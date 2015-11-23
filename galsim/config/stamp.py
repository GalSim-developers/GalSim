# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import galsim

def BuildStamps(nobjects, config, nproc=1, logger=None, obj_num=0,
                xsize=0, ysize=0, do_noise=True,
                make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a number of postage stamp images as specified by the config dict.

    @param nobjects         How many postage stamps to build.
    @param config           A configuration dict.
    @param nproc            How many processes to use. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]
    @param obj_num          If given, the current obj_num. [default: 0]
    @param xsize            The size of a single stamp in the x direction. [default: 0,
                            which means to look for config.image.stamp_xsize, and if that's
                            not there, use automatic sizing.]
    @param ysize            The size of a single stamp in the y direction. [default: 0,
                            which means to look for config.image.stamp_xsize, and if that's
                            not there, use automatic sizing.]
    @param do_noise         Whether to add noise to the image (according to config['noise']).
                            [default: True]
    @param make_psf_image   Whether to make psf_image. [default: False]
    @param make_weight_image  Whether to make weight_image. [default: False]
    @param make_badpix_image  Whether to make badpix_image. [default: False]

    @returns the tuple (images, psf_images, weight_images, badpix_images, current_vars).
             All in tuple are lists.
    """
    config['obj_num'] = obj_num

    def worker(input, output):
        proc = current_process().name
        for job in iter(input.get, 'STOP'):
            try :
                (kwargs, obj_num, nobj, info, logger) = job
                if logger:
                    logger.debug('%s: Received job to do %d stamps, starting with %d',
                                 proc,nobj,obj_num)
                results = []
                for k in range(nobj):
                    kwargs['obj_num'] = obj_num + k
                    kwargs['logger'] = logger
                    result = BuildSingleStamp(**kwargs)
                    results.append(result)
                    # Note: numpy shape is y,x
                    ys, xs = result[0].array.shape
                    t = result[5]
                    if logger:
                        logger.info('%s: Stamp %d: size = %d x %d, time = %f sec', 
                                    proc, obj_num+k, xs, ys, t)
                output.put( (results, info, proc) )
                if logger:
                    logger.debug('%s: Finished job %d -- %d',proc,obj_num,obj_num+nobj-1)
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger:
                    logger.error('%s: Caught exception %s\n%s',proc,str(e),tr)
                output.put( (e, info, tr) )
        if logger:
            logger.debug('%s: Received STOP',proc)
    
    # The kwargs to pass to build_func.
    # We'll be adding to this below...
    kwargs = {
        'xsize' : xsize, 'ysize' : ysize, 
        'do_noise' : do_noise,
        'make_psf_image' : make_psf_image,
        'make_weight_image' : make_weight_image,
        'make_badpix_image' : make_badpix_image
    }

    if nproc > nobjects:
        if logger:
            logger.warn(
                "Trying to use more processes than objects: image.nproc=%d, "%nproc +
                "nobjects=%d.  Reducing nproc to %d."%(nobjects,nobjects))
        nproc = nobjects

    if nproc <= 0:
        # Try to figure out a good number of processes to use
        try:
            from multiprocessing import cpu_count
            ncpu = cpu_count()
            if ncpu > nobjects:
                nproc = nobjects
            else:
                nproc = ncpu
            if logger:
                logger.info("ncpu = %d.  Using %d processes",ncpu,nproc)
        except:
            if logger:
                logger.warn("config.image.nproc <= 0, but unable to determine number of cpus.")
            nproc = 1
            if logger:
                logger.info("Unable to determine ncpu.  Using %d processes",nproc)
    
    if nproc > 1:
        from multiprocessing import Process, Queue, current_process
        from multiprocessing.managers import BaseManager

        # Initialize the images list to have the correct size.
        # This is important here, since we'll be getting back images in a random order,
        # and we need them to go in the right places (in order to have deterministic
        # output files).  So we initialize the list to be the right size.
        images = [ None for i in range(nobjects) ]
        psf_images = [ None for i in range(nobjects) ]
        weight_images = [ None for i in range(nobjects) ]
        badpix_images = [ None for i in range(nobjects) ]
        current_vars = [ None for i in range(nobjects) ]

        # Number of objects to do in each task:
        # At most nobjects / nproc.
        # At least 1 normally, but number in Ring if doing a Ring test
        # Shoot for geometric mean of these two.
        max_nobj = nobjects / nproc
        min_nobj = 1
        if ( 'gal' in config and isinstance(config['gal'],dict) and 'type' in config['gal'] and
             config['gal']['type'] == 'Ring' and 'num' in config['gal'] ):
            min_nobj = galsim.config.ParseValue(config['gal'], 'num', config, int)[0]
        if max_nobj < min_nobj: 
            nobj_per_task = min_nobj
        else:
            import math
            # This formula keeps nobj a multiple of min_nobj, so Rings are intact.
            nobj_per_task = min_nobj * int(math.sqrt(float(max_nobj) / float(min_nobj)))
        
        # The logger is not picklable, se we set up a proxy object.  See comments in process.py
        # for more details about how this works.
        class LoggerManager(BaseManager): pass
        if logger:
            logger_generator = galsim.utilities.SimpleGenerator(logger)
            LoggerManager.register('logger', callable = logger_generator)
            logger_manager = LoggerManager()
            logger_manager.start()

        # Set up the task list
        task_queue = Queue()
        for k in range(0,nobjects,nobj_per_task):
            import copy
            kwargs1 = copy.copy(kwargs)
            kwargs1['config'] = galsim.config.CopyConfig(config)
            if logger:
                logger_proxy = logger_manager.logger()
            else:
                logger_proxy = None
            nobj1 = min(nobj_per_task, nobjects-k)
            task_queue.put( ( kwargs1, obj_num+k, nobj1, k, logger_proxy ) )

        # Run the tasks
        # Each Process command starts up a parallel process that will keep checking the queue 
        # for a new task. If there is one there, it grabs it and does it. If not, it waits 
        # until there is one to grab. When it finds a 'STOP', it shuts down. 
        done_queue = Queue()
        p_list = []
        for j in range(nproc):
            # The name is actually the default name for the first time we do this,
            # but after that it just keeps incrementing the numbers, rather than starting
            # over at Process-1.  As far as I can tell, it's not actually spawning more 
            # processes, so for the sake of the info output, we name the processes 
            # explicitly.
            p = Process(target=worker, args=(task_queue, done_queue), name='Process-%d'%(j+1))
            p.start()
            p_list.append(p)

        # In the meanwhile, the main process keeps going.  We pull each set of images off of the 
        # done_queue and put them in the appropriate place in the lists.
        # This loop is happening while the other processes are still working on their tasks.
        # You'll see that these logging statements get print out as the stamp images are still 
        # being drawn.  
        for i in range(0,nobjects,nobj_per_task):
            results, k0, proc = done_queue.get()
            if isinstance(results,Exception):
                # results is really the exception, e
                # proc is really the traceback
                if logger:
                    logger.error('Exception caught during job starting with stamp %d', k0)
                    logger.error('Aborting the rest of this image')
                for j in range(nproc):
                    p_list[j].terminate()
                raise results
            k = k0
            for result in results:
                images[k] = result[0]
                psf_images[k] = result[1]
                weight_images[k] = result[2]
                badpix_images[k] = result[3]
                current_vars[k] = result[4]
                k += 1
            if logger:
                logger.debug('%s: Successfully returned results for stamps %d--%d', proc, k0, k-1)

        # Stop the processes
        # The 'STOP's could have been put on the task list before starting the processes, or you
        # can wait.  In some cases it can be useful to clear out the done_queue (as we just did)
        # and then add on some more tasks.  We don't need that here, but it's perfectly fine to do.
        # Once you are done with the processes, putting nproc 'STOP's will stop them all.
        # This is important, because the program will keep running as long as there are running
        # processes, even if the main process gets to the end.  So you do want to make sure to 
        # add those 'STOP's at some point!
        for j in range(nproc):
            task_queue.put('STOP')
        for j in range(nproc):
            p_list[j].join()
        task_queue.close()

    else : # nproc == 1

        images = []
        psf_images = []
        weight_images = []
        badpix_images = []
        current_vars = []

        for k in range(nobjects):
            kwargs['config'] = config
            kwargs['obj_num'] = obj_num+k
            kwargs['logger'] = logger
            result = BuildSingleStamp(**kwargs)
            images += [ result[0] ]
            psf_images += [ result[1] ]
            weight_images += [ result[2] ]
            badpix_images += [ result[3] ]
            current_vars += [ result[4] ]
            if logger:
                # Note: numpy shape is y,x
                ys, xs = result[0].array.shape
                t = result[5]
                logger.info('Stamp %d: size = %d x %d, time = %f sec', obj_num+k, xs, ys, t)


    if logger:
        logger.debug('image %d: Done making stamps',config.get('image_num',0))

    return images, psf_images, weight_images, badpix_images, current_vars
 

def BuildSingleStamp(config, xsize=0, ysize=0,
                     obj_num=0, do_noise=True, logger=None,
                     make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a single image using the given config file

    @param config           A configuration dict.
    @param xsize            The xsize of the image to build (if known). [default: 0]
    @param ysize            The ysize of the image to build (if known). [default: 0]
    @param obj_num          If given, the current obj_num [default: 0]
    @param do_noise         Whether to add noise to the image (according to config['noise']).
                            [default: True]
    @param logger           If given, a logger object to log progress. [default: None]
    @param make_psf_image   Whether to make psf_image. [default: False]
    @param make_weight_image  Whether to make weight_image. [default: False]
    @param make_badpix_image  Whether to make badpix_image. [default: False]

    @returns the tuple (image, psf_image, weight_image, badpix_image, current_var, time)
    """
    import time
    t1 = time.time()

    config['index_key'] = 'obj_num'
    config['obj_num'] = obj_num

    # Initialize the random number generator we will be using.
    if 'random_seed' in config['image']:
        seed = galsim.config.ParseValue(config['image'],'random_seed',config,int)[0]
        if logger:
            logger.debug('obj %d: seed = %d',obj_num,seed)
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate()

    # Store the rng in the config for use by BuildGSObject function.
    config['rng'] = rng
    if 'gd' in config:
        del config['gd']  # In case it was set.

    if 'image' in config and 'retry_failures' in config['image']:
        ntries = galsim.config.ParseValue(config['image'],'retry_failures',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    for itry in range(ntries):

        try:  # The rest of the stamp generation stage is wrapped in a try/except block.
              # If we catch an exception, we continue the for loop to try again.
              # On the last time through, we reraise any exception caught.
              # If no exception is thrown, we simply break the loop and return.

            # Determine the size of this stamp
            if not xsize:
                if 'stamp_xsize' in config['image']:
                    xsize = galsim.config.ParseValue(config['image'],'stamp_xsize',config,int)[0]
                elif 'stamp_size' in config['image']:
                    xsize = galsim.config.ParseValue(config['image'],'stamp_size',config,int)[0]
            if not ysize:
                if 'stamp_ysize' in config['image']:
                    ysize = galsim.config.ParseValue(config['image'],'stamp_ysize',config,int)[0]
                elif 'stamp_size' in config['image']:
                    ysize = galsim.config.ParseValue(config['image'],'stamp_size',config,int)[0]
            if False:
                logger.debug('obj %d: xsize,ysize = %d,%d',obj_num,xsize,ysize)
            if xsize: config['stamp_xsize'] = xsize
            if ysize: config['stamp_ysize'] = ysize

            # Determine where this object is going to go:
            if 'image_pos' in config['image'] and 'world_pos' in config['image']:
                image_pos = galsim.config.ParseValue(
                    config['image'], 'image_pos', config, galsim.PositionD)[0]
                world_pos = galsim.config.ParseValue(
                    config['image'], 'world_pos', config, galsim.PositionD)[0]

            elif 'image_pos' in config['image']:
                image_pos = galsim.config.ParseValue(
                    config['image'], 'image_pos', config, galsim.PositionD)[0]
                # Calculate and save the position relative to the image center
                world_pos = config['wcs'].toWorld(image_pos)

                # Wherever we use the world position, we expect a Euclidean position, not a 
                # CelestialCoord.  So if it is the latter, project it onto a tangent plane at the 
                # image center.
                if isinstance(world_pos, galsim.CelestialCoord):
                    # Then project this position relative to the image center.
                    world_center = config['wcs'].toWorld(config['image_center'])
                    world_pos = world_center.project(world_pos, projection='gnomonic')

            elif 'world_pos' in config['image']:
                world_pos = galsim.config.ParseValue(
                    config['image'], 'world_pos', config, galsim.PositionD)[0]
                # Calculate and save the position relative to the image center
                image_pos = config['wcs'].toImage(world_pos)

            else:
                image_pos = None
                world_pos = None

            # Save these values for possible use in Evals or other modules
            if image_pos is not None:
                config['image_pos'] = image_pos
                if logger:
                    logger.debug('obj %d: image_pos = %s',obj_num,str(config['image_pos']))
            if world_pos is not None:
                config['world_pos'] = world_pos
                if logger:
                    logger.debug('obj %d: world_pos = %s',obj_num,str(config['world_pos']))

            if image_pos is not None:
                import math
                # The image_pos refers to the location of the true center of the image, which is 
                # not necessarily the nominal center we need for adding to the final image.  In 
                # particular, even-sized images have their nominal center offset by 1/2 pixel up 
                # and to the right.
                # N.B. This works even if xsize,ysize == 0, since the auto-sizing always produces
                # even sized images.
                nominal_x = image_pos.x        # Make sure we don't change image_pos, which is
                nominal_y = image_pos.y        # stored in config['image_pos'].
                if xsize % 2 == 0: nominal_x += 0.5
                if ysize % 2 == 0: nominal_y += 0.5
                if False:
                    logger.debug('obj %d: nominal pos = %f,%f',obj_num,nominal_x,nominal_y)

                icenter = galsim.PositionI(
                    int(math.floor(nominal_x+0.5)),
                    int(math.floor(nominal_y+0.5)) )
                if False:
                    logger.debug('obj %d: nominal icenter = %s',obj_num,str(icenter))
                offset = galsim.PositionD(nominal_x-icenter.x , nominal_y-icenter.y)
                if False:
                    logger.debug('obj %d: offset = %s',obj_num,str(offset))

            else:
                icenter = None
                offset = galsim.PositionD(0.,0.)
                # Set the image_pos to (0,0) in case the wcs needs it.  Probably, if 
                # there is no image_pos or world_pos defined, then it is unlikely a
                # non-trivial wcs will have been set.  So anything would actually be fine.
                config['image_pos'] = galsim.PositionD(0.,0.)
                if False:
                    logger.debug('obj %d: no offset',obj_num)

            gsparams = {}
            if 'gsparams' in config['image']:
                gsparams = galsim.config.UpdateGSParams(
                    gsparams, config['image']['gsparams'], 'gsparams', config)

            skip = False
            try :
                t4=t3=t2=t1  # in case we throw.
        
                psf = BuildPSF(config,logger,gsparams)
                t2 = time.time()

                gal = BuildGal(config,logger,gsparams)
                t4 = time.time()

                # Check that we have at least gal or psf.
                if not (gal or psf):
                    raise AttributeError("At least one of gal or psf must be specified in config.")

            except galsim.config.gsobject.SkipThisObject, e:
                if logger:
                    logger.debug('obj %d: Caught SkipThisObject: e = %s',obj_num,e.msg)
                    if e.msg:
                        # If there is a message, upgrade to info level
                        logger.info('Skipping object %d: %s',obj_num,e.msg)
                skip = True

            if not skip and 'offset' in config['image']:
                offset1 = galsim.config.ParseValue(config['image'], 'offset', config,
                                                   galsim.PositionD)[0]
                offset += offset1

            if 'image' in config and 'draw_method' in config['image']:
                method = galsim.config.ParseValue(config['image'],'draw_method',config,str)[0]
            else:
                method = 'auto'
            if method not in ['auto', 'fft', 'phot', 'real_space', 'no_pixel', 'sb']:
                raise AttributeError("Invalid draw_method: %s"%method)

            if skip: 
                if xsize and ysize:
                    # If the size is set, we need to do something reasonable to return this size.
                    im = galsim.ImageF(xsize, ysize)
                    im.setOrigin(config['image_origin'])
                    im.setZero()
                    if do_noise:
                        galsim.config.AddNoise(config,'skip',im,weight_im,current_var,logger)
                else:
                    # Otherwise, we don't set the bounds, so it will be noticed as invalid upstream.
                    im = galsim.ImageF()

                if make_weight_image:
                    weight_im = galsim.ImageF(im.bounds, wcs=im.wcs)
                    weight_im.setZero()
                else:
                    weight_im = None
                current_var = 0

            else:
                im, current_var = DrawStamp(psf,gal,config,xsize,ysize,offset,method)
                if icenter:
                    im.setCenter(icenter.x, icenter.y)
                if make_weight_image:
                    weight_im = galsim.ImageF(im.bounds, wcs=im.wcs)
                    weight_im.setZero()
                else:
                    weight_im = None
                if do_noise:
                    # The default indexing for the noise is image_num, not obj_num
                    config['index_key'] = 'image_num'
                    galsim.config.AddNoise(config,method,im,weight_im,current_var,logger)
                    config['index_key'] = 'obj_num'

            if make_badpix_image:
                badpix_im = galsim.ImageS(im.bounds, wcs=im.wcs)
                badpix_im.setZero()
            else:
                badpix_im = None

            t5 = time.time()

            if make_psf_image:
                psf_im = DrawPSFStamp(psf,config,im.bounds,offset,method)
                if ('output' in config and 'psf' in config['output'] and 
                        'signal_to_noise' in config['output']['psf'] and
                        'noise' in config['image']):
                    config['index_key'] = 'image_num'
                    galsim.config.AddNoise(config,'fft',psf_im,None,0,logger,add_sky=False)
                    config['index_key'] = 'obj_num'
            else:
                psf_im = None

            t6 = time.time()

            if logger:
                logger.debug('obj %d: Times: %f, %f, %f, %f, %f',
                             obj_num, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)

        except Exception as e:

            if itry == ntries-1:
                # Then this was the last try.  Just re-raise the exception.
                raise
            else:
                if logger:
                    logger.info('Object %d: Caught exception %s',obj_num,str(e))
                    logger.info('This is try %d/%d, so trying again.',itry+1,ntries)
                # Need to remove the "current_val"s from the config dict.  Otherwise,
                # the value generators will do a quick return with the cached value.
                galsim.config.process.RemoveCurrent(config, keep_safe=True)
                continue

    return im, psf_im, weight_im, badpix_im, current_var, t6-t1


def BuildPSF(config, logger=None, gsparams={}):
    """
    Parse the field config['psf'] returning the built psf object.
    """
 
    if 'psf' in config:
        if not isinstance(config['psf'], dict):
            raise AttributeError("config.psf is not a dict.")
        if False:
            logger.debug('obj %d: Start BuildPSF with %s',config['obj_num'],str(config['psf']))
        psf = galsim.config.BuildGSObject(config, 'psf', config, gsparams, logger)[0]
    else:
        psf = None

    return psf


def BuildGal(config, logger=None, gsparams={}):
    """
    Parse the field config['gal'] returning the built gal object.
    """
 
    if 'gal' in config:
        if not isinstance(config['gal'], dict):
            raise AttributeError("config.gal is not a dict.")
        if False:
            logger.debug('obj %d: Start BuildGal with %s',config['obj_num'],str(config['gal']))
        gal = galsim.config.BuildGSObject(config, 'gal', config, gsparams, logger)[0]
    else:
        gal = None
    return gal



def DrawStamp(psf, gal, config, xsize, ysize, offset, method):
    """
    Draw an image using the given psf and gal profiles (which may be None)
    using the FFT method for doing the convolution.

    @returns the resulting image.
    """

    # Setup the object to draw:
    prof_list = [ prof for prof in (gal,psf) if prof is not None ]
    assert len(prof_list) > 0  # Should have already been checked.
    if len(prof_list) > 1:
        final = galsim.Convolve(prof_list)
    else:
        final = prof_list[0]

    # Setup the kwargs to pass to drawImage
    kwargs = {}
    if xsize:
        kwargs['image'] = galsim.ImageF(xsize, ysize)
    kwargs['offset'] = offset
    kwargs['method'] = method
    if 'image' in config and 'wmult' in config['image']:
        kwargs['wmult'] = galsim.config.ParseValue(config['image'], 'wmult', config, float)[0]
    kwargs['wcs'] = config['wcs'].local(image_pos = config['image_pos'])
    if method == 'phot':
        kwargs['rng'] = config['rng']

    # Check validity of extra phot options:
    max_extra_noise = None
    if 'image' in config and 'n_photons' in config['image']:
        if method != 'phot':
            raise AttributeError('n_photons is invalid with method != phot')
        if 'max_extra_noise' in config['image']:
            import warnings
            warnings.warn(
                "Both 'max_extra_noise' and 'n_photons' are set in config['image'], "+
                "ignoring 'max_extra_noise'.")
        kwargs['n_photons'] = galsim.config.ParseValue(config['image'], 'n_photons', config, int)[0]
    elif 'image' in config and 'max_extra_noise' in config['image']:
        if method != 'phot':
            raise AttributeError('max_extra_noise is invalid with method != phot')
        max_extra_noise = galsim.config.ParseValue(
            config['image'], 'max_extra_noise', config, float)[0]
    elif method == 'phot':
        max_extra_noise = 0.01

    if 'image' in config and 'poisson_flux' in config['image']:
        if method != 'phot':
            raise AttributeError('poisson_flux is invalid with method != phot')
        kwargs['poisson_flux'] = galsim.config.ParseValue(
                config['image'], 'poisson_flux', config, bool)[0]

    if max_extra_noise is not None:
        if max_extra_noise < 0.:
            raise ValueError("image.max_extra_noise cannot be negative")
        if max_extra_noise > 0.:
            if 'image' in config and 'noise' in config['image']:
                noise_var = galsim.config.CalculateNoiseVar(config)
            else:
                raise AttributeError(
                    "Need to specify noise level when using draw_method = phot")
            if noise_var < 0.:
                raise ValueError("noise_var calculated to be < 0.")
            max_extra_noise *= noise_var
            kwargs['max_extra_noise'] = max_extra_noise

    im = final.drawImage(**kwargs)
    im.setOrigin(config['image_origin'])

    # If the object has a noise attribute, then check if we need to do anything with it.
    current_var = 0.  # Default if not overwritten
    if hasattr(final,'noise'):
        if 'noise' in config['image']:
            noise = config['image']['noise']
            if 'whiten' in noise:
                if 'symmetrize' in noise:
                    raise AttributeError('Only one of whiten or symmetrize is allowed')
                whiten, safe = galsim.config.ParseValue(noise, 'whiten', config, bool)
                current_var = final.noise.whitenImage(im)
                if logger:
                    logger.debug('obj %d: whitening noise brought current var to %f',
                                 config['obj_num'],current_var)

            elif 'symmetrize' in noise:
                symmetrize, safe = galsim.config.ParseValue(noise, 'symmetrize', config, int)
                current_var = final.noise.symmetrizeImage(im, symmetrize)
                if logger:
                    logger.debug('obj %d: symmetrizing noise brought current var to %f',
                                 config['obj_num'],current_var)

    if (('gal' in config and 'signal_to_noise' in config['gal']) or
        ('gal' not in config and 'psf' in config and 'signal_to_noise' in config['psf'])):
        if method == 'phot':
            raise NotImplementedError(
                "signal_to_noise option not implemented for draw_method = phot")
        import math
        import numpy
        if 'gal' in config: root_key = 'gal'
        else: root_key = 'psf'

        if 'flux' in config[root_key]:
            raise AttributeError(
                'Only one of signal_to_noise or flux may be specified for %s'%root_key)

        if 'image' in config and 'noise' in config['image']:
            noise_var = galsim.config.CalculateNoiseVar(config)
        else:
            raise AttributeError(
                "Need to specify noise level when using %s.signal_to_noise"%root_key)
        sn_target = galsim.config.ParseValue(config[root_key], 'signal_to_noise', config, float)[0]
            
        # Now determine what flux we need to get our desired S/N
        # There are lots of definitions of S/N, but here is the one used by Great08
        # We use a weighted integral of the flux:
        # S = sum W(x,y) I(x,y) / sum W(x,y)
        # N^2 = Var(S) = sum W(x,y)^2 Var(I(x,y)) / (sum W(x,y))^2
        # Now we assume that Var(I(x,y)) is dominated by the sky noise, so
        # Var(I(x,y)) = var
        # We also assume that we are using a matched filter for W, so W(x,y) = I(x,y).
        # Then a few things cancel and we find that
        # S/N = sqrt( sum I(x,y)^2 / var )

        sn_meas = math.sqrt( numpy.sum(im.array**2) / noise_var )
        # Now we rescale the flux to get our desired S/N
        flux = sn_target / sn_meas
        im *= flux
        if hasattr(final,'noise'):
            current_var *= flux**2

    return im, current_var


def DrawPSFStamp(psf, config, bounds, offset, method):
    """
    Draw an image using the given psf profile.

    @returns the resulting image.
    """

    if not psf:
        raise AttributeError("DrawPSFStamp requires psf to be provided.")

    if ('output' in config and 'psf' in config['output'] and 
        'draw_method' in config['output']['psf'] ):
        method = galsim.config.ParseValue(config['output']['psf'],'draw_method',config,str)[0]
        if method not in ['auto', 'fft', 'phot', 'real_space', 'no_pixel', 'sb']:
            raise AttributeError("Invalid draw_method: %s"%method)
    else:
        method = 'auto'

    # Special: if the galaxy was shifted, then also shift the psf 
    if 'shift' in config['gal']:
        gal_shift = galsim.config.GetCurrentValue(config['gal'],'shift')
        if False:
            logger.debug('obj %d: psf shift (1): %s',config['obj_num'],str(gal_shift))
        psf = psf.shift(gal_shift)

    wcs = config['wcs'].local(config['image_pos'])
    im = galsim.ImageF(bounds, wcs=wcs)
    im = psf.drawImage(image=im, offset=offset, method=method)

    if (('output' in config and 'psf' in config['output'] 
            and 'signal_to_noise' in config['output']['psf']) or
        ('gal' not in config and 'psf' in config and 'signal_to_noise' in config['psf'])):
        if method == 'phot':
            raise NotImplementedError(
                "signal_to_noise option not implemented for draw_method = phot")
        import math
        import numpy

        if 'image' in config and 'noise' in config['image']:
            noise_var = galsim.config.CalculateNoiseVar(config)
        else:
            raise AttributeError(
                "Need to specify noise level when using psf.signal_to_noise")

        if ('output' in config and 'psf' in config['output'] 
                and 'signal_to_noise' in config['output']['psf']):
            cf = config['output']['psf']
        else:
            cf = config['psf']
        sn_target = galsim.config.ParseValue(cf, 'signal_to_noise', config, float)[0]
            
        sn_meas = math.sqrt( numpy.sum(im.array**2) / noise_var )
        flux = sn_target / sn_meas
        im *= flux

    return im
           

