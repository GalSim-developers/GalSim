# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#

import time
import galsim


def BuildStamps(nobjects, config, nproc=1, logger=None, obj_num=0,
                xsize=0, ysize=0, sky_level_pixel=None, do_noise=True,
                make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a number of postage stamp images as specified by the config dict.

    @param nobjects            How many postage stamps to build.
    @param config              A configuration dict.
    @param nproc               How many processes to use.
    @param logger              If given, a logger object to log progress.
    @param obj_num             If given, the current obj_num (default = 0)
    @param xsize               The size of a single stamp in the x direction.
                               (If 0, look for config.image.stamp_xsize, and if that's
                                not there, use automatic sizing.)
    @param ysize               The size of a single stamp in the y direction.
                               (If 0, look for config.image.stamp_ysize, and if that's
                                not there, use automatic sizing.)
    @param sky_level_pixel     The background sky level to add to the image (in ADU/pixel).
    @param do_noise            Whether to add noise to the image (according to config['noise']).
    @param make_psf_image      Whether to make psf_image.
    @param make_weight_image   Whether to make weight_image.
    @param make_badpix_image   Whether to make badpix_image.

    @return (images, psf_images, weight_images, badpix_images)  (All in tuple are lists)
    """
    def worker(input, output):
        for (kwargs, config, obj_num, nobj, info) in iter(input.get, 'STOP'):
            results = []
            # Make new copies of config and kwargs so we can update them without
            # clobbering the versions for other tasks on the queue.
            # (The config modifications come in BuildSingleStamp.)
            import copy
            kwargs1 = copy.copy(kwargs)
            config1 = copy.deepcopy(config)
            for i in range(nobj):
                kwargs1['config'] = config1
                kwargs1['obj_num'] = obj_num + i
                results.append(BuildSingleStamp(**kwargs1))
            output.put( (results, info, current_process().name) )
    
    # The kwargs to pass to build_func.
    # We'll be adding to this below...
    kwargs = {
        'xsize' : xsize, 'ysize' : ysize, 
        'sky_level_pixel' : sky_level_pixel,
        'do_noise' : do_noise,
        'make_psf_image' : make_psf_image,
        'make_weight_image' : make_weight_image,
        'make_badpix_image' : make_badpix_image
    }
    # Apparently the logger isn't picklable, so can't send that as an arg.

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

        # Initialize the images list to have the correct size.
        # This is important here, since we'll be getting back images in a random order,
        # and we need them to go in the right places (in order to have deterministic
        # output files).  So we initialize the list to be the right size.
        images = [ None for i in range(nobjects) ]
        psf_images = [ None for i in range(nobjects) ]
        weight_images = [ None for i in range(nobjects) ]
        badpix_images = [ None for i in range(nobjects) ]

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

        # Set up the task list
        task_queue = Queue()
        for k in range(0,nobjects,nobj_per_task):
            # Send kwargs, config, obj_num, nobj, k
            if k + nobj_per_task > nobjects:
                task_queue.put( ( kwargs, config, obj_num+k, nobjects-k, k ) )
            else:
                task_queue.put( ( kwargs, config, obj_num+k, nobj_per_task, k ) )

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
            results, k, proc = done_queue.get()
            for result in results:
                images[k] = result[0]
                psf_images[k] = result[1]
                weight_images[k] = result[2]
                badpix_images[k] = result[3]
                if logger:
                    # Note: numpy shape is y,x
                    ys, xs = result[0].array.shape
                    t = result[4]
                    logger.info('%s: Stamp %d: size = %d x %d, time = %f sec', 
                                proc, obj_num+k, xs, ys, t)
                k += 1

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

        for k in range(nobjects):
            kwargs['obj_num'] = obj_num+k
            kwargs['config'] = config
            kwargs['obj_num'] = obj_num+k
            kwargs['logger'] = logger
            result = BuildSingleStamp(**kwargs)
            images += [ result[0] ]
            psf_images += [ result[1] ]
            weight_images += [ result[2] ]
            badpix_images += [ result[3] ]
            if logger:
                # Note: numpy shape is y,x
                ys, xs = result[0].array.shape
                t = result[4]
                logger.info('Stamp %d: size = %d x %d, time = %f sec', obj_num+k, xs, ys, t)


    if logger:
        logger.debug('Done making stamps')

    return images, psf_images, weight_images, badpix_images
 

def BuildSingleStamp(config, xsize=0, ysize=0,
                     obj_num=0, sky_level_pixel=None, do_noise=True, logger=None,
                     make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a single image using the given config file

    @param config              A configuration dict.
    @param xsize               The xsize of the image to build (if known).
    @param ysize               The ysize of the image to build (if known).
    @param obj_num             If given, the current obj_num (default = 0)
    @param sky_level_pixel     The background sky level to add to the image (in ADU/pixel).
    @param do_noise            Whether to add noise to the image (according to config['noise']).
    @param logger              If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image.
    @param make_weight_image   Whether to make weight_image.
    @param make_badpix_image   Whether to make badpix_image.

    @return image, psf_image, weight_image, badpix_image, time
    """
    t1 = time.time()

    config['seq_index'] = obj_num 
    # Initialize the random number generator we will be using.
    if 'random_seed' in config['image']:
        seed = galsim.config.ParseValue(config['image'],'random_seed',config,int)[0]
        #print 'seed = ',seed
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate()
    # Store the rng in the config for use by BuildGSObject function.
    config['rng'] = rng
    if 'gd' in config:
        del config['gd']  # In case it was set.

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

    # Determine where this object is going to go:
    if 'image_pos' in config['image']:
        import math
        image_pos = galsim.config.ParseValue(config['image'],'image_pos',config,
                                             galsim.PositionD)[0]
        # Save this value for possible use in Eval's.
        config['image_pos'] = image_pos
        #print 'image_pos = ',image_pos

        # Calculate and save the position relative to the image center
        config['sky_pos'] = (image_pos - config['image_cen']) * config['pixel_scale']
        #print 'sky_pos = ',config['sky_pos']

    elif 'sky_pos' in config['image']:
        import math
        sky_pos = galsim.config.ParseValue(config['image'],'sky_pos',config,
                                           galsim.PositionD)[0]
        # Save this value for possible use in Eval's.
        config['sky_pos'] = sky_pos
        #print 'sky_pos = ',sky_pos

        # Calculate and save the position relative to the image center
        image_pos = (sky_pos / config['pixel_scale']) + config['image_cen']
        config['image_pos'] = image_pos
        #print 'image_pos = ',config['image_pos']

    else:
        image_pos = None


    if image_pos is not None:
        # The image_pos refers to the location of the true center of the image, which is not 
        # necessarily the nominal center we need for adding to the final image.  In particular,
        # even-sized images have their nominal center offset by 1/2 pixel up and to the right.
        # N.B. This works even if xsize,ysize == 0, since the auto-sizing always produces
        # even sized images.
        if xsize % 2 == 0: image_pos.x += 0.5
        if ysize % 2 == 0: image_pos.y += 0.5
        #print 'nominal image_pos = ',image_pos

        icenter = galsim.PositionI(
            int(math.floor(image_pos.x+0.5)),
            int(math.floor(image_pos.y+0.5)) )
        #print 'icenter = ',icenter
        final_shift = galsim.PositionD(image_pos.x-icenter.x , image_pos.y-icenter.y)
        #print 'final_shift = ',final_shift

    else:
        icenter = None
        final_shift = None

    gsparams = {}
    if 'gsparams' in config['image']:
        gsparams = galsim.config.UpdateGSParams(
            gsparams, config['image']['gsparams'], 'gsparams', config)

    skip = False
    try :
        t4=t3=t2=t1  # in case we throw.

        psf = BuildPSF(config,logger,gsparams)
        t2 = time.time()

        pix = BuildPix(config,logger,gsparams)
        t3 = time.time()

        gal = BuildGal(config,logger,gsparams)
        t4 = time.time()

        # Check that we have at least gal or psf.
        if not (gal or psf):
            raise AttributeError("At least one of gal or psf must be specified in config.")

    except galsim.config.gsobject.SkipThisObject, e:
        #print 'Caught SkipThisObject: e =',e.msg
        if logger:
            if e.msg:
                # If there is a message, upgrade to warn level
                logger.warn('Skipping object: %s',e.msg)
            else:
                logger.debug('Skipping object')
        skip = True

    draw_method = galsim.config.ParseValue(config['image'],'draw_method',config,str)[0]
    if skip: 
        if xsize and ysize:
            im = galsim.ImageF(xsize, ysize)
        else:
            im = galsim.ImageF(1,1)
        im.setOrigin(config['image_origin'])
        im.setScale(im.scale)
        im.setZero()
        if do_noise and sky_level_pixel:
            im += sky_level_pixel
        if make_weight_image:
            weight_im = galsim.ImageF(im.bounds)
            weight_im.setScale(im.scale)
            weight_im.setZero()
        else:
            weight_im = None
    elif draw_method == 'fft':
        im = DrawStampFFT(psf,pix,gal,config,xsize,ysize,sky_level_pixel,final_shift)
        if icenter:
            im.setCenter(icenter.x, icenter.y)
        if make_weight_image:
            weight_im = galsim.ImageF(im.bounds)
            weight_im.setScale(im.scale)
            weight_im.setZero()
        else:
            weight_im = None
        if do_noise:
            if 'noise' in config['image']:
                AddNoiseFFT(im,weight_im,config['image']['noise'],config,rng,sky_level_pixel,
                            logger)
            elif sky_level_pixel:
                im += sky_level_pixel

    elif draw_method == 'phot':
        im = DrawStampPhot(psf,gal,config,xsize,ysize,rng,sky_level_pixel,final_shift)
        if icenter:
            im.setCenter(icenter.x, icenter.y)
        if make_weight_image:
            weight_im = galsim.ImageF(im.bounds)
            weight_im.setScale(im.scale)
            weight_im.setZero()
        else:
            weight_im = None
        if do_noise:
            if 'noise' in config['image']:
                AddNoisePhot(im,weight_im,config['image']['noise'],config,rng,sky_level_pixel,
                             logger)
            elif sky_level_pixel:
                im += sky_level_pixel

    else:
        raise AttributeError("Unknown draw_method %s."%draw_method)

    if make_badpix_image:
        badpix_im = galsim.ImageS(im.bounds)
        badpix_im.setScale(im.scale)
        badpix_im.setZero()
    else:
        badpix_im = None

    t5 = time.time()

    if make_psf_image:
        psf_im = DrawPSFStamp(psf,pix,config,im.bounds,final_shift)
    else:
        psf_im = None

    t6 = time.time()

    if logger:
        logger.debug('   Times: %f, %f, %f, %f, %f', t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)
    return im, psf_im, weight_im, badpix_im, t6-t1


def BuildPSF(config, logger=None, gsparams={}):
    """
    Parse the field config['psf'] returning the built psf object.
    """
 
    if 'psf' in config:
        if not isinstance(config['psf'], dict):
            raise AttributeError("config.psf is not a dict.")
        psf = galsim.config.BuildGSObject(config, 'psf', config, gsparams)[0]
    else:
        psf = None

    return psf

def BuildPix(config, logger=None, gsparams={}):
    """
    Parse the field config['pix'] returning the built pix object.
    """
 
    if 'pix' in config: 
        if not isinstance(config['pix'], dict):
            raise AttributeError("config.pix is not a dict.")
        pix = galsim.config.BuildGSObject(config, 'pix', config, gsparams)[0]
    else:
        pix = None

    return pix


def BuildGal(config, logger=None, gsparams={}):
    """
    Parse the field config['gal'] returning the built gal object.
    """
 
    if 'gal' in config:
        if not isinstance(config['gal'], dict):
            raise AttributeError("config.gal is not a dict.")
        gal = galsim.config.BuildGSObject(config, 'gal', config, gsparams)[0]
    else:
        gal = None
    return gal



def DrawStampFFT(psf, pix, gal, config, xsize, ysize, sky_level_pixel, final_shift):
    """
    Draw an image using the given psf, pix and gal profiles (which may be None)
    using the FFT method for doing the convolution.

    @return the resulting image.
    """
    if 'image' in config and 'wcs' in config['image']:
        wcs_shear = CalculateWCSShear(config['image']['wcs'], config)
    else:
        wcs_shear = None

    if wcs_shear:
        nopix_list = [ prof for prof in (psf,gal) if prof is not None ]
        nopix = galsim.Convolve(nopix_list)
        nopix.applyShear(wcs_shear)
        if pix:
            final = galsim.Convolve([nopix, pix])
        else:
            final = nopix
        config['wcs_shear'] = wcs_shear
    else:
        fft_list = [ prof for prof in (psf,pix,gal) if prof is not None ]
        final = galsim.Convolve(fft_list)

    if 'image' in config and 'pixel_scale' in config['image']:
        pixel_scale = galsim.config.ParseValue(config['image'], 'pixel_scale', config, float)[0]
    else:
        pixel_scale = 1.0

    if 'image' in config and 'wmult' in config['image']:
        wmult = galsim.config.ParseValue(config['image'], 'wmult', config, float)[0]
    else:
        wmult = 1.0

    if final_shift:
        #print 'shift = ',final_shift*pixel_scale
        final.applyShift(final_shift.x*pixel_scale, final_shift.y*pixel_scale)

    if xsize:
        im = galsim.ImageF(xsize, ysize)
    else:
        im = None

    im = final.draw(image=im, dx=pixel_scale, wmult=wmult)
    im.setOrigin(config['image_origin'])

    if 'gal' in config and 'signal_to_noise' in config['gal']:
        import math
        import numpy
        if 'flux' in config['gal']:
            raise AttributeError(
                'Only one of signal_to_noise or flux may be specified for gal')

        if 'image' in config and 'noise' in config['image']:
            noise_var = CalculateNoiseVar(config['image']['noise'], config, pixel_scale, 
                                          sky_level_pixel)
        else:
            raise AttributeError(
                "Need to specify noise level when using gal.signal_to_noise")
        sn_target = galsim.config.ParseValue(config['gal'], 'signal_to_noise', config, float)[0]
            
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

    return im

def AddNoiseFFT(im, weight_im, noise, base, rng, sky_level_pixel, logger=None):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the FFT method.
    """
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")

    if 'type' not in noise:
        noise['type'] = 'Poisson'  # Default is Poisson
    type = noise['type']

    # First add the background sky level, if provided
    if sky_level_pixel:
        im += sky_level_pixel

    # Check if a weight image should include the object variance.
    if weight_im:
        include_obj_var = False
        if ('output' in base and 'weight' in base['output'] and 
            'include_obj_var' in base['output']['weight']):
            include_obj_var = galsim.config.ParseValue(
                base['output']['weight'], 'include_obj_var', base, bool)[0]

    # Then add the correct kind of noise
    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', base, single=single)[0]

        if 'sigma' in params:
            sigma = params['sigma']
        else:
            import math
            sigma = math.sqrt(params['variance'])
        im.addNoise(galsim.GaussianNoise(rng,sigma=sigma))

        if weight_im:
            weight_im += sigma*sigma
        if logger:
            logger.debug('   Added Gaussian noise with sigma = %f',sigma)

    elif type == 'Poisson':
        opt = {}
        single = []
        if sky_level_pixel:
            # The noise sky_level is only required here if the image doesn't have any.
            opt['sky_level'] = float
            opt['sky_level_pixel'] = float
        else:
            single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
            sky_level_pixel = 0.
        params = galsim.config.GetAllParams(noise, 'noise', base, opt=opt, single=single)[0]
        if 'sky_level' in params and 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                "noise.type = %s"%type)
        extra_sky_level_pixel = 0.
        if 'sky_level' in params:
            pixel_scale = im.getScale()
            extra_sky_level_pixel = params['sky_level'] * pixel_scale**2
        if 'sky_level_pixel' in params:
            extra_sky_level_pixel = params['sky_level_pixel']
        sky_level_pixel += extra_sky_level_pixel

        if weight_im:
            import math
            if include_obj_var:
                # The image right now has the variance in each pixel.  So before going on with the 
                # noise, copy these over to the weight image.  (We invert this later...)
                weight_im.copyFrom(im)
            else:
                # Otherwise, just add the sky and read_noise:
                weight_im += sky_level_pixel

        im.addNoise(galsim.PoissonNoise(rng, sky_level=extra_sky_level_pixel))
        if logger:
            logger.debug('   Added Poisson noise with sky_level_pixel = %f',sky_level_pixel)

    elif type == 'CCD':
        opt = { 'gain' : float , 'read_noise' : float }
        single = []
        if sky_level_pixel:
            # The noise sky_level is only required here if the image doesn't have any.
            opt['sky_level'] = float
            opt['sky_level_pixel'] = float
        else:
            single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
            sky_level_pixel = 0.
        params = galsim.config.GetAllParams(noise, 'noise', base, opt=opt, single=single)[0]
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)
        if 'sky_level' in params and 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                "noise.type = %s"%type)
        extra_sky_level_pixel = 0.
        if 'sky_level' in params:
            pixel_scale = im.getScale()
            extra_sky_level_pixel = params['sky_level'] * pixel_scale**2
        if 'sky_level_pixel' in params:
            extra_sky_level_pixel = params['sky_level_pixel']
        sky_level_pixel += extra_sky_level_pixel

        if weight_im:
            import math
            if include_obj_var:
                # The image right now has the variance in each pixel.  So before going on with the 
                # noise, copy these over to the weight image and invert.
                weight_im.copyFrom(im)
                if gain != 1.0:
                    weight_im /= math.sqrt(gain)
                if read_noise != 0.0:
                    weight_im += read_noise*read_noise
            else:
                # Otherwise, just add the sky and read_noise:
                weight_im += sky_level_pixel / gain + read_noise*read_noise

        #print 'CCDNoise with ',extra_sky_level_pixel,gain,read_noise
        im.addNoise(galsim.CCDNoise(rng, sky_level=extra_sky_level_pixel, gain=gain,
                                    read_noise=read_noise))
        if logger:
            logger.debug('   Added CCD noise with sky_level_pixel = %f, ' +
                         'gain = %f, read_noise = %f',sky_level_pixel,gain,read_noise)

    elif type == 'COSMOS':
        req = { 'file_name' : str }
        opt = { 'dx_cosmos' : float, 'variance' : float }
        
        kwargs = galsim.config.GetAllParams(noise, 'noise', base, req=req, opt=opt)[0]

        # Build the correlated noise 
        cn = galsim.correlatednoise.getCOSMOSNoise(rng, **kwargs)
        im.addNoise(cn)

        # Then add the variance to the weight image, using the zero-lag correlation function value
        if weight_im:
            weight_im += cn.getVariance()
        if logger:
            logger.debug(
                '   Added COSMOS correlated noise with variance = %f',
                cn.getVariance())

    else:
        raise AttributeError("Invalid type %s for noise"%type)


def DrawStampPhot(psf, gal, config, xsize, ysize, rng, sky_level_pixel, final_shift):
    """
    Draw an image using the given psf and gal profiles (which may be None)
    using the photon shooting method for doing the convolution.

    @return the resulting image.
    """

    phot_list = [ prof for prof in (psf,gal) if prof is not None ]
    final = galsim.Convolve(phot_list)

    if 'image' in config and 'wcs' in config['image']:
        wcs_shear = CalculateWCSShear(config['image']['wcs'], config)
    else:
        wcs_shear = None

    if wcs_shear:
        final.applyShear(wcs_shear)
        config['wcs_shear'] = wcs_shear
                    
    if 'signal_to_noise' in config['gal']:
        raise NotImplementedError(
            "gal.signal_to_noise not implemented for draw_method = phot")

    if final_shift:
        final.applyShift(final_shift.x, final_shift.y)

    if 'image' in config and 'pixel_scale' in config['image']:
        pixel_scale = galsim.config.ParseValue(config['image'], 'pixel_scale', config, float)[0]
    else:
        pixel_scale = 1.0

    if xsize:
        im = galsim.ImageF(xsize, ysize)
    else:
        im = None

    if 'image' in config and 'n_photons' in config['image']:

        if 'max_extra_noise' in config['image']:
            import warnings
            warnings.warn(
                "Both 'max_extra_noise' and 'n_photons' are set in config['image'], "+
                "ignoring 'max_extra_noise'.")

        n_photons = galsim.config.ParseValue(
            config['image'], 'n_photons', config, int)[0]
        im = final.drawShoot(image=im, dx=pixel_scale, n_photons=n_photons, rng=rng)
        im.setOrigin(config['image_origin'])

    else:

        if 'image' in config and 'max_extra_noise' in config['image']:
            max_extra_noise = galsim.config.ParseValue(
                config['image'], 'max_extra_noise', config, float)[0]
        else:
            max_extra_noise = 0.01

        if max_extra_noise < 0.:
            raise ValueError("image.max_extra_noise cannot be negative")

        if max_extra_noise > 0.:
            if 'image' in config and 'noise' in config['image']:
                noise_var = CalculateNoiseVar(config['image']['noise'], config, pixel_scale, 
                                              sky_level_pixel)
            else:
                raise AttributeError(
                    "Need to specify noise level when using draw_method = phot")
            if noise_var < 0.:
                raise ValueError("noise_var calculated to be < 0.")
            max_extra_noise *= noise_var

        im = final.drawShoot(image=im, dx=pixel_scale, max_extra_noise=max_extra_noise, rng=rng)
        im.setOrigin(config['image_origin'])

    return im
    
def AddNoisePhot(im, weight_im, noise, base, rng, sky_level_pixel, logger=None):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the photon-shooting method.
    """
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")

    if 'type' not in noise:
        noise['type'] = 'Poisson'  # Default is Poisson
    type = noise['type']

    # First add the sky noise, if provided
    if sky_level_pixel:
        im += sky_level_pixel

    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', base, single=single)[0]

        if 'sigma' in params:
            sigma = params['sigma']
        else:
            import math
            sigma = math.sqrt(params['variance'])
        im.addNoise(galsim.GaussianNoise(rng,sigma=sigma))

        if weight_im:
            weight_im += sigma*sigma
        if logger:
            logger.debug('   Added Gaussian noise with sigma = %f',sigma)

    elif type == 'Poisson':
        opt = {}
        if sky_level_pixel:
            opt['sky_level'] = float
            opt['sky_level_pixel'] = float
        else:
            single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
            sky_level_pixel = 0. # Switch from None to 0.
        params = galsim.config.GetAllParams(noise, 'noise', base, opt=opt, single=single)[0]
        if 'sky_level' in params and 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                "noise.type = %s"%type)
        if 'sky_level' in params:
            pixel_scale = im.getScale()
            sky_level_pixel += params['sky_level'] * pixel_scale**2
        if 'sky_level_pixel' in params:
            sky_level_pixel += params['sky_level_pixel']

        # We don't have an exact value for the variance in each pixel, but the drawn image
        # before adding the Poisson noise is our best guess for the variance from the 
        # object's flux, so just use that for starters.
        if weight_im and include_obj_var:
            weight_im.copyFrom(im)

        # For photon shooting, galaxy already has Poisson noise, so we want 
        # to make sure not to add that again!
        if sky_level_pixel != 0.:
            im.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel)))
            im -= sky_level_pixel
            if weight_im:
                weight_im += sky_level_pixel

        if logger:
            logger.debug('   Added Poisson noise with sky_level_pixel = %f',sky_level_pixel)


    elif type == 'CCD':
        opt = { 'gain' : float , 'read_noise' : float }
        if sky_level_pixel:
            opt['sky_level'] = float
            opt['sky_level_pixel'] = float
        else:
            single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
            sky_level_pixel = 0. # Switch from None to 0.
        params = galsim.config.GetAllParams(noise, 'noise', base, opt=opt, single=single)[0]
        if 'sky_level' in params and 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                "noise.type = %s"%type)
        if 'sky_level' in params:
            pixel_scale = im.getScale()
            sky_level_pixel += params['sky_level'] * pixel_scale**2
        if 'sky_level_pixel' in params:
            sky_level_pixel += params['sky_level_pixel']
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)

        # We don't have an exact value for the variance in each pixel, but the drawn image
        # before adding the Poisson noise is our best guess for the variance from the 
        # object's flux, so just use that for starters.
        if weight_im and include_obj_var:
            weight_im.copyFrom(im)

        # For photon shooting, galaxy already has Poisson noise, so we want 
        # to make sure not to add that again!
        if sky_level_pixel != 0.:
            if gain != 1.0: im *= gain
            im.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel*gain)))
            if gain != 1.0: im /= gain
            im -= sky_level_pixel*gain
        if read_noise != 0.:
            im.addNoise(galsim.GaussianNoise(rng, sigma=read_noise))

        # Add in these effects to the weight image:
        if weight_im:
            import math
            if sky_level_pixel != 0.0 or read_noise != 0.0:
                weight_im += sky_level_pixel / gain + read_noise * read_noise
        if logger:
            logger.debug('   Added CCD noise with sky_level_pixel = %f, ' +
                         'gain = %f, read_noise = %f',sky_level_pixel,gain,read_noise)

    elif type == 'COSMOS':
        req = { 'file_name' : str }
        opt = { 'dx_cosmos' : float, 'variance' : float }
        
        kwargs = galsim.config.GetAllParams(noise, 'noise', base, req=req, opt=opt)[0]

        # Build and add the correlated noise 
        cn = galsim.correlatednoise.getCOSMOSNoise(rng, **kwargs)
        im.addNoise(cn)

        # Then add the variance to the weight image, using the zero-lag correlation function value
        if weight_im:
            weight_im += cn.getVariance()
        if logger:
            logger.debug(
                '   Added COSMOS correlated noise with variance = %f',
                cn.getVariance())

    else:
        raise AttributeError("Invalid type %s for noise",type)


def DrawPSFStamp(psf, pix, config, bounds, final_shift):
    """
    Draw an image using the given psf and pix profiles.

    @return the resulting image.
    """

    if not psf:
        raise AttributeError("DrawPSFStamp requires psf to be provided.")

    if 'wcs_shear' in config:
        wcs_shear = config['wcs_shear']
    else:
        wcs_shear = None

    if wcs_shear:
        psf = psf.createSheared(wcs_shear)

    psf_list = [ prof for prof in (psf,pix) if prof is not None ]

    if ('output' in config and 
        'psf' in config['output'] and 
        'real_space' in config['output']['psf'] ):
        real_space = galsim.config.ParseValue(config['output']['psf'],'real_space',config,bool)[0]
    else:
        real_space = None
        
    final_psf = galsim.Convolve(psf_list, real_space=real_space)

    if 'image' in config and 'pixel_scale' in config['image']:
        pixel_scale = galsim.config.ParseValue(config['image'], 'pixel_scale', config, float)[0]
    else:
        pixel_scale = 1.0

    # Special: if the galaxy was shifted, then also shift the psf 
    if 'shift' in config['gal']:
        gal_shift = galsim.config.GetCurrentValue(config['gal'],'shift')
        final_psf.applyShift(gal_shift.x, gal_shift.y)

    # Also apply any "final" shift to the psf.
    if final_shift:
        final_psf.applyShift(final_shift.x, final_shift.y)

    psf_im = galsim.ImageF(bounds)
    psf_im.setScale(pixel_scale)
    final_psf.draw(psf_im, dx=pixel_scale)

    return psf_im
           
def CalculateWCSShear(wcs, base):
    """
    Calculate the WCS shear from the WCS specified in the wcs dict.
    TODO: Should add in more WCS types than just a simple shear
          E.g. a full CD matrix and (eventually) things like TAN and TNX.
    """
    if not isinstance(wcs, dict):
        raise AttributeError("image.wcs is not a dict.")

    if 'type' not in wcs:
        wcs['type'] = 'Shear'  # Default is Shear
    type = wcs['type']

    if type == 'Shear':
        req = { 'shear' : galsim.Shear }
        params = galsim.config.GetAllParams(wcs, 'wcs', base, req=req)[0]
        return params['shear']
    else:
        raise AttributeError("Invalid type %s for wcs",type)

def CalculateNoiseVar(noise, base, pixel_scale, sky_level_pixel):
    """
    Calculate the noise variance from the noise specified in the noise dict.
    """
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")

    if 'type' not in noise:
        noise['type'] = 'Poisson'  # Default is Poisson
    type = noise['type']

    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', base, single=single)[0]
        if 'sigma' in params:
            sigma = params['sigma']
            var = sigma * sigma
        else:
            var = params['variance']

    elif type == 'Poisson':
        opt = {}
        single = []
        if sky_level_pixel:
            opt['sky_level'] = float
            opt['sky_level_pixel'] = float
        else:
            single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
            sky_level_pixel = 0. # Switch from None to 0.
        params = galsim.config.GetAllParams(noise, 'noise', base, opt=opt, single=single)[0]
        if 'sky_level' in params and 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                "noise.type = %s"%type)
        if 'sky_level' in params:
            sky_level_pixel += params['sky_level'] * pixel_scale**2
        if 'sky_level_pixel' in params:
            sky_level_pixel += params['sky_level_pixel']
        var = sky_level_pixel

    elif type == 'CCD':
        opt = { 'gain' : float , 'read_noise' : float }
        single = []
        if sky_level_pixel:
            opt['sky_level'] = float
            opt['sky_level_pixel'] = float
        else:
            single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
            sky_level_pixel = 0. # Switch from None to 0.
        params = galsim.config.GetAllParams(noise, 'noise', base, opt=opt, single=single)[0]
        if 'sky_level' in params and 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                "noise.type = %s"%type)
        if 'sky_level' in params:
            sky_level_pixel += params['sky_level'] * pixel_scale**2
        if 'sky_level_pixel' in params:
            sky_level_pixel += params['sky_level_pixel']
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)
        var = sky_level_pixel / gain + read_noise * read_noise

    elif type == 'COSMOS':
        req = { 'file_name' : str }
        opt = { 'dx_cosmos' : float, 'variance' : float }
        
        kwargs = galsim.config.GetAllParams(noise, 'noise', base, req=req, opt=opt)[0]

        # Build and add the correlated noise (lets the cn internals handle dealing with the options
        # for default variance: quick and ensures we don't needlessly duplicate code) 
        # Note: the rng being passed here is arbitrary, since we don't need it to calculate the
        # variance.  Building a BaseDeviate with a particular seed is the fastest option.
        cn = galsim.correlatednoise.getCOSMOSNoise(galsim.BaseDeviate(123), **kwargs)

        # zero distance correlation function value returned as variance
        var = cn.getVariance()

    else:
        raise AttributeError("Invalid type %s for noise",type)

    return var


