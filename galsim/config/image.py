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
import logging

# The items in each tuple are:
#   - The function to call to build the image
#   - The function to call to get the number of objects that will be built
valid_image_types = { 
    'Single' : ( 'BuildSingleImage', 'GetNObjForSingleImage' ),
    'Tiled' : ( 'BuildTiledImage', 'GetNObjForTiledImage' ),
    'Scattered' : ( 'BuildScatteredImage', 'GetNObjForScatteredImage' ),
}

def BuildImages(nimages, config, nproc=1, logger=None, image_num=0, obj_num=0,
                make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a number of postage stamp images as specified by the config dict.

    @param nimages             How many images to build.
    @param config              A configuration dict.
    @param nproc               How many processes to use. [default: 1]
    @param logger              If given, a logger object to log progress. [default: None]
    @param image_num           If given, the current `image_num` [default: 0]
    @param obj_num             If given, the current `obj_num` [default: 0]
    @param make_psf_image      Whether to make `psf_image`. [default: False]
    @param make_weight_image   Whether to make `weight_image`. [default: False]
    @param make_badpix_image   Whether to make `badpix_image`. [default: False]

    @returns the tuple `(images, psf_images, weight_images, badpix_images)`.
             All in tuple are lists.
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildImages nimages = %d: image, obj = %d,%d',
                     config.get('file_num',0),nimages,image_num,obj_num)

    import time
    def worker(input, output, kwargs, logger):
        proc = current_process().name
        for job in iter(input.get, 'STOP'):
            try :
                (image_num, obj_num, nim, info) = job
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('%s: Received job to do %d images, starting with %d',
                                 proc,nim,image_num)
                results = []
                for k in range(nim):
                    t1 = time.time()
                    kwargs['image_num'] = image_num + k
                    kwargs['obj_num'] = obj_num
                    kwargs['logger'] = logger
                    im = BuildImage(**kwargs)
                    obj_num += galsim.config.GetNObjForImage(kwargs['config'], image_num+k)
                    t2 = time.time()
                    results.append( [im[0], im[1], im[2], im[3], t2-t1 ] )
                    ys, xs = im[0].array.shape
                    if logger and logger.isEnabledFor(logging.INFO):
                        logger.info('%s: Image %d: size = %d x %d, time = %f sec', 
                                    proc, image_num+k, xs, ys, t2-t1)
                output.put( (results, info, proc) )
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('%s: Finished job %d -- %d',proc,image_num,image_num+nim-1)
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('%s: Caught exception %s\n%s',proc,str(e),tr)
                output.put( (e, info, tr) )
    
    # The kwargs to pass to BuildImage
    kwargs = {
        'make_psf_image' : make_psf_image,
        'make_weight_image' : make_weight_image,
        'make_badpix_image' : make_badpix_image
    }

    nproc = galsim.config.UpdateNProc(nproc,logger)
 
    if nproc > 1 and 'current_nproc' in config:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Already multiprocessing.  Ignoring nproc for image processing")
        nproc = 1
 
    if nproc > nimages:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug("There are only %d images.  Reducing nproc to %d."%(nimages,nimages))
        nproc = nimages

    if nproc > 1:
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn("Using %d processes for image processing",nproc)

        from multiprocessing import Process, Queue, current_process
        from multiprocessing.managers import BaseManager

        # Initialize the images list to have the correct size.
        # This is important here, since we'll be getting back images in a random order,
        # and we need them to go in the right places (in order to have deterministic
        # output files).  So we initialize the list to be the right size.
        images = [ None for i in range(nimages) ]
        psf_images = [ None for i in range(nimages) ]
        weight_images = [ None for i in range(nimages) ]
        badpix_images = [ None for i in range(nimages) ]

        # Number of images to do in each task:
        # At most nimages / nproc.
        # At least 1 normally, but number in Ring if doing a Ring test
        # Shoot for gemoetric mean of these two.
        max_nim = nimages / nproc
        min_nim = 1
        if ( ('image' not in config or 'type' not in config['image'] or 
                 config['image']['type'] == 'Single') and
             'gal' in config and isinstance(config['gal'],dict) and 'type' in config['gal'] and
             config['gal']['type'] == 'Ring' and 'num' in config['gal'] ):
            min_nim = galsim.config.ParseValue(config['gal'], 'num', config, int)[0]
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Found ring: num = %d',config.get('file_num',0),min_nim)
        if max_nim < min_nim: 
            nim_per_task = min_nim
        else:
            import math
            # This formula keeps nim a multiple of min_nim, so Rings are intact.
            nim_per_task = min_nim * int(math.sqrt(float(max_nim) / float(min_nim)))
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: nim_per_task = %d',config.get('file_num',0),nim_per_task)

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
        for k in range(0,nimages,nim_per_task):
            nim1 = min(nim_per_task, nimages-k)
            task_queue.put( ( image_num+k, obj_num, nim1, k ) )
            for i in range(nim1):
                obj_num += galsim.config.GetNObjForImage(config, image_num+k+i)

        # Run the tasks
        # Each Process command starts up a parallel process that will keep checking the queue 
        # for a new task. If there is one there, it grabs it and does it. If not, it waits 
        # until there is one to grab. When it finds a 'STOP', it shuts down. 
        done_queue = Queue()
        p_list = []
        import copy
        kwargs1 = copy.copy(kwargs)
        config1 = galsim.config.CopyConfig(config)
        config1['current_nproc'] = nproc
        kwargs1['config'] = config1
        if logger:
            logger_proxy = logger_manager.logger()
        else:
            logger_proxy = None
        for j in range(nproc):
            p = Process(target=worker, args=(task_queue, done_queue, kwargs1, logger_proxy),
                        name='Process-%d'%(j+1))
            p.start()
            p_list.append(p)

        # In the meanwhile, the main process keeps going.  We pull each set of images off of the 
        # done_queue and put them in the appropriate place in the lists.
        # This loop is happening while the other processes are still working on their tasks.
        # You'll see that these logging statements get printed out as the stamp images are still 
        # being drawn.  
        for i in range(0,nimages,nim_per_task):
            results, k0, proc = done_queue.get()
            if isinstance(results,Exception):
                # results is really the exception, e
                # proc is really the traceback
                if logger:
                    logger.error('Exception caught during job starting with image %d', k0)
                    logger.error('%s',proc)
                    logger.error('Aborting the rest of this file')
                for j in range(nproc):
                    p_list[j].terminate()
                raise results
            k = k0
            for result in results:
                images[k] = result[0]
                psf_images[k] = result[1]
                weight_images[k] = result[2]
                badpix_images[k] = result[3]
                k += 1
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('%s: Successfully returned results for images %d--%d', proc, k0, k-1)

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

        for k in range(nimages):
            t1 = time.time()
            kwargs['config'] = config
            kwargs['image_num'] = image_num+k
            kwargs['obj_num'] = obj_num
            kwargs['logger'] = logger
            result = BuildImage(**kwargs)
            images += [ result[0] ]
            psf_images += [ result[1] ]
            weight_images += [ result[2] ]
            badpix_images += [ result[3] ]
            t2 = time.time()
            if logger and logger.isEnabledFor(logging.INFO):
                # Note: numpy shape is y,x
                ys, xs = result[0].array.shape
                logger.info('Image %d: size = %d x %d, time = %f sec', image_num+k, xs, ys, t2-t1)
            obj_num += galsim.config.GetNObjForImage(config, image_num+k)

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Done making images %d--%d',config.get('file_num',0),
                     image_num,image_num+nimages-1)

    return images, psf_images, weight_images, badpix_images
 
def SetupConfigImageNum(config, image_num, obj_num):
    """Do the basic setup of the config dict at the image processing level.

    Includes:
    - Set config['image_num'] = image_num
    - Set config['obj_num'] = obj_num
    - Set config['index_key'] = 'image_num'
    - Make sure config['image'] exists
    - Set config['image']['draw_method'] to 'auto' if not given.

    @param config           A configuration dict.
    @param image_num        The current image_num.
    @param obj_num          The current obj_num.
    """
    config['image_num'] = image_num
    config['obj_num'] = obj_num
    config['index_key'] = 'image_num'
 
    # Make config['image'] exist if it doesn't yet.
    if 'image' not in config:
        config['image'] = {}
    image = config['image']
    if not isinstance(image, dict):
        raise AttributeError("config.image is not a dict.")

    if 'draw_method' not in image:
        image['draw_method'] = 'auto'
    if 'type' not in image:
        image['type'] = 'Single'


def SetupConfigImageSize(config, xsize, ysize, wcs):
    """Do some further setup of the config dict at the image processing level based on
    the provided image size and wcs.

    - Set config['image_xsize'], config['image_ysize'] to the size of the image
    - Set config['image_origin'] to the origin of the image
    - Set config['image_center'] to the center of the image
    - If wcs is None, build the wcs using galsim.config.
    - Set config['wcs'] to the wcs

    @param config           A configuration dict.
    @param xsize            The size of the image in the x-dimension.
    @param ysize            The size of the image in the y-dimension.
    @param wcs              The wcs for the image. [default: None, which means build it from
                            either config['image']['pixel_scale'] or config['image']['wcs'].]
    """
    config['image_xsize'] = xsize
    config['image_ysize'] = ysize

    origin = 1 # default
    if 'index_convention' in config['image']:
        convention = galsim.config.ParseValue(config['image'],'index_convention',config,str)[0]
        if convention.lower() in [ '0', 'c', 'python' ]:
            origin = 0
        elif convention.lower() in [ '1', 'fortran', 'fits' ]:
            origin = 1
        else:
            raise AttributeError("Unknown index_convention: %s"%convention)

    config['image_origin'] = galsim.PositionI(origin,origin)
    config['image_center'] = galsim.PositionD( origin + (xsize-1.)/2., origin + (ysize-1.)/2. )

    config['wcs'] = galsim.config.BuildWCS(config)


def BuildImage(config, logger=None, image_num=0, obj_num=0,
               make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an Image according to the information in config.

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress. [default: None]
    @param image_num           If given, the current `image_num` [default: 0]
    @param obj_num             If given, the current `obj_num` [default: 0]
    @param make_psf_image      Whether to make `psf_image`. [default: False]
    @param make_weight_image   Whether to make `weight_image`. [default: False]
    @param make_badpix_image   Whether to make `badpix_image`. [default: False]

    @returns the tuple `(image, psf_image, weight_image, badpix_image)`.

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: BuildImage: image, obj = %d,%d',image_num,image_num,obj_num)

    if 'image' in config and 'type' in config['image']:   
        image_type = config['image']['type']
    else:
        image_type = 'Single'

    if image_type not in valid_image_types:
        raise AttributeError("Invalid image.type=%s."%image_type)

    build_func = eval(valid_image_types[image_type][0])
    all_images = build_func(
            config=config, logger=logger,
            image_num=image_num, obj_num=obj_num,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)

    # The later image building functions build up the weight image as the total variance 
    # in each pixel.  We need to invert this to produce the inverse variance map.
    # Doing it here means it only needs to be done in this one place.
    if all_images[2]:
        all_images[2].invertSelf()

    return all_images

# Ignore these when parsing the parameters for specific Image types:
image_ignore = [ 'random_seed', 'draw_method', 'noise', 'pixel_scale', 'wcs', 
                 'sky_level', 'sky_level_pixel', 'index_convention',
                 'retry_failures', 'n_photons', 'wmult', 'offset', 'gsparams' ]

def BuildSingleImage(config, logger=None, image_num=0, obj_num=0,
                     make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an Image consisting of a single stamp.

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress. [default: None]
    @param image_num           If given, the current `image_num` [default: 0]
    @param obj_num             If given, the current `obj_num` [default: 0]
    @param make_psf_image      Whether to make `psf_image`. [default: False]
    @param make_weight_image   Whether to make `weight_image`. [default: False]
    @param make_badpix_image   Whether to make `badpix_image`. [default: False]

    @returns the tuple `(image, psf_image, weight_image, badpix_image)`.

    Note: All 4 Images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    Also: It is easier to accumulate the inverse weight map, which is just the noise variance
          in each pixel.  BuildImage will invert this to the more normal weight map, but here
          it is actually a variance map.
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: BuildSingleImage: image, obj = %d,%d',image_num,image_num,obj_num)
    SetupConfigImageNum(config,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: seed = %d',image_num,seed)

    extra_ignore = [ 'image_pos', 'world_pos' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, opt=opt, ignore=image_ignore+extra_ignore)[0]

    # If image_force_xsize and image_force_ysize were set in config, this overrides the 
    # read-in params.
    if 'image_force_xsize' in config and 'image_force_ysize' in config:
        xsize = config['image_force_xsize']
        ysize = config['image_force_ysize']
    else:
        size = params.get('size',0)
        xsize = params.get('xsize',size)
        ysize = params.get('ysize',size)
    if (xsize == 0) != (ysize == 0):
        raise AttributeError(
            "Both (or neither) of image.xsize and image.ysize need to be defined  and != 0.")

    SetupConfigImageSize(config,xsize,ysize,wcs=None)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: image_origin = %s',image_num,str(config['image_origin']))
        logger.debug('image %d: image_center = %s',image_num,str(config['image_center']))

    # We allow world_pos to be in config[image], but we don't want it to lead to a final_shift
    # in BuildSingleStamp.  The easiest way to do this is to set image_pos to (0,0).
    if 'world_pos' in config['image']:
        config['image']['image_pos'] = (0,0)

    galsim.config.SetupExtraOutputsForImage(config,1,logger)

    all_images = galsim.config.BuildSingleStamp(
            config=config, xsize=xsize, ysize=ysize, obj_num=obj_num,
            do_noise=True, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)[:4] # Required due to `current_var, time` being
                                                     # last two elements of the BuildSingleStamp
                                                     # return tuple
    
    galsim.config.ProcessExtraOutputsForImage(config,logger)

    return all_images


def BuildTiledImage(config, logger=None, image_num=0, obj_num=0,
                    make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an Image consisting of a tiled array of postage stamps.

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress. [default: None]
    @param image_num           If given, the current `image_num`. [default: 0]
    @param obj_num             If given, the current `obj_num`. [default: 0]
    @param make_psf_image      Whether to make `psf_image`. [default: False]
    @param make_weight_image   Whether to make `weight_image`. [default: False]
    @param make_badpix_image   Whether to make `badpix_image`. [default: False]

    @returns the tuple `(image, psf_image, weight_image, badpix_image)`.

    Note: All 4 Images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    Also: It is easier to accumulate the inverse weight map, which is just the noise variance
          in each pixel.  BuildImage will invert this to the more normal weight map, but here
          it is actually a variance map.
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: BuildTiledImage: image, obj = %d,%d',image_num,image_num,obj_num)
    SetupConfigImageNum(config,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: seed = %d',image_num,seed)
    rng = config['rng'] # Grab this for use later

    extra_ignore = [ 'image_pos' ] # We create this below, so on subequent passes, we ignore it.
    req = { 'nx_tiles' : int , 'ny_tiles' : int }
    opt = { 'stamp_size' : int , 'stamp_xsize' : int , 'stamp_ysize' : int ,
            'border' : int , 'xborder' : int , 'yborder' : int ,
            'nproc' : int , 'order' : str }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, req=req, opt=opt, ignore=image_ignore+extra_ignore)[0]

    nx_tiles = params['nx_tiles']
    ny_tiles = params['ny_tiles']
    nobjects = nx_tiles * ny_tiles
    config['nx_tiles'] = nx_tiles
    config['ny_tiles'] = ny_tiles
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: n_tiles = %d, %d',image_num,nx_tiles,ny_tiles)

    stamp_size = params.get('stamp_size',0)
    stamp_xsize = params.get('stamp_xsize',stamp_size)
    stamp_ysize = params.get('stamp_ysize',stamp_size)
    config['tile_xsize'] = stamp_xsize
    config['tile_ysize'] = stamp_ysize

    if (stamp_xsize == 0) or (stamp_ysize == 0):
        raise AttributeError(
            "Both image.stamp_xsize and image.stamp_ysize need to be defined and != 0.")

    border = params.get("border",0)
    xborder = params.get("xborder",border)
    yborder = params.get("yborder",border)

    do_noise = xborder >= 0 and yborder >= 0
    # TODO: Note: if one of these is < 0 and the other is > 0, then
    #       this will add noise to the border region.  Not exactly the 
    #       design, but I didn't bother to do the bookkeeping right to 
    #       make the borders pure 0 in that case.
    
    full_xsize = (stamp_xsize + xborder) * nx_tiles - xborder
    full_ysize = (stamp_ysize + yborder) * ny_tiles - yborder

    # If image_force_xsize and image_force_ysize were set in config, make sure it matches.
    if ( ('image_force_xsize' in config and full_xsize != config['image_force_xsize']) or
         ('image_force_ysize' in config and full_ysize != config['image_force_ysize']) ):
        raise ValueError(
            "Unable to reconcile required image xsize and ysize with provided "+
            "nx_tiles=%d, ny_tiles=%d, "%(nx_tiles,ny_tiles) +
            "xborder=%d, yborder=%d\n"%(xborder,yborder) +
            "Calculated full_size = (%d,%d) "%(full_xsize,full_ysize)+
            "!= required (%d,%d)."%(config['image_force_xsize'],config['image_force_ysize']))

    SetupConfigImageSize(config,full_xsize,full_ysize,wcs=None)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: image_size = %d, %d',image_num,full_xsize,full_ysize)
        logger.debug('image %d: image_origin = %s',image_num,str(config['image_origin']))
        logger.debug('image %d: image_center = %s',image_num,str(config['image_center']))

    # Make a list of ix,iy values according to the specified order:
    order = params.get('order','row').lower()
    if order.startswith('row'):
        ix_list = [ ix for iy in range(ny_tiles) for ix in range(nx_tiles) ]
        iy_list = [ iy for iy in range(ny_tiles) for ix in range(nx_tiles) ]
    elif order.startswith('col'):
        ix_list = [ ix for ix in range(nx_tiles) for iy in range(ny_tiles) ]
        iy_list = [ iy for ix in range(nx_tiles) for iy in range(ny_tiles) ]
    elif order.startswith('rand'):
        ix_list = [ ix for ix in range(nx_tiles) for iy in range(ny_tiles) ]
        iy_list = [ iy for ix in range(nx_tiles) for iy in range(ny_tiles) ]
        galsim.random.permute(rng, ix_list, iy_list)
        
    # Define a 'image_pos' field so the stamps can set their position appropriately in case
    # we need it for PowerSpectum or NFWHalo.
    x0 = (stamp_xsize-1)/2. + config['image_origin'].x
    y0 = (stamp_ysize-1)/2. + config['image_origin'].y
    dx = stamp_xsize + xborder
    dy = stamp_ysize + yborder
    config['image']['image_pos'] = { 
        'type' : 'XY' ,
        'x' : { 'type' : 'List',
                'items' : [ x0 + ix*dx for ix in ix_list ]
              },
        'y' : { 'type' : 'List',
                'items' : [ y0 + iy*dy for iy in iy_list ]
              }
    }

    nproc = params.get('nproc',1)

    wcs = config['wcs']
    full_image = galsim.ImageF(full_xsize, full_ysize)
    full_image.setOrigin(config['image_origin'])
    full_image.wcs = wcs
    full_image.setZero()

    config['image_bounds'] = full_image.bounds

    if make_psf_image:
        full_psf_image = galsim.ImageF(full_image.bounds, wcs=wcs)
        full_psf_image.setZero()
    else:
        full_psf_image = None

    if make_weight_image:
        full_weight_image = galsim.ImageF(full_image.bounds, wcs=wcs)
        full_weight_image.setZero()
    else:
        full_weight_image = None

    full_badpix_image = None

    # Sometimes an input field needs to do something special at the start of an image.
    galsim.config.SetupInputsForImage(config,logger)
    galsim.config.SetupExtraOutputsForImage(config,nobjects,logger)

    stamp_images = galsim.config.BuildStamps(
            nobjects=nobjects, config=config,
            nproc=nproc, logger=logger, obj_num=obj_num,
            xsize=stamp_xsize, ysize=stamp_ysize, do_noise=do_noise,
            make_psf_image=make_psf_image,
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)

    images = stamp_images[0]
    psf_images = stamp_images[1]
    weight_images = stamp_images[2]
    badpix_images = stamp_images[3]
    current_vars = stamp_images[4]

    max_current_var = 0
    for k in range(nobjects):
        # This is our signal that the object was skipped.
        if not images[k].bounds.isDefined(): continue
        if False:
            logger.debug('image %d: full bounds = %s',image_num,str(full_image.bounds))
            logger.debug('image %d: stamp %d bounds = %s',image_num,k,str(images[k].bounds))
        assert full_image.bounds.includes(images[k].bounds)
        b = images[k].bounds
        full_image[b] += images[k]
        if make_psf_image:
            full_psf_image[b] += psf_images[k]
        if make_weight_image:
            full_weight_image[b] += weight_images[k]
        if current_vars[k] > max_current_var: max_current_var = current_vars[k]

    # Mark that we are no longer doing a single galaxy by deleting image_pos from config top 
    # level, so it cannot be used for things like wcs.pixelArea(image_pos).  
    if 'image_pos' in config: del config['image_pos']

    # Put the rng back into config['rng'] for use by the AddNoise function.
    config['rng'] = rng

    # Store the current image in the base-level config for reference
    config['current_image'] = full_image

    galsim.config.AddSky(config,full_image)

    if make_weight_image and not do_noise:
        if 'include_obj_var' in config['output']['weight']:
            include_obj_var = galsim.config.ParseValue(
                    config['output']['weight'], 'include_obj_var', config, bool)[0]
        else:
            include_obj_var = False
        galsim.config.AddNoiseVariance(config,full_weight_image,include_obj_var,logger)

    galsim.config.ProcessExtraOutputsForImage(config,logger)

    # If didn't do noise above in the stamps, then need to do it here.
    if not do_noise:
        if 'noise' in config['image']:
            # If we didn't apply noise in each stamp, then we need to apply it now.
            if max_current_var > 0:
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('image %d: maximum noise varance in any stamp is %f',
                                 config['image_num'], max_current_var)
                import numpy
                # Then there was whitening applied in the individual stamps.
                # But there could be a different variance in each postage stamp, so the first
                # thing we need to do is bring everything up to a common level.
                noise_image = galsim.ImageF(full_image.bounds)
                for k in range(nobjects): noise_image[images[k].bounds] += current_vars[k]
                # Update this, since overlapping postage stamps may have led to a larger 
                # value in some pixels.
                max_current_var = numpy.max(noise_image.array)
                # Figure out how much noise we need to add to each pixel.
                noise_image = max_current_var - noise_image
                # Add it.
                full_image.addNoise(galsim.VariableGaussianNoise(rng,noise_image))
            # Now max_current_var is how much noise is in each pixel.

            galsim.config.AddNoise(config,full_image,max_current_var,logger)

    return full_image, full_psf_image, full_weight_image, full_badpix_image


def BuildScatteredImage(config, logger=None, image_num=0, obj_num=0,
                        make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an Image containing multiple objects placed at arbitrary locations.

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress. [default None]
    @param image_num           If given, the current `image_num` [default: 0]
    @param obj_num             If given, the current `obj_num` [default: 0]
    @param make_psf_image      Whether to make `psf_image`. [default: False]
    @param make_weight_image   Whether to make `weight_image`. [default: False]
    @param make_badpix_image   Whether to make `badpix_image`. [default: False]

    @returns the tuple `(image, psf_image, weight_image, badpix_image)`.

    Note: All 4 Images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    Also: It is easier to accumulate the inverse weight map, which is just the noise variance
          in each pixel.  BuildImage will invert this to the more normal weight map, but here
          it is actually a variance map.
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: BuildScatteredImage: image, obj = %d,%d',
                     image_num,image_num,obj_num)
    SetupConfigImageNum(config,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: seed = %d',image_num,seed)
    rng = config['rng'] # Grab this for use later

    if 'nobjects' not in config['image']:
        nobjects = galsim.config.ProcessInputNObjects(config)
        if nobjects is None:
            raise AttributeError("Attribute nobjects is required for image.type = Scattered")
    else:
        nobjects = galsim.config.ParseValue(config['image'],'nobjects',config,int)[0]
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: nobj = %d',image_num,nobjects)

    # These are allowed for Scattered, but we don't use them here.
    extra_ignore = [ 'image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                     'nobjects' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int , 'nproc' : int }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, opt=opt, ignore=image_ignore+extra_ignore)[0]

    # Special check for the size.  Either size or both xsize and ysize is required.
    if 'size' not in params:
        if 'xsize' not in params or 'ysize' not in params:
            raise AttributeError(
                "Either attribute size or both xsize and ysize required for image.type=Scattered")
        full_xsize = params['xsize']
        full_ysize = params['ysize']
    else:
        if 'xsize' in params:
            raise AttributeError(
                "Attributes xsize is invalid if size is set for image.type=Scattered")
        if 'ysize' in params:
            raise AttributeError(
                "Attributes ysize is invalid if size is set for image.type=Scattered")
        full_xsize = params['size']
        full_ysize = params['size']


    # If image_force_xsize and image_force_ysize were set in config, make sure it matches.
    if ( ('image_force_xsize' in config and full_xsize != config['image_force_xsize']) or
         ('image_force_ysize' in config and full_ysize != config['image_force_ysize']) ):
        raise ValueError(
            "Unable to reconcile required image xsize and ysize with provided "+
            "xsize=%d, ysize=%d, "%(full_xsize,full_ysize))

    SetupConfigImageSize(config,full_xsize,full_ysize,wcs=None)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: image_size = %d, %d',image_num,full_xsize,full_ysize)
        logger.debug('image %d: image_origin = %s',image_num,str(config['image_origin']))
        logger.debug('image %d: image_center = %s',image_num,str(config['image_center']))

    if 'image_pos' in config['image'] and 'world_pos' in config['image']:
        raise AttributeError("Both image_pos and world_pos specified for Scattered image.")

    if 'image_pos' not in config['image'] and 'world_pos' not in config['image']:
        xmin = config['image_origin'].x
        xmax = xmin + full_xsize-1
        ymin = config['image_origin'].y
        ymax = ymin + full_ysize-1
        config['image']['image_pos'] = { 
            'type' : 'XY' ,
            'x' : { 'type' : 'Random' , 'min' : xmin , 'max' : xmax },
            'y' : { 'type' : 'Random' , 'min' : ymin , 'max' : ymax }
        }

    nproc = params.get('nproc',1)

    wcs = config['wcs']
    full_image = galsim.ImageF(full_xsize, full_ysize)
    full_image.setOrigin(config['image_origin'])
    full_image.wcs = wcs
    full_image.setZero()

    config['image_bounds'] = full_image.bounds

    if make_psf_image:
        full_psf_image = galsim.ImageF(full_image.bounds, wcs=wcs)
        full_psf_image.setZero()
    else:
        full_psf_image = None

    if make_weight_image:
        full_weight_image = galsim.ImageF(full_image.bounds, wcs=wcs)
        full_weight_image.setZero()
    else:
        full_weight_image = None

    full_badpix_image = None

    # Sometimes an input field needs to do something special at the start of an image.
    galsim.config.SetupInputsForImage(config,logger)
    galsim.config.SetupExtraOutputsForImage(config,nobjects,logger)

    stamp_images = galsim.config.BuildStamps(
            nobjects=nobjects, config=config,
            nproc=nproc, logger=logger,obj_num=obj_num, do_noise=False,
            make_psf_image=make_psf_image,
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)

    images = stamp_images[0]
    psf_images = stamp_images[1]
    weight_images = stamp_images[2]
    badpix_images = stamp_images[3]
    current_vars = stamp_images[4]

    max_current_var = 0.
    for k in range(nobjects):
        # This is our signal that the object was skipped.
        if not images[k].bounds.isDefined(): continue
        bounds = images[k].bounds & full_image.bounds
        if False:
            logger.debug('image %d: full bounds = %s',image_num,str(full_image.bounds))
            logger.debug('image %d: stamp %d bounds = %s',image_num,k,str(images[k].bounds))
            logger.debug('image %d: Overlap = %s',image_num,str(bounds))
        if bounds.isDefined():
            full_image[bounds] += images[k][bounds]
            if make_psf_image:
                full_psf_image[bounds] += psf_images[k][bounds]
            if make_weight_image:
                full_weight_image[bounds] += weight_images[k][bounds]
        else:
            if logger and logger.isEnabledFor(logging.INFO):
                logger.warn(
                    "Object centered at (%d,%d) is entirely off the main image,\n"%(
                        images[k].bounds.center().x, images[k].bounds.center().y) +
                    "whose bounds are (%d,%d,%d,%d)."%(
                        full_image.bounds.xmin, full_image.bounds.xmax,
                        full_image.bounds.ymin, full_image.bounds.ymax))
        if current_vars[k] > max_current_var: max_current_var = current_vars[k]

    # Mark that we are no longer doing a single galaxy by deleting image_pos from config top 
    # level, so it cannot be used for things like wcs.pixelArea(image_pos).  
    if 'image_pos' in config: del config['image_pos']

    # Put the rng back into config['rng'] for use by the AddNoise function.
    config['rng'] = rng

    # Store the current image in the base-level config for reference
    config['current_image'] = full_image

    galsim.config.AddSky(config,full_image)

    if make_weight_image:
        if 'include_obj_var' in config['output']['weight']:
            include_obj_var = galsim.config.ParseValue(
                    config['output']['weight'], 'include_obj_var', config, bool)[0]
        else:
            include_obj_var = False
        galsim.config.AddNoiseVariance(config,full_weight_image,include_obj_var,logger)

    galsim.config.ProcessExtraOutputsForImage(config,logger)

    if 'noise' in config['image']:
        # Apply the noise to the full image
        if max_current_var > 0:
            import numpy
            # Then there was whitening applied in the individual stamps.
            # But there could be a different variance in each postage stamp, so the first
            # thing we need to do is bring everything up to a common level.
            noise_image = galsim.ImageF(full_image.bounds)
            for k in range(nobjects): 
                b = images[k].bounds & full_image.bounds
                if b.isDefined(): noise_image[b] += current_vars[k]
            # Update this, since overlapping postage stamps may have led to a larger 
            # value in some pixels.
            max_current_var = numpy.max(noise_image.array)
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('image %d: maximum noise varance in any pixel is %f',
                             config['image_num'], max_current_var)
            # Figure out how much noise we need to add to each pixel.
            noise_image = max_current_var - noise_image
            # Add it.
            full_image.addNoise(galsim.VariableGaussianNoise(rng,noise_image))
        # Now max_current_var is how much noise is in each pixel.

        galsim.config.AddNoise(config,full_image,max_current_var,logger)

    return full_image, full_psf_image, full_weight_image, full_badpix_image


def GetNObjForImage(config, image_num):
    if 'image' in config and 'type' in config['image']:
        image_type = config['image']['type']
    else:
        image_type = 'Single'

    # Check that the type is valid
    if image_type not in valid_image_types:
        raise AttributeError("Invalid image.type=%s."%type)

    nobj_func = eval(valid_image_types[image_type][1])

    return nobj_func(config,image_num)

def GetNObjForSingleImage(config, image_num):
    return 1

def GetNObjForScatteredImage(config, image_num):

    config['index_key'] = 'image_num'
    config['image_num'] = image_num

    # Allow nobjects to be automatic based on input catalog
    if 'nobjects' not in config['image']:
        nobj = galsim.config.ProcessInputNObjects(config)
        if nobj is None:
            raise AttributeError("Attribute nobjects is required for image.type = Scattered")
        return nobj
    else:
        nobj = galsim.config.ParseValue(config['image'],'nobjects',config,int)[0]
        return nobj

def GetNObjForTiledImage(config, image_num):
    
    config['index_key'] = 'image_num'
    config['image_num'] = image_num

    if 'nx_tiles' not in config['image'] or 'ny_tiles' not in config['image']:
        raise AttributeError(
            "Attributes nx_tiles and ny_tiles are required for image.type = Tiled")
    nx = galsim.config.ParseValue(config['image'],'nx_tiles',config,int)[0]
    ny = galsim.config.ParseValue(config['image'],'ny_tiles',config,int)[0]
    return nx*ny

