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

from .image_scattered import *
from .image_tiled import *

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
    config['index_key'] = 'image_num'
    config['image_num'] = image_num
    config['obj_num'] = obj_num

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildImages nimages = %d: image, obj = %d,%d',
                     config.get('file_num',0),nimages,image_num,obj_num)

    if ( 'image' in config 
         and 'random_seed' in config['image'] 
         and not isinstance(config['image']['random_seed'],dict) ):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = { 'type' : 'Sequence', 'first' : first }

    import time
    def worker(input, output):
        proc = current_process().name
        for job in iter(input.get, 'STOP'):
            try :
                (kwargs, image_num, obj_num, nim, info, logger) = job
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

    if nproc > nimages:
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn(
                "Trying to use more processes than images: output.nproc=%d, "%nproc +
                "nimages=%d.  Reducing nproc to %d."%(nimages,nimages))
        nproc = nimages

    if nproc <= 0:
        # Try to figure out a good number of processes to use
        try:
            from multiprocessing import cpu_count
            ncpu = cpu_count()
            if ncpu > nimages:
                nproc = nimages
            else:
                nproc = ncpu
            if logger and logger.isEnabledFor(logging.WARN):
                logger.warn("ncpu = %d.  Using %d processes",ncpu,nproc)
        except:
            if logger and logger.isEnabledFor(logging.WARN):
                logger.warn("config.output.nproc <= 0, but unable to determine number of cpus.")
            nproc = 1
            if logger and logger.isEnabledFor(logging.INFO):
                logger.info("Unable to determine ncpu.  Using %d processes",nproc)
 
    if nproc > 1:
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


        logger_proxy = galsim.config.GetLoggerProxy(logger)

        # Set up the task list
        task_queue = Queue()
        for k in range(0,nimages,nim_per_task):
            import copy
            kwargs1 = copy.copy(kwargs)
            kwargs1['config'] = galsim.config.CopyConfig(config)
            nim1 = min(nim_per_task, nimages-k)
            task_queue.put( ( kwargs1, image_num+k, obj_num, nim1, k, logger_proxy ) )
            for i in range(nim1):
                obj_num += galsim.config.GetNObjForImage(config, image_num+k+i)

        # Run the tasks
        # Each Process command starts up a parallel process that will keep checking the queue 
        # for a new task. If there is one there, it grabs it and does it. If not, it waits 
        # until there is one to grab. When it finds a 'STOP', it shuts down. 
        done_queue = Queue()
        p_list = []
        for j in range(nproc):
            p = Process(target=worker, args=(task_queue, done_queue), name='Process-%d'%(j+1))
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
    config['index_key'] = 'image_num'
    config['image_num'] = image_num
    config['obj_num'] = obj_num

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: BuildImage: image, obj = %d,%d',image_num,image_num,obj_num)

    # Make config['image'] exist if it doesn't yet.
    if 'image' not in config:
        config['image'] = {}
    image = config['image']
    if not isinstance(image, dict):
        raise AttributeError("config.image is not a dict.")

    if 'draw_method' not in image:
        image['draw_method'] = 'auto'

    if 'type' not in image:
        image['type'] = 'Single'  # Default is Single
    type = image['type']

    if type not in valid_image_types:
        raise AttributeError("Invalid image.type=%s."%type)

    build_func = eval(valid_image_types[type][0])
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


def FlattenNoiseVariance(config, full_image, stamps, current_vars, logger):
    rng = config['rng']
    nobjects = len(stamps)
    max_current_var = max(current_vars)
    if max_current_var > 0:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d: maximum noise varance in any stamp is %f',
                         config['image_num'], max_current_var)
        import numpy
        # Then there was whitening applied in the individual stamps.
        # But there could be a different variance in each postage stamp, so the first
        # thing we need to do is bring everything up to a common level.
        noise_image = galsim.ImageF(full_image.bounds)
        for k in range(nobjects): 
            b = stamps[k].bounds & full_image.bounds
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
    return max_current_var


def _set_image_origin(config, convention):
    """Set `config['image_origin']` appropriately based on the provided `convention`.
    """
    if convention.lower() in [ '0', 'c', 'python' ]:
        origin = 0
    elif convention.lower() in [ '1', 'fortran', 'fits' ]:
        origin = 1
    else:
        raise AttributeError("Unknown index_convention: %s"%convention)
    config['image_origin'] = galsim.PositionI(origin,origin)
    # Also define the overall image center while we're at it.
    xsize = config['image_xsize']
    ysize = config['image_ysize']
    config['image_center'] = galsim.PositionD( origin + (xsize-1.)/2., origin + (ysize-1.)/2. )


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
    """
    config['index_key'] = 'image_num'
    config['image_num'] = image_num
    config['obj_num'] = obj_num

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: BuildSingleImage: image, obj = %d,%d',image_num,image_num,obj_num)

    if 'random_seed' in config['image'] and not isinstance(config['image']['random_seed'],dict):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = { 'type' : 'Sequence', 'first' : first }

    ignore = [ 'random_seed', 'draw_method', 'noise', 'pixel_scale', 'wcs', 'nproc', 
               'sky_level', 'sky_level_pixel',
               'retry_failures', 'n_photons', 'wmult', 'offset', 'gsparams' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int , 'index_convention' : str }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, opt=opt, ignore=ignore)[0]

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

    config['image_xsize'] = xsize
    config['image_ysize'] = ysize
    convention = params.get('index_convention','1')
    _set_image_origin(config,convention)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: image_origin = %s',image_num,str(config['image_origin']))
        logger.debug('image %d: image_center = %s',image_num,str(config['image_center']))

    wcs = galsim.config.BuildWCS(config, logger)

    if 'world_pos' in config['image']:
        config['image']['image_pos'] = (0,0)
        # We allow world_pos to be in config[image], but we don't want it to lead to a final_shift
        # in BuildSingleStamp.  The easiest way to do this is to set image_pos to (0,0).

    return galsim.config.BuildSingleStamp(
            config=config, xsize=xsize, ysize=ysize, obj_num=obj_num,
            do_noise=True, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)[:4] # Required due to `current_var, time` being
                                                     # last two elements of the BuildSingleStamp
                                                     # return tuple

def GetNObjForSingleImage(config, image_num):
    return 1


