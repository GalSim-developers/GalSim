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

import galsim

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
    @param nproc               How many processes to use.
    @param logger              If given, a logger object to log progress.
    @param image_num           If given, the current image_num (default = 0)
    @param obj_num             If given, the current obj_num (default = 0)
    @param make_psf_image      Whether to make psf_image.
    @param make_weight_image   Whether to make weight_image.
    @param make_badpix_image   Whether to make badpix_image.

    @return (images, psf_images, weight_images, badpix_images)  (All in tuple are lists)
    """
    if logger:
        logger.debug('file %d: BuildImages nimages = %d: image, obj = %d,%d',
                      config['file_num'],nimages,image_num,obj_num)

    import time
    def worker(input, output):
        proc = current_process().name
        for job in iter(input.get, 'STOP'):
            try :
                (kwargs, image_num, obj_num, nim, info, logger) = job
                if logger:
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
                    if logger:
                        logger.info('%s: Image %d: size = %d x %d, time = %f sec', 
                                    proc, image_num+k, xs, ys, t2-t1)
                output.put( (results, info, proc) )
                if logger:
                    logger.debug('%s: Finished job %d -- %d',proc,image_num,image_num+nim-1)
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger:
                    logger.debug('%s: Caught exception %s\n%s',proc,str(e),tr)
                output.put( (e, info, tr) )
    
    # The kwargs to pass to BuildImage
    kwargs = {
        'make_psf_image' : make_psf_image,
        'make_weight_image' : make_weight_image,
        'make_badpix_image' : make_badpix_image
    }

    if nproc > nimages:
        if logger:
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
            if logger:
                logger.info("ncpu = %d.  Using %d processes",ncpu,nproc)
        except:
            if logger:
                logger.warn("config.output.nproc <= 0, but unable to determine number of cpus.")
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
            if logger:
                logger.debug('file %d: Found ring: num = %d',config['file_num'],min_nim)
        if max_nim < min_nim: 
            nim_per_task = min_nim
        else:
            import math
            # This formula keeps nim a multiple of min_nim, so Rings are intact.
            nim_per_task = min_nim * int(math.sqrt(float(max_nim) / float(min_nim)))
        if logger:
            logger.debug('file %d: nim_per_task = %d',config['file_num'],nim_per_task)

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
            import copy
            kwargs1 = copy.copy(kwargs)
            kwargs1['config'] = galsim.config.CopyConfig(config)
            if logger:
                logger_proxy = logger_manager.logger()
            else:
                logger_proxy = None
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
            if logger:
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
            if logger:
                # Note: numpy shape is y,x
                ys, xs = result[0].array.shape
                logger.info('Image %d: size = %d x %d, time = %f sec', image_num+k, xs, ys, t2-t1)
            obj_num += galsim.config.GetNObjForImage(config, image_num+k)

    if logger:
        logger.debug('file %d: Done making images %d--%d',config['file_num'],
                     image_num,image_num+nimages-1)

    return images, psf_images, weight_images, badpix_images
 

def BuildImage(config, logger=None, image_num=0, obj_num=0,
               make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image according to the information in config.

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress.
    @param image_num           If given, the current image_num (default = 0)
    @param obj_num             If given, the current obj_num (default = 0)
    @param make_psf_image      Whether to make psf_image.
    @param make_weight_image   Whether to make weight_image.
    @param make_badpix_image   Whether to make badpix_image.

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.
    """
    if logger:
        logger.debug('image %d: BuildImage: image, obj = %d,%d',
                      image_num,image_num,obj_num)

    # Make config['image'] exist if it doesn't yet.
    if 'image' not in config:
        config['image'] = {}
    image = config['image']
    if not isinstance(image, dict):
        raise AttributeError("config.image is not a dict.")

    # Normally, random_seed is just a number, which really means to use that number
    # for the first item and go up sequentially from there for each object.
    # However, we allow for random_seed to be a gettable parameter, so for the 
    # normal case, we just convert it into a Sequence.
    if 'random_seed' in image and not isinstance(image['random_seed'],dict):
        first_seed = galsim.config.ParseValue(image, 'random_seed', config, int)[0]
        image['random_seed'] = { 'type' : 'Sequence' , 'first' : first_seed }

    if 'draw_method' not in image:
        image['draw_method'] = 'fft'

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


def _set_image_origin(config, convention):
    """Set config['image_origin'] appropriately based on the provided convention.
    """
    if convention.lower() in [ '0', 'c', 'python' ]:
        origin = 0
    elif convention.lower() in [ '1', 'fortran', 'fits' ]:
        origin = 1
    else:
        raise AttributeError("Unknown index_convention: %s"%convention)
    config['image_origin'] = galsim.PositionI(origin,origin)


def BuildSingleImage(config, logger=None, image_num=0, obj_num=0,
                     make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image consisting of a single stamp

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress.
    @param image_num           If given, the current image_num (default = 0)
    @param obj_num             If given, the current obj_num (default = 0)
    @param make_psf_image      Whether to make psf_image.
    @param make_weight_image   Whether to make weight_image.
    @param make_badpix_image   Whether to make badpix_image.

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    """
    config['seq_index'] = image_num
    config['image_num'] = image_num
    if logger:
        logger.debug('image %d: BuildSingleImage: image, obj = %d,%d',
                      config['image_num'],image_num,obj_num)

    ignore = [ 'random_seed', 'draw_method', 'noise', 'wcs', 'nproc' ,
               'n_photons', 'wmult', 'offset', 'gsparams' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int , 'index_convention' : str,
            'pixel_scale' : float , 'sky_level' : float , 'sky_level_pixel' : float }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, opt=opt, ignore=ignore)[0]

    convention = params.get('index_convention','1')
    _set_image_origin(config,convention)

    # If image_force_xsize and image_force_ysize were set in config, this overrides the 
    # read-in params.
    if 'image_force_xsize' in config and 'image_force_ysize' in config:
        xsize = config['image_force_xsize']
        ysize = config['image_force_ysize']
    else:
        size = params.get('size',0)
        xsize = params.get('xsize',size)
        ysize = params.get('ysize',size)
    config['image_xsize'] = xsize
    config['image_ysize'] = ysize

    if (xsize == 0) != (ysize == 0):
        raise AttributeError(
            "Both (or neither) of image.xsize and image.ysize need to be defined  and != 0.")

    if 'sky_pos' in config['image']:
        config['image']['image_pos'] = (0,0)
        # We allow sky_pos to be in config[image], but we don't want it to lead to a final_shift
        # in BuildSingleStamp.  The easiest way to do this is to set image_pos to (0,0).

    pixel_scale = params.get('pixel_scale',1.0)
    config['pixel_scale'] = pixel_scale
    if 'pix' not in config:
        config['pix'] = { 'type' : 'Pixel' , 'xw' : pixel_scale }

    if 'sky_level' in params and 'sky_level_pixel' in params:
        raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
            "noise.type = %s"%type)
    sky_level_pixel = params.get('sky_level_pixel',None)
    if 'sky_level' in params:
        sky_level_pixel = params['sky_level'] * pixel_scale**2

    return galsim.config.BuildSingleStamp(
            config=config, xsize=xsize, ysize=ysize, obj_num=obj_num,
            sky_level_pixel=sky_level_pixel, do_noise=True, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)


def BuildTiledImage(config, logger=None, image_num=0, obj_num=0,
                    make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image consisting of a tiled array of postage stamps

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress.
    @param image_num           If given, the current image_num (default = 0)
    @param obj_num             If given, the current obj_num (default = 0)
    @param make_psf_image      Whether to make psf_image.
    @param make_weight_image   Whether to make weight_image.
    @param make_badpix_image   Whether to make badpix_image.

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    """
    config['seq_index'] = image_num
    config['image_num'] = image_num
    if logger:
        logger.debug('image %d: BuildTiledImage: image, obj = %d,%d',
                      config['image_num'],image_num,obj_num)

    ignore = [ 'random_seed', 'draw_method', 'noise', 'wcs', 'nproc' ,
               'image_pos', 'n_photons', 'wmult', 'offset', 'gsparams' ]
    req = { 'nx_tiles' : int , 'ny_tiles' : int }
    opt = { 'stamp_size' : int , 'stamp_xsize' : int , 'stamp_ysize' : int ,
            'border' : int , 'xborder' : int , 'yborder' : int ,
            'pixel_scale' : float , 'nproc' : int , 'index_convention' : str,
            'sky_level' : float , 'sky_level_pixel' : float , 'order' : str }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, req=req, opt=opt, ignore=ignore)[0]

    nx_tiles = params['nx_tiles']
    ny_tiles = params['ny_tiles']
    nobjects = nx_tiles * ny_tiles
    config['nx_tiles'] = nx_tiles
    config['ny_tiles'] = ny_tiles

    stamp_size = params.get('stamp_size',0)
    stamp_xsize = params.get('stamp_xsize',stamp_size)
    stamp_ysize = params.get('stamp_ysize',stamp_size)
    config['tile_xsize'] = stamp_xsize
    config['tile_ysize'] = stamp_ysize

    convention = params.get('index_convention','1')
    _set_image_origin(config,convention)

    if (stamp_xsize == 0) or (stamp_ysize == 0):
        raise AttributeError(
            "Both image.stamp_xsize and image.stamp_ysize need to be defined and != 0.")

    border = params.get("border",0)
    xborder = params.get("xborder",border)
    yborder = params.get("yborder",border)

    pixel_scale = params.get('pixel_scale',1.0)
    config['pixel_scale'] = pixel_scale

    if 'sky_level' in params and 'sky_level_pixel' in params:
        raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
            "noise.type = %s"%type)
    sky_level_pixel = params.get('sky_level_pixel',None)
    if 'sky_level' in params:
        sky_level_pixel = params['sky_level'] * pixel_scale**2

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
    config['image_xsize'] = full_xsize
    config['image_ysize'] = full_ysize

    if 'pix' not in config:
        config['pix'] = { 'type' : 'Pixel' , 'xw' : pixel_scale }

    # Set the rng to use for image stuff.
    if 'random_seed' in config['image']:
        config['seq_index'] = obj_num+nobjects
        config['obj_num'] = obj_num+nobjects
        # Technically obj_num+nobjects will be the index of the random seed used for the next 
        # image's first object (if there is a next image).  But I don't think that will have 
        # any adverse effects.
        seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        if logger:
            logger.debug('image %d: seed = %d',image_num,seed)
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate()
    config['rng'] = rng

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

    full_image = galsim.ImageF(full_xsize, full_ysize, scale=pixel_scale)
    full_image.setOrigin(config['image_origin'])
    full_image.setZero()

    # Also define the overall image center, since we need that to calculate the position 
    # of each stamp relative to the center.
    config['image_center'] = full_image.bounds.trueCenter()
    if logger:
        logger.debug('image %d: image_center = %s',config['image_num'],str(config['image_center']))

    if make_psf_image:
        full_psf_image = galsim.ImageF(full_xsize, full_ysize, scale=pixel_scale)
        full_psf_image.setOrigin(config['image_origin'])
        full_psf_image.setZero()
    else:
        full_psf_image = None

    if make_weight_image:
        full_weight_image = galsim.ImageF(full_xsize, full_ysize, scale=pixel_scale)
        full_weight_image.setOrigin(config['image_origin'])
        full_weight_image.setZero()
    else:
        full_weight_image = None

    if make_badpix_image:
        full_badpix_image = galsim.ImageS(full_xsize, full_ysize, scale=pixel_scale)
        full_badpix_image.setOrigin(config['image_origin'])
        full_badpix_image.setZero()
    else:
        full_badpix_image = None

    # Sometimes an input field needs to do something special at the start of an image.
    if 'input' in config:
        for key in [ k for k in galsim.config.valid_input_types.keys() if k in config['input'] ]:
            if galsim.config.valid_input_types[key][3]:
                assert key in config
                fields = config['input'][key]
                if not isinstance(fields, list):
                    fields = [ fields ]
                input_objs = config[key]

                for i in range(len(fields)):
                    field = fields[i]
                    input_obj = input_objs[i]
                    func = eval(galsim.config.valid_input_types[key][3])
                    func(input_obj, field, config)

    stamp_images = galsim.config.BuildStamps(
            nobjects=nobjects, config=config,
            nproc=nproc, logger=logger, obj_num=obj_num,
            xsize=stamp_xsize, ysize=stamp_ysize,
            sky_level_pixel=sky_level_pixel, do_noise=do_noise,
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
            logger.debug('image %d: full bounds = %s',config['image_num'],str(full_image.bounds))
            logger.debug('image %d: stamp %d bounds = %s',
                         config['image_num'],k,str(images[k].bounds))
        assert full_image.bounds.includes(images[k].bounds)
        b = images[k].bounds
        full_image[b] += images[k]
        if make_psf_image:
            full_psf_image[b] += psf_images[k]
        if make_weight_image:
            full_weight_image[b] += weight_images[k]
        if make_badpix_image:
            full_badpix_image[b] |= badpix_images[k]
        if current_vars[k] > max_current_var: max_current_var = current_vars[k]

    if not do_noise:
        if 'noise' in config['image']:
            # If we didn't apply noise in each stamp, then we need to apply it now.
            draw_method = galsim.config.GetCurrentValue(config['image'],'draw_method')

            if max_current_var > 0:
                import numpy
                # Then there was whitening applied in the individual stamps.
                # But there could be a different variance in each postage stamp, so the first
                # thing we need to do is bring everything up to a common level.
                noise_image = galsim.ImageF(full_image.bounds, full_image.scale)
                for k in range(nobjects): noise_image[images[k].bounds] += current_vars[k]
                # Update this, since overlapping postage stamps may have led to a larger 
                # value in some pixels.
                max_current_var = numpy.max(noise_image.array)
                # Figure out how much noise we need to add to each pixel.
                noise_image = max_current_var - noise_image
                # Add it.
                full_image.addNoise(galsim.VariableGaussianNoise(rng,noise_image))
            # Now max_current_var is how much noise is in each pixel.

            if draw_method == 'fft':
                galsim.config.AddNoiseFFT(
                    full_image,full_weight_image,max_current_var,config['image']['noise'],config,
                    rng,sky_level_pixel)
            elif draw_method == 'phot':
                galsim.config.AddNoisePhot(
                    full_image,full_weight_image,max_current_var,config['image']['noise'],config,
                    rng,sky_level_pixel)
            else:
                raise AttributeError("Unknown draw_method %s."%draw_method)
        elif sky_level_pixel:
            # If we aren't doing noise, we still need to add a non-zero sky_level
            full_image += sky_level_pixel

    return full_image, full_psf_image, full_weight_image, full_badpix_image


def BuildScatteredImage(config, logger=None, image_num=0, obj_num=0,
                        make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image containing multiple objects placed at arbitrary locations.

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress.
    @param image_num           If given, the current image_num (default = 0)
    @param obj_num             If given, the current obj_num (default = 0)
    @param make_psf_image      Whether to make psf_image.
    @param make_weight_image   Whether to make weight_image.
    @param make_badpix_image   Whether to make badpix_image.

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    """
    config['seq_index'] = image_num
    config['image_num'] = image_num
    if logger:
        logger.debug('image %d: BuildScatteredImage: image, obj = %d,%d',
                      config['image_num'],image_num,obj_num)

    nobjects = GetNObjForScatteredImage(config,image_num)

    ignore = [ 'random_seed', 'draw_method', 'noise', 'wcs', 'nproc' ,
               'image_pos', 'sky_pos', 'n_photons', 'wmult', 'offset',
               'stamp_size', 'stamp_xsize', 'stamp_ysize', 'gsparams', 'nobjects' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int , 
            'pixel_scale' : float , 'nproc' : int , 'index_convention' : str,
            'sky_level' : float , 'sky_level_pixel' : float }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, opt=opt, ignore=ignore)[0]

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

    pixel_scale = params.get('pixel_scale',1.0)
    config['pixel_scale'] = pixel_scale

    convention = params.get('index_convention','1')
    _set_image_origin(config,convention)

    if 'sky_level' in params and 'sky_level_pixel' in params:
        raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
            "noise.type = %s"%type)
    sky_level_pixel = params.get('sky_level_pixel',None)
    if 'sky_level' in params:
        sky_level_pixel = params['sky_level'] * pixel_scale**2

    # If image_force_xsize and image_force_ysize were set in config, make sure it matches.
    if ( ('image_force_xsize' in config and full_xsize != config['image_force_xsize']) or
         ('image_force_ysize' in config and full_ysize != config['image_force_ysize']) ):
        raise ValueError(
            "Unable to reconcile required image xsize and ysize with provided "+
            "xsize=%d, ysize=%d, "%(full_xsize,full_ysize))
    config['image_xsize'] = full_xsize
    config['image_ysize'] = full_ysize

    if 'pix' not in config:
        config['pix'] = { 'type' : 'Pixel' , 'xw' : pixel_scale }

    # Set the rng to use for image stuff.
    if 'random_seed' in config['image']:
        config['seq_index'] = obj_num+nobjects
        config['obj_num'] = obj_num+nobjects
        # Technically obj_num+nobjects will be the index of the random seed used for the next 
        # image's first object (if there is a next image).  But I don't think that will have 
        # any adverse effects.
        seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        if logger:
            logger.debug('image %d: seed = %d',image_num,seed)
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate()
    config['rng'] = rng

    if 'image_pos' in config['image'] and 'sky_pos' in config['image']:
        raise AttributeError("Both image_pos and sky_pos specified for Scattered image.")

    if 'image_pos' not in config['image'] and 'sky_pos' not in config['image']:
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

    full_image = galsim.ImageF(full_xsize, full_ysize, scale=pixel_scale)
    full_image.setOrigin(config['image_origin'])
    full_image.setZero()

    # Also define the overall image center, since we need that to calculate the position 
    # of each stamp relative to the center.
    config['image_center'] = full_image.bounds.trueCenter()
    if logger:
        logger.debug('image %d: image_center = %s',config['image_num'],str(config['image_center']))

    if make_psf_image:
        full_psf_image = galsim.ImageF(full_xsize, full_ysize, scale=pixel_scale)
        full_psf_image.setOrigin(config['image_origin'])
        full_psf_image.setZero()
    else:
        full_psf_image = None

    if make_weight_image:
        full_weight_image = galsim.ImageF(full_xsize, full_ysize, scale=pixel_scale)
        full_weight_image.setOrigin(config['image_origin'])
        full_weight_image.setZero()
    else:
        full_weight_image = None

    if make_badpix_image:
        full_badpix_image = galsim.ImageS(full_xsize, full_ysize, scale=pixel_scale)
        full_badpix_image.setOrigin(config['image_origin'])
        full_badpix_image.setZero()
    else:
        full_badpix_image = None

    # Sometimes an input field needs to do something special at the start of an image.
    if 'input' in config:
        for key in [ k for k in galsim.config.valid_input_types.keys() if k in config['input'] ]:
            if galsim.config.valid_input_types[key][3]:
                assert key in config
                fields = config['input'][key]
                if not isinstance(fields, list):
                    fields = [ fields ]
                input_objs = config[key]

                for i in range(len(fields)):
                    field = fields[i]
                    input_obj = input_objs[i]
                    func = eval(galsim.config.valid_input_types[key][3])
                    func(input_obj, field, config)

    stamp_images = galsim.config.BuildStamps(
            nobjects=nobjects, config=config,
            nproc=nproc, logger=logger,obj_num=obj_num,
            sky_level_pixel=sky_level_pixel, do_noise=False,
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
            logger.debug('image %d: full bounds = %s',config['image_num'],str(full_image.bounds))
            logger.debug('image %d: stamp %d bounds = %s',
                         config['image_num'],k,str(images[k].bounds))
            logger.debug('image %d: Overlap = %s',config['image_num'],str(bounds))
        if bounds.isDefined():
            full_image[bounds] += images[k][bounds]
            if make_psf_image:
                full_psf_image[bounds] += psf_images[k][bounds]
            if make_weight_image:
                full_weight_image[bounds] += weight_images[k][bounds]
            if make_badpix_image:
                full_badpix_image[bounds] |= badpix_images[k][bounds]
        else:
            if logger:
                logger.warn(
                    "Object centered at (%d,%d) is entirely off the main image,\n"%(
                        images[k].bounds.center().x, images[k].bounds.center().y) +
                    "whose bounds are (%d,%d,%d,%d)."%(
                        full_image.bounds.xmin, full_image.bounds.xmax,
                        full_image.bounds.ymin, full_image.bounds.ymax))
        if current_vars[k] > max_current_var: max_current_var = current_vars[k]

    if 'noise' in config['image']:
        # Apply the noise to the full image
        draw_method = galsim.config.GetCurrentValue(config['image'],'draw_method')
        if max_current_var > 0:
            import numpy
            # Then there was whitening applied in the individual stamps.
            # But there could be a different variance in each postage stamp, so the first
            # thing we need to do is bring everything up to a common level.
            noise_image = galsim.ImageF(full_image.bounds, full_image.scale)
            for k in range(nobjects): 
                b = images[k].bounds & full_image.bounds
                if b.isDefined(): noise_image[b] += current_vars[k]
            # Update this, since overlapping postage stamps may have led to a larger 
            # value in some pixels.
            max_current_var = numpy.max(noise_image.array)
            # Figure out how much noise we need to add to each pixel.
            noise_image = max_current_var - noise_image
            # Add it.
            full_image.addNoise(galsim.VariableGaussianNoise(rng,noise_image))
        # Now max_current_var is how much noise is in each pixel.

        if draw_method == 'fft':
            galsim.config.AddNoiseFFT(
                full_image,full_weight_image,max_current_var,config['image']['noise'],config,
                rng,sky_level_pixel)
        elif draw_method == 'phot':
            galsim.config.AddNoisePhot(
                full_image,full_weight_image,max_current_var,config['image']['noise'],config,
                rng,sky_level_pixel)
        else:
            raise AttributeError("Unknown draw_method %s."%draw_method)

    elif sky_level_pixel:
        # If we aren't doing noise, we still need to add a non-zero sky_level
        full_image += sky_level_pixel

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

    config['seq_index'] = image_num
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
    
    config['seq_index'] = image_num
    config['image_num'] = image_num

    if 'nx_tiles' not in config['image'] or 'ny_tiles' not in config['image']:
        raise AttributeError(
            "Attributes nx_tiles and ny_tiles are required for image.type = Tiled")
    nx = galsim.config.ParseValue(config['image'],'nx_tiles',config,int)[0]
    ny = galsim.config.ParseValue(config['image'],'ny_tiles',config,int)[0]
    return nx*ny

def PowerSpectrumInit(ps, config, base):
    if 'grid_spacing' in config:
        grid_spacing = galsim.config.ParseValue(config, 'grid_spacing', base, float)[0]
    elif 'tile_xsize' in base:
        # Then we have a tiled image.  Can use the tile spacing as the grid spacing.
        stamp_size = min(base['tile_xsize'], base['tile_ysize'])
        grid_spacing = stamp_size * base['pixel_scale']
    else:
        raise AttributeError("power_spectrum.grid_spacing required for non-tiled images")

    if 'tile_xsize' in base and base['tile_xsize'] == base['tile_ysize']:
        # PowerSpectrum can only do a square FFT, so make it the larger of the two n's.
        ngrid = max(base['nx_tiles'], base['ny_tiles'])
        # Normally that's good, but if tiles aren't square, need to drop through to the
        # second option.
    else:
        import math
        image_size = max(base['image_xsize'], base['image_ysize'])
        ngrid = int(math.ceil(image_size * base['pixel_scale'] / grid_spacing))

    if 'interpolant' in config:
        interpolant = galsim.config.ParseValue(config, 'interpolant', base, str)[0]
    else:
        interpolant = None

    # We don't care about the output here.  This just builds the grid, which we'll
    # access for each object using its position.
    ps.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid, rng=base['rng'], interpolant=interpolant)


