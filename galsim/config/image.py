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
    'Single' : 'BuildSingleImage',
    'Tiled' : 'BuildTiledImage',
    'Scattered' : 'BuildScatteredImage',
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
    import time
    def worker(input, output):
        for (kwargs, config, image_num, obj_num, nim, info) in iter(input.get, 'STOP'):
            results = []
            # Make new copies of config and kwargs so we can update them without
            # clobbering the versions for other tasks on the queue.
            import copy
            kwargs1 = copy.copy(kwargs)
            config1 = copy.deepcopy(config)
            for i in range(nim):
                t1 = time.time()
                kwargs1['config'] = config1
                kwargs1['image_num'] = image_num + i
                kwargs1['obj_num'] = obj_num
                im = BuildImage(**kwargs1)
                obj_num += galsim.config.GetNObjForImage(config, image_num+i)
                t2 = time.time()
                results.append( [im[0], im[1], im[2], im[3], t2-t1 ] )
            output.put( (results, info, current_process().name) )
    
    # The kwargs to pass to BuildImage
    kwargs = {
        'make_psf_image' : make_psf_image,
        'make_weight_image' : make_weight_image,
        'make_badpix_image' : make_badpix_image
    }
    # Apparently the logger isn't picklable, so can't send that as an arg.

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
        #print 'gal' in config
        if ( ('image' not in config or 'type' not in config['image'] or 
                 config['image']['type'] == 'Single') and
             'gal' in config and isinstance(config['gal'],dict) and 'type' in config['gal'] and
             config['gal']['type'] == 'Ring' and 'num' in config['gal'] ):
            min_nim = galsim.config.ParseValue(config['gal'], 'num', config, int)[0]
            #print 'Found ring: num = ',min_nim
        if max_nim < min_nim: 
            nim_per_task = min_nim
        else:
            import math
            # This formula keeps nim a multiple of min_nim, so Rings are intact.
            nim_per_task = min_nim * int(math.sqrt(float(max_nim) / float(min_nim)))
        #print 'nim_per_task = ',nim_per_task

        # Set up the task list
        task_queue = Queue()
        for k in range(0,nimages,nim_per_task):
            # Send kwargs, config, im_num, nim, k
            if k + nim_per_task > nimages:
                task_queue.put( ( kwargs, config, image_num+k, obj_num, nimages-k, k ) )
            else:
                task_queue.put( ( kwargs, config, image_num+k, obj_num, nim_per_task, k ) )
            for i in range(nim_per_task):
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
        # You'll see that these logging statements get print out as the stamp images are still 
        # being drawn.  
        for i in range(0,nimages,nim_per_task):
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
                    logger.info('%s: Image %d: size = %d x %d, time = %f sec', 
                                proc, image_num+k, xs, ys, t)
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
        logger.debug('Done making images')

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

    build_func = eval(valid_image_types[type])
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

    ignore = [ 'random_seed', 'draw_method', 'noise', 'wcs', 'nproc' ,
               'n_photons', 'wmult', 'gsparams' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int , 'index_convention' : str,
            'pixel_scale' : float , 'sky_level' : float , 'sky_level_pixel' : float }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, opt=opt, ignore=ignore)[0]

    convention = params.get('index_convention','1')
    _set_image_origin(config,convention)

    # If image_xsize and image_ysize were set in config, this overrides the read-in params.
    if 'image_xsize' in config and 'image_ysize' in config:
        xsize = config['image_xsize']
        ysize = config['image_ysize']
    else:
        size = params.get('size',0)
        xsize = params.get('xsize',size)
        ysize = params.get('ysize',size)

    if (xsize == 0) != (ysize == 0):
        raise AttributeError(
            "Both (or neither) of image.xsize and image.ysize need to be defined  and != 0.")

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

    ignore = [ 'random_seed', 'draw_method', 'noise', 'wcs', 'nproc' ,
               'image_pos', 'n_photons', 'wmult', 'gsparams' ]
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

    stamp_size = params.get('stamp_size',0)
    stamp_xsize = params.get('stamp_xsize',stamp_size)
    stamp_ysize = params.get('stamp_ysize',stamp_size)

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

    # If image_xsize and image_ysize were set in config, make sure it matches.
    if ( 'image_xsize' in config and 'image_ysize' in config and
         (full_xsize != config['image_xsize'] or full_ysize != config['image_ysize']) ):
        raise ValueError(
            "Unable to reconcile saved image_xsize and image_ysize with provided "+
            "nx_tiles=%d, ny_tiles=%d, "%(nx_tiles,ny_tiles) +
            "xborder=%d, yborder=%d\n"%(xborder,yborder) +
            "Calculated full_size = (%d,%d) "%(full_xsize,full_ysize)+
            "!= required (%d,%d)."%(config['image_xsize'],config['image_ysize']))

    if 'pix' not in config:
        config['pix'] = { 'type' : 'Pixel' , 'xw' : pixel_scale }

    # Set the rng to use for image stuff.
    if 'random_seed' in config['image']:
        config['seq_index'] = obj_num+nobjects
        # Technically obj_num+nobjects will be the index of the random seed used for the next 
        # image's first object (if there is a next image).  But I don't think that will have 
        # any adverse effects.
        seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        #print 'seed = ',seed
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate()

    # If we have a power spectrum in config, we need to get a new realization at the start
    # of each image.
    if 'power_spectrum' in config:
        # PowerSpectrum can only do a square FFT, so make it the larger of the two n's.
        n_tiles = max(nx_tiles, ny_tiles)
        stamp_size = max(stamp_xsize, stamp_ysize)
        if 'grid_spacing' in config['input']['power_spectrum']:
            grid_dx = galsim.config.ParseValue(config['input']['power_spectrum'],
                                               'grid_spacing', config, float)[0]
        else:
            grid_dx = stamp_size * pixel_scale
        if 'interpolant' in config['input']['power_spectrum']:
            interpolant = galsim.config.ParseValue(config['input']['power_spectrum'],
                                                   'interpolant', config, str)[0]
        else:
            interpolant = None

        config['power_spectrum'].buildGrid(grid_spacing=grid_dx, ngrid=n_tiles, rng=rng,
                                           interpolant=interpolant)
        # We don't care about the output here.  This just builds the grid, which we'll
        # access for each object using its position.

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

    full_image = galsim.ImageF(full_xsize,full_ysize)
    full_image.setOrigin(config['image_origin'])
    full_image.setZero()
    full_image.setScale(pixel_scale)

    # Also define the overall image center, since we need that to calculate the position 
    # of each stamp relative to the center.
    config['image_cen'] = full_image.bounds.trueCenter()
    #print 'image_cen = ',full_image.bounds.trueCenter()

    if make_psf_image:
        full_psf_image = galsim.ImageF(full_xsize,full_ysize)
        full_psf_image.setOrigin(config['image_origin'])
        full_psf_image.setZero()
        full_psf_image.setScale(pixel_scale)
    else:
        full_psf_image = None

    if make_weight_image:
        full_weight_image = galsim.ImageF(full_xsize,full_ysize)
        full_weight_image.setOrigin(config['image_origin'])
        full_weight_image.setZero()
        full_weight_image.setScale(pixel_scale)
    else:
        full_weight_image = None

    if make_badpix_image:
        full_badpix_image = galsim.ImageS(full_xsize,full_ysize)
        full_badpix_image.setOrigin(config['image_origin'])
        full_badpix_image.setZero()
        full_badpix_image.setScale(pixel_scale)
    else:
        full_badpix_image = None

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

    for k in range(nobjects):
        ix = ix_list[k]
        iy = iy_list[k]
        xmin = ix * (stamp_xsize + xborder) + 1
        xmax = xmin + stamp_xsize-1
        ymin = iy * (stamp_ysize + yborder) + 1
        ymax = ymin + stamp_ysize-1
        b = galsim.BoundsI(xmin,xmax,ymin,ymax)
        #print 'full bounds = ',full_image.bounds
        #print 'stamp bounds = ',b
        #print 'original stamp bounds = ',images[k].bounds
        full_image[b] += images[k]
        if make_psf_image:
            full_psf_image[b] += psf_images[k]
        if make_weight_image:
            full_weight_image[b] += weight_images[k]
        if make_badpix_image:
            full_badpix_image[b] |= badpix_images[k]

    if not do_noise:
        if 'noise' in config['image']:
            # If we didn't apply noise in each stamp, then we need to apply it now.
            draw_method = galsim.config.GetCurrentValue(config['image'],'draw_method')
            if draw_method == 'fft':
                galsim.config.AddNoiseFFT(
                    full_image,full_weight_image,config['image']['noise'],config,rng,
                    sky_level_pixel)
            elif draw_method == 'phot':
                galsim.config.AddNoisePhot(
                    full_image,full_weight_image,config['image']['noise'],config,rng,
                    sky_level_pixel)
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

    ignore = [ 'random_seed', 'draw_method', 'noise', 'wcs', 'nproc' ,
               'image_pos', 'sky_pos', 'n_photons', 'wmult',
               'stamp_size', 'stamp_xsize', 'stamp_ysize', 'gsparams' ]
    req = { 'nobjects' : int }
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int , 
            'pixel_scale' : float , 'nproc' : int , 'index_convention' : str,
            'sky_level' : float , 'sky_level_pixel' : float }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, req=req, opt=opt, ignore=ignore)[0]

    nobjects = params['nobjects']

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

    # If image_xsize and image_ysize were set in config, make sure it matches.
    if ( 'image_xsize' in config and 'image_ysize' in config and
         (full_xsize != config['image_xsize'] or full_ysize != config['image_ysize']) ):
        raise ValueError(
            "Unable to reconcile saved image_xsize and image_ysize with provided "+
            "xsize=%d, ysize=%d, "%(full_xsize,full_ysize))

    if 'pix' not in config:
        config['pix'] = { 'type' : 'Pixel' , 'xw' : pixel_scale }

    # Set the rng to use for image stuff.
    if 'random_seed' in config['image']:
        #print 'random_seed = ',config['image']['random_seed']
        config['seq_index'] = obj_num+nobjects
        #print 'seq_index = ',config['seq_index']
        # Technically obj_num+nobjects will be the index of the random seed used for the next 
        # image's first object (if there is a next image).  But I don't think that will have 
        # any adverse effects.
        seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        #print 'seed = ',seed
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate()

    # If we have a power spectrum in config, we need to get a new realization at the start
    # of each image.
    if 'power_spectrum' in config:
        if 'grid_spacing' not in config['input']['power_spectrum']:
            raise AttributeError(
                "power_spectrum.grid_spacing required for image.type=Scattered")
        grid_dx = galsim.config.ParseValue(config['input']['power_spectrum'],
                                           'grid_spacing', config, float)[0]
        full_size = max(full_xsize, full_ysize)
        grid_nx = full_size * pixel_scale / grid_dx + 1
        if 'interpolant' in config['input']['power_spectrum']:
            interpolant = galsim.config.ParseValue(config['input']['power_spectrum'],
                                                   'interpolant', config, str)[0]
        else:
            interpolant = None

        config['power_spectrum'].buildGrid(grid_spacing=grid_dx, ngrid=grid_nx, rng=rng,
                                           interpolant=interpolant)
        # We don't care about the output here.  This just builds the grid, which we'll
        # access for each object using its position.

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

    full_image = galsim.ImageF(full_xsize,full_ysize)
    full_image.setOrigin(config['image_origin'])
    full_image.setZero()
    full_image.setScale(pixel_scale)

    # Also define the overall image center, since we need that to calculate the position 
    # of each stamp relative to the center.
    config['image_cen'] = full_image.bounds.trueCenter()
    #print 'image_cen = ',full_image.bounds.trueCenter()

    if make_psf_image:
        full_psf_image = galsim.ImageF(full_xsize,full_ysize)
        full_psf_badpix_image.setOrigin(config['image_origin'])
        full_psf_image.setZero()
        full_psf_image.setScale(pixel_scale)
    else:
        full_psf_image = None

    if make_weight_image:
        full_weight_image = galsim.ImageF(full_xsize,full_ysize)
        full_weight_image.setOrigin(config['image_origin'])
        full_weight_image.setZero()
        full_weight_image.setScale(pixel_scale)
    else:
        full_weight_image = None

    if make_badpix_image:
        full_badpix_image = galsim.ImageS(full_xsize,full_ysize)
        full_badpix_image.setOrigin(config['image_origin'])
        full_badpix_image.setZero()
        full_badpix_image.setScale(pixel_scale)
    else:
        full_badpix_image = None

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

    for k in range(nobjects):
        bounds = images[k].bounds & full_image.bounds
        #print 'stamp bounds = ',images[k].bounds
        #print 'full bounds = ',full_image.bounds
        #print 'Overlap = ',bounds
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

    if 'noise' in config['image']:
        # Apply the noise to the full image
        draw_method = galsim.config.GetCurrentValue(config['image'],'draw_method')
        if draw_method == 'fft':
            galsim.config.AddNoiseFFT(
                full_image,full_weight_image,config['image']['noise'],config,rng,sky_level_pixel)
        elif draw_method == 'phot':
            galsim.config.AddNoisePhot(
                full_image,full_weight_image,config['image']['noise'],config,rng,sky_level_pixel)
        else:
            raise AttributeError("Unknown draw_method %s."%draw_method)

    elif sky_level_pixel:
        # If we aren't doing noise, we still need to add a non-zero sky_level
        full_image += sky_level_pixel

    return full_image, full_psf_image, full_weight_image, full_badpix_image



