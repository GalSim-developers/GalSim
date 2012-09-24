
import galsim

def BuildImage(config, logger=None, seeds=None,
               make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image according the information in config.

    This function acts as a wrapper for:
        BuildSingleImage 
        BuildTiledImage 
        BuildScatteredImage 
    choosing between these three using the contents of config if specified (default = Single)

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress.
    @param seeds               If given, a list of seeds to use
    @param make_psf_image      Whether to make psf_image
    @param make_weight_image   Whether to make weight_image
    @param make_badpix_image   Whether to make badpix_image

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

    #print 'BuildImage: seeds = ',seeds
    #print 'len(seeds) = ',len(seeds)
    if seeds:
        image['random_seed'] = { 'type' : 'List' , 'items' : seeds }
    else:
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

    valid_types = [ 'Single', 'Tiled', 'Scattered' ]
    if type not in valid_types:
        raise AttributeError("Invalid image.type=%s."%type)

    build_func = eval('Build' + type + 'Image')
    all_images = build_func(
            config=config, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)

    # The later image building functions build up the weight image as the total variance 
    # in each pixel.  We need to invert this to produce the inverse variance map.
    # Doing it here means it only needs to be done in this one place.
    if all_images[2]:
        all_images[2].invertSelf()

    return all_images


def BuildSingleImage(config, logger=None,
                     make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image consisting of a single stamp

    @param config     A configuration dict.
    @param logger     If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image
    @param make_weight_image   Whether to make weight_image
    @param make_badpix_image   Whether to make badpix_image

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    """
    ignore = [ 'draw_method', 'noise', 'wcs', 'nproc' ]
    opt = { 'random_seed' : int , 'size' : int , 'xsize' : int , 'ysize' : int ,
            'pixel_scale' : float , 'sky_level' : float }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, opt=opt, ignore=ignore)[0]

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

    if 'random_seed' in params:
        seed = params['random_seed']
    else:
        seed = None

    sky_level = params.get('sky_level',None)

    return galsim.config.BuildSingleStamp(
            seed=seed, config=config, xsize=xsize, ysize=ysize, 
            sky_level=sky_level, do_noise=True, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)


def BuildTiledImage(config, logger=None,
                    make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image consisting of a tiled array of postage stamps

    @param config     A configuration dict.
    @param logger     If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image
    @param make_weight_image   Whether to make weight_image
    @param make_badpix_image   Whether to make badpix_image

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    """
    ignore = [ 'random_seed', 'draw_method', 'noise', 'wcs', 'nproc', 'center' ]
    req = { 'nx_tiles' : int , 'ny_tiles' : int }
    opt = { 'stamp_size' : int , 'stamp_xsize' : int , 'stamp_ysize' : int ,
            'border' : int , 'xborder' : int , 'yborder' : int ,
            'pixel_scale' : float , 'nproc' : int , 'sky_level' : float }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, req=req, opt=opt, ignore=ignore)[0]

    nx_tiles = params['nx_tiles']
    ny_tiles = params['ny_tiles']
    nstamps = nx_tiles * ny_tiles

    stamp_size = params.get('stamp_size',0)
    stamp_xsize = params.get('stamp_xsize',stamp_size)
    stamp_ysize = params.get('stamp_ysize',stamp_size)

    if (stamp_xsize == 0) != (stamp_ysize == 0):
        raise AttributeError(
            "Both (or neither) of image.stamp_xsize and image.stamp_ysize need to be "+
            "defined and != 0.")

    border = params.get("border",0)
    xborder = params.get("xborder",border)
    yborder = params.get("yborder",border)

    sky_level = params.get('sky_level',None)

    do_noise = xborder >= 0 and yborder >= 0
    # TODO: Note: if one of these is < 0 and the other is > 0, then
    #       this will add noise to the border region.  Not exactly the 
    #       design, but I didn't bother to do the bookkeeping right to 
    #       make the borders pure 0 in that case.

    # If image_xsize and image_ysize were set in config, this overrides the read-in params.
    if 'image_xsize' in config and 'image_ysize' in config:
        stamp_xsize = (config['image_xsize']+xborder) / nx_tiles - xborder
        stamp_ysize = (config['image_ysize']+yborder) / ny_tiles - yborder
        full_xsize = (stamp_xsize + xborder) * nx_tiles - xborder
        full_ysize = (stamp_ysize + yborder) * ny_tiles - yborder
        if ( full_xsize != config['image_xsize'] or full_ysize != config['image_ysize'] ):
            raise ValueError(
                "Unable to reconcile saved image_xsize and image_ysize with current "+
                "nx_tiles=%d, ny_tiles=%d, "%(nx_tiles,ny_tiles) +
                "xborder=%d, yborder=%d\n"%(xborder,yborder) +
                "Calculated full_size = (%d,%d) "%(full_xsize,full_ysize)+
                "!= required (%d,%d)."%(config['image_xsize'],config['image_ysize']))

    if stamp_xsize == 0:
        if 'random_seed' in config['image']:
            seed = galsim.config.ParseValue(config['image'],'random_seed',config,int)[0]
            #print 'First tile: seed = ',seed
        else:
            seed = None
        first_images = galsim.config.BuildSingleStamp(
            seed=seed, config=config, xsize=stamp_xsize, ysize=stamp_ysize, 
            sky_level=sky_level, do_noise=do_noise, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)
        # Note: numpy shape is y,x
        stamp_ysize, stamp_xsize = first_images[0].array.shape
        images = [ first_images[0] ]
        psf_images = [ first_images[1] ]
        weight_images = [ first_images[2] ]
        badpix_images = [ first_images[3] ]
        nstamps -= 1
    else:
        images = []
        psf_images = []
        weight_images = []
        badpix_images = []

    full_xsize = (stamp_xsize + xborder) * nx_tiles - xborder
    full_ysize = (stamp_ysize + yborder) * ny_tiles - yborder

    pixel_scale = params.get('pixel_scale',1.0)
    config['pixel_scale'] = pixel_scale
    if 'pix' not in config:
        config['pix'] = { 'type' : 'Pixel' , 'xw' : pixel_scale }

    # Define a 'center' field so the stamps can set their position appropriately in case
    # we need it for PowerSpectum or NFWHalo.
    config['image']['center'] = { 
        'type' : 'XY' ,
        'x' : { 'type' : 'Sequence' , 
                'first' : stamp_xsize/2+1 , 
                'step' : stamp_xsize + xborder ,
                'last' : full_xsize ,
                'repeat' : ny_tiles
              },
        'y' : { 'type' : 'Sequence' , 
                'first' : stamp_ysize/2+1 , 
                'step' : stamp_ysize + yborder ,
                'last' : full_ysize 
              },
    }

    # Set the rng to None.  We might create it next for building the power spectrum.
    # Or we might build it later for adding noise.  But we don't want to create two.
    rng = None

    if 'power_spectrum' in config:
        # Get the next random number seed.
        if 'random_seed' in config['image']:
            seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
            #print 'For power spectrum, seed = ',seed
            rng = galsim.BaseDeviate(seed)
        else:
            rng = galsim.BaseDeviate()
        # PowerSpectrum can only do a square FFT, so make it the larger of the two n's.
        n_tiles = max(nx_tiles, ny_tiles)
        stamp_size = max(stamp_xsize, stamp_ysize)
        grid_dx = stamp_size * pixel_scale
        #print 'n_tiles = ',n_tiles
        #print 'stamp_size = ',stamp_size

        g1,g2 = config['power_spectrum'].getShear(grid_spacing=grid_dx, grid_nx=n_tiles, rng=rng)
        #print 'g1,g2 = ',g1,g2
        # We don't care about the output here.  This just builds the grid, which we'll
        # access for each object using its position.
        #print 'g1.xvalue(0,0) = ',config['power_spectrum'].sbii_g1.xValue(galsim.PositionD(0,0))
        x = grid_dx
        y = grid_dx
        #print 'g1.xvalue(%f,%f) = '%(x,y),config['power_spectrum'].sbii_g1.xValue(galsim.PositionD(x,y))
        x = -(n_tiles/2) * grid_dx
        y = -(n_tiles/2) * grid_dx
        #print 'g1.xvalue(%f,%f) = '%(x,y),config['power_spectrum'].sbii_g1.xValue(galsim.PositionD(x,y))

    nproc = params.get('nproc',1)

    full_image = galsim.ImageF(full_xsize,full_ysize)
    full_image.setZero()
    full_image.setScale(pixel_scale)

    # Also define the overall image center, since we need that to calculate the position 
    # of each stamp relative to the center.
    image_cen = full_image.bounds.center()
    config['image_cen'] = galsim.PositionD(image_cen.x,image_cen.y)

    if make_psf_image:
        full_psf_image = galsim.ImageF(full_xsize,full_ysize)
        full_psf_image.setZero()
        full_psf_image.setScale(pixel_scale)
    else:
        full_psf_image = None

    if make_weight_image:
        full_weight_image = galsim.ImageF(full_xsize,full_ysize)
        full_weight_image.setZero()
        full_weight_image.setScale(pixel_scale)
    else:
        full_weight_image = None

    if make_badpix_image:
        full_badpix_image = galsim.ImageS(full_xsize,full_ysize)
        full_badpix_image.setZero()
        full_badpix_image.setScale(pixel_scale)
    else:
        full_badpix_image = None

    stamp_images = galsim.config.BuildStamps(
            nstamps=nstamps, config=config, xsize=stamp_xsize, ysize=stamp_ysize,
            nproc=nproc, sky_level=sky_level, do_noise=do_noise, logger=logger,
            make_psf_image=make_psf_image,
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)

    images += stamp_images[0]
    psf_images += stamp_images[1]
    weight_images += stamp_images[2]
    badpix_images += stamp_images[3]

    k = 0
    for ix in range(nx_tiles):
        for iy in range(ny_tiles):
            if k < len(images):
                xmin = ix * (stamp_xsize + xborder) + 1
                xmax = xmin + stamp_xsize-1
                ymin = iy * (stamp_ysize + yborder) + 1
                ymax = ymin + stamp_ysize-1
                b = galsim.BoundsI(xmin,xmax,ymin,ymax)
                full_image[b] += images[k]
                if make_psf_image:
                    full_psf_image[b] += psf_images[k]
                if make_weight_image:
                    full_weight_image[b] += weight_images[k]
                if make_badpix_image:
                    full_badpix_image[b] |= badpix_images[k]
                k = k+1

    if not do_noise:
        if 'noise' in config['image']:
            # If we didn't apply noise in each stamp, then we need to apply it now.

            if not rng:
                # Get the next random number seed.
                if 'random_seed' in config['image']:
                    seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
                    #print 'For noise, seed = ',seed
                    rng = galsim.BaseDeviate(seed)
                else:
                    rng = galsim.BaseDeviate()

            draw_method = galsim.config.GetCurrentValue(config['image'],'draw_method')
            if draw_method == 'fft':
                galsim.config.AddNoiseFFT(
                    full_image,full_weight_image,config['image']['noise'],rng,sky_level)
            elif draw_method == 'phot':
                galsim.config.AddNoisePhot(
                    full_image,full_weight_image,config['image']['noise'],rng,sky_level)
            else:
                raise AttributeError("Unknown draw_method %s."%draw_method)
        elif sky_level:
            # If we aren't doing noise, we still need to add a non-zero sky_level
            full_image += sky_level * pixel_scale**2

    return full_image, full_psf_image, full_weight_image, full_badpix_image


def BuildScatteredImage(config, logger=None,
                        make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image containing multiple objects placed at arbitrary locations.

    @param config     A configuration dict.
    @param logger     If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image
    @param make_weight_image   Whether to make weight_image
    @param make_badpix_image   Whether to make badpix_image

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    """
    ignore = [ 'random_seed', 'draw_method', 'noise', 'wcs', 'nproc' , 'center' ]
    req = { 'nobjects' : int }
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int , 
            'stamp_size' : int , 'stamp_xsize' : int , 'stamp_ysize' : int ,
            'pixel_scale' : float , 'nproc' : int , 'sky_level' : float }
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

    stamp_size = params.get('stamp_size',0)
    stamp_xsize = params.get('stamp_xsize',stamp_size)
    stamp_ysize = params.get('stamp_ysize',stamp_size)

    sky_level = params.get('sky_level',None)

    # If image_xsize and image_ysize were set in config, this overrides the read-in params.
    if 'image_xsize' in config and 'image_ysize' in config:
        full_xsize = config['image_xsize']
        full_ysize = config['image_ysize']

    pixel_scale = params.get('pixel_scale',1.0)
    config['pixel_scale'] = pixel_scale
    if 'pix' not in config:
        config['pix'] = { 'type' : 'Pixel' , 'xw' : pixel_scale }

    if 'center' not in config['image']:
        config['image']['center'] = { 
            'type' : 'XY' ,
            'x' : { 'type' : 'Random' , 'min' : 1 , 'max' : full_xsize },
            'y' : { 'type' : 'Random' , 'min' : 1 , 'max' : full_ysize }
        }

    rng = None

    nproc = params.get('nproc',1)

    full_image = galsim.ImageF(full_xsize,full_ysize)
    full_image.setZero()
    full_image.setScale(pixel_scale)

    # Also define the overall image center, since we need that to calculate the position 
    # of each stamp relative to the center.
    image_cen = full_image.bounds.center()
    config['image_cen'] = galsim.PositionD(image_cen.x,image_cen.y)

    if make_psf_image:
        full_psf_image = galsim.ImageF(full_xsize,full_ysize)
        full_psf_image.setZero()
        full_psf_image.setScale(pixel_scale)
    else:
        full_psf_image = None

    if make_weight_image:
        full_weight_image = galsim.ImageF(full_xsize,full_ysize)
        full_weight_image.setZero()
        full_weight_image.setScale(pixel_scale)
    else:
        full_weight_image = None

    if make_badpix_image:
        full_badpix_image = galsim.ImageS(full_xsize,full_ysize)
        full_badpix_image.setZero()
        full_badpix_image.setScale(pixel_scale)
    else:
        full_badpix_image = None

    stamp_images = galsim.config.BuildStamps(
            nstamps=nobjects, config=config, xsize=stamp_xsize, ysize=stamp_ysize,
            nproc=nproc, sky_level=sky_level, do_noise=False, logger=logger,
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
        if (not bounds.isDefined()):
            import warnings
            warnings.warn(
                "Object centered at (%d,%d) is entirely off the main image,\n"%(
                        image[k].bounds.center().x, image[k].bounds.center().y) +
                "whose bounds are (%d,%d,%d,%d)."%(
                        full_image.bounds.xmin, full_image.bounds.xmax,
                        full_image.bounds.ymin, full_image.bounds.ymax))
        full_image[bounds] += images[k][bounds]

        if make_psf_image:
            full_psf_image[bounds] += psf_images[k][bounds]
        if make_weight_image:
            full_weight_image[bounds] += weight_images[k][bounds]
        if make_badpix_image:
            full_badpix_image[bounds] |= badpix_images[k][bounds]

    if 'noise' in config['image']:
        # Apply the noise to the full image

        if not rng:
            # Get the next random number seed.
            if 'random_seed' in config['image']:
                seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
                #print 'For noise, seed = ',seed
                rng = galsim.BaseDeviate(seed)
            else:
                rng = galsim.BaseDeviate()

        draw_method = galsim.config.GetCurrentValue(config['image'],'draw_method')
        if draw_method == 'fft':
            galsim.config.AddNoiseFFT(
                full_image,full_weight_image,config['image']['noise'],rng,sky_level)
        elif draw_method == 'phot':
            galsim.config.AddNoisePhot(
                full_image,full_weight_image,config['image']['noise'],rng,sky_level)
        else:
            raise AttributeError("Unknown draw_method %s."%draw_method)

    elif sky_level:
        # If we aren't doing noise, we still need to add a non-zero sky_level
        full_image += sky_level * pixel_scale**2

    return full_image, full_psf_image, full_weight_image, full_badpix_image



