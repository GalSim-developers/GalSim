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
    """
    config['index_key'] = 'image_num'
    config['image_num'] = image_num
    config['obj_num'] = obj_num

    if logger:
        logger.debug('image %d: BuildTiledImage: image, obj = %d,%d',image_num,image_num,obj_num)

    if 'random_seed' in config['image'] and not isinstance(config['image']['random_seed'],dict):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = { 'type' : 'Sequence', 'first' : first }

    ignore = [ 'random_seed', 'draw_method', 'noise', 'pixel_scale', 'wcs', 'nproc',
               'sky_level', 'sky_level_pixel',
               'retry_failures', 'image_pos', 'n_photons', 'wmult', 'offset', 'gsparams' ]
    req = { 'nx_tiles' : int , 'ny_tiles' : int }
    opt = { 'stamp_size' : int , 'stamp_xsize' : int , 'stamp_ysize' : int ,
            'border' : int , 'xborder' : int , 'yborder' : int ,
            'nproc' : int , 'index_convention' : str, 'order' : str }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, req=req, opt=opt, ignore=ignore)[0]

    nx_tiles = params['nx_tiles']
    ny_tiles = params['ny_tiles']
    nobjects = nx_tiles * ny_tiles
    config['nx_tiles'] = nx_tiles
    config['ny_tiles'] = ny_tiles
    if logger:
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
    config['image_xsize'] = full_xsize
    config['image_ysize'] = full_ysize
    if logger:
        logger.debug('image %d: image_size = %d, %d',image_num,full_xsize,full_ysize)

    convention = params.get('index_convention','1')
    galsim.config.image._set_image_origin(config,convention)
    if logger:
        logger.debug('image %d: image_origin = %s',image_num,str(config['image_origin']))
        logger.debug('image %d: image_center = %s',image_num,str(config['image_center']))

    wcs = galsim.config.BuildWCS(config, logger)

    # Set the rng to use for image stuff.
    if 'random_seed' in config['image']:
        # Technically obj_num+nobjects will be the index of the random seed used for the next 
        # image's first object (if there is a next image).  But I don't think that will have 
        # any adverse effects.
        config['obj_num'] = obj_num + nobjects
        config['index_key'] = 'obj_num'
        seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['index_key'] = 'image_num'
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

    full_image = galsim.ImageF(full_xsize, full_ysize)
    full_image.setOrigin(config['image_origin'])
    full_image.wcs = wcs
    full_image.setZero()

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

    if make_badpix_image:
        full_badpix_image = galsim.ImageS(full_image.bounds, wcs=wcs)
        full_badpix_image.setZero()
    else:
        full_badpix_image = None

    # Sometimes an input field needs to do something special at the start of an image.
    if 'input' in config:
        for key in [ k for k in galsim.config.valid_input_types.keys() if k in config['input'] ]:
            if galsim.config.valid_input_types[key][4]:
                assert key in config
                fields = config['input'][key]
                if not isinstance(fields, list):
                    fields = [ fields ]
                input_objs = config[key]

                for i in range(len(fields)):
                    field = fields[i]
                    input_obj = input_objs[i]
                    func = eval(galsim.config.valid_input_types[key][4])
                    func(input_obj, field, config)

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
        if make_badpix_image:
            full_badpix_image[b] |= badpix_images[k]
        if current_vars[k] > max_current_var: max_current_var = current_vars[k]

    # Mark that we are no longer doing a single galaxy by deleting image_pos from config top 
    # level, so it cannot be used for things like wcs.pixelArea(image_pos).  
    if 'image_pos' in config: del config['image_pos']

    # If didn't do noise above in the stamps, then need to do it here.
    if not do_noise:
        if 'noise' in config['image']:
            # If we didn't apply noise in each stamp, then we need to apply it now.
            draw_method = galsim.config.GetCurrentValue(config['image'],'draw_method')

            if max_current_var > 0:
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

            config['rng'] = rng
            galsim.config.AddNoise(
                config,draw_method,full_image,full_weight_image,max_current_var,logger)

        else:
            # If we aren't doing noise, we still may need to add a non-zero sky_level.
            # The same noise function does this with the 'skip' draw method.
            galsim.config.AddNoise(
                config,'skip',full_image,full_weight_image,max_current_var,logger)

    return full_image, full_psf_image, full_weight_image, full_badpix_image


def GetNObjForTiledImage(config, image_num):
    
    config['index_key'] = 'image_num'
    config['image_num'] = image_num

    if 'nx_tiles' not in config['image'] or 'ny_tiles' not in config['image']:
        raise AttributeError(
            "Attributes nx_tiles and ny_tiles are required for image.type = Tiled")
    nx = galsim.config.ParseValue(config['image'],'nx_tiles',config,int)[0]
    ny = galsim.config.ParseValue(config['image'],'ny_tiles',config,int)[0]
    return nx*ny

