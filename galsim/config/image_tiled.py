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

# This file adds image type Tiled, which builds a larger image by tiling nx x ny individual
# postage stamps.

def SetupTiled(config, image_num, obj_num, ignore, logger):
    """
    Build an Image consisting of a tiled array of postage stamps.

    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.
    @param ignore           A list of parameters that are allowed to be in config['image']
                            that we can ignore here.  i.e. it won't be an error if these
                            parameters are present.
    @param logger           If given, a logger object to log progress.

    @returns the final image
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: Building Tiled: image, obj = %d,%d',image_num,image_num,obj_num)

    extra_ignore = [ 'image_pos' ] # We create this below, so on subequent passes, we ignore it.
    req = { 'nx_tiles' : int , 'ny_tiles' : int }
    opt = { 'stamp_size' : int , 'stamp_xsize' : int , 'stamp_ysize' : int ,
            'border' : int , 'xborder' : int , 'yborder' : int , 'order' : str }
    params = galsim.config.GetAllParams(config['image'], config, req=req, opt=opt,
                                        ignore=ignore+extra_ignore)[0]

    nx_tiles = params['nx_tiles']
    ny_tiles = params['ny_tiles']
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
    config['do_noise_in_stamps'] = do_noise

    full_xsize = (stamp_xsize + xborder) * nx_tiles - xborder
    full_ysize = (stamp_ysize + yborder) * ny_tiles - yborder

    config['tile_xborder'] = xborder
    config['tile_yborder'] = yborder

    # If image_force_xsize and image_force_ysize were set in config, make sure it matches.
    if ( ('image_force_xsize' in config and full_xsize != config['image_force_xsize']) or
         ('image_force_ysize' in config and full_ysize != config['image_force_ysize']) ):
        raise ValueError(
            "Unable to reconcile required image xsize and ysize with provided "+
            "nx_tiles=%d, ny_tiles=%d, "%(nx_tiles,ny_tiles) +
            "xborder=%d, yborder=%d\n"%(xborder,yborder) +
            "Calculated full_size = (%d,%d) "%(full_xsize,full_ysize)+
            "!= required (%d,%d)."%(config['image_force_xsize'],config['image_force_ysize']))

    return full_xsize, full_ysize


def BuildTiled(config, image_num, obj_num, logger):
    """
    Build an Image consisting of a tiled array of postage stamps.

    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.
    @param logger           If given, a logger object to log progress.

    @returns the final image
    """
    full_xsize = config['image_xsize']
    full_ysize = config['image_ysize']
    wcs = config['wcs']

    full_image = galsim.ImageF(full_xsize, full_ysize)
    full_image.setOrigin(config['image_origin'])
    full_image.wcs = wcs
    full_image.setZero()

    if 'nproc' in config['image']:
        nproc = galsim.config.ParseValue(config['image'],'nproc',config,int)[0]
    else:
        nproc = 1

    do_noise = config['do_noise_in_stamps']
    xsize = config['tile_xsize']
    ysize = config['tile_ysize']
    xborder = config['tile_xborder']
    yborder = config['tile_yborder']

    nx_tiles = config['nx_tiles']
    ny_tiles = config['ny_tiles']
    nobjects = nx_tiles * ny_tiles

    # Make a list of ix,iy values according to the specified order:
    if 'order' in config['image']:
        order = galsim.config.ParseValue(config['image'],'order',config,str)[0].lower()
    else:
        order = 'row'
    if order.startswith('row'):
        ix_list = [ ix for iy in range(ny_tiles) for ix in range(nx_tiles) ]
        iy_list = [ iy for iy in range(ny_tiles) for ix in range(nx_tiles) ]
    elif order.startswith('col'):
        ix_list = [ ix for ix in range(nx_tiles) for iy in range(ny_tiles) ]
        iy_list = [ iy for ix in range(nx_tiles) for iy in range(ny_tiles) ]
    elif order.startswith('rand'):
        ix_list = [ ix for ix in range(nx_tiles) for iy in range(ny_tiles) ]
        iy_list = [ iy for ix in range(nx_tiles) for iy in range(ny_tiles) ]
        rng = config['rng']
        galsim.random.permute(rng, ix_list, iy_list)
    else:
        raise ValueError("Invalid order.  Must be row, column, or random")

    # Define a 'image_pos' field so the stamps can set their position appropriately in case
    # we need it for PowerSpectum or NFWHalo.
    x0 = (xsize-1)/2. + config['image_origin'].x
    y0 = (ysize-1)/2. + config['image_origin'].y
    dx = xsize + xborder
    dy = ysize + yborder
    config['image']['image_pos'] = {
        'type' : 'XY' ,
        'x' : { 'type' : 'List',
                'items' : [ x0 + ix*dx for ix in ix_list ]
              },
        'y' : { 'type' : 'List',
                'items' : [ y0 + iy*dy for iy in iy_list ]
              }
    }

    stamps, current_vars = galsim.config.BuildStamps(
            nobjects, config, nproc=nproc, logger=logger, obj_num=obj_num,
            xsize=xsize, ysize=ysize, do_noise=do_noise)

    for k in range(nobjects):
        # This is our signal that the object was skipped.
        if stamps[k] is None: continue
        if False:
            logger.debug('image %d: full bounds = %s',image_num,str(full_image.bounds))
            logger.debug('image %d: stamp %d bounds = %s',image_num,k,str(stamps[k].bounds))
        assert full_image.bounds.includes(stamps[k].bounds)
        b = stamps[k].bounds
        full_image[b] += stamps[k]

    current_var = 0
    if not do_noise:
        if 'noise' in config['image']:
            # First bring the image so far up to a flat noise variance
            current_var = galsim.config.FlattenNoiseVariance(
                    config, full_image, stamps, current_vars, logger)
    config['current_var'] = current_var
    return full_image


def AddNoiseTiled(image, config, image_num, obj_num, logger):
    """
    Add the final noise to a Tiled image

    @param image            The image onto which to add the noise.
    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.
    @param logger           If given, a logger object to log progress.
    """
    # If didn't do noise above in the stamps, then need to do it here.
    do_noise = config['do_noise_in_stamps']
    if not do_noise:
        # Apply the sky and noise to the full image
        galsim.config.AddSky(config,image)
        if 'noise' in config['image']:
            current_var = config['current_var']
            galsim.config.AddNoise(config,image,current_var,logger)

def GetNObjTiled(config, image_num):

    config['index_key'] = 'image_num'
    config['image_num'] = image_num

    if 'nx_tiles' not in config['image'] or 'ny_tiles' not in config['image']:
        raise AttributeError(
            "Attributes nx_tiles and ny_tiles are required for image.type = Tiled")
    nx = galsim.config.ParseValue(config['image'],'nx_tiles',config,int)[0]
    ny = galsim.config.ParseValue(config['image'],'ny_tiles',config,int)[0]
    return nx*ny

# Register this as a valid image type
from .image import RegisterImageType
RegisterImageType('Tiled', SetupTiled, BuildTiled, AddNoiseTiled, GetNObjTiled)
