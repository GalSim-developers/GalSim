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
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: Building Scattered: image, obj = %d,%d',
                     image_num,image_num,obj_num)

    if 'nobjects' not in config['image']:
        nobjects = galsim.config.ProcessInputNObjects(config)
        if nobjects is None:
            raise AttributeError("Attribute nobjects is required for image.type = Scattered")
    else:
        nobjects = galsim.config.ParseValue(config['image'],'nobjects',config,int)[0]
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: nobj = %d',image_num,nobjects)
    config['nobjects'] = nobjects

    # These are allowed for Scattered, but we don't use them here.
    extra_ignore = [ 'image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                     'nobjects' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int }
    params = galsim.config.GetAllParams(config['image'], config, opt=opt,
                                        ignore=galsim.config.image_ignore+extra_ignore)[0]

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

    galsim.config.SetupConfigImageSize(config, full_xsize, full_ysize)
    wcs = config['wcs']
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: image_origin = %s',image_num,str(config['image_origin']))
        logger.debug('image %d: image_center = %s',image_num,str(config['image_center']))

    seed = galsim.config.SetupConfigRNG(config, nobjects)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: seed = %d',image_num,seed)
    rng = config['rng']

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
    galsim.config.SetupInputsForImage(config, logger)

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
            if make_badpix_image:
                full_badpix_image[bounds] |= badpix_images[k][bounds]
        else:
            if logger and logger.isEnabledFor(logging.WARN):
                logger.warn(
                    "Object centered at (%d,%d) is entirely off the main image,\n"%(
                        images[k].bounds.center().x, images[k].bounds.center().y) +
                    "whose bounds are (%d,%d,%d,%d)."%(
                        full_image.bounds.xmin, full_image.bounds.xmax,
                        full_image.bounds.ymin, full_image.bounds.ymax))

    # Mark that we are no longer doing a single galaxy by deleting image_pos from config top 
    # level, so it cannot be used for things like wcs.pixelArea(image_pos).  
    if 'image_pos' in config: del config['image_pos']

    galsim.config.AddSky(config, full_image)
    if 'noise' in config['image']:
        config['rng'] = rng
        # Bring noise up to a a constant level if necessary.
        current_var = galsim.config.FlattenNoiseVariance(
                config, full_image, images, current_vars, logger)
        # Apply the noise to the full image
        galsim.config.AddNoise(config,full_image,full_weight_image,current_var,logger)

    return full_image, full_psf_image, full_weight_image, full_badpix_image


def GetNObjScattered(config, image_num):

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


