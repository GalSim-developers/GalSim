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

# This file adds image type Scattered, which places individual stamps at arbitrary
# locations on a larger image.

def SetupScattered(config, image_num, obj_num, ignore, logger):
    """
    Build an Image containing multiple objects placed at arbitrary locations.

    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.
    @param ignore           A list of parameters that are allowed to be in config['image']
                            that we can ignore here.  i.e. it won't be an error if these
                            parameters are present.
    @param logger           If given, a logger object to log progress.

    @returns the final image
    """
    if logger:
        logger.debug('image %d: Building Scattered: image, obj = %d,%d',
                     image_num,image_num,obj_num)

    nobjects = GetNObjScattered(config, image_num)
    if logger:
        logger.debug('image %d: nobj = %d',image_num,nobjects)
    config['nobjects'] = nobjects

    # These are allowed for Scattered, but we don't use them here.
    extra_ignore = [ 'image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                     'nobjects' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int }
    params = galsim.config.GetAllParams(config['image'], config, opt=opt,
                                        ignore=ignore+extra_ignore)[0]

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

    return full_xsize, full_ysize


def BuildScattered(config, image_num, obj_num, logger):
    """
    Build an Image containing multiple objects placed at arbitrary locations.

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

    nobjects = config['nobjects']

    stamps, current_vars = galsim.config.BuildStamps(
            nobjects, config, logger=logger, obj_num=obj_num, do_noise=False)

    config['index_key'] = 'image_num'

    for k in range(nobjects):
        # This is our signal that the object was skipped.
        if stamps[k] is None: continue
        bounds = stamps[k].bounds & full_image.bounds
        if logger:
            logger.debug('image %d: full bounds = %s',image_num,str(full_image.bounds))
            logger.debug('image %d: stamp %d bounds = %s',image_num,k,str(stamps[k].bounds))
            logger.debug('image %d: Overlap = %s',image_num,str(bounds))
        if bounds.isDefined():
            full_image[bounds] += stamps[k][bounds]
        else:
            if logger:
                logger.warn(
                    "Object centered at (%d,%d) is entirely off the main image,\n"%(
                        stamps[k].bounds.center().x, stamps[k].bounds.center().y) +
                    "whose bounds are (%d,%d,%d,%d)."%(
                        full_image.bounds.xmin, full_image.bounds.xmax,
                        full_image.bounds.ymin, full_image.bounds.ymax))

    current_var = 0
    if 'noise' in config['image']:
        # Bring the image so far up to a flat noise variance
        current_var = galsim.config.FlattenNoiseVariance(
                config, full_image, stamps, current_vars, logger)
    config['current_var'] = current_var

    return full_image


def AddNoiseScattered(image, config, image_num, obj_num, logger):
    """
    Add the final noise to a Scattered image

    @param image            The image onto which to add the noise.
    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.
    @param logger           If given, a logger object to log progress.
    """
    galsim.config.AddSky(config,image)
    if 'noise' in config['image']:
        current_var = config['current_var']
        galsim.config.AddNoise(config,image,current_var,logger)


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

# Register this as a valid image type
from .image import RegisterImageType
RegisterImageType('Scattered', SetupScattered, BuildScattered, AddNoiseScattered, GetNObjScattered)


