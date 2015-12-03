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

import os
import galsim
import logging

def BuildDataCube(config, file_num, image_num, obj_num, nproc, logger):
    """
    Build a multi-image fits data cube as specified in config.
    
    @param config           A configuration dict.
    @param file_num         The current file_num.
    @param image_num        The current image_num.
    @param obj_num          The current obj_num.
    @param nproc            How many processes to use.
    @param logger           If given, a logger object to log progress.

    @returns a list of images, all of which are the same size (so amenable to making a data cube)
    """
    import time
    # Allow nimages to be automatic based on input catalog if image type is Single
    if ( 'nimages' not in config['output'] and 
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) ):
        nobjects = galsim.config.ProcessInputNObjects(config)
        if nobjects:
            config['output']['nimages'] = nobjects
    if 'nimages' not in config['output']:
        raise AttributeError("Attribute output.nimages is required for output.type = DataCube")
    nimages = galsim.config.ParseValue(config['output'],'nimages',config,int)[0]

    # All images need to be the same size for a data cube.
    # Enforce this by buliding the first image outside the below loop and setting
    # config['image_force_xsize'] and config['image_force_ysize'] to be the size of the first 
    # image.
    t1 = time.time()
    config1 = galsim.config.CopyConfig(config)
    image0 = galsim.config.BuildImage(config1, logger=logger, image_num=image_num, obj_num=obj_num)
    t2 = time.time()
    if logger and logger.isEnabledFor(logging.INFO):
        # Note: numpy shape is y,x
        ys, xs = image0.array.shape
        logger.info('Image %d: size = %d x %d, time = %f sec', image_num, xs, ys, t2-t1)

    # Note: numpy shape is y,x
    image_ysize, image_xsize = image0.array.shape
    config['image_force_xsize'] = image_xsize
    config['image_force_ysize'] = image_ysize

    images = [ image0 ]

    if nimages > 1:
        obj_num += galsim.config.GetNObjForImage(config, image_num)
        images += galsim.config.BuildImages(nimages-1, config, nproc=nproc, logger=logger,
                                            image_num=image_num+1, obj_num=obj_num)

    return images

def GetNObjDataCube(config, file_num, image_num):
    req = { 'nimages' : int }
    # Allow nimages to be automatic based on input catalog if image type is Single
    if ( 'nimages' not in config['output'] and 
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) ):
        nobjects = galsim.config.ProcessInputNObjects(config)
        if nobjects:
            config['output']['nimages'] = nobjects
    ignore = galsim.config.output_ignore + galsim.config.valid_extra_outputs.keys()
    params = galsim.config.GetAllParams(config['output'], config, ignore=ignore, req=req)[0]
    config['index_key'] = 'file_num'
    config['file_num'] = file_num
    config['image_num'] = image_num
    nimages = params['nimages']
    try :
        nobj = [ galsim.config.GetNObjForImage(config, image_num+j) for j in range(nimages) ]
    except ValueError : # (This may be raised if something needs the input stuff)
        galsim.config.ProcessInput(config, file_num=file_num)
        nobj = [ galsim.config.GetNObjForImage(config, image_num+j) for j in range(nimages) ]
    return nobj
 
# Register this as a valid output type
from .output import valid_output_types
valid_output_types['DataCube'] = (
    BuildDataCube, galsim.fits.writeCube, False, '.fits', GetNObjDataCube
)

