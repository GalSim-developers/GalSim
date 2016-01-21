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

def BuildDataCube(config, file_num, image_num, obj_num, ignore, logger):
    """
    Build a multi-image fits data cube as specified in config.
    
    @param config           A configuration dict.
    @param file_num         The current file_num.
    @param image_num        The current image_num.
    @param obj_num          The current obj_num.
    @param ignore           A list of parameters that are allowed to be in config['output']
                            that we can ignore here.  i.e. it won't be an error if these
                            parameters are present.
    @param logger           If given, a logger object to log progress.

    @returns a list of images, all of which are the same size (so amenable to making a data cube)
    """
    import time
    # Allow nimages to be automatic based on input catalog if image type is Single
    nimages = GetNImagesDataCube(config, file_num)

    # The above call sets up a default nimages if appropriate.  Now, check that there are no
    # invalid parameters in the config dict.
    req = { 'nimages' : int }
    galsim.config.CheckAllParams(config['output'], ignore=ignore, req=req)

    # All images need to be the same size for a data cube.
    # Enforce this by building the first image outside the below loop and setting
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
        images += galsim.config.BuildImages(nimages-1, config, logger=logger,
                                            image_num=image_num+1, obj_num=obj_num)

    return images

def GetNImagesDataCube(config, file_num):
    """
    Get the number of images for a DataCube file type.

    @param config           The configuration dict.
    @param file_num         The current file number.

    @returns the number of images
    """
    # Allow nimages to be automatic based on input catalog if image type is Single
    if ( 'nimages' not in config['output'] and 
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) ):
        nimages = galsim.config.ProcessInputNObjects(config)
        if nimages:
            config['output']['nimages'] = nimages
    if 'nimages' not in config['output']:
        raise AttributeError("Attribute output.nimages is required for output.type = MultiFits")
    return galsim.config.ParseValue(config['output'],'nimages',config,int)[0]


# Register this as a valid output type
from .output import RegisterOutputType
RegisterOutputType('DataCube', BuildDataCube, galsim.fits.writeCube, GetNImagesDataCube)
