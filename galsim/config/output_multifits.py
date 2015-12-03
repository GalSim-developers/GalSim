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

def BuildMultiFits(config, file_num, image_num, obj_num, nproc, ignore, logger):
    """
    Build a multi-extension fits file as specified in config.
    
    @param config           A configuration dict.
    @param file_num         The current file_num.
    @param image_num        The current image_num.
    @param obj_num          The current obj_num.
    @param nproc            How many processes to use.
    @param ignore           A list of parameters that are allowed to be in config['output']
                            that we can ignore here.  i.e. it won't be an error if these
                            parameters are present.
    @param logger           If given, a logger object to log progress.

    @returns a list of images
    """
    nimages = GetNImagesMultiFits(config, file_num)

    # The above call sets up a default nimages if appropriate.  Now, check that there are no
    # invalid parameters in the config dict.
    req = { 'nimages' : int }
    galsim.config.CheckAllParams(config['output'], ignore=ignore, req=req)

    return galsim.config.BuildImages(nimages, config, nproc=nproc, logger=logger,
                                     image_num=image_num, obj_num=obj_num)


def GetNImagesMultiFits(config, file_num):
    """
    Get the number of images for a MultiFits file type.

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
from .output import valid_output_types
valid_output_types['MultiFits'] = (
    BuildMultiFits, galsim.fits.writeMulti, False, '.fits', GetNImagesMultiFits
)

