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


def BuildMultiFits(file_name, config, nproc=1, logger=None,
                   file_num=0, image_num=0, obj_num=0,
                   psf_file_name=None, weight_file_name=None, badpix_file_name=None):
    """
    Build a multi-extension fits file as specified in config.
    
    @param file_name        The name of the output file.
    @param config           A configuration dict.
    @param nproc            How many processes to use. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]
    @param file_num         If given, the current file_num. [default: 0]
    @param image_num        If given, the current image_num. [default: 0]
    @param obj_num          If given, the current obj_num. [default: 0]
    @param psf_file_name    If given, write a psf image to this file. [default: None]
    @param weight_file_name If given, write a weight image to this file. [default: None]
    @param badpix_file_name If given, write a badpix image to this file. [default: None]

    @returns the time taken to build file.
    """
    import time
    t1 = time.time()

    galsim.config.SetupConfigFileNum(config,file_num,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: seed = %d',file_num,seed)

    nimages = galsim.config.GetNImagesForFile(config, file_num)

    if psf_file_name:
        make_psf_image = True
    else:
        make_psf_image = False

    if weight_file_name:
        make_weight_image = True
    else:
        make_weight_image = False

    if badpix_file_name:
        make_badpix_image = True
    else:
        make_badpix_image = False

    if nproc > nimages:
        # Only warn if nproc was specifically set, not if it is -1.
        if (logger and
            not ('nproc' in config['output'] and 
                 galsim.config.ParseValue(config['output'],'nproc',config,int)[0] == -1)):
            logger.warn(
                "Trying to use more processes than images: output.nproc=%d, "%nproc +
                "nimages=%d.  Reducing nproc to %d."%(nimages,nimages))
        nproc = nimages

    all_images = galsim.config.BuildImages(
        nimages, config=config, nproc=nproc, logger=logger,
        image_num=image_num, obj_num=obj_num,
        make_psf_image=make_psf_image, 
        make_weight_image=make_weight_image,
        make_badpix_image=make_badpix_image)

    main_images = all_images[0]
    psf_images = all_images[1]
    weight_images = all_images[2]
    badpix_images = all_images[3]

    if 'output' in config and 'retry_io' in config['output']:
        ntries = galsim.config.ParseValue(config['output'],'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    galsim.config.output.RetryIO(galsim.fits.writeMulti, (main_images, file_name), ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Wrote images to multi-extension fits file %r',
                     config['file_num'],file_name)

    if psf_file_name:
        galsim.config.output.RetryIO(galsim.fits.writeMulti, (psf_images, psf_file_name),
                  ntries, psf_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote psf images to multi-extension fits file %r',
                         config['file_num'],psf_file_name)

    if weight_file_name:
        galsim.config.output.RetryIO(galsim.fits.writeMulti, (weight_images, weight_file_name),
                  ntries, weight_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote weight images to multi-extension fits file %r',
                         config['file_num'],weight_file_name)

    if badpix_file_name:
        galsim.config.output.RetryIO(galsim.fits.writeMulti, (all_images, badpix_file_name),
                  ntries, badpix_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote badpix images to multi-extension fits file %r',
                         config['file_num'],badpix_file_name)


    t2 = time.time()
    return t2-t1


def GetNImagesMultiFits(config, file_num):
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


