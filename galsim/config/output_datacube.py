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


def BuildDataCube(file_name, config, nproc=1, logger=None, 
                  file_num=0, image_num=0, obj_num=0,
                  psf_file_name=None, weight_file_name=None, badpix_file_name=None):
    """
    Build a multi-image fits data cube as specified in config.
    
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

    config['index_key'] = 'file_num'
    config['file_num'] = file_num
    config['image_num'] = image_num
    config['start_obj_num'] = obj_num
    config['obj_num'] = obj_num
    if logger:
        logger.debug('file %d: BuildDataCube for %s: file, image, obj = %d,%d,%d',
                      config['file_num'],file_name,file_num,image_num,obj_num)

    if ( 'image' in config 
         and 'random_seed' in config['image'] 
         and not isinstance(config['image']['random_seed'],dict) ):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = { 'type' : 'Sequence', 'first' : first }

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

    if 'output' not in config or 'nimages' not in config['output']:
        raise AttributeError("Attribute output.nimages is required for output.type = DataCube")
    nimages = galsim.config.ParseValue(config['output'],'nimages',config,int)[0]

    # All images need to be the same size for a data cube.
    # Enforce this by buliding the first image outside the below loop and setting
    # config['image_force_xsize'] and config['image_force_ysize'] to be the size of the first 
    # image.
    t2 = time.time()
    config1 = galsim.config.CopyConfig(config)
    all_images = galsim.config.BuildImage(
            config=config1, logger=logger, image_num=image_num, obj_num=obj_num,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)
    obj_num += galsim.config.GetNObjForImage(config, image_num)
    t3 = time.time()
    if logger:
        # Note: numpy shape is y,x
        ys, xs = all_images[0].array.shape
        logger.info('Image %d: size = %d x %d, time = %f sec', image_num, xs, ys, t3-t2)

    # Note: numpy shape is y,x
    image_ysize, image_xsize = all_images[0].array.shape
    config['image_force_xsize'] = image_xsize
    config['image_force_ysize'] = image_ysize

    main_images = [ all_images[0] ]
    psf_images = [ all_images[1] ]
    weight_images = [ all_images[2] ]
    badpix_images = [ all_images[3] ]

    if nimages > 1:
        if nproc > nimages-1:
            # Only warn if nproc was specifically set, not if it is -1.
            if (logger and
                not ('nproc' in config['output'] and
                     galsim.config.ParseValue(config['output'],'nproc',config,int)[0] == -1)):
                logger.warn(
                    "Trying to use more processes than (nimages-1): output.nproc=%d, "%nproc +
                    "nimages=%d.  Reducing nproc to %d."%(nimages,nimages-1))
            nproc = nimages-1

        all_images = galsim.config.BuildImages(
            nimages-1, config=config, nproc=nproc, logger=logger,
            image_num=image_num+1, obj_num=obj_num,
            make_psf_image=make_psf_image,
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)

        main_images += all_images[0]
        psf_images += all_images[1]
        weight_images += all_images[2]
        badpix_images += all_images[3]

    if 'output' in config and 'retry_io' in config['output']:
        ntries = galsim.config.ParseValue(config['output'],'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    galsim.config.output._retry_io(galsim.fits.writeCube, (main_images, file_name), ntries, file_name, logger)
    if logger:
        logger.debug('file %d: Wrote image to fits data cube %r',
                     config['file_num'],file_name)

    if psf_file_name:
        galsim.config.output._retry_io(galsim.fits.writeCube, (psf_images, psf_file_name),
                  ntries, psf_file_name, logger)
        if logger:
            logger.debug('file %d: Wrote psf images to fits data cube %r',
                         config['file_num'],psf_file_name)

    if weight_file_name:
        galsim.config.output._retry_io(galsim.fits.writeCube, (weight_images, weight_file_name),
                  ntries, weight_file_name, logger)
        if logger:
            logger.debug('file %d: Wrote weight images to fits data cube %r',
                         config['file_num'],weight_file_name)

    if badpix_file_name:
        galsim.config.output._retry_io(galsim.fits.writeCube, (badpix_images, badpix_file_name),
                  ntries, badpix_file_name, logger)
        if logger:
            logger.debug('file %d: Wrote badpix images to fits data cube %r',
                         config['file_num'],badpix_file_name)

    t4 = time.time()
    return t4-t1

def GetNObjForDataCube(config, file_num, image_num):
    ignore = [ 'file_name', 'dir', 'nfiles', 'psf', 'weight', 'badpix', 'nproc',
               'skip', 'noclobber', 'retry_io' ]
    req = { 'nimages' : int }
    # Allow nimages to be automatic based on input catalog if image type is Single
    if ( 'nimages' not in config['output'] and 
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) ):
        nobjects = galsim.config.ProcessInputNObjects(config)
        if nobjects:
            config['output']['nimages'] = nobjects
    params = galsim.config.GetAllParams(config['output'],'output',config,ignore=ignore,req=req)[0]
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

