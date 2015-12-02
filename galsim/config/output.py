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

valid_output_types = { 
    # The values are tuples with:
    # - the build function to call
    # - a function that merely counts the number of objects that will be built by the function
    # - whether the Builder takes nproc.
    # See the des module for examples of how to extend this from a module.
    'Fits' : ('galsim.config.BuildFits', 'galsim.config.GetNObjForFits', False),
    'MultiFits' : ('galsim.config.BuildMultiFits', 'galsim.config.GetNObjForMultiFits', True),
    'DataCube' : ('galsim.config.BuildDataCube', 'galsim.config.GetNObjForDataCube', True),
}

# A helper function to retry io commands
def _retry_io(func, args, ntries, file_name, logger):
    for itry in range(ntries):
        try: 
            ret = func(*args)
        except IOError as e:
            if itry == ntries-1:
                # Then this was the last try.  Just re-raise the exception.
                raise
            else:
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('File %s: Caught IOError: %s',file_name,str(e))
                    logger.warn('This is try %d/%d, so sleep for %d sec and try again.',
                                itry+1,ntries,itry+1)
                import time
                time.sleep(itry+1)
                continue
        else:
            break
    return ret

def SetupConfigFileNum(config, file_num, image_num, obj_num):
    """Do the basic setup of the config dict at the file processing level.

    Includes:
    - Set config['file_num'] = file_num
    - Set config['image_num'] = image_num
    - Set config['obj_num'] = obj_num
    - Set config['index_key'] = 'file_num'
    - Set config['start_obj_num'] = obj_num

    @param config           A configuration dict.
    @param file_num         The current file_num. (If file_num=None, then don't set file_num or
                            start_obj_num items in the config dict.)
    @param image_num        The current image_num.
    @param obj_num          The current obj_num.
    """
    if file_num is None:
        if 'file_num' not in config: config['file_num'] = 0
        if 'start_obj_num' not in config: config['start_obj_num'] = obj_num
    else:
        config['file_num'] = file_num
        config['start_obj_num'] = obj_num
    config['image_num'] = image_num
    config['obj_num'] = obj_num
    config['index_key'] = 'file_num'
    if 'output' not in config: config['output'] = {}

output_ignore = [ 'file_name', 'dir', 'nfiles', 'nproc', 'skip', 'noclobber', 'retry_io' ]

def BuildFits(file_name, config, logger=None, 
              file_num=0, image_num=0, obj_num=0):
    """
    Build a regular fits file as specified in config.
    
    @param file_name        The name of the output file.
    @param config           A configuration dict.
    @param logger           If given, a logger object to log progress. [default: None]
    @param file_num         If given, the current file_num. [default: 0]
    @param image_num        If given, the current image_num. [default: 0]
    @param obj_num          If given, the current obj_num. [default: 0]

    @returns the time taken to build file.
    """
    import time
    t1 = time.time()

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildFits for %s: file, image, obj = %d,%d,%d',
                      file_num,file_name,file_num,image_num,obj_num)

    SetupConfigFileNum(config,file_num,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: seed = %d',file_num,seed)

    image = galsim.config.BuildImage(config, logger=logger, image_num=image_num, obj_num=obj_num)

    hdulist = [ image ] + galsim.config.BuildExtraOutputHDUs(config,logger)

    if 'output' in config and 'retry_io' in config['output']:
        ntries = galsim.config.ParseValue(config['output'],'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    _retry_io(galsim.fits.writeMulti, (hdulist, file_name), ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        if len(hdulist) == 1:
            logger.debug('file %d: Wrote image to fits file %r',file_num,file_name)
        else:
            logger.debug('file %d: Wrote image (with extra hdus) to multi-extension fits file %r',
                         file_num,file_name)

    galsim.config.WriteExtraOutputs(config,logger)

    t2 = time.time()
    return t2-t1


def BuildMultiFits(file_name, config, nproc=1, logger=None,
                   file_num=0, image_num=0, obj_num=0):
    """
    Build a multi-extension fits file as specified in config.
    
    @param file_name        The name of the output file.
    @param config           A configuration dict.
    @param nproc            How many processes to use. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]
    @param file_num         If given, the current file_num. [default: 0]
    @param image_num        If given, the current image_num. [default: 0]
    @param obj_num          If given, the current obj_num. [default: 0]

    @returns the time taken to build file.
    """
    import time
    t1 = time.time()

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildMultiFits for %s: file, image, obj = %d,%d,%d',
                      config['file_num'],file_name,file_num,image_num,obj_num)
    SetupConfigFileNum(config,file_num,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: seed = %d',file_num,seed)

    # Allow nimages to be automatic based on input catalog if image type is Single
    if ( 'nimages' not in config['output'] and 
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) ):
        nobjects = galsim.config.ProcessInputNObjects(config)
        if nobjects:
            config['output']['nimages'] = nobjects
    if 'nimages' not in config['output']:
        raise AttributeError("Attribute output.nimages is required for output.type = MultiFits")
    nimages = galsim.config.ParseValue(config['output'],'nimages',config,int)[0]

    images = galsim.config.BuildImages(nimages, config, nproc=nproc, logger=logger,
                                       image_num=image_num, obj_num=obj_num)

    if 'retry_io' in config['output']:
        ntries = galsim.config.ParseValue(config['output'],'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    _retry_io(galsim.fits.writeMulti, (images, file_name), ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Wrote images to multi-extension fits file %r',
                     config['file_num'],file_name)

    galsim.config.WriteExtraOutputs(config,logger)

    t2 = time.time()
    return t2-t1


def BuildDataCube(file_name, config, nproc=1, logger=None, 
                  file_num=0, image_num=0, obj_num=0):
    """
    Build a multi-image fits data cube as specified in config.
    
    @param file_name        The name of the output file.
    @param config           A configuration dict.
    @param nproc            How many processes to use. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]
    @param file_num         If given, the current file_num. [default: 0]
    @param image_num        If given, the current image_num. [default: 0]
    @param obj_num          If given, the current obj_num. [default: 0]

    @returns the time taken to build file.
    """
    import time
    t1 = time.time()

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildDataCube for %s: file, image, obj = %d,%d,%d',
                      file_num,file_name,file_num,image_num,obj_num)
    SetupConfigFileNum(config,file_num,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: seed = %d',file_num,seed)

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
    t2 = time.time()
    config1 = galsim.config.CopyConfig(config)
    image0 = galsim.config.BuildImage(config1, logger=logger, image_num=image_num, obj_num=obj_num)
    t3 = time.time()
    if logger and logger.isEnabledFor(logging.INFO):
        # Note: numpy shape is y,x
        ys, xs = image0.array.shape
        logger.info('Image %d: size = %d x %d, time = %f sec', image_num, xs, ys, t3-t2)

    # Note: numpy shape is y,x
    image_ysize, image_xsize = image0.array.shape
    config['image_force_xsize'] = image_xsize
    config['image_force_ysize'] = image_ysize

    images = [ image0 ]

    if nimages > 1:
        obj_num += galsim.config.GetNObjForImage(config, image_num)
        images += galsim.config.BuildImages(nimages-1, config, nproc=nproc, logger=logger,
                                            image_num=image_num+1, obj_num=obj_num)

    if 'retry_io' in config['output']:
        ntries = galsim.config.ParseValue(config['output'],'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    _retry_io(galsim.fits.writeCube, (images, file_name), ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Wrote image to fits data cube %r',
                     config['file_num'],file_name)

    galsim.config.WriteExtraOutputs(config,logger)

    t4 = time.time()
    return t4-t1

def GetNObjForFits(config, file_num, image_num):
    ignore = output_ignore + galsim.config.valid_extra_outputs.keys()
    galsim.config.CheckAllParams(config['output'], 'output', ignore=ignore)
    try : 
        nobj = [ galsim.config.GetNObjForImage(config, image_num) ]
    except ValueError : # (This may be raised if something needs the input stuff)
        galsim.config.ProcessInput(config, file_num=file_num)
        nobj = [ galsim.config.GetNObjForImage(config, image_num) ]
    return nobj
    
def GetNObjForMultiFits(config, file_num, image_num):
    req = { 'nimages' : int }
    # Allow nimages to be automatic based on input catalog if image type is Single
    if ( 'nimages' not in config['output'] and 
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) ):
        nobjects = galsim.config.ProcessInputNObjects(config)
        if nobjects:
            config['output']['nimages'] = nobjects
    ignore = output_ignore + galsim.config.valid_extra_outputs.keys()
    params = galsim.config.GetAllParams(config['output'],'output',config, ignore=ignore,req=req)[0]
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

def GetNObjForDataCube(config, file_num, image_num):
    req = { 'nimages' : int }
    # Allow nimages to be automatic based on input catalog if image type is Single
    if ( 'nimages' not in config['output'] and 
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) ):
        nobjects = galsim.config.ProcessInputNObjects(config)
        if nobjects:
            config['output']['nimages'] = nobjects
    ignore = output_ignore + galsim.config.valid_extra_outputs.keys()
    params = galsim.config.GetAllParams(config['output'],'output',config, ignore=ignore,req=req)[0]
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
 
def SetDefaultExt(config, ext):
    """
    Some items have a default extension for a NumberedFile type.
    """
    if ( isinstance(config,dict) and 'type' in config and 
         config['type'] == 'NumberedFile' and 'ext' not in config ):
        config['ext'] = ext

