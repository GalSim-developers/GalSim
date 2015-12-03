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

# This file handles building the output files according to the specifications in config['output'].
# This file includes the basic functionality, but it calls out to helper functions for 
# the different types of output files.  It includes the implementation of the default output
# type, 'Fits'.  See output_multifits.py for 'MultiFits' and output_datacube.py for 'DataCube'.


def BuildFile(config, file_num=0, image_num=0, obj_num=0, nproc=1, logger=None):
    """
    Build a regular fits file as specified in config.
    
    @param config           A configuration dict.
    @param file_num         If given, the current file_num. [default: 0]
    @param image_num        If given, the current image_num. [default: 0]
    @param obj_num          If given, the current obj_num. [default: 0]
    @param nproc            How many processes to use for building the images. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns a tuple of the file name and the time taken to build file: (file_name, t)
    Note: t==0 indicates that this file was skipped.
    """
    import time
    t1 = time.time()

    if 'output' not in config:
        config['output'] = {}
    output = config['output']

    if 'type' in config['output']:
        output_type = config['output']['type']
    else:
        output_type = 'Fits'

    if output_type not in valid_output_types:
        raise AttributeError("Invalid output.type=%s."%output_type)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Build File with type=%s for file, image, obj = %d,%d,%d',
                      file_num,output_type,file_num,image_num,obj_num)

    SetupConfigFileNum(config,file_num,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: seed = %d',file_num,seed)

    # Get the file name
    default_ext = valid_output_types[output_type][3]
    file_name = GetFilename(output, config, default_ext)

    # Check if we ought to skip this file
    if 'skip' in output and galsim.config.ParseValue(output, 'skip', config, bool)[0]:
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn('Skipping file %d = %s because output.skip = True',file_num,file_name)
        t2 = time.time()
        return file_name, 0
    if ('noclobber' in output
        and galsim.config.ParseValue(output, 'noclobber', config, bool)[0]
        and os.path.isfile(file_name)):
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn('Skipping file %d = %s because output.noclobber = True' +
                        ' and file exists',file_num,file_name)
        t2 = time.time()
        return file_name, 0

    if logger: 
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: file_name = %s',file_num,file_name)
        elif logger.isEnabledFor(logging.WARN):
            logger.warn('Start file %d = %s', file_num, file_name)

    build_func = valid_output_types[output_type][0]
    data = build_func(config, file_num, image_num, obj_num, nproc, logger)

    can_add_hdus = valid_output_types[output_type][2]
    if can_add_hdus:
        data = data + galsim.config.BuildExtraOutputHDUs(config,logger,len(data))

    if 'retry_io' in output:
        ntries = galsim.config.ParseValue(output,'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    write_func = valid_output_types[output_type][1]
    args = (data, file_name)
    RetryIO(write_func, args, ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Wrote %s to file %r',file_num,output_type,file_name)

    galsim.config.WriteExtraOutputs(config,logger)
    t2 = time.time()

    return file_name, t2-t1

def GetNObjForFile(config, file_num, image_num):
    """
    Get the number of objects that will be made for the file number file_num, which starts
    at image number image_num, based on the information in the config dict.

    @param config           The configuration dict.
    @param file_num         The current file number.
    @param image_num        The current image number.

    @returns the number of objects
    """
    if 'output' in config and 'type' in config['output']:
        output_type = config['output']['type']
    else:
        output_type = 'Fits'

    # Check that the type is valid
    if output_type not in valid_output_types:
        raise AttributeError("Invalid output.type=%s."%output_type)

    nobj_func = valid_output_types[output_type][4]
    return nobj_func(config, file_num, image_num)


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


def SetDefaultExt(config, ext):
    """Set a default ext if appropriate"""
    if ext is not None:
        if ( isinstance(config,dict) and 'type' in config and 
            config['type'] == 'NumberedFile' and 'ext' not in config ):
            config['ext'] = ext


def GetFilename(config, base, default_ext=None):
    """Get the file_name for the current file being worked on.
    """
    if 'file_name' in config:
        SetDefaultExt(config['file_name'],default_ext)
        file_name = galsim.config.ParseValue(config, 'file_name', base, str)[0]
    elif 'root' in config and default_ext is not None:
        # If a file_name isn't specified, we use the name of the config file + '.fits'
        file_name = config['root'] + default_ext
    else:
        raise AttributeError("No file_name specified and unable to generate it automatically.")

    # Prepend a dir to the beginning of the filename if requested.
    if 'dir' in config:
        dir = galsim.config.ParseValue(config, 'dir', base, str)[0]
        if dir and not os.path.isdir(dir): os.makedirs(dir)
        file_name = os.path.join(dir,file_name)

    return file_name


# A helper function to retry io commands
def RetryIO(func, args, ntries, file_name, logger):
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


output_ignore = [ 'file_name', 'dir', 'nfiles', 'nproc', 'skip', 'noclobber', 'retry_io' ]


def BuildFits(config, file_num, image_num, obj_num, nproc, logger):
    """
    Build a regular fits file as specified in config.
    
    @param config           A configuration dict.
    @param file_num         The current file_num.
    @param image_num        The current image_num.
    @param obj_num          The current obj_num.
    @param nproc            How many processes to use. (ignored)
    @param logger           If given, a logger object to log progress.

    @returns the image in a list with one item: [ image ]
    """
    image = galsim.config.BuildImage(config, logger=logger, image_num=image_num, obj_num=obj_num)
    return [ image ]

def GetNObjFits(config, file_num, image_num):
    ignore = output_ignore + galsim.config.valid_extra_outputs.keys()
    galsim.config.CheckAllParams(config['output'], ignore=ignore)
    try : 
        nobj = [ galsim.config.GetNObjForImage(config, image_num) ]
    except ValueError : # (This may be raised if something needs the input stuff)
        galsim.config.ProcessInput(config, file_num=file_num)
        nobj = [ galsim.config.GetNObjForImage(config, image_num) ]
    return nobj
    
# The values are tuples with:
# - the build function to call
# - the function to use for writing the data
# - whether extra hdus can be added to the end of the data for the extra output items
# - the default file extension
# - a function that merely counts the number of objects that will be built by the function

valid_output_types = {
    'Fits' : (BuildFits, galsim.fits.writeMulti, True, '.fits', GetNObjFits),
}

