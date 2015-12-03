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
import inspect

# This file handles the processing of extra output items in addition to the primary output file
# in config['output']. The ones that are defined natively in GalSim are psf, weight, badpix,
# and truth.  See extra_*.py for the specific functions for each of these.

def SetupExtraOutput(config, file_num=0, logger=None):
    """
    Set up the extra output items as necessary, including building Managers for them
    so their objects can be updated safely in multi-processing mode.

    For example, the truth item needs to have the OutputCatalog set up and managed
    so each process can add rows to it without getting race conditions or clobbering
    each others' rows.

    Each item that gets built will be placed in config['extra_objs'][key] where key is
    the key in galsim.config.valid_extra_outputs.  The objects will actually be proxy objects
    using a multiprocessing.Manager so that multiple processes can all communicate with it
    correctly.  The objects here are what will eventually be written out. 

    This also sets up a scratch dict that is similarly safe to use from multiple processes
    called config['extra_scratch'][key].

    @param config       The configuration dict.
    @param file_num     The file number being worked on currently. [default: 0]
    @param logger       If given, a logger object to log progress. [default: None]
    """
    if 'output' in config:
        output = config['output']

        # We'll iterate through this list of keys a few times
        all_keys = [ k for k in valid_extra_outputs.keys()
                     if (k in output and valid_extra_outputs[k][0] is not None) ]

        # We don't need the manager stuff if we (a) are already in a multiprocessing Process, or
        # (b) config.image.nproc == 1.
        use_manager = (
                'current_nproc' not in config and
                'image' in config and 'nproc' in config['image'] and
                galsim.config.ParseValue(config['image'], 'nproc', config, int)[0] != 1 )
 
        if use_manager and 'output_manager' not in config:
            from multiprocessing.managers import BaseManager, DictProxy
            class OutputManager(BaseManager): pass
 
            # Register each input field with the OutputManager class
            for key in all_keys:
                fields = output[key]
                # Register this object with the manager
                init_func = valid_extra_outputs[key][0]
                OutputManager.register(key, init_func)
            # Also register dict to use for scratch space
            OutputManager.register('dict', dict, DictProxy)
            # Start up the output_manager
            config['output_manager'] = OutputManager()
            config['output_manager'].start()

        if 'extra_objs' not in config:
            config['extra_objs'] = {}
        if 'extra_scratch' not in config:
            config['extra_scratch'] = {}

        for key in all_keys:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Setup output item %s',file_num,key)
            field = config['output'][key]
            kwargs_func = valid_extra_outputs[key][1]
            if kwargs_func is not None:
                kwargs = kwargs_func(field, config, logger)
            else:
                # use default constructor
                kwargs = {}
 
            if use_manager:
                output_obj = getattr(config['output_manager'],key)(**kwargs)
                scratch = config['output_manager'].dict()
            else: 
                init_func = valid_extra_outputs[key][0]
                output_obj = init_func(**kwargs)
                scratch = dict()
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Setup output %s object',file_num,key)
            config['extra_objs'][key] = output_obj
            config['extra_scratch'][key] = scratch


def SetupExtraOutputsForImage(config, logger=None):
    """Perform any necessary setup for the extra output items at the start of a new image.

    @param config       The configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]
    """
    if 'output' in config:
        for key in [ k for k in valid_extra_outputs.keys() if k in config['output'] ]:
            # Always clear out anything in the scratch space
            extra_scratch = config['extra_scratch'][key]
            extra_scratch.clear()
            setup_func = valid_extra_outputs[key][2]
            if setup_func is not None:
                extra_obj = config['extra_objs'][key]
                field = config['output'][key]
                setup_func(extra_obj, extra_scratch, field, config, logger)


def ProcessExtraOutputsForStamp(config, logger=None):
    """Run the appropriate processing code for any extra output items that need to do something
    at the end of building each object.

    This gets called after all the object flux is added to the stamp, but before the sky level
    and noise are added.

    @param config       The configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]
    """
    if 'output' in config:
        obj_num = config['obj_num'] - config['start_obj_num']
        for key in [ k for k in valid_extra_outputs.keys() if k in config['output'] ]:
            stamp_func = valid_extra_outputs[key][3]
            if stamp_func is not None:
                extra_obj = config['extra_objs'][key]
                extra_scratch = config['extra_scratch'][key]
                field = config['output'][key]
                stamp_func(extra_obj, extra_scratch, field, config, obj_num, logger)


def ProcessExtraOutputsForImage(config, logger=None):
    """Run the appropriate processing code for any extra output items that need to do something
    at the end of building each image

    @param config       The configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]
    """
    if 'output' in config:
        for key in [ k for k in valid_extra_outputs.keys() if k in config['output'] ]:
            image_func = valid_extra_outputs[key][4]
            if image_func is not None:
                extra_obj = config['extra_objs'][key]
                extra_scratch = config['extra_scratch'][key]
                field = config['output'][key]
                image_func(extra_obj, extra_scratch, field, config, logger)


def WriteExtraOutputs(config, logger=None):
    """Write the extra output objects to files.

    This gets run at the end of the functions for building the regular output files.

    @param config       The configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]
    """
    config['index_key'] = 'file_num'
    if 'output' in config:
        output = config['output']
        if 'retry_io' in output:
            ntries = galsim.config.ParseValue(config['output'],'retry_io',config,int)[0]
            # This is how many retries.  Do at least 1, so ntries is 1 more than this.
            ntries = ntries + 1
        else:
            ntries = 1

        if 'dir' in output:
            default_dir = galsim.config.ParseValue(output,'dir',config,str)[0]
        else:
            default_dir = None

        if 'noclobber' in output:
            noclobber = galsim.config.ParseValue(output,'noclobber',config,bool)[0]
        else:
            noclobber = False

        if 'extra_objs_last_file' not in config:
            config['extra_objs_last_file'] = {}

        for key in [ k for k in valid_extra_outputs.keys() if k in output ]:
            write_func = valid_extra_outputs[key][5]
            if write_func is None: continue

            field = output[key]
            if 'file_name' in field:
                galsim.config.SetDefaultExt(field, '.fits')
                file_name = galsim.config.ParseValue(field,'file_name',config,str)[0]
            else:
                # If no file_name, then probably writing to hdu
                continue
            if 'dir' in field:
                dir = galsim.config.ParseValue(field,'file_name',config,str)[0]
            else:
                dir = default_dir

            if dir is not None:
                file_name = os.path.join(dir,file_name)

            if noclobber and os.path.isfile(file_name):
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('Not writing %s file %d = %s because output.noclobber = True' +
                                ' and file exists',key,config['file_num'],file_name)
                continue

            if config['extra_objs_last_file'].get(key, None) == file_name:
                # If we already wrote this file, skip it this time around.
                # (Typically this is applicable for psf, where we may only want 1 psf file.)
                if logger and logger.isEnabledFor(logging.INFO):
                    logger.info('Not writing %s file %d = %s because already written',
                                key,config['file_num'],file_name)
                continue

            extra_obj = config['extra_objs'][key]

            # If we have a method, we need to attach it to the extra_obj, since it might
            # be a proxy, in which case the method call won't work.
            if inspect.ismethod(write_func):
                write_func = eval('extra_obj.' + write_func.__name__)
                args = (file_name,)
            else:
                args = (extra_obj, file_name)
            galsim.config.RetryIO(write_func, args, ntries, file_name, logger)
            config['extra_objs_last_file'][key] = file_name
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Wrote %s to %r',config['file_num'],key,file_name)


def BuildExtraOutputHDUs(config, logger=None, first=1):
    """Write the extra output objects to either HDUS or images as appropriate.

    This gets run at the end of the functions for building the regular output files.

    Note: the extra items must have hdu numbers ranging continuously (in any order) starting
    at first.  Typically first = 1, since the main image is the primary HDU, numbered 0.

    @param config       The configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]
    @param first        The first number allowed for the extra hdus. [default: 1]

    @returns a list of HDUs and/or Images to put in the output FITS file.
    """
    config['index_key'] = 'file_num'
    if 'output' in config:
        output = config['output']
        hdus = {}
        for key in [ k for k in valid_extra_outputs.keys() if k in output ]:
            hdu_func = valid_extra_outputs[key][6]
            if hdu_func is None: continue

            field = output[key]
            if 'hdu' in field:
                hdu = galsim.config.ParseValue(field,'hdu',config,int)[0]
            else:
                # If no hdu, then probably writing to file
                continue
            if hdu <= 0 or hdu in hdus.keys():
                raise ValueError("%s hdu = %d is invalid or a duplicate."%hdu)

            extra_obj = config['extra_objs'][key]
            extra_type = valid_extra_outputs[key][0]

            # If we have a method, we need to attach it to the extra_obj, since it might
            # be a proxy, in which case the method call won't work.
            if inspect.ismethod(hdu_func):
                hdu_func = eval('extra_obj.' + hdu_func.__name__)
                hdus[hdu] = hdu_func()
            else:
                hdus[hdu] = hdu_func(extra_obj)

        for h in range(first,len(hdus)+first):
            if h not in hdus.keys():
                raise ValueError("Cannot skip hdus.  Not output found for hdu %d"%h)
        # Turn hdus into a list (in order)
        hdulist = [ hdus[k] for k in range(first,len(hdus)+first) ]
        return hdulist
    else:
        return []


# valid_extra_outputs is a dict that defines how to process each of the extra output items.
# The dict is empty here.  The appropriate items are added to the dict in extra_*.py, and
# this provides the hook for other modules to add additional output items.
# The keys here are the names of the output items, and the values are tuples with:
# - the class name of the object to build to be output.
# - a function to get the initialization kwargs if building something.
#   The call signature is GetKwargs(config, base, logger)
# - a function to call at the start of each image
#   The call signature is Setup(output_obj, scratch, config, base, logger)
# - a function to call at the end of building each stamp
#   The call signature is ProcessStamp(output_obj, scratch, config, base, obj_num, logger)
# - a function to call at the end of building each image
#   The call signature is ProcessImage(output_obj, scratch, config, base, logger)
# - a function to call to write the output file
#   The call signature is WriteFile(output_obj, file_name)
# - a function to call to build either a FITS HDU or an Image to put in an HDU
#   The call signature is hdu = WriteToHDU(output_obj)

valid_extra_outputs = {}


