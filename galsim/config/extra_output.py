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

valid_extra_outputs = { 
    # The values are tuples with:
    # - the class name to build, if any.
    # - a function to get the initialization kwargs if building something.
    # - a function to call at the start of each image
    # - a function to call at the end of building each stamp
    # - a function to call at the end of building each image
    # - a function to call to write the output file
    # - a function to call to build either a FITS HDU or an Image to put in an HDU
    'psf' : ('galsim.ImageF', None, None, None, None, None, 'galsim.ImageF.view',
             ['draw_method', 'signal_to_noise']),
    'weight' : ('galsim.ImageF', None,
                'SetupWeight', 'ProcessWeightStamp', 'ProcessWeightImage', 
                'galsim.ImageF.write', 'galsim.ImageF.view',
                ['weight']),
    'badpix' : ('galsim.ImageS', None,
                'SetupBadPix', 'ProcessBadPixStamp', 'ProcessBadPixImage', 
                'galsim.ImageS.write', 'galsim.ImageS.view',
                []),
    'truth' : ('galsim.OutputCatalog', 'GetTruthKwargs',
                None, 'ProcessTruth', None,
               'galsim.OutputCatalog.write', 'galsim.OutputCatalog.write_fits_hdu',
               ['columns']),
}

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
 
        if 'output_manager' not in config:
            from multiprocessing.managers import BaseManager, DictProxy
            class OutputManager(BaseManager): pass
 
            # Register each input field with the OutputManager class
            for key in all_keys:
                fields = output[key]
                # Register this object with the manager
                extra_type = valid_extra_outputs[key][0]
                if extra_type in galsim.__dict__:
                    init_func = eval("galsim."+extra_type)
                else:
                    init_func = eval(extra_type)
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
                kwargs_func = eval(kwargs_func)
                kwargs = kwargs_func(field, config, logger)
            else:
                # use default constructor
                kwargs = {}
 
            output_obj = getattr(config['output_manager'],key)(**kwargs)
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Setup output %s object',file_num,key)
            config['extra_objs'][key] = output_obj
            config['extra_scratch'][key] = config['output_manager'].dict()


def SetupExtraOutputsForImage(config, nobjects, logger=None):
    """Perform any necessary setup for the extra output items at the start of a new image.

    @param config       The configuration dict.
    @param nobjects     The number of objects that will be built for this image.
    @param logger       If given, a logger object to log progress. [default: None]
    """
    if 'output' in config:
        for key in [ k for k in valid_extra_outputs.keys() if k in config['output'] ]:
            setup_func = valid_extra_outputs[key][2]
            if setup_func is not None:
                extra_obj = config['extra_objs'][key]
                extra_scratch = config['extra_scratch'][key]
                func = eval(setup_func)
                field = config['output'][key]
                func(extra_obj, extra_scratch, field, config, nobjects, logger)


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
                func = eval(stamp_func)
                field = config['output'][key]
                func(extra_obj, extra_scratch, field, config, obj_num, logger)


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
                func = eval(image_func)
                field = config['output'][key]
                func(extra_obj, extra_scratch, field, config, logger)


def WriteExtraOutputs(config, logger=None):
    """Write the extra output objects to files.

    This gets run at the end of the functions for building the regular output files.

    @param config       The configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]
    """
    from galsim.config.output import _retry_io
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
                                ' and file exists',key,output['file_num'],file_name)
                continue

            if config['extra_objs_last_file'].get(key, None) == file_name:
                # If we already wrote this file, skip it this time around.
                # (Typically this is applicable for psf, where we may only want 1 psf file.)
                if logger and logger.isEnabledFor(logging.INFO):
                    logger.info('Not writing %s file %d = %s because already written',
                                key,output['file_num'],file_name)
                continue

            extra_obj = config['extra_objs'][key]
            extra_type = valid_extra_outputs[key][0]
            if write_func.startswith(extra_type):
                # Methods can't be called with proxy objects as the self parameter,
                # so we need to actually call the method on the proxy instead.
                method = write_func[len(extra_type)+1:]
                func = eval('extra_obj.' + method)
                _retry_io(func, (file_name,), ntries, file_name, logger)
            else:
                func = eval(write_func)
                _retry_io(func, (extra_obj, file_name), ntries, file_name, logger)
            config['extra_objs_last_file'][key] = file_name
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Wrote %s to %r',file_num,key,file_name)


def BuildExtraOutputHDUs(config, logger=None):
    """Write the extra output objects to either HDUS or images as appropriate.

    This gets run at the end of the functions for building the regular output files.

    @param config       The configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]

    @returns a list of HDUs and/or Images to put in the output FITS file.
    """
    from galsim.config.output import _retry_io
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
            if hdu_func.startswith(extra_type):
                # Methods can't be called with proxy objects as the self parameter,
                # so we need to actually call the method on the proxy instead.
                method = hdu_func[len(extra_type)+1:]
                func = eval('extra_obj.' + method)
                hdus[hdu] = func()
            else:
                func = eval(hdu_func)
                hdus[hdu] = func(extra_obj)

        for h in range(1,len(hdus)+1):
            if h not in hdus.keys():
                raise ValueError("Cannot skip hdus.  Not output found for hdu %d"%h)
        # Turn hdus into a list (in order)
        hdulist = [ hdus[k] for k in range(1,len(hdus)+1) ]
        return hdulist
    else:
        return []


#
# The functions for weight
#

def SetupWeight(image, scratch, config, base, nobjects, logger=None):
    image.resize(base['current_image'].bounds, wcs=base['wcs'])
    image.setZero()
    scratch.clear()

def ProcessWeightStamp(image, scratch, config, base, obj_num, logger=None):
    if base['do_noise_in_stamps']:
        weight_im = galsim.ImageF(base['current_stamp'].bounds, wcs=base['wcs'], init_value=0)
        if 'include_obj_var' in base['output']['weight']:
            include_obj_var = galsim.config.ParseValue(
                    base['output']['weight'], 'include_obj_var', config, bool)[0]
        else:
            include_obj_var = False
        galsim.config.AddNoiseVariance(base,weight_im,include_obj_var,logger)
        scratch[obj_num] = weight_im

def ProcessWeightImage(image, scratch, config, base, logger=None):
    if len(scratch) > 0.:
        # If we have been accumulating the variance on the stamps, build the total from them.
        for stamp in scratch.values():
            b = stamp.bounds & image.getBounds()
            if b.isDefined():
                # This next line is equivalent to:
                #    image[b] += stamp[b]
                # except that this doesn't work through the proxy.  We can only call methods
                # that don't start with _.  Hence using the more verbose form here.
                image.setSubImage(b, image.subImage(b) + stamp[b])
    else:
        # Otherwise, build the variance map now.
        if 'include_obj_var' in base['output']['weight']:
            include_obj_var = galsim.config.ParseValue(
                    base['output']['weight'], 'include_obj_var', config, bool)[0]
        else:
            include_obj_var = False
        if isinstance(image, galsim.Image):
            galsim.config.AddNoiseVariance(base,image,include_obj_var,logger)
        else:
            # If we are using a Proxy for the image, the code in AddNoiseVar won't work properly.
            # The easiest workaround is to build a new image here and copy it over.
            im2 = galsim.ImageF(image.getBounds(), wcs=base['wcs'], init_value=0)
            galsim.config.AddNoiseVariance(base,im2,include_obj_var,logger)
            image.copyFrom(im2)
 
    # Now invert the variance image to get weight map.
    image.invertSelf()


#
# The functions for badpix
#

def SetupBadPix(image, scratch, config, base, nobjects, logger=None):
    image.resize(base['current_image'].bounds, wcs=base['wcs'])
    image.setZero()
    scratch.clear()

def ProcessBadPixStamp(image, scratch, config, base, obj_num, logger=None):
    # Note: This is just a placeholder for now.  Once we implement defects, saturation, etc.,
    # these features should be marked in the badpix mask.  For now though, all pixels = 0.
    if base['do_noise_in_stamps']:
        badpix_im = galsim.ImageF(base['current_stamp'].bounds, wcs=base['wcs'], init_value=0)
        scratch[obj_num] = badpix_im

def ProcessBadPixImage(image, scratch, config, base, logger=None):
    if len(scratch) > 0.:
        # If we have been accumulating the variance on the stamps, build the total from them.
        for stamp in scratch.values():
            b = stamp.bounds & image.getBounds()
            if b.isDefined():
                # This next line is equivalent to:
                #    image[b] |= stamp[b]
                # except that this doesn't work through the proxy.  We can only call methods
                # that don't start with _.  Hence using the more verbose form here.
                image.setSubImage(b, image.subImage(b) | stamp[b])
    else:
        # Otherwise, build the bad pixel mask here.
        # Again, nothing here yet.
        pass
 
    # Now invert the variance image to get weight map.
    image.invertSelf()

 
#
# The functions for truth
#

def GetTruthKwargs(config, base, logger=None):
    columns = config['columns']
    truth_names = columns.keys()
    return { 'names' : truth_names }
 

def ProcessTruth(truth_cat, scratch, config, base, obj_num, logger=None):
    truth_cat.lock_acquire()
    cols = config['columns']
    row = []
    types = []
    for name in truth_cat.getNames():
        key = cols[name]
        if isinstance(key, dict):
            # Then the "key" is actually something to be parsed in the normal way.
            # Caveat: We don't know the value_type here, so we give None.  This allows
            # only a limited subset of the parsing.  Usually enough for truth items, but
            # not fully featured.
            value = galsim.config.ParseValue(cols,name,base,None)[0]
            t = type(value)
        elif not isinstance(key,basestring):
            # The item can just be a constant value.
            value = key
            t = type(value)
        elif key[0] == '$':
            # This can also be handled by ParseValue
            value = galsim.config.ParseValue(cols,name,base,None)[0]
            t = type(value)
        else:
            value, t = galsim.config.GetCurrentValue(key, base)
        row.append(value)
        types.append(t)
    if truth_cat.getNObjects() == 0:
        truth_cat.setTypes(types)
    elif truth_cat.getTypes() != types:
        if logger:
            logger.error("Type mismatch found when building truth catalog at object %d",
                base['obj_num'])
            logger.error("Types for current object = %s",repr(types))
            logger.error("Expecting types = %s",repr(truth_cat.getTypes()))
        raise RuntimeError("Type mismatch found when building truth catalog.")
    truth_cat.add_row(row, obj_num)
    truth_cat.lock_release()


