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
    # - a list of keys to ignore on the initial creation.
    # - a function to get the initialization kwargs if building something.
    # - a function to call at the start of each file
    # - a function to call at the end of each stamp processing
    # - a function to call at the end of each file
    'psf' : (None, ['draw_method', 'signal_to_noise'], None, None, None, None),
    'weight' : (None, ['weight'], None, None, None, None),
    'badpix' : (None, [], None, None, None, None),
    'truth' : ('galsim.OutputCatalog', ['columns'],
               'galsim.config.GetTruthKwargs', None,
               'galsim.config.ProcessTruth', 'galsim.config.WriteTruth'),
}

def SetupExtraOutput(config, file_num=0, logger=None):
    """
    Set up the extra output items as necessary, including building Managers for them
    so their objects can be updated safely in multi-processing mode.

    For example, the truth item needs to have the OutputCatalog set up and managed
    so each process can add rows to it without getting race conditions or clobbering
    each others' rows.

    Each item that gets built will be placed in config['extra_objs'].  The objects will
    actually be proxy objects using a multiprocessing.Manager so that multiple processes
    can all communicate with it correctly.

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
            from multiprocessing.managers import BaseManager
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
            # Start up the output_manager
            config['output_manager'] = OutputManager()
            config['output_manager'].start()

        if 'extra_objs' not in config:
            config['extra_objs'] = {}

        for key in all_keys:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Setup output item %s',file_num,key)
            field = config['output'][key]
            kwargs_func = valid_extra_outputs[key][2]
            if kwargs_func is None:
                extra_type = valid_extra_outputs[key][0]
                if extra_type in galsim.__dict__:
                    init_func = eval("galsim."+extra_type)
                else:
                    init_func = eval(extra_type)
                ignore = valid_extra_outputs[key][1]
                ignore += ['file_name', 'hdu', 'file_type', 'dir']
                kwargs = galsim.config.GetAllParams(field, key, config,
                                                    req = init_func._req_params,
                                                    opt = init_func._opt_params,
                                                    single = init_func._single_params,
                                                    ignore = ignore)[0]
            else:
                kwargs_func = eval(kwargs_func)
                kwargs = kwargs_func(field, config, logger)
 
            output_obj = getattr(config['output_manager'],key)(**kwargs)
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Setup output %s object',file_num,key)
            config['extra_objs'][key] = output_obj

def ProcessExtraOutputsForStamp(config, logger=None):
    """Run the appropriate processing code for any extra output items that need to do something
    at the end of building each object

    @param config       The configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]
    """
    if 'output' in config:
        for key in [ k for k in valid_extra_outputs.keys() if k in config['output'] ]:
            stamp_func = valid_extra_outputs[key][4]
            if stamp_func is not None:
                extra_obj = config['extra_objs'][key]
                func = eval(stamp_func)
                field = config['output'][key]
                func(extra_obj, field, config, logger)


def GetTruthKwargs(config, base, logger=None):
    """Get the kwargs needed to build the truth OutputCatalog.

    @param config       The configuration dict for 'truth'.
    @param base         The base configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]

    @returns the kwargs to use for building the OutputCatalog.
    """
    columns = config['columns']
    truth_names = columns.keys()
    return { 'names' : truth_names }
 

def WriteTruth(truth_cat, file_name, config, base, logger=None):
    """Write the truth catalog to a file.

    @param truth_cat    The OutputCatalog to write.
    @param file_name    The file name to write to.
    @param config       The configuration dict for 'truth'.
    @param base         The base configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]
    """
    truth_cat.write(file_name)


def ProcessTruth(truth_cat, config, base, logger=None):
    """
    Put the appropriate current_val's into the truth catalog.

    @param truth_cat    The OutputCatalog in which to put the truth information.
    @param config       The configuration dict for 'truth'
    @param base         The base configuration dict.
    @param logger       If given, a logger object to log progress. [default: None]
    """
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
            value, t = galsim.config.GetCurrentValue(key, name, base)
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
    truth_cat.add_row(row, base['obj_num'])
    truth_cat.lock_release()


