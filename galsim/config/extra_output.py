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

valid_extra_output_items = { 
    # The values are tuples with:
    # - the class name to build, if any.
    # - a list of keys to ignore on the initial creation.
    # - a function to get the initialization kwargs if building something.
    # - a function to call at the start of each file
    # - a function to call at the end of each stamp processing (or None)
    # - a function to call at the end of each file
    'psf' : (None, ['draw_method', 'signal_to_noise'], None, None, None),
    'weight' : (None, ['weight'], None, None, None),
    'badpix' : (None, [], None, None, None),
    'truth' : ('galsim.OutputCatalog', ['columns'], 'galsim.config.GetTruthKwargs',
               'galsim.config.ProcessTruth', 'galsim.config.WriteTruth'),
}

def SetupExtraOutput(config, file_num=0, logger=None):
    """
    Set up the extra output items as necessary, including building Managers for them
    so their objects can be updated safely in multi-processing mode.

    For example, the truth item needs to have the OutputCatalog set up and managed
    so each process can add rows to it without getting race conditions or clobbering
    each others' rows.
    """
    if 'output' in config:
        output = config['output']
        if not isinstance(output, dict):
            raise AttributeError("config.output is not a dict.")

        # We'll iterate through this list of keys a few times
        all_keys = [ k for k in valid_extra_output_items.keys()
                     if (k in output and valid_extra_output_items[k][0] is not None) ]
 
        if 'output_manager' not in config:
            from multiprocessing.managers import BaseManager
            class OutputManager(BaseManager): pass
 
            # Register each input field with the OutputManager class
            for key in all_keys:
                fields = output[key]
                # Register this object with the manager
                type = valid_extra_output_items[key][0]
                if type in galsim.__dict__:
                    init_func = eval("galsim."+type)
                else:
                    init_func = eval(type)
                OutputManager.register(key, init_func)
            # Start up the output_manager
            config['output_manager'] = OutputManager()
            config['output_manager'].start()

        for key in all_keys:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Setup output item %s',file_num,key)
            kwargs_func = valid_extra_output_items[key][2]
            if kwargs_func is None:
                type = valid_extra_output_items[key][0]
                if type in galsim.__dict__:
                    init_func = eval("galsim."+type)
                else:
                    init_func = eval(type)
                ignore = valid_extra_output_items[key][1]
                ignore += ['file_name', 'hdu', 'file_type', 'dir']
                kwargs = galsim.config.GetAllParams(field, key, config,
                                                    req = init_func._req_params,
                                                    opt = init_func._opt_params,
                                                    single = init_func._single_params,
                                                    ignore = ignore)[0]
            else:
                kwargs_func = eval(kwargs_func)
                kwargs = kwargs_func(config, logger)
 
            output_obj = getattr(config['output_manager'],key)(**kwargs)
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Setup output %s object',file_num,key)
            config[key] = output_obj


def GetTruthKwargs(config, logger):
    if 'truth' not in config['output']:
        raise AttributeError("No 'truth' field found in config.output")
    if 'columns' not in config['output']['truth']:
        raise AttributeError("No 'columns' listed for config.output.truth")
    columns = config['output']['truth']['columns']
    truth_names = columns.keys()
    return { 'names' : truth_names }
 

def WriteTruth(config, file_name):
    """Write the truth catalog to a file
    """
    config['truth'].write(file_name)


def ProcessTruth(config, logger=None):
    """
    Put the appropriate current_val's into the truth catalog.

    @param config           A configuration dict.
    @param logger           If given, a logger object to log progress. [default: None]
    """
    if ('output' not in config or 'truth' not in config['output'] or 
        'columns' not in config['output']['truth']):
        raise RuntimeError("config has no output.truth.columns field")
    if 'truth' not in config:
        raise RuntimeError("config has no truth catalog")
    cat = config['truth']
    cat.lock_acquire()
    cols = config['output']['truth']['columns']
    row = []
    types = []
    for name in cat.getNames():
        key = cols[name]
        if isinstance(key, dict):
            # Then the "key" is actually something to be parsed in the normal way.
            # Caveat: We don't know the value_type here, so we give None.  This allows
            # only a limited subset of the parsing.  Usually enough for truth items, but
            # not fully featured.
            value = galsim.config.ParseValue(cols,name,config,None)[0]
            t = type(value)
        elif not isinstance(key,basestring):
            # The item can just be a constant value.
            value = key
            t = type(value)
        elif key[0] == '$':
            # This can also be handled by ParseValue
            value = galsim.config.ParseValue(cols,name,config,None)[0]
            t = type(value)
        else:
            value, t = galsim.config.GetCurrentValue(key, name, config)
        row.append(value)
        types.append(t)
    if cat.getNObjects() == 0:
        cat.setTypes(types)
    elif cat.getTypes() != types:
        if logger:
            logger.error("Type mismatch found when building truth catalog at object %d",
                config['obj_num'])
            logger.error("Types for current object = %s",repr(types))
            logger.error("Expecting types = %s",repr(cat.getTypes()))
        raise RuntimeError("Type mismatch found when building truth catalog.")
    cat.add_row(row, config['obj_num'])
    cat.lock_release()


