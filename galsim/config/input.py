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

from .input_powerspectrum import PowerSpectrumInit

valid_input_types = { 
    # The values are tuples with:
    # - The class name to build.
    # - A list of keys to ignore on the initial creation (e.g. PowerSpectrum has values that are 
    #   used later in PowerSpectrumInit).
    # - Whether the class has a getNObjects method, in which case it also must have a constructor
    #   kwarg _nobjects_only to efficiently do only enough to calculate nobjects.
    # - Whether the class might be relevant at the file- or image-scope level, rather than just
    #   at the object level.  Notably, this is true for dict.
    # - A function to call at the start of each image (or None)
    # - A list of types that should have their "current" values invalidated when the input
    #   object changes.
    # See the des module for examples of how to extend this from a module.
    'catalog' : ('galsim.Catalog', [], True, False, None, ['Catalog']), 
    'dict' : ('galsim.Dict', [], False, True, None, ['Dict']), 
    'real_catalog' : ('galsim.RealGalaxyCatalog', [], True, False, None, 
                      ['RealGalaxy', 'RealGalaxyOriginal']),
    'cosmos_catalog' : ('galsim.COSMOSCatalog', [], True, False, None, ['COSMOSGalaxy']),
    'nfw_halo' : ('galsim.NFWHalo', [], False, False, None,
                  ['NFWHaloShear','NFWHaloMagnification']),
    'power_spectrum' : ('galsim.PowerSpectrum',
                        # power_spectrum uses these extra parameters in PowerSpectrumInit
                        ['grid_spacing', 'interpolant'], 
                        False, False,
                        'galsim.config.PowerSpectrumInit',
                        ['PowerSpectrumShear','PowerSpectrumMagnification']),
    'fits_header' : ('galsim.FitsHeader', [], False, True, None, ['FitsHeader']), 
}


class InputGetter:
    """A simple class that is returns a given config[key][i] when called with obj()
    """
    def __init__(self, config, key, i):
        self.config = config
        self.key = key
        self.i = i
    def __call__(self): return self.config[self.key][self.i]


def ProcessInput(config, file_num=0, logger=None, file_scope_only=False, safe_only=False):
    """
    Process the input field, reading in any specified input files or setting up
    any objects that need to be initialized.

    Each item in the above valid_input_types will be built and available at the top level
    of config.  e.g.;
        config['catalog'] = the catalog specified by config.input.catalog, if provided.
        config['real_catalog'] = the catalog specified by config.input.real_catalog, if provided.
        etc.
    """
    config['index_key'] = 'file_num'
    config['file_num'] = file_num
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Start ProcessInput',file_num)
    # Process the input field (read any necessary input files)
    if 'input' in config:
        # We'll iterate through this list of keys a few times
        all_keys = [ k for k in valid_input_types.keys() if k in config['input'] ]

        # First, make sure all the input fields are lists.  If not, then we make them a 
        # list with one element.
        for key in all_keys:
            if not isinstance(config['input'][key], list):
                config['input'][key] = [ config['input'][key] ]
 
        # The input items can be rather large.  Especially RealGalaxyCatalog.  So it is
        # unwieldy to copy them in the config file for each process.  Instead we use proxy
        # objects which are implemented using multiprocessing.BaseManager.  See
        #
        #     http://docs.python.org/2/library/multiprocessing.html
        #
        # The input manager keeps track of all the real objects for us.  We use it to put
        # a proxy object in the config dict, which is copyable to other processes.
        # The input manager itself should not be copied, so the function CopyConfig makes
        # sure to only keep that in the original config dict, not the one that gets passed
        # to other processed.
        # The proxy objects are  able to call public functions in the real object via 
        # multiprocessing communication channels.  (A Pipe, I believe.)  The BaseManager 
        # base class handles all the details.  We just need to register each class we need 
        # with a name (called tag below) and then construct it by calling that tag function.
        if 'input_manager' not in config:
            from multiprocessing.managers import BaseManager
            class InputManager(BaseManager): pass
 
            # Register each input field with the InputManager class
            for key in all_keys:
                fields = config['input'][key]

                # Register this object with the manager
                for i in range(len(fields)):
                    field = fields[i]
                    tag = key + str(i)
                    # This next bit mimics the operation of BuildSimple, except that we don't
                    # actually build the object here.  Just register the class name.
                    input_type = valid_input_types[key][0]
                    if input_type in galsim.__dict__:
                        init_func = eval("galsim."+input_type)
                    else:
                        init_func = eval(input_type)
                    InputManager.register(tag, init_func)
            # Start up the input_manager
            config['input_manager'] = InputManager()
            config['input_manager'].start()

        if 'input_objs' not in config:
            config['input_objs'] = {}
            for key in all_keys:
                fields = config['input'][key]
                config['input_objs'][key] = [ None for i in range(len(fields)) ]
                config['input_objs'][key+'_safe'] = [ None for i in range(len(fields)) ]

        # Read all input fields provided and create the corresponding object
        # with the parameters given in the config file.
        for key in all_keys:
            # Skip this key if not relevant for file_scope_only run.
            if file_scope_only and not valid_input_types[key][3]: continue

            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Process input key %s',file_num,key)
            fields = config['input'][key]

            for i in range(len(fields)):
                field = fields[i]
                input_objs = config['input_objs'][key]
                input_objs_safe = config['input_objs'][key+'_safe']
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file %d: Current values for %s are %s, safe = %s',
                                 file_num, key, str(input_objs[i]), input_objs_safe[i])
                input_type, ignore = valid_input_types[key][0:2]
                field['type'] = input_type
                if input_objs[i] is not None and input_objs_safe[i]:
                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug('file %d: Using %s already read in',file_num,key)
                else:
                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug('file %d: Build input type %s',file_num,input_type)
                    # This is almost identical to the operation of BuildSimple.  However,
                    # rather than call the regular function here, we have input_manager do so.
                    if input_type in galsim.__dict__:
                        init_func = eval("galsim."+input_type)
                    else:
                        init_func = eval(input_type)
                    kwargs, safe = galsim.config.GetAllParams(field, config,
                                                              req = init_func._req_params,
                                                              opt = init_func._opt_params,
                                                              single = init_func._single_params,
                                                              ignore = ignore)
                    if init_func._takes_rng:
                        if 'rng' not in config:
                            raise ValueError("No config['rng'] available for %s.type = %s"%(
                                             key,input_type))
                        kwargs['rng'] = config['rng']
                        safe = False

                    if safe_only and not safe:
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug('file %d: Skip %s %d, since not safe',file_num,key,i)
                        input_objs[i] = None
                        input_objs_safe[i] = None
                        continue

                    tag = key + str(i)
                    input_obj = getattr(config['input_manager'],tag)(**kwargs)
                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug('file %d: Built input object %s %d',file_num,key,i)
                        if 'file_name' in kwargs:
                            logger.debug('file %d: file_name = %s',file_num,kwargs['file_name'])
                    if logger and logger.isEnabledFor(logging.INFO):
                        if valid_input_types[key][2]:
                            logger.info('Read %d objects from %s',input_obj.getNObjects(),key)
                    # Store input_obj in the config for use by BuildGSObject function.
                    input_objs[i] = input_obj
                    input_objs_safe[i] = safe
                    # Invalidate any currently cached values that use this kind of input object:
                    # TODO: This isn't quite correct if there are multiple versions of this input
                    #       item.  e.g. you might want to invalidate dict0, but not dict1.
                    for value_type in valid_input_types[key][5]:
                        galsim.config.RemoveCurrent(config, type=value_type)
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug('file %d: Cleared current_vals for items with type %s',
                                         file_num,value_type)

        # Check that there are no other attributes specified.
        valid_keys = valid_input_types.keys()
        galsim.config.CheckAllParams(config['input'], ignore=valid_keys)


def ProcessInputNObjects(config, logger=None):
    """Process the input field, just enough to determine the number of objects.
    """
    config['index_key'] = 'file_num'
    if 'input' in config:
        for key in valid_input_types:
            has_nobjects = valid_input_types[key][2]
            if key in config['input'] and has_nobjects:
                field = config['input'][key]

                if key in config['input_objs'] and config['input_objs'][key+'_safe'][0]:
                    input_obj = config['input_objs'][key][0]
                else:
                    # If it's a list, just use the first one.
                    if isinstance(field, list): field = field[0]

                    input_type, ignore = valid_input_types[key][0:2]
                    if input_type in galsim.__dict__:
                        init_func = eval("galsim."+input_type)
                    else:
                        init_func = eval(input_type)
                    kwargs = galsim.config.GetAllParams(field, config,
                                                        req = init_func._req_params,
                                                        opt = init_func._opt_params,
                                                        single = init_func._single_params,
                                                        ignore = ignore)[0]
                    kwargs['_nobjects_only'] = True
                    input_obj = init_func(**kwargs)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file %d: Found nobjects = %d for %s',
                                 config['file_num'],input_obj.getNOjects(),key)
                return input_obj.getNObjects()
    # If didn't find anything, return None.
    return None


def SetupInputsForImage(config, logger):
    """Do any necessary setup of the input items at the start of an image.

    @param config       The configuutation dict to process
    @param logger       If given, a logger object to log progress. [default: None]
    """
    if 'input' in config:
        for key in valid_input_types.keys():
            image_func = valid_input_types[key][4]
            if key in config['input'] and image_func is not None:
                fields = config['input'][key]
                if not isinstance(fields, list):
                    fields = [ fields ]
                input_objs = config['input_objs'][key]

                for i in range(len(fields)):
                    field = fields[i]
                    input_obj = input_objs[i]
                    func = eval(image_func)
                    func(input_obj, field, config)


# A helper function for getting the input object needed for generating a value or building
# a gsobject.
def GetInputObj(input_type, config, base, param_name):
    """Get the input object needed for generating a particular value
    @param input_type   The type of input object to get
    @param config       The config dict for this value item
    @param base         The base config dict
    @param param_name   The type of value that we are trying to construct (only used for
                        error messages).
    """
    if input_type not in base['input_objs']:
        raise ValueError("No input %s available for type = %s"%(input_type,param_name))

    if 'num' in config:
        num = galsim.config.ParseValue(config, 'num', base, int)[0]
    else:
        num = 0

    if num < 0:
        raise ValueError("Invalid num < 0 supplied for %s: num = %d"%(param_name,num))
    if num >= len(base['input_objs'][input_type]):
        raise ValueError("Invalid num supplied for %s (too large): num = %d"%(param_name,num))

    return base['input_objs'][input_type][num]


def _GenerateFromCatalog(config, base, value_type):
    """@brief Return a value read from an input catalog
    """
    input_cat = GetInputObj('catalog', config, base, 'Catalog')

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a Catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    galsim.config.SetDefaultIndex(config, input_cat.getNObjects())

    # Coding note: the and/or bit is equivalent to a C ternary operator:
    #     input_cat.isFits() ? str : int
    # which of course doesn't exist in python.  This does the same thing (so long as the 
    # middle item evaluates to true).
    req = { 'col' : input_cat.isFits() and str or int , 'index' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, ignore=['num'])

    if value_type is str:
        val = input_cat.get(**kwargs)
    elif value_type is float:
        val = input_cat.getFloat(**kwargs)
    elif value_type is int:
        val = input_cat.getInt(**kwargs)
    elif value_type is bool:
        val = galsim.config.value._GetBoolValue(input_cat.get(**kwargs))

    #print base['file_num'],
    #print 'Catalog: col = %s, index = %s, val = %s'%(kwargs['col'],kwargs['index'],val)
    return val, safe


def _GenerateFromDict(config, base, value_type):
    """@brief Return a value read from an input dict.
    """
    input_dict = GetInputObj('dict', config, base, 'Dict')

    req = { 'key' : str }
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    key = kwargs['key']

    val = input_dict.get(key)
    #print base['file_num'],'Dict: key = %s, val = %s'%(key,val)
    return val, safe


