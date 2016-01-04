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

# This file handles processing the input items according to the specifications in config['input'].
# This file includes the basic functionality, which is often sufficient for simple input types,
# but it has hooks to allow more customized behavior where necessary. See input_*.py for examples.

def ProcessInput(config, file_num=0, logger=None, file_scope_only=False, safe_only=False):
    """
    Process the input field, reading in any specified input files or setting up
    any objects that need to be initialized.

    Each item registered as a valid input type will be built and available at the top level
    of config in config['input_objs'].  Since there is allowed to be more than one of each type
    of input object (e.g. multilpe catalogs or multiple dicts), these are actually lists.
    If there is only one e.g. catalog entry in config['input'], then this list will have one
    element.

    e.g. config['input_objs']['catalog'][0] holds the first catalog item defined in
    config['input']['catalog'] (if any).

    @param config           The configuutation dict to process
    @param file_num         The file number being worked on currently [default: 0]
    @param logger           If given, a logger object to log progress. [default: None]
    @param file_scope_only  If True, only process the input items that are marked as being
                            possibly relevant for file- and image-level items. [default: False]
    @param safe_only        If True, only process the input items whose construction parameters
                            are not going to change every file, so it can be made once and
                            used by multiple processes if appropriate. [default: False]
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

        # We don't need the manager stuff if we (a) are already in a multiprocessing Process, or
        # (b) we are only loading for file scope, or (c) both config.image.nproc and
        # config.output.nproc == 1.
        use_manager = (
                'current_nproc' not in config and
                not file_scope_only and
                ( ('image' in config and 'nproc' in config['image'] and
                   galsim.config.ParseValue(config['image'], 'nproc', config, int)[0] != 1) or
                  ('output' in config and 'nproc' in config['output'] and
                   galsim.config.ParseValue(config['output'], 'nproc', config, int)[0] != 1) ) )

        if use_manager and 'input_manager' not in config:
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
                    init_func = valid_input_types[key]['init']
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
            if file_scope_only and not valid_input_types[key]['file']: continue

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
                init_func = valid_input_types[key]['init']
                if input_objs[i] is not None and input_objs_safe[i]:
                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug('file %d: Using %s already read in',file_num,key)
                else:
                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug('file %d: Build input type %s',file_num,key)
                    try:
                        kwargs, safe = GetInputKwargs(key, field, config)
                    except Exception as e:
                        if safe_only:
                            if logger and logger.isEnabledFor(logging.DEBUG):
                                logger.debug('file %d: Skip %s %d, since caugt exception: %s',
                                             file_num,key,i,e)
                            input_objs[i] = None
                            input_objs_safe[i] = None
                            continue
                        else:
                            raise

                    if safe_only and not safe:
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug('file %d: Skip %s %d, since not safe',file_num,key,i)
                        input_objs[i] = None
                        input_objs_safe[i] = None
                        continue

                    if use_manager:
                        tag = key + str(i)
                        input_obj = getattr(config['input_manager'],tag)(**kwargs)
                    else:
                        init_func = valid_input_types[key]['init']
                        input_obj = init_func(**kwargs)

                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug('file %d: Built input object %s %d',file_num,key,i)
                        if 'file_name' in kwargs:
                            logger.debug('file %d: file_name = %s',file_num,kwargs['file_name'])
                    if logger and logger.isEnabledFor(logging.INFO):
                        if valid_input_types[key]['nobj']:
                            logger.info('Read %d objects from %s',input_obj.getNObjects(),key)

                    # Store input_obj in the config for use by BuildGSObject function.
                    input_objs[i] = input_obj
                    input_objs_safe[i] = safe
                    # Invalidate any currently cached values that use this kind of input object:
                    # TODO: This isn't quite correct if there are multiple versions of this input
                    #       item.  e.g. you might want to invalidate dict0, but not dict1.
                    for value_type in valid_input_types[key]['types']:
                        galsim.config.RemoveCurrent(config, type=value_type)
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug('file %d: Cleared current_vals for items with type %s',
                                         file_num,value_type)

        # Check that there are no other attributes specified.
        valid_keys = valid_input_types.keys()
        galsim.config.CheckAllParams(config['input'], ignore=valid_keys)

def GetInputKwargs(key, config, base):
    """Get the kwargs to use for initializing the input object

    @param key      The name of the input type
    @param config   The config dict for this input item, typically base['input'][key]
    @param base     The base config dict
    """
    kwargs_func = valid_input_types[key]['kwargs']
    if kwargs_func is None:
        init_func = valid_input_types[key]['init']
        req = init_func._req_params
        opt = init_func._opt_params
        single = init_func._single_params
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, single=single)
        if init_func._takes_rng:
            if 'rng' not in config:
                raise ValueError("No config['rng'] available for %s.type"%key)
            kwargs['rng'] = config['rng']
            safe = False
        return kwargs, safe
    else:
        return kwargs_func(config, base)

 
def ProcessInputNObjects(config, logger=None):
    """Process the input field, just enough to determine the number of objects.

    Some input items are relevant for determining the number of objects in a file or image.
    This means we need to have them processed before splitting up jobs over multiple processes
    (since the seed increments based on the number of objects).  So this function builds
    the input items that have a getNObjects() method using the _nobject_only construction
    argument and returns the number of objects.

    Caveat: This function tries each input type in galsim.config.valid_input_types in
            order and returns the nobjects for the first one that works.  If multiple input
            items have nobjects and they are inconsistent, this function may return a
            number of objects that isn't what you wanted.  In this case, you should explicitly
            set nobjects or nimages in the configuratin dict, rather than relying on this
            galsim.config "magic".

    @param config       The configuutation dict to process
    @param logger       If given, a logger object to log progress. [default: None]

    @returns the number of objects to use.
    """
    config['index_key'] = 'file_num'
    if 'input' in config:
        for key in valid_input_types:
            has_nobjects = valid_input_types[key]['nobj']
            if key in config['input'] and has_nobjects:
                field = config['input'][key]

                if key in config['input_objs'] and config['input_objs'][key+'_safe'][0]:
                    input_obj = config['input_objs'][key][0]
                else:
                    # If it's a list, just use the first one.
                    if isinstance(field, list): field = field[0]

                    init_func = valid_input_types[key]['init']
                    kwargs, safe = GetInputKwargs(key, field, config)
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
            setup_func = valid_input_types[key]['setup']
            if key in config['input'] and setup_func is not None:
                fields = config['input'][key]
                if not isinstance(fields, list):
                    fields = [ fields ]
                input_objs = config['input_objs'][key]

                for i in range(len(fields)):
                    field = fields[i]
                    input_obj = input_objs[i]
                    setup_func(input_obj, field, config, logger)


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


valid_input_types = {}

def RegisterInputType(input_type, init_func, types, kwargs_func=None, has_nobj=False, 
                      file_scope=False, setup_func=None):
    """Register an input type for use by the config apparatus.

    @param input_type       The name of the type in config['input']
    @param init_func        A function or class name to use to build the input object.
    @param types            A list of value or object types that use this input type.
                            These items will have their "current" values invalidated when the
                            input object changes.
    @param kwargs_func      A function to get the initialization kwargs if the regular
                            _req, _opt, etc. kind of initialization will not work. The call
                            signature is:
                                kwargs, safe = GetKwargs(config, base)
                            [default: None, which means use the regular initialization]
    @param has_nobj         Whether the object can be used to automatically determine the number
                            of objects to build for a given file or image.  If True, it must have
                            a getNObjects() method and also a construction kwargs _nobjects_only
                            to efficiently do only enough to calculate nobjects. [default: False]
    @param file_scope       Whether the class might be relevant at the file- or image-scope level,
                            rather than just at the object level.  Notably, this is true for dict.
                            [default: False]
    @param setup_func       A function to call at the start of each image. The call signature is
                                Setup(input_obj, config, base, logger)
                            [default: None]
    """
    valid_input_types[input_type] = {
        'init' : init_func,
        'types' : types,
        'kwargs' : kwargs_func,
        'nobj' : has_nobj,
        'file' : file_scope,
        'setup' : setup_func
    }

# We define in this file two simple input types: catalog and dict, which read in a Catalog
# or Dict from a file and then can use that to generate values.
RegisterInputType('catalog', galsim.Catalog, ['Catalog'], has_nobj=True)
RegisterInputType('dict', galsim.Dict, ['Dict'], file_scope=True)



# Now define the value generators connected to the catalog and dict input types.
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
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    col = kwargs['col']
    index = kwargs['index']

    if value_type is str:
        val = input_cat.get(index, col)
    elif value_type is float:
        val = input_cat.getFloat(index, col)
    elif value_type is int:
        val = input_cat.getInt(index, col)
    elif value_type is bool:
        val = galsim.config.value._GetBoolValue(input_cat.get(index, col))

    #print base['file_num'],'Catalog: col = %s, index = %s, val = %s'%(col, index, val)
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

# Register these as valid value types
from .value import RegisterValueType
RegisterValueType('Catalog', _GenerateFromCatalog, [ float, int, bool, str ])
RegisterValueType('Dict', _GenerateFromDict, [ float, int, bool, str ])
