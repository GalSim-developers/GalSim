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
    'real_catalog' : ('galsim.RealGalaxyCatalog', [],
                      False, # Actually it does have getNObjects, but that's probably not
                             # the right number of objects to use for a single file or image.
                      False, None,
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

def ProcessInput(config, file_num=0, logger=None, file_scope_only=False, safe_only=False):
    """
    Process the input field, reading in any specified input files or setting up
    any objects that need to be initialized.

    Each item in galsim.config.valid_input_types will be built and available at the top level
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
                    kwargs, safe = galsim.config.GetAllParams(field, key, config,
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

                    if use_manager:
                        tag = key + str(i)
                        input_obj = getattr(config['input_manager'],tag)(**kwargs)
                    else:
                        input_type = valid_input_types[key][0]
                        if input_type in galsim.__dict__:
                            init_func = eval("galsim."+input_type)
                        else:
                            init_func = eval(input_type)
                        input_obj = init_func(**kwargs)
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
        galsim.config.CheckAllParams(config['input'], 'input', ignore=valid_keys)

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
                    kwargs = galsim.config.GetAllParams(field, key, config,
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
                    func(input_obj, field, config, logger)


def PowerSpectrumInit(ps, config, base, logger=None):
    """Initialize the PowerSpectrum input object's gridded values based on the
    size of the image and the grid spacing.

    @param ps           The PowerSpectrum object to use
    @param config       The configuration dict for 'power_spectrum'
    @param base         The base configuration dict.
    @param logger       If given, a logger object to log progress.  [default: None]
    """
    if 'grid_spacing' in config:
        grid_spacing = galsim.config.ParseValue(config, 'grid_spacing', base, float)[0]
    elif 'tile_xsize' in base:
        # Then we have a tiled image.  Can use the tile spacing as the grid spacing.
        stamp_size = min(base['tile_xsize'], base['tile_ysize'])
        # Note: we use the (max) pixel scale at the image center.  This isn't
        # necessarily optimal, but it seems like the best choice.
        scale = base['wcs'].maxLinearScale(base['image_center'])
        grid_spacing = stamp_size * scale
    else:
        raise AttributeError("power_spectrum.grid_spacing required for non-tiled images")

    if 'tile_xsize' in base and base['tile_xsize'] == base['tile_ysize']:
        # PowerSpectrum can only do a square FFT, so make it the larger of the two n's.
        ngrid = max(base['nx_tiles'], base['ny_tiles'])
        # Normally that's good, but if tiles aren't square, need to drop through to the
        # second option.
    else:
        import math
        image_size = max(base['image_xsize'], base['image_ysize'])
        scale = base['wcs'].maxLinearScale(base['image_center'])
        ngrid = int(math.ceil(image_size * scale / grid_spacing))

    if 'interpolant' in config:
        interpolant = galsim.config.ParseValue(config, 'interpolant', base, str)[0]
    else:
        interpolant = None

    # We don't care about the output here.  This just builds the grid, which we'll
    # access for each object using its position.
    if base['wcs'].isCelestial():
        world_center = galsim.PositionD(0,0)
    else:
        world_center = base['wcs'].toWorld(base['image_center'])
    ps.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid, center=world_center,
                 rng=base['rng'], interpolant=interpolant)

    # Make sure this process gives consistent results regardless of the number of processes
    # being used.
    if not isinstance(ps, galsim.PowerSpectrum):
        # Then ps is really a proxy, which means the rng was pickled, so we need to
        # discard the same number of random calls from the one in the config dict.
        base['rng'].discard(ps.nRandCallsForBuildGrid())

