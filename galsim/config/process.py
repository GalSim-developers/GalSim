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
    # - the class name to build.
    # - a list of keys to ignore on the initial creation (e.g. PowerSpectrum has values that are 
    #   used later in PowerSpectrumInit).
    # - whether the class has a getNObjects method, in which case it also must have a constructor
    #   kwarg _nobjects_only to efficiently do only enough to calculate nobjects.
    # - whether the class might be relevant at the file- or image-scope level, rather than just
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

valid_output_types = { 
    # The values are tuples with:
    # - the build function to call
    # - a function that merely counts the number of objects that will be built by the function
    # - whether the Builder takes nproc.
    # - whether the Builder takes psf_file_name, weight_file_name, and badpix_file_name.
    # - whether the Builder takes psf_hdu, weight_hdu, and badpix_hdu.
    # See the des module for examples of how to extend this from a module.
    'Fits' : ('BuildFits', 'GetNObjForFits', False, True, True),
    'MultiFits' : ('BuildMultiFits', 'GetNObjForMultiFits', True, True, False),
    'DataCube' : ('BuildDataCube', 'GetNObjForDataCube', True, True, False),
}


def RemoveCurrent(config, keep_safe=False, type=None):
    """
    Remove any "current values" stored in the config dict at any level.

    @param config       The config dict to process.
    @param keep_safe    Should current values that are marked as safe be preserved? 
                        [default: False]
    @param type         If provided, only clear the current value of objects that use this 
                        particular type.  [default: None, which means to remove current values
                        of all types.]
    """
    # End recursion if this is not a dict.
    if not isinstance(config,dict): return

    # Recurse to lower levels, if any
    force = False  # If lower levels removed anything, then force removal at this level as well.
    for key in config:
        if isinstance(config[key],list):
            for item in config[key]:
                force = RemoveCurrent(item, keep_safe, type) or force
        else:
            force = RemoveCurrent(config[key], keep_safe, type) or force
    if force: 
        keep_safe = False
        type = None

    # Delete the current_val at this level, if any
    if ( 'current_val' in config 
          and not (keep_safe and config['current_safe'])
          and (type == None or ('type' in config and config['type'] == type)) ):
        del config['current_val']
        del config['current_safe']
        if 'current_obj_num' in config:
            del config['current_obj_num']
            del config['current_image_num']
            del config['current_file_num']
            del config['current_value_type']
        return True
    else:
        return force


class InputGetter:
    """A simple class that is returns a given config[key][i] when called with obj()
    """
    def __init__(self, config, key, i):
        self.config = config
        self.key = key
        self.i = i
    def __call__(self): return self.config[self.key][self.i]

def CopyConfig(config):
    """
    If you want to use a config dict for multiprocessing, you need to deep copy
    the gal, psf, and pix fields, since they cache values that are not picklable.
    If you don't do the deep copy, then python balks when trying to send the updated
    config dict back to the root process.  We do this a few different times, so encapsulate
    the copy semantics once here.
    """
    import copy
    config1 = copy.copy(config)
  
    # Make sure the input_manager isn't in the copy
    if 'input_manager' in config1:
        del config1['input_manager']

    # Now deepcopy all the regular config fields to make sure things like current_val don't
    # get clobbered by two processes writing to the same dict.
    if 'gal' in config:
        config1['gal'] = copy.deepcopy(config['gal'])
    if 'psf' in config:
        config1['psf'] = copy.deepcopy(config['psf'])
    if 'pix' in config:
        config1['pix'] = copy.deepcopy(config['pix'])
    if 'image' in config:
        config1['image'] = copy.deepcopy(config['image'])
    if 'input' in config:
        config1['input'] = copy.deepcopy(config['input'])
    if 'output' in config:
        config1['output'] = copy.deepcopy(config['output'])
    if 'eval_variables' in config:
        config1['eval_variables'] = copy.deepcopy(config['eval_variables'])

    return config1


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
        input = config['input']
        if not isinstance(input, dict):
            raise AttributeError("config.input is not a dict.")

        # We'll iterate through this list of keys a few times
        all_keys = [ k for k in valid_input_types.keys() if k in input ]

        # First, make sure all the input fields are lists.  If not, then we make them a 
        # list with one element.
        for key in all_keys:
            if not isinstance(input[key], list): input[key] = [ input[key] ]
 
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
                fields = input[key]

                # Register this object with the manager
                for i in range(len(fields)):
                    field = fields[i]
                    tag = key + str(i)
                    # This next bit mimics the operation of BuildSimple, except that we don't
                    # actually build the object here.  Just register the class name.
                    type = valid_input_types[key][0]
                    if type in galsim.__dict__:
                        init_func = eval("galsim."+type)
                    else:
                        init_func = eval(type)
                    InputManager.register(tag, init_func)
            # Start up the input_manager
            config['input_manager'] = InputManager()
            config['input_manager'].start()

        # Read all input fields provided and create the corresponding object
        # with the parameters given in the config file.
        for key in all_keys:
            # Skip this key if not relevant for file_scope_only run.
            if file_scope_only and not valid_input_types[key][3]: continue

            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Process input key %s',file_num,key)
            fields = input[key]

            if key not in config:
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file %d: %s not currently in config',file_num,key)
                config[key] = [ None for i in range(len(fields)) ]
                config[key+'_safe'] = [ None for i in range(len(fields)) ]
            for i in range(len(fields)):
                field = fields[i]
                ck = config[key]
                ck_safe = config[key+'_safe']
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file %d: Current values for %s are %s, safe = %s',
                                 file_num, key, str(ck[i]), ck_safe[i])
                type, ignore = valid_input_types[key][0:2]
                field['type'] = type
                if ck[i] is not None and ck_safe[i]:
                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug('file %d: Using %s already read in',file_num,key)
                else:
                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug('file %d: Build input type %s',file_num,type)
                    # This is almost identical to the operation of BuildSimple.  However,
                    # rather than call the regular function here, we have input_manager do so.
                    if type in galsim.__dict__:
                        init_func = eval("galsim."+type)
                    else:
                        init_func = eval(type)
                    kwargs, safe = galsim.config.GetAllParams(field, key, config,
                                                              req = init_func._req_params,
                                                              opt = init_func._opt_params,
                                                              single = init_func._single_params,
                                                              ignore = ignore)
                    if init_func._takes_rng:
                        if 'rng' not in config:
                            raise ValueError("No config['rng'] available for %s.type = %s"%(
                                             key,type))
                        kwargs['rng'] = config['rng']
                        safe = False

                    if safe_only and not safe:
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug('file %d: Skip %s %d, since not safe',file_num,key,i)
                        ck[i] = None
                        ck_safe[i] = None
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
                    ck[i] = input_obj
                    ck_safe[i] = safe
                    # Invalidate any currently cached values that use this kind of input object:
                    # TODO: This isn't quite correct if there are multiple versions of this input
                    #       item.  e.g. you might want to invalidate dict0, but not dict1.
                    for value_type in valid_input_types[key][5]:
                        RemoveCurrent(config, type=value_type)
                        if logger and logger.isEnabledFor(logging.DEBUG):
                            logger.debug('file %d: Cleared current_vals for items with type %s',
                                         file_num,value_type)

        # Check that there are no other attributes specified.
        valid_keys = valid_input_types.keys()
        galsim.config.CheckAllParams(input, 'input', ignore=valid_keys)


def ProcessInputNObjects(config, logger=None):
    """Process the input field, just enough to determine the number of objects.
    """
    if 'input' in config:
        config['index_key'] = 'file_num'
        input = config['input']
        if not isinstance(input, dict):
            raise AttributeError("config.input is not a dict.")

        for key in valid_input_types:
            has_nobjects = valid_input_types[key][2]
            if key in input and has_nobjects:
                field = input[key]

                if key in config and config[key+'_safe'][0]:
                    input_obj = config[key][0]
                else:
                    # If it's a list, just use the first one.
                    if isinstance(field, list): field = field[0]

                    type, ignore = valid_input_types[key][0:2]
                    if type in galsim.__dict__:
                        init_func = eval("galsim."+type)
                    else:
                        init_func = eval(type)
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

def UpdateNProc(nproc, logger=None):
    """Update nproc to ncpu if nproc <= 0
    """
    if nproc <= 0:
        # Try to figure out a good number of processes to use
        try:
            from multiprocessing import cpu_count
            nproc = cpu_count()
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("ncpu = %d.",nproc)
        except:
            if logger and logger.isEnabledFor(logging.WARN):
                logger.warn("nproc <= 0, but unable to determine number of cpus.")
                logger.warn("Using single process")
            nproc = 1
    return nproc
 

def Process(config, logger=None):
    """
    Do all processing of the provided configuration dict.  In particular, this
    function handles processing the output field, calling other functions to
    build and write the specified files.  The input field is processed before
    building each file.
    """
    # First thing to do is deep copy the input config to make sure we don't modify the original.
    import copy
    config = copy.deepcopy(config)

    # If we don't have a root specified yet, we generate it from the current script.
    if 'root' not in config:
        import inspect
        script_name = os.path.basename(
            inspect.getfile(inspect.currentframe())) # script filename (usually with path)
        # Strip off a final suffix if present.
        config['root'] = os.path.splitext(script_name)[0]

    # Make config['output'] exist if it doesn't yet.
    if 'output' not in config:
        config['output'] = {}
    output = config['output']
    if not isinstance(output, dict):
        raise AttributeError("config.output is not a dict.")

    # Get the output type.  Default = Fits
    if 'type' not in output:
        output['type'] = 'Fits' 
    type = output['type']

    # Check that the type is valid
    if type not in valid_output_types:
        raise AttributeError("Invalid output.type=%s."%type)

    # build_func is the function we'll call to build each file.
    build_func = eval(valid_output_types[type][0])

    # nobj_func is the function that builds the nobj_per_file list
    nobj_func = eval(valid_output_types[type][1])

    # can_do_multiple says whether the function can in principal do multiple images.
    can_do_multiple = valid_output_types[type][2]

    # extra_file_name says whether the function takes psf_file_name, etc.
    extra_file_name = valid_output_types[type][3]

    # extra_hdu says whether the function takes psf_hdu, etc.
    extra_hdu = valid_output_types[type][4]
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('type = %s',type)
        logger.debug('extra_file_name = %s',extra_file_name)
        logger.debug('extra_hdu = %d',extra_hdu)

    # We need to know how many objects we'll need for each file (and each image within each file)
    # to get the indexing correct for any sequence items.  (e.g. random_seed)
    # If we use multiple processors and let the regular sequencing happen, 
    # it will get screwed up by the multi-processing potentially happening out of order.
    # Start with the number of files.
    if 'nfiles' in output:
        nfiles = galsim.config.ParseValue(output, 'nfiles', config, int)[0]
    else:
        nfiles = 1 
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('nfiles = %d',nfiles)

    # Figure out how many processes we will use for building the files.
    # (If nfiles = 1, but nimages > 1, we'll do the multi-processing at the image stage.)
    if 'nproc' in output:
        nproc = galsim.config.ParseValue(output, 'nproc', config, int)[0]
        nproc = UpdateNProc(nproc,logger)
    else:
        nproc = 1 

    # If set, nproc_image will be passed to the build function to be acted on at that level.
    nproc_image = None
    if nproc > nfiles:
        if nfiles == 1 and can_do_multiple:
            nproc_image = nproc 
            nproc = 1
        else:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("There are only %d files.  Reducing nproc to %d."%(nfiles,nfiles))
            nproc = nfiles

    def worker(input, output, config, logger):
        proc = current_process().name
        for job in iter(input.get, 'STOP'):
            try:
                (kwargs, file_num, file_name) = job
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: Start file %d, %s',proc,file_num,file_name)
                ProcessInput(config, file_num=file_num, logger=logger)
                kwargs['config'] = config
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('%s: After ProcessInput for file %d',proc,file_num)
                kwargs['logger'] = logger
                t = build_func(**kwargs)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('%s: After %s for file %d',proc,build_func,file_num)
                output.put( (t, file_num, file_name, proc) )
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: Caught exception %s\n%s',proc,str(e),tr)
                output.put( (e, file_num, file_name, tr) )
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('%s: Received STOP',proc)

    # Set up the multi-process task_queue if we're going to need it.
    if nproc > 1:
        # NB: See the function BuildStamps for more verbose comments about how
        # the multiprocessing stuff works.
        from multiprocessing import Process, Queue, current_process
        task_queue = Queue()

    # The logger is not picklable, so we use the same trick for it as we used for the 
    # input fields in CopyConfig to allow the worker processes to log their progress.
    # The real logger stays in this process, and the workers all get a proxy logger which 
    # they can use normally.  We use galsim.utilities.SimpleGenerator as the callable that
    # just returns the existing logger object.
    from multiprocessing.managers import BaseManager
    class LoggerManager(BaseManager): pass
    if logger:
        logger_generator = galsim.utilities.SimpleGenerator(logger)
        LoggerManager.register('logger', callable = logger_generator)
        logger_manager = LoggerManager()
        logger_manager.start()
        logger_proxy = logger_manager.logger()
    else:
        logger_proxy = None

    # Now start working on the files.
    image_num = 0
    obj_num = 0
    config['file_num'] = 0
    config['image_num'] = 0
    config['obj_num'] = 0

    extra_keys = [ 'psf', 'weight', 'badpix' ]
    last_file_name = {}
    for key in extra_keys:
        last_file_name[key] = None

    # Process the input field for the first file.  Often there are "safe" input items
    # that won't need to be reprocessed each time.  So do them here once and keep them
    # in the config for all file_nums.  This is more important if nproc != 1.
    ProcessInput(config, file_num=0, logger=logger_proxy, safe_only=True)

    # Normally, random_seed is just a number, which really means to use that number
    # for the first item and go up sequentially from there for each object.
    # However, we allow for random_seed to be a gettable parameter, so for the 
    # normal case, we just convert it into a Sequence.
    if ( 'image' in config 
         and 'random_seed' in config['image'] 
         and not isinstance(config['image']['random_seed'],dict) ):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = {
            'type' : 'Sequence' ,
            'index_key' : 'obj_num',
            'first' : first
        }

    nfiles_use = nfiles
    # We'll want a pristine version later to give to the workers.
    orig_config = CopyConfig(config)
    for file_num in range(nfiles):
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file_num, image_num, obj_num = %d,%d,%d',file_num,image_num,obj_num)
        # Set the index for any sequences in the input or output parameters.
        # These sequences are indexed by the file_num.
        # (In image, they are indexed by image_num, and after that by obj_num.)
        config['index_key'] = 'file_num'
        config['file_num'] = file_num
        config['image_num'] = image_num
        config['start_obj_num'] = obj_num
        config['obj_num'] = obj_num

        # Process the input fields that might be relevant at file scope:
        ProcessInput(config, file_num=file_num, logger=logger_proxy, file_scope_only=True)

        # It is possible that some items at image scope could need a random number generator.
        # For example, in demo9, we have a random number of objects per image.
        # So we need to build an rng here.
        if 'random_seed' in config['image']:
            config['index_key'] = 'obj_num'
            seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
            config['index_key'] = 'file_num'
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: seed = %d',file_num,seed)
            rng = galsim.BaseDeviate(seed)
        else:
            rng = galsim.BaseDeviate()
        config['rng'] = rng

        # Get the file_name
        if 'file_name' in output:
            SetDefaultExt(output['file_name'],'.fits')
            file_name = galsim.config.ParseValue(output, 'file_name', config, str)[0]
        elif 'root' in config:
            # If a file_name isn't specified, we use the name of the config file + '.fits'
            file_name = config['root'] + '.fits'
        else:
            raise AttributeError(
                "No output.file_name specified and unable to generate it automatically.")
        
        # Prepend a dir to the beginning of the filename if requested.
        if 'dir' in output:
            dir = galsim.config.ParseValue(output, 'dir', config, str)[0]
            if dir and not os.path.isdir(dir): os.makedirs(dir)
            file_name = os.path.join(dir,file_name)
        else:
            dir = None

        # Assign some of the kwargs we know now:
        kwargs = {
            'file_name' : file_name,
            'file_num' : file_num,
            'image_num' : image_num,
            'obj_num' : obj_num
        }
        if nproc_image:
            kwargs['nproc'] = nproc_image

        output = config['output']
        # This also updates nimages or nobjects as needed if they are being automatically
        # set from an input catalog.
        nobj = nobj_func(config,file_num,image_num)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: nobj = %s',file_num,str(nobj))

        # nobj is a list of nobj for each image in that file.
        # So len(nobj) = nimages and sum(nobj) is the total number of objects
        # This gets the values of image_num and obj_num ready for the next loop.
        image_num += len(nobj)
        obj_num += sum(nobj)

        # Check if we ought to skip this file
        if ('skip' in output 
                and galsim.config.ParseValue(output, 'skip', config, bool)[0]):
            if logger and logger.isEnabledFor(logging.WARN):
                logger.warn('Skipping file %d = %s because output.skip = True',file_num,file_name)
            nfiles_use -= 1
            continue
        if ('noclobber' in output 
                and galsim.config.ParseValue(output, 'noclobber', config, bool)[0]
                and os.path.isfile(file_name)):
            if logger and logger.isEnabledFor(logging.WARN):
                logger.warn('Skipping file %d = %s because output.noclobber = True' +
                            ' and file exists',file_num,file_name)
            nfiles_use -= 1
            continue

        # Check if we need to build extra images to write out as well
        for extra_key in [ key for key in extra_keys if key in output ]:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('extra_key = %s',extra_key)
            output_extra = output[extra_key]

            output_extra['type'] = 'default'
            req = {}
            single = []
            opt = {}
            ignore = []
            if extra_file_name and extra_hdu:
                single += [ { 'file_name' : str, 'hdu' : int } ]
                opt['dir'] = str
            elif extra_file_name:
                req['file_name'] = str
                opt['dir'] = str
            elif extra_hdu:
                req['hdu'] = int

            if extra_key == 'psf': 
                ignore += ['draw_method', 'signal_to_noise']
            if extra_key == 'weight': 
                ignore += ['include_obj_var']
            if 'file_name' in output_extra:
                SetDefaultExt(output_extra['file_name'],'.fits')
            params, safe = galsim.config.GetAllParams(output_extra,extra_key,config,
                                                      req=req, opt=opt, single=single,
                                                      ignore=ignore)

            if 'file_name' in params:
                f = params['file_name']
                if 'dir' in params:
                    dir = params['dir']
                    if dir and not os.path.isdir(dir): os.makedirs(dir)
                # else keep dir from above.
                if dir:
                    f = os.path.join(dir,f)
                # If we already wrote this file, skip it this time around.
                # (Typically this is applicable for psf, where we may only want 1 psf file.)
                if last_file_name[key] == f:
                    if logger and logger.isEnabledFor(logging.WARN):
                        logger.warn('skipping %s, since already written',f)
                    continue
                kwargs[ extra_key+'_file_name' ] = f
                last_file_name[key] = f
            elif 'hdu' in params:
                kwargs[ extra_key+'_hdu' ] = params['hdu']

        # This is where we actually build the file.
        # If we're doing multiprocessing, we send this information off to the task_queue.
        # Otherwise, we just call build_func.
        if nproc > 1:
            import copy
            # Make new copies of kwargs so we can update them without
            # clobbering the versions for other tasks on the queue.
            kwargs1 = copy.copy(kwargs)
            task_queue.put( (kwargs1, file_num, file_name) )
        else:
            try:
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('Start file %d = %s', file_num, file_name)
                ProcessInput(config, file_num=file_num, logger=logger_proxy)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file %d: After ProcessInput',file_num)
                kwargs['config'] = config
                kwargs['logger'] = logger 
                t = build_func(**kwargs)
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('File %d = %s, time = %f sec', file_num, file_name, t)
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger:
                    logger.error('Exception caught for file %d = %s', file_num, file_name)
                    logger.error('%s',tr)
                    logger.error('%s',e)
                    logger.error('File %s not written! Continuing on...',file_name)

    # If we're doing multiprocessing, here is the machinery to run through the task_queue
    # and process the results.
    if nproc > 1:
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn("Using %d processes for file processing",nproc)
        import time
        t1 = time.time()
        # Run the tasks
        done_queue = Queue()
        p_list = []
        config1 = galsim.config.CopyConfig(orig_config)
        config1['current_nproc'] = nproc
        for j in range(nproc):
            p = Process(target=worker, args=(task_queue, done_queue, config1, logger_proxy),
                        name='Process-%d'%(j+1))
            p.start()
            p_list.append(p)

        # Log the results.
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('nfiles_use = %d',nfiles_use)
        for k in range(nfiles_use):
            t, file_num, file_name, proc = done_queue.get()
            if isinstance(t,Exception):
                # t is really the exception, e
                # proc is really the traceback
                if logger:
                    logger.error('Exception caught for file %d = %s', file_num, file_name)
                    logger.error('%s',proc)
                    logger.error('%s',t)
                    logger.error('File %s not written! Continuing on...',file_name)
            else:
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: File %d = %s: time = %f sec', proc, file_num, file_name, t)

        # Stop the processes
        for j in range(nproc):
            task_queue.put('STOP')
        for j in range(nproc):
            p_list[j].join()
        task_queue.close()
        t2 = time.time()
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn('Total time for %d files with %d processes = %f sec', 
                        nfiles_use,nproc,t2-t1)

    if logger and logger.isEnabledFor(logging.WARN):
        logger.warn('Done building files')


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

def BuildFits(file_name, config, logger=None, 
              file_num=0, image_num=0, obj_num=0,
              psf_file_name=None, psf_hdu=None,
              weight_file_name=None, weight_hdu=None,
              badpix_file_name=None, badpix_hdu=None):
    """
    Build a regular fits file as specified in config.
    
    @param file_name        The name of the output file.
    @param config           A configuration dict.
    @param logger           If given, a logger object to log progress. [default: None]
    @param file_num         If given, the current file_num. [default: 0]
    @param image_num        If given, the current image_num. [default: 0]
    @param obj_num          If given, the current obj_num. [default: 0]
    @param psf_file_name    If given, write a psf image to this file. [default: None]
    @param psf_hdu          If given, write a psf image to this hdu in file_name. [default: None]
    @param weight_file_name If given, write a weight image to this file. [default: None]
    @param weight_hdu       If given, write a weight image to this hdu in file_name.  [default: 
                            None]
    @param badpix_file_name If given, write a badpix image to this file. [default: None]
    @param badpix_hdu       If given, write a badpix image to this hdu in file_name. [default:
                            None]

    @returns the time taken to build file.
    """
    import time
    t1 = time.time()

    config['index_key'] = 'file_num'
    config['file_num'] = file_num
    config['image_num'] = image_num
    config['start_obj_num'] = obj_num
    config['obj_num'] = obj_num
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildFits for %s: file, image, obj = %d,%d,%d',
                      config['file_num'],file_name,file_num,image_num,obj_num)

    if ( 'image' in config 
         and 'random_seed' in config['image'] 
         and not isinstance(config['image']['random_seed'],dict) ):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = { 
                'type' : 'Sequence',
                'index_key' : 'obj_num',
                'first' : first 
        }

    if 'random_seed' in config['image']:
        config['index_key'] = 'obj_num'
        seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['index_key'] = 'file_num'
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: seed = %d',file_num,seed)
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate()
    config['rng'] = rng

    # hdus is a dict with hdus[i] = the item in all_images to put in the i-th hdu.
    hdus = {}
    # The primary hdu is always the main image.
    hdus[0] = 0

    if psf_file_name or psf_hdu:
        make_psf_image = True
        if psf_hdu: 
            if psf_hdu <= 0 or psf_hdu in hdus.keys():
                raise ValueError("psf_hdu = %d is invalid or a duplicate."%pdf_hdu)
            hdus[psf_hdu] = 1
    else:
        make_psf_image = False

    if weight_file_name or weight_hdu:
        make_weight_image = True
        if weight_hdu: 
            if weight_hdu <= 0 or weight_hdu in hdus.keys():
                raise ValueError("weight_hdu = %d is invalid or a duplicate."&weight_hdu)
            hdus[weight_hdu] = 2
    else:
        make_weight_image = False

    if badpix_file_name or badpix_hdu:
        make_badpix_image = True
        if badpix_hdu: 
            if badpix_hdu <= 0 or badpix_hdu in hdus.keys():
                raise ValueError("badpix_hdu = %d is invalid or a duplicate."&badpix_hdu)
            hdus[badpix_hdu] = 3
    else:
        make_badpix_image = False

    for h in range(len(hdus.keys())):
        if h not in hdus.keys():
            raise ValueError("Image for hdu %d not found.  Cannot skip hdus."%h)

    all_images = galsim.config.BuildImage(
            config=config, logger=logger, image_num=image_num, obj_num=obj_num,
            make_psf_image=make_psf_image,
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)
    # returns a tuple ( main_image, psf_image, weight_image, badpix_image )

    hdulist = []
    for h in range(len(hdus.keys())):
        assert h in hdus.keys()  # Checked for this above.
        hdulist.append(all_images[hdus[h]])
    # We can use hdulist in writeMulti even if the main image is the only one in the list.

    if 'output' in config and 'retry_io' in config['output']:
        ntries = galsim.config.ParseValue(config['output'],'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    _retry_io(galsim.fits.writeMulti, (hdulist, file_name), ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        if len(hdus.keys()) == 1:
            logger.debug('file %d: Wrote image to fits file %r',
                         config['file_num'],file_name)
        else:
            logger.debug('file %d: Wrote image (with extra hdus) to multi-extension fits file %r',
                         config['file_num'],file_name)

    if psf_file_name:
        _retry_io(galsim.fits.write, (all_images[1], psf_file_name),
                  ntries, psf_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote psf image to fits file %r',
                         config['file_num'],psf_file_name)

    if weight_file_name:
        _retry_io(galsim.fits.write, (all_images[2], weight_file_name),
                  ntries, weight_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote weight image to fits file %r',
                         config['file_num'],weight_file_name)

    if badpix_file_name:
        _retry_io(galsim.fits.write, (all_images[3], badpix_file_name),
                  ntries, badpix_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote badpix image to fits file %r',
                         config['file_num'],badpix_file_name)

    t2 = time.time()
    return t2-t1


def BuildMultiFits(file_name, config, nproc=1, logger=None,
                   file_num=0, image_num=0, obj_num=0,
                   psf_file_name=None, weight_file_name=None, badpix_file_name=None):
    """
    Build a multi-extension fits file as specified in config.
    
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
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildMultiFits for %s: file, image, obj = %d,%d,%d',
                      config['file_num'],file_name,file_num,image_num,obj_num)

    if ( 'image' in config 
         and 'random_seed' in config['image'] 
         and not isinstance(config['image']['random_seed'],dict) ):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = { 
                'type' : 'Sequence',
                'index_key' : 'obj_num',
                'first' : first 
        }

    if 'random_seed' in config['image']:
        config['index_key'] = 'obj_num'
        seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['index_key'] = 'file_num'
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: seed = %d',file_num,seed)
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate()
    config['rng'] = rng

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
        raise AttributeError("Attribute output.nimages is required for output.type = MultiFits")
    nimages = galsim.config.ParseValue(config['output'],'nimages',config,int)[0]

    if nproc > nimages:
        # Only warn if nproc was specifically set, not if it is -1.
        if logger and logger.isEnabledFor(logging.WARN):
            if not ('nproc' in config['output'] and 
                 galsim.config.ParseValue(config['output'],'nproc',config,int)[0] == -1):
                logger.warn(
                    "Trying to use more processes than images: output.nproc=%d, "%nproc +
                    "nimages=%d.  Reducing nproc to %d."%(nimages,nimages))
        nproc = nimages

    all_images = galsim.config.BuildImages(
        nimages, config=config, nproc=nproc, logger=logger,
        image_num=image_num, obj_num=obj_num,
        make_psf_image=make_psf_image, 
        make_weight_image=make_weight_image,
        make_badpix_image=make_badpix_image)

    main_images = all_images[0]
    psf_images = all_images[1]
    weight_images = all_images[2]
    badpix_images = all_images[3]

    if 'output' in config and 'retry_io' in config['output']:
        ntries = galsim.config.ParseValue(config['output'],'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    _retry_io(galsim.fits.writeMulti, (main_images, file_name), ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Wrote images to multi-extension fits file %r',
                     config['file_num'],file_name)

    if psf_file_name:
        _retry_io(galsim.fits.writeMulti, (psf_images, psf_file_name),
                  ntries, psf_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote psf images to multi-extension fits file %r',
                         config['file_num'],psf_file_name)

    if weight_file_name:
        _retry_io(galsim.fits.writeMulti, (weight_images, weight_file_name),
                  ntries, weight_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote weight images to multi-extension fits file %r',
                         config['file_num'],weight_file_name)

    if badpix_file_name:
        _retry_io(galsim.fits.writeMulti, (all_images, badpix_file_name),
                  ntries, badpix_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote badpix images to multi-extension fits file %r',
                         config['file_num'],badpix_file_name)


    t2 = time.time()
    return t2-t1


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
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildDataCube for %s: file, image, obj = %d,%d,%d',
                      config['file_num'],file_name,file_num,image_num,obj_num)

    if ( 'image' in config 
         and 'random_seed' in config['image'] 
         and not isinstance(config['image']['random_seed'],dict) ):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = { 
                'type' : 'Sequence',
                'index_key' : 'obj_num',
                'first' : first 
        }

    if 'random_seed' in config['image']:
        config['index_key'] = 'obj_num'
        seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['index_key'] = 'file_num'
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: seed = %d',file_num,seed)
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate()
    config['rng'] = rng

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
    config1 = CopyConfig(config)
    all_images = galsim.config.BuildImage(
            config=config1, logger=logger, image_num=image_num, obj_num=obj_num,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)
    obj_num += galsim.config.GetNObjForImage(config, image_num)
    t3 = time.time()
    if logger and logger.isEnabledFor(logging.INFO):
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
            if logger and logger.isEnabledFor(logging.WARN):
                if not ('nproc' in config['output'] and
                     galsim.config.ParseValue(config['output'],'nproc',config,int)[0] == -1):
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

    _retry_io(galsim.fits.writeCube, (main_images, file_name), ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Wrote image to fits data cube %r',
                     config['file_num'],file_name)

    if psf_file_name:
        _retry_io(galsim.fits.writeCube, (psf_images, psf_file_name),
                  ntries, psf_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote psf images to fits data cube %r',
                         config['file_num'],psf_file_name)

    if weight_file_name:
        _retry_io(galsim.fits.writeCube, (weight_images, weight_file_name),
                  ntries, weight_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote weight images to fits data cube %r',
                         config['file_num'],weight_file_name)

    if badpix_file_name:
        _retry_io(galsim.fits.writeCube, (badpix_images, badpix_file_name),
                  ntries, badpix_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote badpix images to fits data cube %r',
                         config['file_num'],badpix_file_name)

    t4 = time.time()
    return t4-t1

def GetNObjForFits(config, file_num, image_num):
    ignore = [ 'file_name', 'dir', 'nfiles', 'psf', 'weight', 'badpix', 'nproc',
               'skip', 'noclobber', 'retry_io' ]
    galsim.config.CheckAllParams(config['output'], 'output', ignore=ignore)
    try : 
        nobj = [ galsim.config.GetNObjForImage(config, image_num) ]
    except ValueError : # (This may be raised if something needs the input stuff)
        ProcessInput(config, file_num=file_num)
        nobj = [ galsim.config.GetNObjForImage(config, image_num) ]
    return nobj
    
def GetNObjForMultiFits(config, file_num, image_num):
    ignore = [ 'file_name', 'dir', 'nfiles', 'psf', 'weight', 'badpix', 'nproc', 
               'skip', 'noclobber', 'retry_io' ]
    req = { 'nimages' : int }
    # Allow nimages to be automatic based on input catalog if image type is Single
    if ( 'nimages' not in config['output'] and 
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) ):
        nobjects = ProcessInputNObjects(config)
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
        ProcessInput(config, file_num=file_num)
        nobj = [ galsim.config.GetNObjForImage(config, image_num+j) for j in range(nimages) ]
    return nobj

def GetNObjForDataCube(config, file_num, image_num):
    ignore = [ 'file_name', 'dir', 'nfiles', 'psf', 'weight', 'badpix', 'nproc',
               'skip', 'noclobber', 'retry_io' ]
    req = { 'nimages' : int }
    # Allow nimages to be automatic based on input catalog if image type is Single
    if ( 'nimages' not in config['output'] and 
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) ):
        nobjects = ProcessInputNObjects(config)
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
        ProcessInput(config, file_num=file_num)
        nobj = [ galsim.config.GetNObjForImage(config, image_num+j) for j in range(nimages) ]
    return nobj
 
def SetDefaultExt(config, ext):
    """
    Some items have a default extension for a NumberedFile type.
    """
    if ( isinstance(config,dict) and 'type' in config and 
         config['type'] == 'NumberedFile' and 'ext' not in config ):
        config['ext'] = ext

