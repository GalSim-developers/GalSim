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

# First, some helper functions that will be useful at various points in the processing.

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
        del config['current_index']
        if 'current_value_type' in config:
            del config['current_value_type']
        return True
    else:
        return force

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
 

def SetDefaultExt(config, ext):
    """
    Some items have a default extension for a NumberedFile type.
    """
    if ( isinstance(config,dict) and 'type' in config and 
         config['type'] == 'NumberedFile' and 'ext' not in config ):
        config['ext'] = ext

def SetupConfigRNG(config, seed_offset=0):
    """Set up the RNG in the config dict.

    - Setup config['image']['random_seed'] if necessary
    - Set config['rng'] based on appropriate random_seed 

    @param config           A configuration dict.
    @param seed_offset      An offset to use relative to what config['image']['random_seed'] gives.

    @returns the seed used to initialize the RNG.
    """
    # Normally, random_seed is just a number, which really means to use that number
    # for the first item and go up sequentially from there for each object.
    # However, we allow for random_seed to be a gettable parameter, so for the 
    # normal case, we just convert it into a Sequence.
    if ( 'image' in config 
         and 'random_seed' in config['image'] 
         and not isinstance(config['image']['random_seed'],dict) ):
         # The "first" is actually the seed value to use for anything at file or image scope
         # using the obj_num of the first object in the file or image.  Seeds for objects
         # will start at 1 more than this.
         first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
         config['image']['random_seed'] = { 
                 'type' : 'Sequence',
                 'index_key' : 'obj_num',
                 'first' : first
         }

    if 'random_seed' in config['image']:
        orig_key = config['index_key']
        config['index_key'] = 'obj_num'
        seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['index_key'] = orig_key
        seed += seed_offset
    else:
        seed = 0

    config['seed'] = seed
    config['rng'] = galsim.BaseDeviate(seed)

    # This can be present for efficiency, since GaussianDeviates produce two values at a time, 
    # so it is more efficient to not create a new GaussianDeviate object each time.
    # But if so, we need to remove it now.
    if 'gd' in config:
        del config['gd']

    return seed
 
# This is the main script to process everything in the configuration dict.
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
    if type not in galsim.config.valid_output_types:
        raise AttributeError("Invalid output.type=%s."%type)

    # build_func is the function we'll call to build each file.
    build_func = eval(galsim.config.valid_output_types[type][0])

    # nobj_func is the function that builds the nobj_per_file list
    nobj_func = eval(galsim.config.valid_output_types[type][1])

    # can_do_multiple says whether the function can in principal do multiple images.
    can_do_multiple = galsim.config.valid_output_types[type][2]

    # extra_file_name says whether the function takes psf_file_name, etc.
    extra_file_name = galsim.config.valid_output_types[type][3]

    # extra_hdu says whether the function takes psf_hdu, etc.
    extra_hdu = galsim.config.valid_output_types[type][4]
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
                RemoveCurrent(config, keep_safe=True)
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: Start file %d, %s',proc,file_num,file_name)
                galsim.config.ProcessInput(config, file_num=file_num, logger=logger)
                galsim.config.SetupExtraOutput(config, file_num=file_num, logger=logger)
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

    extra_keys = galsim.config.valid_extra_outputs.keys()
    last_file_name = {}
    for key in extra_keys:
        last_file_name[key] = None

    # Process the input field for the first file.  Often there are "safe" input items
    # that won't need to be reprocessed each time.  So do them here once and keep them
    # in the config for all file_nums.  This is more important if nproc != 1.
    galsim.config.ProcessInput(config, file_num=0, logger=logger_proxy, safe_only=True)

    nfiles_use = nfiles
    # We'll want a pristine version later to give to the workers.
    orig_config = CopyConfig(config)
    for file_num in range(nfiles):
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file_num, image_num, obj_num = %d,%d,%d',file_num,image_num,obj_num)
        galsim.config.SetupConfigFileNum(config,file_num,image_num,obj_num)
        seed = SetupConfigRNG(config)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: seed = %d',file_num,seed)

        # Process the input fields that might be relevant at file scope:
        galsim.config.ProcessInput(config, file_num=file_num, logger=logger_proxy,
                                   file_scope_only=True)

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
        main_dir = dir
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

            ignore += galsim.config.valid_extra_outputs[extra_key][1]

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
                else:
                    dir = main_dir
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
                config1 = galsim.config.CopyConfig(orig_config)
                RemoveCurrent(config1, keep_safe=True)
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('Start file %d = %s', file_num, file_name)
                galsim.config.ProcessInput(config1, file_num=file_num, logger=logger)
                galsim.config.SetupExtraOutput(config1, file_num=file_num, logger=logger)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file %d: After ProcessInput',file_num)
                kwargs['config'] = config1
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

