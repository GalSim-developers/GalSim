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

def GetLoggerProxy(logger):
    """Make a proxy for the given logger that can be passed into multiprocessing Processes
    and used safely.
    """
    from multiprocessing.managers import BaseManager
    if logger:
        class LoggerManager(BaseManager): pass
        logger_generator = galsim.utilities.SimpleGenerator(logger)
        LoggerManager.register('logger', callable = logger_generator)
        logger_manager = LoggerManager()
        logger_manager.start()
        logger_proxy = logger_manager.logger()
    else:
        logger_proxy = None
    return logger_proxy


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

    # nproc_image will be passed to the build function to be acted on at that level.
    nproc_image = 1
    if nproc > nfiles:
        if nfiles == 1:
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
                (kwargs, file_num) = job
                RemoveCurrent(config, keep_safe=True)
                galsim.config.ProcessInput(config, file_num=file_num, logger=logger)
                galsim.config.SetupExtraOutput(config, file_num=file_num, logger=logger)
                kwargs['config'] = config
                kwargs['logger'] = logger
                result = galsim.config.BuildFile(**kwargs)
                output.put( (result, file_num, proc) )
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: Caught exception %s\n%s',proc,str(e),tr)
                output.put( (e, file_num, tr) )
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('%s: Received STOP',proc)

    # Set up the multi-process task_queue if we're going to need it.
    if nproc > 1:
        # NB: See the function BuildStamps for more verbose comments about how
        # the multiprocessing stuff works.
        from multiprocessing import Process, Queue, current_process
        task_queue = Queue()

    # Now start working on the files.
    image_num = 0
    obj_num = 0
    config['file_num'] = 0
    config['image_num'] = 0
    config['obj_num'] = 0

    # Process the input field for the first file.  Often there are "safe" input items
    # that won't need to be reprocessed each time.  So do them here once and keep them
    # in the config for all file_nums.  This is more important if nproc != 1.
    galsim.config.ProcessInput(config, file_num=0, logger=logger, safe_only=True)

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
        galsim.config.ProcessInput(config, file_num=file_num, logger=logger, file_scope_only=True)

        # Assign some of the kwargs we know now:
        kwargs = {
            'nproc' : nproc_image,
            'file_num' : file_num,
            'image_num' : image_num,
            'obj_num' : obj_num
        }

        output = config['output']
        # This also updates nimages or nobjects as needed if they are being automatically
        # set from an input catalog.
        nobj = galsim.config.GetNObjForFile(config,file_num,image_num)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: nobj = %s',file_num,str(nobj))

        # nobj is a list of nobj for each image in that file.
        # So len(nobj) = nimages and sum(nobj) is the total number of objects
        # This gets the values of image_num and obj_num ready for the next loop.
        image_num += len(nobj)
        obj_num += sum(nobj)

        # This is where we actually build the file.
        # If we're doing multiprocessing, we send this information off to the task_queue.
        # Otherwise, we just call BuildFile
        if nproc > 1:
            import copy
            # Make new copies of kwargs so we can update them without
            # clobbering the versions for other tasks on the queue.
            kwargs1 = copy.copy(kwargs)
            task_queue.put( (kwargs1, file_num) )
        else:
            try:
                config1 = galsim.config.CopyConfig(orig_config)
                RemoveCurrent(config1, keep_safe=True)
                galsim.config.ProcessInput(config1, file_num=file_num, logger=logger)
                galsim.config.SetupExtraOutput(config1, file_num=file_num, logger=logger)
                kwargs['config'] = config1
                kwargs['logger'] = logger 
                file_name, t = galsim.config.BuildFile(**kwargs)
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('File %d = %s: time = %f sec', file_num, file_name, t)
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger:
                    logger.error('Exception caught for file %d', file_num)
                    logger.error('%s',tr)
                    logger.error('%s',e)
                    try:
                        # If possible say which file it was.
                        default_ext = galsim.config.valid_output_types[output_type][3]
                        file_name = galsim.config.GetFilename(output, config, default_ext)
                        logger.error('File %s not written! Continuing on...',file_name)
                    except:
                        pass

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
        logger_proxy = GetLoggerProxy(logger)
        for j in range(nproc):
            p = Process(target=worker, args=(task_queue, done_queue, config1, logger_proxy),
                        name='Process-%d'%(j+1))
            p.start()
            p_list.append(p)

        # Log the results.
        nfiles_written = 0  # Don't count skipped files.
        for k in range(nfiles):
            result, file_num, proc = done_queue.get()
            if isinstance(result,Exception):
                # result is really the exception, e
                # proc is really the traceback
                if logger:
                    logger.error('Exception caught for file %d', file_num)
                    logger.error('%s',proc)
                    logger.error('%s',result)
                    try:
                        # If possible say which file it was.
                        default_ext = galsim.config.valid_output_types[output_type][3]
                        file_name = galsim.config.GetFilename(output, config, default_ext)
                        logger.error('File %s not written! Continuing on...',file_name)
                    except:
                        pass
            else:
                file_name, t = result
                if t != 0 and logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: File %d = %s: time = %f sec', proc, file_num, file_name, t)
                    nfiles_written += 1

        # Stop the processes
        for j in range(nproc):
            task_queue.put('STOP')
        for j in range(nproc):
            p_list[j].join()
        task_queue.close()
        t2 = time.time()
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn('Total time for %d files with %d processes = %f sec', 
                        nfiles_written,nproc,t2-t1)

    if logger and logger.isEnabledFor(logging.WARN):
        logger.warn('Done building files')

