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

from .input import *
from .output import *

def ReadYaml(config_file):
    """Read in a YAML configuration file and return the corresponding dicts.

    A YAML file is allowed to define several dicts using multiple documents. The GalSim parser
    treats this as a set of multiple jobs to be done.  The first document is taken to be a "base"
    dict that has common definitions for all the jobs.  Then each subsequent document has the
    (usually small) modifications to the base dict for each job.  See demo6.yaml, demo8.yaml and
    demo9.yaml in the GalSim/examples directory for example usage.

    On output, base_config is the dict for the first document if there are multiple documents.
    Then all_config is a list of all the other documents that should be joined with base_dict
    for each job to be processed.  If there is only one document defined, base_dict will be
    empty, and all_config will be a list of one dict, which is the one to use.

    @param config_file      The name of the configuration file to read.

    @returns (base_config, all_config)
    """
    import yaml

    with open(config_file) as f:
        all_config = [ c for c in yaml.load_all(f.read()) ]

    # If there is only 1 yaml document, then it is of course used for the configuration.
    # If there are multiple yaml documents, then the first one defines a common starting
    # point for the later documents.
    # So the configurations are taken to be:
    #   all_config[0] + all_config[1]
    #   all_config[0] + all_config[2]
    #   all_config[0] + all_config[3]
    #   ...
    # See demo6.yaml and demo8.yaml in the examples directory for examples of this feature.

    if len(all_config) > 1:
        # Break off the first one if more than one:
        base_config = all_config[0]
        all_config = all_config[1:]
    else:
        # Else just use an empty base_config dict.
        base_config = {}

    return base_config, all_config


def ReadJson(config_file):
    """Read in a JSON configuration file and return the corresponding dicts.

    A JSON file only defines a single dict.  However to be parallel to the functionality of
    ReadYaml, the output is base_config, all_config, where base_config is an empty dict,
    and all_config is a list with a single item, which is the dict defined by the JSON file.

    @param config_file      The name of the configuration file to read.

    @returns (base_config, all_config)
    """
    import json

    with open(config_file) as f:
        config = json.load(f)

    # JSON files are just processed as is.  This is equivalent to having an empty 
    # base_config, so we just do that and use the same structure.
    base_config = {}
    all_config = [ config ]

    return base_config, all_config

def ReadConfig(config_file, file_type=None, logger=None):
    """Read in a configuration file and return the corresponding dicts.

    A YAML file is allowed to define several dicts using multiple documents. The GalSim parser
    treats this as a set of multiple jobs to be done.  The first document is taken to be a "base"
    dict that has common definitions for all the jobs.  Then each subsequent document has the
    (usually small) modifications to the base dict for each job.  See demo6.yaml, demo8.yaml and
    demo9.yaml in the GalSim/examples directory for example usage.

    On output, base_config is the dict for the first document if there are multiple documents.
    Then all_config is a list of all the other documents that should be joined with base_dict
    for each job to be processed.  If there is only one document defined, base_dict will be
    empty, and all_config will be a list of one dict, which is the one to use.

    A JSON file does not have this feature, but to be consistent, we always return the tuple
    (base_config, all_config).

    @param config_file      The name of the configuration file to read.
    @param file_type        If given, the type of file to read.  [default: None, which mean
                            infer the file type from the extension.]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns (base_config, all_config)
    """
    # Determine the file type from the extension if necessary:
    if file_type is None:
        import os
        name, ext = os.path.splitext(config_file)
        if ext.lower().startswith('.j'):
            file_type = 'json'
        else:
            # Let YAML be the default if the extension is not .y* or .j*.
            file_type = 'yaml'
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('File type determined to be %s', file_type)
    else:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('File type specified to be %s', file_type)

    if file_type == 'yaml':
        if logger and logger.isEnabledFor(logging.INFO):
            logger.info('Reading YAML config file %s', config_file)
        return galsim.config.ReadYaml(config_file)
    else:
        if logger and logger.isEnabledFor(logging.INFO):
            logger.info('Reading JSON config file %s', config_file)
        return galsim.config.ReadJson(config_file)

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


def CopyConfig(config):
    """
    If you want to use a config dict for multiprocessing, you need to deep copy
    the gal, psf, and pix fields, since they get cache values that are not picklable.
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


def UpdateNProc(nproc, ntot, config, logger=None):
    """Update nproc
    - If nproc < 0, set nproc to ncpu
    - Make sure nproc <= ntot
    @param nproc        The nominal number of processes from the config dict
    @param ntot         The total number of files/images/stamps to do, so the maximum number of
                        processes that would make sense.
    @param config       The configuration dict to copy.
    @param logger       If given, a logger object to log progress. [default: None]
    @returns the number of processes to use.
    """
    # First if nproc < 0, update based on ncpu
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
 
    # Second, make sure we aren't already in a multiprocessing mode
    if nproc > 1 and 'current_nproc' in config:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Already multiprocessing.  Ignoring image.nproc")
        nproc = 1

    # Finally, don't try to use more processes than jobs.  It wouldn't fail or anything.
    # It just looks bad to have 3 images processed with 8 processes or something like that.
    if nproc > ntot:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug("There are only %d jobs to do.  Reducing nproc to %d."%(ntot,ntot))
        nproc = ntot
    return nproc


def SetupConfigRNG(config, seed_offset=0):
    """Set up the RNG in the config dict.
    - Setup config['image']['random_seed'] if necessary
    - Set config['rng'] based on appropriate random_seed 
    @param config           The configuration dict.
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


def ParseExtendedKey(config, key):
    """Traverse all but the last item in an extended key and return the resulting config, key.

    If key is an extended key like gal.items.0.ellip.e, then this will return the tuple.
    (config['gal']['items'][0]['ellip'], 'e').

    If key is a regular string, then is just returns the original (config, key).

    @param config       The configuration dict.
    @param key          The possibly extended key.

    @returns the equivalent (config, key) where key is now a regular non-extended key.
    """
    # This is basically identical to the code for Dict.get(key) in catalog.py.
    chain = key.split('.')
    d = config
    while len(chain) > 1:
        k = chain.pop(0)
        try: k = int(k)
        except ValueError: pass
        d = d[k]
    return d, chain[0]

def GetFromConfig(config, key):
    """Get the value for the (possibly extended) key from a config dict.

    If key is a simple string, then this is equivalent to config[key].
    However, key is allowed to be a chain of keys such as 'gal.items.0.ellip.e', in which
    case this function will return config['gal']['items'][0]['ellip']['e'].

    @param config       The configuration dict.
    @param key          The possibly extended key.

    @returns the value of that key from the config.
    """
    config, key = ParseExtendedKey(config, key)
    return config[key]

def SetInConfig(config, key, value):
    """Set the value of a (possibly extended) key in a config dict.

    If key is a simple string, then this is equivalent to config[key] = value.
    However, key is allowed to be a chain of keys such as 'gal.items.0.ellip.e', in which
    case this function will set config['gal']['items'][0]['ellip']['e'] = value.

    @param config       The configuration dict.
    @param key          The possibly extended key.

    @returns the value of that key from the config.
    """
    config, key = ParseExtendedKey(config, key)
    config[key] = value


def UpdateConfig(config, new_params):
    """Update the given config dict with additional parameters/values.

    @param config           The configuration dict to update.
    @param new_params       A dict of parameters to update.  The keys of this dict may be
                            chained field names, such as gal.first.dilate, which will be
                            parsed to update config['gal']['first']['dilate'].
    """
    for key, value in new_params.items():
        SetInConfig(config, key, value)


def Process(config, logger=None, new_params=None):
    """
    Do all processing of the provided configuration dict.  In particular, this
    function handles processing the output field, calling other functions to
    build and write the specified files.  The input field is processed before
    building each file.
    """
    # First thing to do is deep copy the input config to make sure we don't modify the original.
    import copy
    config = copy.deepcopy(config)

    # Update using any new_params that are given:
    if new_params is not None:
        UpdateConfig(config, new_params)

    # If we don't have a root specified yet, we generate it from the current script.
    if 'root' not in config:
        import inspect
        script_name = os.path.basename(
            inspect.getfile(inspect.currentframe())) # script filename (usually with path)
        # Strip off a final suffix if present.
        config['root'] = os.path.splitext(script_name)[0]

    if logger and logger.isEnabledFor(logging.DEBUG):
        import pprint
        logger.debug("Final config dict to be processed: \n%s", pprint.pformat(config))

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
    else:
        nproc = 1 

    galsim.config.BuildFiles(nfiles, config, nproc=nproc, logger=logger)


def MultiProcess(nproc, config, job_func, jobs, item, logger=None,
                 njobs_per_task=1, done_func=None, except_func=None, except_abort=True):
    """A helper function for performing a task using multiprocessing.

    @param nproc            How many processes to use.
    @param config           The configuration dict.
    @param job_func         The function to run for each job.
    @param jobs             A list of jobs to run.  Each item is a tuple (kwargs, info).
    @param item             A string indicating what is being worked on.
    @param logger           If given, a logger object to log progress. [default: None]
    @param njobs_per_task   The number of jobs to send to the worker at a time. [default: 1]
    @param done_func        A function to run upon completion of each job. [default: None]
    @param except_func      A function to run if an exception is encountered. [default: None]
    @param except_abort     Whether an exception should abort the rest of the processing.
                            [default: True]

    @returns a list of the outputs from job_func for each job
    """
    import time

    # The worker function will be run once in each process.
    # It pulls tasks off the task_queue, runs them, and puts the results onto the results_queue
    # to send them back to the main process.
    # The *tasks* can be made up of more than one *job*.  Each job involves calling job_func
    # with the kwargs from the list of jobs.
    # Each job also carries with it some kind of information, like the file_num, file_name, etc.
    # to help identify what the job is about.  This is mostly useful for the done_func and
    # except_func to write something appropriate in the logger.
    def worker(task_queue, results_queue, config, logger):
        proc = current_process().name
        for task in iter(task_queue.get, 'STOP'):
            try :
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('%s: Received job to do %d %ss, starting with %s',
                                 proc,len(task),item,task[0][1])
                for kwargs, k in task:
                    t1 = time.time()
                    kwargs['config'] = config
                    kwargs['logger'] = logger
                    result = job_func(**kwargs)
                    t2 = time.time()
                    results_queue.put( (result, k, t2-t1, proc) )
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: Caught exception: %s\n%s',proc,str(e),tr)
                results_queue.put( (e, k, tr, proc) )
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('%s: Received STOP', proc)

    if nproc > 1:
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn("Using %d processes for %s processing",nproc,item)

        from multiprocessing import Process, Queue, current_process
        from multiprocessing.managers import BaseManager

        # Set up the task list.  Each task is defined as a list of jobs and the index k of the
        # first job to be done.  The index k is mostly important for putting the results in
        # the right order, since they will be done out of order.
        task_queue = Queue()
        njobs = len(jobs)
        for k in range(0, njobs, njobs_per_task):
            k2 = k + njobs_per_task
            if k2 > njobs: k2 = njobs
            task = [ (jobs[k][0], k) for k in range(k,k2) ]
            task_queue.put(task)

        # Temporarily mark that we are multiprocessing, so we know not to start another
        # round of multiprocessing later.
        config['current_nproc'] = nproc

        # The logger is not picklable, so we need to make a proxy for it so all the 
        # processes can emit logging information safely.
        logger_proxy = GetLoggerProxy(logger)

        # Run the tasks.
        # Each Process command starts up a parallel process that will keep checking the queue
        # for a new task. If there is one there, it grabs it and does it. If not, it waits
        # until there is one to grab. When it finds a 'STOP', it shuts down.
        results_queue = Queue()
        p_list = []
        for j in range(nproc):
            # The process name is actually the default name that Process would generate on its
            # own for the first time we do this. But after that, if we start another round of
            # multiprocessing, then it just keeps incrementing the numbers, rather than starting
            # over at Process-1.  As far as I can tell, it's not actually spawning more
            # processes, so for the sake of the logging output, we name the processes explicitly.
            p = Process(target=worker, args=(task_queue, results_queue, config, logger_proxy),
                        name='Process-%d'%(j+1))
            p.start()
            p_list.append(p)

        # In the meanwhile, the main process keeps going.  We pull each set of images off of the
        # results_queue and put them in the appropriate place in the lists.
        # This loop is happening while the other processes are still working on their tasks.
        results = [ None for k in range(len(jobs)) ]
        for kk in range(njobs):
            res, k, t, proc = results_queue.get()
            if isinstance(res,Exception):
                # res is really the exception, e
                # t is really the traceback
                # k is the job number that failed, so jobs[k][1] is the info for that job.
                except_func(logger, proc, res, t, jobs[k][1])
                if except_abort:
                    for j in range(nproc):
                        p_list[j].terminate()
                    raise res
            else:
                # The normal case
                done_func(logger, proc, jobs[k][1], res, t)
                results[k] = res
 
        # Stop the processes
        # The 'STOP's could have been put on the task list before starting the processes, or you
        # can wait.  In some cases it can be useful to clear out the results_queue (as we just did)
        # and then add on some more tasks.  We don't need that here, but it's perfectly fine to do.
        # Once you are done with the processes, putting nproc 'STOP's will stop them all.
        # This is important, because the program will keep running as long as there are running
        # processes, even if the main process gets to the end.  So you do want to make sure to
        # add those 'STOP's at some point!
        for j in range(nproc):
            task_queue.put('STOP')
        for j in range(nproc):
            p_list[j].join()
        task_queue.close()

        # And clear this out, so we know that we're not multiprocessing anymore.
        config['current_nproc'] = nproc

    else : # nproc == 1
        results = []
        for kwargs, info in jobs:
            try:
                t1 = time.time()
                kwargs['config'] = config
                kwargs['logger'] = logger
                result = job_func(**kwargs)
                t2 = time.time()
                done_func(logger, None, info, result, t2-t1)
                results.append(result)
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                except_func(logger, None, e, tr, info)
                if except_abort: raise
 
    return results
 
