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

from .input import *
from .output import *
from .output_datacube import *
from .output_multifits import *


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
        if logger:
            logger.debug('File type determined to be %s', file_type)
    else:
        if logger:
            logger.debug('File type specified to be %s', file_type)

    if file_type == 'yaml':
        if logger:
            logger.info('Reading YAML config file %s', config_file)
        return galsim.config.ReadYaml(config_file)
    else:
        if logger:
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


def SetDefaultExt(config, ext):
    """
    Some items have a default extension for a NumberedFile type.
    """
    if ( isinstance(config,dict) and 'type' in config and 
         config['type'] == 'NumberedFile' and 'ext' not in config ):
        config['ext'] = ext


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

    if logger:
        import pprint
        logger.debug("Final config dict to be processed: \n%s", pprint.pformat(config))

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

    # can_do_multiple says whether the function can in principal do multiple files
    can_do_multiple = valid_output_types[type][2]

    # extra_file_name says whether the function takes psf_file_name, etc.
    extra_file_name = valid_output_types[type][3]

    # extra_hdu says whether the function takes psf_hdu, etc.
    extra_hdu = valid_output_types[type][4]
    if logger:
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
    if logger:
        logger.debug('nfiles = %d',nfiles)

    # Figure out how many processes we will use for building the files.
    # (If nfiles = 1, but nimages > 1, we'll do the multi-processing at the image stage.)
    if 'nproc' in output:
        nproc = galsim.config.ParseValue(output, 'nproc', config, int)[0]
    else:
        nproc = 1 

    # If set, nproc2 will be passed to the build function to be acted on at that level.
    nproc2 = None
    if nproc > nfiles:
        if nfiles == 1 and can_do_multiple:
            nproc2 = nproc 
            nproc = 1
        else:
            if logger:
                logger.warn(
                    "Trying to use more processes than files: output.nproc=%d, "%nproc +
                    "output.nfiles=%d.  Reducing nproc to %d."%(nfiles,nfiles))
            nproc = nfiles

    if nproc <= 0:
        # Try to figure out a good number of processes to use
        try:
            from multiprocessing import cpu_count
            ncpu = cpu_count()
            if nfiles == 1 and can_do_multiple:
                nproc2 = ncpu # Use this value in BuildImages rather than here.
                nproc = 1
                if logger:
                    logger.debug("ncpu = %d.",ncpu)
            else:
                if ncpu > nfiles:
                    nproc = nfiles
                else:
                    nproc = ncpu
                if logger:
                    logger.info("ncpu = %d.  Using %d processes",ncpu,nproc)
        except:
            if logger:
                logger.warn("config.output.nproc <= 0, but unable to determine number of cpus.")
            nproc = 1

    def worker(input, output):
        proc = current_process().name
        for job in iter(input.get, 'STOP'):
            try:
                (kwargs, file_num, file_name, logger) = job
                if logger:
                    logger.debug('%s: Received job to do file %d, %s',proc,file_num,file_name)
                ProcessInput(kwargs['config'], file_num=file_num, logger=logger)
                if logger:
                    logger.debug('%s: After ProcessInput for file %d',proc,file_num)
                kwargs['logger'] = logger
                t = build_func(**kwargs)
                if logger:
                    logger.debug('%s: After %s for file %d',proc,build_func,file_num)
                output.put( (t, file_num, file_name, proc) )
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger:
                    logger.debug('%s: Caught exception %s\n%s',proc,str(e),tr)
                output.put( (e, file_num, file_name, tr) )

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
        config['first_seed'] = galsim.config.ParseValue(
                config['image'], 'random_seed', config, int)[0]

    nfiles_use = nfiles
    for file_num in range(nfiles):
        if logger:
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

        # Set up random_seed appropriately if necessary.
        if 'first_seed' in config:
            config['image']['random_seed'] = {
                'type' : 'Sequence' ,
                'first' : config['first_seed']
            }

        # It is possible that some items at image scope could need a random number generator.
        # For example, in demo9, we have a random number of objects per image.
        # So we need to build an rng here.
        if 'random_seed' in config['image']:
            config['index_key'] = 'obj_num'
            seed = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
            config['index_key'] = 'file_num'
            if logger:
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
        if nproc2:
            kwargs['nproc'] = nproc2

        output = config['output']
        # This also updates nimages or nobjects as needed if they are being automatically
        # set from an input catalog.
        nobj = nobj_func(config,file_num,image_num)
        if logger:
            logger.debug('file %d: nobj = %s',file_num,str(nobj))

        # nobj is a list of nobj for each image in that file.
        # So len(nobj) = nimages and sum(nobj) is the total number of objects
        # This gets the values of image_num and obj_num ready for the next loop.
        image_num += len(nobj)
        obj_num += sum(nobj)

        # Check if we ought to skip this file
        if ('skip' in output 
                and galsim.config.ParseValue(output, 'skip', config, bool)[0]):
            if logger:
                logger.warn('Skipping file %d = %s because output.skip = True',file_num,file_name)
            nfiles_use -= 1
            continue
        if ('noclobber' in output 
                and galsim.config.ParseValue(output, 'noclobber', config, bool)[0]
                and os.path.isfile(file_name)):
            if logger:
                logger.warn('Skipping file %d = %s because output.noclobber = True' +
                            ' and file exists',file_num,file_name)
            nfiles_use -= 1
            continue

        # Check if we need to build extra images for write out as well
        for extra_key in [ key for key in extra_keys if key in output ]:
            if logger:
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
                    if logger:
                        logger.debug('skipping %s, since already written',f)
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
            # Make new copies of config and kwargs so we can update them without
            # clobbering the versions for other tasks on the queue.
            kwargs1 = copy.copy(kwargs)
            # Clear out unsafe proxy objects, since there seems to be a bug in the manager
            # package where this can cause strange KeyError exceptions in the incref function.
            # It seems to be related to having multiple identical proxy objects that then
            # get deleted.  e.g. if the first N files use one dict, then the next N use another,
            # and so forth.  I don't really get it, but clearing them out here seems to 
            # fix the problem.
            ProcessInput(config, file_num=file_num, logger=logger_proxy, safe_only=True)
            kwargs1['config'] = CopyConfig(config)
            task_queue.put( (kwargs1, file_num, file_name, logger_proxy) )
        else:
            try:
                ProcessInput(config, file_num=file_num, logger=logger_proxy)
                if logger:
                    logger.debug('file %d: After ProcessInput',file_num)
                kwargs['config'] = config
                kwargs['logger'] = logger 
                t = build_func(**kwargs)
                if logger:
                    logger.warn('File %d = %s: time = %f sec', file_num, file_name, t)
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
        if logger:
            logger.warn("Using %d processes",nproc)
        import time
        t1 = time.time()
        # Run the tasks
        done_queue = Queue()
        p_list = []
        for j in range(nproc):
            p = Process(target=worker, args=(task_queue, done_queue), name='Process-%d'%(j+1))
            p.start()
            p_list.append(p)

        # Log the results.
        if logger:
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
                if logger:
                    logger.warn('%s: File %d = %s: time = %f sec', proc, file_num, file_name, t)

        # Stop the processes
        for j in range(nproc):
            task_queue.put('STOP')
        for j in range(nproc):
            p_list[j].join()
        task_queue.close()
        t2 = time.time()
        if logger:
            logger.warn('Total time for %d files with %d processes = %f sec', 
                        nfiles_use,nproc,t2-t1)

    if logger:
        logger.debug('Done building files')

