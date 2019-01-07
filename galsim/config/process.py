# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
import copy
from collections import OrderedDict

def MergeConfig(config1, config2, logger=None):
    """
    Merge config2 into config1 such that it has all the information from either config1 or
    config2 including places where both input dicts have some of a field defined.
    e.g. config1 has image.pixel_scale, and config2 has image.noise.
            Then the returned dict will have both.
    For real conflicts (the same value in both cases), config1's value takes precedence
    """
    logger = LoggerWrapper(logger)
    for (key, value) in config2.items():
        if not key in config1:
            # If this key isn't in config1 yet, just add it
            config1[key] = copy.deepcopy(value)
        elif isinstance(value,dict) and isinstance(config1[key],dict):
            # If they both have a key, first check if the values are dicts
            # If they are, just recurse this process and merge those dicts.
            MergeConfig(config1[key],value,logger)
        else:
            # Otherwise config1 takes precedence
            logger.info("Not merging key %s from the base config, since the later "
                        "one takes precedence",key)
            pass

def ReadYaml(config_file):
    """Read in a YAML configuration file and return the corresponding dicts.

    A YAML file is allowed to define several dicts using multiple documents. The GalSim parser
    treats this as a set of multiple jobs to be done.  The first document is taken to be a "base"
    dict that has common definitions for all the jobs.  Then each subsequent document has the
    (usually small) modifications to the base dict for each job.  See demo6.yaml, demo8.yaml and
    demo9.yaml in the GalSim/examples directory for example usage.

    The return value will be a list of dicts, one dict for each job to be done.

    @param config_file      The name of the configuration file to read.

    @returns list of config dicts
    """
    import yaml

    # cf. coldfix's answer here:
    # http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    class OrderedLoader(yaml.SafeLoader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    with open(config_file) as f:
        all_config = [ c for c in yaml.load_all(f.read(), OrderedLoader) ]

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
        # And merge it into each of the other dicts
        for c in all_config:
            MergeConfig(c, base_config)

    return all_config


def ReadJson(config_file):
    """Read in a JSON configuration file and return the corresponding dicts.

    A JSON file only defines a single dict.  However to be parallel to the functionality of
    ReadYaml, the output is a list with a single item, which is the dict defined by the JSON file.

    @param config_file      The name of the configuration file to read.

    @returns [config_dict]
    """
    import json

    with open(config_file) as f:
        # cf. http://stackoverflow.com/questions/6921699/can-i-get-json-to-load-into-an-ordereddict-in-python
        config = json.load(f, object_pairs_hook=OrderedDict)

    # JSON files only ever define a single job, but we need to return a list with this one item.
    return [config]


def ConvertNones(config):
    """Convert any items whose value is 'None' to None.

    To allow some parameters to be set to None in the config dict (e.g. in a list, where only
    some values need to be None), we convert all values == 'None' to None.

    @param config       The config dict to process
    """
    if isinstance(config, dict):
        keys = config.keys()
    else:
        keys = range(len(config))

    for key in keys:
        # Recurse to lower levels, if any
        if isinstance(config[key],(list,dict)):
            ConvertNones(config[key])

        # Convert any Nones at this level
        elif config[key] == 'None':
            config[key] = None


def ReadConfig(config_file, file_type=None, logger=None):
    """Read in a configuration file and return the corresponding dicts.

    A YAML file is allowed to define several dicts using multiple documents. The GalSim parser
    treats this as a set of multiple jobs to be done.  The first document is taken to be a "base"
    dict that has common definitions for all the jobs.  Then each subsequent document has the
    (usually small) modifications to the base dict for each job.  See demo6.yaml, demo8.yaml and
    demo9.yaml in the GalSim/examples directory for example usage.

    On output, the returned list will have an entry for each job to be done.  If there are
    multiple documents, then the first dict is a merge of the first two documents, the
    second a merge of the first and third, and so on.  Each job includes the first document
    merged with each subseqent document in turn.  If there is only one document defined,
    the returned list will have one element, which is this dict.

    A JSON file does not have this feature, but to be consistent, we always return a list,
    which would only have one element in this case.

    Also, we actually read in the config file into an OrderedDict.  The main advantage of
    this is for the truth catalog.  This lets the columns be in the same order as the entries
    in the config file.  With a normal dict, they get scrambled.

    @param config_file      The name of the configuration file to read.
    @param file_type        If given, the type of file to read.  [default: None, which mean
                            infer the file type from the extension.]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns list of config dicts
    """
    logger = LoggerWrapper(logger)
    # Determine the file type from the extension if necessary:
    if file_type is None:
        import os
        name, ext = os.path.splitext(config_file)
        if ext.lower().startswith('.j'):
            file_type = 'json'
        else:
            # Let YAML be the default if the extension is not .y* or .j*.
            file_type = 'yaml'
        logger.debug('File type determined to be %s', file_type)
    else:
        logger.debug('File type specified to be %s', file_type)

    if file_type == 'yaml':
        logger.info('Reading YAML config file %s', config_file)
        config = galsim.config.ReadYaml(config_file)
    else:
        logger.info('Reading JSON config file %s', config_file)
        config = galsim.config.ReadJson(config_file)

    galsim.config.ConvertNones(config)

    return config


def RemoveCurrent(config, keep_safe=False, type=None, index_key=None):
    """
    Remove any "current" values stored in the config dict at any level.

    @param config       The configuration dict.
    @param keep_safe    Should current values that are marked as safe be preserved?
                        [default: False]
    @param type         If provided, only clear the current value of objects that use this
                        particular type.  [default: None, which means to remove current values
                        of all types.]
    @param index_key    If provided, only clear the current value of objects that use this
                        index_key (or start with this index_key, so obj_num also does
                        obj_num_in_file).  [default: None]
    """
    # End recursion if this is not a dict.
    if not isinstance(config,dict): return

    # Recurse to lower levels, if any
    force = False  # If lower levels removed anything, then force removal at this level as well.
    for key in config:
        if key[0] == '_': continue  # These are our own implementation details, not the normal dict.
        if isinstance(config[key],list):
            for item in config[key]:
                force = RemoveCurrent(item, keep_safe, type, index_key) or force
        else:
            force = RemoveCurrent(config[key], keep_safe, type, index_key) or force
    if force:
        keep_safe = False
        type = None
        index_key = None

    # Delete the current_val at this level, if any
    if 'current' in config:
        cval, csafe, ctype, cindex, cindex_key = config['current']
        if (not (keep_safe and csafe)
                and (type is None or ('type' in config and config['type'] == type))
                and (index_key is None or cindex_key.startswith(index_key))):
            del config['current']
            return True
    return force

top_level_fields = ['psf', 'gal', 'stamp', 'image', 'input', 'output',
                    'eval_variables', 'root', 'modules', 'profile']

rng_fields = ['rng', 'obj_num_rng', 'image_num_rng', 'file_num_rng',
              'obj_num_rngs', 'image_num_rngs', 'file_num_rngs']

def CopyConfig(config):
    """
    If you want to use a config dict for multiprocessing, you need to deep copy
    the gal, psf, and pix fields, since they cache values that are not picklable.
    If you don't do the deep copy, then python balks when trying to send the updated
    config dict back to the root process.  We do this a few different times, so encapsulate
    the copy semantics once here.

    @param config           The configuration dict to copy.

    @returns a deep copy of the config dict.
    """
    import copy
    config1 = copy.copy(config)

    # Make sure the input_manager isn't in the copy
    config1.pop('_input_manager',None)

    # Now deepcopy all the regular config fields to make sure things like current don't
    # get clobbered by two processes writing to the same dict.  Also the rngs.
    for field in top_level_fields + rng_fields:
        if field in config:
            config1[field] = copy.deepcopy(config[field])

    return config1

def GetLoggerProxy(logger):
    """Make a proxy for the given logger that can be passed into multiprocessing Processes
    and used safely.

    @param logger       The logger to make a copy of

    @returns a proxy for the given logger
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

class LoggerWrapper(object):
    """A wrap around a Logger object that checks whether a debug or info or warn call will
    actually produce any output before calling the functions.

    This seems like a gratuitous wrapper, and it is if the object being wrapped is a real
    Logger object.  However, we use it to wrap proxy objects (returned from GetLoggerProxy)
    that would otherwise send the arguments of logger.debug(...) calls through a multiprocessing
    pipe before (typically) being ignored.  Here, we check whether the call will actually
    produce any output before calling the functions.

    @param logger       The logger object to wrap.
    """
    def __init__(self, logger):
        if isinstance(logger,LoggerWrapper):
            self.logger = logger.logger
        else:
            self.logger = logger

    def __bool__(self):
        return self.logger is not None
    __nonzero__ = __bool__

    def debug(self, *args, **kwargs):
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if self.logger and self.logger.isEnabledFor(logging.WARNING):
            self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        if self.logger and self.logger.isEnabledFor(logging.ERROR):
            self.logger.error(*args, **kwargs)

    def log(self, lvl, *args, **kwargs):
        if self.logger and self.logger.isEnabledFor(lvl):
            self.logger.log(lvl, *args, **kwargs)

    def isEnabledFor(self, *args, **kwargs):
        if self.logger:
            return self.logger.isEnabledFor(*args,**kwargs)
        else:
            return False


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
    logger = LoggerWrapper(logger)
    # First if nproc < 0, update based on ncpu
    if nproc <= 0:
        # Try to figure out a good number of processes to use
        try:
            from multiprocessing import cpu_count
            nproc = cpu_count()
            logger.debug("ncpu = %d.",nproc)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # pragma: no cover
            logger.warning("nproc <= 0, but unable to determine number of cpus.")
            logger.warning("Caught error: %s",e)
            logger.warning("Using single process")
            nproc = 1

    # Second, make sure we aren't already in a multiprocessing mode
    if nproc > 1 and 'current_nproc' in config:
        logger.debug("Already multiprocessing.  Ignoring image.nproc")
        nproc = 1

    # Finally, don't try to use more processes than jobs.  It wouldn't fail or anything.
    # It just looks bad to have 3 images processed with 8 processes or something like that.
    if nproc > ntot:
        logger.debug("There are only %d jobs to do.  Reducing nproc to %d."%(ntot,ntot))
        nproc = ntot
    return nproc


def ParseRandomSeed(config, param_name, base, seed_offset):
    # Normally, random_seed parameter is just a number, which really means to use that number
    # for the first item and go up sequentially from there for each object.
    # However, we allow for random_seed to be a gettable parameter, so for the
    # normal case, we just convert it into a Sequence.
    if isinstance(config[param_name], int):
         # The "first" is actually the seed value to use for anything at file or image scope
         # using the obj_num of the first object in the file or image.  Seeds for objects
         # will start at 1 more than this.
         first = galsim.config.ParseValue(config, param_name, base, int)[0]
         config[param_name] = {
                 'type' : 'Sequence',
                 'index_key' : 'obj_num',
                 'first' : first
         }

    index_key = base['index_key']
    if index_key == 'obj_num':
        # The normal case
        seed = galsim.config.ParseValue(config, param_name, base, int)[0]
    else:
        # If we are setting either the file_num or image_num rng, we need to be careful.
        base['index_key'] = 'obj_num'
        seed = galsim.config.ParseValue(config, param_name, base, int)[0]
        base['index_key'] = index_key
    seed += seed_offset

    # Normally, the file_num rng and the image_num rng can be the same.  So if seed
    # comes out the same as what we already built, then just use the existing file_num_rng.
    if index_key == 'image_num' and seed == base.get('file_num_seed',None):
        rng = base['file_num_rng']
    else:
        rng = galsim.BaseDeviate(seed)

    return seed, rng

def PropagateIndexKeyRNGNum(config, index_key, rng_num):
    """Propagate any index_key or rng_num specification in a dict to all sub-fields
    """
    if isinstance(config, list):
        for item in config:
            PropagateIndexKeyRNGNum(item, index_key, rng_num)
        return

    if not isinstance(config, dict): return

    if 'index_key' in config:
        index_key = config['index_key']
    elif index_key is not None:
        config['index_key'] = index_key

    if 'rng_num' in config:
        rng_num = config['rng_num']
    elif rng_num is not None:
        config['rng_num'] = rng_num

    for key, field in config.items():
        if key[0] == '_': continue
        PropagateIndexKeyRNGNum(field, index_key, rng_num)


def SetupConfigRNG(config, seed_offset=0, logger=None):
    """Set up the RNG in the config dict.

    - Setup config['image']['random_seed'] if necessary
    - Set config['rng'] and other related values based on appropriate random_seed

    @param config           The configuration dict.
    @param seed_offset      An offset to use relative to what config['image']['random_seed'] gives.
                            [default: 0]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns the seed used to initialize the RNG.
    """
    logger = LoggerWrapper(logger)

    # If we are starting a new file, clear out the existing rngs.
    index_key = config['index_key']
    if index_key == 'file_num':
        for key in rng_fields + ['seed', 'obj_num_seed', 'image_num_seed', 'file_num_seed']:
            config.pop(key, None)

    # This can be present for efficiency, since GaussianDeviates produce two values at a time,
    # so it is more efficient to not create a new GaussianDeviate object each time.
    # But if so, we need to remove it now.
    config.pop('gd', None)

    # All fields have a default index_key that they would normally use as well as a default
    # rng_num = 0 if the random_seed is a list.  But we allow users to specify alternate values
    # of index_key and/or rng_num at any level in the dict.  We want that specification to
    # propagate down to lower levels from that point forward.  The easiest way to do so is to
    # explicitly run through the dict once and propagate any index_key and/or rng_num fields
    # to all sub-fields.
    if not config.get('_propagated_index_key_rng_num',False):
        logger.debug('Propagating any index_key or rng_num specifications')
        for field in config.values():
            PropagateIndexKeyRNGNum(field, None, None)
        config['_propagated_index_key_rng_num'] = True

    if 'image' not in config or 'random_seed' not in config['image']:
        logger.debug('obj %d: No random_seed specified.  Using /dev/urandom',
                     config.get('obj_num',0))
        config['seed'] = 0
        rng = galsim.BaseDeviate()
        config['rng'] = rng
        config[index_key + '_rng'] = rng
        return 0

    image = config['image']
    # Normally, there is just one random_seed and it is just an integer.  However, sometimes
    # it is useful to have 2 (or more) rngs going with different seed sequences.  To enable this,
    # image.random_seed is allowed to be a list, each item of which may be either an integer or
    # a dict defining an integer sequence.  If it is a list, we parse each item separately
    # and then put the combined results into config['rng'] as a list.
    if isinstance(image['random_seed'], list):
        lst = image['random_seed']
        logger.debug('random_seed = %s',CleanConfig(lst))
        logger.debug('seed_offset = %s',seed_offset)
        seeds = []
        rngs = []
        for i in range(len(lst)):
            seed, rng = ParseRandomSeed(lst, i, config, seed_offset)
            logger.debug('seed %d = %s',i,seed)
            seeds.append(seed)
            rngs.append(rng)
            if i == 0:
                # Helpful to get this done right away, because later seeds might be based
                # on a random number that uses the first rng.
                # cf. test_eval_full_word in test_config_output.py.
                config['seed'] = seed
                config['rng'] = rng
                config[index_key + '_seed'] = seed
                config[index_key + '_rng'] = rng
        config[index_key + '_rngs'] = rngs
        logger.debug('obj %d: random_seed is a list. Initializing rngs with seeds %s',
                     config.get('obj_num',0), seeds)
        return seeds[0]
    else:
        seed, rng = ParseRandomSeed(image, 'random_seed', config, seed_offset)
        config['seed'] = seed
        config['rng'] = rng
        config[index_key + '_seed'] = seed
        config[index_key + '_rng'] = rng
        logger.debug('obj %d: Initializing rng with seed %s', config.get('obj_num',0), seed)
        return seed


def ImportModules(config):
    """Import any modules listed in config['modules'].

    These won't be brought into the running scope of the config processing, but any side
    effects of the import statements will persist.  In particular, these are allowed to
    register additional custom types that can then be used in the current config dict.

    @param config           The configuration dict.
    """
    import sys
    if 'modules' in config:
        for module in config['modules']:
            try:
                # Do this first to let user modules take precedence
                exec('import '+module)
            except ImportError:
                try:
                    exec('import galsim.'+module)
                except ImportError:
                    # But do it again if everything fails to give a better error message.
                    # Also make sure '.' in the path to load local modules.
                    if '.' not in sys.path:
                        sys.path.append('.')
                    exec('import '+module)


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
    while True:
        d = config
        k = chain.pop(0)
        try: k = int(k)
        except ValueError: pass
        if len(chain) == 0: break
        try:
            config = d[k]
        except (TypeError, KeyError):
            # TypeError for the case where d is a float or Position2D, so d[k] is invalid.
            # KeyError for the case where d is a dict, but k is not a valid key.
            raise galsim.GalSimConfigError(
                "Unable to parse extended key %s.  Field %s is invalid."%(key,k))
    return d, k

def GetFromConfig(config, key):
    """Get the value for the (possibly extended) key from a config dict.

    If key is a simple string, then this is equivalent to config[key].
    However, key is allowed to be a chain of keys such as 'gal.items.0.ellip.e', in which
    case this function will return config['gal']['items'][0]['ellip']['e'].

    @param config       The configuration dict.
    @param key          The possibly extended key.

    @returns the value of that key from the config.
    """
    d, k = ParseExtendedKey(config, key)
    try:
        value = d[k]
    except Exception as e:
        raise galsim.GalSimConfigError(
            "Unable to parse extended key %s.  Field %s is invalid."%(key,k))
    return value

def SetInConfig(config, key, value):
    """Set the value of a (possibly extended) key in a config dict.

    If key is a simple string, then this is equivalent to config[key] = value.
    However, key is allowed to be a chain of keys such as 'gal.items.0.ellip.e', in which
    case this function will set config['gal']['items'][0]['ellip']['e'] = value.

    @param config       The configuration dict.
    @param key          The possibly extended key.

    @returns the value of that key from the config.
    """
    d, k = ParseExtendedKey(config, key)
    if value == '':
        # This means remove it, if it is there.
        d.pop(k,None)
    else:
        try:
            d[k] = value
        except Exception as e:
            raise galsim.GalSimConfigError(
                "Unable to parse extended key %s.  Field %s is invalid."%(key,k))


def UpdateConfig(config, new_params):
    """Update the given config dict with additional parameters/values.

    @param config           The configuration dict to update.
    @param new_params       A dict of parameters to update.  The keys of this dict may be
                            chained field names, such as gal.first.dilate, which will be
                            parsed to update config['gal']['first']['dilate'].
    """
    for key, value in new_params.items():
        SetInConfig(config, key, value)


def ProcessTemplate(config, base, logger=None):
    """If the config dict has a 'template' item, read in the appropriate file and
    make any requested updates.

    @param config           The configuration dict.
    @param base             The base configuration dict.
    @param logger           If given, a logger object to log progress. [default: None]
    """
    logger = LoggerWrapper(logger)
    if 'template' in config:
        template_string = config.pop('template')
        logger.debug("Processing template specified as %s",template_string)

        # Parse the template string
        if ':' in template_string:
            config_file, field = template_string.split(':')
        else:
            config_file, field = template_string, None

        # Read the config file if appropriate
        if config_file != '':
            template = ReadConfig(config_file, logger=logger)[0]
        else:
            template = base

        # Pull out the specified field, if any
        if field is not None:
            template = GetFromConfig(template, field)

        # Copy over the template config into this one.
        new_params = config.copy()  # N.B. Already popped config['template'].
        config.clear()
        config.update(template)

        # Update the config with the requested changes
        UpdateConfig(config, new_params)


def ProcessAllTemplates(config, logger=None, base=None):
    """Check through the full config dict and process any fields that have a 'template' item.

    @param config           The configuration dict.
    @param base             The base configuration dict.
    @param logger           If given, a logger object to log progress. [default: None]
    """
    if base is None: base = config
    ProcessTemplate(config, base, logger)
    for (key, field) in config.items():
        if isinstance(field, dict):
            ProcessAllTemplates(field, logger, base)
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, dict):
                    ProcessAllTemplates(item, logger, base)

# This is the main script to process everything in the configuration dict.
def Process(config, logger=None, njobs=1, job=1, new_params=None, except_abort=False):
    """
    Do all processing of the provided configuration dict.  In particular, this
    function handles processing the output field, calling other functions to
    build and write the specified files.  The input field is processed before
    building each file.

    Sometimes, it can be helpful to split up a processing jobs over multiple machines
    (i.e. not just multiple processes, which can be handled natively with the output.nproc
    or image.nproc options).  In this case, you can ask the Process command to split up
    the total amount of work into njobs and only do one of those jobs here.  To do this,
    set njobs to be the number of jobs total and job to be which job should be done here.

    @param config           The configuration dict.
    @param logger           If given, a logger object to log progress. [default: None]
    @param njobs            The total number of jobs to split the work into. [default: 1]
    @param job              Which job should be worked on here (1..njobs). [default: 1]
    @param new_params       A dict of new parameter values that should be used to update the config
                            dict after any template loading (if any). [default: None]
    @param except_abort     Whether to abort processing when a file raises an exception (True)
                            or just report errors and continue on (False). [default: False]
    """
    logger = LoggerWrapper(logger)
    import pprint
    if njobs < 1:
        raise galsim.GalSimValueError("Invalid number of jobs",njobs)
    if job < 1:
        raise galsim.GalSimValueError("Invalid job number.  Must be >= 1.",job)
    if job > njobs:
        raise galsim.GalSimValueError("Invalid job number.  Must be <= njobs (%d)"%(njobs),job)

    # First thing to do is deep copy the input config to make sure we don't modify the original.
    config = CopyConfig(config)

    # Process any template specifications in the dict.
    ProcessAllTemplates(config, logger)

    # Update using any new_params that are given:
    if new_params is not None:
        UpdateConfig(config, new_params)

    # Import any modules if requested
    ImportModules(config)

    logger.debug("Final config dict to be processed: \n%s", pprint.pformat(config))

    # Warn about any unexpected fields.
    unexpected = [ k for k in config if k not in top_level_fields ]
    if len(unexpected) > 0 and logger:
        logger.warning("Warning: config dict contains the following unexpected fields: %s.",
                       unexpected)
        logger.warning("These fields are not (directly) processed by the config processing.")

    # Determine how many files we will be processing in total.
    # Usually, this is just output.nfiles, but different output types may define this differently.
    nfiles = galsim.config.output.GetNFiles(config)
    logger.debug('nfiles = %d',nfiles)

    if njobs > 1:
        # Start each job at file_num = nfiles * job / njobs
        start = nfiles * (job-1) // njobs
        end = nfiles * job // njobs
        logger.warning('Splitting work into %d jobs.  Doing job %d',njobs,job)
        logger.warning('Building %d out of %d total files: file_num = %d .. %d',
                       end-start,nfiles,start,end-1)
        nfiles = end-start
    else:
        start = 0

    if nfiles == 1:
        except_abort = True  # Mostly just so the message reads better.

    #BuildFiles returns the config dictionary, which can includes stuff added
    #by custom output types during the run.
    config_out = galsim.config.BuildFiles(nfiles, config, file_num=start, logger=logger,
                                         except_abort=except_abort)
    #Return config_out in case useful
    return config_out

def MultiProcess(nproc, config, job_func, tasks, item, logger=None,
                 done_func=None, except_func=None, except_abort=True):
    """A helper function for performing a task using multiprocessing.

    A note about the nomenclature here.  We use the term "job" to mean the job of building a single
    file or image or stamp.  The output of each job is gathered into the list of results that
    is returned.  A task is a collection of one or more jobs that are all done by the same
    processor.  For simple cases, each task is just a single job, but for things like a Ring
    test, the task needs to have the jobs for a full ring.

    The tasks argument is a list of tasks.
    Each task in that list is a list of jobs.
    Each job is a tuple consisting of (kwargs, k), where kwargs is the dict of kwargs to pass to
    the job_func and k is the index of this job in the full list of jobs.

    @param nproc            How many processes to use.
    @param config           The configuration dict.
    @param job_func         The function to run for each job.  It will be called as
                                result = job_func(**kwargs)
                            where kwargs is from one of the jobs in the task list.
    @param tasks            A list of tasks to run.  Each task is a list of jobs, each of which is
                            a tuple (kwargs, k).
    @param item             A string indicating what is being worked on.
    @param logger           If given, a logger object to log progress. [default: None]
    @param done_func        A function to run upon completion of each job.  It will be called as
                                done_func(logger, proc, k, result, t)
                            where proc is the process name, k is the index of the job, result is
                            the return value of that job, and t is the time taken. [default: None]
    @param except_func      A function to run if an exception is encountered.  It will be called as
                                except_func(logger, proc, k, ex, tr)
                            where proc is the process name, k is the index of the job that failed,
                            ex is the exception caught, and tr is the traceback. [default: None]
    @param except_abort     Whether an exception should abort the rest of the processing.
                            If False, then the returned results list will not include anything
                            for the jobs that failed.  [default: True]

    @returns a list of the outputs from job_func for each job
    """
    import time
    import traceback

    # The worker function will be run once in each process.
    # It pulls tasks off the task_queue, runs them, and puts the results onto the results_queue
    # to send them back to the main process.
    # The *tasks* can be made up of more than one *job*.  Each job involves calling job_func
    # with the kwargs from the list of jobs.
    # Each job also carries with it its index in the original list of all jobs.
    def worker(task_queue, results_queue, config, logger):
        proc = current_process().name

        # The logger object passed in here is a proxy object.  This means that all the arguments
        # to any logging commands are passed through the pipe to the real Logger object on the
        # other end of the pipe.  This tends to produce a lot of unnecessary communication, since
        # most of those commands don't actually produce any output (e.g. logger.debug(..) commands
        # when the logging level is not DEBUG).  So it is helpful to wrap this object in a
        # LoggerWrapper that checks whether it is worth sending the arguments back to the original
        # Logger before calling the functions.
        logger = LoggerWrapper(logger)

        if 'profile' in config and config['profile']:
            import cProfile, pstats, io
            pr = cProfile.Profile()
            pr.enable()
        else:
            pr = None

        for task in iter(task_queue.get, 'STOP'):
            try :
                logger.debug('%s: Received job to do %d %ss, starting with %s',
                             proc,len(task),item,task[0][1])
                for kwargs, k in task:
                    t1 = time.time()
                    kwargs['config'] = config
                    kwargs['logger'] = logger
                    result = job_func(**kwargs)
                    t2 = time.time()
                    results_queue.put( (result, k, t2-t1, proc) )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                tr = traceback.format_exc()
                logger.debug('%s: Caught exception: %s\n%s',proc,str(e),tr)
                results_queue.put( (e, k, tr, proc) )
        logger.debug('%s: Received STOP', proc)
        if pr is not None:
            pr.disable()
            try:
                from StringIO import StringIO
            except ImportError:
                from io import StringIO
            s = StringIO()
            sortby = 'time'  # Note: This is now called tottime, but time seems to be a valid
                             # alias for this that is backwards compatible to older versions
                             # of pstats.
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby).reverse_order()
            ps.print_stats()
            logger.error("*** Start profile for %s ***\n%s\n*** End profile for %s ***",
                         proc,s.getvalue(),proc)

    njobs = sum([len(task) for task in tasks])

    if nproc > 1:
        logger.warning("Using %d processes for %s processing",nproc,item)

        from multiprocessing import Process, Queue, current_process
        from multiprocessing.managers import BaseManager

        # Send the tasks to the task_queue.
        task_queue = Queue()
        for task in tasks:
            task_queue.put(task)

        # Temporarily mark that we are multiprocessing, so we know not to start another
        # round of multiprocessing later.
        config['current_nproc'] = nproc

        if 'profile' in config and config['profile']:
            logger.info("Starting separate profiling for each of the %d processes.",nproc)

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

        raise_error = None

        try:
            # In the meanwhile, the main process keeps going.  We pull each set of images off of the
            # results_queue and put them in the appropriate place in the lists.
            # This loop is happening while the other processes are still working on their tasks.
            results = [ None for k in range(njobs) ]
            for kk in range(njobs):
                res, k, t, proc = results_queue.get()
                if isinstance(res, Exception):
                    # res is really the exception, e
                    # t is really the traceback
                    # k is the index for the job that failed
                    if except_func is not None:  # pragma: no branch
                        except_func(logger, proc, k, res, t)
                    if except_abort or isinstance(res, KeyboardInterrupt):
                        for j in range(nproc):
                            p_list[j].terminate()
                        raise_error = res
                        break
                else:
                    # The normal case
                    if done_func is not None:  # pragma: no branch
                        done_func(logger, proc, k, res, t)
                    results[k] = res

        except Exception as e:  # pragma: no cover
            logger.error("Caught a fatal exception during multiprocessing:\n%r",e)
            logger.error("%s",traceback.format_exc())
            # Clear any unclaimed jobs that are still in the queue
            while not task_queue.empty():
                task_queue.get()
            # And terminate any jobs that might still be running.
            for j in range(nproc):
                p_list[j].terminate()
            raise_error = e

        finally:
            # Stop the processes
            # Once you are done with the processes, putting nproc 'STOP's will stop them all.
            # This is important, because the program will keep running as long as there are running
            # processes, even if the main process gets to the end.  So you do want to make sure to
            # add those 'STOP's at some point!
            for j in range(nproc):
                task_queue.put('STOP')
            for j in range(nproc):
                p_list[j].join()
            task_queue.close()

            del config['current_nproc']

        if raise_error is not None:
            raise raise_error

    else : # nproc == 1
        results = [ None ] * njobs
        for task in tasks:
            for kwargs, k in task:
                try:
                    t1 = time.time()
                    kwargs['config'] = config
                    kwargs['logger'] = logger
                    result = job_func(**kwargs)
                    t2 = time.time()
                    if done_func is not None:  # pragma: no branch
                        done_func(logger, None, k, result, t2-t1)
                    results[k] = result
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    tr = traceback.format_exc()
                    if except_func is not None: # pragma: no branch
                        except_func(logger, None, k, e, tr)
                    if except_abort or isinstance(e, KeyboardInterrupt):
                        raise

    # If there are any failures, then there will still be some Nones in the results list.
    # Remove them.
    results = [ r for r in results if r is not None ]

    return results


valid_index_keys = [ 'obj_num_in_file', 'obj_num', 'image_num', 'file_num' ]

def GetIndex(config, base, is_sequence=False):
    """Return the index to use for the current object or parameter and the index_key.

    First check for an explicit index_key value given by the user.
    Then if base[index_key] is other than obj_num, use that.
    Finally, if this is a sequence, default to 'obj_num_in_file', otherwise 'obj_num'.

    @returns index, index_key
    """
    if 'index_key' in config:
        index_key = config['index_key']
        if index_key not in valid_index_keys:
            raise galsim.GalSimConfigValueError("Invalid index_key.", index_key, valid_index_keys)
    else:
        index_key = base.get('index_key','obj_num')
        if index_key == 'obj_num' and is_sequence:
            index_key = 'obj_num_in_file'

    if index_key == 'obj_num_in_file':
        index = base.get('obj_num',0) - base.get('start_obj_num',0)
        index_key = 'obj_num'
    else:
        index = base.get(index_key,0)

    return index, index_key


def GetRNG(config, base, logger=None, tag=''):
    """Get the appropriate current rng according to whatever the current index_key is.

    If a logger is provided, then it will emit a warning if there is no current rng setup.

    @param config           The configuration dict for the current item being worked on.
    @param base             The base configuration dict.
    @param logger           If given, a logger object to log progress. [default: None]
    @param tag              If given, an appropriate name for the current item to use in the
                            warning message. [default: '']

    @returns either the appropriate rng for the current index_key or None
    """
    logger = LoggerWrapper(logger)
    index, index_key = GetIndex(config, base)
    logger.debug("GetRNG for %s: %s",index_key,index)

    rng_num = config.get('rng_num', 0)
    if rng_num != 0:
        if int(rng_num) != rng_num:
            raise galsim.GalSimConfigValueError("rng_num must be an integer", rng_num)
        rngs = base.get(index_key + '_rngs', None)
        if rngs is None:
            raise galsim.GalSimConfigError(
                "rng_num is only allowed when image.random_seed is a list")
        if rng_num < 0 or rng_num > len(rngs):
            raise galsim.GalSimConfigError(
                "rng_num is invalid.  Must be in [0,%d]"%(len(rngs)))
        rng = rngs[int(rng_num)]
    else:
        rng = base.get(index_key + '_rng', None)

    if rng is None:
        logger.debug("No index_key_rng.  Use base[rng]")
        rng = base.get('rng',None)

    if rng is None and logger:
        # Only report the warning the first time.
        rng_tag = tag + '_reported_no_rng'
        if rng_tag not in base:
            base[rng_tag] = True
            logger.warning("No base['rng'] available for %s.  Using /dev/urandom.",tag)

    return rng

def CleanConfig(config, keep_current=False):
    """Return a "clean" config dict without any leading-underscore values

    GalSim config dicts store a lot of ancillary information internally to help improve
    efficiency.  However, some of these are actually pointers to other places in the dict, so
    printing a config dict, or even what should be a small portion of one, can have infinite loops.

    This helper function is useful when debugging config processing to strip out all of these
    leading-underscore values, so that printing the dict is reasonable.

        >>> print(galsim.config.CleanConfig(config_dict))
    """
    if isinstance(config, dict):
        return { k : CleanConfig(config[k], keep_current) for k in config
                 if k[0] != '_' and (keep_current or k != 'current') }
    elif isinstance(config, list):
        return [ CleanConfig(item, keep_current) for item in config ]
    else:
        return config
