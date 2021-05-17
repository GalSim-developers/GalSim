# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
import logging
import copy
import json
from collections import OrderedDict

# Lots of function that used to be here are now in util, so import them back here in case users
# were using them as galsim.config.process.*.  But having them in util means that we can safely
# import from .util in other files without triggering a circular import cycle.
from .util import *

from .value import ParseValue
from .output import GetNFiles, BuildFiles
from ..utilities import SimpleGenerator
from ..random import BaseDeviate
from ..errors import GalSimConfigError, GalSimConfigValueError, GalSimValueError

top_level_fields = ['psf', 'gal', 'stamp', 'image', 'input', 'output',
                    'eval_variables', 'root', 'modules', 'profile']

rng_fields = ['rng', 'obj_num_rng', 'image_num_rng', 'file_num_rng',
              'obj_num_rngs', 'image_num_rngs', 'file_num_rngs']

valid_index_keys = [ 'obj_num_in_file', 'obj_num', 'image_num', 'file_num' ]


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

    Parameters:
        config_file:    The name of the configuration file to read.
        file_type:      If given, the type of file to read.  [default: None, which mean
                        infer the file type from the extension.]
        logger:         If given, a logger object to log progress. [default: None]

    Returns:
        list of config dicts
    """
    logger = LoggerWrapper(logger)
    logger.warning('Reading config file %s', config_file)
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
        config = ReadYaml(config_file)
    else:
        logger.info('Reading JSON config file %s', config_file)
        config = ReadJson(config_file)

    ConvertNones(config)
    logger.debug('Successfully read in config file.')

    return config

def ImportModules(config, gdict=None):
    """Import any modules listed in config['modules'].

    These won't be brought into the running scope of the config processing, but any side
    effects of the import statements will persist.  In particular, these are allowed to
    register additional custom types that can then be used in the current config dict.

    Parameters:
        config:     The configuration dict.
    """
    import sys
    from importlib import import_module
    if gdict is None:
        gdict = globals()
    if 'modules' in config:
        for module in config['modules']:
            try:
                gdict[module] = import_module(module)
            except ImportError:
                # Try adding '.' to path, in case loading a local module and '.' not present.
                if '.' not in sys.path:
                    sys.path.append('.')
                    gdict[module] = import_module(module)
                else:
                    raise

def ProcessTemplate(config, base, logger=None):
    """If the config dict has a 'template' item, read in the appropriate file and
    make any requested updates.

    Parameters:
        config:         The configuration dict.
        base:           The base configuration dict.
        logger:         If given, a logger object to log progress. [default: None]
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

    Parameters:
        config:         The configuration dict.
        logger:         If given, a logger object to log progress. [default: None]
        base:           The base configuration dict. [default: None]
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

    Parameters:
        config:         The configuration dict.
        logger:         If given, a logger object to log progress. [default: None]
        njobs:          The total number of jobs to split the work into. [default: 1]
        job:            Which job should be worked on here (1..njobs). [default: 1]
        new_params:     A dict of new parameter values that should be used to update the config
                        dict after any template loading (if any). [default: None]
        except_abort:   Whether to abort processing when a file raises an exception (True)
                        or just report errors and continue on (False). [default: False]

    Returns:
        the final config dict that was used.
    """
    logger = LoggerWrapper(logger)
    if njobs < 1:
        raise GalSimValueError("Invalid number of jobs",njobs)
    if job < 1:
        raise GalSimValueError("Invalid job number.  Must be >= 1.",job)
    if job > njobs:
        raise GalSimValueError("Invalid job number.  Must be <= njobs (%d)"%(njobs),job)

    # First thing to do is deep copy the input config to make sure we don't modify the original.
    config = CopyConfig(config)

    # Process any template specifications in the dict.
    ProcessAllTemplates(config, logger)

    # Update using any new_params that are given:
    if new_params is not None:
        UpdateConfig(config, new_params)

    # Import any modules if requested
    ImportModules(config)

    logger.debug("Final config dict to be processed: \n%s",
                 json.dumps(config, default=lambda o: repr(o), indent=4))

    # Warn about any unexpected fields.
    unexpected = [ k for k in config if k not in top_level_fields ]
    if len(unexpected) > 0 and logger:
        logger.warning("Warning: config dict contains the following unexpected fields: %s.",
                       unexpected)
        logger.warning("These fields are not (directly) processed by the config processing.")

    # Determine how many files we will be processing in total.
    # Usually, this is just output.nfiles, but different output types may define this differently.
    nfiles = GetNFiles(config, logger=logger)
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
    config_out = BuildFiles(nfiles, config, file_num=start, logger=logger,
                            except_abort=except_abort)
    #Return config_out in case useful
    return config_out
