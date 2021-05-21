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
"""
The main driver program for making images of galaxies whose parameters are specified
in a configuration file.
"""

from __future__ import print_function
import sys
import os
import logging
import json
import argparse

from ._version import __version__ as version
from .config import ReadConfig, Process
from .errors import GalSimError, GalSimValueError, GalSimRangeError

def parse_args(command_args):
    """Handle the command line arguments using either argparse (if available) or optparse.
    """
    # Short description strings common to both parsing mechanisms
    version_str = "GalSim Version %s"%version
    description = "galsim: configuration file parser for %s.  "%version_str
    description += "See https://github.com/GalSim-developers/GalSim/wiki/Config-Documentation "
    description += "for documentation about using this program."
    epilog = "Works with both YAML and JSON markup formats."

    # Build the parser and add arguments
    parser = argparse.ArgumentParser(prog='galsim', description=description, add_help=True,
                                     epilog=epilog)
    parser.add_argument('config_file', type=str, nargs='?', help='the configuration file')
    parser.add_argument(
        'variables', type=str, nargs='*',
        help='additional variables or modifications to variables in the config file. '
             'e.g. galsim foo.yaml output.nproc=-1 gal.rotate="{type : Random}"')
    parser.add_argument(
        '-v', '--verbosity', type=int, action='store', default=1, choices=(0, 1, 2, 3),
        help='integer verbosity level: min=0, max=3 [default=1]')
    parser.add_argument(
        '-l', '--log_file', type=str, action='store', default=None,
        help='filename for storing logging output [default is to stream to stdout]')
    parser.add_argument(
        '-f', '--file_type', type=str, action='store', choices=('yaml','json'),
        default=None,
        help='type of config_file: yaml or json are currently supported. '
             '[default is to automatically determine the type from the extension]')
    parser.add_argument(
        '-m', '--module', type=str, action='append', default=None,
        help='python module to import before parsing config file')
    parser.add_argument(
        '-p', '--profile', action='store_const', default=False, const=True,
        help='output profiling information at the end of the run')
    parser.add_argument(
        '-n', '--njobs', type=int, action='store', default=1,
        help='set the total number of jobs that this run is a part of. ' +
        'Used in conjunction with -j (--job)')
    parser.add_argument(
        '-j', '--job', type=int, action='store', default=1,
        help='set the job number for this particular run. Must be in [1,njobs]. '
             'Used in conjunction with -n (--njobs)')
    parser.add_argument(
        '-x', '--except_abort', action='store_const', default=False, const=True,
        help='abort the whole job whenever any file raises an exception rather than '
             'continuing on')
    parser.add_argument(
        '--version', action='store_const', default=False, const=True,
        help='show the version of GalSim')
    args = parser.parse_args(command_args)

    if args.config_file == None:
        if args.version:
            print(version_str)
        else:
            parser.print_help()
        sys.exit()
    elif args.version:
        print(version_str)

    if args.njobs < 1:
        raise GalSimValueError("Invalid number of jobs", args.njobs)
    if args.job < 1:
        raise GalSimRangeError("Invalid job number.  Must be >= 1", args.job, 1, args.njobs)
    if args.job > args.njobs:
        raise GalSimRangeError("Invalid job number.  Must be <= njobs",args.job, 1, args.njobs)

    # Return the args
    return args

def parse_variables(variables, logger):
    """Parse any command-line variables, returning them as a dict
    """
    new_params = {}
    for v in variables:
        logger.debug('Parsing additional variable: %s',v)
        if '=' not in v:
            raise GalSimError('Improper variable specification.  Use field.item=value.')
        key, value = v.split('=',1)
        # Try to evaluate the value string to allow people to input things like
        # gal.rotate='{type : Rotate}'
        # But if it fails (particularly with json), just assign the value as a string.
        try:
            try:
                import yaml
                value = yaml.safe_load(value)
            except ImportError:
                # Don't require yaml.  json usually works for these.
                import json
                value = json.loads(value)
        except Exception as e:
            logger.debug('Caught exception: %s'%e)
            logger.info('Unable to parse %s.  Treating it as a string.'%value)
        new_params[key] = value

    return new_params

def add_modules(config, modules):
    """If modules are given on the command line, add them to the config dict as a modules field.
    """
    if modules:
        if 'modules' not in config:
            config['modules'] = modules
        else:
            config['modules'].extend(modules)

def make_logger(args):
    """Make a logger object according to the command-line specifications.
    """
    # Make a logger
    logger = logging.getLogger('galsim')

    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL,
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    level = logging_levels[args.verbosity]
    logger.setLevel(level)

    # Setup logging to go to sys.stdout or (if requested) to an output file
    if args.log_file is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(level)
    else:
        handler = logging.FileHandler(args.log_file, mode='w')
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(level)
    logger.addHandler(handler)
    return logger

def process_config(all_config, args, logger):
    """Process the config dict according to the command-line specifications.
    """
    # If requested, load the profiler
    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    # Process each config document
    for config in all_config:

        root = os.path.splitext(args.config_file)[0]
        if 'root' not in config:
            config['root'] = root

        # Parse the command-line variables:
        new_params = parse_variables(args.variables, logger)

        # Add modules to the config['modules'] list
        add_modules(config, args.module)

        # Profiling doesn't work well with multiple processes.  We'll need to separately
        # enable profiling withing the workers and output when the process ends.  Set
        # config['profile'] = True to enable this.
        if args.profile:
            config['profile'] = True

        logger.debug("Process config dict: \n%s", json.dumps(config, indent=4))

        # Process the configuration
        Process(config, logger, njobs=args.njobs, job=args.job, new_params=new_params,
                except_abort=args.except_abort)

    if args.profile:
        # cf. example code here: https://docs.python.org/2/library/profile.html
        pr.disable()
        try:
            from StringIO import StringIO
        except ImportError:
            from io import StringIO
        pr.dump_stats(root + '.pstats')
        s = StringIO()
        sortby = 'time'  # Note: This is now called tottime, but time seems to be a valid
                         # alias for this that is backwards compatible to older versions
                         # of pstats.
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby).reverse_order()
        ps.print_stats()
        logger.error(s.getvalue())
        logger.error("Stats file also output to %s for further analysis.",root+'.pstats')

def main(command_args):
    """The whole process given command-line parameters in their native (non-ArgParse) form.
    """
    args = parse_args(command_args)
    logger = make_logger(args)
    all_config = ReadConfig(args.config_file, args.file_type, logger)
    process_config(all_config, args, logger)

def run_main():
    """Kick off the process grabbing the command-line parameters from sys.argv
    """
    main(sys.argv[1:])
