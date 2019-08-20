# Copyright (c) 2012-2019 by the GalSim developers team on GitHub
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
import pprint

from .errors import GalSimError, GalSimValueError, GalSimRangeError

def parse_args():
    """Handle the command line arguments using either argparse (if available) or optparse.
    """
    from ._version import __version__ as version

    # Short description strings common to both parsing mechanisms
    version_str = "GalSim Version %s"%version
    description = "galsim: configuration file parser for %s.  "%version_str
    description += "See https://github.com/GalSim-developers/GalSim/wiki/Config-Documentation "
    description += "for documentation about using this program."
    epilog = "Works with both YAML and JSON markup formats."

    try:
        import argparse

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
        args = parser.parse_args()

        if args.config_file == None:
            if args.version:
                print(version_str)
            else:
                parser.print_help()
            sys.exit()
        elif args.version:
            print(version_str)

    except ImportError:
        # Use optparse instead
        import optparse

        # Usage string not automatically generated for optparse, so generate it
        usage = """usage: galsim [-h] [-v {0,1,2,3}] [-l LOG_FILE] [-f {yaml,json}] [-m MODULE]
              [--version] config_file [variables ...]"""
        # Build the parser
        parser = optparse.OptionParser(usage=usage, epilog=epilog, description=description)
        # optparse only allows string choices, so take verbosity as a string and make it int later
        parser.add_option(
            '-v', '--verbosity', type="choice", action='store', choices=('0', '1', '2', '3'),
            default='1', help='integer verbosity level: min=0, max=3 [default=1]')
        parser.add_option(
            '-l', '--log_file', type=str, action='store', default=None,
            help='filename for storing logging output [default is to stream to stdout]')
        parser.add_option(
            '-f', '--file_type', type="choice", action='store', choices=('yaml','json'),
            default=None,
            help=('type of config_file: yaml or json are currently supported. '
                  '[default is to automatically determine the type from the extension]'))
        parser.add_option(
            '-m', '--module', type=str, action='append', default=None,
            help='python module to import before parsing config file')
        parser.add_option(
            '-p', '--profile', action='store_const', default=False, const=True,
            help='output profiling information at the end of the run')
        parser.add_option(
            '-n', '--njobs', type=int, action='store', default=1,
            help='set the total number of jobs that this run is a part of. ' +
            'Used in conjunction with -j (--job)')
        parser.add_option(
            '-j', '--job', type=int, action='store', default=1,
            help='set the job number for this particular run. Must be in [1,njobs]. '
                 'Used in conjunction with -n (--njobs)')
        parser.add_option(
            '-x', '--except_abort', action='store_const', default=False, const=True,
            help='abort the whole job whenever any file raises an exception rather than '
                 'just reporting the exception and continuing on')
        parser.add_option(
            '--version', action='store_const', default=False, const=True,
            help='show the version of GalSim')
        (args, posargs) = parser.parse_args()

        # Remembering to convert to an integer type
        args.verbosity = int(args.verbosity)

        # Store the positional arguments in the args object as well:
        if len(posargs) == 0:
            if args.version:
                print(version_str)
            else:
                parser.print_help()
            sys.exit()
        else:
            args.config_file = posargs[0]
            args.variables = posargs[1:]
            if args.version:
                print(version_str)

    # Return the args
    return args

def ParseVariables(variables, logger):
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
        except:
            logger.debug('Unable to parse %s.  Treating it as a string.'%value)
        new_params[key] = value

    return new_params


def AddModules(config, modules):
    if modules:
        if 'modules' not in config:
            config['modules'] = modules
        else:
            config['modules'].extend(modules)

def main():
    from .config import ReadConfig, Process

    args = parse_args()

    if args.njobs < 1:
        raise GalSimValueError("Invalid number of jobs", args.njobs)
    if args.job < 1:
        raise GalSimRangeError("Invalid job number.  Must be >= 1", args.job, 1, args.njobs)
    if args.job > args.njobs:
        raise GalSimRangeError("Invalid job number.  Must be <= njobs",args.job, 1, args.njobs)

    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL,
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]

    # If requested, load the profiler
    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    # Setup logging to go to sys.stdout or (if requested) to an output file
    if args.log_file is None:
        logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    else:
        logging.basicConfig(format="%(message)s", level=logging_level, filename=args.log_file)
    logger = logging.getLogger('galsim')

    logger.warning('Using config file %s', args.config_file)
    all_config = ReadConfig(args.config_file, args.file_type, logger)
    logger.debug('Successfully read in config file.')

    # Process each config document
    for config in all_config:

        if 'root' not in config:
            config['root'] = os.path.splitext(args.config_file)[0]

        # Parse the command-line variables:
        new_params = ParseVariables(args.variables, logger)

        # Add modules to the config['modules'] list
        AddModules(config, args.module)

        # Profiling doesn't work well with multiple processes.  We'll need to separately
        # enable profiling withing the workers and output when the process ends.  Set
        # config['profile'] = True to enable this.
        if args.profile:
            config['profile'] = True

        logger.debug("Process config dict: \n%s", pprint.pformat(config))

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
        s = StringIO()
        sortby = 'time'  # Note: This is now called tottime, but time seems to be a valid
                         # alias for this that is backwards compatible to older versions
                         # of pstats.
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby).reverse_order()
        ps.print_stats()
        logger.error(s.getvalue())
