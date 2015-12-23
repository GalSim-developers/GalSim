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
"""
The main driver program for making images of galaxies whose parameters are specified
in a configuration file.
"""

import sys

# The only wrinkle about letting this executable be called galsim is that we want to
# make sure that `import galsim` doesn't import itself.  We want it to import the real
# galsim module of course.  So the solution is to get rid of the current directory
# from python's default search path
temp = sys.path[0]
sys.path = sys.path[1:]
import galsim
# Now put it back in case anyone else relies on this feature.
sys.path = [temp] + sys.path

def MergeConfig(config1, config2, logger=None):
    """
    Merge config2 into config1 such that it has all the information from either config1 or 
    config2 including places where both input dicts have some of a field defined.
    e.g. config1 has image.pixel_scale, and config2 has image.noise.
            Then the returned dict will have both.
    For real conflicts (the same value in both cases), config1's value takes precedence
    """
    for (key, value) in config2.items():
        if not key in config1:
            import copy
            # If this key isn't in config1 yet, just add it
            config1[key] = copy.deepcopy(value)
        elif isinstance(value,dict) and isinstance(config1[key],dict):
            # If they both have a key, first check if the values are dicts
            # If they are, just recurse this process and merge those dicts.
            MergeConfig(config1[key],value)
        else:
            # Otherwise config1 takes precedence
            if logger:
                logger.info("Not merging key %s from the base config, since the later "
                            "one takes precedence",key)
            pass

def parse_args():
    """Handle the command line arguments using either argparse (if available) or optparse.
    """

    # Short description strings common to both parsing mechanisms
    version_str = "GalSim Version %s"%galsim.version
    description = "galsim: configuration file parser for %s.  "%version_str 
    description += "See https://github.com/GalSim-developers/GalSim/wiki/Config-Documentation "
    description += "for documentation about using this program."
    epilog = "Works with both YAML and JSON markup formats."
    
    try:
        import argparse
        
        # Build the parser and add arguments
        parser = argparse.ArgumentParser(description=description, add_help=True, epilog=epilog)
        parser.add_argument('config_file', type=str, nargs='?', help='the configuration file')
        parser.add_argument(
            'variables', type=str, nargs='*',
            help='additional variables or modifications to variables in the config file. ' +
            'e.g. galsim foo.yaml output.nproc=-1 gal.rotate="{type : Random}"')
        parser.add_argument(
            '-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3),
            help='integer verbosity level: min=0, max=3 [default=2]')
        parser.add_argument(
            '-l', '--log_file', type=str, action='store', default=None,
            help='filename for storing logging output [default is to stream to stdout]')
        parser.add_argument(
            '-f', '--file_type', type=str, action='store', choices=('yaml','json'),
            default=None,
            help=('type of config_file: yaml or json are currently supported. ' +
                  '[default is to automatically determine the type from the extension]'))
        parser.add_argument(
            '-m', '--module', type=str, action='append', default=None, 
            help='python module to import before parsing config file')
        parser.add_argument(
            '--version', action='store_const', default=False, const=True,
            help='show the version of GalSim')
        args = parser.parse_args()

        if args.config_file == None:
            if args.version:
                print version_str
            else:
                parser.print_help()
            sys.exit()
        elif args.version:
            print version_str

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
            default='2', help='integer verbosity level: min=0, max=3 [default=2]')
        parser.add_option(
            '-l', '--log_file', type=str, action='store', default=None,
            help='filename for storing logging output [default is to stream to stdout]')
        parser.add_option(
            '-f', '--file_type', type="choice", action='store', choices=('yaml','json'),
            default=None,
            help=('type of config_file: yaml or json are currently supported. ' +
                  '[default is to automatically determine the type from the extension]'))
        parser.add_option(
            '-m', '--module', type=str, action='append', default=None, 
            help='python module to import before parsing config file')
        parser.add_option(
            '--version', action='store_const', default=False, const=True,
            help='show the version of GalSim')
        (args, posargs) = parser.parse_args()

        # Remembering to convert to an integer type
        args.verbosity = int(args.verbosity) 

        # Store the positional arguments in the args object as well:
        if len(posargs) == 0:
            if args.version:
                print version_str
            else:
                parser.print_help()
            sys.exit()
        else:
            args.config_file = posargs[0]
            args.variables = posargs[1:]
            if args.version:
                print version_str

    # Return the args
    return args

def ParseVariables(variables, logger):
    new_params = {}
    for v in variables:
        logger.debug('Parsing additional variable: %s',v)
        if '=' not in v:
            raise ValueError('Improper variable specification.  Use field.item=value.')
        key, value = v.split('=',1)
        # Try to evaluate the value string to allow people to input things like
        # gal.rotate='{type : Rotate}'
        # But if it fails (particularly with json), just assign the value as a string.
        try:
            try:
                import yaml
                value = yaml.load(value)
            except ImportError:
                # Don't require yaml.  json usually works for these.
                import json
                value = json.loads(value)
        except:
            logger.debug('Unable to parse %s.  Treating it as a string.'%value)
        new_params[key] = value

    return new_params


def main():
    args = parse_args()

    # Parse the integer verbosity level from the command line args into a logging_level string
    import logging
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]

    # Setup logging to go to sys.stdout or (if requested) to an output file
    if args.log_file is None:
        logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    else:
        logging.basicConfig(format="%(message)s", level=logging_level, filename=args.log_file)
    logger = logging.getLogger('galsim')
    
    logger.warn('Using config file %s', args.config_file)
    base_config, all_config = galsim.config.ReadConfig(args.config_file, args.file_type, logger)
    logger.debug('Successfully read in config file.')

    # Add the additional variables to the config file
    for v in args.variables:
        logger.debug('Parsing additional variable: %s',v)
        if '=' not in v:
            raise ValueError('Improper variable specification.  Use field.item=value.')
        key, value = v.split('=',1)
        # This next bit is basically identical to the code for Dict.get(key) in catalog.py.
        chain = key.split('.')
        if base_config:
            d = base_config
        else:
            d = all_config[0]
        while chain:
            k = chain.pop(0)
            try: k = int(k)
            except ValueError: pass
            if chain: d = d[k]
            else: 
                # Try to evaluate the value string to allow people to input things like
                # gal.rotate='{type : Rotate}'
                # But if it fails (particularly with json), just assign the value as a string.
                try:
                    if args.file_type == 'yaml':
                        import yaml
                        d[k] = yaml.load(value)
                    else:
                        import json
                        d[k] = json.loads(value)
                except:
                    logger.debug('Unable to parse %s.  Treating it as a string.'%value)
                    d[k] = value

    # Set the root value in base_config
    if 'root' not in base_config:
        import os
        base_config['root'] = os.path.splitext(args.config_file)[0]

    # Import any modules if requested
    if args.module:
        for module in args.module:
            try:
                exec('import galsim.'+module)
            except:
                exec('import '+module)

    # Process each config document
    for config in all_config:

        # Merge the base_config information into this config file.
        MergeConfig(config,base_config)

        # Parse the command-line variables:
        new_params = ParseVariables(args.variables, logger)

        import pprint
        logger.debug("Process config dict: \n%s", pprint.pformat(config))

        # Process the configuration
        galsim.config.Process(config, logger, new_params=new_params)


if __name__ == "__main__":
    main()
