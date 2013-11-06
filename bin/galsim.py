# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
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
    description = "galsim: configuration file parser for GalSim.\n" + version_str
    epilog = "Works with both YAML and JSON markup formats."
    
    try:
        import argparse
        
        # Build the parser and add arguments
        parser = argparse.ArgumentParser(description=description, add_help=True, epilog=epilog)
        parser.add_argument('config_file', type=str, nargs='*', help='the configuration file(s)')
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

        if len(args.config_file) == 0:
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
              [--version] config_file [config_file ...]"""
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
            args.config_file = posargs
            if args.version:
                print version_str

    # Return the args
    return args


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
    
    # Determine the file type from the extension if necessary:
    if args.file_type is None:
        import os
        name, ext = os.path.splitext(args.config_file[0])
        if ext.lower().startswith('.j'):
            args.file_type = 'json'
        else:
            # Let YAML be the default if the extension is not .y* or .j*.
            args.file_type = 'yaml'
        logger.debug('File type determined to be %s', args.file_type)
    else:
        logger.debug('File type specified to be %s', args.file_type)

    for config_file in args.config_file:
        logger.warn('Using config file %s', config_file)
    
        if args.file_type == 'yaml':
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

        else:
            import json

            with open(config_file) as f:
                config = json.load(f)

            # JSON files are just processed as is.  This is equivalent to having an empty 
            # base_config, so we just do that and use the same structure.
            base_config = {}
            all_config = [ config ]
            
        logger.debug('Successfully read in config file.')

        # Set the root value in base_config
        if 'root' not in base_config:
            import os
            base_config['root'] = os.path.splitext(config_file)[0]

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

            logger.debug("Process config dict: \n%s", config)

            # Process the configuration
            galsim.config.Process(config, logger)


if __name__ == "__main__":
    main()
