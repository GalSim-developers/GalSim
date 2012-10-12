"""
The main driver program for making images of galaxies whose parameters are specified
in a configuration file.
"""

import sys
import os
import galsim
import logging
import copy
import yaml

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
    description = "galsim_yaml: YAML configuration file parser for GalSim"
    epilog = "For JSON-formatted configuration files, use galsim_json"
    
    try:
        import argparse
        
        # Build the parser and add arguments
        parser = argparse.ArgumentParser(description=description, add_help=True, epilog=epilog)
        parser.add_argument('config_file', type=str, help='the YAML configuration file')
        parser.add_argument(
            '-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3),
            help='integer verbosity level: min=0, max=3 [default=2]')
        parser.add_argument(
            '-l', '--log_file', type=str, action='store', default=None,
            help='filename for storing logging output [default is to stream to stdout]')
        args = parser.parse_args()

    except ImportError:
        # Use optparse instead
        import optparse

        # Usage string not automatically generated for optparse, so generate it
        usage = "Usage: galsim_yaml [-h] [-v {0,1,2,3}] [-l LOG_FILE] config_file"
        # Build the parser
        parser = optparse.OptionParser(usage=usage, epilog=epilog, description=description)
        # optparse only allows string choices, so take verbosity as a string and make it int later
        parser.add_option(
            '-v', '--verbosity', type="choice", action='store', choices=('0', '1', '2', '3'),
            default='2', help='integer verbosity level: min=0, max=3 [default=2]')
        parser.add_option(
            '-l', '--log_file', type=str, action='store', default=None,
            help='filename for storing logging output [default is to stream to stdout]')
        (options, posargs) = parser.parse_args()

        # Since optparse doesn't put all the positional arguments together with the options,
        # make a galsim.AttributeDict() (functionally very similar to an optparse.Values instance
        # such as options) to store everything.
        args = galsim.utilities.AttributeDict()
        args.verbosity = int(options.verbosity) # remembering to convert to an integer type
        args.log_file = options.log_file
        # Parse the positional arguments by hand
        if len(posargs) == 1:
            args.config_file = posargs[0]
        else:
            print usage
            if len(posargs) == 0:
                sys.exit('galsim_yaml: error: too few arguments')
            else:
                argstring = posargs[1]
                for addme in posargs[2:]:
                    argstring = argstring+' '+addme
                sys.exit('galsim_yaml: error: unrecognised arguments: '+argstring)

    # Return the args
    return args


def main():
    args = parse_args()

    # Parse the integer verbosity level from the command line args into a logging_level string
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
    logger = logging.getLogger('galsim_yaml')
    
    config_file = args.config_file
    logger.info('Using config file %s', config_file)

    all_config = [ c for c in yaml.load_all(open(config_file).read()) ]
    logger.debug('Successfully read in config file.')

    # If there is only 1 yaml document, then it is of course used for the configuration.
    # If there are multiple yamls documents, then the first one defines a common starting
    # point for the later documents.
    # So the configurations are taken to be:
    #   all_cong[0] + allconfig[1]
    #   all_cong[0] + allconfig[2]
    #   all_cong[0] + allconfig[3]
    #   ...
    # See demo6.yaml and demo8.yaml in the examples directory for examples of this feature.

    if len(all_config) == 1:
        # If we only have 1, prepend an empty "base_config"
        all_config = [{}] + all_config

    base_config = all_config[0]

    # Set the root value in base_config
    if 'root' not in base_config:
        base_config['root'] = os.path.splitext(config_file)[0]

    for config in all_config[1:]:

        # Merge the base_config information into this config file.
        MergeConfig(config,base_config)

        logger.debug("Process config dict: \n%s", config)

        # Process the configuration
        galsim.config.Process(config, logger)


if __name__ == "__main__":
    main()
