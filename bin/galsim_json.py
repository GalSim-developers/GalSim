"""
The main driver program for making images of galaxies whose parameters are specified
in a configuration file.
"""

import sys
import os
import galsim
import logging
import json

def parse_args():
    """Handle the command line arguments using either argparse (if available) or optparse.
    """

    # Short description strings common to both parsing mechanisms
    description = "galsim_json: JSON configuration file parser for GalSim"
    epilog = "For YAML-formatted configuration files, use galsim_yaml"
    
    try:
        import argparse
        
        # Build the parser and add arguments
        parser = argparse.ArgumentParser(description=description, add_help=True, epilog=epilog)
        parser.add_argument(
            'config_file', type=str, nargs='+', help='the JSON configuration file(s)')
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
        usage = """Usage: galsim_json [-h] [-v {0,1,2,3}] [-l LOG_FILE]
                   config_file [config_file ...]
        """
        
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
        if len(posargs) >= 1:
            args.config_file = posargs
        else:
            print usage
            sys.exit('galsim_json: error: too few arguments')

    # Return the args
    return args

def main():
    args = parse_args()

    # Parse the integer verbosity level from the commandl ine args into a logging_level string
    logging_level = {0: "CRITICAL", 1: "WARNING", 2: "INFO", 3: "DEBUG"}[args.verbosity]

    # Setup logging to go to sys.stdout or (if requested) to an output file
    if args.log_file is None:
        logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    else:
        logging.basicConfig(format="%(message)s", level=logging_level, filename=args.log_file)
    logger = logging.getLogger('galsim_json')
    
    # To turn off logging:
    #logger.propagate = False

    for config_file in args.config_file:
        logger.info('Using config file %s', config_file)

        config = json.load(open(config_file))
        logger.info('Successfully read in config file.')

        # Set the root value
        if 'root' not in config:
            config['root'] = os.path.splitext(config_file)[0]

        # Process the configuration
        galsim.config.Process(config, logger)


if __name__ == "__main__":
    main()
