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

def main(argv):

    if len(argv) < 2: 
        print 'Usage: galsim_yaml config_file'
        sys.exit("No configuration file specified")

    # TODO: Should have a nice way of specifying a verbosity level...
    # Then we can just pass that verbosity into the logger.
    # Can also have the logging go to a file, etc.
    #  -- Note: should use optparse rather than argparse, since we want it to work for 
    #           python 2.6, and we probably don't need any of the extra features that 
    #           argparse provides.
    # But for now, just do a basic setup.
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        stream=sys.stdout
    )
    logger = logging.getLogger('galsim_yaml')

    # To turn off logging:
    #logger.propagate = False

    config_file = argv[1]
    logger.info('Using config file %s',config_file)

    all_config = [ c for c in yaml.load_all(open(config_file).read()) ]
    logger.info('Successfully read in config file.')
    #print 'all_config = ',all_config

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
        #print 'config = ',config

        # Process the configuration
        galsim.config.Process(config, logger)
    
if __name__ == "__main__":
    main(sys.argv)
