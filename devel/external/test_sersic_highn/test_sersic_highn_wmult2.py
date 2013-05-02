#!/usr/bin/env python

import os
import logging
import test_sersic_highn_basic

# Start off with the basic config
config = test_sersic_highn_basic.config_basic
config['image']['gsparams']['wmult'] = 2.

# Output filename
outfile = os.path.join(
    "outputs", "sersic_highn_wmult2_output_N"+str(test_sersic_highn_basic.NOBS)+".pkl")

# Setup the logging
logging.basicConfig(level=test_sersic_highn_basic.LOGLEVEL) 
logger = logging.getLogger("sersic_highn_wmult2")

random_seed = 912424534

test_sersic_highn_basic.run_tests(random_seed, outfile, config=config, logger=logger)
