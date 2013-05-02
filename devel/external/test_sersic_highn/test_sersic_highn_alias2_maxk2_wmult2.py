#!/usr/bin/env python

import os
import logging
import test_sersic_highn_basic

# Start off with the basic config
config = test_sersic_highn_basic.config_basic
config['image']['gsparams']['alias_threshold'] = 2.5e-3
config['image']['gsparams']['maxk_threshold'] = 5.e-4
config['image']['gsparams']['wmult'] = 2.

# Output filename
outfile = os.path.join(
    "outputs", "sersic_highn_alias2_maxk2_wmult2_output_N"+str(test_sersic_highn_basic.NOBS)+".pkl")

# Setup the logging
logging.basicConfig(level=test_sersic_highn_basic.LOGLEVEL) 
logger = logging.getLogger("sersic_highn_alias2_maxk2_wmult2")

random_seed = 912424534

test_sersic_highn_basic.run_tests(random_seed, outfile, config=config, logger=logger)
