#!/usr/bin/env python

import os
import logging
import test_sersic_highn_basic

# Start off with the basic config
config = test_sersic_highn_basic.config_basic
config['image']['gsparams']['shoot_accuracy'] = 5.e-6

# Output filename
if not os.path.isdir("outputs"):
    os.mkdir("outputs")
outfile = os.path.join(
    "outputs", "sersic_highn_shoot_accuracy2_output_N"+str(test_sersic_highn_basic.NOBS)+".asc")

# Setup the logging
logging.basicConfig(level=test_sersic_highn_basic.LOGLEVEL) 
logger = logging.getLogger("sersic_highn_shoot_accuracy2")

random_seed = 912424534

test_sersic_highn_basic.run_tests(
    random_seed, outfile, config=config, logger=logger,
    fail_value=test_sersic_highn_basic.FAIL_VALUE)
