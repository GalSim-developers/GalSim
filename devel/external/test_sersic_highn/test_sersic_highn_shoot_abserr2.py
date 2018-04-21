# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

import os
import logging
import test_sersic_highn_basic

# Start off with the basic config
config = test_sersic_highn_basic.config_basic
config['image']['gsparams']['shoot_abserr'] = 5.e-9

# Output filename
if not os.path.isdir("outputs"):
    os.mkdir("outputs")
outfile = os.path.join(
    "outputs", "sersic_highn_shoot_abserr2_output_N"+str(test_sersic_highn_basic.NOBS)+".asc")

# Setup the logging
logging.basicConfig(level=test_sersic_highn_basic.LOGLEVEL) 
logger = logging.getLogger("sersic_highn_shoot_abserr2")

random_seed = 912424534

test_sersic_highn_basic.run_tests(
    random_seed, outfile, config=config, logger=logger,
    fail_value=test_sersic_highn_basic.FAIL_VALUE)
