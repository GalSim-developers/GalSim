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

import logging
import numpy as np
import galsim
import test_sersic_highn_basic

# Logging level
logging.basicConfig(level=test_sersic_highn_basic.LOGLEVEL)
logger = logging.getLogger("break_test_sersic_highn_basic")

g1 = -7.82782974302381404086e-03
g2 = -2.65197698285174143784e-01
hlr = 4.22360899999999983567e-01
sersic_n = 1.5
random_seed = 912425495
config = test_sersic_highn_basic.config_basic 

# Start the tests, make the config
config['image']['random_seed'] = random_seed
config['gal'] = {
    "type" : "Sersic" , "n" : sersic_n , "half_light_radius" : hlr ,
    "ellip" : {
        "type" : "G1G2" , "g1" : g1 , "g2" : g2
     }
}
config['psf'] = {"type" : "Airy" , "lam_over_diam" : test_sersic_highn_basic.PSF_LAM_OVER_DIAM }

# Try the test
try:
    results = galsim.utilities.compare_dft_vs_photon_config(
        config, abs_tol_ellip=test_sersic_highn_basic.TOL_ELLIP,
        abs_tol_size=test_sersic_highn_basic.TOL_SIZE, logger=logger)
    test_ran = True
except RuntimeError as err:
    print err
    test_ran = False
    # Uncomment lines below to ouput a check image
    import copy
    checkimage = galsim.config.BuildImage(copy.deepcopy(config))[0] #im = first element
    checkimage.write('junk_'+str(incr_seed)+'.fits')
    pass

