# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

from __future__ import print_function
import numpy as np
import os
import shutil
import sys
import logging

import galsim
from galsim_test_helpers import timer


@timer
def test_input_init():
    """Test using the init method for inputs.
    """
    # Most of the tests in this file write to the 'output' directory.
    # Here we write to a different directory and make sure that it properly
    # creates the directory if necessary.
    if os.path.exists('output_fits'):
        shutil.rmtree('output_fits')
    config = {
        'modules': ['config_input_test_modules'],
        'input': {
            'input_size_module': {'size': 55},
        },
        'image': {
            'type': 'Single',
            'random_seed': 1234,
            'size': "$input_size"
        },
        'gal': {
            'type': 'Gaussian',
            'sigma': {'type': 'Random', 'min': 1, 'max': 2},
            'flux': 100,
        },
        'output': {
            'type': 'Fits',
            'file_name': "output_fits/test_fits.fits"
        },
    }

    logger = logging.getLogger('test_fits')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

    ud = galsim.UniformDeviate(1234 + 0 + 1)
    sigma = ud() + 1.
    gal = galsim.Gaussian(sigma=sigma, flux=100)
    im1 = gal.drawImage(scale=1, nx=55, ny=55)

    galsim.config.Process(config, logger=logger)
    file_name = 'output_fits/test_fits.fits'
    im2 = galsim.fits.read(file_name)
    np.testing.assert_array_equal(im2.array, im1.array)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
