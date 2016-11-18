# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
import sys
import logging
import math

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


@timer
def test_fits():
    """Test the default output type = Fits
    """
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'output' : {
            'nfiles' : 6,
            'file_name' : "$'output/test_fits_%d.fits'%file_num"
        },
    }

    logger = logging.getLogger('test_single')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

    im1_list = []
    nfiles = 6
    for k in range(nfiles):
        ud = galsim.UniformDeviate(1234 + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        im1 = gal.drawImage(scale=1)
        im1_list.append(im1)

        galsim.config.BuildFile(config, file_num=k, image_num=k, obj_num=k, logger=logger)
        file_name = 'output/test_fits_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1.array)

    # Build all files at once
    galsim.config.RemoveCurrent(config)
    galsim.config.BuildFiles(nfiles, config)
    for k in range(nfiles):
        file_name = 'output/test_fits_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1_list[k].array)

    # Can also use Process to do this
    galsim.config.RemoveCurrent(config)
    galsim.config.Process(config)
    for k in range(nfiles):
        file_name = 'output/test_fits_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1_list[k].array)

    # For the first file, you don't need the file_num.
    os.remove('output/test_fits_0.fits')
    galsim.config.RemoveCurrent(config)
    galsim.config.BuildFile(config)
    im2 = galsim.fits.read('output/test_fits_0.fits')
    np.testing.assert_array_equal(im2.array, im1_list[0].array)

    # If there is no output field, the default behavior is to write to root.fits.
    os.remove('output/test_fits_0.fits')
    del config['output']
    config['root'] = 'output/test_fits_0'
    galsim.config.RemoveCurrent(config)
    galsim.config.BuildFile(config)
    im2 = galsim.fits.read('output/test_fits_0.fits')
    np.testing.assert_array_equal(im2.array, im1_list[0].array)


@timer
def test_skip():
    """Test the skip and noclobber options
    """
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'output' : {
            'nfiles' : 6,
            'file_name' : "$'output/test_skip_%d.fits'%file_num",
            'skip' : { 'type' : 'Random', 'p' : 0.4 }
        },
    }

    im1_list = []
    skip_list = []
    nfiles = 6
    for k in range(nfiles):
        file_name = 'output/test_skip_%d.fits'%k
        if os.path.exists(file_name):
            os.remove(file_name)
        ud_file = galsim.UniformDeviate(1234 + k)
        if ud_file() < 0.4:
            print('skip k = ',k)
            skip_list.append(True)
        else:
            skip_list.append(False)
        ud = galsim.UniformDeviate(1234 + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        im1 = gal.drawImage(scale=1)
        im1_list.append(im1)

    galsim.config.Process(config)
    for k in range(nfiles):
        file_name = 'output/test_skip_%d.fits'%k
        if skip_list[k]:
            assert not os.path.exists(file_name)
        else:
            im2 = galsim.fits.read(file_name)
            np.testing.assert_array_equal(im2.array, im1_list[k].array)

    # Build the ones we skipped using noclobber option
    del config['output']['skip']
    config['output']['noclobber'] = True
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    #print(cl.output)
    assert "Skipping file 1 = output/test_skip_1.fits because output.noclobber" in cl.output
    assert "Skipping file 2 = output/test_skip_2.fits because output.noclobber" in cl.output
    assert "Skipping file 3 = output/test_skip_3.fits because output.noclobber" in cl.output
    assert "Skipping file 5 = output/test_skip_5.fits because output.noclobber" in cl.output
    for k in range(nfiles):
        file_name = 'output/test_skip_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1_list[k].array)


if __name__ == "__main__":
    test_fits()
    test_skip()
