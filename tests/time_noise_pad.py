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

"""Unit tests for the InterpolatedImage class.
"""

from __future__ import print_function
import numpy as np
import os
import sys

n_iter = 50

import galsim

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_corr_padding_cf():
    """Test for correlated noise padding of InterpolatedImage."""
    import time
    t1 = time.time()

    imgfile = 'fits_files/blankimg.fits'
    orig_nx = 147
    orig_ny = 124
    orig_seed = 151241
    rng = galsim.BaseDeviate(orig_seed)

    # Make an ImageCorrFunc
    cf = galsim.CorrelatedNoise(rng, galsim.fits.read(imgfile))

    # first, make the base image
    orig_img = galsim.ImageF(orig_nx, orig_ny, scale=1.)
    gal = galsim.Gaussian(sigma=2.5, flux=100.)
    gal.drawImage(orig_img, method='no_pixel')

    for iter in range(n_iter):
        # make it into an InterpolatedImage padded with cf
        int_im = galsim.InterpolatedImage(orig_img, noise_pad=cf)

        # do it again with a particular seed
        int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                          noise_pad = cf)

        # repeat
        int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                          noise_pad = cf)

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


def test_corr_padding_im():
    """Test for correlated noise padding of InterpolatedImage."""
    import time
    t1 = time.time()

    imgfile = 'fits_files/blankimg.fits'
    orig_nx = 147
    orig_ny = 124
    orig_seed = 151241

    # Make an Image
    im = galsim.fits.read(imgfile)

    # first, make the base image
    orig_img = galsim.ImageF(orig_nx, orig_ny, scale=1.)
    gal = galsim.Gaussian(sigma=2.5, flux=100.)
    gal.drawImage(orig_img, method='no_pixel')

    for iter in range(n_iter):
        # make it into an InterpolatedImage padded with im
        int_im = galsim.InterpolatedImage(orig_img, noise_pad=im)

        # do it again with a particular seed
        int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                          noise_pad = im)

        # repeat
        int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                          noise_pad = im)

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


def test_corr_padding_imgfile():
    """Test for correlated noise padding of InterpolatedImage."""
    import time
    t1 = time.time()

    imgfile = 'fits_files/blankimg.fits'
    orig_nx = 147
    orig_ny = 124
    orig_seed = 151241

    # Make an Image

    # first, make the base image
    orig_img = galsim.ImageF(orig_nx, orig_ny, scale=1.)
    gal = galsim.Gaussian(sigma=2.5, flux=100.)
    gal.drawImage(orig_img, method='no_pixel')

    for iter in range(n_iter):
        # make it into an InterpolatedImage padded with imgfile
        int_im = galsim.InterpolatedImage(orig_img, noise_pad=imgfile)

        # do it again with a particular seed
        int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                          noise_pad = imgfile)

        # repeat
        int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                          noise_pad = imgfile)

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))

def test_corr_nopadding():
    import time
    t1 = time.time()

    imgfile = 'fits_files/blankimg.fits'
    orig_nx = 147
    orig_ny = 124
    orig_seed = 151241

    # first, make the base image
    orig_img = galsim.ImageF(orig_nx, orig_ny, scale=1.)
    gal = galsim.Gaussian(sigma=2.5, flux=100.)
    gal.drawImage(orig_img, method='no_pixel')

    for iter in range(n_iter):
        # make it into an InterpolatedImage padded with imgfile
        int_im = galsim.InterpolatedImage(orig_img)

        # do it again with a particular seed
        int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed))

        # repeat
        int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed))

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))

if __name__ == "__main__":
    test_corr_padding_cf()
    test_corr_padding_im()
    test_corr_padding_imgfile()
    test_corr_nopadding()
