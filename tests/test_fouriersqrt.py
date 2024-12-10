# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

import numpy as np
import os

import galsim
from galsim_test_helpers import *

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images.


@timer
def test_fourier_sqrt():
    """Test that the FourierSqrt operator is the inverse of auto-convolution.
    """
    dx = 0.4
    myImg1 = galsim.ImageF(80,80, scale=dx)
    myImg1.setCenter(0,0)
    myImg2 = galsim.ImageF(80,80, scale=dx)
    myImg2.setCenter(0,0)

    # Test trivial case, where we could (but don't) analytically collapse the
    # chain of profiles by recognizing that FourierSqrt is the inverse of
    # AutoConvolve.
    psf = galsim.Moffat(beta=3.8, fwhm=1.3, flux=5)
    psf.drawImage(myImg1, method='no_pixel')
    sqrt1 = galsim.FourierSqrt(psf)
    psf2 = galsim.AutoConvolve(sqrt1)
    np.testing.assert_almost_equal(psf.stepk, psf2.stepk)
    psf2.drawImage(myImg2, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Moffat sqrt convolved with self disagrees with original")

    check_basic(sqrt1, "FourierSqrt", do_x=False)

    # Test non-trivial case where we compare (in Fourier space) sqrt(a*a + b*b + 2*a*b) against (a + b)
    a = galsim.Moffat(beta=3.8, fwhm=1.3, flux=5)
    a.shift(dx=0.5, dy=-0.3)  # need nonzero centroid to test
    b = galsim.Moffat(beta=2.5, fwhm=1.6, flux=3)
    check = galsim.Sum([a, b])
    sqrt = galsim.FourierSqrt(
        galsim.Sum([
            galsim.AutoConvolve(a),
            galsim.AutoConvolve(b),
            2*galsim.Convolve([a, b])
        ])
    )
    np.testing.assert_almost_equal(check.stepk, sqrt.stepk)
    check.drawImage(myImg1, method='no_pixel')
    sqrt.drawImage(myImg2, method='no_pixel')
    np.testing.assert_almost_equal(check.centroid.x, sqrt.centroid.x)
    np.testing.assert_almost_equal(check.centroid.y, sqrt.centroid.y)
    np.testing.assert_almost_equal(check.flux, sqrt.flux)
    np.testing.assert_almost_equal(check.xValue(check.centroid), check.max_sb)
    print('check.max_sb = ',check.max_sb)
    print('sqrt.max_sb = ',sqrt.max_sb)
    # This isn't super accurate...
    np.testing.assert_allclose(check.max_sb, sqrt.max_sb, rtol=0.1)
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Fourier square root of expanded square disagrees with original")

    # Check picklability
    check_pickle(sqrt1, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(sqrt1)

    # Should raise an exception for invalid arguments
    assert_raises(TypeError, galsim.FourierSqrt)
    assert_raises(TypeError, galsim.FourierSqrt, myImg1)
    assert_raises(TypeError, galsim.FourierSqrt, [psf])
    assert_raises(TypeError, galsim.FourierSqrt, psf, psf)
    assert_raises(TypeError, galsim.FourierSqrt, psf, real_space=False)
    assert_raises(TypeError, galsim.FourierSqrtProfile)
    assert_raises(TypeError, galsim.FourierSqrtProfile, myImg1)
    assert_raises(TypeError, galsim.FourierSqrtProfile, [psf])
    assert_raises(TypeError, galsim.FourierSqrtProfile, psf, psf)
    assert_raises(TypeError, galsim.FourierSqrtProfile, psf, real_space=False)

    assert_raises(NotImplementedError, sqrt1.xValue, galsim.PositionD(0,0))
    assert_raises(NotImplementedError, sqrt1.drawReal, myImg1)
    assert_raises(NotImplementedError, sqrt1.shoot, 1)

if __name__ == "__main__":
    runtests(__file__)
