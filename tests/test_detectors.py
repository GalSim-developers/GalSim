# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
"""Unit tests for inclusion of detector effects (nonlinearity, etc.).
"""

import numpy as np
from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    import os
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

DECIMAL = 14

def test_nonlinearity_basic():
    """Check for overall sensible behavior of the nonlinearity routine."""
    import time
    t1 = time.time()

    # Make an image with non-trivially interesting scale and bounds.
    g = galsim.Gaussian(sigma=3.7)
    im = g.draw(scale=0.25)
    im.shift(dx=-5, dy=3)
    im_save = im.copy()

    # Basic - exceptions / bad usage (invalid function, does not return NumPy array).
    try:
        np.testing.assert_raises(ValueError, im.applyNonlinearity, lambda x : 1.0)
    except ImportError:
        print 'The assert_raises tests require nose'

    # Check for constant function as NLfunc
    im_new = im.copy()
    im_new.applyNonlinearity(lambda x: x**0.0)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_new.array, np.ones_like(im),
        err_msg='Image not constant when the nonlinearity function is constant')

    # Check that calling a NLfunc with no parameter works
    NLfunc = lambda x: x + 0.001*(x**2)
    im_new = im.copy()
    im_new.applyNonlinearity(NLfunc)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_new.array, im.array+0.001*((im.array)**2),
        err_msg = 'Nonlinearity function with no argument does not function as desired.')

    # Check that calling a NLfunc with a parameter works
    NLfunc = lambda x, beta: x + beta*(x**2)
    im_new = im.copy()
    im_new.applyNonlinearity(NLfunc, 0.001)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_new.array, im.array + 0.001*((im.array)**2),
        err_msg = 'Nonlinearity function with one argument does not function as desired.')

    # Check that calling a NLfunc with multiple parameters works
    NLfunc = lambda x, beta1, beta2: x + beta1*(x**2) + beta2*(x**3)
    im_new = im.copy()
    im_new.applyNonlinearity(NLfunc, 0.001, -0.0001)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_new.array, im.array + 0.001*((im.array)**2) -0.0001*((im.array)**3),
        err_msg = 'Nonlinearity function with multiple arguments does not function as desired')

    # Check for preservation for identity function as NLfunc.
    im_new = im.copy()
    im_new.applyNonlinearity(lambda x : x)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_new.array, im.array,
        err_msg='Image not preserved when applying identity nonlinearity function')

    # Check that lambda func vs. LookupTable agree.
    max_val = np.max(im.array)
    x_vals = np.linspace(0.0, max_val, num=500)
    f_vals = x_vals + 0.1*(x_vals**2)
    lut = galsim.LookupTable(x=x_vals, f=f_vals)
    im1 = im.copy()
    im2 = im.copy()
    im1.applyNonlinearity(lambda x : x + 0.1*(x**2))
    im2.applyNonlinearity(lut)
    
    assert im1.scale == im.scale
    assert im1.wcs == im.wcs
    assert im1.dtype == im.dtype
    assert im1.bounds == im.bounds

    assert im2.scale == im.scale
    assert im2.wcs == im.wcs
    assert im2.dtype == im.dtype
    assert im2.bounds == im.bounds
    # Note, don't be quite as stringent as in previous test; there can be small interpolation
    # errors.
    np.testing.assert_array_almost_equal(
        im1.array, im2.array, int(0.5*DECIMAL),
        err_msg='Image differs when using LUT vs. lambda function')

    # Check that lambda func vs. interpolated function from SciPy agree
    # GalSim doesn't have SciPy dependence and this is NOT the preferred way to construct smooth
    # functions from tables but our routine can handle it anyway.
    try:
        from scipy import interpolate
        max_val = np.max(im.array)
        x_vals = np.linspace(0.0,max_val,num=500)
        y_vals = x_vals + 0.1*(x_vals**2)
        lut = interpolate.interp1d(x=x_vals,y=y_vals)
        im1 = im.copy()
        im2 = im.copy()
        im1.applyNonlinearity(lambda x: x + 0.1*(x**2))
        im2.applyNonlinearity(lut)

        assert im1.scale == im.scale
        assert im1.wcs == im.wcs
        assert im1.dtype == im.dtype
        assert im1.bounds == im.bounds

        assert im2.scale == im.scale
        assert im2.wcs == im.wcs
        assert im2.dtype == im.dtype
        assert im2.bounds == im.bounds

        #Let the user know that this test happened
        print "SciPy was found installed. Using SciPy modules in the unit test for",
        "'applyNonlinearity'"
        # Note, don't be quite as stringent as in previous test; there can be small interpolation
        # errors.
        np.testing.assert_array_almost_equal(
            im1.array, im2.array, int(0.5*DECIMAL),
            err_msg="Image differs when using SciPy's interpolation vs. lambda function")
    except:
        pass
        # GalSim doesn't have SciPy dependence. So if SciPy is not installed, then this test is
        # skipped. The user is not alerted.

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_recipfail_basic():
    """Check for overall sensible behavior of the reciprocity failure routine."""
    import time
    t1 = time.time()

    # Make an image with non-trivially interesting scale and bounds.
    g = galsim.Gaussian(sigma=3.7)
    im = g.draw(scale=0.25)
    im.shift(dx=-5, dy=3)
    im_save = im.copy()

    # Basic - exceptions / bad usage.
    try:
        np.testing.assert_raises(ValueError, im.addReciprocityFailure, -1.0, 200, 1.0)
    except ImportError:
        print 'The assert_raises tests require nose'

    # Preservation of data type / scale / bounds
    im_new = im.copy()
    im_new.addReciprocityFailure(exp_time=200., alpha=0.0065, base_flux=1.0)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_save.array, im.array,
        err_msg = 'Input image was modified after addition of reciprocity failure')

    # Check for preservation for certain alpha.
    im_new = im.copy()
    im_new.addReciprocityFailure(exp_time=200., alpha=0.0, base_flux=1.0)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_new.array, im.array,
        err_msg='Image not preserved when applying null reciprocity failure')

    # Check for proper scaling with alpha
    alpha1 = 0.006
    alpha2 = 0.007
    im1 = im.copy()
    im2 = im.copy()
    im1.addReciprocityFailure(exp_time=200.,alpha=alpha1, base_flux=1.0)
    im2.addReciprocityFailure(exp_time=200.,alpha=alpha2, base_flux=1.0)
    dim1 = im1.array/im.array
    dim2 = im2.array/im.array
    # We did new / old image, which should equal (old_image/normalization)^alpha.
    # The old image is the same, as is the factor inside the log.  So the log ratio log(dim2)/log(
    # dim1) should just be alpha2/alpha1
    expected_ratio = np.zeros(im.array.shape) + (alpha2/alpha1)

    assert im1.scale == im.scale
    assert im1.wcs == im.wcs
    assert im1.dtype == im.dtype
    assert im1.bounds == im.bounds

    assert im2.scale == im.scale
    assert im2.wcs == im.wcs
    assert im2.dtype == im.dtype
    assert im2.bounds == im.bounds

    np.testing.assert_array_almost_equal(
        np.log(dim2)/np.log(dim1), expected_ratio, int(DECIMAL/3),
        err_msg='Did not get expected change in reciprocity failure when varying alpha')

    #Check math is right
    alpha, exp_time, base_flux = 0.0065, 10.0, 5.0
    im_new = im.copy()
    im_new.addReciprocityFailure(alpha=alpha, exp_time=exp_time, base_flux=base_flux)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_almost_equal(
        (np.log10(im_new.array)-np.log10(im.array)), (alpha/np.log(10))*np.log10(im.array/ \
            (exp_time*base_flux))
        ,int(DECIMAL/3), err_msg='Difference in images is not alpha times the log of original')

    # Check power law against logarithmic behavior
    alpha, exp_time, base_flux = 0.0065, 2.0, 4.0
    im_new = im.copy()
    im_new.addReciprocityFailure(alpha=alpha, exp_time=exp_time, base_flux=base_flux)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_almost_equal(
        im_new.array,im.array*(1+alpha*np.log10(im.array/exp_time)),int(DECIMAL/3),
        err_msg='Difference between power law and log behavior')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2-t1)


if __name__ == "__main__":
    test_nonlinearity_basic()
    test_recipfail_basic()
