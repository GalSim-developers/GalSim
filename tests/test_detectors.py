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
from scipy import interpolate

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

    ## Basic - exceptions / bad usage (invalid function, does not return NumPy array).
    #try:
    #    np.testing.assert_raises(ValueError, im.applyNonlinearity, lambda x : 1.0)
    #except ImportError:
    #    print 'The assert_raises tests require nose'

    # Preservation of data type / scale / bounds / input image
    im_new = im.copy()
    im_new.applyNonlinearity(lambda x : x + 0.001*(x**2))
    assert im_new.scale == im.scale
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_save.array, im.array,
        err_msg = 'Input image was modified after addition of nonlinearity')

    # Check that calling a NLfunc with no parameter works
    NLfunc = lambda x: x + 0.001*(x**2)
    im_new = im.copy()
    im_new.applyNonlinearity(NLfunc)
    assert im_new.scale == im.scale
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
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_new.array, im.array + 0.001*((im.array)**2) -0.0001*((im.array)**3),
        err_msg = 'Nonlinearity function with multiple arguments does not function as desired')

    # Check for preservation for identity function as NLfunc.
    im_new = im.copy()
    im_new.applyNonlinearity(lambda x : x)
    np.testing.assert_array_equal(
        im_new.array, im.array, err_msg='Image not preserved when applying identity nonlinearity function')

    # Check for constant function as NLfunc
    im_new = im.copy()
    im_new.applyNonlinearity(lambda x: 1.0)
    np.testing.assert_array_equal(im_new.array,np.ones_like(im),err_msg='Image not constant when the nonlinearity function is constant')

    # Check that lambda func vs. LookupTable agree.
    max_val = np.max(im.array)
    x_vals = np.linspace(0.0, max_val, num=500)
    f_vals = x_vals + 0.1*(x_vals**2)
    lut = galsim.LookupTable(x=x_vals, f=f_vals)
    im1 = im.copy()
    im2 = im.copy()
    im1.applyNonlinearity(lambda x : x + 0.1*(x**2))
    im2.applyNonlinearity(lut)
    # Note, don't be quite as stringent as in previous test; there can be small interpolation
    # errors.
    np.testing.assert_array_almost_equal(
        im1.array, im2.array, int(0.5*DECIMAL),
        err_msg='Image differs when using LUT vs. lambda function')

    # Check that lambda func vs. interpolated function from SciPy agree
    # This is NOT the preferred way to construct smooth functions from tables but our routine can handle it anyway.
    max_val = np.max(im.array)
    x_vals = np.linspace(0.0,max_val,num=500)
    f_vals = x_vals + 0.1*(x_vals**2)
    lut = interpolate.interp1d(x=x_vals,y=f_vals)
    im1 = im.copy()
    im2 = im.copy()
    im1.applyNonlinearity(lambda x: x + 0.1*(x**2))
    im2.applyNonlinearity(lut)
    # Note, don't be quite as stringent as in previous test; there can be small interpolation errors.
    np.testing.assert_array_almost_equal(im1.array,im2.array,int(0.5*DECIMAL),err_msg="Image differs when using SciPy's interpolation vs. lambda function")

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

    ## Basic - exceptions / bad usage.
    #try:
    #    np.testing.assert_raises(ValueError, im.addReciprocityFailure, -1.0)
    #except ImportError:
    #    print 'The assert_raises tests require nose'

    # Preservation of data type / scale / bounds
    im_new = im.copy()
    im_new.addReciprocityFailure(exp_time=200.,alpha=0.0065)
    assert im_new.scale == im.scale
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_equal(
        im_save.array, im.array,
        err_msg = 'Input image was modified after addition of reciprocity failure')

    # Check for preservation for certain alpha.
    im_new = im.copy()
    im_new.addReciprocityFailure(exp_time=200.,alpha=0.0)
    np.testing.assert_array_equal(
        im_new.array, im.array, err_msg='Image not preserved when applying null reciprocity failure')

    # Check for proper scaling with alpha
    alpha1 = 0.006
    alpha2 = 0.007
    im1 = im.copy()
    im2 = im.copy()
    im1.addReciprocityFailure(exp_time=200.,alpha=alpha1)
    im2.addReciprocityFailure(exp_time=200.,alpha=alpha2)
    dim1 = im1.array-im.array
    dim2 = im2.array-im.array
    # We did new - old image, which should equal (old image)*alpha*log(...).
    # The old image is the same, as is the factor inside the log.  So the ratio dim2/dim1 should
    # just be alpha2/alpha1.
    expected_ratio = np.zeros(im.array.shape) + alpha2/alpha1
    np.testing.assert_array_almost_equal(
        dim2/dim1, expected_ratio, int(DECIMAL/3),
        err_msg='Did not get expected change in reciprocity failure when varying alpha')

    #Check math is right
    alpha, exp_time = 0.0065, 1.0
    im_new = im.copy()
    im_new.addReciprocityFailure(alpha=alpha,exp_time=exp_time)
    np.testing.assert_array_almost_equal((im_new.array-im.array),alpha*im.array*np.log10(im.array/exp_time),int(DECIMAL/2),err_msg='Difference in images is not alpha times the log of original')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_IPC_basic():
    import time
    t1 = time.time()

    # Make an image with non-trivially interesting scale and bounds.
    g = galsim.Gaussian(sigma=3.7)
    im = g.draw(scale=0.25)
    im.shift(dx=-5, dy=3)
    im_save = im.copy()

    # Check for no IPC
    ipc_kernel = np.zeros((3,3))
    ipc_kernel[1,1] = 1.0
    im_new = im.copy()

    im_new.applyIPC(IPC_kernel=ipc_kernel, edge_effects='extend')
    np.testing.assert_array_equal(im_new.array, im.array, err_msg="Image is altered for no IPC with edge_effects = 'extend'" )

    im_new.applyIPC(IPC_kernel=ipc_kernel, edge_effects='warp')
    np.testing.assert_array_equal(im_new.array, im.array, err_msg="Image is altered for no IPC with edge_effects = 'warp'" )

    im_new.applyIPC(IPC_kernel=ipc_kernel, edge_effects='crop')
    np.testing.assert_array_almost_equal(im_new.array, im.array, err_msg="Image is altered for no IPC with edge_effects = 'crop'" )

    try:
        from scipy import signal
        print "SciPy found installed. Checking IPC kernel convolution against SciPy's `convolve2d`...."
        im_new = im.copy()
        im_new.applyIPC(IPC_kernel=ipc_kernel, edge_effects='extend')
        np.testing/assert_array_almost_equal(im_new, signal.convolve2d(im, ipc_kernel, mode='same', boundary='fill')[1:-1,1:-1], DECIMAL, err_msg="Image differs from SciPy's result using `mode='same'` and `boundary='fill`.")

        im_new = im.copy()
        im_new.applyIPC(IPC_kernel=ipc_kernel, edge_effects='crop')
        np.testing.assert_array_almost_equal(im_new[1:-1], signal.convolve2d(im, ipc_kernel, mode='valid', boundary = 'fill'), DECIMAL, err_msg="Image differs from SciPy's result using `mode=valid'` and `boundary='fill'`.")

        im_new = im.copy()
        im_new.applyIPC(IPC_kernel=ipc_kernel, edge_effects='warp')
        np.testing.assert_array_almost_equal(im_new, signal.convolve2d(im, ipc_kernel, mode='same', boundary='wrap'), DECIMAL, err_msg="Image differs from SciPy's result using `mode=same'` and `boundary='wrap'`.")
    except:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_nonlinearity_basic()
    test_recipfail_basic()
    test_IPC_basic()
