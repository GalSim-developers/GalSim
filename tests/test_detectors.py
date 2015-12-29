# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
import warnings
from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    import os
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def test_nonlinearity_basic():
    """Check for overall sensible behavior of the nonlinearity routine."""
    import time
    t1 = time.time()

    # Make an image with non-trivially interesting scale and bounds.
    g = galsim.Gaussian(sigma=3.7)
    im = g.drawImage(scale=0.25)
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
        im1.array, im2.array, 7,
        err_msg='Image differs when using LUT vs. lambda function')

    # Check that lambda func vs. interpolated function from SciPy agree
    # GalSim doesn't have SciPy dependence and this is NOT the preferred way to construct smooth
    # functions from tables but our routine can handle it anyway.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
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

        # Let the user know that this test happened
        print "SciPy was found installed. Using SciPy modules in the unit test for",\
        "'applyNonlinearity'"
        # Note, don't be quite as stringent as in previous test; there can be small interpolation
        # errors.
        np.testing.assert_array_almost_equal(
            im1.array, im2.array, 7,
            err_msg="Image differs when using SciPy's interpolation vs. lambda function")
    except ImportError:
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
    im = g.drawImage(scale=0.25)
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
        np.log(dim2)/np.log(dim1), expected_ratio, 5,
        err_msg='Did not get expected change in reciprocity failure when varying alpha')

    # Check math is right
    alpha, exp_time, base_flux = 0.0065, 10.0, 5.0
    im_new = im.copy()
    im_new.addReciprocityFailure(alpha=alpha, exp_time=exp_time, base_flux=base_flux)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_almost_equal(
        (np.log10(im_new.array)-np.log10(im.array)), (alpha/np.log(10))*np.log10(im.array/ \
            (exp_time*base_flux)), 5,
        err_msg='Difference in images is not alpha times the log of original')

    # Check power law against logarithmic behavior
    alpha, exp_time, base_flux = 0.0065, 2.0, 4.0
    im_new = im.copy()
    im_new.addReciprocityFailure(alpha=alpha, exp_time=exp_time, base_flux=base_flux)
    assert im_new.scale == im.scale
    assert im_new.wcs == im.wcs
    assert im_new.dtype == im.dtype
    assert im_new.bounds == im.bounds
    np.testing.assert_array_almost_equal(
        im_new.array,im.array*(1+alpha*np.log10(im.array/(exp_time*base_flux))),6,
        err_msg='Difference between power law and log behavior')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2-t1)

def test_quantize():
    """Check behavior of the image quantization routine."""
    import time
    t1 = time.time()

    # Choose a set of types.
    dtypes = [np.float64, np.float32]
    for dtype in dtypes:

        # Set up some array and image with this type.
        arr = np.arange(-10.2,9.8,0.5,dtype=dtype).reshape(5,8)
        image = galsim.Image(arr, scale=0.3, xmin=37, ymin=-14)
        image_q = image.copy()

        # Do quantization.
        image_q.quantize()

        # Check for correctness of numerical values.
        # Note that quantize uses np.round, which rounds x.5 to the nearest even number.
        # cf. http://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html#numpy.around
        # This is different from floor(array+0.5), so don't use x.5 in the test array here.
        # For all other values, the two prescriptions should be equivalent.
        np.testing.assert_array_almost_equal(
            image_q.array, np.floor(image.array+0.5), decimal=8,
            err_msg='Array contents not as expected after quantization, for dtype=%s'%dtype)

        # Make sure quantizing an image with values that are already integers does nothing.
        save_image = image_q.copy()
        image_q.quantize()
        np.testing.assert_array_almost_equal(
            image_q.array, save_image.array, decimal=8,
            err_msg='Array contents of integer-valued array modified by quantization, '+
            'for dtype=%s'%dtype)

        # Check for preservation of WCS etc.
        assert image_q.scale == image.scale
        assert image_q.wcs == image.wcs
        assert image_q.dtype == image.dtype
        assert image_q.bounds == image.bounds

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2-t1)

def test_IPC_basic():
    import time
    t1 = time.time()

    # Make an image with non-trivially interesting scale.
    g = galsim.Gaussian(sigma=3.7)
    im = g.drawImage(scale=0.25)
    im_save = im.copy()

    # Check for no IPC
    ipc_kernel = galsim.Image(3,3)
    ipc_kernel.setValue(2,2,1.0)
    im_new = im.copy()

    im_new.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='extend')
    np.testing.assert_array_equal(
        im_new.array, im.array,
        err_msg="Image is altered for no IPC with edge_treatment = 'extend'" )

    im_new.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='wrap')
    np.testing.assert_array_equal(
        im_new.array, im.array,
        err_msg="Image is altered for no IPC with edge_treatment = 'wrap'" )

    im_new.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='crop')
    np.testing.assert_array_equal(
        im_new.array, im.array,
        err_msg="Image is altered for no IPC with edge_treatment = 'crop'" )

    # Test with a scalar fill_value
    fill_value = np.pi # a non-trivial one
    im_new.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='crop',fill_value=fill_value)

    #Input arrays and output arrays will differ at the edges for this option.
    np.testing.assert_array_equal(
        im_new.array[1:-1,1:-1], im.array[1:-1,1:-1],
        err_msg="Image is altered for no IPC with edge_treatment = 'crop' and with a fill_value" )
    # Check if the edges are filled with fill_value
    np.testing.assert_array_equal(
        im_new.array[0,:], fill_value,
        err_msg="Top edge is not filled with the correct value by applyIPC")
    np.testing.assert_array_equal(
        im_new.array[-1,:], fill_value,
        err_msg="Bottom edge is not filled with the correct value by applyIPC")
    np.testing.assert_array_equal(
        im_new.array[:,0], fill_value,
        err_msg="Left edge is not filled with the correct value by applyIPC")
    np.testing.assert_array_equal(
        im_new.array[:,-1], fill_value,
        err_msg="Left edge is not filled with the correct value by applyIPC")

    # Testing for flux conservation
    np.random.seed(1234)
    ipc_kernel = galsim.Image(abs(np.random.randn(3,3))) # a random kernel
    ipc_kernel /= ipc_kernel.array.sum() # but make it normalized so we do not get warnings
    im_new = im.copy()
    # Set edges to zero since flux is not conserved at the edges otherwise
    im_new.array[0,:] = 0.0
    im_new.array[-1,:] = 0.0
    im_new.array[:,0] = 0.0
    im_new.array[:,-1] = 0.0
    im_new.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='extend', kernel_normalization=True)
    np.testing.assert_almost_equal(im_new.array.sum(), im.array[1:-1,1:-1].sum(), 4,
        err_msg="Normalized IPC kernel does not conserve the total flux for 'extend' option.")

    im_new = im.copy()
    im_new.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='wrap', kernel_normalization=True)
    np.testing.assert_almost_equal(im_new.array.sum(), im.array.sum(), 4,
        err_msg="Normalized IPC kernel does not conserve the total flux for 'wrap' option.")

    # Checking directionality
    ipc_kernel = galsim.Image(3,3)
    ipc_kernel.setValue(2,2,0.875)
    ipc_kernel.setValue(2,3,0.125)
    # This kernel should correspond to each pixel getting contribution from the pixel above it.
    im1 = im.copy()
    im1.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='crop',kernel_normalization=False)
    np.testing.assert_array_almost_equal(0.875*im.array[1:-1,1:-1]+0.125*im.array[2:,1:-1],
        im1.array[1:-1,1:-1], 7, err_msg="Difference in directionality for up kernel in applyIPC")
    # Checking for one pixel in the central bulk
    np.testing.assert_almost_equal(im1(2,2), 0.875*im(2,2)+0.125*im(2,3), 7,
        err_msg="Direction is not as intended for up kernel in applyIPC")

    ipc_kernel = galsim.Image(3,3)
    ipc_kernel.setValue(2,2,0.875)
    ipc_kernel.setValue(1,2,0.125)
    # This kernel should correspond to each pixel getting contribution from the pixel to its left.
    im1 = im.copy()
    im1.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='crop',kernel_normalization=False)
    np.testing.assert_array_almost_equal(im1.array[1:-1,1:-1], im1.array[1:-1,1:-1], 7,
        err_msg="Difference in directionality for left kernel in applyIPC")
    # Checking for one pixel in the central bulk
    np.testing.assert_almost_equal(im1(2,3), 0.875*im(2,3)+0.125*im(2,2), 7,
        err_msg="Direction is not as intended for left kernel in applyIPC")

    # Check using GalSim's native Convolve routine for GSObjects for a realisitic kernel
    ipc_kernel = galsim.Image(np.array(
        [[0.01,0.1,0.01],
         [0.1,1.0,0.1],
         [0.01,0.1,0.01]]))
    ipc_kernel /= ipc_kernel.array.sum()
    ipc_kernel_int = galsim.InterpolatedImage(ipc_kernel,x_interpolant='nearest',scale=im.scale)
    im1 = im.copy()
    im1.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='crop',kernel_normalization=False)
    im2 = im.copy()
    im2_int = galsim.InterpolatedImage(im2,x_interpolant='nearest')
    ipc_kernel_int = galsim.InterpolatedImage(ipc_kernel,x_interpolant='nearest',scale=im.scale)
    im_int = galsim.Convolve(ipc_kernel_int,im2_int,real_space=False)
    im_int.drawImage(im2,method='no_pixel',scale=im.scale)
    np.testing.assert_array_almost_equal(im1.array,im2.array,6,
        err_msg="Output of applyIPC does not match the output from Convolve")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            from scipy import signal
        print "SciPy found installed. Checking IPC kernel convolution against SciPy's `convolve2d`"
        # SciPy is going to emit a warning that we don't want to worry about, so let's deliberately
        # ignore it by going into a `catch_warnings` context.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Generate an arbitrary kernel
            np.random.seed(2345)
            ipc_kernel = galsim.Image(abs(np.random.randn(3,3)))
            ipc_kernel /= ipc_kernel.array.sum()
            # Convolution requires the kernel to be flipped up-down and left-right.
            im_new = im.copy()
            im_new.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='extend',
                            kernel_normalization=False)
            np.testing.assert_array_almost_equal(
                im_new.array, signal.convolve2d(im.array, np.flipud(np.fliplr(ipc_kernel.array)),
                                                mode='same', boundary='fill'), 7,
                err_msg="Image differs from SciPy's result using `mode='same'` and "
                "`boundary='fill`.")

            im_new = im.copy()
            im_new.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='crop',
                            kernel_normalization=False)
            np.testing.assert_array_almost_equal(
                im_new.array[1:-1,1:-1], signal.convolve2d(im.array,
                                                           np.fliplr(np.flipud(ipc_kernel.array)),
                                                           mode='valid', boundary = 'fill'), 7,
                err_msg="Image differs from SciPy's result using `mode=valid'` and "
                "`boundary='fill'`.")

            im_new = im.copy()
            im_new.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='wrap',
                            kernel_normalization=False)
            np.testing.assert_array_almost_equal(
                im_new.array, signal.convolve2d(im.array, np.fliplr(np.flipud(ipc_kernel.array)),
                                                mode='same', boundary='wrap'), 7,
                err_msg="Image differs from SciPy's result using `mode=same'` and "
                "boundary='wrap'`.")

    except ImportError:
        # Skip without any warning if SciPy is not installed
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_nonlinearity_basic()
    test_recipfail_basic()
    test_quantize()
    test_IPC_basic()
