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
import time
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

# Use a deterministic random number generator so we don't fail tests because of rare flukes
# in the random numbers.
rseed=12345

smallim_size = 16 # size of image when we test correlated noise properties using small inputs
smallim_size_odd = 17 # odd-sized version of the above for odd/even relevant tests (e.g. draw)
largeim_size = 12 * smallim_size # ditto, but when we need a larger image
xlargeim_size =long(np.ceil(1.41421356 * largeim_size)) # sometimes, for precision tests, we 
                                                        # need a very large image that will 
                                                        # fit a large image within it, even if 
                                                        # rotated

# Decimals for comparison (one for fine detail, another for comparing stochastic quantities)
decimal_approx = 2
decimal_precise = 7

# Number of positions to test in nonzero lag uncorrelated tests
npos_test = 3

# Number of CorrelatedNoises to sum over to get slightly better statistics for noise generation test
nsum_test = 7

# Set up some variables for testing of noise symmetry.  If we run this test suite directly, do a
# more stringent test that takes more time, otherwise do a looser test that takes less time.
if __name__ == "__main__":
    symm_divide = 1. # test to precision determined by decimal_approx (usually 1%)
    symm_size_mult = 6 # make really huge images
else:
    symm_divide = 3. # test to 3x worse precision than determined by decimal_approx (usually 3%)
    symm_size_mult = 3 # make only moderately huge images


def setup_uncorrelated_noise(deviate, size):
    """Makes and returns uncorrelated noise fields for later use in generating derived correlated
    noise fields.  Field has unit variance.
    """
    gn = galsim.GaussianNoise(deviate, sigma=1.)
    uncorr_noise = galsim.ImageD(size, size)
    uncorr_noise.addNoise(gn)
    return uncorr_noise

def make_xcorr_from_uncorr(uncorr_image):
    """Make some x-correlated noise using shift and add using an input uncorrelated noise field.
    """
    xnoise_image = galsim.ImageD(
        uncorr_image.array + np.roll(uncorr_image.array, 1, axis=1)) # note NumPy thus [y,x]
    xnoise_image *= (np.sqrt(2.) / 2.) # Preserve variance
    return xnoise_image

def make_ycorr_from_uncorr(uncorr_image):
    """Make some y-correlated noise using shift and add using an input uncorrelated noise field.
    """
    ynoise_image = galsim.ImageD(
        uncorr_image.array + np.roll(uncorr_image.array, 1, axis=0)) # note NumPy thus [y,x]
    ynoise_image *= (np.sqrt(2.) / 2.) # Preserve variance
    return ynoise_image

def test_uncorrelated_noise_zero_lag():
    """Test that the zero lag correlation of an input uncorrelated noise field matches its variance.
    """
    t1 = time.time()
    sigmas = [3.e-9, 49., 1.11e11]  # some wide ranging sigma values for the noise field
    # loop through the sigmas
    cf_zero = 0.
    gd = galsim.GaussianDeviate(rseed)
    for sigma in sigmas:
        # Test the estimated value is good to 1% of the input variance; we expect this!
        # Note we make multiple correlation funcs and average their zero lag to beat down noise
        for i in range(nsum_test):
            uncorr_noise_image = setup_uncorrelated_noise(gd, largeim_size) * sigma
            cn = galsim.CorrelatedNoise(uncorr_noise_image, gd, scale=1.)
            cf_zero += cn._profile.xValue(galsim.PositionD(0., 0.))
        cf_zero /= float(nsum_test)
        np.testing.assert_almost_equal(
            cf_zero / sigma**2, 1., decimal=decimal_approx,
            err_msg="Zero distance noise correlation value does not match input noise variance.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_uncorrelated_noise_nonzero_lag():
    """Test that the non-zero lag correlation of an input uncorrelated noise field is zero at some
    randomly chosen positions.
    """
    t1 = time.time()
    # Set up some random positions (within and outside) the bounds of the table inside the
    # CorrelatedNoise then test
    uncorr_noise_image = galsim.ImageD(largeim_size, largeim_size)
    ud = galsim.UniformDeviate(rseed)
    gn = galsim.GaussianNoise(ud, sigma=1.)
    for i in range(npos_test):
        # Note we make multiple noise fields and correlation funcs and average non-zero lag values
        # to beat down noise
        cf_test_value = 0.
        for i in range(nsum_test):
            uncorr_noise_image.addNoise(gn)
            cn = galsim.CorrelatedNoise(uncorr_noise_image, ud, scale=1.)
            # generate the test position at least one pixel away from the origin
            rpos = 2. + ud() * (largeim_size - 2.) # this can go outside table bounds
            tpos = 2. * np.pi * ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            cf_test_value += cn._profile.xValue(pos)
            uncorr_noise_image.setZero()
        cf_test_value /= float(nsum_test)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_almost_equal(
            cf_test_value, 0., decimal=decimal_approx,
            err_msg="Non-zero distance noise correlation value not sufficiently close to target "+
            "value of zero.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_uncorrelated_noise_symmetry_90degree_rotation():
    """Test that the non-zero lag correlation of an input uncorrelated noise field has two-fold
    rotational symmetry and that CorrelatedNoise rotation methods produce the same output when 
    initializing with a 90 degree-rotated input field.
    """
    t1 = time.time()
    ud = galsim.UniformDeviate(rseed)
    uncorr_noise_small = setup_uncorrelated_noise(ud, smallim_size)
    cn = galsim.CorrelatedNoise(uncorr_noise_small, ud, scale=1.) # small image is fine here
    # Set up some random positions (within and outside) the bounds of the table inside the corrfunc
    # then test
    for i in range(npos_test):
        rpos = ud() * smallim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = cn._profile.xValue(pos)
        cf_test2 = cn._profile.xValue(-pos)
        np.testing.assert_almost_equal(
            cf_test1, cf_test2,
            decimal=decimal_precise,
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric.")
    # Then test that CorrelatedNoise rotation methods produces the same output as initializing 
    # with a 90 degree-rotated input field
    angles = [
        90. * galsim.degrees, 180. * galsim.degrees, 270. * galsim.degrees, 360. * galsim.degrees]
    # loop over rotation angles and check
    for i, angle in zip(range(len(angles)), angles):
        noise_ref = galsim.ImageD(
            np.ascontiguousarray(np.rot90(uncorr_noise_small.array, k=i+1)))
        cn_ref = galsim.CorrelatedNoise(noise_ref, ud, scale=1.)
        # First we'll check the createRotated() method
        cn_test1 = cn.createRotated(angle)
        # Then we'll check the applyRotation() method
        cn_test2 = cn.copy()
        cn_test2.applyRotation(angle)
        # Then check some positions inside the bounds of the original image
        for i in range(npos_test):
            rpos = .5 * ud() * smallim_size
            tpos = 2. * np.pi * ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            cf_ref = cn_ref._profile.xValue(pos)
            cf_test1 = cn_test1._profile.xValue(pos)
            cf_test2 = cn_test2._profile.xValue(pos)
            # Then test these estimated value is good to within our chosen decimal place
            np.testing.assert_almost_equal(
                cf_test1, cf_ref, decimal=decimal_precise,
                err_msg="Uncorrelated noise failed 90 degree createRotated() method test.")
            np.testing.assert_almost_equal(
                cf_test2, cf_ref, decimal=decimal_precise,
                err_msg="Uncorrelated noise failed 90 degree applyRotation() method test.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_xcorr_noise_basics_symmetry_90degree_rotation():
    """Test the basic properties of a noise field, correlated in the x direction, generated using
    a simple shift-add prescription, check it has two-fold rotational symmetry and behaves
    correctly under 90 degree rotations.
    """
    t1 = time.time()
    ud = galsim.UniformDeviate(rseed)
    # We make multiple correlation funcs and average their zero lag to beat down noise
    cf_zero = 0.
    cf_10 = 0.
    for i in range(nsum_test):
        uncorr_noise = setup_uncorrelated_noise(ud, largeim_size)
        xnoise = make_xcorr_from_uncorr(uncorr_noise)
        xcn = galsim.CorrelatedNoise(xnoise, ud, scale=1.)
        cf_zero += xcn._profile.xValue(galsim.PositionD(0., 0.))
        cf_10 += xcn._profile.xValue(galsim.PositionD(1., 0.))
    cf_zero /= float(nsum_test)
    cf_10 /= float(nsum_test)
    # Then test the zero-lag value is good to 1% of the input variance; we expect this!
    np.testing.assert_almost_equal(
        cf_zero, 1., decimal=decimal_approx,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (1, 0) value is good to 1% of the input variance (0.5); we expect this!
    np.testing.assert_almost_equal(
        cf_10, .5, decimal=decimal_approx,
        err_msg="Noise correlation value at (1, 0) does not match input covariance.")
    # Then set up some random positions (within and outside) the bounds of the table inside the 
    # corrfunc (the last one made is fine) then test for symmetry
    for i in range(npos_test):
        rpos = ud() * smallim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = xcn._profile.xValue(pos)
        cf_test2 = xcn._profile.xValue(-pos)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_almost_equal(
            cf_test1, cf_test2, decimal=decimal_precise, # should be good to machine precision
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric for x correlated noise field.")
    # Then test that CorrelatedNoise rotation methods produces the same output as initializing 
    # with a 90 degree-rotated input field
    angles = [
        90. * galsim.degrees, 180. * galsim.degrees, 270. * galsim.degrees, 360. * galsim.degrees]
    # loop over rotation angles and check
    for i, angle in zip(range(len(angles)), angles):
        noise_ref = galsim.ImageD(
            np.ascontiguousarray(np.rot90(xnoise.array, k=i+1)))
        xcn_ref = galsim.CorrelatedNoise(noise_ref, ud, scale=1.)
        # First we'll check the createRotated() method
        xcn_test1 = xcn.createRotated(angle)
        # Then we'll check the applyRotation() method
        xcn_test2 = xcn.copy()
        xcn_test2.applyRotation(angle)
        # Then check some positions inside the bounds of the original image
        for i in range(npos_test):
            rpos = .5 * ud() * smallim_size
            tpos = 2. * np.pi * ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            xcf_ref = xcn_ref._profile.xValue(pos)
            xcf_test1 = xcn_test1._profile.xValue(pos)
            xcf_test2 = xcn_test2._profile.xValue(pos)
            # Then test these estimated value is good to within our chosen decimal place
            np.testing.assert_almost_equal(
                xcf_test1, xcf_ref, decimal=decimal_precise,
                err_msg="x-correlated noise failed 90 degree createRotated() method test.")
            np.testing.assert_almost_equal(
                xcf_test2, xcf_ref, decimal=decimal_precise,
                err_msg="x-correlated noise failed 90 degree applyRotation() method test.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_ycorr_noise_basics_symmetry_90degree_rotation():
    """Test the basic properties of a noise field, correlated in the y direction, generated using
    a simple shift-add prescription, check it has two-fold rotational symmetry and behaves
    correctly under 90 degree rotations.
    """
    t1 = time.time()
    ud = galsim.UniformDeviate(rseed)
    # We make multiple correlation funcs and average their zero lag to beat down noise
    cf_zero = 0.
    cf_01 = 0.
    for i in range(nsum_test):
        uncorr_noise = setup_uncorrelated_noise(ud, largeim_size)
        ynoise = make_ycorr_from_uncorr(uncorr_noise)
        ycn = galsim.CorrelatedNoise(ynoise, ud, scale=1.)
        cf_zero += ycn._profile.xValue(galsim.PositionD(0., 0.))
        cf_01 += ycn._profile.xValue(galsim.PositionD(0., 1.))
    cf_zero /= float(nsum_test)
    cf_01 /= float(nsum_test)
    # Then test the zero-lag value is good to 1% of the input variance; we expect this!
    np.testing.assert_almost_equal(
        cf_zero, 1., decimal=decimal_approx,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (0, 1) value is good to 1% of the input variance (0.5); we expect this!
    np.testing.assert_almost_equal(
        cf_01, .5, decimal=decimal_approx,
        err_msg="Noise correlation value at (0, 1) does not match input covariance.")
    # Then set up some random positions (within and outside) the bounds of the table inside the 
    # corrfunc (the last one made is fine) then test for symmetry
    for i in range(npos_test):
        rpos = ud() * smallim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = ycn._profile.xValue(pos)
        cf_test2 = ycn._profile.xValue(-pos)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_almost_equal(
            cf_test1, cf_test2, decimal=decimal_precise, # should be good to machine precision
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric for x correlated noise field.")
    # Then test that CorrelatedNoise rotation methods produces the same output as initializing 
    # with a 90 degree-rotated input field
    angles = [
        90. * galsim.degrees, 180. * galsim.degrees, 270. * galsim.degrees, 360. * galsim.degrees]
    # loop over rotation angles and check
    for i, angle in zip(range(len(angles)), angles):
        noise_ref = galsim.ImageD(
            np.ascontiguousarray(np.rot90(ynoise.array, k=i+1)))
        ycn_ref = galsim.CorrelatedNoise(noise_ref, ud, scale=1.)
        # First we'll check the createRotated() method
        ycn_test1 = ycn.createRotated(angle)
        # Then we'll check the applyRotation() method
        ycn_test2 = ycn.copy()
        ycn_test2.applyRotation(angle)
        # Then check some positions inside the bounds of the original image
        for i in range(npos_test):
            rpos = .5 * ud() * smallim_size
            tpos = 2. * np.pi * ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            ycf_ref = ycn_ref._profile.xValue(pos)
            ycf_test1 = ycn_test1._profile.xValue(pos)
            ycf_test2 = ycn_test2._profile.xValue(pos)
            # Then test these estimated value is good to within our chosen decimal place
            np.testing.assert_almost_equal(
                ycf_test1, ycf_ref, decimal=decimal_precise,
                err_msg="y-correlated noise failed 90 degree createRotated() method test.")
            np.testing.assert_almost_equal(
                ycf_test2, ycf_ref, decimal=decimal_precise,
                err_msg="y-correlated noise failed 90 degree applyRotation() method test.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_arbitrary_rotation():
    """Check that rotated correlated noise xValues() are correct for a correlated noise with
    something in it.
    """
    # Just do the ycorr direction, if tests above pass this should be sufficient
    t1 = time.time()
    ud = galsim.UniformDeviate(rseed)
    ynoise_small = make_ycorr_from_uncorr(setup_uncorrelated_noise(ud, smallim_size))
    cn = galsim.CorrelatedNoise(ynoise_small, ud, scale=1.) # use something >0
    for i in range(npos_test):
        rot_angle = 2. * np.pi * ud()
        rpos = ud() * smallim_size # look in the vicinity of the action near the centre
        tpos = 2. * np.pi * ud()
        # get reference test position
        pos_ref = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        # then a rotated version
        pos_rot = galsim.PositionD(pos_ref.x * np.cos(rot_angle) + pos_ref.y * np.sin(rot_angle),
                                   -pos_ref.x * np.sin(rot_angle) + pos_ref.y * np.cos(rot_angle))
        # then create rotated cns for comparison
        cn_rot1 = cn.createRotated(rot_angle * galsim.radians)
        cn_rot2 = cn.copy()
        cn_rot2.applyRotation(rot_angle * galsim.radians)
        np.testing.assert_almost_equal(
            cn._profile.xValue(pos_rot), cn_rot1._profile.xValue(pos_ref), 
            decimal=decimal_precise, # this should be good at very high accuracy 
            err_msg="Noise correlated in the y direction failed createRotated() "+
            "method test for arbitrary rotations.")
        np.testing.assert_almost_equal(
            cn._profile.xValue(pos_rot), cn_rot2._profile.xValue(pos_ref), 
            decimal=decimal_precise, # ditto
            err_msg="Noise correlated in the y direction failed applyRotation() "+
            "method test for arbitrary rotations.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_scaling():
    """Test the scaling of correlation functions, specifically that the applyExpansion and
    createExpanded methods work correctly when querying the profile with xValue().
    """
    # Again, only use the x direction correlated noise, will be sufficient given tests above
    t1 = time.time()
    ud = galsim.UniformDeviate(rseed)
    xnoise_small = make_xcorr_from_uncorr(setup_uncorrelated_noise(ud, smallim_size))
    cn = galsim.CorrelatedNoise(xnoise_small, ud, scale=1.)
    scalings = [7.e-13, 424., 7.9e23]
    for scale in scalings:
       cn_test1 = cn.createExpanded(scale)
       cn_test2 = cn.copy() 
       cn_test2.applyExpansion(scale)
       for i in range(npos_test):
           rpos = ud() * 0.1 * smallim_size * scale # look in vicinity of the centre
           tpos = 2. * np.pi * ud()
           pos_ref = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
           np.testing.assert_almost_equal(
               cn_test1._profile.xValue(pos_ref), cn._profile.xValue(pos_ref / scale),
               decimal=decimal_precise,
               err_msg="Noise correlated in the y direction failed createExpanded() scaling test.")
           np.testing.assert_almost_equal(
               cn_test2._profile.xValue(pos_ref), cn._profile.xValue(pos_ref / scale),
               decimal=decimal_precise,
               err_msg="Noise correlated in the y direction failed applyExpansion() scaling "+
               "test.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_jacobian():
    """Check that transformed correlated noise xValues() are correct for a correlated noise with
    something in it.
    """
    t1 = time.time()
    ud = galsim.UniformDeviate(rseed)
    ynoise_small = make_ycorr_from_uncorr(setup_uncorrelated_noise(ud, smallim_size))
    cn = galsim.CorrelatedNoise(ynoise_small, ud, scale=1.)
    dudx = 0.241
    dudy = 0.051
    dvdx = -0.098
    dvdy = -0.278
    cn_test1 = cn.createTransformed(dudx,dudy,dvdx,dvdy)
    cn_test2 = cn.copy()
    cn_test2.applyTransformation(dudx,dudy,dvdx,dvdy)
    for i in range(npos_test):
        rpos = ud() * smallim_size
        tpos = 2. * np.pi * ud()
        pos_ref = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        pos_test = galsim.PositionD(pos_ref.x * dudx + pos_ref.y * dudy,
                                    pos_ref.x * dvdx + pos_ref.y * dvdy)
        np.testing.assert_almost_equal(
            cn_test1._profile.xValue(pos_test), cn._profile.xValue(pos_ref),
            decimal=decimal_precise,
            err_msg="Noise correlated in the y direction failed createTransformed() test")
        np.testing.assert_almost_equal(
            cn_test2._profile.xValue(pos_test), cn._profile.xValue(pos_ref),
            decimal=decimal_precise,
            err_msg="Noise correlated in the y direction failed applyTransformation() test")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)


def test_draw():
    """Test that the CorrelatedNoise draw() method matches its internal, NumPy-derived estimate of
    the correlation function, and an independent calculation of the same thing.
    """
    t1 = time.time()
    from galsim import utilities
    gd = galsim.GaussianDeviate(rseed)
    # We have slightly different expectations for how the CorrelatedNoise will represent and store
    # CFs from even and odd sized noise fields, so we will test both here.  
    #
    # First let's do odd (an uncorrelated noise field is fine for the tests we want to do):
    uncorr_noise_small_odd = setup_uncorrelated_noise(gd, smallim_size_odd) 
    uncorr_noise_small_odd -= uncorr_noise_small_odd.array.mean() # Subtract mean as in CF estimates
    # Build a noise correlated noise using DFTs
    ft_array = np.fft.fft2(uncorr_noise_small_odd.array)
    # Calculate the power spectrum then correlated noise
    ps_array = (ft_array * ft_array.conj()).real
    cf_array = (np.fft.ifft2(ps_array)).real / float(np.product(np.shape(ft_array)))
    cf_array = utilities.roll2d(cf_array, (cf_array.shape[0] / 2, cf_array.shape[1] / 2))
    # Then use the CorrelatedNoise class for comparison (don't use periodicity correction for
    # comparison with naive results above)
    cn = galsim.CorrelatedNoise(uncorr_noise_small_odd, gd, scale=1., correct_periodicity=False)
    testim1 = galsim.ImageD(smallim_size_odd, smallim_size_odd)
    cn.draw(testim1, scale=1.)
    # Then compare the odd-sized arrays:
    np.testing.assert_array_almost_equal(
        testim1.array, cf_array, decimal=decimal_precise, 
        err_msg="Drawn image (odd-sized) does not match independently calculated correlated noise.")
    # Now we do even
    uncorr_noise_small = setup_uncorrelated_noise(gd, smallim_size)
    uncorr_noise_small -= uncorr_noise_small.array.mean() # Subtract mean as in CF estimates
    ft_array = np.fft.fft2(uncorr_noise_small.array)
    # Calculate the power spectrum then correlated noise
    ps_array = (ft_array * ft_array.conj()).real
    cf_array = (np.fft.ifft2(ps_array)).real / float(np.product(np.shape(ft_array)))
    cf_array = utilities.roll2d(cf_array, (cf_array.shape[0] / 2, cf_array.shape[1] / 2))
    # Then use the CorrelatedNoise class for comparison (as above don't correct for periodicity)
    cn = galsim.CorrelatedNoise(uncorr_noise_small, gd, scale=1., correct_periodicity=False)
    testim1 = galsim.ImageD(smallim_size, smallim_size)
    cn.draw(testim1, scale=1.)
    # Then compare the even-sized arrays:
    np.testing.assert_array_almost_equal(
        testim1.array, cf_array, decimal=decimal_precise, 
        err_msg="Drawn image (even-sized) does not match independently calculated correlated "+
        "noise.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_output_generation_basic():
    """Test that noise generated by a CorrelatedNoise matches the correlated noise.  Averages over
    CorrelatedNoise instances from a number of realizations.
    """
    t1 = time.time()
    ud = galsim.UniformDeviate(rseed)
    # Get the correlated noise from an image of some x-correlated noise
    xnoise_large = make_xcorr_from_uncorr(setup_uncorrelated_noise(ud, largeim_size))
    # Note the noise we generate in these tests *is* periodic, so use non-default settings
    cn = galsim.CorrelatedNoise(xnoise_large, ud, scale=.18, correct_periodicity=False)
    refim = galsim.ImageD(smallim_size, smallim_size)
    # Draw this for reference
    cn.draw(refim, scale=.18)
    # Generate a large image containing noise according to this function
    outimage = galsim.ImageD(xnoise_large.bounds, scale=0.18)
    outimage.addNoise(cn)
    # Summed (average) CorrelatedNoises should be approximately equal to the input, so average
    # multiple CFs
    cn_2ndlevel = galsim.CorrelatedNoise(outimage, ud, scale=.18, correct_periodicity=False)
    # Draw the summed CF to an image for comparison 
    testim = galsim.ImageD(smallim_size, smallim_size)
    cn_2ndlevel.draw(testim, scale=.18, add_to_image=True)
    for i in range(nsum_test - 1):
        # Then repeat
        outimage.setZero()
        outimage.addNoise(cn)
        cn_2ndlevel = galsim.CorrelatedNoise(outimage, ud, scale=.18, correct_periodicity=False)
        cn_2ndlevel.draw(testim, scale=.18, add_to_image=True)
    # Then take average
    testim /= float(nsum_test)
    np.testing.assert_array_almost_equal(
        testim.array, refim.array, decimal=decimal_approx,
        err_msg="Generated noise field (basic) does not match input correlation properties.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_output_generation_rotated():
    """Test that noise generated by a rotated CorrelatedNoise matches the parent correlated noise.
    """
    t1 = time.time()
    # Get the correlated noise
    # Note that here we use an extra large image: this is because rotating the noise correlation
    # function (CF) brings in beyond-edge regions (imagine rotating a square but trimming within a
    # fixed square border of the same size).  These seem to add excess variance, perhaps due to
    # interpolant behaviour across transition to formal zero in the CF, which ruins agreement at
    # 2dp (still OK to 1dp or better).  This behaviour is quite strongly dependent on interpolant,
    # with Linear seeming to provide the best performance.  This is also likely to be related to
    # the fact that we do not zero-pad while generating the noise field as we might while generating
    # a galaxy in an empty patch of sky: the Linear interpolatant has a limited real space support.
    #
    # Update: See https://github.com/GalSim-developers/GalSim/pull/452 for a clearer discussion of
    # why the Linear provides a good (but not perfect) approximate treatment.
    #
    # Therefore, we rotate a CF with a support larger than the output region we simulate: this works
    # well at 2dp.
    #
    # TODO: It would be good to understand more about the detailed interpolant behaviour though...
    ud = galsim.UniformDeviate(rseed)
    # Get the correlated noise from an image of some y-correlated noise
    ynoise_xlarge = make_ycorr_from_uncorr(setup_uncorrelated_noise(ud, xlargeim_size))
    # Subtract the mean
    ynoise_xlarge -= ynoise_xlarge.array.mean()
    cn = galsim.CorrelatedNoise(ynoise_xlarge, ud, scale=1., correct_periodicity=False)
    # Then loop over some angles
    angles = [28.7 * galsim.degrees, 135. * galsim.degrees]
    for angle in angles:
        cn_rot = cn.createRotated(angle)
        refim = galsim.ImageD(smallim_size, smallim_size)
        # Draw this for reference
        cn_rot.draw(refim, scale=1.)
        # Generate a large image containing noise according to this function
        outimage = galsim.ImageD(largeim_size, largeim_size, scale=1.)
        outimage.addNoise(cn_rot)
        # Summed (average) CorrelatedNoises should be approximately equal to the input, so avg
        # multiple CFs
        cn_2ndlevel = galsim.CorrelatedNoise(
            outimage, ud, scale=1., subtract_mean=False, correct_periodicity=False)
        for i in range(nsum_test - 1):
            # Then repeat
            outimage.setZero()
            outimage.addNoise(cn_rot)
            cn_2ndlevel += galsim.CorrelatedNoise(
                outimage, ud, scale=1., subtract_mean=False, correct_periodicity=False)
        cn_2ndlevel /= float(nsum_test)
        # Then draw the summed CF to an image for comparison 
        testim = galsim.ImageD(smallim_size, smallim_size)
        cn_2ndlevel.draw(testim, scale=1.)
        np.testing.assert_array_almost_equal(
            testim.array, refim.array, decimal=decimal_approx,
            err_msg="Generated noise field (rotated) does not match input correlation properties.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_output_generation_magnified():
    """Test that noise generated by a magnified CorrelatedNoise matches the parent correlated noise.
    """
    t1 = time.time()
    ud = galsim.UniformDeviate(rseed)
    # Get the correlated noise from an image of some y-correlated noise
    ynoise_large = make_ycorr_from_uncorr(setup_uncorrelated_noise(ud, largeim_size))
    # Get the correlated noise
    cn = galsim.CorrelatedNoise(ynoise_large, ud, scale=1., correct_periodicity=False)
    refim = galsim.ImageD(smallim_size, smallim_size)
    # Draw this for reference
    cn.draw(refim, scale=1.)
    # Then loop over some scales, using `applyNoiseTo` with the relevant scaling in the `dx` to
    # argument check that the underlying correlated noise is preserved when both `dx` and
    # a magnification factor `scale` change in the same sense
    scales = [0.03, 11.]
    for scale in scales:
        cn_scl = cn.createExpanded(scale)
        # Generate a large image containing noise according to this function
        outimage = galsim.ImageD(largeim_size, largeim_size, scale=scale)
        outimage.addNoise(cn_scl)
        # Summed (average) CorrelatedNoises should be approximately equal to the input, so avg
        # multiple CFs
        cn_2ndlevel = galsim.CorrelatedNoise(outimage, ud, scale=1., correct_periodicity=False)
        for i in range(nsum_test - 1): # Need to add here to nsum_test to beat down noise
            # Then repeat
            outimage.setZero()
            outimage.addNoise(cn_scl) # apply noise using scale
            cn_2ndlevel += galsim.CorrelatedNoise(outimage, ud, scale=1., correct_periodicity=False)
        # Divide by nsum_test to get average quantities
        cn_2ndlevel /= float(nsum_test)
        # Then draw the summed CF to an image for comparison 
        testim = galsim.ImageD(smallim_size, smallim_size)
        cn_2ndlevel.draw(testim, scale=1.)
        np.testing.assert_array_almost_equal(
            testim.array, refim.array, decimal=decimal_approx,
            err_msg="Generated noise does not match (magnified) input correlation properties.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_copy():
    """Check that a copied correlated noise instance correctly represents the parent correlation
    properties.
    """
    t1 = time.time()
    ud = galsim.UniformDeviate(rseed)
    noise_image = setup_uncorrelated_noise(ud, smallim_size)
    cn = galsim.CorrelatedNoise(noise_image, ud, subtract_mean=True, correct_periodicity=False)
    cn_copy = cn.copy()
    # Fundamental checks on RNG
    assert cn.getRNG() is cn_copy.getRNG(), "Copied correlated noise does not keep same RNG."
    cn_copy.setRNG(galsim.UniformDeviate(rseed + 1))
    assert cn.getRNG() is not cn_copy.getRNG(), \
        "Copied correlated noise keeps same RNG despite reset."
    # Then check the profile in the copy is *NOT* shared, so that changes in one aren't manifest
    # in the other
    cn_copy = cn.copy()
    assert cn._profile is not cn_copy._profile, \
        "Copied correlated noise erroneously retains reference to parent's correlation function."
    # Set up some random positions within the bounds of the correlation funtion and check that
    # the xValues are nonetheless the same
    for i in range(npos_test):
        rpos = .5 * ud() * smallim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = cn._profile.xValue(pos)
        cf_test2 = cn_copy._profile.xValue(pos)
        np.testing.assert_equal(
            cf_test1, cf_test2,
            err_msg="Copied correlation function does not replicate the parent correlation "+
            "funtion when queried using ._profile.xValue().")
    # Check that the copied correlated noise generates the same noise field as its parent when
    # they are initialized with the same RNG immediately prior to noise generation
    outim1 = galsim.ImageD(smallim_size, smallim_size, scale=1.)
    outim2 = galsim.ImageD(smallim_size, smallim_size, scale=1.)
    cn_copy = cn.copy()
    cn.setRNG(galsim.UniformDeviate(rseed))
    cn_copy.setRNG(galsim.UniformDeviate(rseed))
    outim1.addNoise(cn)
    outim2.addNoise(cn_copy)
    # The test below does not yield *exactly* equivalent results, plausibly due to the fact that the
    # rootps_store cache is *not* copied over and when the CF is redrawn the GSObject may make
    # slight approximations.  So we'll just test at high precision:
    np.testing.assert_array_almost_equal(
        outim1.array, outim2.array, decimal=decimal_precise,
        err_msg="Copied correlated noise does not produce the same noise field as the parent "+
        "despite sharing the same RNG.")
    # To illustrate the point above, we'll run a test in which the rootps is not stored in cn when
    # created, by setting correct_periodicity=True and testing at very high precision:
    outim1.setZero()
    outim2.setZero()
    cn = galsim.CorrelatedNoise(noise_image, ud, subtract_mean=True, correct_periodicity=True)
    cn_copy = cn.copy()
    cn.setRNG(galsim.UniformDeviate(rseed))
    cn_copy.setRNG(galsim.UniformDeviate(rseed))
    outim1.addNoise(cn)
    outim2.addNoise(cn_copy)
    decimal_very_precise = 14
    np.testing.assert_array_almost_equal(
        outim1.array, outim2.array, decimal=decimal_precise,
        err_msg="Copied correlated noise does not produce the same noise field as the parent "+
        "despite sharing the same RNG (high precision test).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_cosmos_and_whitening():
    """Test that noise generated by an HST COSMOS correlated noise is correct and correctly
    whitened.  Includes test for a magnified, sheared, and rotated version of the COSMOS noise, and
    tests convolution with a ground-based PSF.
    """
    t1 = time.time()
    gd = galsim.GaussianDeviate(rseed)
    cosmos_scale = 7.5 # Use some non-default, non-unity value of COSMOS pixel spacing
    ccn = galsim.getCOSMOSNoise(
        '../examples/data/acs_I_unrot_sci_20_cf.fits', gd, cosmos_scale=cosmos_scale)
    # large image to beat down noise
    outimage = galsim.ImageD(3 * largeim_size, 3 * largeim_size, scale=cosmos_scale)
    outimage.addNoise(ccn)  # Add the COSMOS noise
    # Then estimate correlation function from generated noise
    cntest_correlated = galsim.CorrelatedNoise(outimage, ccn.getRNG())
    # Check basic correlation function values of the 3x3 pixel region around (0,0)
    pos = galsim.PositionD(0., 0.)
    cf00 = ccn._profile.xValue(pos)
    cftest00 = cntest_correlated._profile.xValue(pos)
    # Test variances first
    np.testing.assert_almost_equal(
        cftest00 / cf00, 1., decimal=decimal_approx,
        err_msg="Noise field generated with COSMOS CorrelatedNoise does not approximately match "+
        "input variance")
    # Then test (1, 0), (0, 1), (1,-1) and (1,1) values
    for xpos, ypos in zip((cosmos_scale, 0., cosmos_scale, cosmos_scale), 
                          (0., cosmos_scale, -cosmos_scale, cosmos_scale)):
        pos = galsim.PositionD(xpos, ypos)
        cf = ccn._profile.xValue(pos)
        cftest = cntest_correlated._profile.xValue(pos)
        np.testing.assert_almost_equal(
            cftest / cftest00, cf / cf00, decimal=decimal_approx,
            err_msg="Noise field generated with COSMOS CorrelatedNoise does not have "+
            "approximately matching interpixel covariances")
    # Now whiten the noise field, and check that its variance and covariances are as expected
    # (non-zero distance correlations ~ 0!)
    whitened_variance = ccn.whitenImage(outimage)
    cntest_whitened = galsim.CorrelatedNoise(outimage, ccn.getRNG()) # Get the correlation function
    cftest00 = cntest_whitened._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        cftest00 / whitened_variance, 1., decimal=decimal_approx,
        err_msg="Noise field generated by whitening COSMOS CorrelatedNoise does not approximately "+
        "match theoretical variance")
    # Then test (1, 0), (0, 1), (1,-1) and (1,1) values
    for xpos, ypos in zip((cosmos_scale, 0., cosmos_scale, cosmos_scale), 
                          (0., cosmos_scale, -cosmos_scale, cosmos_scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest = cntest_whitened._profile.xValue(pos)
        np.testing.assert_almost_equal(
            cftest / cftest00, 0., decimal=decimal_approx,
            err_msg="Noise field generated by whitening COSMOS CorrelatedNoise does not have "+
            "approximately zero interpixel covariances")
    # Now test whitening but having first expanded and sheared the COSMOS noise correlation
    ccn_transformed = ccn.createSheared(g1=-0.03, g2=0.07)
    ccn_transformed.applyRotation(313. * galsim.degrees)
    ccn_transformed.applyExpansion(3.9)
    outimage.setZero()
    outimage.addNoise(ccn_transformed)
    wht_variance = ccn_transformed.whitenImage(outimage)  # Whiten noise correlation
    cntest_whitened = galsim.CorrelatedNoise(outimage, ccn.getRNG()) # Get the correlation function
    cftest00 = cntest_whitened._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        cftest00 / wht_variance, 1., decimal=decimal_approx,
        err_msg="Noise field generated by whitening rotated, sheared, magnified COSMOS "+
        "CorrelatedNoise does not approximately match theoretical variance")
    # Then test (1, 0), (0, 1), (1,-1) and (1,1) values
    for xpos, ypos in zip((cosmos_scale, 0.,  cosmos_scale, cosmos_scale), 
                          (0., cosmos_scale, -cosmos_scale, cosmos_scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest = cntest_whitened._profile.xValue(pos)
        np.testing.assert_almost_equal(
            cftest / cftest00, 0., decimal=decimal_approx,
            err_msg="Noise field generated by whitening rotated, sheared, magnified COSMOS "+
            "CorrelatedNoise does not have approximately zero interpixel covariances")
    # Then convolve with a ground-based PSF and pixel, generate some more correlated noise
    # and whiten it
    scale = cosmos_scale * 9. # simulates a 0.03 arcsec * 9 = 0.27 arcsec pitch ground image
    psf_ground = galsim.Moffat(beta=3., fwhm=2.5*scale) # FWHM=0.675 arcsec seeing
    pix_ground = galsim.Pixel(scale)
    ccn_convolved = ccn_transformed.copy()
    # Convolve the correlated noise field with each of the psf, pix
    ccn_convolved.convolveWith(galsim.Convolve([psf_ground, pix_ground]))
    # Reset the outimage, and set its pixel scale to now be the ground-based resolution
    outimage.setZero()
    outimage.scale = scale
    # Add correlated noise
    outimage.addNoise(ccn_convolved)
    # Then whiten
    wht_variance = ccn_convolved.whitenImage(outimage)
    # Then test
    cntest_whitened = galsim.CorrelatedNoise(outimage, ccn.getRNG()) # Get the correlation function
    cftest00 = cntest_whitened._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        cftest00 / wht_variance, 1., decimal=decimal_approx,
        err_msg="Noise field generated by whitening rotated, sheared, magnified, convolved COSMOS "+
        "CorrelatedNoise does not approximately match theoretical variance")
    # Then test (1, 0), (0, 1), (1,-1) and (1,1) values
    for xpos, ypos in zip((scale, 0.,  scale, scale), 
                          (0., scale, -scale, scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest = cntest_whitened._profile.xValue(pos)
        np.testing.assert_almost_equal(
            cftest / cftest00, 0., decimal=decimal_approx,
            err_msg="Noise field generated by whitening rotated, sheared, magnified, convolved "+
            "COSMOS CorrelatedNoise does not have approximately zero interpixel covariances")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_symmetrizing():
    """Test that noise generated by an HST COSMOS correlated noise is correctly symmetrized.
    Includes test for a magnified, sheared, and rotated version of the COSMOS noise, and tests
    convolution with a ground-based PSF.
    """
    t1 = time.time()
    gd = galsim.GaussianDeviate(rseed)
    cosmos_scale = 7.5 # Use some non-default, non-unity value of COSMOS pixel spacing
    ccn = galsim.getCOSMOSNoise(
        '../examples/data/acs_I_unrot_sci_20_cf.fits', gd, cosmos_scale=cosmos_scale)
    # large image to beat down noise
    outimage = galsim.ImageD(symm_size_mult * largeim_size,
                             symm_size_mult * largeim_size, scale=cosmos_scale)
    outimage.addNoise(ccn)  # Add the COSMOS noise
    # Then estimate correlation function from generated noise
    cntest_correlated = galsim.CorrelatedNoise(outimage, ccn.getRNG())
    # Now apply 4-fold symmetry to the noise field, and check that its variance and covariances are
    # as expected (non-zero distance correlations should be symmetric)
    symmetrized_variance = ccn.symmetrizeImage(outimage, order=4)
    cntest_symmetrized = galsim.CorrelatedNoise(outimage, ccn.getRNG()) # Get the correlation function
    cftest00 = cntest_symmetrized._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        (cftest00/symmetrized_variance - 1.)/symm_divide, 0, decimal=decimal_approx,
        err_msg="Noise field generated by symmetrizing COSMOS CorrelatedNoise does not approximately "+
        "match theoretical variance")
    # Then test (1, 0), (0, 1), (-1,0) and (0,-1) values
    cftest = []
    for xpos, ypos in zip((cosmos_scale, 0., -cosmos_scale, 0.), 
                          (0., cosmos_scale, 0., -cosmos_scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest.append(cntest_symmetrized._profile.xValue(pos))
    # They should be approximately equal, so check ratios of each one to the next one
    for ind in range(len(cftest)-1):
        np.testing.assert_almost_equal(
            (cftest[ind]/cftest[ind+1] - 1.)/symm_divide, 0., decimal=decimal_approx,
            err_msg="Noise field generated by symmetrizing COSMOS CorrelatedNoise does not have "+
            "approximate 4-fold symmetry")
    # Now test symmetrizing, but having first expanded and sheared the COSMOS noise correlation
    ccn_transformed = ccn.createSheared(g1=-0.05, g2=0.11)
    ccn_transformed.applyRotation(313. * galsim.degrees)
    ccn_transformed.applyExpansion(2.1)
    outimage.setZero()
    outimage.addNoise(ccn_transformed)
    sym_variance = ccn_transformed.symmetrizeImage(outimage, order=4)  # Symmetrize noise correlation
    cntest_symmetrized = galsim.CorrelatedNoise(outimage, ccn.getRNG()) # Get the correlation function
    cftest00 = cntest_symmetrized._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        (cftest00/sym_variance - 1.)/symm_divide, 0., decimal=decimal_approx,
        err_msg="Noise field generated by symmetrizing rotated, sheared, magnified COSMOS "+
        "CorrelatedNoise does not approximately match theoretical variance")
    # Then test (1, 0), (0, 1), (-1,0) and (0,-1) values
    cftest = []
    for xpos, ypos in zip((cosmos_scale, 0., -cosmos_scale, 0.), 
                          (0., cosmos_scale, 0., -cosmos_scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest.append(cntest_symmetrized._profile.xValue(pos))
    # They should be approximately equal, so check ratios of each one to the next one
    for ind in range(len(cftest)-1):
        np.testing.assert_almost_equal(
            (cftest[ind]/cftest[ind+1]-1.)/symm_divide, 0., decimal=decimal_approx,
            err_msg="Noise field generated by symmetrizing rotated, sheared, magnified COSMOS "+
            "CorrelatedNoise does not have approximate 4-fold symmetry")
    # Then convolve with a ground-based PSF and pixel, generate some more correlated noise
    # and symmetrize it
    scale = cosmos_scale * 9. # simulates a 0.03 arcsec * 9 = 0.27 arcsec pitch ground image
    psf_ground = galsim.Moffat(beta=3., fwhm=2.5*scale) # FWHM=0.675 arcsec seeing
    psf_ground = psf_ground.shear(g1=0.1)
    pix_ground = galsim.Pixel(scale)
    ccn_convolved = ccn_transformed.copy()
    # Convolve the correlated noise field with each of the psf, pix
    ccn_convolved.convolveWith(galsim.Convolve([psf_ground, pix_ground]))
    # Reset the outimage, and set its pixel scale to now be the ground-based resolution
    outimage.setZero()
    outimage.scale = scale
    # Add correlated noise
    outimage.addNoise(ccn_convolved)
    # Then symmetrize
    sym_variance = ccn_convolved.symmetrizeImage(outimage, order=4)
    # Then test
    cntest_symmetrized = galsim.CorrelatedNoise(outimage, ccn.getRNG()) # Get the correlation function
    cftest00 = cntest_symmetrized._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        (cftest00/sym_variance - 1.)/symm_divide, 0., decimal=decimal_approx,
        err_msg="Noise field generated by symmetrizing rotated, sheared, magnified, convolved COSMOS "+
        "CorrelatedNoise does not approximately match theoretical variance")
    # Then test (1, 0), (0, 1), (-1, 0) and (0, -1) values
    cftest = []
    for xpos, ypos in zip((scale, 0.,  -scale, 0), 
                          (0., scale, 0, -scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest.append(cntest_symmetrized._profile.xValue(pos))
    # They should be approximately equal, so check ratios of each one to the next one
    for ind in range(len(cftest)-1):
        np.testing.assert_almost_equal(
            (cftest[ind]/cftest[ind+1] - 1.)/symm_divide, 0., decimal=decimal_approx,
            err_msg="Noise field generated by symmetrizing rotated, sheared, magnified, convolved "+
            "COSMOS CorrelatedNoise does not have approximate 4-fold symmetry")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_convolve_cosmos():
    """Test that a COSMOS noise field convolved with a ground based PSF-style kernel matches the
    output of the correlated noise model modified with the convolveWith method.
    """
    t1 = time.time()
    gd = galsim.GaussianDeviate(rseed)
    cosmos_scale=0.03 # Non-unity, non-default value to be used below
    cn = galsim.getCOSMOSNoise(
        '../examples/data/acs_I_unrot_sci_20_cf.fits', gd, cosmos_scale=cosmos_scale)
    cn.setVariance(300.) # Again chosen to be non-unity and so as to produce ~unity output variance
    # Define a PSF with which to convolve the noise field, one WITHOUT 2-fold rotational symmetry
    # (see test_autocorrelate in test_SBProfile.py for more info as to why this is relevant)
    # Make a relatively realistic mockup of a GREAT3 target image
    lam_over_diam_cosmos = (814.e-9 / 2.4) * (180. / np.pi) * 3600. # ~lamda/D in arcsec
    lam_over_diam_ground = lam_over_diam_cosmos * 2.4 / 4. # Generic 4m at same lambda
    psf_cosmos = galsim.Convolve([
        galsim.Airy(lam_over_diam=lam_over_diam_cosmos, obscuration=0.4), galsim.Pixel(0.05)])
    psf_ground = galsim.Convolve([
        galsim.Kolmogorov(fwhm=0.8), galsim.Pixel(0.18),
        galsim.OpticalPSF(lam_over_diam=lam_over_diam_ground, coma2=0.4, defocus=-0.6)])
    psf_shera = galsim.Convolve([
        psf_ground, (galsim.Deconvolve(psf_cosmos)).createSheared(g1=0.03, g2=-0.01)])
    # Then define the convolved cosmos correlated noise model
    conv_cn = cn.copy()
    conv_cn.convolveWith(psf_shera)
    # Then draw the correlation function for this correlated noise as the reference
    refim = galsim.ImageD(smallim_size, smallim_size)
    conv_cn.draw(refim, scale=0.18)
    # Now start the test...
    # First we generate a COSMOS noise field (cosimage), read it into an InterpolatedImage and
    # then convolve it with psf, making sure we pad the edges
    interp=galsim.Linear(tol=1.e-4) # interpolation kernel to use in making convimages
    # Number of tests needs to be a little larger to beat down noise here, but see the script
    # in devel/external/test_cf/test_cf_convolution_detailed.py
    cosimage_padded = galsim.ImageD(
        (2 * smallim_size) * 6 + 256, # Note 6 here since 0.18 = 6 * 0.03
        (2 * smallim_size) * 6 + 256, # large image to beat down noise + padding
        scale = cosmos_scale)            # Use COSMOS pixel scale
    cosimage_padded.addNoise(cn) # Add cosmos noise
    # Put this noise into a GSObject and then convolve
    imobj_padded = galsim.InterpolatedImage(
        cosimage_padded, calculate_stepk=False, calculate_maxk=False,
        normalization='sb', scale=cosmos_scale, x_interpolant=interp)
    cimobj_padded = galsim.Convolve(imobj_padded, psf_shera)

    # We draw, calculate a correlation function for the resulting field, and repeat to get an
    # average over nsum_test trials
    convimage = galsim.ImageD(2 * smallim_size, 2 * smallim_size)
    cimobj_padded.draw(convimage, scale=0.18, normalization='sb')
    cn_test = galsim.CorrelatedNoise(
        convimage, gd, scale=0.18, correct_periodicity=True, subtract_mean=False)
    testim = galsim.ImageD(smallim_size, smallim_size)
    cn_test.draw(testim, scale=0.18)
    # Start some lists to store image info
    conv_list = [convimage.array.copy()] # Don't forget Python reference/assignment semantics, we
                                         # zero convimage and write over it later!
    mnsq_list = [np.mean(convimage.array**2)]
    var_list = [convimage.array.var()]
    #nsum_test = 500 - uncomment this line to pass test below at 2dp
    for i in range(nsum_test - 1):
        cosimage_padded.setZero()
        cosimage_padded.addNoise(cn)
        imobj_padded = galsim.InterpolatedImage(
            cosimage_padded, calculate_stepk=False, calculate_maxk=False,
            normalization='sb', scale=cosmos_scale, x_interpolant=interp)
        cimobj_padded = galsim.Convolve(imobj_padded, psf_shera) 
        convimage.setZero() # See above 
        # Draw convolved image into convimage
        cimobj_padded.draw(convimage, scale=0.18, normalization='sb')
        conv_list.append(convimage.array.copy()) # See above
        mnsq_list.append(np.mean(convimage.array**2))
        var_list.append(convimage.array.var())
        cn_test = galsim.CorrelatedNoise(
            convimage, gd, scale=0.18, correct_periodicity=True, subtract_mean=False) 
        cn_test.draw(testim, scale=0.18, add_to_image=True)
        del imobj_padded
        del cimobj_padded
        del cn_test

    mnsq_individual = sum(mnsq_list) / float(nsum_test)
    var_individual = sum(var_list) / float(nsum_test)
    mnsq_individual = sum(mnsq_list) / float(nsum_test)
    testim /= float(nsum_test) # Take average CF of trials   
    conv_array = np.asarray(conv_list)
    mnsq_all = np.mean(conv_array**2)
    var_all = conv_array.var()
    print "Mean square estimate from avg. of individual field mean squares = "+str(mnsq_individual)
    print "Mean square estimate from all fields = "+str(mnsq_all)
    print "Ratio of mean squares = %e" % (mnsq_individual / mnsq_all)
    print "Variance estimate from avg. of individual field variances = "+str(var_individual)
    print "Variance estimate from all fields = "+str(var_all)
    print "Ratio of variances = %e" % (var_individual / var_all)
    print "Zero lag CF from avg. of individual field CFs = "+str(testim.array[8, 8])
    print "Zero lag CF in reference case = "+str(refim.array[8, 8])
    print "Ratio of zero lag CFs = %e" % (testim.array[8, 8] / refim.array[8, 8])
    print "Printing analysis of central 4x4 of CF:"
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'var diff = ',np.var(testim.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'min diff = ',np.min(testim.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'max diff = ',np.max(testim.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim.array[4:12, 4:12] / refim.array[4:12, 4:12])

    # Test (this is a crude regression test at best, for a much more precise test of this behaviour
    # see devel/external/test_cf/test_cf_convolution_detailed.py)
    np.testing.assert_array_almost_equal(
        testim.array, refim.array, decimal=1,#decimal_approx, - if you want to pass at 2dp, make
                                             # nsum_test=500 above, takes ~100s on a fast laptop
        err_msg="Convolved COSMOS noise fields do not match the convolved correlated noise model.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_uncorrelated_noise_tracking():
    """Test that we can track various processes that convert uncorrelated noise to correlated noise.
    """
    t1 = time.time()

    # Start with an UncorrelatedNoise instance that we attach to an InterpolatedImage GSObject as a
    # 'noise' attribute
    gal_sigma = 1.
    noise_var = 1.
    seed = 1234
    pix_scale = 0.1
    orig_object = galsim.Gaussian(sigma=gal_sigma)
    orig_ucn = galsim.UncorrelatedNoise(noise_var, rng=galsim.BaseDeviate(seed), scale=pix_scale)
    im = orig_object.draw(scale=pix_scale)
    int_im = galsim.InterpolatedImage(im)
    # Note, I'm including the noise attribute without actually adding noise.  It doesn't matter
    # here, we just want to check the ability of GalSim to track what happens to `noise'
    # attributes.
    int_im.noise = orig_ucn

    # Manipulate the object in various non-trivial ways: shear, magnify, rotate, convolve
    test_shear = 0.15
    test_mag = 0.92
    rot_ang = 21. # degrees
    new_int_im = int_im.shear(g1=test_shear)
    new_int_im = new_int_im.magnify(test_mag)
    new_int_im = new_int_im.rotate(rot_ang*galsim.degrees)
    print 'new_int_im.noise = ',new_int_im.noise
    new_int_im = galsim.Convolve(new_int_im, orig_object)
    print 'new_int_im.noise => ',new_int_im.noise
    final_noise = new_int_im.noise

    # Now, make a correlated noise object directly based on a realization of the original
    # uncorrelated noise object.
    test_im = galsim.Image(512,512)
    orig_ucn.applyTo(test_im)
    cn = galsim.CorrelatedNoise(test_im, galsim.BaseDeviate(seed), scale=pix_scale)

    # Run it through the same operations.
    new_cn = cn.shear(g1=test_shear)
    new_cn = new_cn.magnify(test_mag)
    new_cn = new_cn.rotate(rot_ang*galsim.degrees)
    new_cn = new_cn.convolvedWith(orig_object)

    # Make sure that it's basically the same as the manipulated 'noise' object from the first case,
    # i.e., compare final_noise with new_cn.
    # Allow for some error due to inferring the CorrelatedNoise object 'cn' from a single
    # realization.  For now we'll do the simplest possible comparison of just the variance.  This is
    # probably not adequate but it's a start.
    np.testing.assert_almost_equal(final_noise.getVariance(), new_cn.getVariance(), decimal=3,
                                   err_msg='Failure in tracking noise properties through operations')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_variance_changes():
    """Test that we can change and check the variance for CorrelatedNoise objects.
    """
    t1 = time.time()

    # Make an UncorrelatedNoise object.
    noise_var = 1.24
    seed = 1234
    pix_scale = 0.1
    orig_ucn = galsim.UncorrelatedNoise(noise_var, rng=galsim.BaseDeviate(seed), scale=pix_scale)
    # Reset variance to something else.
    new_var = 1.07
    orig_ucn.setVariance(new_var)
    np.testing.assert_equal(orig_ucn.getVariance(), new_var,
                            err_msg='Failure to reset and then get variance for UncorrelatedNoise')

    # Now do this for a CorrelatedNoise object.
    gd = galsim.GaussianDeviate()
    cosmos_scale=0.03
    cn = galsim.getCOSMOSNoise(
        '../examples/data/acs_I_unrot_sci_20_cf.fits', gd, cosmos_scale=cosmos_scale)
    cn.setVariance(new_var)
    np.testing.assert_equal(cn.getVariance(), new_var,
                            err_msg='Failure to reset and then get variance for CorrelatedNoise')


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

if __name__ == "__main__":
    test_uncorrelated_noise_zero_lag()
    test_uncorrelated_noise_nonzero_lag()
    test_uncorrelated_noise_symmetry_90degree_rotation()
    test_xcorr_noise_basics_symmetry_90degree_rotation()
    test_ycorr_noise_basics_symmetry_90degree_rotation()
    test_arbitrary_rotation()
    test_scaling()
    test_jacobian()
    test_draw()
    test_output_generation_basic()
    test_output_generation_rotated()
    test_output_generation_magnified()
    test_copy()
    test_cosmos_and_whitening()
    test_symmetrizing()
    test_convolve_cosmos()
    test_uncorrelated_noise_tracking()
    test_variance_changes()

