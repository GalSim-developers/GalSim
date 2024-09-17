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


# Use a deterministic random number generator so we don't fail tests because of rare flukes
# in the random numbers.
rseed=12345

smallim_size = 16 # size of image when we test correlated noise properties using small inputs
smallim_size_odd = 17 # odd-sized version of the above for odd/even relevant tests (e.g. draw)
largeim_size = 12 * smallim_size # ditto, but when we need a larger image

# Number of positions to test in nonzero lag uncorrelated tests
npos_test = 3

# Number of CorrelatedNoises to sum over to get slightly better statistics for noise generation test
nsum_test = 7

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


@timer
def test_uncorrelated_noise_zero_lag():
    """Test that the zero lag correlation of an input uncorrelated noise field matches its variance.
    """
    sigmas = [3.e-9, 49., 1.11e11]  # some wide ranging sigma values for the noise field
    # loop through the sigmas
    gd = galsim.GaussianDeviate(rseed)
    for sigma in sigmas:
        # Test the estimated value is good to 1% of the input variance; we expect this!
        # Note we make multiple correlation funcs and average their zero lag to beat down noise
        cf_zero = 0.
        for i in range(nsum_test):
            uncorr_noise_image = setup_uncorrelated_noise(gd, largeim_size) * sigma
            cn = galsim.CorrelatedNoise(uncorr_noise_image, gd)
            cf_zero += cn._profile.xValue(galsim.PositionD(0., 0.))
        cf_zero /= float(nsum_test)
        np.testing.assert_almost_equal(
            cf_zero / sigma**2, 1., decimal=2,
            err_msg="Zero distance noise correlation value does not match input noise variance.")

        # Repeat using UncorrelatedNoise
        ucn = galsim.UncorrelatedNoise(variance=sigma**2)
        ucf_zero = ucn._profile.xValue(galsim.PositionD(0.,0.))
        np.testing.assert_almost_equal(
            ucf_zero / sigma**2, 1., decimal=5,
            err_msg="Zero distance noise correlation value does not match variance value " +
            "provided to UncorrelatedNoise.")

        # Check picklability
        check_pickle(ucn, lambda x: (x.rng.serialize(), x.getVariance(), x.wcs))
        check_pickle(ucn, drawNoise)
        check_pickle(cn, lambda x: (x.rng.serialize(), x.getVariance(), x.wcs))
        check_pickle(cn, drawNoise)
        check_pickle(ucn)
        check_pickle(cn)

    assert_raises(TypeError, galsim.UncorrelatedNoise)
    assert_raises(ValueError, galsim.UncorrelatedNoise, variance = -1.0)
    assert_raises(TypeError, galsim.UncorrelatedNoise, 1, scale=1, wcs=galsim.PixelScale(3))
    assert_raises(TypeError, galsim.UncorrelatedNoise, 1, wcs=1)
    assert_raises(ValueError, galsim.UncorrelatedNoise, 1,
                  wcs=galsim.FitsWCS('fits_files/tpv.fits'))
    assert_raises(TypeError, galsim.UncorrelatedNoise, 1, rng=10)


@timer
def test_uncorrelated_noise_nonzero_lag():
    """Test that the non-zero lag correlation of an input uncorrelated noise field is zero at some
    randomly chosen positions.
    """
    ud = galsim.UniformDeviate(rseed)
    sigma = 1.7  # Don't use sigma = 1, to notice sqrt error.
    gn = galsim.GaussianNoise(ud, sigma=sigma)
    # Set up some random positions (within and outside) the bounds of the table inside the
    # CorrelatedNoise then test
    for rpos in [ 1.3, 23., 84., 243. ]:
        # Note we make multiple noise fields and correlation funcs and average non-zero lag values
        # to beat down noise
        cf_test_value = 0.
        for i in range(nsum_test):
            tpos = 2. * np.pi * ud()  # A new random direction each time
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            uncorr_noise_image = setup_uncorrelated_noise(ud, largeim_size) * sigma
            cn = galsim.CorrelatedNoise(uncorr_noise_image, ud)
            # generate the test position at least one pixel away from the origin
            cf_test_value += cn._profile.xValue(pos)
        cf_test_value /= float(nsum_test)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_almost_equal(
            cf_test_value, 0., decimal=2,
            err_msg="Non-zero distance noise correlation value not sufficiently close to target "+
            "value of zero.")

        # Repeat using UncorrelatedNoise
        ucn = galsim.UncorrelatedNoise(variance=sigma**2)
        ucf_test_value = ucn._profile.xValue(pos)
        np.testing.assert_almost_equal(
            ucf_test_value, 0., decimal=5,
            err_msg="Non-zero distance noise correlation value not sufficiently close to target "+
            "value of zero.")


@timer
def test_uncorrelated_noise_output():
    """Test that noise generated by a UncorrelatedNoise has the expected variance.
    """
    import math

    ud = galsim.UniformDeviate(rseed)
    sigma = 1.7

    # Note: Test non-square and odd size in y.
    im = galsim.ImageD(2*largeim_size, 3*largeim_size+1, init_value=0.)

    # If no wcs is given, then the variance is the pixel variance
    ucn = galsim.UncorrelatedNoise(rng=ud, variance=sigma**2)
    im.addNoise(ucn)
    im_var = im.array.var()
    np.testing.assert_array_almost_equal(
        im_var/sigma**2, 1.0, decimal=2,
        err_msg="Generated uncorrelated noise field (2) does not have expected variance")

    # If we provide a wcs, then the wcs given to UncorrelatedNoise should match the wcs
    # of the image.  In this case, the resulting pixel variance should still match the
    # provided variance parameter.
    if False:
        # The last one is a sufficient unit test, but if it fails, it is helpful to
        # test the simpler ones first.
        wcs_list = [ galsim.PixelScale(1.0),
                     galsim.PixelScale(0.23),
                     galsim.JacobianWCS(0.23, 0.01, -0.05, 0.21),
                     galsim.AffineTransform(0.23, 0.31, -0.15, 0.21, galsim.PositionD(73,45)) ]
    else:
        wcs_list = [ galsim.AffineTransform(0.23, 0.31, -0.15, 0.21, galsim.PositionD(73,45)) ]
    for wcs in wcs_list:
        scale = math.sqrt(wcs.pixelArea())
        im.wcs = wcs
        print('wcs = ',wcs)
        print('scale = ',scale)

        # First test with an UncorrelatedNoise object
        ucn = galsim.UncorrelatedNoise(rng=ud, variance=sigma**2, wcs=wcs)
        im.setZero()
        im.addNoise(ucn)
        im_var = im.array.var()
        print('im_mean = ',im.array.mean())
        print('im_std = ',im.array.std())
        print('im_var = ',im_var)
        np.testing.assert_array_almost_equal(
            im_var/sigma**2, 1.0, decimal=2,
            err_msg="Generated uncorrelated noise field (1) does not have expected variance")

        # Test with a regular CorrelatedNoise object built from an image with this variance
        uncorr_noise_image = setup_uncorrelated_noise(ud, 10*largeim_size) * sigma
        uncorr_noise_image.wcs = wcs
        cn = galsim.CorrelatedNoise(uncorr_noise_image, ud)
        im.setZero()
        im.addNoise(cn)
        im_var = im.array.var()
        print('im_mean = ',im.array.mean())
        print('im_std = ',im.array.std())
        print('im_var = ',im_var)
        np.testing.assert_array_almost_equal(
            im_var/sigma**2, 1.0, decimal=2,
            err_msg="Generated uncorrelated noise field (2) does not have expected variance")


@timer
def test_uncorrelated_noise_symmetry_90degree_rotation():
    """Test that the non-zero lag correlation of an input uncorrelated noise field has two-fold
    rotational symmetry and that CorrelatedNoise rotation methods produce the same output when
    initializing with a 90 degree-rotated input field.
    """
    ud = galsim.UniformDeviate(rseed)
    uncorr_noise_small = setup_uncorrelated_noise(ud, smallim_size)
    cn = galsim.CorrelatedNoise(uncorr_noise_small, ud) # small image is fine here
    # Set up some random positions (within and outside) the bounds of the table inside the corrfunc
    # then test
    for i in range(npos_test):
        rpos = ud() * smallim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = cn._profile.xValue(pos)
        cf_test2 = cn._profile.xValue(-pos)
        np.testing.assert_almost_equal(
            cf_test1, cf_test2, decimal=7,
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
        cn_ref = galsim.CorrelatedNoise(noise_ref, ud)
        cn_test1 = cn.rotate(angle)
        # Then check some positions inside the bounds of the original image
        for i in range(npos_test):
            rpos = .5 * ud() * smallim_size
            tpos = 2. * np.pi * ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            cf_ref = cn_ref._profile.xValue(pos)
            cf_test1 = cn_test1._profile.xValue(pos)
            # Then test these estimated value is good to within our chosen decimal place
            np.testing.assert_almost_equal(
                cf_test1, cf_ref, decimal=7,
                err_msg="Uncorrelated noise failed 90 degree rotate() method test.")


@timer
def test_xcorr_noise_basics_symmetry_90degree_rotation():
    """Test the basic properties of a noise field, correlated in the x direction, generated using
    a simple shift-add prescription, check it has two-fold rotational symmetry and behaves
    correctly under 90 degree rotations.
    """
    ud = galsim.UniformDeviate(rseed)
    # We make multiple correlation funcs and average their zero lag to beat down noise
    cf_zero = 0.
    cf_10 = 0.
    for i in range(nsum_test):
        uncorr_noise = setup_uncorrelated_noise(ud, largeim_size)
        xnoise = make_xcorr_from_uncorr(uncorr_noise)
        xcn = galsim.CorrelatedNoise(xnoise, ud)
        cf_zero += xcn._profile.xValue(galsim.PositionD(0., 0.))
        cf_10 += xcn._profile.xValue(galsim.PositionD(1., 0.))
    cf_zero /= float(nsum_test)
    cf_10 /= float(nsum_test)
    # Then test the zero-lag value is good to 1% of the input variance; we expect this!
    np.testing.assert_almost_equal(
        cf_zero, 1., decimal=2,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (1, 0) value is good to 1% of the input variance (0.5); we expect this!
    np.testing.assert_almost_equal(
        cf_10, .5, decimal=2,
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
            cf_test1, cf_test2, decimal=12, # should be good to machine precision
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
        xcn_ref = galsim.CorrelatedNoise(noise_ref, ud)
        xcn_test1 = xcn.rotate(angle)
        # Then check some positions inside the bounds of the original image
        for i in range(npos_test):
            rpos = .5 * ud() * smallim_size
            tpos = 2. * np.pi * ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            xcf_ref = xcn_ref._profile.xValue(pos)
            xcf_test1 = xcn_test1._profile.xValue(pos)
            # Then test these estimated value is good to within our chosen decimal place
            np.testing.assert_almost_equal(
                xcf_test1, xcf_ref, decimal=7,
                err_msg="x-correlated noise failed 90 degree rotate() method test.")


@timer
def test_ycorr_noise_basics_symmetry_90degree_rotation():
    """Test the basic properties of a noise field, correlated in the y direction, generated using
    a simple shift-add prescription, check it has two-fold rotational symmetry and behaves
    correctly under 90 degree rotations.
    """
    ud = galsim.UniformDeviate(rseed)
    # We make multiple correlation funcs and average their zero lag to beat down noise
    cf_zero = 0.
    cf_01 = 0.
    for i in range(nsum_test):
        uncorr_noise = setup_uncorrelated_noise(ud, largeim_size)
        ynoise = make_ycorr_from_uncorr(uncorr_noise)
        ycn = galsim.CorrelatedNoise(ynoise, ud)
        cf_zero += ycn._profile.xValue(galsim.PositionD(0., 0.))
        cf_01 += ycn._profile.xValue(galsim.PositionD(0., 1.))
    cf_zero /= float(nsum_test)
    cf_01 /= float(nsum_test)
    # Then test the zero-lag value is good to 1% of the input variance; we expect this!
    np.testing.assert_almost_equal(
        cf_zero, 1., decimal=2,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (0, 1) value is good to 1% of the input variance (0.5); we expect this!
    np.testing.assert_almost_equal(
        cf_01, .5, decimal=2,
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
            cf_test1, cf_test2, decimal=12, # should be good to machine precision
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
        ycn_ref = galsim.CorrelatedNoise(noise_ref, ud)
        ycn_test1 = ycn.rotate(angle)
        # Then check some positions inside the bounds of the original image
        for i in range(npos_test):
            rpos = .5 * ud() * smallim_size
            tpos = 2. * np.pi * ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            ycf_ref = ycn_ref._profile.xValue(pos)
            ycf_test1 = ycn_test1._profile.xValue(pos)
            # Then test these estimated value is good to within our chosen decimal place
            np.testing.assert_almost_equal(
                ycf_test1, ycf_ref, decimal=7,
                err_msg="y-correlated noise failed 90 degree rotate() method test.")


@timer
def test_arbitrary_rotation():
    """Check that rotated correlated noise xValues() are correct for a correlated noise with
    something in it.
    """
    # Just do the ycorr direction, if tests above pass this should be sufficient
    ud = galsim.UniformDeviate(rseed)
    ynoise_small = make_ycorr_from_uncorr(setup_uncorrelated_noise(ud, smallim_size))
    cn = galsim.CorrelatedNoise(ynoise_small, ud)
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
        cn_rot1 = cn.rotate(rot_angle * galsim.radians)
        np.testing.assert_almost_equal(
            cn._profile.xValue(pos_rot), cn_rot1._profile.xValue(pos_ref), decimal=12,
            err_msg="Noise correlated in the y direction failed rotate() "+
            "method test for arbitrary rotations.")


@timer
def test_scaling():
    """Test the scaling of correlation functions, specifically that the expand
    method works correctly when querying the profile with xValue().
    """
    # Again, only use the x direction correlated noise, will be sufficient given tests above
    ud = galsim.UniformDeviate(rseed)
    xnoise_small = make_xcorr_from_uncorr(setup_uncorrelated_noise(ud, smallim_size))
    cn = galsim.CorrelatedNoise(xnoise_small, ud)
    scalings = [7.e-13, 424., 7.9e23]
    for scale in scalings:
       cn_test1 = cn.expand(scale)
       for i in range(npos_test):
           rpos = ud() * 0.1 * smallim_size * scale # look in vicinity of the centre
           tpos = 2. * np.pi * ud()
           pos_ref = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
           np.testing.assert_almost_equal(
               cn_test1._profile.xValue(pos_ref), cn._profile.xValue(pos_ref / scale), decimal=7,
               err_msg="Noise correlated in the y direction failed expand() scaling test.")


@timer
def test_jacobian():
    """Check that transformed correlated noise xValues() are correct for a correlated noise with
    something in it.
    """
    ud = galsim.UniformDeviate(rseed)
    ynoise_small = make_ycorr_from_uncorr(setup_uncorrelated_noise(ud, smallim_size))
    cn = galsim.CorrelatedNoise(ynoise_small, ud)
    dudx = 0.241
    dudy = 0.051
    dvdx = -0.098
    dvdy = -0.278
    cn_test1 = cn.transform(dudx,dudy,dvdx,dvdy)
    for i in range(npos_test):
        rpos = ud() * smallim_size
        tpos = 2. * np.pi * ud()
        pos_ref = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        pos_test = galsim.PositionD(pos_ref.x * dudx + pos_ref.y * dudy,
                                    pos_ref.x * dvdx + pos_ref.y * dvdy)
        np.testing.assert_almost_equal(
            cn_test1._profile.xValue(pos_test), cn._profile.xValue(pos_ref), decimal=7,
            err_msg="Noise correlated in the y direction failed transform() test")


@timer
def test_drawImage():
    """Test that the CorrelatedNoise drawImage() method matches its internal, NumPy-derived
    estimate of the correlation function, and an independent calculation of the same thing.
    """
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
    cf_array = (np.fft.ifft2(ps_array)).real / float(np.prod(np.shape(ft_array)))
    cf_array = utilities.roll2d(cf_array, (cf_array.shape[0] // 2, cf_array.shape[1] // 2))
    # Then use the CorrelatedNoise class for comparison (don't use periodicity correction for
    # comparison with naive results above)
    cn = galsim.CorrelatedNoise(uncorr_noise_small_odd, gd, correct_periodicity=False)
    testim1 = galsim.ImageD(smallim_size_odd, smallim_size_odd)
    cn.drawImage(testim1, scale=1.)
    # Then compare the odd-sized arrays:
    np.testing.assert_array_almost_equal(
        testim1.array, cf_array, decimal=7,
        err_msg="Drawn image (odd-sized) does not match independently calculated correlated noise.")
    # Now we do even
    uncorr_noise_small = setup_uncorrelated_noise(gd, smallim_size)
    uncorr_noise_small -= uncorr_noise_small.array.mean() # Subtract mean as in CF estimates
    ft_array = np.fft.fft2(uncorr_noise_small.array)
    # Calculate the power spectrum then correlated noise
    ps_array = (ft_array * ft_array.conj()).real
    cf_array = (np.fft.ifft2(ps_array)).real / float(np.prod(np.shape(ft_array)))
    cf_array = utilities.roll2d(cf_array, (cf_array.shape[0] // 2, cf_array.shape[1] // 2))
    # Then use the CorrelatedNoise class for comparison (as above don't correct for periodicity)
    cn = galsim.CorrelatedNoise(uncorr_noise_small, gd, correct_periodicity=False)
    testim1 = galsim.ImageD(smallim_size, smallim_size)
    cn.drawImage(testim1, scale=1.)
    # Then compare the even-sized arrays:
    np.testing.assert_array_almost_equal(
        testim1.array, cf_array, decimal=7,
        err_msg="Drawn image (even-sized) does not match independently calculated correlated "+
        "noise.")


@timer
def test_output_generation_basic():
    """Test that noise generated by a CorrelatedNoise matches the correlated noise.  Averages over
    CorrelatedNoise instances from a number of realizations.
    """
    ud = galsim.UniformDeviate(rseed)
    # Get the correlated noise from an image of some x-correlated noise
    xnoise_large = make_xcorr_from_uncorr(setup_uncorrelated_noise(ud, largeim_size))
    # Note the noise we generate in these tests *is* periodic, so use non-default settings
    xnoise_large.scale = 0.18
    cn = galsim.CorrelatedNoise(xnoise_large, ud, correct_periodicity=False)
    # Note non-square and y is odd
    refim = galsim.ImageD(smallim_size, smallim_size+11)
    # Draw this for reference
    cn.drawImage(refim, scale=.18)
    # Generate a large image containing noise according to this function
    outimage = galsim.ImageD(xnoise_large.bounds, scale=0.18)
    rng2 = ud.duplicate()
    outimage.addNoise(cn)

    # Can also use applyTo syntax.
    outimage2 = galsim.ImageD(xnoise_large.bounds, scale=0.18)
    cn.rng.reset(rng2)
    cn.applyTo(outimage2)
    np.testing.assert_equal(outimage2.array, outimage.array)

    assert_raises(TypeError, cn.applyTo, outimage2.array)
    assert_raises(galsim.GalSimUndefinedBoundsError, cn.applyTo, galsim.Image())

    # Summed (average) CorrelatedNoises should be approximately equal to the input, so average
    # multiple CFs
    cn_2ndlevel = galsim.CorrelatedNoise(outimage, ud, correct_periodicity=False)
    # Draw the summed CF to an image for comparison
    testim = galsim.ImageD(smallim_size, smallim_size+11)
    cn_2ndlevel.drawImage(testim, scale=.18, add_to_image=True)
    for i in range(nsum_test - 1):
        # Then repeat
        outimage.setZero()
        outimage.addNoise(cn)
        cn_2ndlevel = galsim.CorrelatedNoise(outimage, ud, correct_periodicity=False)
        cn_2ndlevel.drawImage(testim, scale=.18, add_to_image=True)
    # Then take average
    testim /= float(nsum_test)
    np.testing.assert_array_almost_equal(
        testim.array, refim.array, decimal=2,
        err_msg="Generated noise field (basic) does not match input correlation properties.")

    assert_raises(TypeError, galsim.CorrelatedNoise)
    assert_raises(TypeError, galsim.CorrelatedNoise, outimage.array)
    assert_raises(TypeError, galsim.CorrelatedNoise, outimage, scale=1, wcs=galsim.PixelScale(3))
    assert_raises(TypeError, galsim.CorrelatedNoise, outimage, wcs=1)
    assert_raises(ValueError, galsim.CorrelatedNoise, outimage,
                  wcs=galsim.FitsWCS('fits_files/tpv.fits'))
    assert_raises(TypeError, galsim.CorrelatedNoise, outimage, rng=10)
    assert_raises(ValueError, galsim.CorrelatedNoise, outimage, x_interpolant='invalid')


@timer
def test_output_generation_rotated():
    """Test that noise generated by a rotated CorrelatedNoise matches the parent correlated noise.
    """
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
    xlargeim_size =int(np.ceil(1.41421356 * largeim_size))
    # need a very large image that will fit a large image within it, even if rotated
    ynoise_xlarge = make_ycorr_from_uncorr(setup_uncorrelated_noise(ud, xlargeim_size))
    # Subtract the mean
    ynoise_xlarge -= ynoise_xlarge.array.mean()
    cn = galsim.CorrelatedNoise(ynoise_xlarge, ud, correct_periodicity=False)
    # Then loop over some angles
    angles = [28.7 * galsim.degrees, 135. * galsim.degrees]
    for angle in angles:
        cn_rot = cn.rotate(angle)
        refim = galsim.ImageD(smallim_size, smallim_size)
        # Draw this for reference
        cn_rot.drawImage(refim, scale=1.)
        # Generate a large image containing noise according to this function
        outimage = galsim.ImageD(largeim_size, largeim_size, scale=1.)
        outimage.addNoise(cn_rot)
        # Summed (average) CorrelatedNoises should be approximately equal to the input, so avg
        # multiple CFs
        cn_2ndlevel = galsim.CorrelatedNoise(outimage, ud, subtract_mean=False,
                                             correct_periodicity=False)
        for i in range(nsum_test - 1):
            # Then repeat
            outimage.setZero()
            outimage.addNoise(cn_rot)
            cn_2ndlevel += galsim.CorrelatedNoise(outimage, ud, subtract_mean=False,
                                                  correct_periodicity=False)
        cn_2ndlevel /= float(nsum_test)
        # Then draw the summed CF to an image for comparison
        testim = galsim.ImageD(smallim_size, smallim_size)
        cn_2ndlevel.drawImage(testim, scale=1.)
        np.testing.assert_array_almost_equal(
            testim.array, refim.array, decimal=2,
            err_msg="Generated noise field (rotated) does not match input correlation properties.")


@timer
def test_output_generation_magnified():
    """Test that noise generated by a magnified CorrelatedNoise matches the parent correlated noise.
    """
    ud = galsim.UniformDeviate(rseed)
    # Get the correlated noise from an image of some y-correlated noise
    ynoise_large = make_ycorr_from_uncorr(setup_uncorrelated_noise(ud, largeim_size))
    # Get the correlated noise
    cn = galsim.CorrelatedNoise(ynoise_large, ud, correct_periodicity=False)
    refim = galsim.ImageD(smallim_size, smallim_size)
    # Draw this for reference
    cn.drawImage(refim, scale=1.)
    # Then loop over some scales, using `applyNoiseTo` with the relevant scaling in the `dx` to
    # argument check that the underlying correlated noise is preserved when both `dx` and
    # a magnification factor `scale` change in the same sense
    scales = [0.03, 11.]
    for scale in scales:
        cn_scl = cn.expand(scale)
        # Generate a large image containing noise according to this function
        outimage = galsim.ImageD(largeim_size, largeim_size, scale=scale)
        outimage.addNoise(cn_scl)
        # Summed (average) CorrelatedNoises should be approximately equal to the input, so avg
        # multiple CFs
        # Make the CorrelatedNoise object with scale=1. rather than image scale
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
        cn_2ndlevel.drawImage(testim, scale=1.)
        np.testing.assert_array_almost_equal(
            testim.array, refim.array, decimal=2,
            err_msg="Generated noise does not match (magnified) input correlation properties.")


@timer
def test_copy():
    """Check that a copied correlated noise instance correctly represents the parent correlation
    properties.
    """
    ud = galsim.UniformDeviate(rseed)
    noise_image = setup_uncorrelated_noise(ud, smallim_size)
    cn = galsim.CorrelatedNoise(noise_image, ud, subtract_mean=True, correct_periodicity=False)
    cn_copy = cn.copy()
    # Fundamental checks on RNG
    assert cn.rng is cn_copy.rng, "Copied correlated noise does not keep same RNG."
    cn_copy = cn.copy(rng=galsim.UniformDeviate(rseed + 1))
    assert cn.rng is not cn_copy.rng, "Copied correlated noise keeps same RNG despite reset."
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
    cn_copy = cn.copy(rng=cn.rng.duplicate())
    outim1.addNoise(cn)
    outim2.addNoise(cn_copy)
    # The test below does not yield *exactly* equivalent results, plausibly due to the fact that the
    # rootps_store cache is *not* copied over and when the CF is redrawn the GSObject may make
    # slight approximations.  So we'll just test at high precision:
    np.testing.assert_array_almost_equal(
        outim1.array, outim2.array, decimal=7,
        err_msg="Copied correlated noise does not produce the same noise field as the parent "+
        "despite sharing the same RNG.")
    # To illustrate the point above, we'll run a test in which the rootps is not stored in cn when
    # created, by setting correct_periodicity=True and testing at very high precision:
    outim1.setZero()
    outim2.setZero()
    cn = galsim.CorrelatedNoise(noise_image, ud, subtract_mean=True, correct_periodicity=True)
    cn_copy = cn.copy(rng=cn.rng.duplicate())
    cn = cn.copy(rng=cn.rng.duplicate())
    outim1.addNoise(cn)
    outim2.addNoise(cn_copy)
    np.testing.assert_array_almost_equal(
        outim1.array, outim2.array, decimal=12,
        err_msg="Copied correlated noise does not produce the same noise field as the parent "+
        "despite sharing the same RNG (high precision test).")

    # Check picklability
    check_pickle(cn, lambda x: (x.rng.serialize(), x.getVariance(), x.wcs))
    check_pickle(cn, drawNoise)
    check_pickle(cn)


@timer
def test_add():
    """Adding two correlated noise objects, just adds their profiles.
    """
    rng = galsim.BaseDeviate(1234)
    cosmos_scale = 0.03
    ccn = galsim.getCOSMOSNoise(rng=rng)
    print('ccn.variance = ',ccn.getVariance())
    ucn1 = galsim.UncorrelatedNoise(variance=5.e-6, scale=cosmos_scale)
    print('ucn1.variance = ',ucn1.getVariance())
    ucn2 = galsim.UncorrelatedNoise(variance=5.e-6, scale=1.)
    print('ucn2.variance = ',ucn2.getVariance())

    sum = ccn + ucn1
    print('sum.variance = ',sum.getVariance())
    np.testing.assert_allclose(sum.getVariance(), ccn.getVariance() + ucn1.getVariance())

    with assert_warns(galsim.GalSimWarning):
        sum = ccn + ucn2
    print('sum.variance = ',sum.getVariance())
    np.testing.assert_allclose(sum.getVariance(), ccn.getVariance() + ucn2.getVariance())

    diff = ccn - ucn1
    print('diff.variance = ',diff.getVariance())
    np.testing.assert_allclose(diff.getVariance(), ccn.getVariance() - ucn1.getVariance())

    with assert_warns(galsim.GalSimWarning):
        diff = ccn - ucn2
    print('diff.variance = ',diff.getVariance())
    np.testing.assert_allclose(diff.getVariance(), ccn.getVariance() - ucn2.getVariance())


@timer
def test_cosmos_and_whitening():
    """Test that noise generated by an HST COSMOS correlated noise is correct and correctly
    whitened.  Includes test for a magnified, sheared, and rotated version of the COSMOS noise, and
    tests convolution with a ground-based PSF.
    """
    gd = galsim.GaussianDeviate(rseed)
    cosmos_scale = 7.5 # Use some non-default, non-unity value of COSMOS pixel spacing
    ccn = galsim.getCOSMOSNoise(rng=gd, cosmos_scale=cosmos_scale)
    # large image to beat down noise
    # Non-square and x size is odd.
    outimage = galsim.ImageD(3 * largeim_size + 11, 3 * largeim_size, scale=cosmos_scale)
    outimage.addNoise(ccn)  # Add the COSMOS noise
    # Then estimate correlation function from generated noise
    cntest_correlated = galsim.CorrelatedNoise(outimage, ccn.rng, scale=cosmos_scale)
    # Check basic correlation function values of the 3x3 pixel region around (0,0)
    pos = galsim.PositionD(0., 0.)
    cf00 = ccn._profile.xValue(pos)
    cftest00 = cntest_correlated._profile.xValue(pos)
    # Test variances first
    np.testing.assert_almost_equal(
        cftest00 / cf00, 1., decimal=2,
        err_msg="Noise field generated with COSMOS CorrelatedNoise does not approximately match "+
        "input variance")
    # Then test (1, 0), (0, 1), (1,-1) and (1,1) values
    for xpos, ypos in zip((cosmos_scale, 0., cosmos_scale, cosmos_scale),
                          (0., cosmos_scale, -cosmos_scale, cosmos_scale)):
        pos = galsim.PositionD(xpos, ypos)
        cf = ccn._profile.xValue(pos)
        cftest = cntest_correlated._profile.xValue(pos)
        np.testing.assert_almost_equal(
            cftest / cftest00, cf / cf00, decimal=2,
            err_msg="Noise field generated with COSMOS CorrelatedNoise does not have "+
            "approximately matching interpixel covariances")
    # Now whiten the noise field, and check that its variance and covariances are as expected
    # (non-zero distance correlations ~ 0!)
    whitened_variance = ccn.whitenImage(outimage)
    cntest_whitened = galsim.CorrelatedNoise(outimage, ccn.rng) # Get the correlation function
    cftest00 = cntest_whitened._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        cftest00 / whitened_variance, 1., decimal=2,
        err_msg="Noise field generated by whitening COSMOS CorrelatedNoise does not approximately "+
        "match theoretical variance")
    # Then test (1, 0), (0, 1), (1,-1) and (1,1) values
    for xpos, ypos in zip((cosmos_scale, 0., cosmos_scale, cosmos_scale),
                          (0., cosmos_scale, -cosmos_scale, cosmos_scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest = cntest_whitened._profile.xValue(pos)
        np.testing.assert_almost_equal(
            cftest / cftest00, 0., decimal=2,
            err_msg="Noise field generated by whitening COSMOS CorrelatedNoise does not have "+
            "approximately zero interpixel covariances")
    # Now test whitening but having first expanded and sheared the COSMOS noise correlation
    ccn_transformed = ccn.shear(g1=-0.03, g2=0.07).rotate(313. * galsim.degrees).expand(3.9)
    outimage.setZero()
    outimage.addNoise(ccn_transformed)
    wht_variance = ccn_transformed.whitenImage(outimage)  # Whiten noise correlation
    cntest_whitened = galsim.CorrelatedNoise(outimage, ccn.rng) # Get the correlation function
    cftest00 = cntest_whitened._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        cftest00 / wht_variance, 1., decimal=2,
        err_msg="Noise field generated by whitening rotated, sheared, magnified COSMOS "+
        "CorrelatedNoise does not approximately match theoretical variance")
    # Then test (1, 0), (0, 1), (1,-1) and (1,1) values
    for xpos, ypos in zip((cosmos_scale, 0.,  cosmos_scale, cosmos_scale),
                          (0., cosmos_scale, -cosmos_scale, cosmos_scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest = cntest_whitened._profile.xValue(pos)
        np.testing.assert_almost_equal(
            cftest / cftest00, 0., decimal=2,
            err_msg="Noise field generated by whitening rotated, sheared, magnified COSMOS "+
            "CorrelatedNoise does not have approximately zero interpixel covariances")
    # Then convolve with a ground-based PSF and pixel, generate some more correlated noise
    # and whiten it
    scale = cosmos_scale * 9. # simulates a 0.03 arcsec * 9 = 0.27 arcsec pitch ground image
    psf_ground = galsim.Moffat(beta=3., fwhm=2.5*scale) # FWHM=0.675 arcsec seeing
    pix_ground = galsim.Pixel(scale)
    # Convolve the correlated noise field with each of the psf, pix
    ccn_convolved = ccn_transformed.convolvedWith(galsim.Convolve([psf_ground, pix_ground]))
    # Reset the outimage, and set its pixel scale to now be the ground-based resolution
    # Also, check both odd-size and non-square here.  Both should be ok.
    outimage = galsim.ImageD(3 * largeim_size + 1, 3 * largeim_size + 43)
    # Add correlated noise
    outimage.addNoise(ccn_convolved)
    # Then whiten
    # Note: Use alternate syntax here.  Equivalent to
    #       wht_variance = ccn_convolved.whitenImage(outimage)
    wht_variance = outimage.whitenNoise(ccn_convolved)
    # Then test
    cntest_whitened = galsim.CorrelatedNoise(outimage, ccn.rng) # Get the correlation function
    cftest00 = cntest_whitened._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        cftest00 / wht_variance, 1., decimal=2,
        err_msg="Noise field generated by whitening rotated, sheared, magnified, convolved COSMOS "+
        "CorrelatedNoise does not approximately match theoretical variance")
    # Then test (1, 0), (0, 1), (1,-1) and (1,1) values
    for xpos, ypos in zip((scale, 0.,  scale, scale),
                          (0., scale, -scale, scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest = cntest_whitened._profile.xValue(pos)
        np.testing.assert_almost_equal(
            cftest / cftest00, 0., decimal=2,
            err_msg="Noise field generated by whitening rotated, sheared, magnified, convolved "+
            "COSMOS CorrelatedNoise does not have approximately zero interpixel covariances")

    assert_raises(TypeError, ccn.whitenImage, outimage.array)
    assert_raises(galsim.GalSimUndefinedBoundsError, ccn.whitenImage, galsim.Image())


@timer
def test_symmetrizing(run_slow):
    """Test that noise generated by an HST COSMOS correlated noise is correctly symmetrized.
    Includes test for a magnified, sheared, and rotated version of the COSMOS noise, and tests
    convolution with a ground-based PSF.
    """
    if run_slow:
        # symm_divide determines how close to zero we need to be.  Bigger is looser tolerance.
        symm_divide = 2.
        symm_size_mult = 6 # make really huge images
    else:
        symm_divide = 3.
        symm_size_mult = 3 # make only moderately huge images

    gd = galsim.GaussianDeviate(rseed)
    cosmos_scale = 7.5 # Use some non-default, non-unity value of COSMOS pixel spacing
    ccn = galsim.getCOSMOSNoise(
        '../examples/data/acs_I_unrot_sci_20_cf.fits', gd, cosmos_scale=cosmos_scale)
    # large image to beat down noise
    outimage = galsim.ImageD(symm_size_mult * largeim_size,
                             symm_size_mult * largeim_size, scale=cosmos_scale)
    outimage.addNoise(ccn)  # Add the COSMOS noise
    outimage2 = outimage.copy()
    # Now apply 4-fold symmetry to the noise field, and check that its variance and covariances are
    # as expected (non-zero distance correlations should be symmetric)
    symmetrized_variance = ccn.symmetrizeImage(outimage, order=4)
    cntest_symmetrized = galsim.CorrelatedNoise(outimage, ccn.rng) # Get the correlation function
    cftest00 = cntest_symmetrized._profile.xValue(galsim.PositionD(0., 0.))

    # Test variances first
    np.testing.assert_almost_equal(
        (cftest00/symmetrized_variance - 1.)/symm_divide, 0, decimal=2,
        err_msg="Noise field generated by symmetrizing COSMOS CorrelatedNoise does not "+
        "approximately match theoretical variance")
    # Then test (1, 0), (0, 1), (-1,0) and (0,-1) values
    cftest = []
    for xpos, ypos in zip((cosmos_scale, 0., -cosmos_scale, 0.),
                          (0., cosmos_scale, 0., -cosmos_scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest.append(cntest_symmetrized._profile.xValue(pos))
    # They should be approximately equal, so check ratios of each one to the next one
    for ind in range(len(cftest)-1):
        np.testing.assert_almost_equal(
            (cftest[ind]/cftest[ind+1] - 1.)/symm_divide, 0., decimal=2,
            err_msg="Noise field generated by symmetrizing COSMOS CorrelatedNoise does not have "+
            "approximate 4-fold symmetry")

    # If outimage doesn't have a scale set, then it uses the ccn scale.
    outimage2.wcs = None
    symmetrized_variance2 = ccn.symmetrizeImage(outimage2, order=4)
    np.testing.assert_almost_equal(symmetrized_variance2, symmetrized_variance)

    # Now test symmetrizing, but having first expanded and sheared the COSMOS noise correlation.
    # Also we'll make the output image odd-sized, so as to ensure we test that option.
    ccn_transformed = ccn.shear(g1=-0.05, g2=0.11).rotate(313. * galsim.degrees).expand(2.1)
    # Note: symmetrize currently cannot handle non-square images.  But odd is ok.
    outimage = galsim.ImageD(symm_size_mult*largeim_size + 1,
                             symm_size_mult*largeim_size + 1, scale=cosmos_scale)
    outimage.addNoise(ccn_transformed)
    sym_variance = ccn_transformed.symmetrizeImage(outimage, order=4)  # Symmetrize noise correlation
    cntest_symmetrized = galsim.CorrelatedNoise(outimage, ccn.rng) # Get the correlation function
    cftest00 = cntest_symmetrized._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        (cftest00/sym_variance - 1.)/symm_divide, 0., decimal=2,
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
            (cftest[ind]/cftest[ind+1]-1.)/symm_divide, 0., decimal=2,
            err_msg="Noise field generated by symmetrizing rotated, sheared, magnified COSMOS "+
            "CorrelatedNoise does not have approximate 4-fold symmetry")

    # Then convolve with a ground-based PSF and pixel, generate some more correlated noise
    # and symmetrize it.  This time we'll go to higher order (20).  And we will use the image method
    # instead of the CorrelatedNoise method.
    scale = cosmos_scale * 9. # simulates a 0.03 arcsec * 9 = 0.27 arcsec pitch ground image
    psf_ground = galsim.Moffat(beta=3., fwhm=2.5*scale) # FWHM=0.675 arcsec seeing
    psf_ground = psf_ground.shear(g1=0.1)
    pix_ground = galsim.Pixel(scale)
    # Convolve the correlated noise field with each of the psf, pix
    ccn_convolved = ccn_transformed.convolvedWith(galsim.Convolve([psf_ground, pix_ground]))
    # Reset the outimage, and set its pixel scale to now be the ground-based resolution
    outimage.setZero()
    outimage.scale = scale
    # Add correlated noise
    outimage.addNoise(ccn_convolved)
    # Then symmetrize
    sym_variance = outimage.symmetrizeNoise(ccn_convolved, order=20)
    # Then test
    cntest_symmetrized = galsim.CorrelatedNoise(outimage, ccn.rng) # Get the correlation function
    cftest00 = cntest_symmetrized._profile.xValue(galsim.PositionD(0., 0.))
    # Test variances first
    np.testing.assert_almost_equal(
        (cftest00/sym_variance - 1.)/symm_divide, 0., decimal=2,
        err_msg="Noise field generated by symmetrizing rotated, sheared, magnified, convolved "+
        "COSMOS CorrelatedNoise does not approximately match theoretical variance")
    # Then test (1, 0), (0, 1), (-1, 0) and (0, -1) values
    cftest = []
    for xpos, ypos in zip((scale, 0.,  -scale, 0),
                          (0., scale, 0, -scale)):
        pos = galsim.PositionD(xpos, ypos)
        cftest.append(cntest_symmetrized._profile.xValue(pos))
    # They should be approximately equal, so check ratios of each one to the next one
    for ind in range(len(cftest)-1):
        np.testing.assert_almost_equal(
            (cftest[ind]/cftest[ind+1] - 1.)/symm_divide, 0., decimal=2,
            err_msg="Noise field generated by symmetrizing rotated, sheared, magnified, convolved "+
            "COSMOS CorrelatedNoise does not have approximate 4-fold symmetry")

    assert_raises(TypeError, ccn.symmetrizeImage)
    assert_raises(TypeError, ccn.symmetrizeImage, outimage.array)
    assert_raises(ValueError, ccn.symmetrizeImage, galsim.Image(24,20))
    assert_raises(ValueError, ccn.symmetrizeImage, outimage, order=2)
    assert_raises(ValueError, ccn.symmetrizeImage, outimage, order=5)
    assert_raises(galsim.GalSimUndefinedBoundsError, ccn.symmetrizeImage, galsim.Image())


@timer
def test_convolve_cosmos(run_slow):
    """Test that a COSMOS noise field convolved with a ground based PSF-style kernel matches the
    output of the correlated noise model modified with the convolvedWith method.
    """
    gd = galsim.GaussianDeviate(rseed)
    cosmos_scale=0.03 # Non-unity, non-default value to be used below
    cn = galsim.getCOSMOSNoise(rng=gd, cosmos_scale=cosmos_scale)
    cn = cn.withVariance(300.) # Again non-unity so as to produce ~unity output variance
    # Define a PSF with which to convolve the noise field, one WITHOUT 2-fold rotational symmetry
    # (see test_autocorrelate in test_compound.py for more info as to why this is relevant)
    # Make a relatively realistic mockup of a GREAT3 target image
    lam_over_diam_cosmos = (814.e-9 / 2.4) * (180. / np.pi) * 3600. # ~lamda/D in arcsec
    lam_over_diam_ground = lam_over_diam_cosmos * 2.4 / 4. # Generic 4m at same lambda
    psf_cosmos = galsim.Convolve([
        galsim.Airy(lam_over_diam=lam_over_diam_cosmos, obscuration=0.4), galsim.Pixel(0.05)])
    psf_ground = galsim.Convolve([
        galsim.Kolmogorov(fwhm=0.8), galsim.Pixel(0.18),
        galsim.OpticalPSF(lam_over_diam=lam_over_diam_ground, coma2=0.4, defocus=-0.6)])
    psf_shera = galsim.Convolve([
        psf_ground, (galsim.Deconvolve(psf_cosmos)).shear(g1=0.03, g2=-0.01)])
    # Then define the convolved cosmos correlated noise model
    conv_cn = cn.convolvedWith(psf_shera)
    # Then draw the correlation function for this correlated noise as the reference
    # Note: non-square and odd size for y
    refim = galsim.ImageD(smallim_size, smallim_size + 11)
    conv_cn.drawImage(refim, scale=0.18)
    # Now start the test...
    # First we generate a COSMOS noise field (cosimage), read it into an InterpolatedImage and
    # then convolve it with psf, making sure we pad the edges
    interp=galsim.Linear() # interpolation kernel to use in making convimages
    # Number of tests needs to be a little larger to beat down noise here, but see the script
    # in devel/external/test_cf/test_cf_convolution_detailed.py
    cosimage_padded = galsim.ImageD(
        (2 * smallim_size) * 6 + 355, # Note 6 here since 0.18 = 6 * 0.03
        (2 * smallim_size) * 6 + 256, # large image to beat down noise + padding
        scale=cosmos_scale)           # Use COSMOS pixel scale
    cosimage_padded.addNoise(cn) # Add cosmos noise
    # Put this noise into a GSObject and then convolve
    imobj_padded = galsim.InterpolatedImage(
        cosimage_padded, calculate_stepk=False, calculate_maxk=False,
        normalization='sb', scale=cosmos_scale, x_interpolant=interp)
    cimobj_padded = galsim.Convolve(imobj_padded, psf_shera)

    if False:  # Switch this for more rigorous tests (takes a long time!)
    #if not run_slow:
        # The convolve_cosmos test, which includes a lot of the correlated noise functionality is
        # fairly sensitive at 2dp, but takes ~200s on a mid-range laptop
        decimal_convolve_cosmos = 2
        nsum_test_convolve_cosmos = 1000
    else:
        # Basic settings for convolve_cosmos, will only catch basic screwups
        decimal_convolve_cosmos = 1
        nsum_test_convolve_cosmos = 10

    # We draw, calculate a correlation function for the resulting field, and repeat to get an
    # average over nsum_test_convolve_cosmos trials
    convimage = galsim.ImageD(2 * smallim_size, 2 * smallim_size + 11)
    cimobj_padded.drawImage(convimage, scale=0.18, method='sb')
    cn_test = galsim.CorrelatedNoise(convimage, gd, correct_periodicity=True, subtract_mean=False)
    testim = galsim.ImageD(smallim_size, smallim_size + 11)
    cn_test.drawImage(testim, scale=0.18)
    # Start some lists to store image info
    conv_list = [convimage.array.copy()] # Don't forget Python reference/assignment semantics, we
                                         # zero convimage and write over it later!
    mnsq_list = [np.mean(convimage.array**2)]
    var_list = [convimage.array.var()]
    print('start set of {0} iterations to build up the correlation function'.format(
            nsum_test_convolve_cosmos))
    for i in range(nsum_test_convolve_cosmos - 1):
        print('iteration ',i)
        cosimage_padded.setZero()
        cosimage_padded.addNoise(cn)
        imobj_padded = galsim.InterpolatedImage(
            cosimage_padded, calculate_stepk=False, calculate_maxk=False,
            normalization='sb', scale=cosmos_scale, x_interpolant=interp)
        cimobj_padded = galsim.Convolve(imobj_padded, psf_shera)
        convimage.setZero() # See above
        # Draw convolved image into convimage
        cimobj_padded.drawImage(convimage, scale=0.18, method='sb')
        conv_list.append(convimage.array.copy()) # See above
        mnsq_list.append(np.mean(convimage.array**2))
        var_list.append(convimage.array.var())
        cn_test = galsim.CorrelatedNoise(convimage, gd, correct_periodicity=True,
                                         subtract_mean=False)
        cn_test.drawImage(testim, scale=0.18, add_to_image=True)
        del imobj_padded
        del cimobj_padded
        del cn_test

    mnsq_individual = sum(mnsq_list) / float(nsum_test_convolve_cosmos)
    var_individual = sum(var_list) / float(nsum_test_convolve_cosmos)
    mnsq_individual = sum(mnsq_list) / float(nsum_test_convolve_cosmos)
    testim /= float(nsum_test_convolve_cosmos) # Take average CF of trials
    conv_array = np.asarray(conv_list)
    mnsq_all = np.mean(conv_array**2)
    var_all = conv_array.var()
    print("Mean square estimate from avg. of individual field mean squares = "+str(mnsq_individual))
    print("Mean square estimate from all fields = "+str(mnsq_all))
    print("Ratio of mean squares = %e" % (mnsq_individual / mnsq_all))
    print("Variance estimate from avg. of individual field variances = "+str(var_individual))
    print("Variance estimate from all fields = "+str(var_all))
    print("Ratio of variances = %e" % (var_individual / var_all))
    print("Zero lag CF from avg. of individual field CFs = "+str(testim.array[8, 8]))
    print("Zero lag CF in reference case = "+str(refim.array[8, 8]))
    print("Ratio of zero lag CFs = %e" % (testim.array[8, 8] / refim.array[8, 8]))
    print("Printing analysis of central 4x4 of CF:")
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print('mean diff = ',np.mean(testim.array[4:12, 4:12] - refim.array[4:12, 4:12]))
    print('var diff = ',np.var(testim.array[4:12, 4:12] - refim.array[4:12, 4:12]))
    print('min diff = ',np.min(testim.array[4:12, 4:12] - refim.array[4:12, 4:12]))
    print('max diff = ',np.max(testim.array[4:12, 4:12] - refim.array[4:12, 4:12]))
    print('mean ratio = %e' % np.mean(testim.array[4:12, 4:12] / refim.array[4:12, 4:12]))
    print('var ratio = ',np.var(testim.array[4:12, 4:12] / refim.array[4:12, 4:12]))
    print('min ratio = %e' % np.min(testim.array[4:12, 4:12] / refim.array[4:12, 4:12]))
    print('max ratio = %e' % np.max(testim.array[4:12, 4:12] / refim.array[4:12, 4:12]))

    # Test (this is a crude regression test at best, for a much more precise test of this behaviour
    # see devel/external/test_cf/test_cf_convolution_detailed.py)
    np.testing.assert_array_almost_equal(
        testim.array, refim.array, decimal=decimal_convolve_cosmos,
        err_msg="Convolved COSMOS noise fields do not match the convolved correlated noise model.")


@timer
def test_uncorrelated_noise_tracking():
    """Test that we can track various processes that convert uncorrelated noise to correlated noise.
    """
    # Start with an UncorrelatedNoise instance that we attach to an InterpolatedImage GSObject as a
    # 'noise' attribute
    gal_sigma = 1.7
    noise_var = 1.3
    seed = 1234
    pix_scale = 0.1
    orig_object = galsim.Gaussian(sigma=gal_sigma)
    orig_ucn = galsim.UncorrelatedNoise(variance=noise_var, scale=pix_scale)
    im = orig_object.drawImage(scale=pix_scale, method='no_pixel')
    int_im = galsim.InterpolatedImage(im)
    # Note, I'm including the noise attribute without actually adding noise.  It doesn't matter
    # here, we just want to check the ability of GalSim to track what happens to `noise'
    # attributes.
    int_im.noise = orig_ucn
    print('int_im.noise = ',int_im.noise)

    # Manipulate the object in various non-trivial ways: shear, magnify, rotate, convolve
    test_shear = 0.15
    test_mag = 0.92
    rot_ang = 21. # degrees
    new_int_im = int_im.shear(g1=test_shear)
    new_int_im = new_int_im.magnify(test_mag)
    new_int_im = new_int_im.rotate(rot_ang*galsim.degrees)
    print('new_int_im.noise = ',new_int_im.noise)
    new_int_im = galsim.Convolve(new_int_im, orig_object)
    print('new_int_im.noise => ',new_int_im.noise)
    final_noise = new_int_im.noise

    # Now, make a correlated noise object directly based on a realization of the original
    # uncorrelated noise object.
    test_im = galsim.Image(512,512, scale=pix_scale)
    orig_ucn.applyTo(test_im)
    cn = galsim.CorrelatedNoise(test_im, galsim.BaseDeviate(seed))

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
    np.testing.assert_almost_equal(
            final_noise.getVariance(), new_cn.getVariance(), decimal=3,
            err_msg='Failure in tracking noise properties through operations')

    # Convolving two objects with noise works fine, but accessing the resulting noise attribute
    # leads to a warning.
    conv_obj = galsim.Convolve(int_im, int_im)
    with assert_warns(galsim.GalSimWarning):
        noise = conv_obj.noise
    # The noise should be correlated, not just the original UncorrelatedNoise
    assert isinstance(noise, galsim.BaseCorrelatedNoise)
    assert not isinstance(noise, galsim.UncorrelatedNoise)


@timer
def test_variance_changes():
    """Test that we can change and check the variance for CorrelatedNoise objects.
    """
    # Make an UncorrelatedNoise object.
    noise_var = 1.24
    seed = 1234
    pix_scale = 0.1
    orig_ucn = galsim.UncorrelatedNoise(noise_var, rng=galsim.BaseDeviate(seed), scale=pix_scale)
    # Reset variance to something else.
    new_var = 1.07
    ucn = orig_ucn.withVariance(new_var)
    np.testing.assert_equal(ucn.getVariance(), new_var,
                            err_msg='Failure to reset and then get variance for UncorrelatedNoise')

    # Now do this for a CorrelatedNoise object.
    gd = galsim.GaussianDeviate()
    cosmos_scale=0.03
    cn = galsim.getCOSMOSNoise(rng=gd, cosmos_scale=cosmos_scale)
    cn = cn.withVariance(new_var)
    np.testing.assert_equal(cn.getVariance(), new_var,
                            err_msg='Failure to reset and then get variance for CorrelatedNoise')

    # Also directly using from_file
    file_name = '../share/acs_I_unrot_sci_20_cf.fits'
    cn = galsim.BaseCorrelatedNoise.from_file(file_name, cosmos_scale, rng=gd, variance=new_var)
    np.testing.assert_equal(cn.getVariance(), new_var,
                            err_msg='Failure to set variance with from_file')

    # Also check some errors here
    assert_raises(ValueError, cn.withVariance, -1.0)
    assert_raises(OSError, galsim.getCOSMOSNoise, file_name='not_a_file')
    assert_raises(OSError, galsim.getCOSMOSNoise, file_name='config_input/catalog.fits')
    assert_raises(TypeError, galsim.getCOSMOSNoise, rng='invalid')
    assert_raises(ValueError, galsim.getCOSMOSNoise, variance = -1.0)
    assert_raises(ValueError, galsim.getCOSMOSNoise, x_interpolant='invalid')
    assert_raises(OSError, galsim.BaseCorrelatedNoise.from_file, 'not_a_file', 0.1)
    # Image must be square:
    assert_raises(galsim.GalSimError, galsim.BaseCorrelatedNoise.from_file,
                  os.path.join('real_comparison_images','AEGIS_F606w_images_01.fits'), 0.1)
    # Image must have odd sides:
    assert_raises(galsim.GalSimError, galsim.BaseCorrelatedNoise.from_file,
                  os.path.join('real_comparison_images','AEGIS_F606w_PSF_images_01.fits'), 0.1)
    # Image must be rotationally symmetric
    assert_raises(galsim.GalSimError, galsim.BaseCorrelatedNoise.from_file,
                  os.path.join('real_comparison_images', 'shera_target_PSF.fits'), 0.1)
    # Image must be centrally peaked
    im = galsim.fits.read(file_name)
    im[im.center] *= 0.1
    bad_center_file_name = os.path.join('fits_files','test_acs_bad_center_cf.fits')
    im.write(bad_center_file_name)
    assert_raises(galsim.GalSimError, galsim.BaseCorrelatedNoise.from_file,
                  bad_center_file_name, 0.1)


@timer
def test_cosmos_wcs(run_slow):
    """Test how getCOSMOSNoise works when applied to an image with a WCS.
    """
    var = 1.7
    rng = galsim.BaseDeviate(8675309)
    cn_cosmos = galsim.getCOSMOSNoise(rng=rng, variance=var)
    cosmos_scale = 0.03

    # Shear it significantly to amplify the directionality of the correlations.
    cn_orig = cn_cosmos.shear(e2=0.7)

    test_wcs_list = [
        galsim.PixelScale(cosmos_scale),                            # Same as original
        #galsim.PixelScale(2.*cosmos_scale),                         # 2x larger pixels
        galsim.PixelScale(0.5*cosmos_scale),                        # 2x smaller pixels
        galsim.JacobianWCS(0., cosmos_scale, -cosmos_scale, 0.),    # 90 degrees rotated
    ]
    # The test doesn't work for the big pixels.  When drawing on such an image, there
    # is too much information lost by sampling on the effectively larger grid, so the
    # measured correlation function is unable to recover the original noise correlations
    # correctly.  I don't think this is a problem, so don't include that in the test list.

    # This test isn't super fast, so for regular unit tests, just do the last one.
    if not run_slow:
        test_wcs_list = test_wcs_list[-1:]

    for k, test_wcs in enumerate(test_wcs_list):
        print(test_wcs)
        test_im = galsim.ImageD(3 * largeim_size, 3 * largeim_size, wcs=test_wcs)

        # This adds the noise respecting the different WCS functions in the two cases.
        test_im.addNoise(cn_orig)
        cn_test = galsim.CorrelatedNoise(test_im)

        # The "raw" correlation function treats the image as having the cosmos pixel_scale
        cn_raw = galsim.CorrelatedNoise(test_im.view(scale=cosmos_scale))

        # Check basic correlation function values of the 3x3 pixel region around (0,0)
        for xpos, ypos in zip((0., cosmos_scale, 0., cosmos_scale, cosmos_scale),
                              (0., 0., cosmos_scale, -cosmos_scale, cosmos_scale)):
            pos = galsim.PositionD(xpos, ypos)
            cf_orig = cn_orig._profile.xValue(pos)
            cf_test = cn_test._profile.xValue(pos)
            cf_raw = cn_raw._profile.xValue(pos)
            print(pos, cf_orig, cf_test, cf_raw)
            np.testing.assert_almost_equal(
                    cf_orig/var, cf_test/var, decimal=2,
                    err_msg='Drawing COSMOS noise on image with WCS did not ' +
                    'recover correct covariance at positions '+str(pos))

        # Repeat, but this time adding the noise to a view with the cosmos pixel scale
        test_im.setZero()
        test_im.view(wcs=cn_orig.wcs).addNoise(cn_orig)
        cn_test = galsim.CorrelatedNoise(test_im)
        cn_raw = galsim.CorrelatedNoise(test_im, wcs=galsim.PixelScale(cosmos_scale))

        # This time it is the raw cf values that should match.
        for xpos, ypos in zip((0., cosmos_scale, 0., cosmos_scale, cosmos_scale),
                              (0., 0., cosmos_scale, -cosmos_scale, cosmos_scale)):
            pos = galsim.PositionD(xpos, ypos)
            cf_orig = cn_orig._profile.xValue(pos)
            cf_test = cn_test._profile.xValue(pos)
            cf_raw = cn_raw._profile.xValue(pos)
            print(pos, cf_orig, cf_test, cf_raw)
            np.testing.assert_almost_equal(
                    cf_orig/var, cf_raw/var, decimal=2,
                    err_msg='Drawing COSMOS noise on view with cosmos pixel scale did not '+
                    'recover correct covariance at positions '+str(pos))

        # Check picklability
        check_pickle(cn_test, lambda x: (x.rng.serialize(), x.getVariance(), x.wcs))
        check_pickle(cn_test, drawNoise)
        check_pickle(cn_test)


@timer
def test_covariance_spectrum():
    """Just do some pickling tests of CovarianceSpectrum."""
    bd = galsim.BaseDeviate(rseed)
    Sigma = {}
    for i in range(2):
        for j in range(2):
            if i > j: continue
            Sigma[(i, j)] = galsim.Gaussian(fwhm=1)  # anything with a drawKImage will do...
    SEDs = [galsim.SED('1', 'nm', 'fphotons'), galsim.SED('wave', 'nm', 'fphotons')]
    covspec = galsim.CovarianceSpectrum(Sigma, SEDs)

    check_pickle(covspec)

    wcs = galsim.PixelScale(0.1)
    psf = galsim.Gaussian(fwhm=1)
    bp = galsim.Bandpass('1', 'nm', blue_limit=500.0, red_limit=600.0)
    check_pickle(covspec, lambda x: x.toNoise(bp, psf, wcs, rng=bd))

    covspec = covspec.transform(1.1, 0.2, 0.1, 0.9)
    check_pickle(covspec)
    check_pickle(covspec, lambda x: x.toNoise(bp, psf, wcs, rng=bd))


@timer
def test_gsparams():
    """Test withGSParams
    """
    rng = galsim.BaseDeviate(1234)
    ucn = galsim.UncorrelatedNoise(rng=rng, variance=1.e3)
    gsp = galsim.GSParams(folding_threshold=1.e-4, maxk_threshold=1.e-4, maximum_fft_size=1.e4)

    ucn1 = ucn.withGSParams(gsp)
    ucn2 = galsim.UncorrelatedNoise(rng=rng, variance=1.e3, gsparams=gsp)
    ucn3 = ucn.withGSParams(folding_threshold=1.e-4, maxk_threshold=1.e-4, maximum_fft_size=1.e4)
    print('ucn1 = ',repr(ucn1))
    print('ucn2 = ',repr(ucn2))
    print('ucn3 = ',repr(ucn3))
    assert ucn != ucn1
    assert ucn1 == ucn2
    assert ucn1 == ucn3
    assert ucn2 == ucn3
    assert ucn.withGSParams(ucn.gsparams) is ucn
    assert ucn1.withGSParams(ucn.gsparams) is not ucn
    assert ucn1.withGSParams(ucn.gsparams) == ucn

    ccn = galsim.getCOSMOSNoise(rng=rng)

    ccn1 = ccn.withGSParams(gsp)
    ccn2 = galsim.getCOSMOSNoise(rng=rng, gsparams=gsp)
    ccn3 = ccn.withGSParams(folding_threshold=1.e-4, maxk_threshold=1.e-4, maximum_fft_size=1.e4)
    assert ccn != ccn1
    assert ccn1 == ccn2
    assert ccn1 == ccn3
    assert ccn2 == ccn3
    assert ccn.withGSParams(ccn.gsparams) is ccn
    assert ccn1.withGSParams(ccn.gsparams) is not ccn
    assert ccn1.withGSParams(ccn.gsparams) == ccn


if __name__ == "__main__":
    runtests(__file__)
