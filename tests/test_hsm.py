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

"""Unit tests for the PSF correction and shear estimation routines.

There are two types of tests: tests that use Gaussian profiles, for which the ideal results are
known; and tests that use real galaxies in SDSS for which results were tabulated using the same code
before it was integrated into GalSim (so we can make sure we are not breaking anything as we modify
the code).
"""

import os
import numpy as np
import math
import glob

import galsim
from galsim_test_helpers import *


# define a range of input parameters for the Gaussians that we are testing
gaussian_sig_values = [0.5, 1.0, 2.0]
shear_values = [0.01, 0.03, 0.10, 0.30]
pixel_scale = 0.2
decimal = 2 # decimal place at which to require equality in sizes
decimal_shape = 3 # decimal place at which to require equality in shapes

# The timing tests can be unreliable in environments with other processes running at the
# same time.  So we disable them by default.  However, on a clean system, they should all pass.
test_timing = False

# define inputs and expected results for tests that use real SDSS galaxies
img_dir = os.path.join(".","HSM_precomputed")
gal_file_prefix = "image."
psf_file_prefix = "psf."
img_suff = ".fits"
file_indices = [0, 2, 4, 6, 8]
x_centroid = [35.888, 19.44, 8.74, 20.193, 57.94]
y_centroid = [19.845, 25.047, 11.92, 38.93, 27.73]
sky_var = [35.01188, 35.93418, 35.15456, 35.11146, 35.16454]
correction_methods = ["KSB", "BJ", "LINEAR", "REGAUSS"]
# Note: expected results give shear for KSB and distortion for others, but the results below have
# converted KSB expected results to distortion for the sake of consistency
e1_expected = np.array([
        [0.467603106752, 0.381211727, 0.398856937, 0.401755571],
        [0.28618443944, 0.199222784, 0.233883543, 0.234257525],
        [0.271533794146, 0.158049396, 0.183517068, 0.184893412],
        [-0.293754156071, -0.457024541, 0.123946584, -0.609233462],
        [0.557720893779, 0.374143023, 0.714147448, 0.435404409] ])
e2_expected = np.array([
        [-0.867225166489, -0.734855778, -0.777027588, -0.774684891],
        [-0.469354341577, -0.395520479, -0.502540961, -0.464466257],
        [-0.519775291311, -0.471589061, -0.574750641, -0.529664935],
        [0.345688365839, -0.342047099, 0.120603755, -0.44609129428863525],
        [0.525728304099, 0.370691830, 0.702724807, 0.433999442] ])
resolution_expected = np.array([
        [0.796144249, 0.835624917, 0.835624917, 0.827796187],
        [0.685023735, 0.699602704, 0.699602704, 0.659457638],
        [0.634736458, 0.651040481, 0.651040481, 0.614663396],
        [0.477027015, 0.477210752, 0.477210752, 0.423157447],
        [0.595205998, 0.611824797, 0.611824797, 0.563582092] ])
sigma_e_expected = np.array([
        [0.016924826, 0.014637648, 0.014637648, 0.014465546],
        [0.075769504, 0.073602324, 0.073602324, 0.064414520],
        [0.110253112, 0.106222900, 0.106222900, 0.099357106],
        [0.185276702, 0.184300955, 0.184300955, 0.173478300],
        [0.073020065, 0.070270966, 0.070270966, 0.061856263] ])

def equal_hsmshapedata(res1, res2):
    """Utility to check that all entries in two ShapeData objects are equal."""
    if not isinstance(res1, galsim.hsm.ShapeData) or not isinstance(res2, galsim.hsm.ShapeData):
        raise TypeError("Objects that were passed in are not ShapeData objects")

    if res1.corrected_e1 != res2.corrected_e1: return False
    if res1.corrected_e2 != res2.corrected_e2: return False
    if res1.corrected_g1 != res2.corrected_g1: return False
    if res1.corrected_g2 != res2.corrected_g2: return False
    if res1.corrected_shape_err != res2.corrected_shape_err: return False
    if res1.error_message != res2.error_message: return False
    if res1.image_bounds.xmin != res2.image_bounds.xmin: return False
    if res1.image_bounds.xmax != res2.image_bounds.xmax: return False
    if res1.image_bounds.ymin != res2.image_bounds.ymin: return False
    if res1.image_bounds.ymax != res2.image_bounds.ymax: return False
    if res1.meas_type != res2.meas_type: return False
    if res1.moments_amp != res2.moments_amp: return False
    if res1.moments_centroid != res2.moments_centroid: return False
    if res1.moments_n_iter != res2.moments_n_iter: return False
    if res1.moments_rho4 != res2.moments_rho4: return False
    if res1.moments_sigma != res2.moments_sigma: return False
    if res1.moments_status != res2.moments_status: return False
    if res1.observed_shape != res2.observed_shape: return False
    if res1.resolution_factor != res2.resolution_factor: return False
    return True


@timer
def test_moments_basic():
    """Test that we can properly recover adaptive moments for Gaussians."""
    first_test=True
    for sig in gaussian_sig_values:
        for g1 in shear_values:
            for g2 in shear_values:
                total_shear = np.sqrt(g1**2 + g2**2)
                conversion_factor = np.tanh(2.0*math.atanh(total_shear))/total_shear
                distortion_1 = g1*conversion_factor
                distortion_2 = g2*conversion_factor
                gal = galsim.Gaussian(flux = 1.0, sigma = sig)
                gal = gal.shear(g1=g1, g2=g2)
                gal_image = gal.drawImage(scale = pixel_scale, method='no_pixel')
                result = gal_image.FindAdaptiveMom()
                # make sure we find the right Gaussian sigma
                np.testing.assert_almost_equal(np.fabs(result.moments_sigma-sig/pixel_scale), 0.0,
                                               err_msg = "- incorrect dsigma", decimal = decimal)
                # make sure we find the right e
                np.testing.assert_almost_equal(result.observed_shape.e1,
                                               distortion_1, err_msg = "- incorrect e1",
                                               decimal = decimal_shape)
                np.testing.assert_almost_equal(result.observed_shape.e2,
                                               distortion_2, err_msg = "- incorrect e2",
                                               decimal = decimal_shape)

                # test for moments with a circular weight function
                result_round = gal_image.FindAdaptiveMom(round_moments=True)

                # The ellipticities calculated here are we expect when integrating an
                # elliptical Gaussian light profile with a round Gaussian weight function
                # with the same sigma.
                q = np.exp(-2.*math.atanh(total_shear))
                theta = 0.5*math.atan2(g2, g1)
                e_round = (1 - q**2)/(1 + q**2 + 2*q)
                e1_round = e_round*math.cos(2*theta)
                e2_round = e_round*math.sin(2*theta)

                np.testing.assert_almost_equal(np.fabs(result_round.moments_sigma-sig/pixel_scale), 0.0,
                                               err_msg = "- incorrect round sigma", decimal = decimal)
                np.testing.assert_almost_equal(result_round.observed_shape.e1,
                                               e1_round, err_msg = "- incorrect round e1",
                                               decimal = decimal_shape)
                np.testing.assert_almost_equal(result_round.observed_shape.e2,
                                               e2_round, err_msg = "- incorrect round e2",
                                               decimal = decimal_shape)

                # if this is the first time through this loop, just make sure it runs and gives the
                # same result whether const or not.
                if first_test:
                    result = gal_image.view().FindAdaptiveMom()
                    first_test=False
                    np.testing.assert_almost_equal(
                        np.fabs(result.moments_sigma-sig/pixel_scale), 0.0,
                        err_msg = "- incorrect dsigma (ImageView)", decimal = decimal)
                    np.testing.assert_almost_equal(
                        result.observed_shape.e1,
                        distortion_1, err_msg = "- incorrect e1 (ImageView)",
                        decimal = decimal_shape)
                    np.testing.assert_almost_equal(
                        result.observed_shape.e2,
                        distortion_2, err_msg = "- incorrect e2 (ImageView)",
                        decimal = decimal_shape)
                    result = gal_image.view(make_const=True).FindAdaptiveMom()
                    np.testing.assert_almost_equal(
                        np.fabs(result.moments_sigma-sig/pixel_scale), 0.0,
                        err_msg = "- incorrect dsigma (make_const=True)", decimal = decimal)
                    np.testing.assert_almost_equal(
                        result.observed_shape.e1,
                        distortion_1, err_msg = "- incorrect e1 (make_const=True)",
                        decimal = decimal_shape)
                    np.testing.assert_almost_equal(
                        result.observed_shape.e2,
                        distortion_2, err_msg = "- incorrect e2 (make_const=True)",
                        decimal = decimal_shape)

@timer
def test_moments_wcs():
    """Test adaptive moments for Gaussians when the wcs is non-trivial."""

    wcs_list = [
        galsim.PixelScale(0.2),
        galsim.JacobianWCS(0.2, 0.03, -0.03, 0.2),
        galsim.JacobianWCS(0.23, 0.03, -0.06, 0.19),
        galsim.AffineTransform(0.23, 0.03, 0.06, -0.19,
                               origin=galsim.PositionD(75,105),
                               world_origin=galsim.PositionD(3475,1005)),
        galsim.TanWCS(galsim.AffineTransform(0.03, 0.23, 0.21, -0.01),
                      galsim.CelestialCoord(74.2 * galsim.degrees, -32 * galsim.degrees)),
    ]

    rng = galsim.BaseDeviate(1234)

    for sig in gaussian_sig_values:
        for g1 in shear_values:
            for g2 in shear_values:
                for wcs in wcs_list:
                    du = rng.np.uniform(-2,2)
                    dv = rng.np.uniform(-2,2)
                    gal = galsim.Gaussian(sigma=sig).shear(g1=g1, g2=g2).shift(du,dv)
                    image = gal.drawImage(nx=200, ny=200, wcs=wcs, method='no_pixel')
                    result = image.FindAdaptiveMom()

                    # Convert to sky coords
                    result = result.applyWCS(wcs, image_pos=image.true_center)
                    print(result.moments_sigma, sig,
                          result.observed_shape.g1, g1,
                          result.observed_shape.g2, g2,
                          result.moments_centroid.x, du,
                          result.moments_centroid.y, dv)
                    np.testing.assert_allclose(result.moments_sigma, sig, rtol=1.e-5)
                    np.testing.assert_allclose(result.observed_shape.g1, g1, rtol=1.e-5)
                    np.testing.assert_allclose(result.observed_shape.g2, g2, rtol=1.e-5)
                    np.testing.assert_allclose(result.moments_centroid.x, du, atol=1.e-7)
                    np.testing.assert_allclose(result.moments_centroid.y, dv, atol=1.e-7)

                    # Can also do this directly with FindAdaptiveMom(use_sky_coords=True)
                    result = image.FindAdaptiveMom(use_sky_coords=True)
                    np.testing.assert_allclose(result.moments_sigma, sig, rtol=1.e-5)
                    np.testing.assert_allclose(result.observed_shape.g1, g1, rtol=1.e-5)
                    np.testing.assert_allclose(result.observed_shape.g2, g2, rtol=1.e-5)
                    np.testing.assert_allclose(result.moments_centroid.x, du, atol=1.e-7)
                    np.testing.assert_allclose(result.moments_centroid.y, dv, atol=1.e-7)


@timer
def test_shearest_basic():
    """Test that we can recover shears for Gaussian galaxies and PSFs."""
    for sig in gaussian_sig_values:
        for g1 in shear_values:
            for g2 in shear_values:
                total_shear = np.sqrt(g1**2 + g2**2)
                conversion_factor = np.tanh(2.0*math.atanh(total_shear))/total_shear
                distortion_1 = g1*conversion_factor
                distortion_2 = g2*conversion_factor
                gal = galsim.Gaussian(flux = 1.0, sigma = sig)
                psf = galsim.Gaussian(flux = 1.0, sigma = sig)
                gal = gal.shear(g1=g1, g2=g2)
                psf = psf.shear(g1=0.1*g1, g2=0.05*g2)
                final = galsim.Convolve([gal, psf])
                final_image = final.drawImage(scale=pixel_scale, method='no_pixel')
                epsf_image = psf.drawImage(scale=pixel_scale, method='no_pixel')
                result = galsim.hsm.EstimateShear(final_image, epsf_image)
                # make sure we find the right e after PSF correction
                # with regauss, which returns a distortion
                np.testing.assert_almost_equal(result.corrected_e1,
                                               distortion_1, err_msg = "- incorrect e1",
                                               decimal = decimal_shape)
                np.testing.assert_almost_equal(result.corrected_e2,
                                               distortion_2, err_msg = "- incorrect e2",
                                               decimal = decimal_shape)


@timer
def test_shearest_precomputed():
    """Test that we can recover shears the same as before the code was put into GalSim."""
    # loop over real galaxies
    for index in range(len(file_indices)):
        # define input filenames
        img_file = os.path.join(img_dir, gal_file_prefix + str(file_indices[index]) + img_suff)
        psf_file = os.path.join(img_dir, psf_file_prefix + str(file_indices[index]) + img_suff)

        # read in information for objects and expected results
        imgR = galsim.fits.read(img_file)
        # perform a cast to int as the images on file are unsigned,
        # which leads to problems when we subtract 1000 below
        img  = galsim.Image(imgR, dtype = int)
        img -= 1000
        psfR = galsim.fits.read(psf_file)
        psf = galsim.Image(psfR, dtype = np.int16)
        psf -= 1000

        # get PSF moments for later tests
        psf_mom = psf.FindAdaptiveMom()

        # loop over methods
        for method_index in range(len(correction_methods)):
            # call PSF correction
            result = galsim.hsm.EstimateShear(
                img, psf, sky_var = sky_var[index], shear_est = correction_methods[method_index],
                guess_centroid = galsim.PositionD(x_centroid[index], y_centroid[index]))

            # compare results with precomputed
            print(result.meas_type, correction_methods[method_index])
            if result.meas_type == 'e':
                np.testing.assert_almost_equal(
                    result.corrected_e1, e1_expected[index][method_index], decimal = decimal_shape)
                np.testing.assert_almost_equal(
                    result.corrected_e2, e2_expected[index][method_index], decimal = decimal_shape)
            else:
                gval = np.sqrt(result.corrected_g1**2 + result.corrected_g2**2)
                if gval <= 1.0:
                    s = galsim.Shear(g1=result.corrected_g1, g2=result.corrected_g2)
                    np.testing.assert_almost_equal(
                        s.e1, e1_expected[index][method_index], decimal = decimal_shape)
                    np.testing.assert_almost_equal(
                        s.e2, e2_expected[index][method_index], decimal = decimal_shape)
            # also compare resolutions and estimated errors
            np.testing.assert_almost_equal(
                result.resolution_factor, resolution_expected[index][method_index],
                decimal = decimal_shape)
            np.testing.assert_almost_equal(
                result.corrected_shape_err, sigma_e_expected[index][method_index],
                decimal = decimal_shape)
            # Also check that the PSF properties that come out of EstimateShear are the same as
            # what we would get from measuring directly.
            np.testing.assert_almost_equal(
                psf_mom.moments_sigma, result.psf_sigma, decimal=decimal_shape,
                err_msg = "PSF sizes from FindAdaptiveMom vs. EstimateShear disagree")
            np.testing.assert_almost_equal(
                psf_mom.observed_shape.e1, result.psf_shape.e1, decimal=decimal_shape,
                err_msg = "PSF e1 from FindAdaptiveMom vs. EstimateShear disagree")
            np.testing.assert_almost_equal(
                psf_mom.observed_shape.e2, result.psf_shape.e2, decimal=decimal_shape,
                err_msg = "PSF e2 from FindAdaptiveMom vs. EstimateShear disagree")
            first = False


@timer
def test_masks():
    """Test that moments and shear estimation routines respond appropriately to masks."""
    # set up some toy galaxy and PSF
    my_sigma = 1.0
    my_pixscale = 0.1
    my_g1 = 0.15
    my_g2 = -0.4
    imsize = 256
    g = galsim.Gaussian(sigma = my_sigma)
    p = galsim.Gaussian(sigma = my_sigma) # the ePSF is Gaussian (kind of silly but it means we can
                                     # predict results exactly)
    g = g.shear(g1=my_g1, g2=my_g2)
    obj = galsim.Convolve(g, p)
    im = galsim.ImageF(imsize, imsize)
    p_im = galsim.ImageF(imsize, imsize)
    obj.drawImage(image=im, scale=my_pixscale, method='no_pixel')
    p.drawImage(image=p_im, scale=my_pixscale, method='no_pixel')

    # make some screwy weight and badpix images that should cause issues, and check that the
    # exception is thrown
    good_weight_im = galsim.ImageI(imsize, imsize, init_value=1)
    ## different size from image
    weight_im = galsim.ImageI(imsize, 2*imsize)
    assert_raises(ValueError, galsim.hsm.FindAdaptiveMom, im, weight_im)
    assert_raises(ValueError, galsim.hsm.EstimateShear, im, p_im, weight_im)
    badpix_im = galsim.ImageI(imsize, 2*imsize)
    assert_raises(ValueError, galsim.hsm.FindAdaptiveMom, im, badpix_im)
    assert_raises(ValueError, galsim.hsm.EstimateShear, im, p_im, good_weight_im, badpix_im)
    ## weird values
    weight_im = galsim.ImageI(imsize, imsize, init_value = -3)
    assert_raises(ValueError, galsim.hsm.FindAdaptiveMom, im, weight_im)
    assert_raises(ValueError, galsim.hsm.EstimateShear, im, p_im, weight_im)
    ## excludes all pixels
    weight_im = galsim.ImageI(imsize, imsize)
    assert_raises(galsim.GalSimError, galsim.hsm.FindAdaptiveMom, im, weight_im)
    assert_raises(galsim.GalSimError, galsim.hsm.EstimateShear, im, p_im, weight_im)
    badpix_im = galsim.ImageI(imsize, imsize, init_value = -1)
    assert_raises(galsim.GalSimError, galsim.hsm.FindAdaptiveMom, im, good_weight_im, badpix_im)
    assert_raises(galsim.GalSimError, galsim.hsm.EstimateShear, im, p_im, good_weight_im, badpix_im)

    # check moments, shear without mask
    resm = im.FindAdaptiveMom()
    ress = galsim.hsm.EstimateShear(im, p_im)

    # check moments, shear with weight image that includes all pixels
    weightall1 = galsim.ImageI(imsize, imsize, init_value = 1)
    resm_weightall1 = im.FindAdaptiveMom(weightall1)
    ress_weightall1 = galsim.hsm.EstimateShear(im, p_im, weightall1)

    # We'll do this series of tests a few times, so encapsulate the code here.
    def check_equal(resm, ress, resm_test, ress_test, tag):
        np.testing.assert_equal(resm.observed_shape.e1, resm_test.observed_shape.e1,
            err_msg="e1 from FindAdaptiveMom changes "+tag)
        np.testing.assert_equal(resm.observed_shape.e2, resm_test.observed_shape.e2,
            err_msg="e2 from FindAdaptiveMom changes "+tag)
        np.testing.assert_equal(resm.moments_sigma, resm_test.moments_sigma,
            err_msg="sigma from FindAdaptiveMom changes "+tag)
        np.testing.assert_equal(ress.observed_shape.e1, ress_test.observed_shape.e1,
            err_msg="observed e1 from EstimateShear changes "+tag)
        np.testing.assert_equal(ress.observed_shape.e2, ress_test.observed_shape.e2,
            err_msg="observed e2 from EstimateShear changes "+tag)
        np.testing.assert_equal(ress.moments_sigma, ress_test.moments_sigma,
            err_msg="observed sigma from EstimateShear changes "+tag)
        np.testing.assert_equal(ress.corrected_e1, ress_test.corrected_e1,
            err_msg="corrected e1 from EstimateShear changes "+tag)
        np.testing.assert_equal(ress.corrected_e2, ress_test.corrected_e2,
            err_msg="corrected e2 from EstimateShear changes "+tag)
        np.testing.assert_equal(ress.resolution_factor, ress_test.resolution_factor,
            err_msg="resolution factor from EstimateShear changes "+tag)
    check_equal(resm,ress,resm_weightall1,ress_weightall1, "when using inclusive weight")

    # check moments and shears with mask of edges, should be nearly the same
    # (this seems dumb, but it's helpful for keeping track of whether the pointers in the C++ code
    # are being properly updated despite the masks.  If we monkey in that code again, it will be a
    # useful check.)
    maskedge = galsim.ImageI(imsize, imsize, init_value = 1)
    xmin = maskedge.xmin
    xmax = maskedge.xmax
    ymin = maskedge.ymin
    ymax = maskedge.ymax
    edgenum = 3
    for ind1 in range(xmin, xmax+1):
        for ind2 in range(ymin, ymax+1):
            if (ind1 <= (xmin+edgenum)) or (ind1 >= (xmax-edgenum)) or (ind2 <= (ymin+edgenum)) or (ind2 >= (ymax-edgenum)):
                maskedge.setValue(ind1, ind2, 0)
    resm_maskedge = im.FindAdaptiveMom(maskedge)
    ress_maskedge = galsim.hsm.EstimateShear(im, p_im, maskedge)
    test_decimal = 4
    np.testing.assert_almost_equal(resm.observed_shape.e1, resm_maskedge.observed_shape.e1,
        decimal=test_decimal, err_msg="e1 from FindAdaptiveMom changes when masking edge")
    np.testing.assert_almost_equal(resm.observed_shape.e2, resm_maskedge.observed_shape.e2,
        decimal=test_decimal, err_msg="e2 from FindAdaptiveMom changes when masking edge")
    np.testing.assert_almost_equal(resm.moments_sigma, resm_maskedge.moments_sigma,
        decimal=test_decimal, err_msg="sigma from FindAdaptiveMom changes when masking edge")
    np.testing.assert_almost_equal(ress.observed_shape.e1, ress_maskedge.observed_shape.e1,
        decimal=test_decimal, err_msg="observed e1 from EstimateShear changes when masking edge")
    np.testing.assert_almost_equal(ress.observed_shape.e2, ress_maskedge.observed_shape.e2,
        decimal=test_decimal, err_msg="observed e2 from EstimateShear changes when masking edge")
    np.testing.assert_almost_equal(ress.moments_sigma, ress_maskedge.moments_sigma,
        decimal=test_decimal,
        err_msg="observed sigma from EstimateShear changes when masking edge")
    np.testing.assert_almost_equal(ress.corrected_e1, ress_maskedge.corrected_e1,
        decimal=test_decimal,
        err_msg="corrected e1 from EstimateShear changes when masking edge")
    np.testing.assert_almost_equal(ress.corrected_e2, ress_maskedge.corrected_e2,
        decimal=test_decimal,
        err_msg="corrected e2 from EstimateShear changes when masking edge")
    np.testing.assert_almost_equal(ress.resolution_factor, ress_maskedge.resolution_factor,
        decimal=test_decimal,
        err_msg="resolution factor from EstimateShear changes when masking edge")

    # check that results don't change *at all* i.e. using assert_equal when we do this edge masking
    # in different ways:
    ## do the same as the previous test, but with weight map that is floats (0.0 or 1.0)
    maskedge = galsim.ImageF(imsize, imsize, init_value = 1.)
    for ind1 in range(xmin, xmax+1):
        for ind2 in range(ymin, ymax+1):
            if (ind1 <= (xmin+edgenum)) or (ind1 >= (xmax-edgenum)) or (ind2 <= (ymin+edgenum)) or (ind2 >= (ymax-edgenum)):
                maskedge.setValue(ind1, ind2, 0.)
    resm_maskedge1 = im.FindAdaptiveMom(maskedge)
    ress_maskedge1 = galsim.hsm.EstimateShear(im, p_im, maskedge)
    check_equal(resm_maskedge,ress_maskedge,resm_maskedge1,ress_maskedge1,
                "when masking with floats")

    ## make the weight map for allowed pixels a nonzero value that also != 1
    maskedge = galsim.ImageF(imsize, imsize, init_value = 2.3)
    for ind1 in range(xmin, xmax+1):
        for ind2 in range(ymin, ymax+1):
            if (ind1 <= (xmin+edgenum)) or (ind1 >= (xmax-edgenum)) or (ind2 <= (ymin+edgenum)) or (ind2 >= (ymax-edgenum)):
                maskedge.setValue(ind1, ind2, 0.)
    resm_maskedge1 = im.FindAdaptiveMom(maskedge)
    ress_maskedge1 = galsim.hsm.EstimateShear(im, p_im, maskedge)
    check_equal(resm_maskedge,ress_maskedge,resm_maskedge1,ress_maskedge1,
                "when masking with floats != 1")

    ## make the weight map all equal to 1, and use a badpix map with a range of nonzero values
    maskedge = galsim.ImageI(imsize, imsize, init_value = 1)
    badpixedge = galsim.ImageI(imsize, imsize, init_value = 0)
    for ind1 in range(xmin, xmax+1):
        for ind2 in range(ymin, ymax+1):
            if (ind1 <= (xmin+edgenum)) or (ind1 >= (xmax-edgenum)) or (ind2 <= (ymin+edgenum)) or (ind2 >= (ymax-edgenum)):
                badpixedge.setValue(ind1, ind2, ind1+1)
    resm_maskedge1 = im.FindAdaptiveMom(maskedge, badpixedge)
    ress_maskedge1 = galsim.hsm.EstimateShear(im, p_im, maskedge, badpixedge)
    check_equal(resm_maskedge,ress_maskedge,resm_maskedge1,ress_maskedge1,
                "when masking with badpix")

    ## same as previous, but with badpix of floats
    maskedge = galsim.ImageI(imsize, imsize, init_value = 1)
    badpixedge = galsim.ImageF(imsize, imsize, init_value = 0.)
    for ind1 in range(xmin, xmax+1):
        for ind2 in range(ymin, ymax+1):
            if (ind1 <= (xmin+edgenum)) or (ind1 >= (xmax-edgenum)) or (ind2 <= (ymin+edgenum)) or (ind2 >= (ymax-edgenum)):
                badpixedge.setValue(ind1, ind2, float(ind1+1))
    resm_maskedge1 = im.FindAdaptiveMom(maskedge, badpixedge)
    ress_maskedge1 = galsim.hsm.EstimateShear(im, p_im, maskedge, badpixedge)
    check_equal(resm_maskedge,ress_maskedge,resm_maskedge1,ress_maskedge1,
                "when masking with badpix (floats)")

    ## do some of the masking using weight map, and the rest using badpix
    maskedge = galsim.ImageI(imsize, imsize, init_value = 1)
    badpixedge = galsim.ImageI(imsize, imsize, init_value = 0)
    meanval = int(0.5*(xmin+xmax))
    for ind1 in range(xmin, xmax+1):
        for ind2 in range(ymin, ymax+1):
            if (ind1 <= (xmin+edgenum)) or (ind1 >= (xmax-edgenum)) or (ind2 <= (ymin+edgenum)) or (ind2 >= (ymax-edgenum)):
                if ind1 < meanval:
                    badpixedge.setValue(ind1, ind2, 1)
                else:
                    maskedge.setValue(ind1, ind2, 0)
    resm_maskedge1 = im.FindAdaptiveMom(maskedge, badpixedge)
    ress_maskedge1 = galsim.hsm.EstimateShear(im, p_im, maskedge, badpixedge)
    check_equal(resm_maskedge,ress_maskedge,resm_maskedge1,ress_maskedge1,
                "when masking with badpix and weight map")


@timer
def test_masks_with_check():
    """Test which errors in masks for moments and shear estimation routines are caught with check."""
    # Set up some toy galaxy and PSF
    my_sigma = 1.0
    my_pixscale = 0.1
    my_g1 = 0.15
    my_g2 = -0.4
    imsize = 256
    g = galsim.Gaussian(sigma = my_sigma)
    p = galsim.Gaussian(sigma = my_sigma)

    g = g.shear(g1=my_g1, g2=my_g2)
    obj = galsim.Convolve(g, p)
    im = galsim.ImageF(imsize, imsize)
    p_im = galsim.ImageF(imsize, imsize)
    obj.drawImage(image=im, scale=my_pixscale, method='no_pixel')
    p.drawImage(image=p_im, scale=my_pixscale, method='no_pixel')

    ## Create a weight image with size different from image
    ## Let it be smaller in one dimension, and larger in the other.
    bad_weight_im = galsim.ImageI(imsize, 2*imsize, init_value=1)
    small_weight_im = galsim.ImageI(2*imsize, imsize//4, init_value=1)
    # Setting strict=True (default) catches this error even if check=False.
    galsim.hsm.FindAdaptiveMom(im, small_weight_im, strict=False, check=False)
    assert_raises(galsim.errors.GalSimHSMError, galsim.hsm.FindAdaptiveMom, im, small_weight_im, strict=True, check=False)

    # A zero weight image raises errors,
    zero_weight_im = galsim.ImageI(imsize, 2*imsize, init_value=0)
    assert_raises(galsim.errors.GalSimHSMError, galsim.hsm.FindAdaptiveMom, im, zero_weight_im, check=False)
    assert_raises(galsim.errors.GalSimHSMError, galsim.hsm.EstimateShear, im, p_im, zero_weight_im, strict=True, check=False)
    # but a negative weight image can slip through with check=False.
    negative_weight_im = galsim.ImageI(imsize, 2*imsize, init_value=-1)
    galsim.hsm.FindAdaptiveMom(im, negative_weight_im, check=False)
    galsim.hsm.EstimateShear(im, p_im, negative_weight_im, check=False)
    # But these are unique to ImageI. Passing the same as ImageS will catch the error, but is more expensive due to deep copy.
    negative_weight_im_singleprecision = galsim.ImageS(imsize, imsize, init_value=-1)
    assert_raises(galsim.errors.GalSimHSMError, galsim.hsm.FindAdaptiveMom, im, negative_weight_im_singleprecision, check=False)

    # But all of these errors are caught with check=True.
    assert_raises(galsim.errors.GalSimIncompatibleValuesError, galsim.hsm.FindAdaptiveMom, im, bad_weight_im, check=True)
    assert_raises(galsim.errors.GalSimIncompatibleValuesError, galsim.hsm.FindAdaptiveMom, im, zero_weight_im, check=True)
    assert_raises(galsim.errors.GalSimIncompatibleValuesError, galsim.hsm.FindAdaptiveMom, im, negative_weight_im, check=True)
    assert_raises(galsim.errors.GalSimValueError, galsim.hsm.FindAdaptiveMom, im, negative_weight_im_singleprecision, check=True)
    assert_raises(galsim.errors.GalSimIncompatibleValuesError, galsim.hsm.EstimateShear, im, p_im, bad_weight_im, check=True)
    assert_raises(galsim.errors.GalSimIncompatibleValuesError, galsim.hsm.EstimateShear, im, p_im, zero_weight_im, check=True)
    assert_raises(galsim.errors.GalSimIncompatibleValuesError, galsim.hsm.EstimateShear, im, p_im, negative_weight_im, check=True)
    assert_raises(galsim.errors.GalSimValueError, galsim.hsm.EstimateShear, im, p_im, negative_weight_im_singleprecision, check=True)


@timer
def test_shearest_shape():
    """Test that shear estimation is insensitive to shape of input images."""
    # this test can help reveal bugs having to do with x / y indexing issues
    # just do test for one particular gaussian
    g1 = shear_values[1]
    g2 = shear_values[2]
    e1_psf = 0.05
    e2_psf = -0.04
    total_shear = np.sqrt(g1**2 + g2**2)
    conversion_factor = np.tanh(2.0*math.atanh(total_shear))/total_shear
    distortion_1 = g1*conversion_factor
    distortion_2 = g2*conversion_factor
    gal = galsim.Exponential(flux = 1.0, half_light_radius = 1.)
    gal = gal.shear(g1=g1, g2=g2)
    psf = galsim.Kolmogorov(flux = 1.0, fwhm = 0.7)
    psf = psf.shear(e1=e1_psf, e2=e2_psf)
    final = galsim.Convolve([gal, psf])

    imsize = [128, 256]
    for method_index in range(len(correction_methods)):
        print(correction_methods[method_index])

        save_e1 = -100.
        save_e2 = -100.
        for gal_x_imsize in imsize:
            for gal_y_imsize in imsize:
                for psf_x_imsize in imsize:
                    for psf_y_imsize in imsize:
                        final_image = galsim.ImageF(gal_x_imsize, gal_y_imsize)
                        epsf_image = galsim.ImageF(psf_x_imsize, psf_y_imsize)

                        final.drawImage(image=final_image, scale=pixel_scale, method='no_pixel')
                        psf.drawImage(image=epsf_image, scale=pixel_scale, method='no_pixel')
                        result = galsim.hsm.EstimateShear(final_image, epsf_image,
                            shear_est = correction_methods[method_index])
                        e1 = result.corrected_e1
                        e2 = result.corrected_e2
                        # make sure answers don't change as we vary image size

                        tot_e = np.sqrt(save_e1**2 + save_e2**2)
                        if tot_e < 99.:
                            np.testing.assert_almost_equal(e1, save_e1,
                                err_msg = "- incorrect e1",
                                decimal = decimal_shape)
                            np.testing.assert_almost_equal(e2, save_e2,
                                err_msg = "- incorrect e2",
                                decimal = decimal_shape)
                        save_e1 = e1
                        save_e2 = e2


@timer
def test_hsmparams():
    """Test the ability to set/change parameters that define how moments/shape estimation are done."""
    # First make some profile, and make sure that we get the same answers when we specify default
    # hsmparams or don't specify hsmparams at all.
    default_hsmparams = galsim.hsm.HSMParams(nsig_rg=3.0,
                                             nsig_rg2=3.6,
                                             regauss_too_small=1,
                                             adapt_order=2,
                                             convergence_threshold=1.e-6,
                                             max_mom2_iter=400,
                                             num_iter_default=-1,
                                             bound_correct_wt=0.25,
                                             max_amoment=8000.,
                                             max_ashift=15.,
                                             ksb_moments_max=4,
                                             failed_moments=-1000.)
    bulge = galsim.DeVaucouleurs(half_light_radius = 0.3)
    disk = galsim.Exponential(half_light_radius = 0.5)
    disk = disk.shear(e1=0.2, e2=-0.3)
    psf = galsim.Kolmogorov(fwhm = 0.6)
    gal = bulge + disk   # equal weighting, i.e., B/T=0.5
    tot_gal = galsim.Convolve(gal, psf)
    tot_gal_image = tot_gal.drawImage(scale=0.18)
    tot_psf_image = psf.drawImage(scale=0.18)

    res = tot_gal_image.FindAdaptiveMom()
    res_def = tot_gal_image.FindAdaptiveMom(hsmparams = default_hsmparams)
    assert(equal_hsmshapedata(res, res_def)), 'Moment outputs differ when using default HSMParams'
    assert res == res_def, 'Moment outputs differ when using default HSMParams'

    res2 = galsim.hsm.EstimateShear(tot_gal_image, tot_psf_image)
    res2_def = galsim.hsm.EstimateShear(tot_gal_image, tot_psf_image, hsmparams = default_hsmparams)
    assert(equal_hsmshapedata(res, res_def)), 'Shear outputs differ when using default HSMParams'
    assert res == res_def, 'Shear outputs differ when using default HSMParams'

    check_pickle(default_hsmparams)
    check_pickle(galsim.hsm.HSMParams(nsig_rg=1.0,
                                   nsig_rg2=1.6,
                                   regauss_too_small=0,
                                   adapt_order=0,
                                   convergence_threshold=1.e-8,
                                   max_mom2_iter=100,
                                   num_iter_default=4,
                                   bound_correct_wt=0.05,
                                   max_amoment=80.,
                                   max_ashift=5.,
                                   ksb_moments_max=2,
                                   failed_moments=99.))
    check_pickle(res)
    check_pickle(res2)

    # Then check failure modes: force it to fail by changing HSMParams.
    new_params_niter = galsim.hsm.HSMParams(max_mom2_iter = res.moments_n_iter-1)
    new_params_size = galsim.hsm.HSMParams(max_amoment = 0.3*res.moments_sigma**2)
    assert_raises(galsim.GalSimError, galsim.hsm.FindAdaptiveMom, tot_gal_image,
                  hsmparams=new_params_niter)
    assert_raises(galsim.GalSimError, galsim.hsm.EstimateShear, tot_gal_image, tot_psf_image,
                  hsmparams=new_params_size)

    assert_raises(TypeError, galsim.hsm.EstimateShear, tot_gal_image, tot_psf_image,
                  hsmparams='hsmparams')


@timer
def test_hsmparams_nodefault():
    """Test that when non-default hsmparams are used, the results change."""
    import time
    # First make some profile
    bulge = galsim.DeVaucouleurs(half_light_radius = 0.3)
    disk = galsim.Exponential(half_light_radius = 0.5)
    disk = disk.shear(e1=0.2, e2=-0.3)
    psf = galsim.Kolmogorov(fwhm = 0.6)
    gal = bulge + disk   # equal weighting, i.e., B/T=0.5
    tot_gal = galsim.Convolve(gal, psf)
    tot_gal_image = tot_gal.drawImage(scale=0.18)
    tot_psf_image = psf.drawImage(scale=0.18)

    # Check that recompute_flux changes give results that are as expected
    test_t = time.time()
    res = galsim.hsm.EstimateShear(tot_gal_image, tot_psf_image)
    dt = time.time() - test_t
    res2 = galsim.hsm.EstimateShear(tot_gal_image, tot_psf_image, recompute_flux = 'sum')
    assert(res.moments_amp < res2.moments_amp),'Incorrect behavior with recompute_flux=sum'
    res3 = galsim.hsm.EstimateShear(tot_gal_image, tot_psf_image, recompute_flux = 'none')
    assert(res3.moments_amp == 0),'Incorrect behavior with recompute_flux=none'

    # Check correction_status and error message when recompute_flux is invalid.
    with assert_raises(galsim.GalSimError):
        galsim.hsm.EstimateShear(tot_gal_image, tot_psf_image, recompute_flux='invalid')
    res4 = galsim.hsm.EstimateShear(tot_gal_image, tot_psf_image, recompute_flux='invalid',
                                    strict=False)
    assert res4.correction_status == -1
    assert "Unknown value" in res4.error_message

    # Check that results, timing change as expected with nsig_rg
    # For this, use Gaussian as galaxy and for ePSF, i.e., no extra pixel response
    p = galsim.Gaussian(fwhm=10.)
    g = galsim.Gaussian(fwhm=20.)
    g = g.shear(g1=0.5)
    obj = galsim.Convolve(g, p)
    # HSM allows a slop of 1.e-8 on nsig_rg, which means that default float32 images don't
    # actually end up with different result when using nsig_rg=0. rather than 3.
    im = obj.drawImage(scale=1., method='no_pixel', dtype=float)
    psf_im = p.drawImage(scale=1., method='no_pixel', dtype=float)
    test_t1 = time.time()
    g_res = galsim.hsm.EstimateShear(im, psf_im)
    test_t2 = time.time()
    g_res2 = galsim.hsm.EstimateShear(im, psf_im, hsmparams=galsim.hsm.HSMParams(nsig_rg=0.))
    dt2 = time.time()-test_t2
    dt1 = test_t2-test_t1
    if test_timing:
        assert(dt2 > dt1),'Should take longer to estimate shear without truncation of galaxy'
    assert(not equal_hsmshapedata(g_res, g_res2)),'Results should differ with diff nsig_rg'
    assert g_res != g_res2,'Results should differ with diff nsig_rg'

    # Check that results, timing change as expected with convergence_threshold
    test_t2 = time.time()
    res2 = galsim.hsm.EstimateShear(tot_gal_image, tot_psf_image,
                                    hsmparams=galsim.hsm.HSMParams(convergence_threshold = 1.e-3))
    dt2 = time.time() - test_t2
    if test_timing:
        assert(dt2 < dt),'Should be faster to estimate shear with higher convergence_threshold'
    assert(not equal_hsmshapedata(res, res2)),'Outputs same despite change in convergence_threshold'
    assert res != res2,'Outputs same despite change in convergence_threshold'

    # Check that max_amoment, max_ashift work as expected
    assert_raises(galsim.GalSimError,
        galsim.hsm.EstimateShear, tot_gal_image, tot_psf_image,
        hsmparams=galsim.hsm.HSMParams(max_amoment = 10.))
    assert_raises(galsim.GalSimError,
        galsim.hsm.EstimateShear, tot_gal_image, tot_psf_image,
        guess_centroid=galsim.PositionD(47., tot_gal_image.true_center.y),
        hsmparams=galsim.hsm.HSMParams(max_ashift=0.1))


@timer
def test_shapedata():
    """Check for basic issues with initialization of ShapeData objects."""
    x = 1.
    # Cannot initialize with messed up arguments.
    assert_raises(TypeError, galsim.hsm.ShapeData, x, x)
    assert_raises(TypeError, galsim.hsm.ShapeData, x)

    # Check that if initialized when empty, the resulting object has certain properties.
    foo = galsim.hsm.ShapeData()
    if foo.observed_shape != galsim.Shear() or foo.moments_n_iter != 0 or foo.meas_type != "None":
        raise AssertionError("Default ShapeData object was not as expected!")


@timer
def test_strict():
    """Check that using strict=True results in the behavior we expect."""
    # Set up an image for which moments measurement should fail spectacularly.
    scale = 2.7
    size = 11
    pix = galsim.Pixel(scale)
    image = galsim.Image(size, size)
    im = pix.drawImage(image=image, scale=scale, method='no_pixel')

    # Try to measure moments with strict = False. Make sure there's an error message stored.
    res = im.FindAdaptiveMom(strict = False)
    if res.error_message == '':
        raise AssertionError("Should have error message stored in case of FindAdaptiveMom failure!")

    # Check that measuring moments with strict = True results in the expected exception, and that
    # it is the same one as is stored when running with strict = False.
    with assert_raises(galsim.GalSimError):
        galsim.hsm.FindAdaptiveMom(im)
    try:
        res2 = im.FindAdaptiveMom()
    except galsim.GalSimError as err:
        if str(err) != res.error_message:
            raise AssertionError("Error messages do not match when running identical tests!")

    # Now redo the above for EstimateShear
    res = galsim.hsm.EstimateShear(im, im, strict = False)
    if res.error_message == '':
        raise AssertionError("Should have error message stored in case of EstimateShear failure!")
    with assert_raises(galsim.GalSimError):
        galsim.hsm.EstimateShear(im, im)
    try:
        res2 = galsim.hsm.EstimateShear(im, im)
    except galsim.GalSimError as err:
        if str(err) != res.error_message:
            raise AssertionError("Error messages do not match when running identical tests!")


@timer
def test_bounds_centroid():
    """Check that the input bounds are respected, and centroid coordinates make sense."""
    # Make a simple object drawn into image with non-trivial bounds (even-sized).
    b = galsim.BoundsI(37, 326, 47, 336)
    test_scale = 0.15
    test_sigma = 3.1
    im = galsim.Image(bounds=b)
    im.scale = test_scale

    obj = galsim.Gaussian(sigma=test_sigma)
    obj.drawImage(image=im, scale=test_scale, method='no_pixel')
    mom = im.FindAdaptiveMom()
    np.testing.assert_almost_equal(
        mom.moments_centroid.x, im.true_center.x, decimal=7,
        err_msg='Moments x centroid differs from true center of even-sized image')
    np.testing.assert_almost_equal(
        mom.moments_centroid.y, im.true_center.y, decimal=7,
        err_msg='Moments y centroid differs from true center of even-sized image')

    # Draw the same object into odd-sized image with non-trivial bounds.
    b2 = galsim.BoundsI(b.xmin, b.xmax+1, b.ymin, b.ymax+1)
    im = galsim.Image(bounds=b2)
    im.scale = test_scale
    obj.drawImage(image=im, scale=test_scale, method='no_pixel')
    mom = im.FindAdaptiveMom()
    np.testing.assert_almost_equal(
        mom.moments_centroid.x, im.true_center.x, decimal=7,
        err_msg='Moments x centroid differs from true center of odd-sized image')
    np.testing.assert_almost_equal(
        mom.moments_centroid.y, im.true_center.y, decimal=7,
        err_msg='Moments y centroid differs from true center of odd-sized image')

    # Check that it still works with a symmetric sub-image.
    sub_im = im[galsim.BoundsI(b2.xmin+2, b2.xmax-2, b2.ymin+2, b2.ymax-2)]
    mom = sub_im.FindAdaptiveMom()
    np.testing.assert_almost_equal(
        mom.moments_centroid.x, sub_im.true_center.x, decimal=7,
        err_msg='Moments x centroid differs from true center of odd-sized subimage')
    np.testing.assert_almost_equal(
        mom.moments_centroid.y, sub_im.true_center.y, decimal=7,
        err_msg='Moments y centroid differs from true center of odd-sized subimage')

    # Check that we can take a weird/asymmetric sub-image, and it fails because of centroid shift.
    sub_im = im[galsim.BoundsI(b2.xmin, b2.xmax-100, b2.ymin+27, b2.ymax)]
    with assert_raises(galsim.GalSimError):
        galsim.hsm.FindAdaptiveMom(sub_im)

    # ... and that it passes if we hand in a good centroid guess.  Note that this test is a bit less
    # stringent than some of the previous ones, because our subimage cut off a decent part of the
    # light profile in the x direction, affecting the x centroid estimate.  But the y centroid test
    # is the same precision as before.
    mom = sub_im.FindAdaptiveMom(guess_centroid=im.true_center)
    np.testing.assert_approx_equal(
        mom.moments_centroid.x, im.true_center.x, significant=4,
        err_msg='Moments x centroid differs from true center of asymmetric subimage')
    np.testing.assert_almost_equal(
        mom.moments_centroid.y, im.true_center.y, decimal=7,
        err_msg='Moments y centroid differs from true center of asymmetric subimage')


@timer
def test_ksb_sig():
    """Check that modification of KSB weight function width works."""
    gal = galsim.Gaussian(fwhm=1.0).shear(e1=0.2, e2=0.1)
    psf = galsim.Gaussian(fwhm=0.7)
    gal_img = galsim.Convolve(gal, psf).drawImage(nx=32, ny=32, scale=0.2)
    psf_img = psf.drawImage(nx=16, ny=16, scale=0.2)

    # First just check that combination of ksb_sig_weight and ksb_sig_factor is consistent.
    hsmparams1 = galsim.hsm.HSMParams(ksb_sig_weight=2.0)
    result1 = galsim.hsm.EstimateShear(gal_img, psf_img, shear_est='KSB', hsmparams=hsmparams1)

    hsmparams2 = galsim.hsm.HSMParams(ksb_sig_weight=1.0, ksb_sig_factor=2.0)
    result2 = galsim.hsm.EstimateShear(gal_img, psf_img, shear_est='KSB', hsmparams=hsmparams2)

    np.testing.assert_almost_equal(result1.corrected_g1, result2.corrected_g1, 9,
                                   "KSB weight fn width inconsistently manipulated")
    np.testing.assert_almost_equal(result1.corrected_g2, result2.corrected_g2, 9,
                                   "KSB weight fn width inconsistently manipulated")

    # Now check that if we construct a galaxy with an ellipticity gradient, we see the appropriate
    # sign of the response when we change the width of the weight function.
    narrow = galsim.Gaussian(fwhm=1.0).shear(e1=0.2)
    wide = galsim.Gaussian(fwhm=2.0).shear(e1=-0.2)
    gal = narrow + wide
    gal_img = galsim.Convolve(gal, psf).drawImage(nx=32, ny=32, scale=0.2)
    hsmparams_narrow = galsim.hsm.HSMParams()  # Default sig_factor=1.0
    result_narrow = galsim.hsm.EstimateShear(gal_img, psf_img, shear_est='KSB',
                                             hsmparams=hsmparams_narrow)
    hsmparams_wide = galsim.hsm.HSMParams(ksb_sig_factor=2.0)
    result_wide = galsim.hsm.EstimateShear(gal_img, psf_img, shear_est='KSB',
                                           hsmparams=hsmparams_wide)

    np.testing.assert_array_less(result_wide.corrected_g1, result_narrow.corrected_g1,
                                 "Galaxy ellipticity gradient not captured by ksb_sig_factor.")

@timer
def test_noncontiguous():
    """Test running HSM module with non-C-contiguous images.

    This test was inspired by a bug report in issue #833.
    """
    gal = galsim.Gaussian(sigma=1.3).shear(g1=0.1, g2=0.2)

    # First a normal C-contiguous image as build by drawImage.
    img = gal.drawImage(nx=64, ny=64, scale=0.2)
    meas_shape1 = galsim.hsm.FindAdaptiveMom(img).observed_shape
    print(meas_shape1)
    np.testing.assert_almost_equal(meas_shape1.g1, 0.1, decimal=3,
                                   err_msg="HSM measured wrong shear on normal image")
    np.testing.assert_almost_equal(meas_shape1.g2, 0.2, decimal=3,
                                   err_msg="HSM measured wrong shear on normal image")

    # Transpose the image, which should just flip the sign of g1.
    # Note, though, that this changes the ndarray from C-ordering to FORTRAN-ordering.
    fimg = galsim.Image(img.array.T, scale=0.2)
    meas_shape2 = galsim.hsm.FindAdaptiveMom(fimg).observed_shape
    print(meas_shape2)
    np.testing.assert_almost_equal(meas_shape2.g1, -0.1, decimal=3,
                                   err_msg="HSM measured wrong shear on transposed image")
    np.testing.assert_almost_equal(meas_shape2.g2, 0.2, decimal=3,
                                   err_msg="HSM measured wrong shear on transposed image")

    # Also test the real part of an ImageC, which not contiguous in either direction (step=2)
    # This should have the negative shear from drawing in k space.
    kimg = gal.drawKImage(nx=64, ny=64)
    meas_shape3 = galsim.hsm.FindAdaptiveMom(kimg.real).observed_shape
    print(meas_shape3)
    np.testing.assert_almost_equal(meas_shape3.g1, -0.1, decimal=3,
                                   err_msg="HSM measured wrong shear on image with step=2")
    np.testing.assert_almost_equal(meas_shape3.g2, -0.2, decimal=3,
                                   err_msg="HSM measured wrong shear on image with step=2")

@timer
def test_headers():
    # This isn't really an HSM test per se, but it's testing a feature that we added so
    # LSST DM can use the C++-layer HSM code from their C++ code.
    # It also adds a bit of stability guarantee to the HSM function signatures at least.

    # Check that the named files exist.
    include_dir = galsim.include_dir
    lib_file = galsim.lib_file
    print('include_dir = ',include_dir)
    print('lib_file = ',lib_file)
    assert os.path.isfile(lib_file)
    assert os.path.isfile(os.path.join(include_dir, 'GalSim.h'))

    # Check version code
    version_h = os.path.join(include_dir, 'galsim', 'Version.h')
    with open(version_h) as fin:
        lines = fin.read()
    assert '#define GALSIM_MAJOR %d'%(galsim.__version_info__[0]) in lines
    assert '#define GALSIM_MINOR %d'%(galsim.__version_info__[1]) in lines

    # Check the function signatudes of the code that LSST DM uses from the C++ layer.
    psfcorr_h = os.path.join(include_dir, 'galsim', 'hsm', 'PSFCorr.h')
    assert os.path.isfile(os.path.join(include_dir, 'galsim', 'hsm', 'PSFCorr.h'))

    fam_signature = """
    PUBLIC_API void FindAdaptiveMomView(
        ShapeData& results,
        const BaseImage<T> &object_image, const BaseImage<int> &object_mask_image,
        double guess_sig = 5.0, double precision = 1.0e-6,
        galsim::Position<double> guess_centroid = galsim::Position<double>(-1000.,-1000.),
        bool round_moments = false,
        const HSMParams& hsmparams=HSMParams());"""

    esv_signature = """
    PUBLIC_API void EstimateShearView(
        ShapeData& results,
        const BaseImage<T> &gal_image, const BaseImage<U> &PSF_image,
        const BaseImage<int> &gal_mask_image,
        float sky_var = 0.0, const char *shear_est = "REGAUSS",
        const char* recompute_flux = "FIT",
        double guess_sig_gal = 5.0, double guess_sig_PSF = 3.0, double precision = 1.0e-6,
        galsim::Position<double> guess_centroid = galsim::Position<double>(-1000.,-1000.),
        const HSMParams& hsmparams=HSMParams());"""

    # Note: If these fail (because we change the above signature for some reason), let LSST-DM
    # devs know that their code will break on the next version of GalSim.
    # They know we don't guarantee this to be stable, but it's a courtesy to let them know if
    # they might have problems.
    with open(psfcorr_h) as fin:
        lines = fin.read()
    assert fam_signature in lines
    assert esv_signature in lines

    # Check that the library is loadable.  (C++ name mangling means we can't actually check
    # that the right functions are there, but that should be implicitly tested by the fact
    # that calls from python actaully work...)
    import ctypes
    lib = ctypes.cdll.LoadLibrary(galsim.lib_file)
    # The test was that this doesn't raise an OSError or something.

@timer
def test_failures():
    """Test some images that used to fail, but now work.
    """
    files = glob.glob(os.path.join(img_dir, 'HSM*.fits'))

    for f in files:
        im = galsim.fits.read(f)
        hsm = im.FindAdaptiveMom()
        assert hsm.moments_status == 0

@timer
def test_very_small():
    """Test an unresolved star reported to fail in #1132, but now works.
    """
    profile = galsim.VonKarman(lam=700.0, r0=0.25, L0=11.0, flux=1.0).shift(-0.13, -0.13)
    im = profile.drawImage(nx=31, ny=31, scale=0.26)
    mom = im.FindAdaptiveMom(strict=False)
    print(mom)
    assert mom.moments_status == 0

    # However, if the object is small enough, then HSM cannot recover.  Make sure the
    # error message in such cases is sensible.
    profile = galsim.Gaussian(sigma=0.18).shift(0.02,-0.03)
    im = profile.drawImage(nx=31, ny=31, scale=0.26)
    mom = im.FindAdaptiveMom(strict=False)
    print(mom)
    assert mom.moments_status != 0
    assert "Object is too small" in mom.error_message


@timer
def test_negative_stepstride():
    """In response to #1185, check that hsm works for arrays with negative step or stride.
    """
    img = galsim.Gaussian(fwhm=1).drawImage()
    result1 = galsim.hsm.FindAdaptiveMom(img)
    result2 = galsim.hsm.FindAdaptiveMom(galsim.Image(img.array[::-1]))
    result3 = galsim.hsm.FindAdaptiveMom(galsim.Image(img.array[:,::-1]))
    result4 = galsim.hsm.FindAdaptiveMom(galsim.Image(img.array[::-1,::-1]))

    assert np.isclose(result1.moments_sigma, result2.moments_sigma)
    assert np.isclose(result1.moments_sigma, result3.moments_sigma)
    assert np.isclose(result1.moments_sigma, result4.moments_sigma)


if __name__ == "__main__":
    runtests(__file__)
