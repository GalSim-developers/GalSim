# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

from galsim_test_helpers import *

path, filename = os.path.split(__file__)
imgdir = os.path.join(path, "SBProfile_comparison_images") # Directory containing the reference
                                                           # images.

try:
    import galsim
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# Some values to use in multiple tests below:
test_hlr = 1.8
test_fwhm = 1.8
test_sigma = 1.8
test_sersic_n = [1.5, 2.5, 4, -4]  # -4 means use explicit DeVauc rather than n=4
test_scale = [1.8, 0.05, 0.002, 0.002]
test_sersic_trunc = [0., 8.5]
test_flux = 1.8

test_spergel_nu = [-0.85, -0.5, 0.0, 0.85, 4.0]
test_spergel_scale = [20.0, 1.0, 1.0, 0.5, 0.5]

if __name__ != "__main__":
    # If doing a pytest run, we don't actually need to do all 4 sersic n values.
    # Two should be enough to notice if there is a problem, and the full list will be tested
    # when running python test_base.py to try to diagnose the problem.
    test_sersic_n = [1.5, -4]
    test_scale = [1.8, 0.002]

# These are the default GSParams used when unspecified.  We'll check that specifying
# these explicitly produces the same results.
default_params = galsim.GSParams(
        minimum_fft_size = 128,
        maximum_fft_size = 4096,
        folding_threshold = 5.e-3,
        maxk_threshold = 1.e-3,
        kvalue_accuracy = 1.e-5,
        xvalue_accuracy = 1.e-5,
        shoot_accuracy = 1.e-5,
        realspace_relerr = 1.e-4,
        realspace_abserr = 1.e-6,
        integration_relerr = 1.e-6,
        integration_abserr = 1.e-8)


@timer
def test_sersic():
    """Test the generation of a specific Sersic profile against a known result.
    """
    # Test Sersic
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic disagrees with expected result")

    # Check with default_params
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=default_params)
    sersic.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic with default_params disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    sersic.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic with GSParams() disagrees with expected result")

    # Use non-unity values.
    sersic = galsim.Sersic(n=3, flux=1.7, half_light_radius=2.3)
    check_basic(sersic, "Sersic")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,"Sersic")

    # Test kvalues
    do_kvalue(sersic,myImg,"Sersic")

    # Check picklability
    do_pickle(sersic, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(sersic)

    # Now repeat everything using a truncation.  (Above had no truncation.)

    # Test Truncated Sersic
    # Don't use an integer truncation, since we don't want the truncation line to pass directly
    # through the center of a pixel where numerical rounding differences may decide whether the
    # value is zero or not.
    # This regression test compares to an image built using the code base at 82259f0
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1_10.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, trunc=9.99)
    sersic.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using truncated GSObject Sersic disagrees with expected result")

    # Use non-unity values.
    sersic = galsim.Sersic(n=3, flux=test_flux, half_light_radius=2.3, trunc=5.9)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(sersic.centroid, cen)
    np.testing.assert_almost_equal(sersic.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(sersic.flux, test_flux)
    np.testing.assert_almost_equal(sersic.xValue(cen), sersic.max_sb)

    check_basic(sersic, "Truncated Sersic")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,"Truncated Sersic")

    # Test kvalues
    do_kvalue(sersic,myImg, "Truncated Sersic")

    # Check picklability
    do_pickle(sersic, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(sersic)

    # Check for normalization consistencies with kValue checks. xValues tested in test_sersic_radii.

    # For half-light radius specified truncated Sersic, with flux_untruncated flag set
    sersic = galsim.Sersic(n=3, flux=test_flux, half_light_radius=1, trunc=10,
                           flux_untruncated=True)
    do_kvalue(sersic,myImg, "Truncated Sersic w/ flux_untruncated, half-light radius specified")

    # For scale radius specified Sersic
    sersic = galsim.Sersic(n=3, flux=test_flux, scale_radius=0.05)
    do_kvalue(sersic,myImg, "Sersic w/ scale radius specified")

    # For scale radius specified truncated Sersic
    sersic = galsim.Sersic(n=3, flux=test_flux, scale_radius=0.05, trunc=10)
    do_kvalue(sersic,myImg, "Truncated Sersic w/ scale radius specified")

    # For scale radius specified truncated Sersic, with flux_untruncated flag set
    sersic = galsim.Sersic(n=3, flux=test_flux, scale_radius=0.05, trunc=10, flux_untruncated=True)
    do_kvalue(sersic,myImg, "Truncated Sersic w/ flux_untruncated, scale radius specified")

    # Test severely truncated Sersic
    sersic = galsim.Sersic(n=4, flux=test_flux, half_light_radius=1, trunc=1.45)
    do_kvalue(sersic,myImg, "Severely truncated n=4 Sersic")

    # Should raise an exception if both scale_radius and half_light_radius are provided.
    try:
        np.testing.assert_raises(TypeError, galsim.Sersic, n=1.2, scale_radius=3,
                                 half_light_radius=1)
        np.testing.assert_raises(TypeError, galsim.DeVaucouleurs, scale_radius=3,
                                 half_light_radius=1)
    except ImportError:
        pass


@timer
def test_sersic_radii():
    """Test initialization of Sersic with different types of radius specification.
    """
    import math
    for n, scale in zip(test_sersic_n, test_scale) :

        # Test constructor using half-light-radius
        if n == -4:
            test_gal1 = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=1.)
            test_gal2 = galsim.DeVaucouleurs(half_light_radius=test_hlr, trunc=8.5, flux=1.)
            test_gal3 = galsim.DeVaucouleurs(half_light_radius=test_hlr, trunc=8.5, flux=1.,
                                             flux_untruncated=True)
            gal_labels = ["DeVauc", "truncated DeVauc", "flux_untruncated DeVauc"]
        else:
            test_gal1 = galsim.Sersic(n=n, half_light_radius=test_hlr, flux=1.)
            test_gal2 = galsim.Sersic(n=n, half_light_radius=test_hlr, trunc=8.5, flux=1.)
            test_gal3 = galsim.Sersic(n=n, half_light_radius=test_hlr, trunc=8.5, flux=1.,
                                      flux_untruncated=True)
            gal_labels = ["Sersic", "truncated Sersic", "flux_untruncated Sersic"]
        gal_list = [test_gal1, test_gal2, test_gal3]

        do_pickle(test_gal1)
        do_pickle(test_gal2)
        do_pickle(test_gal3)

        # Check that the returned half-light radius is correct
        print('test_hlr = ',test_hlr)
        print('test_gal1 hlr, sr = ',test_gal1.half_light_radius,test_gal1.scale_radius)
        print('test_gal2 hlr, sr = ',test_gal2.half_light_radius,test_gal2.scale_radius)
        print('test_gal3 hlr, sr = ',test_gal3.half_light_radius,test_gal3.scale_radius)
        np.testing.assert_almost_equal(
            test_gal1.half_light_radius, test_hlr, decimal=5,
            err_msg = "Error in returned HLR for Sersic HLR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal2.half_light_radius, test_hlr, decimal=5,
            err_msg = "Error in returned HLR for truncated Sersic HLR constructor, n=%.1f"%n)

        # 1 and 3 should have the same scale radius, but different hlr.
        np.testing.assert_almost_equal(
            test_gal3.scale_radius, test_gal1.scale_radius, decimal=5,
            err_msg = "Error in returned SR for flux_untruncated Sersic HLR constructor, n=%.1f"%n)

        # Check that the returned flux is correct
        print('test_gal1.flux = ',test_gal1.flux)
        print('test_gal2.flux = ',test_gal2.flux)
        print('test_gal3.flux = ',test_gal3.flux)
        np.testing.assert_almost_equal(
            test_gal1.flux, 1., decimal=5,
            err_msg = "Error in returned Flux for Sersic HLR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal2.flux, 1., decimal=5,
            err_msg = "Error in returned Flux for truncated Sersic HLR constructor, n=%.1f"%n)
        # test_gal3 doesn't match flux, but should have central value match test_gal1.
        center1 = test_gal1.xValue(galsim.PositionD(0,0))
        center3 = test_gal3.xValue(galsim.PositionD(0,0))
        print('peak value 1,3 = ', center1, center3)
        np.testing.assert_almost_equal(
                center1, center3, 9,
                "Error in flux_untruncated Sersic normalization HLR constructor, n=%.1f"%n)

        # (test half-light radii)
        for test_gal, label in zip(gal_list, gal_labels):
            print('flux = ',test_gal.flux)
            print('hlr = ',test_gal.half_light_radius)
            print('scale = ',test_gal.scale_radius)
            got_hlr = test_gal.half_light_radius
            got_flux = test_gal.flux
            hlr_sum = radial_integrate(test_gal, 0., got_hlr)
            print('hlr_sum = ',hlr_sum)
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*got_flux, decimal=4,
                    err_msg = "Error in %s half-light radius constructor, n=%.1f"%(label,n))

        # (test scale radii)
        for test_gal, label in zip(gal_list, gal_labels):
            got_sr = test_gal.scale_radius
            center = test_gal.xValue(galsim.PositionD(0,0))
            ratio = test_gal.xValue(galsim.PositionD(got_sr,0)) / center
            print('scale ratio = ',ratio)
            np.testing.assert_almost_equal(
                    ratio, np.exp(-1.0), decimal=4,
                    err_msg="Error in scale_radius for HLR constructed %s"%label)

        # Test constructor using scale radius (test scale radius)
        if n == -4:
            test_gal1 = galsim.DeVaucouleurs(scale_radius=scale, flux=1.)
            test_gal2 = galsim.DeVaucouleurs(scale_radius=scale, trunc=8.5, flux=1.)
            test_gal3 = galsim.DeVaucouleurs(scale_radius=scale, trunc=8.5, flux=1.,
                                             flux_untruncated=True)
        else:
            test_gal1 = galsim.Sersic(n=n, scale_radius=scale, flux=1.)
            test_gal2 = galsim.Sersic(n=n, scale_radius=scale, trunc=8.5, flux=1.)
            test_gal3 = galsim.Sersic(n=n, scale_radius=scale, trunc=8.5, flux=1.,
                                      flux_untruncated=True)
        gal_list = [test_gal1, test_gal2, test_gal3]

        # Check that the returned scale radius is correct
        print('test_scale = ',scale)
        print('test_gal1 hlr, sr = ',test_gal1.half_light_radius,test_gal1.scale_radius)
        print('test_gal2 hlr, sr = ',test_gal2.half_light_radius,test_gal2.scale_radius)
        print('test_gal3 hlr, sr = ',test_gal3.half_light_radius,test_gal3.scale_radius)
        np.testing.assert_almost_equal(
            test_gal1.scale_radius, scale, decimal=5,
            err_msg = "Error in returned SR for Sersic SR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal2.scale_radius, scale, decimal=5,
            err_msg = "Error in returned SR for truncated Sersic SR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal3.scale_radius, scale, decimal=5,
            err_msg = "Error in returned SR for truncated Sersic SR constructor, n=%.1f"%n)

        # Returned HLR should match for gals 2,3
        got_hlr2 = test_gal2.half_light_radius
        got_hlr3 = test_gal3.half_light_radius
        print('half light radii of truncated, scale_radius constructed Sersic =',got_hlr2,got_hlr3)
        np.testing.assert_almost_equal(
                got_hlr2, got_hlr3, decimal=4,
                err_msg="Error in HLR for scale_radius constructed flux_untruncated Sersic (II).")

        # Check that the returned flux is correct
        print('test_gal1.flux = ',test_gal1.flux)
        print('test_gal2.flux = ',test_gal2.flux)
        print('test_gal3.flux = ',test_gal3.flux)
        np.testing.assert_almost_equal(
            test_gal1.flux, 1., decimal=5,
            err_msg = "Error in returned Flux for Sersic HLR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal2.flux, 1., decimal=5,
            err_msg = "Error in returned Flux for truncated Sersic HLR constructor, n=%.1f"%n)
        center1 = test_gal1.xValue(galsim.PositionD(0,0))
        center3 = test_gal3.xValue(galsim.PositionD(0,0))
        print('peak value 1,3 = ', center1, center3)
        np.testing.assert_almost_equal(
                center1, center3, 9,
                "Error in flux_untruncated Sersic normalization HLR constructor, n=%.1f"%n)

        # (test scale radii)
        for test_gal, label in zip(gal_list, gal_labels):
            center = test_gal.xValue(galsim.PositionD(0,0))
            ratio = test_gal.xValue(galsim.PositionD(scale,0)) / center
            print('scale ratio = ',ratio)
            np.testing.assert_almost_equal(
                    ratio, np.exp(-1.0), decimal=4,
                    err_msg="Error in %s scale radius constructor, n=%.1f"%(label,n))

        # (test half-light radius)
        for test_gal, label in zip(gal_list, gal_labels):
            got_hlr = test_gal.half_light_radius
            got_flux = test_gal.flux
            hlr_sum = radial_integrate(test_gal, 0., got_hlr)
            print('hlr_sum = ',hlr_sum)
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*got_flux, decimal=4,
                    err_msg="Error in HLR for scale_radius constructed %s"%label)

        # Check that the getters don't work after modifying the original.
        test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
        # But not after shear() (or others, but this is a sufficient test here)
        try:
            if n != -4:
                np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "n")
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "scale_radius")
        except ImportError:
            pass


@timer
def test_sersic_flux_scaling():
    """Test flux scaling for Sersic.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12

    # loop through sersic n
    for test_n in test_sersic_n:
        # loop through sersic truncation
        for test_trunc in test_sersic_trunc:
            # init with hlr and flux only (should be ok given last tests)
            # n=-4 is code to use explicit DeVaucouleurs rather than Sersic(n=4).
            # It should be identical.
            if test_n == -4:
                init_obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux,
                                           trunc=test_trunc)
            else:
                init_obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux,
                                    trunc=test_trunc)

            obj2 = init_obj * 2.
            np.testing.assert_almost_equal(
                init_obj.flux, test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (original).")
            np.testing.assert_almost_equal(
                obj2.flux, test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (result).")

            obj2 = 2. * init_obj
            np.testing.assert_almost_equal(
                init_obj.flux, test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (original).")
            np.testing.assert_almost_equal(
                obj2.flux, test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (result).")

            obj2 = init_obj / 2.
            np.testing.assert_almost_equal(
                 init_obj.flux, test_flux, decimal=param_decimal,
                 err_msg="Flux param inconsistent after __div__ (original).")
            np.testing.assert_almost_equal(
                obj2.flux, test_flux / 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __div__ (result).")


@timer
def test_sersic_05():
    """Test the equivalence of Sersic with n=0.5 and Gaussian
    """
    # hlr/sigma = sqrt(2 ln(2)) = 1.177410022515475
    hlr_sigma = 1.177410022515475

    # cf test_gaussian()
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_1.fits"))
    savedImg.setCenter(0,0)
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    sersic = galsim.Sersic(n=0.5, flux=1, half_light_radius=1 * hlr_sigma)
    myImg = sersic.drawImage(myImg, method="sb", use_true_center=False)
    print('saved image center = ',savedImg(0,0))
    print('image center = ',myImg(0,0))
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using Sersic with n=0.5 disagrees with expected result for Gaussian")

    check_basic(sersic, "n=0.5 Sersic")

    do_kvalue(sersic,myImg,"n=0.5 Sersic")

    # cf test_gaussian_properties()
    sersic = galsim.Sersic(n=0.5, flux=test_flux, half_light_radius=test_sigma * hlr_sigma)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(sersic.centroid, cen)
    np.testing.assert_almost_equal(sersic.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(sersic.flux, test_flux)
    import math
    np.testing.assert_almost_equal(sersic.xValue(cen), 1./(2.*math.pi) * test_flux / test_sigma**2,
                                   decimal=5)
    np.testing.assert_almost_equal(sersic.xValue(cen), sersic.max_sb)

    # Also test some random values other than the center:
    gauss = galsim.Gaussian(flux=test_flux, sigma=test_sigma)
    for (x,y) in [ (0.1,0.2), (-0.5, 0.4), (0, 0.9), (1.2, 0.1), (2,2) ]:
        pos = galsim.PositionD(x,y)
        np.testing.assert_almost_equal(sersic.xValue(pos), gauss.xValue(pos), decimal=5)
        np.testing.assert_almost_equal(sersic.kValue(pos), gauss.kValue(pos), decimal=5)


@timer
def test_sersic_1():
    """Test the equivalence of Sersic with n=1 and Exponential
    """
    # cf test_exponential()
    re = 1.0
    r0 = re/1.67839
    # The real value of re/r0 = 1.6783469900166605
    hlr_r0 =  1.6783469900166605
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_1.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    sersic = galsim.Sersic(n=1, flux=1., half_light_radius=r0 * hlr_r0)
    sersic.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using Sersic n=1 disagrees with expected result for Exponential")

    check_basic(sersic, "n=1 Sersic")

    do_kvalue(sersic,myImg,"n=1 Sersic")

    # cf test_exponential_properties()
    sersic = galsim.Sersic(n=1, flux=test_flux, half_light_radius=test_scale[0] * hlr_r0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(sersic.centroid, cen)
    np.testing.assert_almost_equal(sersic.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(sersic.flux, test_flux)
    import math
    np.testing.assert_almost_equal(sersic.xValue(cen), 1./(2.*math.pi)*test_flux/test_scale[0]**2,
                                   decimal=5)
    np.testing.assert_almost_equal(sersic.xValue(cen), sersic.max_sb)

    # Also test some random values other than the center:
    expon = galsim.Exponential(flux=test_flux, scale_radius=test_scale[0])
    for (x,y) in [ (0.1,0.2), (-0.5, 0.4), (0, 0.9), (1.2, 0.1), (2,2) ]:
        pos = galsim.PositionD(x,y)
        np.testing.assert_almost_equal(sersic.xValue(pos), expon.xValue(pos), decimal=5)
        np.testing.assert_almost_equal(sersic.kValue(pos), expon.kValue(pos), decimal=5)


@timer
def test_kolmogorov():
    """Test the generation of a specific Kolmogorov profile against a known result.
    """
    import math
    dx = 0.2
    # This savedImg was created from the SBKolmogorov implementation in
    # commit c8efd74d1930157b1b1ffc0bfcfb5e1bf6fe3201
    # It would be nice to get an independent calculation here...
    #mySBP = galsim.SBKolmogorov(lam_over_r0=1.5, flux=test_flux)
    #savedImg = galsim.ImageF(128,128)
    #mySBP.drawImage(image=savedImg, dx=dx, method="sb")
    #savedImg.write(os.path.join(imgdir, "kolmogorov.fits"))
    savedImg = galsim.fits.read(os.path.join(imgdir, "kolmogorov.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=test_flux)
    kolm.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Kolmogorov disagrees with expected result")

    # Check with default_params
    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=test_flux, gsparams=default_params)
    kolm.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Kolmogorov with default_params disagrees with expected result")
    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=test_flux, gsparams=galsim.GSParams())
    kolm.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Kolmogorov with GSParams() disagrees with expected result")

    check_basic(kolm, "Kolmogorov")

    # Test photon shooting.
    do_shoot(kolm,myImg,"Kolmogorov")

    # Test kvalues
    do_kvalue(kolm,myImg, "Kolmogorov")

    # Check picklability
    do_pickle(kolm, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(kolm)

    # Test initialization separately with lam and r0, in various units.  Since the above profiles
    # have lam/r0 = 3./2. in arbitrary units, we will tell it that lam=3.e9 nm and r0=2.0 m,
    # and use `scale_unit` of galsim.radians.  This is rather silly, but it should work.
    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=test_flux)
    kolm2 = galsim.Kolmogorov(lam=3.e9, r0=2.0, scale_unit=galsim.radians, flux=test_flux)
    gsobject_compare(kolm,kolm2)
    # For lam/r0 = 1.5 arcsec, and r0 = 0.2, lam = (1.5/3600/180*pi) * 0.2 * 1.e9
    lam = 1.5 * 0.2 / 3600. / 180. * math.pi * 1.e9
    print('lam = ',lam)
    kolm3 = galsim.Kolmogorov(lam=lam, r0=0.2, scale_unit='arcsec', flux=test_flux)
    gsobject_compare(kolm,kolm3)
    # arcsec is the default scale_unit, so can leave this off.
    kolm4 = galsim.Kolmogorov(lam=lam, r0=0.2, flux=test_flux)
    gsobject_compare(kolm,kolm4)
    # Test using r0_500 instead
    r0_500 = 0.2 * (lam/500)**-1.2
    kolm5 = galsim.Kolmogorov(lam=lam, r0_500=r0_500, flux=test_flux)
    gsobject_compare(kolm,kolm5)

    # Should raise an exception if >= 2 radius specifications are provided and/or lam and r0 are not
    # paired together.
    try:
        np.testing.assert_raises(TypeError, galsim.Kolmogorov,
                                 lam_over_r0=3, fwhm=2, half_light_radius=1, lam=3, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov,
                                 fwhm=2, half_light_radius=1, lam=3, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov,
                                 lam_over_r0=3, half_light_radius=1, lam=3, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, fwhm=2, lam=3, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov,
                                 lam_over_r0=3, fwhm=2, half_light_radius=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, half_light_radius=1, lam=3, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, fwhm=2, lam=3, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, fwhm=2, half_light_radius=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, lam=3, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, half_light_radius=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, fwhm=2)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, lam=3)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, fwhm=2, lam=3)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, fwhm=2, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, half_light_radius=1, lam=3)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, half_light_radius=1, r0=1)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, lam=3)
        np.testing.assert_raises(TypeError, galsim.Kolmogorov, r0=1)
    except ImportError:
        pass

@timer
def test_kolmogorov_properties():
    """Test some basic properties of the Kolmogorov profile.
    """
    lor = 1.5
    psf = galsim.Kolmogorov(lam_over_r0=lor, flux=test_flux)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid, cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxk, 8.6440505245909858, 9)
    np.testing.assert_almost_equal(psf.stepk, 0.36982048503361376, 9)
    np.testing.assert_almost_equal(psf.kValue(cen), test_flux+0j)
    np.testing.assert_almost_equal(psf.lam_over_r0, lor)
    np.testing.assert_almost_equal(psf.half_light_radius, lor * 0.554811)
    np.testing.assert_almost_equal(psf.fwhm, lor * 0.975865)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.6283160485127478)
    np.testing.assert_almost_equal(psf.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(psf.flux, test_flux)
    np.testing.assert_almost_equal(psf.xValue(cen), psf.max_sb)

    # Check input flux vs output flux
    lors = [1, 0.5, 2, 5]
    for lor in lors:
        psf = galsim.Kolmogorov(lam_over_r0=lor, flux=test_flux)
        out_flux = psf.flux
        np.testing.assert_almost_equal(out_flux, test_flux,
                                       err_msg="Flux of Kolmogorov is incorrect.")

        # Also check the realized flux in a drawn image
        dx = lor / 10.
        img = galsim.ImageF(256,256, scale=dx)
        psf.drawImage(image=img)
        out_flux = img.array.sum(dtype=float)
        np.testing.assert_almost_equal(out_flux, test_flux, 3,
                                       err_msg="Flux of Kolmogorov (image array) is incorrect.")


@timer
def test_kolmogorov_radii():
    """Test initialization of Kolmogorov with different types of radius specification.
    """
    import math
    # Test constructor using lambda/r0
    lors = [1, 0.5, 2, 5]
    for lor in lors:
        print('lor = ',lor)
        test_gal = galsim.Kolmogorov(flux=1., lam_over_r0=lor)

        np.testing.assert_almost_equal(
                lor, test_gal.lam_over_r0, decimal=9,
                err_msg="Error in Kolmogorov, lor != lam_over_r0")

        # test that fwhm property provides correct FWHM
        got_fwhm = test_gal.fwhm
        print('got_fwhm = ',got_fwhm)
        test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
        print('fwhm ratio = ', test_fwhm_ratio)
        np.testing.assert_almost_equal(
                test_fwhm_ratio, 0.5, decimal=4,
                err_msg="Error in FWHM for Kolmogorov initialized with half-light radius")

        # then test that image indeed has the correct HLR properties when radially integrated
        got_hlr = test_gal.half_light_radius
        print('got_hlr = ',got_hlr)
        hlr_sum = radial_integrate(test_gal, 0., got_hlr)
        print('hlr_sum = ',hlr_sum)
        np.testing.assert_almost_equal(
                hlr_sum, 0.5, decimal=3,
                err_msg="Error in half light radius for Kolmogorov initialized with lam_over_r0.")

    # Test constructor using half-light-radius:
    test_gal = galsim.Kolmogorov(flux=1., half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print('hlr_sum = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=3,
            err_msg="Error in Kolmogorov constructor with half-light radius")

    # test that fwhm property provides correct FWHM
    got_fwhm = test_gal.fwhm
    print('got_fwhm = ',got_fwhm)
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                    test_gal.xValue(galsim.PositionD(0., 0.)))
    print('fwhm ratio = ', test_fwhm_ratio)
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Kolmogorov initialized with half-light radius")

    # Test constructor using FWHM:
    test_gal = galsim.Kolmogorov(flux=1., fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print('fwhm ratio = ',ratio)
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Kolmogorov constructor with fwhm")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.half_light_radius
    print('got_hlr = ',got_hlr)
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print('hlr_sum (profile initialized with fwhm) = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=3,
            err_msg="Error in half light radius for Kolmogorov initialized with FWHM.")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "fwhm")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "lam_over_r0")
    except ImportError:
        pass


@timer
def test_kolmogorov_flux_scaling():
    """Test flux scaling for Kolmogorov.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_lor0 = 1.9

    # init with lam_over_r0 and flux only (should be ok given last tests)
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")


@timer
def test_spergel():
    """Test the generation of a specific Spergel profile against a known result.
    """
    mathica_enclosed_fluxes = [3.06256e-2, 9.99995e-6, 6.06443e-10, 2.94117e-11, 6.25011e-12]
    mathica_enclosing_radii = [2.3973e-17, 1.00001e-5, 1.69047e-3, 5.83138e-3, 1.26492e-2]

    for nu, enclosed_flux, enclosing_radius in zip(test_spergel_nu,
                                                   mathica_enclosed_fluxes,
                                                   mathica_enclosing_radii):
        filename = "spergel_nu{0:.2f}.fits".format(nu)
        savedImg = galsim.fits.read(os.path.join(imgdir, filename))
        savedImg.setCenter(0,0)
        dx = 0.2
        myImg = galsim.ImageF(savedImg.bounds, scale=dx)
        myImg.setCenter(0,0)

        spergel = galsim.Spergel(nu=nu, half_light_radius=1.0)
        # Reference images were made with old centering,
        # which is equivalent to use_true_center=False.
        myImg = spergel.drawImage(myImg, scale=dx, method="sb", use_true_center=False)

        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Spergel disagrees with expected result")

        # nu < 0 has inf for xValue(0,0), so the x tests fail for them.
        check_basic(spergel, "Spergel with nu=%f"%nu, do_x=(nu > 0))

        # Only nu >= -0.3 give reasonably sized FFTs,
        # and small nu method='phot' is super slow.
        if nu >= -0.3:
            test_im = galsim.Image(16,16,scale=dx)
            do_kvalue(spergel,test_im, "Spergel(nu={0:1}) ".format(nu))

            # Test photon shooting.
            # Convolve with a small gaussian to smooth out the central peak.
            spergel2 = galsim.Convolve(spergel, galsim.Gaussian(sigma=0.3))
            do_shoot(spergel2,myImg,"Spergel")

        # Test integrated flux routines against Mathematica
        spergel = galsim.Spergel(nu=nu, scale_radius=1.0)
        np.testing.assert_almost_equal(
            spergel.calculateFluxRadius(1.e-5)/enclosing_radius, 1.0, 4,
            err_msg="Calculated incorrect Spergel(nu={0}) flux-enclosing-radius.".format(nu))
        np.testing.assert_almost_equal(
            spergel.calculateIntegratedFlux(1.e-5)/enclosed_flux, 1.0, 4,
            err_msg="Calculated incorrect Spergel(nu={0}) enclosed flux.".format(nu))

    # Use non-unity values.
    spergel = galsim.Spergel(nu=0.37, flux=1.7, half_light_radius=2.3)

    check_basic(spergel, "Spergel")

    # Check picklability
    do_pickle(spergel, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(spergel)
    do_pickle(galsim.Spergel(0,1))

    # Should raise an exception if both scale_radius and half_light_radius are provided.
    try:
        np.testing.assert_raises(TypeError, galsim.Spergel, nu=0, scale_radius=3,
                                 half_light_radius=1)
    except ImportError:
        pass


@timer
def test_spergel_properties():
    """Test some basic properties of the Spergel profile.
    """
    spergel = galsim.Spergel(nu=0.0, flux=test_flux, scale_radius=1.0)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(spergel.centroid, cen)
    # # Check Fourier properties
    np.testing.assert_almost_equal(spergel.kValue(cen), (1+0j) * test_flux)
    maxk = spergel.maxk
    np.testing.assert_array_less(spergel.kValue(maxk,0)/test_flux, galsim.GSParams().maxk_threshold)
    np.testing.assert_almost_equal(spergel.flux, test_flux)
    np.testing.assert_almost_equal(spergel.xValue(cen), spergel.max_sb)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        spergel = galsim.Spergel(nu=0.0, flux=inFlux, scale_radius=1.0)
        outFlux = spergel.flux
        np.testing.assert_almost_equal(outFlux, inFlux)
        np.testing.assert_almost_equal(spergel.xValue(cen), spergel.max_sb)


@timer
def test_spergel_radii():
    """Test initialization of Spergel with different types of radius specification.
    """
    import math
    for nu, scale in zip(test_spergel_nu, test_spergel_scale) :

        test_gal = galsim.Spergel(nu=nu, half_light_radius=test_hlr, flux=1.)

        # Check that the returned half-light radius is correct
        print('test_hlr = ',test_hlr)
        print('test_gal hlr, sr = ',test_gal.half_light_radius,test_gal.scale_radius)
        np.testing.assert_almost_equal(
            test_gal.half_light_radius, test_hlr, decimal=5,
            err_msg = "Error in returned HLR for Spergel HLR constructor, nu=%.1f"%nu)

        # Check that the returned flux is correct
        print('test_gal.flux = ',test_gal.flux)
        np.testing.assert_almost_equal(
            test_gal.flux, 1., decimal=5,
            err_msg = "Error in returned Flux for Spergel HLR constructor, nu=%.1f"%nu)

        # (test half-light radii)
        print('flux = ',test_gal.flux)
        print('hlr = ',test_gal.half_light_radius)
        print('scale = ',test_gal.scale_radius)
        got_hlr = test_gal.half_light_radius
        got_flux = test_gal.flux
        # nu = -0.85 is too difficult to numerically integrate
        if nu > -0.85:
            hlr_sum = radial_integrate(test_gal, 0., got_hlr)
            print('hlr_sum = ',hlr_sum)
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*got_flux, decimal=4,
                    err_msg = "Error in Spergel half-light radius constructor, nu=%.1f"%nu)

        # Test constructor using scale radius (test scale radius)
        test_gal = galsim.Spergel(nu=nu, scale_radius=scale, flux=1.)

        # Check that the returned scale radius is correct
        print('test_scale = ',scale)
        print('test_gal hlr, sr = ',test_gal.half_light_radius,test_gal.scale_radius)
        np.testing.assert_almost_equal(
            test_gal.scale_radius, scale, decimal=5,
            err_msg = "Error in returned SR for Sersic SR constructor, nu=%.1f"%nu)

        # Check that the returned flux is correct
        print('test_gal.flux = ',test_gal.flux)
        np.testing.assert_almost_equal(
            test_gal.flux, 1., decimal=5,
            err_msg = "Error in returned Flux for Spergel HLR constructor, nu=%.1f"%nu)

        # (test half-light radius)
        got_hlr = test_gal.half_light_radius
        got_flux = test_gal.flux
        # nu = -0.85 is too difficult to numerically integrate
        if nu > -0.85:
            hlr_sum = radial_integrate(test_gal, 0., got_hlr)
            print('hlr_sum = ',hlr_sum)
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*got_flux, decimal=4,
                    err_msg="Error in HLR for scale_radius constructed Spergel")

        # Check that the getters don't work after modifying the original.
        test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
        try:
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "nu")
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "scale_radius")
        except ImportError:
            pass


@timer
def test_spergel_flux_scaling():
    """Test flux scaling for Spergel.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12

    # loop through spergel nu
    for test_nu in test_spergel_nu:
        # init with hlr and flux only (should be ok given last tests)
        init_obj = galsim.Spergel(test_nu, half_light_radius=test_hlr, flux=test_flux)

        obj2 = init_obj * 2.
        np.testing.assert_almost_equal(
            init_obj.flux, test_flux, decimal=param_decimal,
            err_msg="Flux param inconsistent after __rmul__ (original).")
        np.testing.assert_almost_equal(
            obj2.flux, test_flux * 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __rmul__ (result).")

        obj2 = 2. * init_obj
        np.testing.assert_almost_equal(
            init_obj.flux, test_flux, decimal=param_decimal,
            err_msg="Flux param inconsistent after __mul__ (original).")
        np.testing.assert_almost_equal(
            obj2.flux, test_flux * 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __mul__ (result).")

        obj2 = init_obj / 2.
        np.testing.assert_almost_equal(
             init_obj.flux, test_flux, decimal=param_decimal,
             err_msg="Flux param inconsistent after __div__ (original).")
        np.testing.assert_almost_equal(
            obj2.flux, test_flux / 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __div__ (result).")


@timer
def test_spergel_05():
    """Test the equivalence of Spergel with nu=0.5 and Exponential
    """
    # cf test_exponential()
    re = 1.0
    r0 = re/1.67839
    # The real value of re/r0 = 1.6783469900166605
    hlr_r0 =  1.6783469900166605
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_1.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    spergel = galsim.Spergel(nu=0.5, flux=1., half_light_radius=r0 * hlr_r0)
    spergel.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using Spergel nu=0.5 disagrees with expected result for Exponential")

    check_basic(spergel, "nu=0.5 Spergel")
    do_kvalue(spergel,myImg,"nu=0.5 Spergel")

    # cf test_exponential_properties()
    spergel = galsim.Spergel(nu=0.5, flux=test_flux, half_light_radius=test_scale[0] * hlr_r0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(spergel.centroid, cen)
    np.testing.assert_almost_equal(spergel.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(spergel.flux, test_flux)
    import math
    np.testing.assert_almost_equal(spergel.xValue(cen), 1./(2.*math.pi)*test_flux/test_scale[0]**2,
                                   decimal=5)
    np.testing.assert_almost_equal(spergel.xValue(cen), spergel.max_sb)

    # Also test some random values other than the center:
    expon = galsim.Exponential(flux=test_flux, scale_radius=test_scale[0])
    for (x,y) in [ (0.1,0.2), (-0.5, 0.4), (0, 0.9), (1.2, 0.1), (2,2) ]:
        pos = galsim.PositionD(x,y)
        np.testing.assert_almost_equal(spergel.xValue(pos), expon.xValue(pos), decimal=5)
        np.testing.assert_almost_equal(spergel.kValue(pos), expon.kValue(pos), decimal=5)

@timer
def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Sersic.  Params include n, half_light_radius, scale_radius, flux, trunc, flux_untruncated
    # and gsparams.
    # The following should all test unequal:
    gals = [galsim.Sersic(n=1.1, half_light_radius=1.0),
            galsim.Sersic(n=1.2, half_light_radius=1.0),
            galsim.Sersic(n=1.1, half_light_radius=1.1),
            galsim.Sersic(n=1.1, scale_radius=1.0),
            galsim.Sersic(n=1.1, half_light_radius=1.0, flux=1.1),
            galsim.Sersic(n=1.1, half_light_radius=1.0, trunc=1.8),
            galsim.Sersic(n=1.1, half_light_radius=1.0, trunc=1.8, flux_untruncated=True),
            galsim.Sersic(n=1.1, half_light_radius=1.0, gsparams=gsp)]
    all_obj_diff(gals)

    # DeVaucouleurs.  Params include half_light_radius, scale_radius, flux, trunc, flux_untruncated,
    # and gsparams.
    # The following should all test unequal:
    gals = [galsim.DeVaucouleurs(half_light_radius=1.0),
            galsim.DeVaucouleurs(half_light_radius=1.1),
            galsim.DeVaucouleurs(scale_radius=1.0),
            galsim.DeVaucouleurs(half_light_radius=1.0, flux=1.1),
            galsim.DeVaucouleurs(half_light_radius=1.0, trunc=2.0),
            galsim.DeVaucouleurs(half_light_radius=1.0, trunc=2.0, flux_untruncated=True),
            galsim.DeVaucouleurs(half_light_radius=1.0, gsparams=gsp)]
    all_obj_diff(gals)

    # Spergel.  Params include nu, half_light_radius, scale_radius, flux, and gsparams.
    # The following should all test unequal:
    gals = [galsim.Spergel(nu=0.0, half_light_radius=1.0),
            galsim.Spergel(nu=0.1, half_light_radius=1.0),
            galsim.Spergel(nu=0.0, half_light_radius=1.1),
            galsim.Spergel(nu=0.0, scale_radius=1.0),
            galsim.Spergel(nu=0.0, half_light_radius=1.0, flux=1.1),
            galsim.Spergel(nu=0.0, half_light_radius=1.0, gsparams=gsp)]
    all_obj_diff(gals)


if __name__ == "__main__":
    test_sersic()
    test_sersic_radii()
    test_sersic_flux_scaling()
    test_sersic_05()
    test_sersic_1()
    test_spergel()
    test_spergel_properties()
    test_spergel_radii()
    test_spergel_flux_scaling()
    test_spergel_05()
    test_ne()
