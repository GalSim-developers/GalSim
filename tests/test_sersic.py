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

import pytest
import numpy as np
import os

import galsim
from galsim_test_helpers import *

path, filename = os.path.split(__file__)
imgdir = os.path.join(path, "SBProfile_comparison_images") # Directory containing the reference
                                                           # images.

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
    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    sersic2 = galsim.Sersic(n=3, flux=1.7, half_light_radius=2.3, gsparams=gsp)
    assert sersic2 != sersic
    assert sersic2 == sersic.withGSParams(gsp)
    assert sersic2 == sersic.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    check_basic(sersic, "Sersic")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,"Sersic")

    # Test kvalues
    do_kvalue(sersic,myImg,"Sersic")

    # Check picklability
    check_pickle(sersic, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(sersic)

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
    test_flux = 1.8
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
    check_pickle(sersic, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(sersic)

    # n=4 is also called DeVaucouleurs
    sersic = galsim.Sersic(n=4, flux=1.7, half_light_radius=2.3, trunc=5.9)
    devauc = galsim.DeVaucouleurs(flux=1.7, half_light_radius=2.3, trunc=5.9)
    assert devauc == sersic
    check_basic(devauc, "DeVaucouleurs")

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
    assert_raises(TypeError, galsim.Sersic, n=1.2, scale_radius=3, half_light_radius=1)
    assert_raises(TypeError, galsim.Sersic, n=1.2)
    assert_raises(TypeError, galsim.DeVaucouleurs, scale_radius=3, half_light_radius=1)
    assert_raises(TypeError, galsim.DeVaucouleurs)

    # Allowed range is [0.3, 6.2]
    assert_raises(ValueError, galsim.Sersic, n=0.2, scale_radius=3)
    assert_raises(ValueError, galsim.Sersic, n=6.3, scale_radius=3)

    # trunc must be > sqrt(2) * hlr
    assert_raises(ValueError, galsim.Sersic, n=3, half_light_radius=1, trunc=1.4)
    assert_raises(ValueError, galsim.DeVaucouleurs, half_light_radius=1, trunc=1.4)

    # Other errors
    assert_raises(TypeError, galsim.Sersic, scale_radius=3)
    assert_raises(ValueError, galsim.Sersic, n=3, scale_radius=3, trunc=-1)
    assert_raises(ValueError, galsim.DeVaucouleurs, scale_radius=3, trunc=-1)


@timer
def test_sersic_radii(run_slow):
    """Test initialization of Sersic with different types of radius specification.
    """
    import math
    test_hlr = 1.8

    if not run_slow:
        # If doing a pytest run, we don't actually need to do all 4 sersic n values.
        # Two should be enough to notice if there is a problem, and the full list will be tested
        # when running python test_sersic.py to try to diagnose the problem.
        test_sersic_n = [1.5, -4]
        test_scale = [1.8, 0.002]
    else:
        test_sersic_n = [1.5, 2.5, 4, -4]  # -4 means use explicit DeVauc rather than n=4
        test_scale = [1.8, 0.05, 0.002, 0.002]

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

        check_pickle(test_gal1)
        check_pickle(test_gal2)
        check_pickle(test_gal3)

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
        if n != -4:
            assert_raises(AttributeError, getattr, test_gal_shear, "n")
        assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
        assert_raises(AttributeError, getattr, test_gal_shear, "scale_radius")


@timer
def test_sersic_flux_scaling(run_slow):
    """Test flux scaling for Sersic.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_hlr = 1.8
    test_flux = 17.9
    test_sersic_trunc = [0., 8.5]

    if not run_slow:
        # If doing a pytest run, we don't actually need to do all 4 sersic n values.
        # Two should be enough to notice if there is a problem, and the full list will be tested
        # when running python test_sersic.py to try to diagnose the problem.
        test_sersic_n = [1.5, -4]
    else:
        test_sersic_n = [1.5, 2.5, 4, -4]  # -4 means use explicit DeVauc rather than n=4

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
    test_flux = 1.8

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
    test_sigma = 1.8
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
    test_flux = 17.9
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
    test_scale = 1.8
    sersic = galsim.Sersic(n=1, flux=test_flux, half_light_radius=test_scale * hlr_r0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(sersic.centroid, cen)
    np.testing.assert_almost_equal(sersic.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(sersic.flux, test_flux)
    import math
    np.testing.assert_almost_equal(sersic.xValue(cen), 1./(2.*math.pi)*test_flux/test_scale**2,
                                   decimal=5)
    np.testing.assert_almost_equal(sersic.xValue(cen), sersic.max_sb)

    # Also test some random values other than the center:
    expon = galsim.Exponential(flux=test_flux, scale_radius=test_scale)
    for (x,y) in [ (0.1,0.2), (-0.5, 0.4), (0, 0.9), (1.2, 0.1), (2,2) ]:
        pos = galsim.PositionD(x,y)
        np.testing.assert_almost_equal(sersic.xValue(pos), expon.xValue(pos), decimal=5)
        np.testing.assert_almost_equal(sersic.kValue(pos), expon.kValue(pos), decimal=5)


@timer
def test_sersic_shoot():
    """Test Sersic with photon shooting.  Particularly the flux of the final image.
    """
    rng = galsim.BaseDeviate(1234)
    obj = galsim.Sersic(n=1.5, half_light_radius=3.5, flux=1.e4)
    im = galsim.Image(100,100, scale=1)
    im.setCenter(0,0)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng)
    assert photons2 == photons, "Sersic makePhot not equivalent to drawPhot"

    obj = galsim.DeVaucouleurs(half_light_radius=3.5, flux=1.e4)
    # Need a larger image for devauc wings
    im = galsim.Image(1000,1000, scale=1)
    im.setCenter(0,0)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng)
    assert photons2 == photons, "Sersic makePhot not equivalent to drawPhot"

    # Can do up to around n=6 with this image if hlr is smaller.
    obj = galsim.Sersic(half_light_radius=0.9, n=6.2, flux=1.e4)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng)
    assert photons2 == photons, "Sersic makePhot not equivalent to drawPhot"


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
    check_all_diff(gals)

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
    check_all_diff(gals)


@timer
def test_near_05():
    """Test from issue #1041, where some values of n near but not exactly equal to 0.5 would
    fail to converge in bracketUpper()
    """
    ser1 = galsim.Sersic(n=0.5, half_light_radius=1)
    ser2 = galsim.Sersic(n=0.499999999999, half_light_radius=1)
    ser3 = galsim.Sersic(n=0.500000000001, half_light_radius=1)

    im1 = ser1.drawImage()
    im2 = ser2.drawImage()
    im3 = ser3.drawImage()

    np.testing.assert_allclose(im2.array, im1.array, atol=1.e-12)
    np.testing.assert_allclose(im3.array, im1.array, atol=1.e-12)


if __name__ == "__main__":
    runtests(__file__)
