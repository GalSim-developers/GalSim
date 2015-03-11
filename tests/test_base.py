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

if __name__ == "__main__":
    # If doing a nosetests run, we don't actually need to do all 4 sersic n values.
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


def test_gaussian():
    """Test the generation of a specific Gaussian profile against a known result.
    """
    import time
    t1 = time.time()
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_1.fits"))
    savedImg.setCenter(0,0)
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    gauss = galsim.Gaussian(flux=1, sigma=1)
    # Reference images were made with old centering, which is equivalent to use_true_center=False.
    myImg = gauss.drawImage(myImg, scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.sum() *dx**2, myImg.added_flux, 5,
            err_msg="Gaussian profile GSObject::draw returned wrong added_flux")

    # Check a non-square image
    print myImg.bounds
    recImg = galsim.ImageF(45,66)
    recImg.setCenter(0,0)
    recImg = gauss.drawImage(recImg, scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            recImg[savedImg.bounds].array, savedImg.array, 5,
            err_msg="Drawing Gaussian on non-square image disagrees with expected result")
    np.testing.assert_almost_equal(
            recImg.array.sum() *dx**2, recImg.added_flux, 5,
            err_msg="Gaussian profile GSObject::draw on non-square image returned wrong added_flux")

    # Check with default_params
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=default_params)
    gauss.drawImage(myImg,scale=0.2, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian with default_params disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=galsim.GSParams())
    gauss.drawImage(myImg,scale=0.2, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian with GSParams() disagrees with expected result")

    # Use non-unity values.
    gauss = galsim.Gaussian(flux=1.7, sigma=2.3)

    # Test photon shooting.
    do_shoot(gauss,myImg,"Gaussian")

    # Test kvalues
    do_kvalue(gauss,myImg,"Gaussian")

    # Check picklability
    do_pickle(galsim.GSParams())  # Check GSParams explicitly here too.
    do_pickle(galsim.GSParams(
        minimum_fft_size = 12,
        maximum_fft_size = 40,
        folding_threshold = 1.e-1,
        maxk_threshold = 2.e-1,
        kvalue_accuracy = 3.e-1,
        xvalue_accuracy = 4.e-1,
        shoot_accuracy = 5.e-1,
        realspace_relerr = 6.e-1,
        realspace_abserr = 7.e-1,
        integration_relerr = 8.e-1,
        integration_abserr = 9.e-1))
    do_pickle(gauss.SBProfile, lambda x: (x.getSigma(), x.getFlux(), x.getGSParams()))
    do_pickle(gauss, lambda x: x.drawImage(method='no_pixel'))

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_gaussian_properties():
    """Test some basic properties of the Gaussian profile.
    """
    import time
    t1 = time.time()
    gauss = galsim.Gaussian(flux=test_flux, sigma=test_sigma)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(gauss.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(gauss.maxK(), 3.7169221888498383 / test_sigma)
    np.testing.assert_almost_equal(gauss.stepK(), 0.533644625664 / test_sigma)
    np.testing.assert_equal(gauss.kValue(cen), (1+0j) * test_flux)
    import math
    np.testing.assert_almost_equal(gauss.xValue(cen), 1./(2.*math.pi) * test_flux / test_sigma**2)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        gauss = galsim.Gaussian(flux=inFlux, sigma=2.)
        outFlux = gauss.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_gaussian_radii():
    """Test initialization of Gaussian with different types of radius specification.
    """
    import time
    t1 = time.time()
    import math
    # Test constructor using half-light-radius:
    test_gal = galsim.Gaussian(flux = 1., half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Gaussian constructor with half-light radius")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Gaussian initialized with half-light radius")

    # test that getSigma() method provides correct sigma
    got_sigma = test_gal.getSigma()
    test_sigma_ratio = (test_gal.xValue(galsim.PositionD(got_sigma, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'sigma ratio = ', test_sigma_ratio
    np.testing.assert_almost_equal(
            test_sigma_ratio, math.exp(-0.5), decimal=4,
            err_msg="Error in sigma for Gaussian initialized with half-light radius")

    # Test constructor using sigma:
    test_gal = galsim.Gaussian(flux = 1., sigma = test_sigma)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_sigma,0)) / center
    print 'sigma ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, np.exp(-0.5), decimal=4,
            err_msg="Error in Gaussian constructor with sigma")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print 'hlr_sum (profile initialized with sigma) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Gaussian initialized with sigma.")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Gaussian initialized with sigma.")

    # Test constructor using FWHM:
    test_gal = galsim.Gaussian(flux = 1., fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Gaussian constructor with fwhm")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print 'hlr_sum (profile initialized with fwhm) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Gaussian initialized with FWHM.")

    # test that getSigma() method provides correct sigma
    got_sigma = test_gal.getSigma()
    test_sigma_ratio = (test_gal.xValue(galsim.PositionD(got_sigma, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'sigma ratio = ', test_sigma_ratio
    np.testing.assert_almost_equal(
            test_sigma_ratio, math.exp(-0.5), decimal=4,
            err_msg="Error in sigma for Gaussian initialized with FWHM.")

    # Check that the getters don't work after modifying the original.
    # Note: I test all the modifiers here.  For the rest of the profile types, I'll
    # just confirm that it is true of shear.  I don't think that has any chance
    # of missing anything.
    test_gal_copy = test_gal.copy()
    print 'fwhm = ',test_gal_copy.getFWHM()
    print 'hlr = ',test_gal_copy.getHalfLightRadius()
    print 'sigma = ',test_gal_copy.getSigma()
    # They still work after copy
    test_gal_flux1 = test_gal_copy * 3.
    # But not after rescaling the flux.
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux1, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux1, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux1, "getSigma")
    except ImportError:
        # assert_raises requires nose, which we don't want to force people to install.
        # So if they are running this without nose, we just skip these tests.
        pass

    test_gal_flux2 = test_gal.withFlux(3.)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux2, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux2, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux2, "getSigma")
    except ImportError:
        pass

    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getSigma")
    except ImportError:
        pass

    test_gal_rot = test_gal.rotate(theta = 0.5 * galsim.radians)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_rot, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_rot, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_rot, "getSigma")
    except ImportError:
        pass

    test_gal_shift = test_gal.shift(dx=0.11, dy=0.04)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shift, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shift, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shift, "getSigma")
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_gaussian_flux_scaling():
    """Test flux scaling for Gaussian.
    """
    import time
    t1 = time.time()

    # decimal point to go to for parameter value comparisons
    param_decimal = 12

    # init with sigma and flux only (should be ok given last tests)
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_exponential():
    """Test the generation of a specific exp profile against a known result.
    """
    import time
    t1 = time.time()
    re = 1.0
    # Note the factor below should really be 1.6783469900166605, but the value of 1.67839 is
    # retained here as it was used by SBParse to generate the original known result (this changed
    # in commit b77eb05ab42ecd31bc8ca03f1c0ae4ee0bc0a78b.
    # The value of this test for regression purposes is not harmed by retaining the old scaling, it
    # just means that the half light radius chosen for the test is not really 1, but 0.999974...
    r0 = re/1.67839
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_1.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    expon = galsim.Exponential(flux=1., scale_radius=r0)
    expon.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential disagrees with expected result")

    # Check with default_params
    expon = galsim.Exponential(flux=1., scale_radius=r0, gsparams=default_params)
    expon.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential with default_params disagrees with expected result")
    expon = galsim.Exponential(flux=1., scale_radius=r0, gsparams=galsim.GSParams())
    expon.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential with GSParams() disagrees with expected result")

    # Use non-unity values.
    expon = galsim.Exponential(flux=1.7, scale_radius=0.91)

    # Test photon shooting.
    do_shoot(expon,myImg,"Exponential")

    # Test kvalues
    do_kvalue(expon,myImg,"Exponential")

    # Check picklability
    do_pickle(expon.SBProfile, lambda x: (x.getScaleRadius(), x.getFlux(), x.getGSParams()))
    do_pickle(expon, lambda x: x.drawImage(method='no_pixel'))

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_exponential_properties():
    """Test some basic properties of the Exponential profile.
    """
    import time
    t1 = time.time()
    expon = galsim.Exponential(flux=test_flux, scale_radius=test_scale[0])
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(expon.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(expon.maxK(), 10 / test_scale[0])
    np.testing.assert_almost_equal(expon.stepK(), 0.37436747851 / test_scale[0])
    np.testing.assert_equal(expon.kValue(cen), (1+0j) * test_flux)
    import math
    np.testing.assert_almost_equal(expon.xValue(cen), 1./(2.*math.pi)*test_flux/test_scale[0]**2)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        expon = galsim.Exponential(flux=inFlux, scale_radius=1.8)
        outFlux = expon.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_exponential_radii():
    """Test initialization of Exponential with different types of radius specification.
    """
    import time
    t1 = time.time()
    import math
    # Test constructor using half-light-radius:
    test_gal = galsim.Exponential(flux = 1., half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Exponential constructor with half-light radius")

    # then test scale getter
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_gal.getScaleRadius(),0)) / center
    print 'scale ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, np.exp(-1.0), decimal=4,
            err_msg="Error in getScaleRadius for Exponential constructed with half light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Exponential(flux = 1., scale_radius = test_scale[0])
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale[0],0)) / center
    print 'scale ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, np.exp(-1.0), decimal=4,
            err_msg="Error in Exponential constructor with scale")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print 'hlr_sum (profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Exponential initialized with scale_radius.")

    # Check that the getters don't work after modifying the original.
    test_gal_copy = test_gal.copy()
    print 'hlr = ',test_gal_copy.getHalfLightRadius()
    print 'scale = ',test_gal_copy.getScaleRadius()
    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getScaleRadius")
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_exponential_flux_scaling():
    """Test flux scaling for Exponential.
    """
    import time
    t1 = time.time()

    # decimal point to go to for parameter value comparisons
    param_decimal = 12

    # init with scale and flux only (should be ok given last tests)
    obj = galsim.Exponential(scale_radius=test_scale[0], flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Exponential(scale_radius=test_scale[0], flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Exponential(scale_radius=test_scale[0], flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Exponential(scale_radius=test_scale[0], flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Exponential(scale_radius=test_scale[0], flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sersic():
    """Test the generation of a specific Sersic profile against a known result.
    """
    import time
    t1 = time.time()

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

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,"Sersic")

    # Test kvalues
    do_kvalue(sersic,myImg,"Sersic")

    # Check picklability
    do_pickle(sersic.SBProfile,
              lambda x: (x.getScaleRadius(), x.getTrunc(), x.getFlux(), x.getGSParams()))
    do_pickle(sersic, lambda x: x.drawImage(method='no_pixel'))


    # Now repeat everything using a truncation.  (Above had no truncation.)

    # Test Truncated Sersic
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1_10.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, trunc=10)
    sersic.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using truncated GSObject Sersic disagrees with expected result")

    # Use non-unity values.
    sersic = galsim.Sersic(n=3, flux=1.7, half_light_radius=2.3, trunc=5.9)

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,"Truncated Sersic")

    # Test kvalues
    do_kvalue(sersic,myImg, "Truncated Sersic")

    # Check picklability
    do_pickle(sersic.SBProfile,
              lambda x: (x.getScaleRadius(), x.getTrunc(), x.getFlux(), x.getGSParams()))
    do_pickle(sersic, lambda x: x.drawImage(method='no_pixel'))

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sersic_radii():
    """Test initialization of Sersic with different types of radius specification.
    """
    import time
    t1 = time.time()
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

        # Check that the returned half-light radius is correct
        print 'test_hlr = ',test_hlr
        print 'test_gal1 hlr, sr = ',test_gal1.getHalfLightRadius(),test_gal1.getScaleRadius()
        print 'test_gal2 hlr, sr = ',test_gal2.getHalfLightRadius(),test_gal2.getScaleRadius()
        print 'test_gal3 hlr, sr = ',test_gal3.getHalfLightRadius(),test_gal3.getScaleRadius()
        np.testing.assert_almost_equal(
            test_gal1.getHalfLightRadius(), test_hlr, decimal=5,
            err_msg = "Error in returned HLR for Sersic HLR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal2.getHalfLightRadius(), test_hlr, decimal=5,
            err_msg = "Error in returned HLR for truncated Sersic HLR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal3.getScaleRadius(), test_gal1.getScaleRadius(), decimal=5,
            err_msg = "Error in returned SR for flux_untruncated Sersic HLR constructor, n=%.1f"%n)

        # Check that the returned flux is correct
        print 'test_gal1.getFlux() = ',test_gal1.getFlux()
        print 'test_gal2.getFlux() = ',test_gal2.getFlux()
        print 'test_gal3.getFlux() = ',test_gal3.getFlux()
        np.testing.assert_almost_equal(
            test_gal1.getFlux(), 1., decimal=5,
            err_msg = "Error in returned Flux for Sersic HLR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal2.getFlux(), 1., decimal=5,
            err_msg = "Error in returned Flux for truncated Sersic HLR constructor, n=%.1f"%n)
        # test_gal3 doesn't match getFlux(), but should have central value match test_gal1.
        center1 = test_gal1.xValue(galsim.PositionD(0,0))
        center3 = test_gal3.xValue(galsim.PositionD(0,0))
        print 'peak value 1,3 = ', center1, center3
        np.testing.assert_almost_equal(
                center1, center3, 9,
                "Error in flux_untruncated Sersic normalization HLR constructor, n=%.1f"%n)

        # (test half-light radii)
        for test_gal, label in zip(gal_list, gal_labels):
            print 'flux = ',test_gal.getFlux()
            print 'hlr = ',test_gal.getHalfLightRadius()
            print 'scale = ',test_gal.getScaleRadius()
            got_hlr = test_gal.getHalfLightRadius()
            got_flux = test_gal.getFlux()
            hlr_sum = radial_integrate(test_gal, 0., got_hlr)
            print 'hlr_sum = ',hlr_sum
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*got_flux, decimal=4,
                    err_msg = "Error in %s half-light radius constructor, n=%.1f"%(label,n))

        # (test scale radii)
        for test_gal, label in zip(gal_list, gal_labels):
            got_sr = test_gal.getScaleRadius()
            center = test_gal.xValue(galsim.PositionD(0,0))
            ratio = test_gal.xValue(galsim.PositionD(got_sr,0)) / center
            print 'scale ratio = ',ratio
            np.testing.assert_almost_equal(
                    ratio, np.exp(-1.0), decimal=4,
                    err_msg="Error in getScaleRadius for HLR constructed %s"%label)

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
        print 'test_scale = ',scale
        print 'test_gal1 hlr, sr = ',test_gal1.getHalfLightRadius(),test_gal1.getScaleRadius()
        print 'test_gal2 hlr, sr = ',test_gal2.getHalfLightRadius(),test_gal2.getScaleRadius()
        print 'test_gal3 hlr, sr = ',test_gal3.getHalfLightRadius(),test_gal3.getScaleRadius()
        np.testing.assert_almost_equal(
            test_gal1.getScaleRadius(), scale, decimal=5,
            err_msg = "Error in returned SR for Sersic SR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal2.getScaleRadius(), scale, decimal=5,
            err_msg = "Error in returned SR for truncated Sersic SR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal3.getScaleRadius(), scale, decimal=5,
            err_msg = "Error in returned SR for truncated Sersic SR constructor, n=%.1f"%n)

        # Returned HLR should match for gals 2,3
        got_hlr2 = test_gal2.getHalfLightRadius()
        got_hlr3 = test_gal3.getHalfLightRadius()
        print 'half light radii of truncated, scale_radius constructed Sersic =',got_hlr2,got_hlr3
        np.testing.assert_almost_equal(
                got_hlr2, got_hlr3, decimal=4,
                err_msg="Error in HLR for scale_radius constructed flux_untruncated Sersic (II).")

        # Check that the returned flux is correct
        print 'test_gal1.getFlux() = ',test_gal1.getFlux()
        print 'test_gal2.getFlux() = ',test_gal2.getFlux()
        print 'test_gal3.getFlux() = ',test_gal3.getFlux()
        np.testing.assert_almost_equal(
            test_gal1.getFlux(), 1., decimal=5,
            err_msg = "Error in returned Flux for Sersic HLR constructor, n=%.1f"%n)
        np.testing.assert_almost_equal(
            test_gal2.getFlux(), 1., decimal=5,
            err_msg = "Error in returned Flux for truncated Sersic HLR constructor, n=%.1f"%n)
        center1 = test_gal1.xValue(galsim.PositionD(0,0))
        center3 = test_gal3.xValue(galsim.PositionD(0,0))
        print 'peak value 1,3 = ', center1, center3
        np.testing.assert_almost_equal(
                center1, center3, 9,
                "Error in flux_untruncated Sersic normalization HLR constructor, n=%.1f"%n)

        # (test scale radii)
        for test_gal, label in zip(gal_list, gal_labels):
            center = test_gal.xValue(galsim.PositionD(0,0))
            ratio = test_gal.xValue(galsim.PositionD(scale,0)) / center
            print 'scale ratio = ',ratio
            np.testing.assert_almost_equal(
                    ratio, np.exp(-1.0), decimal=4,
                    err_msg="Error in %s scale radius constructor, n=%.1f"%(label,n))

        # (test half-light radius)
        for test_gal, label in zip(gal_list, gal_labels):
            got_hlr = test_gal.getHalfLightRadius()
            got_flux = test_gal.getFlux()
            hlr_sum = radial_integrate(test_gal, 0., got_hlr)
            print 'hlr_sum = ',hlr_sum
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*got_flux, decimal=4,
                    err_msg="Error in HLR for scale_radius constructed %s"%label)

        # Check that the getters don't work after modifying the original.
        test_gal_copy = test_gal1.copy()
        # They still work after copy()
        if n != -4:
            print 'n = ',test_gal_copy.getN()
        print 'hlr = ',test_gal_copy.getHalfLightRadius()
        print 'sr = ',test_gal_copy.getScaleRadius()
        test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
        # But not after shear() (or others, but this is a sufficient test here)
        try:
            if n != -4:
                np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getN")
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getScaleRadius")
        except ImportError:
            pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sersic_flux_scaling():
    """Test flux scaling for Sersic.
    """
    import time
    t1 = time.time()

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

            # Test in place *= and /=
            obj = init_obj.copy()
            obj *= 2.
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __imul__.")
            np.testing.assert_almost_equal(
                init_obj.getFlux(), test_flux, decimal=param_decimal,
                err_msg="obj.copy() didn't produce a separate copy.")
            obj = init_obj.copy()
            obj /= 2.
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux / 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __idiv__.")

            obj2 = init_obj * 2.
            np.testing.assert_almost_equal(
                init_obj.getFlux(), test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (original).")
            np.testing.assert_almost_equal(
                obj2.getFlux(), test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (result).")

            obj2 = 2. * init_obj
            np.testing.assert_almost_equal(
                init_obj.getFlux(), test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (original).")
            np.testing.assert_almost_equal(
                obj2.getFlux(), test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (result).")

            obj2 = init_obj / 2.
            np.testing.assert_almost_equal(
                 init_obj.getFlux(), test_flux, decimal=param_decimal,
                 err_msg="Flux param inconsistent after __div__ (original).")
            np.testing.assert_almost_equal(
                obj2.getFlux(), test_flux / 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __div__ (result).")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


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
    print 'saved image center = ',savedImg(0,0)
    print 'image center = ',myImg(0,0)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using Sersic with n=0.5 disagrees with expected result for Gaussian")

    do_kvalue(sersic,myImg,"n=0.5 Sersic")

    # cf test_gaussian_properties()
    sersic = galsim.Sersic(n=0.5, flux=test_flux, half_light_radius=test_sigma * hlr_sigma)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(sersic.centroid(), cen)
    np.testing.assert_equal(sersic.kValue(cen), (1+0j) * test_flux)
    import math
    np.testing.assert_almost_equal(sersic.xValue(cen), 1./(2.*math.pi) * test_flux / test_sigma**2,
                                   decimal=5)

    # Also test some random values other than the center:
    gauss = galsim.Gaussian(flux=test_flux, sigma=test_sigma)
    for (x,y) in [ (0.1,0.2), (-0.5, 0.4), (0, 0.9), (1.2, 0.1), (2,2) ]:
        pos = galsim.PositionD(x,y)
        np.testing.assert_almost_equal(sersic.xValue(pos), gauss.xValue(pos), decimal=5)
        np.testing.assert_almost_equal(sersic.kValue(pos), gauss.kValue(pos), decimal=5)


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

    do_kvalue(sersic,myImg,"n=1 Sersic")

    # cf test_exponential_properties()
    sersic = galsim.Sersic(n=1, flux=test_flux, half_light_radius=test_scale[0] * hlr_r0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(sersic.centroid(), cen)
    np.testing.assert_equal(sersic.kValue(cen), (1+0j) * test_flux)
    import math
    np.testing.assert_almost_equal(sersic.xValue(cen), 1./(2.*math.pi)*test_flux/test_scale[0]**2,
                                   decimal=5)

    # Also test some random values other than the center:
    expon = galsim.Exponential(flux=test_flux, scale_radius=test_scale[0])
    for (x,y) in [ (0.1,0.2), (-0.5, 0.4), (0, 0.9), (1.2, 0.1), (2,2) ]:
        pos = galsim.PositionD(x,y)
        np.testing.assert_almost_equal(sersic.xValue(pos), expon.xValue(pos), decimal=5)
        np.testing.assert_almost_equal(sersic.kValue(pos), expon.kValue(pos), decimal=5)


def test_airy():
    """Test the generation of a specific Airy profile against a known result.
    """
    import time
    t1 = time.time()
    savedImg = galsim.fits.read(os.path.join(imgdir, "airy_.8_.1.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1)
    airy.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy disagrees with expected result")

    # Check with default_params
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1, gsparams=default_params)
    airy.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy with default_params disagrees with expected result")
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1, gsparams=galsim.GSParams())
    airy.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy with GSParams() disagrees with expected result")

    # Test photon shooting.
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.0, flux=test_flux)
    do_shoot(airy,myImg,"Airy obscuration=0.0")
    airy2 = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=test_flux)
    do_shoot(airy2,myImg,"Airy obscuration=0.1")

    # Test kvalues
    do_kvalue(airy,myImg, "Airy obscuration=0.0")
    do_kvalue(airy2,myImg, "Airy obscuration=0.1")

    # Check picklability
    do_pickle(airy.SBProfile, lambda x: (x.getLamOverD(), x.getFlux(), x.getGSParams()))
    do_pickle(airy, lambda x: x.drawImage(method='no_pixel'))

    # Test initialization separately with lam and diam, in various units.  Since the above profiles
    # have lam/diam = 1./0.8 in arbitrary units, we will tell it that lam=1.e9 nm and diam=0.8 m,
    # and use `scale_unit` of galsim.radians.  This is rather silly, but it should work.
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1.7)
    test_im1 = airy.drawImage(scale=0.2)
    test_im2 = test_im1.copy()
    airy2 = galsim.Airy(lam=1.e9, diam=0.8, scale_unit=galsim.radians, obscuration=0.1, flux=1.7)
    airy2.drawImage(image=test_im2, scale=0.2)
    np.testing.assert_array_almost_equal(
            test_im1.array, test_im2.array, 8,
            err_msg="Using GSObject Airy with different kwargs disagrees with expected result")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_airy_radii():
    """Test Airy half light radius and FWHM correctly set and match image.
    """
    import time
    t1 = time.time()
    import math
    # Test constructor using lam_over_diam: (only option for Airy)
    test_gal = galsim.Airy(lam_over_diam= 1./0.8, flux=1.)
    # test half-light-radius getter
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Airy half-light radius")

    # test FWHM getter
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(.5 * test_gal.getFWHM(),0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in getFWHM() for Airy.")

    # Check that the getters don't work after modifying the original.
    test_gal_copy = test_gal.copy()
    print 'fwhm = ',test_gal_copy.getFWHM()
    print 'hlr = ',test_gal_copy.getHalfLightRadius()
    print 'lod = ',test_gal_copy.getLamOverD()
    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getFWHM");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getLamOverD")
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_airy_flux_scaling():
    """Test flux scaling for Airy.
    """
    import time
    t1 = time.time()

    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_loD = 1.9
    test_obscuration = 0.32

    # init with lam_over_r0 and flux only (should be ok given last tests)
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_box():
    """Test the generation of a specific box profile against a known result.
    """
    import time
    t1 = time.time()
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    myImg.setCenter(0,0)

    pixel = galsim.Pixel(scale=1, flux=1)
    pixel.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel disagrees with expected result")

    # Check with default_params
    pixel = galsim.Pixel(scale=1, flux=1, gsparams=default_params)
    pixel.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel with default_params disagrees with expected result")
    pixel = galsim.Pixel(scale=1, flux=1, gsparams=galsim.GSParams())
    pixel.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel with GSParams() disagrees with expected result")

    # Use non-unity values.
    pixel = galsim.Pixel(flux=1.7, scale=2.3)

    # Test photon shooting.
    do_shoot(pixel,myImg,"Pixel")

    # Check picklability
    do_pickle(pixel.SBProfile,
              lambda x: (x.getWidth(), x.getHeight(), x.getFlux(), x.getGSParams()))
    do_pickle(pixel, lambda x: x.drawImage(method='no_pixel'))

    # Check that non-square Box profiles work correctly
    scale = 0.2939  # Use a strange scale here to make sure that the centers of the pixels
                    # never fall on the box edge, otherwise it gets a bit weird to know what
                    # the correct SB value is for that pixel.
    im = galsim.ImageF(16,16, scale=scale)
    gsp = galsim.GSParams(maximum_fft_size = 30000)
    for (width,height) in [ (3,2), (1.7, 2.7), (2.2222, 3.1415) ]:
        print 'width, height = ',width,height
        box = galsim.Box(width=width, height=height, flux=test_flux, gsparams=gsp)
        print 'im.bounds = ',im.bounds
        print 'im.scale = ',im.scale
        do_shoot(box,im,"Box with width,height = %f,%f"%(width,height))
        if __name__ == '__main__':
            # These are slow because they require a pretty huge fft.
            # So only do them if running as main.
            do_kvalue(box,im,"Box with width,height = %f,%f"%(width,height))

    # Check picklability
    do_pickle(box.SBProfile, lambda x: (x.getWidth(), x.getHeight(), x.getFlux(), x.getGSParams()))
    do_pickle(box, lambda x: x.drawImage(method='no_pixel'))

    # Check sheared boxes the same way
    box = galsim.Box(width=3, height=2, flux=test_flux, gsparams=gsp)
    box = box.shear(galsim.Shear(g1=0.2, g2=-0.3))
    do_shoot(box,im, "Sheared Box")
    if __name__ == '__main__':
        do_kvalue(box,im, "Sheared Box")
        do_pickle(box, lambda x: x.drawImage(method='no_pixel'))

    # This is also a profile that may be convolved using real space convolution, so test that.
    if __name__ == '__main__':
        conv = galsim.Convolve(box, galsim.Pixel(scale=scale), real_space=True)
        do_kvalue(conv,im, "Sheared Box convolved with pixel in real space")
        do_pickle(conv, lambda x: x.xValue(0.123,-0.456))

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_tophat():
    """Test the generation of a specific tophat profile against a known result.
    """
    import time
    t1 = time.time()
    savedImg = galsim.fits.read(os.path.join(imgdir, "tophat_101.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    myImg.setCenter(0,0)

    # There are numerical issues with using radius = 1, since many points are right on the edge
    # of the circle.  e.g. (+-1,0), (0,+-1), (+-0.6,+-0.8), (+-0.8,+-0.6).  And in practice, some
    # of these end up getting drawn and not others, which means it's not a good choice for a unit
    # test since it wouldn't be any less correct for a different subset of these points to be
    # drawn. Using r = 1.01 solves this problem and makes the result symmetric.
    tophat = galsim.TopHat(radius=1.01, flux=1)
    tophat.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject TopHat disagrees with expected result")

    # Check with default_params
    tophat = galsim.TopHat(radius=1.01, flux=1, gsparams=default_params)
    tophat.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject TopHat with default_params disagrees with expected result")
    tophat = galsim.TopHat(radius=1.01, flux=1, gsparams=galsim.GSParams())
    tophat.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject TopHat with GSParams() disagrees with expected result")

    # Use non-unity values.
    tophat = galsim.TopHat(flux=1.7, radius=2.3)

    # Test photon shooting.
    do_shoot(tophat,myImg,"TopHat")

    # Test shoot and kvalue
    scale = 0.2939
    im = galsim.ImageF(16,16, scale=scale)
    # The choices of radius here are fairly specific.  If the edge of the circle comes too close
    # to the center of one of the pixels, then the test will fail, since the Fourier draw method
    # will blur the edge a bit and give some flux to that pixel.
    for radius in [ 1.2, 0.83, 2.11 ]:
        print 'radius = ',radius
        tophat = galsim.TopHat(radius=radius, flux=test_flux)
        print 'im.bounds = ',im.bounds
        print 'im.scale = ',im.scale
        do_shoot(tophat,im,"TopHat with radius = %f"%radius)
        do_kvalue(tophat,im,"TopHat with radius = %f"%radius)

        # This is also a profile that may be convolved using real space convolution, so test that.
        conv = galsim.Convolve(tophat, galsim.Pixel(scale=scale), real_space=True)
        do_kvalue(conv,im, "Sheared TopHat convolved with pixel in real space")

    # Check picklability
    do_pickle(tophat.SBProfile, lambda x: (x.getRadius(), x.getFlux(), x.getGSParams()))
    do_pickle(tophat, lambda x: x.drawImage(method='no_pixel'))

    # Check sheared tophat the same way
    tophat = galsim.TopHat(radius=1.2, flux=test_flux)
    # Again, the test is very sensitive to the choice of shear here.  Most values fail because 
    # some pixel center gets too close to the resulting ellipse for the fourier draw to match
    # the real-space draw at the required accuracy.
    tophat = tophat.shear(galsim.Shear(g1=0.15, g2=-0.33))
    do_shoot(tophat,im, "Sheared TopHat")
    do_kvalue(tophat,im, "Sheared TopHat")

    # Check picklability
    do_pickle(tophat, lambda x: x.drawImage(method='no_pixel'))

    # Check real-space convolution of the sheared tophat.
    conv = galsim.Convolve(tophat, galsim.Pixel(scale=scale), real_space=True)
    do_kvalue(conv,im, "Sheared TopHat convolved with pixel in real space")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_moffat():
    """Test the generation of a specific Moffat profile against a known result.
    """
    import time
    t1 = time.time()
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_2_5.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    # Code was formerly:
    # moffat = galsim.Moffat(beta=2, truncationFWHM=5, flux=1, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in
    # physical units.  However, the same profile can be constructed using
    # fwhm=1.3178976627539716
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    fwhm_backwards_compatible = 1.3178976627539716
    moffat = galsim.Moffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1)
    moffat.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat disagrees with expected result")

    # Check with default_params
    moffat = galsim.Moffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1,
                           gsparams=default_params)
    moffat.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat with default_params disagrees with expected result")
    moffat = galsim.Moffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1,
                           gsparams=galsim.GSParams())
    moffat.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat with GSParams() disagrees with expected result")

    # Use non-unity values.
    moffat = galsim.Moffat(beta=3.7, flux=1.7, half_light_radius=2.3, trunc=8.2)

    # Test photon shooting.
    do_shoot(moffat,myImg,"Moffat")

    # Test kvalues
    do_kvalue(moffat,myImg, "Moffat")

    # Check picklability
    do_pickle(moffat.SBProfile,
              lambda x: (x.getBeta(), x.getScaleRadius(), x.getTrunc(), x.getFlux(),\
                         x.getGSParams()))
    do_pickle(moffat, lambda x: x.drawImage(method='no_pixel'))

    # The code for untruncated Moffat profiles is specialized for particular beta values, so
    # test each of these:
    for beta in [ 1.5, 2, 2.5, 3, 3.5, 4, 2.3 ]:  # The one last is for the generic case.
        moffat = galsim.Moffat(beta=beta, half_light_radius=0.7, flux=test_flux)
        do_kvalue(moffat,myImg,"Untruncated Moffat with beta=%f"%beta)
        # Don't bother repeating the do_shoot tests, since they are rather slow, and the code
        # isn't different for the different beta values.

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_moffat_properties():
    """Test some basic properties of the Moffat profile.
    """
    import time
    t1 = time.time()
    # Code was formerly:
    # psf = galsim.Moffat(beta=2.0, truncationFWHM=2, flux=test_flux, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in
    # physical units.  However, the same profile can be constructed using
    # fwhm=1.4686232496771867,
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    fwhm_backwards_compatible = 1.4686232496771867
    psf = galsim.Moffat(beta=2.0, fwhm=fwhm_backwards_compatible,
                        trunc=2*fwhm_backwards_compatible, flux=test_flux)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxK(), 11.613036117918105)
    np.testing.assert_almost_equal(psf.stepK(), 0.62831853071795873)
    np.testing.assert_almost_equal(psf.kValue(cen), test_flux+0j)
    np.testing.assert_almost_equal(psf.getHalfLightRadius(), 1.0)
    np.testing.assert_almost_equal(psf.getFWHM(), fwhm_backwards_compatible)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.50654651638242509)

    # Now create the same profile using the half_light_radius:
    psf = galsim.Moffat(beta=2.0, half_light_radius=1.,
                        trunc=2*fwhm_backwards_compatible, flux=test_flux)
    np.testing.assert_equal(psf.centroid(), cen)
    np.testing.assert_almost_equal(psf.maxK(), 11.613036112206663)
    np.testing.assert_almost_equal(psf.stepK(), 0.62831853071795862)
    np.testing.assert_almost_equal(psf.kValue(cen), test_flux+0j)
    np.testing.assert_almost_equal(psf.getHalfLightRadius(), 1.0)
    np.testing.assert_almost_equal(psf.getFWHM(), fwhm_backwards_compatible)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.50654651638242509)

    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        psfFlux = galsim.Moffat(2.0, fwhm=fwhm_backwards_compatible,
                                trunc=2*fwhm_backwards_compatible, flux=inFlux)
        outFlux = psfFlux.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_moffat_radii():
    """Test initialization of Moffat with different types of radius specification.
    """
    import time
    t1 = time.time()
    import math

    test_beta = 2.

    # Test constructor using half-light-radius:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with half-light radius")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with half-light radius")

    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with half-light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, scale_radius = test_scale[0])
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale[0],0)) / center
    print 'scale ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, pow(2,-test_beta), decimal=4,
            err_msg="Error in Moffat constructor with scale")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print 'hlr_sum (profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Moffat initialized with scale radius.")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with scale radius")

    # Test constructor using FWHM:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with fwhm")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print 'hlr_sum (profile initialized with FWHM) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Moffat initialized with FWHM.")
    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with scale radius")

    # Now repeat everything using a severe truncation.  (Above had no truncation.)

    # Test constructor using half-light-radius:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, half_light_radius = test_hlr,
                             trunc=2*test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with half-light radius")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with half-light radius")

    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with half-light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Moffat(flux=1., beta=test_beta, trunc=2*test_scale[0],
                             scale_radius=test_scale[0])
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale[0],0)) / center
    print 'scale ratio = ', ratio
    np.testing.assert_almost_equal(
            ratio, pow(2,-test_beta), decimal=4,
            err_msg="Error in Moffat constructor with scale")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print 'hlr_sum (truncated profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for truncated Moffat "+
                    "initialized with scale radius.")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for truncated Moffat initialized with scale radius")

    # Test constructor using FWHM:
    test_gal = galsim.Moffat(flux=1, beta=test_beta, trunc=2.*test_fwhm,
                             fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ', ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with fwhm")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print 'hlr_sum (truncated profile initialized with FWHM) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for truncated Moffat initialized with FWHM.")

    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for truncated Moffat initialized with scale radius")

    # Check that the getters don't work after modifying the original.
    test_gal_copy = test_gal.copy()
    print 'beta = ',test_gal_copy.getBeta()
    print 'fwhm = ',test_gal_copy.getFWHM()
    print 'hlr = ',test_gal_copy.getHalfLightRadius()
    print 'scale = ',test_gal_copy.getScaleRadius()
    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getBeta");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getFWHM");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getScaleRadius");
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_moffat_flux_scaling():
    """Test flux scaling for Moffat.
    """
    import time
    t1 = time.time()

    # decimal point to go to for parameter value comparisons
    param_decimal = 12

    for test_beta in [ 1.5, 2., 2.5, 3., 3.8 ]:
        for test_trunc in [ 0., 8.5 ]:

            # init with scale_radius only (should be ok given last tests)
            obj = galsim.Moffat(scale_radius=test_scale[0], beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
            obj *= 2.
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __imul__.")
            obj = galsim.Moffat(scale_radius=test_scale[0], beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
            obj /= 2.
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux / 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __idiv__.")
            obj = galsim.Moffat(scale_radius=test_scale[0], beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
            obj2 = obj * 2.
            # First test that original obj is unharmed... (also tests that .copy() is working)
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (original).")
            # Then test new obj2 flux
            np.testing.assert_almost_equal(
                obj2.getFlux(), test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (result).")
            obj = galsim.Moffat(scale_radius=test_scale[0], beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
            obj2 = 2. * obj
            # First test that original obj is unharmed... (also tests that .copy() is working)
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (original).")
            # Then test new obj2 flux
            np.testing.assert_almost_equal(
                obj2.getFlux(), test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (result).")
            obj = galsim.Moffat(scale_radius=test_scale[0], beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
            obj2 = obj / 2.
            # First test that original obj is unharmed... (also tests that .copy() is working)
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __div__ (original).")
            # Then test new obj2 flux
            np.testing.assert_almost_equal(
                obj2.getFlux(), test_flux / 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __div__ (result).")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_kolmogorov():
    """Test the generation of a specific Kolmogorov profile against a known result.
    """
    import time
    t1 = time.time()
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

    # Test photon shooting.
    do_shoot(kolm,myImg,"Kolmogorov")

    # Test kvalues
    do_kvalue(kolm,myImg, "Kolmogorov")

    # Check picklability
    do_pickle(kolm.SBProfile, lambda x: (x.getLamOverR0(), x.getFlux(), x.getGSParams()))
    do_pickle(kolm, lambda x: x.drawImage(method='no_pixel'))

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_kolmogorov_properties():
    """Test some basic properties of the Kolmogorov profile.
    """
    import time
    t1 = time.time()

    lor = 1.5
    psf = galsim.Kolmogorov(lam_over_r0=lor, flux=test_flux)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxK(), 8.6440505245909858, 9)
    np.testing.assert_almost_equal(psf.stepK(), 0.36982048503361376, 9)
    np.testing.assert_almost_equal(psf.kValue(cen), test_flux+0j)
    np.testing.assert_almost_equal(psf.getLamOverR0(), lor)
    np.testing.assert_almost_equal(psf.getHalfLightRadius(), lor * 0.554811)
    np.testing.assert_almost_equal(psf.getFWHM(), lor * 0.975865)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.6283160485127478)

    # Check input flux vs output flux
    lors = [1, 0.5, 2, 5]
    for lor in lors:
        psf = galsim.Kolmogorov(lam_over_r0=lor, flux=test_flux)
        out_flux = psf.getFlux()
        np.testing.assert_almost_equal(out_flux, test_flux,
                                       err_msg="Flux of Kolmogorov (getFlux) is incorrect.")

        # Also check the realized flux in a drawn image
        dx = lor / 10.
        img = galsim.ImageF(256,256, scale=dx)
        psf.drawImage(image=img)
        out_flux = img.array.sum()
        np.testing.assert_almost_equal(out_flux, test_flux, 3,
                                       err_msg="Flux of Kolmogorov (image array) is incorrect.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_kolmogorov_radii():
    """Test initialization of Kolmogorov with different types of radius specification.
    """
    import time
    t1 = time.time()
    import math
    # Test constructor using lambda/r0
    lors = [1, 0.5, 2, 5]
    for lor in lors:
        print 'lor = ',lor
        test_gal = galsim.Kolmogorov(flux=1., lam_over_r0=lor)

        np.testing.assert_almost_equal(
                lor, test_gal.getLamOverR0(), decimal=9,
                err_msg="Error in Kolmogorov, lor != getLamOverR0")

        # test that getFWHM() method provides correct FWHM
        got_fwhm = test_gal.getFWHM()
        print 'got_fwhm = ',got_fwhm
        test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
        print 'fwhm ratio = ', test_fwhm_ratio
        np.testing.assert_almost_equal(
                test_fwhm_ratio, 0.5, decimal=4,
                err_msg="Error in FWHM for Kolmogorov initialized with half-light radius")

        # then test that image indeed has the correct HLR properties when radially integrated
        got_hlr = test_gal.getHalfLightRadius()
        print 'got_hlr = ',got_hlr
        hlr_sum = radial_integrate(test_gal, 0., got_hlr)
        print 'hlr_sum = ',hlr_sum
        np.testing.assert_almost_equal(
                hlr_sum, 0.5, decimal=3,
                err_msg="Error in half light radius for Kolmogorov initialized with lam_over_r0.")

    # Test constructor using half-light-radius:
    test_gal = galsim.Kolmogorov(flux=1., half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=3,
            err_msg="Error in Kolmogorov constructor with half-light radius")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    print 'got_fwhm = ',got_fwhm
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                    test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Kolmogorov initialized with half-light radius")

    # Test constructor using FWHM:
    test_gal = galsim.Kolmogorov(flux=1., fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Kolmogorov constructor with fwhm")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    print 'got_hlr = ',got_hlr
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print 'hlr_sum (profile initialized with fwhm) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=3,
            err_msg="Error in half light radius for Gaussian initialized with FWHM.")

    # Check that the getters don't work after modifying the original.
    test_gal_copy = test_gal.copy()
    print 'fwhm = ',test_gal_copy.getFWHM()
    print 'hlr = ',test_gal_copy.getHalfLightRadius()
    print 'lor = ',test_gal_copy.getLamOverR0()
    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getFWHM");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getLamOverR0");
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_kolmogorov_flux_scaling():
    """Test flux scaling for Kolmogorov.
    """
    import time
    t1 = time.time()

    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_lor0 = 1.9

    # init with lam_over_r0 and flux only (should be ok given last tests)
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_spergel():
    """Test the generation of a specific Spergel profile against a known result.
    """
    import time
    t1 = time.time()

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
        myImg = spergel.drawImage(myImg, scale=dx, method="sb",
                             use_true_center=False)

        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Spergel disagrees with expected result")

        np.testing.assert_almost_equal(
            myImg.array.sum()*dx**2, myImg.added_flux, 5,
            err_msg="Spergel profile GSObject::draw returned wrong added_flux")

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
            spergel.SBProfile.calculateFluxRadius(1.e-5)/enclosing_radius, 1.0, 4,
            err_msg="Calculated incorrect Spergel(nu={}) flux-enclosing-radius.".format(nu))
        np.testing.assert_almost_equal(
            spergel.SBProfile.calculateIntegratedFlux(1.e-5)/enclosed_flux, 1.0, 4,
            err_msg="Calculated incorrect Spergel(nu={}) enclosed flux.".format(nu))

        # Use non-unity values.
        spergel = galsim.Spergel(nu=0.37, flux=1.7, half_light_radius=2.3)

        # Check picklability
        do_pickle(spergel.SBProfile,
                  lambda x: (x.getNu(), x.getScaleRadius(), x.getFlux(), x.getGSParams()))
        do_pickle(spergel, lambda x: x.drawImage(method='no_pixel'))

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

        
def test_spergel_properties():
    """Test some basic properties of the Spergel profile.
    """
    import time
    t1 = time.time()
    spergel = galsim.Spergel(nu=0.0, flux=test_flux, scale_radius=1.0)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(spergel.centroid(), cen)
    # # Check Fourier properties
    np.testing.assert_equal(spergel.kValue(cen), (1+0j) * test_flux)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        spergel = galsim.Spergel(nu=0.0, flux=inFlux, scale_radius=1.0)
        outFlux = spergel.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_spergel_radii():
    """Test initialization of Spergel with different types of radius specification.
    """
    import time
    t1 = time.time()
    import math
    for nu, scale in zip(test_spergel_nu, test_spergel_scale) :

        test_gal = galsim.Spergel(nu=nu, half_light_radius=test_hlr, flux=1.)

        # Check that the returned half-light radius is correct
        print 'test_hlr = ',test_hlr
        print 'test_gal hlr, sr = ',test_gal.getHalfLightRadius(),test_gal.getScaleRadius()
        np.testing.assert_almost_equal(
            test_gal.getHalfLightRadius(), test_hlr, decimal=5,
            err_msg = "Error in returned HLR for Spergel HLR constructor, nu=%.1f"%nu)

        # Check that the returned flux is correct
        print 'test_gal.getFlux() = ',test_gal.getFlux()
        np.testing.assert_almost_equal(
            test_gal.getFlux(), 1., decimal=5,
            err_msg = "Error in returned Flux for Spergel HLR constructor, nu=%.1f"%nu)

        # (test half-light radii)
        print 'flux = ',test_gal.getFlux()
        print 'hlr = ',test_gal.getHalfLightRadius()
        print 'scale = ',test_gal.getScaleRadius()
        got_hlr = test_gal.getHalfLightRadius()
        got_flux = test_gal.getFlux()
        # nu = -0.85 is too difficult to numerically integrate
        if nu > -0.85:
            hlr_sum = radial_integrate(test_gal, 0., got_hlr)
            print 'hlr_sum = ',hlr_sum
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*got_flux, decimal=4,
                    err_msg = "Error in Spergel half-light radius constructor, nu=%.1f"%nu)

        # Test constructor using scale radius (test scale radius)
        test_gal = galsim.Spergel(nu=nu, scale_radius=scale, flux=1.)

        # Check that the returned scale radius is correct
        print 'test_scale = ',scale
        print 'test_gal hlr, sr = ',test_gal.getHalfLightRadius(),test_gal.getScaleRadius()
        np.testing.assert_almost_equal(
            test_gal.getScaleRadius(), scale, decimal=5,
            err_msg = "Error in returned SR for Sersic SR constructor, nu=%.1f"%nu)

        # Check that the returned flux is correct
        print 'test_gal.getFlux() = ',test_gal.getFlux()
        np.testing.assert_almost_equal(
            test_gal.getFlux(), 1., decimal=5,
            err_msg = "Error in returned Flux for Spergel HLR constructor, nu=%.1f"%nu)

        # (test half-light radius)
        got_hlr = test_gal.getHalfLightRadius()
        got_flux = test_gal.getFlux()
        # nu = -0.85 is too difficult to numerically integrate
        if nu > -0.85:
            hlr_sum = radial_integrate(test_gal, 0., got_hlr)
            print 'hlr_sum = ',hlr_sum
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*got_flux, decimal=4,
                    err_msg="Error in HLR for scale_radius constructed Spergel")

        # Check that the getters don't work after modifying the original.
        test_gal_copy = test_gal.copy()
        # They still work after copy()
        print 'nu = ',test_gal_copy.getNu()
        print 'hlr = ',test_gal_copy.getHalfLightRadius()
        print 'sr = ',test_gal_copy.getScaleRadius()
        test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
        # But not after shear() (or others, but this is a sufficient test here)
        try:
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getNu")
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getScaleRadius")
        except ImportError:
            pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_spergel_flux_scaling():
    """Test flux scaling for Spergel.
    """
    import time
    t1 = time.time()

    # decimal point to go to for parameter value comparisons
    param_decimal = 12

    # loop through spergel nu
    for test_nu in test_spergel_nu:
        # init with hlr and flux only (should be ok given last tests)
        init_obj = galsim.Spergel(test_nu, half_light_radius=test_hlr, flux=test_flux)

        # Test in place *= and /=
        obj = init_obj.copy()
        obj *= 2.
        np.testing.assert_almost_equal(
            obj.getFlux(), test_flux * 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __imul__.")
        np.testing.assert_almost_equal(
            init_obj.getFlux(), test_flux, decimal=param_decimal,
            err_msg="obj.copy() didn't produce a separate copy.")
        obj = init_obj.copy()
        obj /= 2.
        np.testing.assert_almost_equal(
            obj.getFlux(), test_flux / 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __idiv__.")

        obj2 = init_obj * 2.
        np.testing.assert_almost_equal(
            init_obj.getFlux(), test_flux, decimal=param_decimal,
            err_msg="Flux param inconsistent after __rmul__ (original).")
        np.testing.assert_almost_equal(
            obj2.getFlux(), test_flux * 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __rmul__ (result).")

        obj2 = 2. * init_obj
        np.testing.assert_almost_equal(
            init_obj.getFlux(), test_flux, decimal=param_decimal,
            err_msg="Flux param inconsistent after __mul__ (original).")
        np.testing.assert_almost_equal(
            obj2.getFlux(), test_flux * 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __mul__ (result).")

        obj2 = init_obj / 2.
        np.testing.assert_almost_equal(
             init_obj.getFlux(), test_flux, decimal=param_decimal,
             err_msg="Flux param inconsistent after __div__ (original).")
        np.testing.assert_almost_equal(
            obj2.getFlux(), test_flux / 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __div__ (result).")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_spergel_05():
    """Test the equivalence of Spergel with nu=0.5 and Exponential
    """
    import time
    t1 = time.time()

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

    do_kvalue(spergel,myImg,"nu=0.5 Spergel")

    # cf test_exponential_properties()
    spergel = galsim.Spergel(nu=0.5, flux=test_flux, half_light_radius=test_scale[0] * hlr_r0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(spergel.centroid(), cen)
    np.testing.assert_equal(spergel.kValue(cen), (1+0j) * test_flux)
    import math
    np.testing.assert_almost_equal(spergel.xValue(cen), 1./(2.*math.pi)*test_flux/test_scale[0]**2,
                                   decimal=5)

    # Also test some random values other than the center:
    expon = galsim.Exponential(flux=test_flux, scale_radius=test_scale[0])
    for (x,y) in [ (0.1,0.2), (-0.5, 0.4), (0, 0.9), (1.2, 0.1), (2,2) ]:
        pos = galsim.PositionD(x,y)
        np.testing.assert_almost_equal(spergel.xValue(pos), expon.xValue(pos), decimal=5)
        np.testing.assert_almost_equal(spergel.kValue(pos), expon.kValue(pos), decimal=5)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_gaussian()
    test_gaussian_properties()
    test_gaussian_radii()
    test_gaussian_flux_scaling()
    test_exponential()
    test_exponential_properties()
    test_exponential_radii()
    test_exponential_flux_scaling()
    test_sersic()
    test_sersic_radii()
    test_sersic_flux_scaling()
    test_sersic_05()
    test_sersic_1()
    test_airy()
    test_airy_radii()
    test_airy_flux_scaling()
    test_box()
    test_tophat()
    test_moffat()
    test_moffat_properties()
    test_moffat_radii()
    test_moffat_flux_scaling()
    test_kolmogorov()
    test_kolmogorov_properties()
    test_kolmogorov_radii()
    test_kolmogorov_flux_scaling()
    test_spergel()
    test_spergel_properties()
    test_spergel_radii()
    test_spergel_flux_scaling()
    test_spergel_05()
