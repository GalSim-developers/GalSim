# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
import numpy as np
import os
import sys

from galsim_test_helpers import *

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images. 

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
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

# These are the default GSParams used when unspecified.  We'll check that specifying 
# these explicitly produces the same results.
default_params = galsim.GSParams(
        minimum_fft_size = 128,
        maximum_fft_size = 4096,
        alias_threshold = 5.e-3,
        maxk_threshold = 1.e-3,
        kvalue_accuracy = 1.e-5,
        xvalue_accuracy = 1.e-5,
        shoot_accuracy = 1.e-5,
        realspace_relerr = 1.e-3,
        realspace_abserr = 1.e-6,
        integration_relerr = 1.e-5,
        integration_abserr = 1.e-7)


def test_gaussian():
    """Test the generation of a specific Gaussian profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_1.fits"))
    savedImg.setCenter(0,0)
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)
    tot = mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Gaussian profile disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.sum() *dx**2, tot, 5,
            err_msg="Gaussian profile SBProfile::draw returned wrong tot")

    # Repeat with the GSObject version of this:
    gauss = galsim.Gaussian(flux=1, sigma=1)
    # Reference images were made with old centering, which is equivalent to use_true_center=False.
    myImg = gauss.draw(myImg, dx=dx, normalization="surface brightness", use_true_center=False)
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
    recImg = gauss.draw(recImg, dx=dx, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            recImg[savedImg.bounds].array, savedImg.array, 5,
            err_msg="Drawing Gaussian on non-square image disagrees with expected result")
    np.testing.assert_almost_equal(
            recImg.array.sum() *dx**2, recImg.added_flux, 5,
            err_msg="Gaussian profile GSObject::draw on non-square image returned wrong added_flux")

    # Check with default_params
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=default_params)
    gauss.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian with default_params disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=galsim.GSParams())
    gauss.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(gauss,myImg,"Gaussian")

    # Test kvalues
    do_kvalue(gauss,"Gaussian")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_gaussian_properties():
    """Test some basic properties of the SBGaussian profile.
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
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
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
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
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
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
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
    # just confirm that it is true of applyShear.  I don't think that has any chance
    # of missing anything.
    test_gal_flux1 = test_gal.copy()
    print 'fwhm = ',test_gal_flux1.getFWHM()
    print 'hlr = ',test_gal_flux1.getHalfLightRadius()
    print 'sigma = ',test_gal_flux1.getSigma()
    test_gal_flux1.setFlux(3.)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux1, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux1, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux1, "getSigma")
    except ImportError:
        # assert_raises requires nose, which we don't want to force people to install.
        # So if they are running this without nose, we just skip these tests.
        pass

    test_gal_flux2 = test_gal.copy()
    print 'fwhm = ',test_gal_flux2.getFWHM()
    print 'hlr = ',test_gal_flux2.getHalfLightRadius()
    print 'sigma = ',test_gal_flux2.getSigma()
    test_gal_flux2.setFlux(3.)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux2, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux2, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux2, "getSigma")
    except ImportError:
        pass

    test_gal_shear = test_gal.copy()
    print 'fwhm = ',test_gal_shear.getFWHM()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'sigma = ',test_gal_shear.getSigma()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getSigma")
    except ImportError:
        pass

    test_gal_rot = test_gal.copy()
    print 'fwhm = ',test_gal_rot.getFWHM()
    print 'hlr = ',test_gal_rot.getHalfLightRadius()
    print 'sigma = ',test_gal_rot.getSigma()
    test_gal_rot.applyRotation(theta = 0.5 * galsim.radians)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_rot, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_rot, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_rot, "getSigma")
    except ImportError:
        pass

    test_gal_shift = test_gal.copy()
    print 'fwhm = ',test_gal_shift.getFWHM()
    print 'hlr = ',test_gal_shift.getHalfLightRadius()
    print 'sigma = ',test_gal_shift.getSigma()
    test_gal_shift.applyShift(dx=0.11, dy=0.04)
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
    """Test the generation of a specific exp profile using SBProfile against a known result. 
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
    mySBP = galsim.SBExponential(flux=1., scale_radius=r0)
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Exponential profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    expon = galsim.Exponential(flux=1., scale_radius=r0)
    expon.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential disagrees with expected result")

    # Check with default_params
    expon = galsim.Exponential(flux=1., scale_radius=r0, gsparams=default_params)
    expon.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential with default_params disagrees with expected result")
    expon = galsim.Exponential(flux=1., scale_radius=r0, gsparams=galsim.GSParams())
    expon.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(expon,myImg,"Exponential")

    # Test kvalues
    do_kvalue(expon,"Exponential")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_exponential_properties():
    """Test some basic properties of the SBExponential profile.
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
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
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
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Exponential initialized with scale_radius.")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.copy()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'scale = ',test_gal_shear.getScaleRadius()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
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
    """Test the generation of a specific Sersic profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()

    # Test SBSersic
    mySBP = galsim.SBSersic(n=3, flux=1, half_light_radius=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Sersic profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic disagrees with expected result")

    # Check with default_params
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=default_params)
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic with default_params disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic with GSParams() disagrees with expected result")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,"Sersic")

    # Test kvalues
    do_kvalue(sersic,"Sersic")


    # Now repeat everything using a truncation.  (Above had no truncation.)

    # Test Truncated SBSersic
    mySBP = galsim.SBSersic(n=3, flux=1, half_light_radius=1, trunc=10)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1_10.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Truncated Sersic profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, trunc=10)
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using truncated GSObject Sersic disagrees with expected result")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,"Truncated Sersic")

    # Test kvalues
    do_kvalue(sersic, "Truncated Sersic")

    # Check for normalization consistencies with kValue checks. xValues tested in test_sersic_radii.

    # For half-light radius specified truncated Sersic, with flux_untruncated flag set
    sersic = galsim.Sersic(n=3, flux=test_flux, half_light_radius=1, trunc=10,
                           flux_untruncated=True)
    do_kvalue(sersic, "Truncated Sersic w/ flux_untruncated, half-light radius specified")

    # For scale radius specified Sersic
    sersic = galsim.Sersic(n=3, flux=test_flux, scale_radius=0.05)
    do_kvalue(sersic, "Sersic w/ scale radius specified")

    # For scale radius specified truncated Sersic
    sersic = galsim.Sersic(n=3, flux=test_flux, scale_radius=0.05, trunc=10)
    do_kvalue(sersic, "Truncated Sersic w/ scale radius specified")

    # For scale radius specified truncated Sersic, with flux_untruncated flag set
    sersic = galsim.Sersic(n=3, flux=test_flux, scale_radius=0.05, trunc=10, flux_untruncated=True)
    do_kvalue(sersic, "Truncated Sersic w/ flux_untruncated, scale radius specified")

    # Test severely truncated Sersic
    sersic = galsim.Sersic(n=4, flux=test_flux, half_light_radius=1, trunc=1.45)
    do_kvalue(sersic, "Severely truncated n=4 Sersic")

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
            hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
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
            hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
            print 'hlr_sum = ',hlr_sum
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*got_flux, decimal=4,
                    err_msg="Error in HLR for scale_radius constructed %s"%label)

        # Check that the getters don't work after modifying the original.
        test_gal_shear = test_gal1.copy()
        # They still work after copy()
        if n != -4:
            print 'n = ',test_gal_shear.getN()
        print 'hlr = ',test_gal_shear.getHalfLightRadius()
        print 'sr = ',test_gal_shear.getScaleRadius()
        test_gal_shear.applyShear(g1=0.3, g2=0.1)
        # But not after applyShear() (or others, but this is a sufficient test here)
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
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    sersic = galsim.Sersic(n=0.5, flux=1, half_light_radius=1 * hlr_sigma)
    myImg = sersic.draw(myImg, normalization="surface brightness", use_true_center=False)
    print 'saved image center = ',savedImg(0,0)
    print 'image center = ',myImg(0,0)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using Sersic with n=0.5 disagrees with expected result for Gaussian")

    do_kvalue(sersic,"n=0.5 Sersic")

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
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    sersic = galsim.Sersic(n=1, flux=1., half_light_radius=r0 * hlr_r0)
    sersic.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using Sersic n=1 disagrees with expected result for Exponential")

    do_kvalue(sersic,"n=1 Sersic")

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
    """Test the generation of a specific Airy profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBAiry(lam_over_diam=1./0.8, obscuration=0.1, flux=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "airy_.8_.1.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Airy profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1)
    airy.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy disagrees with expected result")

    # Check with default_params
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1, gsparams=default_params)
    airy.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy with default_params disagrees with expected result")
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1, gsparams=galsim.GSParams())
    airy.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy with GSParams() disagrees with expected result")

    # Test photon shooting.
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.0, flux=test_flux)
    do_shoot(airy,myImg,"Airy obscuration=0.0")
    airy2 = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=test_flux)
    do_shoot(airy2,myImg,"Airy obscuration=0.1")

    # Test kvalues
    do_kvalue(airy, "Airy obscuration=0.0")
    do_kvalue(airy2, "Airy obscuration=0.1")

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
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
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
    test_gal_shear = test_gal.copy()
    print 'fwhm = ',test_gal_shear.getFWHM()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'lod = ',test_gal_shear.getLamOverD()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
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
    """Test the generation of a specific box profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBBox(xw=1, yw=1, flux=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Box profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    pixel = galsim.Pixel(xw=1, yw=1, flux=1)
    pixel.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel disagrees with expected result")

    # Check with default_params
    pixel = galsim.Pixel(xw=1, yw=1, flux=1, gsparams=default_params)
    pixel.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel with default_params disagrees with expected result")
    pixel = galsim.Pixel(xw=1, yw=1, flux=1, gsparams=galsim.GSParams())
    pixel.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(pixel,myImg,"Pixel")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_moffat():
    """Test the generation of a specific Moffat profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    # Code was formerly:
    # mySBP = galsim.SBMoffat(beta=2, truncationFWHM=5, flux=1, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in 
    # physical units.  However, the same profile can be constructed using 
    # fwhm=1.3178976627539716
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    fwhm_backwards_compatible = 1.3178976627539716
    mySBP = galsim.SBMoffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_2_5.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Moffat profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    moffat = galsim.Moffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1)
    moffat.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat disagrees with expected result")

    # Check with default_params
    moffat = galsim.Moffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1, 
                           gsparams=default_params)
    moffat.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat with default_params disagrees with expected result")
    moffat = galsim.Moffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1, 
                           gsparams=galsim.GSParams())
    moffat.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(moffat,myImg,"Moffat")

    # Test kvalues
    do_kvalue(moffat, "Moffat")

    # The code for untruncated Moffat profiles is specialized for particular beta values, so 
    # test each of these:
    for beta in [ 1.5, 2, 2.5, 3, 3.5, 4, 2.3 ]:  # The one last is for the generic case.
        moffat = galsim.Moffat(beta=beta, half_light_radius=0.7, flux=test_flux)
        do_kvalue(moffat,"Untruncated Moffat with beta=%f"%beta)
        # Don't bother repeating the do_shoot tests, since they are rather slow, and the code
        # isn't different for the different beta values.

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_moffat_properties():
    """Test some basic properties of the SBMoffat profile.
    """
    import time
    t1 = time.time()
    # Code was formerly:
    # mySBP = galsim.Moffat(beta=2.0, truncationFWHM=2, flux=test_flux, half_light_radius=1)
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
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
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
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
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
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
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
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
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
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
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
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
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
    test_gal_shear = test_gal.copy()
    print 'beta = ',test_gal_shear.getBeta()
    print 'fwhm = ',test_gal_shear.getFWHM()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'scale = ',test_gal_shear.getScaleRadius()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
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
    """Test the generation of a specific Kolmogorov profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBKolmogorov(lam_over_r0=1.5, flux=test_flux)
    # This savedImg was created from the SBKolmogorov implementation in
    # commit c8efd74d1930157b1b1ffc0bfcfb5e1bf6fe3201
    # It would be nice to get an independent calculation here...
    #savedImg = galsim.ImageF(128,128)
    #mySBP.draw(image=savedImg, dx=0.2)
    #savedImg.write(os.path.join(imgdir, "kolmogorov.fits"))
    savedImg = galsim.fits.read(os.path.join(imgdir, "kolmogorov.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Kolmogorov profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=test_flux)
    kolm.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Kolmogorov disagrees with expected result")

    # Check with default_params
    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=test_flux, gsparams=default_params)
    kolm.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Kolmogorov with default_params disagrees with expected result")
    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=test_flux, gsparams=galsim.GSParams())
    kolm.draw(myImg, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Kolmogorov with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(kolm,myImg,"Kolmogorov")

    # Test kvalues
    do_kvalue(kolm, "Kolmogorov")

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
        pix = galsim.Pixel(dx)
        conv = galsim.Convolve([psf,pix])
        conv.draw(image=img)
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
        hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
        print 'hlr_sum = ',hlr_sum
        np.testing.assert_almost_equal(
                hlr_sum, 0.5, decimal=3,
                err_msg="Error in half light radius for Kolmogorov initialized with lam_over_r0.")

    # Test constructor using half-light-radius:
    test_gal = galsim.Kolmogorov(flux=1., half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
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
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with fwhm) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=3,
            err_msg="Error in half light radius for Gaussian initialized with FWHM.")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.copy()
    print 'fwhm = ',test_gal_shear.getFWHM()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'lor = ',test_gal_shear.getLamOverR0()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
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
    test_moffat()
    test_moffat_properties()
    test_moffat_radii()
    test_moffat_flux_scaling()
    test_kolmogorov()
    test_kolmogorov_properties()
    test_kolmogorov_radii()
    test_kolmogorov_flux_scaling()

