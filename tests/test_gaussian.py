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

path, filename = os.path.split(__file__)
imgdir = os.path.join(path, "SBProfile_comparison_images") # Directory containing the reference
                                                           # images.


@timer
def test_gaussian():
    """Test the generation of a specific Gaussian profile against a known result.
    """
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
            myImg.array.sum(dtype=float) *dx**2, myImg.added_flux, 5,
            err_msg="Gaussian profile GSObject::draw returned wrong added_flux")

    # Check a non-square image
    print(myImg.bounds)
    recImg = galsim.ImageF(45,66)
    recImg.setCenter(0,0)
    recImg = gauss.drawImage(recImg, scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            recImg[savedImg.bounds].array, savedImg.array, 5,
            err_msg="Drawing Gaussian on non-square image disagrees with expected result")
    np.testing.assert_almost_equal(
            recImg.array.sum(dtype=float) *dx**2, recImg.added_flux, 5,
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
    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    gauss2 = galsim.Gaussian(flux=1.7, sigma=2.3, gsparams=gsp)
    assert gauss2 != gauss
    assert gauss2 == gauss.withGSParams(gsp)
    assert gauss2 == gauss.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    assert gauss2 == gauss.withGSParams(xvalue_accuracy=1.e-8).withGSParams(kvalue_accuracy=1.e-8)
    assert gauss2 == gauss.withGSParams(galsim.GSParams(xvalue_accuracy=1.e-8),
                                        kvalue_accuracy=1.e-8)
    assert gauss2 == gauss.withGSParams(gsp).withGSParams(kvalue_accuracy=1.e-8)
    assert gauss2 == gauss.withGSParams(galsim.GSParams(xvalue_accuracy=1.e-8)).withGSParams(
                                            kvalue_accuracy=1.e-8)
    check_basic(gauss, "Gaussian")

    # Check invalid parameters
    assert_raises(TypeError, gauss.withGSParams, xvalue_threshold=1.e-8)
    assert_raises(TypeError, gauss.withGSParams, xvalue_accuracy=1.e-8, kvalue=1.e-8)

    # Test photon shooting.
    do_shoot(gauss,myImg,"Gaussian")

    # Test kvalues
    do_kvalue(gauss,myImg,"Gaussian")

    # Check picklability
    check_pickle(galsim.GSParams())  # Check GSParams explicitly here too.
    check_pickle(galsim.GSParams(
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
    check_pickle(gauss, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(gauss)

    # Should raise an exception if >=2 radii are provided.
    assert_raises(TypeError, galsim.Gaussian, sigma=3, half_light_radius=1, fwhm=2)
    assert_raises(TypeError, galsim.Gaussian, half_light_radius=1, fwhm=2)
    assert_raises(TypeError, galsim.Gaussian, sigma=3, fwhm=2)
    assert_raises(TypeError, galsim.Gaussian, sigma=3, half_light_radius=1)
    # Or none.
    assert_raises(TypeError, galsim.Gaussian)

    # Finally, test the noise property for things that don't have any noise set.
    assert gauss.noise is None
    # And accessing the attribute from the class should indicate that it is a lazyproperty
    assert 'lazy_property' in str(galsim.GSObject._noise)

    # And check that trying to use GSObject directly is an error.
    assert_raises(NotImplementedError, galsim.GSObject)


@timer
def test_gaussian_properties():
    """Test some basic properties of the Gaussian profile.
    """
    test_flux = 17.9
    test_sigma = 1.8
    gauss = galsim.Gaussian(flux=test_flux, sigma=test_sigma)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(gauss.centroid, cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(gauss.maxk, 3.7169221888498383 / test_sigma)
    np.testing.assert_almost_equal(gauss.stepk, 0.533644625664 / test_sigma)
    np.testing.assert_almost_equal(gauss.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(gauss.flux, test_flux)
    import math
    np.testing.assert_almost_equal(gauss.xValue(cen), 1./(2.*math.pi) * test_flux / test_sigma**2)
    np.testing.assert_almost_equal(gauss.xValue(cen), gauss.max_sb)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        gauss = galsim.Gaussian(flux=inFlux, sigma=2.)
        outFlux = gauss.flux
        np.testing.assert_almost_equal(outFlux, inFlux)

    # Check some valid and invalid ways to pass arguments to xValue
    # Same code applies to kValue and others, so just do this one.
    assert gauss.xValue(cen.x, cen.y) == gauss.xValue(cen)
    assert gauss.xValue(x=cen.x, y=cen.y) == gauss.xValue(cen)
    assert gauss.xValue( (cen.x, cen.y) ) == gauss.xValue(cen)
    assert_raises(TypeError, gauss.xValue, cen.x)
    assert_raises(TypeError, gauss.xValue, x=cen.x)
    assert_raises(TypeError, gauss.xValue, cen.x, y=cen.y)
    assert_raises(TypeError, gauss.xValue, dx=cen.x, dy=cen.y)
    assert_raises(TypeError, gauss.xValue, dx=cen.x, y=cen.y)
    assert_raises(TypeError, gauss.xValue, x=cen.x, dy=cen.y)
    assert_raises(TypeError, gauss.xValue, cen.x, cen.y, cen.y)
    assert_raises(TypeError, gauss.xValue, cen.x, cen.y, invalid=True)
    assert_raises(TypeError, gauss.xValue, pos=cen)



@timer
def test_gaussian_radii():
    """Test initialization of Gaussian with different types of radius specification.
    """
    import math
    test_hlr = 1.7
    test_fwhm = 1.8
    test_sigma = 1.9

    # Test constructor using half-light-radius:
    test_gal = galsim.Gaussian(flux=1., half_light_radius=test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print('hlr_sum = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Gaussian constructor with half-light radius")

    # test that fwhm provides correct FWHM
    got_fwhm = test_gal.fwhm
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print('fwhm ratio = ', test_fwhm_ratio)
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Gaussian initialized with half-light radius")

    # test that sigma provides correct sigma
    got_sigma = test_gal.sigma
    test_sigma_ratio = (test_gal.xValue(galsim.PositionD(got_sigma, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print('sigma ratio = ', test_sigma_ratio)
    np.testing.assert_almost_equal(
            test_sigma_ratio, math.exp(-0.5), decimal=4,
            err_msg="Error in sigma for Gaussian initialized with half-light radius")

    # Test constructor using sigma:
    test_gal = galsim.Gaussian(flux=1., sigma=test_sigma)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_sigma,0)) / center
    print('sigma ratio = ',ratio)
    np.testing.assert_almost_equal(
            ratio, np.exp(-0.5), decimal=4,
            err_msg="Error in Gaussian constructor with sigma")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.half_light_radius
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print('hlr_sum (profile initialized with sigma) = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Gaussian initialized with sigma.")

    # test that fwhm provides correct FWHM
    got_fwhm = test_gal.fwhm
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print('fwhm ratio = ', test_fwhm_ratio)
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Gaussian initialized with sigma.")

    # Test constructor using FWHM:
    test_gal = galsim.Gaussian(flux=1., fwhm=test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print('fwhm ratio = ',ratio)
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Gaussian constructor with fwhm")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.half_light_radius
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print('hlr_sum (profile initialized with fwhm) = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Gaussian initialized with FWHM.")

    # test that sigma provides correct sigma
    got_sigma = test_gal.sigma
    test_sigma_ratio = (test_gal.xValue(galsim.PositionD(got_sigma, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print('sigma ratio = ', test_sigma_ratio)
    np.testing.assert_almost_equal(
            test_sigma_ratio, math.exp(-0.5), decimal=4,
            err_msg="Error in sigma for Gaussian initialized with FWHM.")

    # Check that the properties don't work after modifying the original.
    # Note: I test all the modifiers here.  For the rest of the profile types, I'll
    # just confirm that it is true of shear.  I don't think that has any chance
    # of missing anything.
    test_gal_flux1 = test_gal * 3.
    assert_raises(AttributeError, getattr, test_gal_flux1, "fwhm")
    assert_raises(AttributeError, getattr, test_gal_flux1, "half_light_radius")
    assert_raises(AttributeError, getattr, test_gal_flux1, "sigma")

    test_gal_flux2 = test_gal.withScaledFlux(3.)
    assert_raises(AttributeError, getattr, test_gal_flux2, "fwhm")
    assert_raises(AttributeError, getattr, test_gal_flux2, "half_light_radius")
    assert_raises(AttributeError, getattr, test_gal_flux2, "sigma")

    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    assert_raises(AttributeError, getattr, test_gal_shear, "fwhm")
    assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
    assert_raises(AttributeError, getattr, test_gal_shear, "sigma")

    test_gal_rot = test_gal.rotate(theta = 0.5 * galsim.radians)
    assert_raises(AttributeError, getattr, test_gal_rot, "fwhm")
    assert_raises(AttributeError, getattr, test_gal_rot, "half_light_radius")
    assert_raises(AttributeError, getattr, test_gal_rot, "sigma")

    test_gal_shift = test_gal.shift(dx=0.11, dy=0.04)
    assert_raises(AttributeError, getattr, test_gal_shift, "fwhm")
    assert_raises(AttributeError, getattr, test_gal_shift, "half_light_radius")
    assert_raises(AttributeError, getattr, test_gal_shift, "sigma")


@timer
def test_gaussian_flux_scaling():
    """Test flux scaling for Gaussian.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_flux = 17.9
    test_sigma = 1.8

    # init with sigma and flux only (should be ok given last tests)
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
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
def test_gaussian_shoot():
    """Test Gaussian with photon shooting.  Particularly the flux of the final image.
    """
    rng = galsim.BaseDeviate(1234)
    obj = galsim.Gaussian(fwhm=3.5, flux=1.e4)
    im = galsim.Image(100,100, scale=1)
    im.setCenter(0,0)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng.duplicate())
    assert photons2 == photons, "Gaussian makePhot not equivalent to drawPhot"

    # Can treat the profile as a convolution of a delta function and put it in a photon_ops list.
    delta = galsim.DeltaFunction(flux=1.e4)
    psf = galsim.Gaussian(fwhm=3.5)
    photons3 = delta.makePhot(poisson_flux=False, rng=rng.duplicate(), photon_ops=[psf])
    assert photons3 == photons, "Using Gaussian in photon_ops not equivalent to drawPhot"


@timer
def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Gaussian.  Params include sigma, fwhm, half_light_radius, flux, and gsparams.
    # The following should all test unequal:
    gals = [galsim.Gaussian(sigma=1.0),
            galsim.Gaussian(sigma=1.1),
            galsim.Gaussian(fwhm=1.0),
            galsim.Gaussian(half_light_radius=1.0),
            galsim.Gaussian(half_light_radius=1.1),
            galsim.Gaussian(sigma=1.2, flux=1.0),
            galsim.Gaussian(sigma=1.2, flux=1.1),
            galsim.Gaussian(sigma=1.2, gsparams=gsp)]
    # Check that setifying doesn't remove any duplicate items.
    check_all_diff(gals)


@timer
def test_accurate_shift():
    """Test that shifted Gaussian looks the same with real and fourier drawing.
    """
    # This is in response to issue #1231

    gal = galsim.Gaussian(sigma=1.).shift([5,5]).withFlux(200)

    real_im = gal.drawImage(nx=128, ny=128, scale=0.2, method='real_space')
    fft_im = gal.drawImage(nx=128, ny=128, scale=0.2, method='fft')
    print('max abs diff = ',np.max(np.abs(real_im.array - fft_im.array)))
    np.testing.assert_allclose(fft_im.array, real_im.array, rtol=1.e-7, atol=1.e-7)

    # In double precision it's almost perfect.
    real_im = gal.drawImage(nx=128, ny=128, scale=0.2, method='real_space', dtype=float)
    fft_im = gal.drawImage(nx=128, ny=128, scale=0.2, method='fft', dtype=float)
    print('max abs diff = ',np.max(np.abs(real_im.array - fft_im.array)))
    np.testing.assert_allclose(fft_im.array, real_im.array, rtol=1.e-14, atol=1.e-14)


if __name__ == "__main__":
    runtests(__file__)
