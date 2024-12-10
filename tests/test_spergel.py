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
def test_spergel():
    """Test the generation of a specific Spergel profile against a known result.
    """
    test_spergel_nu = [-0.85, -0.5, 0.0, 0.85, 4.0]
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

        gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
        spergel2 = galsim.Spergel(nu=nu, half_light_radius=1.0, gsparams=gsp)
        assert spergel2 != spergel
        assert spergel2 == spergel.withGSParams(gsp)
        assert spergel2 == spergel.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

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
    check_pickle(spergel, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(spergel)
    check_pickle(galsim.Spergel(0,1))

    # Should raise an exception if both scale_radius and half_light_radius are provided.
    assert_raises(TypeError, galsim.Spergel, nu=0, scale_radius=3, half_light_radius=1)
    assert_raises(TypeError, galsim.Spergel, nu=0)
    assert_raises(TypeError, galsim.Spergel, scale_radius=3)

    # Allowed range = [-0.85, 4.0]
    assert_raises(ValueError, galsim.Spergel, nu=-0.9)
    assert_raises(ValueError, galsim.Spergel, nu=4.1)


@timer
def test_spergel_properties():
    """Test some basic properties of the Spergel profile.
    """
    test_flux = 17.9
    spergel = galsim.Spergel(nu=0.0, flux=test_flux, scale_radius=1.0)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(spergel.centroid, cen)
    # # Check Fourier properties
    np.testing.assert_almost_equal(spergel.kValue(cen), (1+0j) * test_flux)
    maxk = spergel.maxk
    assert spergel.kValue(maxk,0).real/test_flux <= galsim.GSParams().maxk_threshold
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
    test_spergel_nu = [-0.85, -0.5, 0.0, 0.85, 4.0]
    test_spergel_scale = [20.0, 1.0, 1.0, 0.5, 0.5]
    test_hlr = 1.8

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
        assert_raises(AttributeError, getattr, test_gal_shear, "nu")
        assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
        assert_raises(AttributeError, getattr, test_gal_shear, "scale_radius")


@timer
def test_spergel_flux_scaling():
    """Test flux scaling for Spergel.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_flux = 17.9
    test_spergel_nu = [-0.85, -0.5, 0.0, 0.85, 4.0]
    test_hlr = 1.8

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
    test_flux = 17.9
    test_scale = 1.8
    spergel = galsim.Spergel(nu=0.5, flux=test_flux, half_light_radius=test_scale * hlr_r0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(spergel.centroid, cen)
    np.testing.assert_almost_equal(spergel.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(spergel.flux, test_flux)
    import math
    np.testing.assert_almost_equal(spergel.xValue(cen), 1./(2.*math.pi)*test_flux/test_scale**2,
                                   decimal=5)
    np.testing.assert_almost_equal(spergel.xValue(cen), spergel.max_sb)

    # Also test some random values other than the center:
    expon = galsim.Exponential(flux=test_flux, scale_radius=test_scale)
    for (x,y) in [ (0.1,0.2), (-0.5, 0.4), (0, 0.9), (1.2, 0.1), (2,2) ]:
        pos = galsim.PositionD(x,y)
        np.testing.assert_almost_equal(spergel.xValue(pos), expon.xValue(pos), decimal=5)
        np.testing.assert_almost_equal(spergel.kValue(pos), expon.kValue(pos), decimal=5)


@timer
def test_spergel_shoot():
    """Test Spergel with photon shooting.  Particularly the flux of the final image.
    """
    rng = galsim.BaseDeviate(1234)
    obj = galsim.Spergel(nu=0, half_light_radius=3.5, flux=1.e4)
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
    assert photons2 == photons, "Spergel makePhot not equivalent to drawPhot"

    obj = galsim.Spergel(nu=3.2, half_light_radius=3.5, flux=1.e4)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng)
    assert photons2 == photons, "Spergel makePhot not equivalent to drawPhot"

    obj = galsim.Spergel(nu=-0.6, half_light_radius=3.5, flux=1.e4)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng)
    assert photons2 == photons, "Spergel makePhot not equivalent to drawPhot"


@timer
def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Spergel.  Params include nu, half_light_radius, scale_radius, flux, and gsparams.
    # The following should all test unequal:
    gals = [galsim.Spergel(nu=0.0, half_light_radius=1.0),
            galsim.Spergel(nu=0.1, half_light_radius=1.0),
            galsim.Spergel(nu=0.0, half_light_radius=1.1),
            galsim.Spergel(nu=0.0, scale_radius=1.0),
            galsim.Spergel(nu=0.0, half_light_radius=1.0, flux=1.1),
            galsim.Spergel(nu=0.0, half_light_radius=1.0, gsparams=gsp)]
    check_all_diff(gals)


if __name__ == "__main__":
    runtests(__file__)
