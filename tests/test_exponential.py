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
def test_exponential():
    """Test the generation of a specific exp profile against a known result.
    """
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
    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    expon2 = galsim.Exponential(flux=1.7, scale_radius=0.91, gsparams=gsp)
    assert expon2 != expon
    assert expon2 == expon.withGSParams(gsp)
    assert expon2 == expon.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

    check_basic(expon, "Exponential")

    # Test photon shooting.
    do_shoot(expon,myImg,"Exponential")

    # Test kvalues
    do_kvalue(expon,myImg,"Exponential")

    # Check picklability
    check_pickle(expon, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(expon)

    # Should raise an exception if both scale_radius and half_light_radius are provided.
    assert_raises(TypeError, galsim.Exponential, scale_radius=3, half_light_radius=1)
    # Or neither.
    assert_raises(TypeError, galsim.Exponential)


@timer
def test_exponential_properties():
    """Test some basic properties of the Exponential profile.
    """
    test_flux = 17.9
    test_scale = 1.8
    expon = galsim.Exponential(flux=test_flux, scale_radius=test_scale)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(expon.centroid, cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(expon.maxk, 10 / test_scale)
    np.testing.assert_almost_equal(expon.stepk, 0.37436747851 / test_scale)
    np.testing.assert_almost_equal(expon.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(expon.flux, test_flux)
    import math
    np.testing.assert_almost_equal(expon.xValue(cen), 1./(2.*math.pi)*test_flux/test_scale**2)
    np.testing.assert_almost_equal(expon.xValue(cen), expon.max_sb)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        expon = galsim.Exponential(flux=inFlux, scale_radius=1.8)
        outFlux = expon.flux
        np.testing.assert_almost_equal(outFlux, inFlux)


@timer
def test_exponential_radii():
    """Test initialization of Exponential with different types of radius specification.
    """
    test_hlr = 1.8
    test_scale = 1.8

    import math
    # Test constructor using half-light-radius:
    test_gal = galsim.Exponential(flux=1., half_light_radius=test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print('hlr_sum = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Exponential constructor with half-light radius")

    # then test scale getter
    center = test_gal.xValue(galsim.PositionD(0,0))
    got_sr = test_gal.scale_radius
    ratio = test_gal.xValue(galsim.PositionD(got_sr,0)) / center
    print('scale ratio = ',ratio)
    np.testing.assert_almost_equal(
            ratio, np.exp(-1.0), decimal=4,
            err_msg="Error in scale_radius for Exponential constructed with half light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Exponential(flux=1., scale_radius=test_scale)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale,0)) / center
    print('scale ratio = ',ratio)
    np.testing.assert_almost_equal(
            ratio, np.exp(-1.0), decimal=4,
            err_msg="Error in Exponential constructor with scale")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.half_light_radius
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print('hlr_sum (profile initialized with scale_radius) = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Exponential initialized with scale_radius.")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
    assert_raises(AttributeError, getattr, test_gal_shear, "scale_radius")


@timer
def test_exponential_flux_scaling():
    """Test flux scaling for Exponential.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_flux = 17.9
    test_scale = 1.8

    # init with scale and flux only (should be ok given last tests)
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
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
def test_exponential_shoot():
    """Test Exponential with photon shooting.  Particularly the flux of the final image.
    """
    rng = galsim.BaseDeviate(1234)
    obj = galsim.Exponential(half_light_radius=3.5, flux=1.e4)
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
    assert photons2 == photons, "Exponential makePhot not equivalent to drawPhot"


@timer
def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Exponential.  Params include half_light_radius, scale_radius, flux, gsparams
    # The following should all test unequal:
    gals = [galsim.Exponential(half_light_radius=1.0),
            galsim.Exponential(half_light_radius=1.1),
            galsim.Exponential(scale_radius=1.0),
            galsim.Exponential(half_light_radius=1.0, flux=1.1),
            galsim.Exponential(half_light_radius=1.0, gsparams=gsp)]
    check_all_diff(gals)


if __name__ == "__main__":
    runtests(__file__)
