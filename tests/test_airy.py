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
def test_airy():
    """Test the generation of a specific Airy profile against a known result.
    """
    import math
    savedImg = galsim.fits.read(os.path.join(imgdir, "airy_.8_.1.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)
    test_flux = 17.9

    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1)
    airy.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy disagrees with expected result")
    np.testing.assert_array_almost_equal(
            airy.lam_over_diam, 1./0.8, 5,
            err_msg="Airy lam_over_diam returned wrong value")
    np.testing.assert_array_equal(
            airy.obscuration, 0.1,
            err_msg="Airy obscuration returned wrong value")
    check_pickle(airy)

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

    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    airy2 = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1, gsparams=gsp)
    assert airy2 != airy
    assert airy2 == airy.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

    # Check some properties
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=test_flux)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(airy.centroid, cen)
    np.testing.assert_almost_equal(airy.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(airy.flux, test_flux)
    np.testing.assert_almost_equal(airy.xValue(cen), airy.max_sb)

    check_basic(airy, "Airy obscuration=0.1")

    # Check with obscuration == 0
    airy0 = galsim.Airy(lam_over_diam=1./0.7, flux=test_flux)
    np.testing.assert_equal(airy0.centroid, cen)
    np.testing.assert_almost_equal(airy0.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(airy0.flux, test_flux)
    np.testing.assert_almost_equal(airy0.xValue(cen), airy0.max_sb)
    np.testing.assert_array_almost_equal(
            airy0.lam_over_diam, 1./0.7, 5,
            err_msg="Airy lam_over_diam returned wrong value")
    np.testing.assert_array_equal(
            airy0.obscuration, 0.0,
            err_msg="Airy obscuration returned wrong value")
    check_basic(airy0, "Airy obscuration=0.0")

    # Test photon shooting.
    do_shoot(airy0,myImg,"Airy obscuration=0.0")
    do_shoot(airy,myImg,"Airy obscuration=0.1")

    # Test kvalues
    do_kvalue(airy0,myImg, "Airy obscuration=0.0")
    do_kvalue(airy,myImg, "Airy obscuration=0.1")

    # Check picklability
    check_pickle(airy0, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(airy0)
    check_pickle(airy, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(airy)

    # Test initialization separately with lam and diam, in various units.  Since the above profiles
    # have lam/diam = 1./0.8 in arbitrary units, we will tell it that lam=1.e9 nm and diam=0.8 m,
    # and use `scale_unit` of galsim.radians.  This is rather silly, but it should work.
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1.7)
    airy2 = galsim.Airy(lam=1.e9, diam=0.8, scale_unit=galsim.radians, obscuration=0.1, flux=1.7)
    gsobject_compare(airy,airy2)
    # For lam/diam = 1.25 arcsec, and diam = 0.3 m, lam = (1.25/3600/180*pi) * 0.3 * 1.e9
    lam = 1.25 * 0.3 / 3600. / 180. * math.pi * 1.e9
    print('lam = ',lam)
    airy3 = galsim.Airy(lam=lam, diam=0.3, scale_unit='arcsec', obscuration=0.1, flux=1.7)
    gsobject_compare(airy,airy3)
    # arcsec is the default scale_unit, so can leave this off.
    airy4 = galsim.Airy(lam=lam, diam=0.3, obscuration=0.1, flux=1.7)
    gsobject_compare(airy,airy4)
    check_pickle(airy)
    check_pickle(airy2)
    check_pickle(airy3)
    check_pickle(airy4)

    # Should raise an exception if both lam, lam_over_diam are provided
    assert_raises(TypeError, galsim.Airy, lam_over_diam=3, lam=3, diam=1)
    assert_raises(TypeError, galsim.Airy, lam_over_diam=3, lam=3)
    assert_raises(TypeError, galsim.Airy, lam_over_diam=3, diam=1)
    assert_raises(TypeError, galsim.Airy, lam=3)
    assert_raises(TypeError, galsim.Airy, diam=1)


@timer
def test_airy_radii():
    """Test Airy half light radius and FWHM correctly set and match image.
    """
    import math
    # Test constructor using lam_over_diam: (only option for Airy)
    test_gal = galsim.Airy(lam_over_diam= 1./0.8, flux=1.)
    # test half-light-radius getter
    got_hlr = test_gal.half_light_radius
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print('hlr_sum = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Airy half-light radius")

    # test FWHM getter
    center = test_gal.xValue(galsim.PositionD(0,0))
    got_fwhm = test_gal.fwhm
    ratio = test_gal.xValue(galsim.PositionD(0.5 * got_fwhm,0)) / center
    print('fwhm ratio = ',ratio)
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in fwhm for Airy.")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    assert_raises(AttributeError, getattr, test_gal_shear, "fwhm")
    assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
    assert_raises(AttributeError, getattr, test_gal_shear, "lam_over_diam")

    # hlr and fwhm not implemented for obscuration != 0
    airy2 = galsim.Airy(lam_over_diam= 1./0.8, flux=1., obscuration=0.2)
    with assert_raises(galsim.GalSimNotImplementedError):
        airy2.half_light_radius
    with assert_raises(galsim.GalSimNotImplementedError):
        airy2.fwhm


@timer
def test_airy_flux_scaling():
    """Test flux scaling for Airy.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_loD = 1.9
    test_obscuration = 0.32
    test_flux = 17.9

    # init with lam_over_diam and flux only (should be ok given last tests)
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
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
def test_airy_shoot():
    """Test Airy with photon shooting.  Particularly the flux of the final image.
    """
    # Airy patterns have *very* extended wings, so make this really small and the image really
    # big to make sure we capture all the photons.
    rng = galsim.BaseDeviate(1234)
    obj = galsim.Airy(lam_over_diam=0.01, flux=1.e4)
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
    assert photons2 == photons, "Airy makePhot not equivalent to drawPhot"

    obj = galsim.Airy(lam_over_diam=0.01, flux=1.e4, obscuration=0.4)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng.duplicate())
    assert photons2 == photons, "Airy makePhot not equivalent to drawPhot"

    # Can treat the profile as a convolution of a delta function and put it in a photon_ops list.
    delta = galsim.DeltaFunction(flux=1.e4)
    psf = galsim.Airy(lam_over_diam=0.01, obscuration=0.4)
    photons3 = delta.makePhot(poisson_flux=False, rng=rng.duplicate(), photon_ops=[psf])
    assert photons3 == photons, "Using Airy in photon_ops not equivalent to drawPhot"


@timer
def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Airy.  Params include lam_over_diam, lam, diam, obscuration, flux, and gsparams.
    # The following should all test unequal:
    gals = [galsim.Airy(lam_over_diam=1.0),
            galsim.Airy(lam_over_diam=1.1),
            galsim.Airy(lam=1.0, diam=1.2),
            galsim.Airy(lam=1.0, diam=1.2, scale_unit=galsim.arcmin),
            galsim.Airy(lam=1.0, diam=1.2, scale_unit='degrees'),
            galsim.Airy(lam=1.0, diam=1.0, obscuration=0.1),
            galsim.Airy(lam_over_diam=1.0, flux=1.1),
            galsim.Airy(lam_over_diam=1.0, gsparams=gsp)]
    check_all_diff(gals)


if __name__ == "__main__":
    runtests(__file__)
