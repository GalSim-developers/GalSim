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
import sys

import galsim
from galsim_test_helpers import *

path, filename = os.path.split(__file__)
imgdir = os.path.join(path, "SBProfile_comparison_images") # Directory containing the reference
                                                           # images.

@timer
def test_kolmogorov():
    """Test the generation of a specific Kolmogorov profile against a known result.
    """
    import math
    dx = 0.2
    test_flux = 1.8
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

    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    kolm2 = galsim.Kolmogorov(lam_over_r0=1.5, flux=test_flux, gsparams=gsp)
    assert kolm2 != kolm
    assert kolm2 == kolm.withGSParams(gsp)
    assert kolm2 == kolm.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

    check_basic(kolm, "Kolmogorov")

    # Test photon shooting.
    do_shoot(kolm,myImg,"Kolmogorov")

    # Test kvalues
    do_kvalue(kolm,myImg, "Kolmogorov")

    # Check picklability
    check_pickle(kolm, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(kolm)

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
    assert_raises(TypeError, galsim.Kolmogorov,
                  lam_over_r0=3, fwhm=2, half_light_radius=1, lam=3, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, fwhm=2, half_light_radius=1, lam=3, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, half_light_radius=1, lam=3, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, fwhm=2, lam=3, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, fwhm=2, half_light_radius=1)
    assert_raises(TypeError, galsim.Kolmogorov, half_light_radius=1, lam=3, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, fwhm=2, lam=3, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, fwhm=2, half_light_radius=1)
    assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, lam=3, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, half_light_radius=1)
    assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, fwhm=2)
    assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, lam=3)
    assert_raises(TypeError, galsim.Kolmogorov, lam_over_r0=3, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, fwhm=2, lam=3)
    assert_raises(TypeError, galsim.Kolmogorov, fwhm=2, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, half_light_radius=1, lam=3)
    assert_raises(TypeError, galsim.Kolmogorov, half_light_radius=1, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov, lam=3)
    assert_raises(TypeError, galsim.Kolmogorov, r0=1)
    assert_raises(TypeError, galsim.Kolmogorov)

@timer
def test_kolmogorov_properties():
    """Test some basic properties of the Kolmogorov profile.
    """
    test_flux = 1.8
    lor = 1.5
    psf = galsim.Kolmogorov(lam_over_r0=lor, flux=test_flux)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid, cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxk, 8.644067599028375, 5)
    np.testing.assert_almost_equal(psf.stepk, 0.379103528681262, 5)
    np.testing.assert_almost_equal(psf.kValue(cen), test_flux+0j)
    np.testing.assert_almost_equal(psf.lam_over_r0, lor)
    np.testing.assert_almost_equal(psf.half_light_radius, lor * 0.5548101137)
    np.testing.assert_almost_equal(psf.fwhm, lor * 0.9758634299)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.6283185307179587)
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
    test_hlr = 1.9
    test_fwhm = 1.8

    # Test constructor using lambda/r0
    lors = [1, 0.5, 2, 5]
    for lor in lors:
        print('lor = ',lor)
        test_gal = galsim.Kolmogorov(flux=1., lam_over_r0=lor)

        np.testing.assert_almost_equal(
                lor, test_gal.lam_over_r0, decimal=9,
                err_msg="Error in Kolmogorov, lor != lam_over_r0")

        # test that fwhm provides correct FWHM
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
    test_gal = galsim.Kolmogorov(flux=1., half_light_radius=test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print('hlr_sum = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=3,
            err_msg="Error in Kolmogorov constructor with half-light radius")

    # test that fwhm provides correct FWHM
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
    assert_raises(AttributeError, getattr, test_gal_shear, "fwhm")
    assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
    assert_raises(AttributeError, getattr, test_gal_shear, "lam_over_r0")


@timer
def test_kolmogorov_flux_scaling():
    """Test flux scaling for Kolmogorov.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_lor0 = 1.9
    test_flux = 17.9

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
def test_kolmogorov_folding_threshold():
    """Test Kolmogorov with low folding_threshold.
    """
    # This test reproduces a bug reported by Jim Chiang when Kolmogorov has a less than
    # default folding_threshold.  Reported in Issue #952.
    # It turned out the problem was in OneDimensionalDeviate's findExtremum going into an
    # endless loop, mostly because it didn't preserve the intended invariant that x1 < x2 < x3.
    # In this case, x1 became the largest of the three values and ended up getting larger and
    # larger indefinitely, thus resulting in an endless loop.

    # The test is really just to construct the object in finite time, but we do a few sanity
    # checks afterward for good measure.

    fwhm = 0.55344217545630736
    folding_threshold=0.0008316873626901008
    gsparams = galsim.GSParams(folding_threshold=folding_threshold)

    obj = galsim.Kolmogorov(fwhm=fwhm, gsparams=gsparams)
    print('obj = ',obj)

    assert obj.flux == 1.0
    assert obj.fwhm == fwhm
    assert obj.gsparams.folding_threshold == folding_threshold
    check_basic(obj, 'Kolmogorov with low folding_threshold')
    check_pickle(obj)


@timer
def test_kolmogorov_shoot():
    """Test Kolmogorov with photon shooting.  Particularly the flux of the final image.
    """
    rng = galsim.BaseDeviate(1234)
    obj = galsim.Kolmogorov(fwhm=1.5, flux=1.e4)
    im = galsim.Image(500,500, scale=1)
    im.setCenter(0,0)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng.duplicate())
    assert photons2 == photons, "Kolmogorov makePhot not equivalent to drawPhot"

    # Can treat the profile as a convolution of a delta function and put it in a photon_ops list.
    delta = galsim.DeltaFunction(flux=1.e4)
    psf = galsim.Kolmogorov(fwhm=1.5)
    photons3 = delta.makePhot(poisson_flux=False, rng=rng.duplicate(), photon_ops=[psf])
    assert photons3 == photons, "Using Kolmogorov in photon_ops not equivalent to drawPhot"


@timer
def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Kolmogorov.  Params include lam_over_r0, fwhm, half_light_radius, lam/r0, lam/r0_500, flux
    # gsparams.
    # The following should all test unequal:
    gals = [galsim.Kolmogorov(lam_over_r0=1.0),
            galsim.Kolmogorov(lam=1.0, r0=1.1),
            galsim.Kolmogorov(fwhm=1.0),
            galsim.Kolmogorov(half_light_radius=1.0),
            galsim.Kolmogorov(lam=1.0, r0=1.0),
            galsim.Kolmogorov(lam=1.0, r0=1.0, scale_unit=galsim.arcmin),
            galsim.Kolmogorov(lam=1.0, r0=1.0, scale_unit='degrees'),
            galsim.Kolmogorov(lam=1.0, r0_500=1.0),
            galsim.Kolmogorov(lam=1.0, r0=1.0, flux=1.1),
            galsim.Kolmogorov(lam=1.0, r0=1.0, flux=1.1, gsparams=gsp)]
    check_all_diff(gals)

@timer
def test_low_folding_threshold():
    """Test Kolmogorov with a very low folding_threshold.
    """
    # Jim Chiang identified a bug introduced in commit 56003a938963ba4bef875c2
    # where stepk was much too large for Kolmogorov profiles with very low folding_threshold.
    # This test checks that we have in fact fixed the bug.
    folding_threshold = 1e-6
    pixel_scale = 0.2

    gsparams = galsim.GSParams(folding_threshold=folding_threshold)

    fwhm = 0.7313  # This was the particular fwhm from Jim's test showing the failure.
    psf = galsim.Kolmogorov(fwhm=fwhm, gsparams=gsparams)
    image_size = psf.getGoodImageSize(pixel_scale)
    print('ft = 1.e-6: psf.getGoodImageSize:', image_size)
    # Note: Older versions gave 574 for this choice, which was actually too small due to
    # numerical rounding errors. Then it was wrongly zero for a while (only on main, not on
    # a release branch), and now I think this is correct.
    assert image_size == 6862

    # I think going forward, 1.e-6 is much too small a folding_threshold for the desired
    # effect Jim wants in imSim.  I think 1.e-4 is more appropriate to get a stamp that will
    # include most of the visible photons from a bright star.
    folding_threshold = 1e-4
    gsparams = galsim.GSParams(folding_threshold=folding_threshold)
    psf = galsim.Kolmogorov(fwhm=fwhm, gsparams=gsparams)
    image_size = psf.getGoodImageSize(pixel_scale)
    print('ft = 1.e-4: psf.getGoodImageSize:', image_size)
    assert image_size == 434


if __name__ == "__main__":
    runtests(__file__)
