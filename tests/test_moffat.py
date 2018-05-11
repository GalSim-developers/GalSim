# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

import galsim
from galsim_test_helpers import *

path, filename = os.path.split(__file__)
imgdir = os.path.join(path, "SBProfile_comparison_images") # Directory containing the reference
                                                           # images.

@timer
def test_moffat():
    """Test the generation of a specific Moffat profile against a known result.
    """
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
    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    moffat2 = galsim.Moffat(beta=3.7, flux=1.7, half_light_radius=2.3, trunc=8.2, gsparams=gsp)
    assert moffat2 != moffat
    assert moffat2 == moffat.withGSParams(gsp)

    check_basic(moffat, "Moffat")

    # Test photon shooting.
    do_shoot(moffat,myImg,"Moffat")

    # Test kvalues
    do_kvalue(moffat,myImg, "Moffat")

    # Check picklability
    do_pickle(moffat, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(moffat)

    # The code for untruncated Moffat profiles is specialized for particular beta values, so
    # test each of these:
    test_flux = 17.9
    for beta in [ 1.5, 2, 2.5, 3, 3.5, 4, 2.3 ]:  # The one last is for the generic case.
        moffat = galsim.Moffat(beta=beta, half_light_radius=0.7, flux=test_flux)
        check_basic(moffat, "Untruncated Moffat with beta=%f"%beta)
        do_kvalue(moffat,myImg,"Untruncated Moffat with beta=%f"%beta)
        # Don't bother repeating the do_shoot tests, since they are rather slow, and the code
        # isn't different for the different beta values.
        cen = galsim.PositionD(0, 0)
        np.testing.assert_equal(moffat.centroid, cen)
        np.testing.assert_almost_equal(moffat.kValue(cen), (1+0j) * test_flux)
        np.testing.assert_almost_equal(moffat.flux, test_flux)
        np.testing.assert_almost_equal(moffat.xValue(cen), moffat.max_sb)

    # Should raise an exception if >=2 radii are provided.
    assert_raises(TypeError, galsim.Moffat, beta=3, scale_radius=3, half_light_radius=1, fwhm=2)
    assert_raises(TypeError, galsim.Moffat, beta=3, half_light_radius=1, fwhm=2)
    assert_raises(TypeError, galsim.Moffat, beta=3, scale_radius=3, fwhm=2)
    assert_raises(TypeError, galsim.Moffat, beta=3, scale_radius=3, half_light_radius=1)
    assert_raises(TypeError, galsim.Moffat, beta=3)

    # beta <= 1.1 needs to be truncated.
    assert_raises(ValueError, galsim.Moffat, beta=1.1, scale_radius=3)
    assert_raises(ValueError, galsim.Moffat, beta=0.9, scale_radius=3)

    # trunc must be > sqrt(2) * hlr
    assert_raises(ValueError, galsim.Moffat, beta=3, half_light_radius=1, trunc=1.4)

    # Other errors
    assert_raises(TypeError, galsim.Moffat, scale_radius=3)
    assert_raises(ValueError, galsim.Moffat, beta=3, scale_radius=3, trunc=-1)

@timer
def test_moffat_properties():
    """Test some basic properties of the Moffat profile.
    """
    # Code was formerly:
    # psf = galsim.Moffat(beta=2.0, truncationFWHM=2, flux=test_flux, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in
    # physical units.  However, the same profile can be constructed using
    # fwhm=1.4686232496771867,
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    test_flux = 1.8
    fwhm_backwards_compatible = 1.4686232496771867
    psf = galsim.Moffat(beta=2.0, fwhm=fwhm_backwards_compatible,
                        trunc=2*fwhm_backwards_compatible, flux=test_flux)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid, cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxk, 11.613036117918105)
    np.testing.assert_almost_equal(psf.stepk, 0.62831853071795873)
    np.testing.assert_almost_equal(psf.kValue(cen), test_flux+0j)
    np.testing.assert_almost_equal(psf.half_light_radius, 1.0)
    np.testing.assert_almost_equal(psf.fwhm, fwhm_backwards_compatible)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.50654651638242509)
    np.testing.assert_almost_equal(psf.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(psf.flux, test_flux)
    np.testing.assert_almost_equal(psf.xValue(cen), psf.max_sb)

    # Now create the same profile using the half_light_radius:
    psf = galsim.Moffat(beta=2.0, half_light_radius=1.,
                        trunc=2*fwhm_backwards_compatible, flux=test_flux)
    np.testing.assert_equal(psf.centroid, cen)
    np.testing.assert_almost_equal(psf.maxk, 11.613036112206663)
    np.testing.assert_almost_equal(psf.stepk, 0.62831853071795862)
    np.testing.assert_almost_equal(psf.kValue(cen), test_flux+0j)
    np.testing.assert_almost_equal(psf.half_light_radius, 1.0)
    np.testing.assert_almost_equal(psf.fwhm, fwhm_backwards_compatible)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.50654651638242509)
    np.testing.assert_almost_equal(psf.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(psf.flux, test_flux)
    np.testing.assert_almost_equal(psf.xValue(cen), psf.max_sb)

    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        psfFlux = galsim.Moffat(2.0, fwhm=fwhm_backwards_compatible,
                                trunc=2*fwhm_backwards_compatible, flux=inFlux)
        outFlux = psfFlux.flux
        np.testing.assert_almost_equal(outFlux, inFlux)


@timer
def test_moffat_radii():
    """Test initialization of Moffat with different types of radius specification.
    """
    import math
    test_hlr = 1.7
    test_fwhm = 1.8
    test_scale = 1.9
    test_beta = 2.

    # Test constructor using half-light-radius:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, half_light_radius=test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print('hlr_sum = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with half-light radius")
    np.testing.assert_equal(
            test_gal.half_light_radius, test_hlr,
            err_msg="Moffat half_light_radius returned wrong value")

    # test that fwhm provides correct FWHM
    got_fwhm = test_gal.fwhm
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print('fwhm ratio = ', test_fwhm_ratio)
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with half-light radius")

    # test that scale_radius provides correct scale
    got_scale = test_gal.scale_radius
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print('scale ratio = ', test_scale_ratio)
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with half-light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, scale_radius = test_scale)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale,0)) / center
    print('scale ratio = ',ratio)
    np.testing.assert_almost_equal(
            ratio, pow(2,-test_beta), decimal=4,
            err_msg="Error in Moffat constructor with scale")
    np.testing.assert_equal(
            test_gal.scale_radius, test_scale,
            err_msg="Moffat scale_radius not correct")
    np.testing.assert_equal(
            test_gal.beta, test_beta,
            err_msg="Moffat beta not correct")
    np.testing.assert_equal(
            test_gal.trunc, 0,
            err_msg="Moffat trunc not correct")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.half_light_radius
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print('hlr_sum (profile initialized with scale_radius) = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Moffat initialized with scale radius.")

    # test that fwhm provides correct FWHM
    got_fwhm = test_gal.fwhm
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print('fwhm ratio = ', test_fwhm_ratio)
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with scale radius")

    # Test constructor using FWHM:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print('fwhm ratio = ',ratio)
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with fwhm")
    np.testing.assert_equal(
            test_gal.fwhm, test_fwhm,
            err_msg="Moffat fwhm returned wrong value")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.half_light_radius
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print('hlr_sum (profile initialized with FWHM) = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Moffat initialized with FWHM.")
    # test that scale_radius provides correct scale
    got_scale = test_gal.scale_radius
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print('scale ratio = ', test_scale_ratio)
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with scale radius")

    # Now repeat everything using a severe truncation.  (Above had no truncation.)

    # Test constructor using half-light-radius:
    test_gal = galsim.Moffat(flux=1., beta=test_beta, half_light_radius=test_hlr,
                             trunc=2*test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr)
    print('hlr_sum = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with half-light radius")
    np.testing.assert_equal(
            test_gal.half_light_radius, test_hlr,
            err_msg="Moffat hlr incorrect")

    # test that fwhm provides correct FWHM
    got_fwhm = test_gal.fwhm
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print('fwhm ratio = ', test_fwhm_ratio)
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with half-light radius")

    # test that scale_radius provides correct scale
    got_scale = test_gal.scale_radius
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print('scale ratio = ', test_scale_ratio)
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with half-light radius")
    np.testing.assert_equal(
            test_gal.scale_radius, got_scale,
            err_msg="Moffat scale_radius returned wrong value")
    np.testing.assert_equal(
            test_gal.trunc, 2*test_hlr,
            err_msg="Moffat trunc returned wrong value")

    # Test constructor using scale radius:
    test_gal = galsim.Moffat(flux=1., beta=test_beta, trunc=2*test_scale,
                             scale_radius=test_scale)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale,0)) / center
    print('scale ratio = ', ratio)
    np.testing.assert_almost_equal(
            ratio, pow(2,-test_beta), decimal=4,
            err_msg="Error in Moffat constructor with scale")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.half_light_radius
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print('hlr_sum (truncated profile initialized with scale_radius) = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for truncated Moffat "+
                    "initialized with scale radius.")

    # test that fwhm provides correct FWHM
    got_fwhm = test_gal.fwhm
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) /
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print('fwhm ratio = ', test_fwhm_ratio)
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for truncated Moffat initialized with scale radius")

    # Test constructor using FWHM:
    test_gal = galsim.Moffat(flux=1, beta=test_beta, trunc=2.*test_fwhm,
                             fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print('fwhm ratio = ', ratio)
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with fwhm")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.half_light_radius
    hlr_sum = radial_integrate(test_gal, 0., got_hlr)
    print('hlr_sum (truncated profile initialized with FWHM) = ',hlr_sum)
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for truncated Moffat initialized with FWHM.")

    # test that scale_radius provides correct scale
    got_scale = test_gal.scale_radius
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) /
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print('scale ratio = ', test_scale_ratio)
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for truncated Moffat initialized with scale radius")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.shear(g1=0.3, g2=0.1)
    assert_raises(AttributeError, getattr, test_gal_shear, "beta")
    assert_raises(AttributeError, getattr, test_gal_shear, "fwhm")
    assert_raises(AttributeError, getattr, test_gal_shear, "half_light_radius")
    assert_raises(AttributeError, getattr, test_gal_shear, "scale_radius")


@timer
def test_moffat_flux_scaling():
    """Test flux scaling for Moffat.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_scale = 1.8
    test_flux = 17.9

    for test_beta in [ 1.5, 2., 2.5, 3., 3.8 ]:
        for test_trunc in [ 0., 8.5 ]:

            # init with scale_radius only (should be ok given last tests)
            obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
            obj *= 2.
            np.testing.assert_almost_equal(
                obj.flux, test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __imul__.")
            obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
            obj /= 2.
            np.testing.assert_almost_equal(
                obj.flux, test_flux / 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __idiv__.")
            obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
            obj2 = obj * 2.
            # First test that original obj is unharmed...
            np.testing.assert_almost_equal(
                obj.flux, test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (original).")
            # Then test new obj2 flux
            np.testing.assert_almost_equal(
                obj2.flux, test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (result).")
            obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
            obj2 = 2. * obj
            # First test that original obj is unharmed...
            np.testing.assert_almost_equal(
                obj.flux, test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (original).")
            # Then test new obj2 flux
            np.testing.assert_almost_equal(
                obj2.flux, test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (result).")
            obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc,
                                flux=test_flux)
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
def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Moffat.  Params include beta, scale_radius, half_light_radius, fwhm, trunc, flux and gsparams.
    # The following should all test unequal:
    gals = [galsim.Moffat(beta=3.0, scale_radius=1.0),
            galsim.Moffat(beta=3.1, scale_radius=1.1),
            galsim.Moffat(beta=3.0, scale_radius=1.1),
            galsim.Moffat(beta=3.0, half_light_radius=1.2),
            galsim.Moffat(beta=3.0, fwhm=1.3),
            galsim.Moffat(beta=3.0, scale_radius=1.1, trunc=2.0),
            galsim.Moffat(beta=3.0, scale_radius=1.0, flux=1.1),
            galsim.Moffat(beta=3.0, scale_radius=1.0, gsparams=gsp)]
    all_obj_diff(gals)


if __name__ == "__main__":
    test_moffat()
    test_moffat_properties()
    test_moffat_radii()
    test_moffat_flux_scaling()
    test_ne()
