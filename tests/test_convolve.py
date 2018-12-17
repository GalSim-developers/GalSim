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

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images.

@timer
def test_convolve():
    """Test the convolution of a Moffat and a Box profile against a known result.
    """
    dx = 0.2
    # Using an exact Maple calculation for the comparison.  Only accurate to 4 decimal places.
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_pixel.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    # Code was formerly:
    # psf = galsim.Moffat(beta=1.5, truncationFWHM=4, flux=1, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in
    # physical units.  However, the same profile can be constructed using
    # fwhm=1.0927449310213702,
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    fwhm_backwards_compatible = 1.0927449310213702
    psf = galsim.Moffat(beta=1.5, fwhm=fwhm_backwards_compatible, trunc=4*fwhm_backwards_compatible,
                        flux=1)
    pixel = galsim.Pixel(scale=dx, flux=1.)
    # Note: Since both of these have hard edges, GalSim wants to do this with real_space=True.
    # Here we are intentionally tesing the Fourier convolution, so we want to suppress the
    # warning that GalSim emits.
    with assert_warns(galsim.GalSimWarning):
        # We'll do the real space convolution below
        conv = galsim.Convolve([psf,pixel],real_space=False)
        conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Moffat convolved with Pixel disagrees with expected result")
        assert psf.gsparams is galsim.GSParams.default
        assert pixel.gsparams is galsim.GSParams.default
        assert conv.gsparams is galsim.GSParams.default

        # Other ways to do the convolution:
        conv = galsim.Convolve(psf,pixel,real_space=False)
        conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
        assert conv.gsparams is galsim.GSParams.default

        # Check with default_params
        conv = galsim.Convolve([psf,pixel],real_space=False,gsparams=default_params)
        conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve([psf,pixel]) with default_params disagrees with"
                "expected result")
        # In this case, it's not the same object, but it should be ==
        assert conv.gsparams is not galsim.GSParams.default
        assert conv.gsparams == galsim.GSParams.default
        assert conv.gsparams is default_params
        # Also the components shouldn't have changed.
        assert conv.obj_list[0] is psf
        assert conv.obj_list[1] is pixel

        conv = galsim.Convolve([psf,pixel],real_space=False,gsparams=galsim.GSParams())
        conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve([psf,pixel]) with GSParams() disagrees with"
                "expected result")
        assert conv.gsparams is not galsim.GSParams.default
        assert conv.gsparams == galsim.GSParams.default

    cen = galsim.PositionD(0,0)
    np.testing.assert_equal(conv.centroid, cen)
    np.testing.assert_almost_equal(conv.flux, psf.flux * pixel.flux)
    # Not almost_equal.  Convolutions don't give a very good estimate.
    # They are almost always too high, which is actually ok for how we use max_sb for phot shooting.
    np.testing.assert_array_less(conv.xValue(cen), conv.max_sb)

    check_basic(conv, "Moffat * Pixel")

    # Test photon shooting.
    with assert_warns(galsim.GalSimWarning):
        do_shoot(conv,myImg,"Moffat * Pixel")

    # Convolution of just one argument should be equivalent to that argument.
    single = galsim.Convolve(psf)
    gsobject_compare(single, psf)
    check_basic(single, "`convolution' of single Moffat")
    do_pickle(single)
    do_shoot(single, myImg, "single Convolution")

    single = galsim.Convolve([psf])
    gsobject_compare(single, psf)
    check_basic(single, "`convolution' of single Moffat")
    do_pickle(single)

    single = galsim.Convolution(psf)
    gsobject_compare(single, psf)
    check_basic(single, "`convolution' of single Moffat")
    do_pickle(single)

    single = galsim.Convolution([psf])
    gsobject_compare(single, psf)
    check_basic(single, "`convolution' of single Moffat")
    do_pickle(single)

    # Should raise an exception for invalid arguments
    assert_raises(TypeError, galsim.Convolve)
    assert_raises(TypeError, galsim.Convolve, myImg)
    assert_raises(TypeError, galsim.Convolve, [myImg])
    assert_raises(TypeError, galsim.Convolve, [psf, myImg])
    assert_raises(TypeError, galsim.Convolve, [psf, psf, myImg])
    assert_raises(TypeError, galsim.Convolve, [psf, psf], realspace=False)
    assert_raises(TypeError, galsim.Convolution)
    assert_raises(TypeError, galsim.Convolution, myImg)
    assert_raises(TypeError, galsim.Convolution, [myImg])
    assert_raises(TypeError, galsim.Convolution, [psf, myImg])
    assert_raises(TypeError, galsim.Convolution, [psf, psf, myImg])
    assert_raises(TypeError, galsim.Convolution, [psf, psf], realspace=False)

    with assert_warns(galsim.GalSimWarning):
        triple = galsim.Convolve(psf, psf, pixel)
    assert_raises(galsim.GalSimError, triple.xValue, galsim.PositionD(0,0))
    assert_raises(galsim.GalSimError, triple.drawReal, myImg)

    deconv = galsim.Convolve(psf, galsim.Deconvolve(pixel))
    assert_raises(galsim.GalSimError, deconv.xValue, galsim.PositionD(0,0))
    assert_raises(galsim.GalSimError, deconv.drawReal, myImg)
    assert_raises(galsim.GalSimError, deconv.drawPhot, myImg, n_photons=10)


@timer
def test_convolve_flux_scaling():
    """Test flux scaling for Convolve.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_flux = 17.9
    test_sigma = 1.8
    test_hlr = 1.9

    # init with Gaussian and DeVauc only (should be ok given last tests)
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
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
def test_shearconvolve():
    """Test the convolution of a sheared Gaussian and a Box profile against a known result.
    """
    e1 = 0.04
    e2 = 0.0
    myShear = galsim.Shear(e1=e1, e2=e2)
    dx = 0.2
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear_convolve_box.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    psf = galsim.Gaussian(flux=1, sigma=1).shear(e1=e1, e2=e2)
    pixel = galsim.Pixel(scale=dx, flux=1.)
    conv = galsim.Convolve([psf,pixel])
    conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")

    # Check with default_params
    conv = galsim.Convolve([psf,pixel],gsparams=default_params)
    conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with default_params disagrees with "
            "expected result")
    conv = galsim.Convolve([psf,pixel],gsparams=galsim.GSParams())
    conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel)
    conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")

    check_basic(conv, "sheared Gaussian * Pixel")

    # Test photon shooting.
    with assert_warns(galsim.GalSimWarning):
        do_shoot(conv,myImg,"sheared Gaussian * Pixel")


@timer
def test_realspace_convolve():
    """Test the real-space convolution of a Moffat and a Box profile against a known result.
    """
    dx = 0.2
    # Note: Using an image created from Maple "exact" calculations.
    saved_img = galsim.fits.read(os.path.join(imgdir, "moffat_pixel.fits"))
    img = galsim.ImageF(saved_img.bounds, scale=dx)
    img.setCenter(0,0)

    # Code was formerly:
    # psf = galsim.Moffat(beta=1.5, truncationFWHM=4, flux=1, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in
    # physical units.  However, the same profile can be constructed using
    # fwhm=1.0927449310213702,
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    fwhm_backwards_compatible = 1.0927449310213702
    psf = galsim.Moffat(beta=1.5, half_light_radius=1,
                        trunc=4*fwhm_backwards_compatible, flux=1)
    #psf = galsim.Moffat(beta=1.5, fwhm=fwhm_backwards_compatible,
                        #trunc=4*fwhm_backwards_compatible, flux=1)
    pixel = galsim.Pixel(scale=dx, flux=1.)
    conv = galsim.Convolve([psf,pixel],real_space=True)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")

    # Check with default_params
    conv = galsim.Convolve([psf,pixel],real_space=True,gsparams=default_params)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with default_params disagrees with "
            "expected result")
    conv = galsim.Convolve([psf,pixel],real_space=True,gsparams=galsim.GSParams())
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel,real_space=True)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")

    # The real-space convolution algorithm is not (trivially) independent of the order of
    # the two things being convolved.  So check the opposite order.
    conv = galsim.Convolve([pixel,psf],real_space=True)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([pixel,psf]) disagrees with expected result")

    check_basic(conv, "Truncated Moffat*Box", approx_maxsb=True)

    # Test kvalues
    do_kvalue(conv,img,"Truncated Moffat*Box")

    # Check picklability
    do_pickle(conv, lambda x: x.drawImage(method='sb'))
    do_pickle(conv)

    # Check some warnings that should be raised
    # More than 2 with only hard edges gives a warning either way. (Different warnings though.)
    assert_warns(galsim.GalSimWarning, galsim.Convolve, [psf, psf, pixel])
    assert_warns(galsim.GalSimWarning, galsim.Convolve, [psf, psf, pixel], real_space=False)
    assert_warns(galsim.GalSimWarning, galsim.Convolve, [psf, psf, pixel], real_space=True)
    # 2 with hard edges gives a warning if we ask it not to use real_space
    assert_warns(galsim.GalSimWarning, galsim.Convolve, [psf, pixel], real_space=False)
    # >2 of any kind give a warning if we ask it to use real_space
    g = galsim.Gaussian(sigma=2)
    assert_warns(galsim.GalSimWarning, galsim.Convolve, [g, g, g], real_space=True)
    # non-analytic profiles cannot do real_space
    d = galsim.Deconvolve(galsim.Gaussian(sigma=2))
    assert_warns(galsim.GalSimWarning, galsim.Convolve, [g, d], real_space=True)
    assert_raises(TypeError, galsim.Convolve, [g, d], real_space='true')

    # Repeat some of the above for AutoConvolve and AutoCorrelate
    conv = galsim.AutoConvolve(psf,real_space=True)
    check_basic(conv, "AutoConvolve Truncated Moffat", approx_maxsb=True)
    do_kvalue(conv,img,"AutoConvolve Truncated Moffat")
    do_pickle(conv)

    conv = galsim.AutoCorrelate(psf,real_space=True)
    check_basic(conv, "AutoCorrelate Truncated Moffat", approx_maxsb=True)
    do_kvalue(conv,img,"AutoCorrelate Truncated Moffat")
    do_pickle(conv)

    assert_warns(galsim.GalSimWarning, galsim.AutoConvolve, psf, real_space=False)
    assert_warns(galsim.GalSimWarning, galsim.AutoConvolve, d, real_space=True)
    assert_warns(galsim.GalSimWarning, galsim.AutoCorrelate, psf, real_space=False)
    assert_warns(galsim.GalSimWarning, galsim.AutoCorrelate, d, real_space=True)
    assert_raises(TypeError, galsim.AutoConvolve, d, real_space='true')
    assert_raises(TypeError, galsim.AutoCorrelate, d, real_space='true')


@timer
def test_realspace_distorted_convolve():
    """
    The same as above, but both the Moffat and the Box are sheared, rotated and shifted
    to stress test the code that deals with this for real-space convolutions that wouldn't
    be tested otherwise.
    """
    dx = 0.2
    saved_img = galsim.fits.read(os.path.join(imgdir, "moffat_pixel_distorted.fits"))
    img = galsim.ImageF(saved_img.bounds, scale=dx)
    img.setCenter(0,0)

    fwhm_backwards_compatible = 1.0927449310213702
    psf = galsim.Moffat(beta=1.5, half_light_radius=1,
                        trunc=4*fwhm_backwards_compatible, flux=1)
    #psf = galsim.Moffat(beta=1.5, fwhm=fwhm_backwards_compatible,
                        #trunc=4*fwhm_backwards_compatible, flux=1)
    psf = psf.shear(g1=0.11,g2=0.17).rotate(13 * galsim.degrees)
    pixel = galsim.Pixel(scale=dx, flux=1.)
    pixel = pixel.shear(g1=0.2,g2=0.0).rotate(80 * galsim.degrees).shift(0.13,0.27)
    # NB: real-space is chosen automatically
    conv = galsim.Convolve([psf,pixel])
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([psf,pixel]) (distorted) disagrees with expected result")

    # Check with default_params
    conv = galsim.Convolve([psf,pixel],gsparams=default_params)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([psf,pixel]) (distorted) with default_params disagrees with "
            "expected result")
    conv = galsim.Convolve([psf,pixel],gsparams=galsim.GSParams())
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([psf,pixel]) (distorted) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve(psf,pixel) (distorted) disagrees with expected result")

    # The real-space convolution algorithm is not (trivially) independent of the order of
    # the two things being convolved.  So check the opposite order.
    conv = galsim.Convolve([pixel,psf])
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([pixel,psf]) (distorted) disagrees with expected result")


@timer
def test_realspace_shearconvolve():
    """Test the real-space convolution of a sheared Gaussian and a Box profile against a
       known result.
    """
    e1 = 0.04
    e2 = 0.0
    myShear = galsim.Shear(e1=e1, e2=e2)
    dx = 0.2
    saved_img = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear_convolve_box.fits"))
    img = galsim.ImageF(saved_img.bounds, scale=dx)
    img.setCenter(0,0)

    psf = galsim.Gaussian(flux=1, sigma=1)
    psf = psf.shear(e1=e1,e2=e2)
    pixel = galsim.Pixel(scale=dx, flux=1.)
    conv = galsim.Convolve([psf,pixel],real_space=True)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")

    # Check with default_params
    conv = galsim.Convolve([psf,pixel],real_space=True,gsparams=default_params)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with default_params disagrees with "
            "expected result")
    conv = galsim.Convolve([psf,pixel],real_space=True,gsparams=galsim.GSParams())
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel,real_space=True)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")

    # The real-space convolution algorithm is not (trivially) independent of the order of
    # the two things being convolved.  So check the opposite order.
    conv = galsim.Convolve([pixel,psf],real_space=True)
    conv.drawImage(img,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([pixel,psf]) disagrees with expected result")

@timer
def test_deconvolve():
    """Test that deconvolution works as expected
    """
    dx = 0.4
    myImg1 = galsim.ImageF(80,80, scale=dx)
    myImg1.setCenter(0,0)
    myImg2 = galsim.ImageF(80,80, scale=dx)
    myImg2.setCenter(0,0)

    psf = galsim.Moffat(beta=3.8, fwhm=1.3, flux=5)
    inv_psf = galsim.Deconvolve(psf)
    psf.drawImage(myImg1, method='no_pixel')
    conv = galsim.Convolve(psf,psf,inv_psf)
    conv.drawImage(myImg2, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Image of Deconvolve * obj^2 doesn't match obj alone")

    cen = galsim.PositionD(0,0)
    np.testing.assert_equal(inv_psf.centroid, cen)
    np.testing.assert_almost_equal(inv_psf.flux, 1./psf.flux)
    # This doesn't really have any meaning, but this is what we've assigned to a deconvolve max_sb.
    np.testing.assert_almost_equal(inv_psf.max_sb, -psf.max_sb / psf.flux**2)

    check_basic(inv_psf, "Deconvolve(Moffat)", do_x=False)

    # Also check Deconvolve with an asymmetric profile.
    obj1 = galsim.Gaussian(sigma=3., flux=4).shift(-0.2, -0.4)
    obj2 = galsim.Gaussian(sigma=6., flux=1.3).shift(0.3, 0.3)
    obj = galsim.Add(obj1, obj2)
    inv_obj = galsim.Deconvolve(obj)
    conv = galsim.Convolve([inv_obj, obj, obj])
    conv.drawImage(myImg1, method='no_pixel')
    obj.drawImage(myImg2, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Image of Deconvolve of asymmetric sum of Gaussians doesn't match obj alone")

    np.testing.assert_equal(inv_obj.centroid, -obj.centroid)
    np.testing.assert_almost_equal(inv_obj.flux, 1./obj.flux)
    np.testing.assert_almost_equal(inv_obj.max_sb, -obj.max_sb / obj.flux**2)

    check_basic(inv_obj, "Deconvolve(asym)", do_x=False)

    # Check picklability
    do_pickle(inv_obj)

    # And a significantly transformed deconvolve object
    jac = (0.3, -0.8, -0.7, 0.4)
    transformed_obj = obj.transform(*jac)
    transformed_inv_obj = inv_obj.transform(*jac)
    # Fix the flux -- most of the transformation commutes with deconvolution, but not flux scaling
    transformed_inv_obj /= transformed_obj.flux * transformed_inv_obj.flux
    check_basic(transformed_inv_obj, "transformed Deconvolve(asym)", do_x=False)
    conv = galsim.Convolve([transformed_inv_obj, transformed_obj, obj])
    conv.drawImage(myImg1, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Transformed Deconvolve didn't cancel transformed original")

    np.testing.assert_equal(transformed_inv_obj.centroid, -transformed_obj.centroid)
    np.testing.assert_almost_equal(transformed_inv_obj.flux, 1./transformed_obj.flux)
    np.testing.assert_almost_equal(transformed_inv_obj.max_sb,
                                   -transformed_obj.max_sb / transformed_obj.flux**2)

    check_basic(transformed_inv_obj, "transformed Deconvolve(asym)", do_x=False)

    # Check picklability
    do_pickle(transformed_inv_obj)

    # Should raise an exception for invalid arguments
    assert_raises(TypeError, galsim.Deconvolve)
    assert_raises(TypeError, galsim.Deconvolve, myImg1)
    assert_raises(TypeError, galsim.Deconvolve, [psf])
    assert_raises(TypeError, galsim.Deconvolve, psf, psf)
    assert_raises(TypeError, galsim.Deconvolve, psf, real_space=False)
    assert_raises(TypeError, galsim.Deconvolution)
    assert_raises(TypeError, galsim.Deconvolution, myImg1)
    assert_raises(TypeError, galsim.Deconvolution, [psf])
    assert_raises(TypeError, galsim.Deconvolution, psf, psf)
    assert_raises(TypeError, galsim.Deconvolution, psf, real_space=False)

    assert_raises(NotImplementedError, inv_obj.xValue, galsim.PositionD(0,0))
    assert_raises(NotImplementedError, inv_obj.drawReal, myImg1)
    assert_raises(NotImplementedError, inv_obj.shoot, 1)


@timer
def test_autoconvolve():
    """Test that auto-convolution works the same as convolution with itself.
    """
    dx = 0.4
    myImg1 = galsim.ImageF(80,80, scale=dx)
    myImg1.setCenter(0,0)
    myImg2 = galsim.ImageF(80,80, scale=dx)
    myImg2.setCenter(0,0)

    psf = galsim.Moffat(beta=3.8, fwhm=1.3, flux=5)
    conv = galsim.Convolve([psf,psf])
    conv.drawImage(myImg1, method='no_pixel')
    conv2 = galsim.AutoConvolve(psf)
    conv2.drawImage(myImg2, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Moffat convolved with self disagrees with AutoConvolve result")

    # Check with default_params
    conv = galsim.AutoConvolve(psf, gsparams=default_params)
    conv.drawImage(myImg1, method='no_pixel')
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Using AutoConvolve with default_params disagrees with expected result")
    conv = galsim.AutoConvolve(psf, gsparams=galsim.GSParams())
    conv.drawImage(myImg1, method='no_pixel')
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Using AutoConvolve with GSParams() disagrees with expected result")

    check_basic(conv, "AutoConvolve(Moffat)")

    cen = galsim.PositionD(0,0)
    np.testing.assert_equal(conv2.centroid, cen)
    np.testing.assert_almost_equal(conv2.flux, psf.flux**2)
    np.testing.assert_array_less(conv2.xValue(cen), conv2.max_sb)

    # Check picklability
    do_pickle(conv2, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(conv2)

    # Test photon shooting.
    do_shoot(conv2,myImg2,"AutoConvolve(Moffat)")

    # For a symmetric profile, AutoCorrelate is the same thing:
    conv2 = galsim.AutoCorrelate(psf)
    conv2.drawImage(myImg2, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Moffat convolved with self disagrees with AutoCorrelate result")

    # And check AutoCorrelate with gsparams:
    conv2 = galsim.AutoCorrelate(psf, gsparams=default_params)
    conv2.drawImage(myImg1, method='no_pixel')
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Using AutoCorrelate with default_params disagrees with expected result")
    conv2 = galsim.AutoCorrelate(psf, gsparams=galsim.GSParams())
    conv2.drawImage(myImg1, method='no_pixel')
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Using AutoCorrelate with GSParams() disagrees with expected result")

    cen = galsim.PositionD(0,0)
    np.testing.assert_equal(conv2.centroid, cen)
    np.testing.assert_almost_equal(conv2.flux, psf.flux**2)
    np.testing.assert_array_less(conv2.xValue(cen), conv2.max_sb)

    # Also check AutoConvolve with an asymmetric profile.
    # (AutoCorrelate with this profile is done below...)
    obj1 = galsim.Gaussian(sigma=3., flux=4).shift(-0.2, -0.4)
    obj2 = galsim.Gaussian(sigma=6., flux=1.3).shift(0.3, 0.3)
    add = galsim.Add(obj1, obj2)
    conv = galsim.Convolve([add, add])
    conv.drawImage(myImg1, method='no_pixel')
    autoconv = galsim.AutoConvolve(add)
    autoconv.drawImage(myImg2, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Asymmetric sum of Gaussians convolved with self disagrees with "+
            "AutoConvolve result")

    cen = 2. * add.centroid
    np.testing.assert_equal(autoconv.centroid, cen)
    np.testing.assert_almost_equal(autoconv.flux, add.flux**2)
    np.testing.assert_array_less(autoconv.xValue(cen), autoconv.max_sb)

    check_basic(autoconv, "AutoConvolve(asym)")

    # Should raise an exception for invalid arguments
    assert_raises(TypeError, galsim.AutoConvolve)
    assert_raises(TypeError, galsim.AutoConvolve, myImg1)
    assert_raises(TypeError, galsim.AutoConvolve, [psf])
    assert_raises(TypeError, galsim.AutoConvolve, psf, psf)
    assert_raises(TypeError, galsim.AutoConvolve, psf, realspace=False)
    assert_raises(TypeError, galsim.AutoConvolution)
    assert_raises(TypeError, galsim.AutoConvolution, myImg1)
    assert_raises(TypeError, galsim.AutoConvolution, [psf])
    assert_raises(TypeError, galsim.AutoConvolution, psf, psf)
    assert_raises(TypeError, galsim.AutoConvolution, psf, realspace=False)


@timer
def test_autocorrelate():
    """Test that auto-correlation works the same as convolution with the mirror image of itself.

    (See the Signal processing Section of http://en.wikipedia.org/wiki/Autocorrelation)
    """
    dx = 0.7
    myImg1 = galsim.ImageF(80,80, scale=dx)
    myImg1.setCenter(0,0)
    myImg2 = galsim.ImageF(80,80, scale=dx)
    myImg2.setCenter(0,0)

    # Use a function that is NOT two-fold rotationally symmetric, e.g. two different flux Gaussians
    obj1 = galsim.Gaussian(sigma=3., flux=4).shift(-0.2, -0.4)
    obj2 = galsim.Gaussian(sigma=6., flux=1.3).shift(0.3, 0.3)
    add1 = galsim.Add(obj1, obj2)
    # Here we rotate by 180 degrees to create mirror image
    add2 = (galsim.Add(obj1, obj2)).rotate(180. * galsim.degrees)
    conv = galsim.Convolve([add1, add2])
    conv.drawImage(myImg1, method='no_pixel')
    corr = galsim.AutoCorrelate(add1)
    corr.drawImage(myImg2, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Asymmetric sum of Gaussians convolved with mirror of self disagrees with "+
            "AutoCorrelate result")

    check_basic(conv, "AutoCorrelate")

    # Test photon shooting.
    do_shoot(corr,myImg2,"AutoCorrelate")

    # Check picklability
    do_pickle(corr, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(corr)

    # Should raise an exception for invalid arguments
    assert_raises(TypeError, galsim.AutoCorrelate)
    assert_raises(TypeError, galsim.AutoCorrelate, myImg1)
    assert_raises(TypeError, galsim.AutoCorrelate, [obj1])
    assert_raises(TypeError, galsim.AutoCorrelate, obj1, obj2)
    assert_raises(TypeError, galsim.AutoCorrelate, obj1, realspace=False)
    assert_raises(TypeError, galsim.AutoCorrelation)
    assert_raises(TypeError, galsim.AutoCorrelation, myImg1)
    assert_raises(TypeError, galsim.AutoCorrelation, [obj1])
    assert_raises(TypeError, galsim.AutoCorrelation, obj1, obj2)
    assert_raises(TypeError, galsim.AutoCorrelation, obj1, realspace=False)


@timer
def test_ne():
    """ Check that inequality works as expected."""
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)
    gal1 = galsim.Gaussian(fwhm=1)
    gal2 = galsim.Gaussian(fwhm=2)

    # Convolution.  Params are objs to convolve and potentially gsparams.
    # The following should test unequal
    gals = [galsim.Convolution(gal1),
            galsim.Convolution(gal1, gal2),
            galsim.Convolution(gal2, gal1),  # Not! commutative.
            galsim.Convolution(gal1, gal2, real_space=True),
            galsim.Convolution(galsim.Convolution(gal1, gal2), gal2),
            galsim.Convolution(gal1, galsim.Convolution(gal2, gal2)),  # Not! associative.
            galsim.Convolution(gal1, gsparams=gsp),
            galsim.Convolution(gal1, gsparams=gsp, propagate_gsparams=False)]
    all_obj_diff(gals)

    # Deconvolution.  Only params here are obj to deconvolve and gsparams.
    gals = [galsim.Deconvolution(gal1),
            galsim.Deconvolution(gal2),
            galsim.Deconvolution(gal1, gsparams=gsp),
            galsim.Deconvolution(gal1, gsparams=gsp, propagate_gsparams=False)]
    all_obj_diff(gals)

    # AutoConvolution.  Only params here are obj to deconvolve and gsparams.
    gals = [galsim.AutoConvolution(gal1),
            galsim.AutoConvolution(gal2),
            galsim.AutoConvolution(gal1, gsparams=gsp),
            galsim.AutoConvolution(gal1, gsparams=gsp, propagate_gsparams=False)]
    all_obj_diff(gals)

    # AutoCorrelation.  Only params here are obj to deconvolve and gsparams.
    gals = [galsim.AutoCorrelation(gal1),
            galsim.AutoCorrelation(gal2),
            galsim.AutoCorrelation(gal1, gsparams=gsp),
            galsim.AutoCorrelation(gal1, gsparams=gsp, propagate_gsparams=False)]
    all_obj_diff(gals)


@timer
def test_convolve_noise():
    """Test that noise propagation works properly for compount objects.
    """
    obj1 = galsim.Gaussian(sigma=1.7)
    obj2 = galsim.Gaussian(sigma=2.3)
    obj1.noise = galsim.UncorrelatedNoise(variance=0.3, scale=0.2)
    obj2.noise = galsim.UncorrelatedNoise(variance=0.5, scale=0.2)
    obj3 = galsim.Gaussian(sigma=2.9)

    # Convolve convolves the noise from a single component
    conv2 = galsim.Convolution([obj1,obj3])
    noise = galsim.Convolve([obj1.noise._profile, obj3, obj3])
    # xValue is too slow here.  Use drawImage to get variance.  (Just as CorrelatedNoise does.)
    variance = noise.drawImage(nx=1, ny=1, scale=1., method='sb')(1,1)
    np.testing.assert_almost_equal(conv2.noise.getVariance(), variance,
            err_msg = "Convolution of two objects did not correctly propagate noise varinace")
    conv2 = galsim.Convolution([obj2,obj3])
    noise = galsim.Convolve([obj2.noise._profile, obj3, obj3])
    variance = noise.drawImage(nx=1, ny=1, scale=1., method='sb')(1,1)
    np.testing.assert_almost_equal(conv2.noise.getVariance(), variance,
            err_msg = "Convolution of two objects did not correctly propagate noise varinace")

    # Convolution of multiple objects with noise attributes raises a warning and fails
    # to propagate noise properly.  (It takes the input noise from the first one.)
    conv2 = galsim.Convolution(obj1, obj2)
    conv3 = galsim.Convolution(obj1, obj2, obj3)
    with assert_warns(galsim.GalSimWarning):
        assert conv2.noise == obj1.noise.convolvedWith(obj2)
    with assert_warns(galsim.GalSimWarning):
        assert conv3.noise == obj1.noise.convolvedWith(galsim.Convolve(obj2,obj3))

    # Convolution with only one object uses that object's noise
    conv1 = galsim.Convolution(obj1)
    assert conv1.noise == obj1.noise

    # Other types don't propagate noise and give a warning about it.
    deconv = galsim.Deconvolution(obj2)
    autoconv = galsim.AutoConvolution(obj2)
    autocorr = galsim.AutoCorrelation(obj2)
    four = galsim.FourierSqrt(obj2)
    with assert_warns(galsim.GalSimWarning):
        assert deconv.noise is None
    with assert_warns(galsim.GalSimWarning):
        assert autoconv.noise is None
    with assert_warns(galsim.GalSimWarning):
        assert autocorr.noise is None
    with assert_warns(galsim.GalSimWarning):
        assert four.noise is None

    obj2.noise = None  # Remove obj2 noise for the rest.
    conv2 = galsim.Convolution(obj1, obj2)
    noise = galsim.Convolve([obj1.noise._profile, obj2, obj2])
    variance = noise.drawImage(nx=1, ny=1, scale=1., method='sb')(1,1)
    np.testing.assert_almost_equal(conv2.noise.getVariance(), variance,
            err_msg = "Convolution of two objects did not correctly propagate noise varinace")

    conv3 = galsim.Convolution(obj1, obj2, obj3)
    noise = galsim.Convolve([obj1.noise._profile, obj2, obj2, obj3, obj3])
    variance = noise.drawImage(nx=1, ny=1, scale=1., method='sb')(1,1)
    np.testing.assert_almost_equal(conv3.noise.getVariance(), variance,
            err_msg = "Convolution of three objects did not correctly propagate noise varinace")

    deconv = galsim.Deconvolution(obj2)
    autoconv = galsim.AutoConvolution(obj2)
    autocorr = galsim.AutoCorrelation(obj2)
    four = galsim.FourierSqrt(obj2)
    assert deconv.noise is None
    assert autoconv.noise is None
    assert autocorr.noise is None
    assert four.noise is None

@timer
def test_gsparams():
    """Test withGSParams with some non-default gsparams
    """
    obj1 = galsim.Exponential(half_light_radius=1.7)
    obj2 = galsim.Pixel(scale=0.2)
    gsp = galsim.GSParams(folding_threshold=1.e-4, maxk_threshold=1.e-4, maximum_fft_size=1.e4)
    gsp2 = galsim.GSParams(folding_threshold=1.e-2, maxk_threshold=1.e-2)

    # Convolve
    conv = galsim.Convolve(obj1, obj2)
    conv1 = conv.withGSParams(gsp)
    assert conv.gsparams == galsim.GSParams()
    assert conv1.gsparams == gsp
    assert conv1.obj_list[0].gsparams == gsp
    assert conv1.obj_list[1].gsparams == gsp

    conv2 = galsim.Convolve(obj1.withGSParams(gsp), obj2.withGSParams(gsp))
    conv3 = galsim.Convolve(galsim.Exponential(half_light_radius=1.7, gsparams=gsp),
                            galsim.Pixel(scale=0.2))
    conv4 = galsim.Convolve(obj1, obj2, gsparams=gsp)
    assert conv != conv1
    assert conv1 == conv2
    assert conv1 == conv3
    assert conv1 == conv4
    print('stepk = ',conv.stepk, conv1.stepk)
    assert conv1.stepk < conv.stepk
    print('maxk = ',conv.maxk, conv1.maxk)
    assert conv1.maxk > conv.maxk

    conv5 = galsim.Convolve(obj1, obj2, gsparams=gsp, propagate_gsparams=False)
    assert conv5 != conv4
    assert conv5.gsparams == gsp
    assert conv5.obj_list[0].gsparams == galsim.GSParams()
    assert conv5.obj_list[1].gsparams == galsim.GSParams()

    conv6 = conv5.withGSParams(gsp2)
    assert conv6 != conv5
    assert conv6.gsparams == gsp2
    assert conv6.obj_list[0].gsparams == galsim.GSParams()
    assert conv6.obj_list[1].gsparams == galsim.GSParams()

    # AutoConvolve
    conv = galsim.AutoConvolve(obj1)
    conv1 = conv.withGSParams(gsp)
    assert conv.gsparams == galsim.GSParams()
    assert conv1.gsparams == gsp
    assert conv1.orig_obj.gsparams == gsp

    conv2 = galsim.AutoConvolve(obj1.withGSParams(gsp))
    conv3 = galsim.AutoConvolve(obj1, gsparams=gsp)
    assert conv != conv1
    assert conv1 == conv2
    assert conv1 == conv3
    print('stepk = ',conv.stepk, conv1.stepk)
    assert conv1.stepk < conv.stepk
    print('maxk = ',conv.maxk, conv1.maxk)
    assert conv1.maxk > conv.maxk

    conv4 = galsim.AutoConvolve(obj1, gsparams=gsp, propagate_gsparams=False)
    assert conv4 != conv3
    assert conv4.gsparams == gsp
    assert conv4.orig_obj.gsparams == galsim.GSParams()

    conv5 = conv4.withGSParams(gsp2)
    assert conv5 != conv4
    assert conv5.gsparams == gsp2
    assert conv5.orig_obj.gsparams == galsim.GSParams()

    # AutoCorrelate
    conv = galsim.AutoCorrelate(obj1)
    conv1 = conv.withGSParams(gsp)
    assert conv.gsparams == galsim.GSParams()
    assert conv1.gsparams == gsp
    assert conv1.orig_obj.gsparams == gsp

    conv2 = galsim.AutoCorrelate(obj1.withGSParams(gsp))
    conv3 = galsim.AutoCorrelate(obj1, gsparams=gsp)
    assert conv != conv1
    assert conv1 == conv2
    assert conv1 == conv3
    print('stepk = ',conv.stepk, conv1.stepk)
    assert conv1.stepk < conv.stepk
    print('maxk = ',conv.maxk, conv1.maxk)
    assert conv1.maxk > conv.maxk

    conv4 = galsim.AutoCorrelate(obj1, gsparams=gsp, propagate_gsparams=False)
    assert conv4 != conv3
    assert conv4.gsparams == gsp
    assert conv4.orig_obj.gsparams == galsim.GSParams()

    conv5 = conv4.withGSParams(gsp2)
    assert conv5 != conv4
    assert conv5.gsparams == gsp2
    assert conv5.orig_obj.gsparams == galsim.GSParams()

    # Deconvolve
    conv = galsim.Convolve(obj1, galsim.Deconvolve(obj2))
    conv1 = conv.withGSParams(gsp)
    assert conv.gsparams == galsim.GSParams()
    assert conv1.gsparams == gsp
    assert conv1.obj_list[0].gsparams == gsp
    assert conv1.obj_list[1].gsparams == gsp
    assert conv1.obj_list[1].orig_obj.gsparams == gsp

    conv2 = galsim.Convolve(obj1, galsim.Deconvolve(obj2.withGSParams(gsp)))
    conv3 = galsim.Convolve(obj1.withGSParams(gsp), galsim.Deconvolve(obj2))
    conv4 = galsim.Convolve(obj1, galsim.Deconvolve(obj2, gsparams=gsp))
    assert conv != conv1
    assert conv1 == conv2
    assert conv1 == conv3
    assert conv1 == conv4
    print('stepk = ',conv.stepk, conv1.stepk)
    assert conv1.stepk < conv.stepk
    print('maxk = ',conv.maxk, conv1.maxk)
    assert conv1.maxk > conv.maxk

    conv5 = galsim.Convolve(obj1, galsim.Deconvolve(obj2, gsparams=gsp, propagate_gsparams=False))
    assert conv5 != conv4
    assert conv5.gsparams == gsp
    assert conv5.obj_list[0].gsparams == gsp
    assert conv5.obj_list[1].gsparams == gsp
    assert conv5.obj_list[1].orig_obj.gsparams == galsim.GSParams()

    conv6 = conv5.withGSParams(gsp2)
    assert conv6 != conv5
    assert conv6.gsparams == gsp2
    assert conv6.obj_list[0].gsparams == gsp2
    assert conv6.obj_list[1].gsparams == gsp2
    assert conv6.obj_list[1].orig_obj.gsparams == galsim.GSParams()

    # FourierSqrt
    conv = galsim.Convolve(obj1, galsim.FourierSqrt(obj2))
    conv1 = conv.withGSParams(gsp)
    assert conv.gsparams == galsim.GSParams()
    assert conv1.gsparams == gsp
    assert conv1.obj_list[0].gsparams == gsp
    assert conv1.obj_list[1].gsparams == gsp
    assert conv1.obj_list[1].orig_obj.gsparams == gsp

    conv2 = galsim.Convolve(obj1, galsim.FourierSqrt(obj2.withGSParams(gsp)))
    conv3 = galsim.Convolve(obj1.withGSParams(gsp), galsim.FourierSqrt(obj2))
    conv4 = galsim.Convolve(obj1, galsim.FourierSqrt(obj2, gsparams=gsp))
    assert conv != conv1
    assert conv1 == conv2
    assert conv1 == conv3
    assert conv1 == conv4
    print('stepk = ',conv.stepk, conv1.stepk)
    assert conv1.stepk < conv.stepk
    print('maxk = ',conv.maxk, conv1.maxk)
    assert conv1.maxk > conv.maxk

    conv5 = galsim.Convolve(obj1, galsim.FourierSqrt(obj2, gsparams=gsp, propagate_gsparams=False))
    assert conv5 != conv4
    assert conv5.gsparams == gsp
    assert conv5.obj_list[0].gsparams == gsp
    assert conv5.obj_list[1].gsparams == gsp
    assert conv5.obj_list[1].orig_obj.gsparams == galsim.GSParams()

    conv6 = conv5.withGSParams(gsp2)
    assert conv6 != conv5
    assert conv6.gsparams == gsp2
    assert conv6.obj_list[0].gsparams == gsp2
    assert conv6.obj_list[1].gsparams == gsp2
    assert conv6.obj_list[1].orig_obj.gsparams == galsim.GSParams()


if __name__ == "__main__":
    test_convolve()
    test_convolve_flux_scaling()
    test_shearconvolve()
    test_realspace_convolve()
    test_realspace_distorted_convolve()
    test_realspace_shearconvolve()
    test_deconvolve()
    test_autoconvolve()
    test_autocorrelate()
    test_ne()
    test_convolve_noise()
    test_gsparams()
