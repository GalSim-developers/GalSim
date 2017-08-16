# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

from galsim_test_helpers import *

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images.

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

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

# Some standard values for testing
test_flux = 1.8
test_hlr = 1.8
test_sigma = 1.8
test_scale = 1.8


@timer
def test_convolve():
    """Test the convolution of a Moffat and a Box SBProfile against a known result.
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
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # We'll do the real space convolution below
        conv = galsim.Convolve([psf,pixel],real_space=False)
        conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Moffat convolved with Pixel disagrees with expected result")

        # Other ways to do the convolution:
        conv = galsim.Convolve(psf,pixel,real_space=False)
        conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")

        # Check with default_params
        conv = galsim.Convolve([psf,pixel],real_space=False,gsparams=default_params)
        conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve([psf,pixel]) with default_params disagrees with"
                "expected result")
        conv = galsim.Convolve([psf,pixel],real_space=False,gsparams=galsim.GSParams())
        conv.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve([psf,pixel]) with GSParams() disagrees with"
                "expected result")

    cen = galsim.PositionD(0,0)
    np.testing.assert_equal(conv.centroid(), cen)
    np.testing.assert_almost_equal(conv.getFlux(), psf.flux * pixel.flux)
    np.testing.assert_almost_equal(conv.flux, psf.flux * pixel.flux)
    # Not almost_equal.  Convolutions don't give a very good estimate.
    # They are almost always too high, which is actually ok for how we use maxSB for phot shooting.
    np.testing.assert_array_less(conv.xValue(cen), conv.maxSB())

    check_basic(conv, "Moffat * Pixel")

    # Test photon shooting.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        do_shoot(conv,myImg,"Moffat * Pixel")
    # Clear the warnings registry for later so we can test that appropriate warnings are raised.
    galsim.Convolution.__init__.__globals__['__warningregistry__'].clear()

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
    try:
        np.testing.assert_raises(TypeError, galsim.Convolve)
        np.testing.assert_raises(TypeError, galsim.Convolve, myImg)
        np.testing.assert_raises(TypeError, galsim.Convolve, [myImg])
        np.testing.assert_raises(TypeError, galsim.Convolve, [psf, myImg])
        np.testing.assert_raises(TypeError, galsim.Convolve, [psf, psf, myImg])
        np.testing.assert_raises(TypeError, galsim.Convolve, [psf, psf], realspace=False)
        np.testing.assert_raises(TypeError, galsim.Convolution)
        np.testing.assert_raises(TypeError, galsim.Convolution, myImg)
        np.testing.assert_raises(TypeError, galsim.Convolution, [myImg])
        np.testing.assert_raises(TypeError, galsim.Convolution, [psf, myImg])
        np.testing.assert_raises(TypeError, galsim.Convolution, [psf, psf, myImg])
        np.testing.assert_raises(TypeError, galsim.Convolution, [psf, psf], realspace=False)
    except ImportError:
        pass

@timer
def test_convolve_flux_scaling():
    """Test flux scaling for Convolve.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12

    # init with Gaussian and DeVauc only (should be ok given last tests)
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = obj / 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")


@timer
def test_shearconvolve():
    """Test the convolution of a sheared Gaussian and a Box SBProfile against a known result.
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
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        do_shoot(conv,myImg,"sheared Gaussian * Pixel")
    # Clear the warnings registry for later so we can test that appropriate warnings are raised.
    galsim.GSObject.drawImage.__globals__['__warningregistry__'].clear()


@timer
def test_realspace_convolve():
    """Test the real-space convolution of a Moffat and a Box SBProfile against a known result.
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
    do_pickle(conv.SBProfile, lambda x: (repr(x.getObjs()), x.isRealSpace(), x.getGSParams()))
    do_pickle(conv, lambda x: x.drawImage(method='sb'))
    do_pickle(conv)
    do_pickle(conv.SBProfile)

    # Check some warnings that should be raised

    try:
        # More than 2 with only hard edges gives a warning either way. (Different warnings though.)
        np.testing.assert_warns(UserWarning, galsim.Convolve, [psf, psf, pixel])
        np.testing.assert_warns(UserWarning, galsim.Convolve, [psf, psf, pixel], real_space=False)
        np.testing.assert_warns(UserWarning, galsim.Convolve, [psf, psf, pixel], real_space=True)
        # 2 with hard edges gives a warning if we ask it not to use real_space
        np.testing.assert_warns(UserWarning, galsim.Convolve, [psf, pixel], real_space=False)
        # >2 of any kind give a warning if we ask it to use real_space
        g = galsim.Gaussian(sigma=2)
        np.testing.assert_warns(UserWarning, galsim.Convolve, [g, g, g], real_space=True)
        # non-analytic profiles cannot do real_space
        d = galsim.Deconvolve(galsim.Gaussian(sigma=2))
        np.testing.assert_warns(UserWarning, galsim.Convolve, [g, d], real_space=True)
    except ImportError:
        pass

    # Repeat some of the above for AutoConvolve and AutoCorrelate
    conv = galsim.AutoConvolve(psf,real_space=True)
    check_basic(conv, "AutoConvolve Truncated Moffat", approx_maxsb=True)
    do_kvalue(conv,img,"AutoConvolve Truncated Moffat")
    do_pickle(conv)
    do_pickle(conv.SBProfile)

    conv = galsim.AutoCorrelate(psf,real_space=True)
    check_basic(conv, "AutoCorrelate Truncated Moffat", approx_maxsb=True)
    do_kvalue(conv,img,"AutoCorrelate Truncated Moffat")
    do_pickle(conv)
    do_pickle(conv.SBProfile)

    try:
        np.testing.assert_warns(UserWarning, galsim.AutoConvolve, psf, real_space=False)
        np.testing.assert_warns(UserWarning, galsim.AutoConvolve, d, real_space=True)
        np.testing.assert_warns(UserWarning, galsim.AutoCorrelate, psf, real_space=False)
        np.testing.assert_warns(UserWarning, galsim.AutoCorrelate, d, real_space=True)
    except ImportError:
        pass



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
    """Test the real-space convolution of a sheared Gaussian and a Box SBProfile against a
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
def test_add():
    """Test the addition of two rescaled Gaussian profiles against a known double Gaussian result.
    """
    savedImg = galsim.fits.read(os.path.join(imgdir, "double_gaussian.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    gauss1 = galsim.Gaussian(flux=0.75, sigma=1)
    gauss2 = galsim.Gaussian(flux=0.25, sigma=3)
    sum_gauss = galsim.Add(gauss1,gauss2)
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) disagrees with expected result")

    cen = galsim.PositionD(0,0)
    np.testing.assert_equal(sum_gauss.centroid(), cen)
    np.testing.assert_almost_equal(sum_gauss.getFlux(), gauss1.flux + gauss2.flux)
    np.testing.assert_almost_equal(sum_gauss.flux, gauss1.flux + gauss2.flux)
    np.testing.assert_almost_equal(sum_gauss.xValue(cen), sum_gauss.maxSB())

    # Check with default_params
    sum_gauss = galsim.Add(gauss1,gauss2,gsparams=default_params)
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) with default_params disagrees with "
            "expected result")
    sum_gauss = galsim.Add(gauss1,gauss2,gsparams=galsim.GSParams())
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the sum:
    sum_gauss = gauss1 + gauss2
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject gauss1 + gauss2 disagrees with expected result")
    sum_gauss = gauss1
    sum_gauss += gauss2
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum = gauss1; sum += gauss2 disagrees with expected result")
    sum_gauss = galsim.Add([gauss1,gauss2])
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add([gauss1,gauss2]) disagrees with expected result")
    gauss1 = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = galsim.Gaussian(flux=1, sigma=3)
    sum_gauss = 0.75 * gauss1 + 0.25 * gauss2
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject 0.75 * gauss1 + 0.25 * gauss2 disagrees with expected result")
    sum_gauss = 0.75 * gauss1
    sum_gauss += 0.25 * gauss2
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum += 0.25 * gauss2 disagrees with expected result")

    check_basic(sum_gauss, "sum of 2 Gaussians")

    # Test photon shooting.
    do_shoot(sum_gauss,myImg,"sum of 2 Gaussians")

    # Test kvalues
    do_kvalue(sum_gauss,myImg,"sum of 2 Gaussians")

    # Check picklability
    do_pickle(sum_gauss.SBProfile, lambda x: (repr(x.getObjs()), x.getGSParams()))
    do_pickle(sum_gauss, lambda x: x.drawImage(method='sb'))
    do_pickle(sum_gauss)
    do_pickle(sum_gauss.SBProfile)

    # Sum of just one argument should be equivalent to that argument.
    single = galsim.Add(gauss1)
    gsobject_compare(single, gauss1)
    check_basic(single, "`sum' of 1 Gaussian")
    do_pickle(single)
    do_shoot(single, myImg, "Single Sum")

    single = galsim.Add([gauss1])
    gsobject_compare(single, gauss1)
    check_basic(single, "`sum' of 1 Gaussian")
    do_pickle(single)

    # Should raise an exception for invalid arguments
    try:
        np.testing.assert_raises(TypeError, galsim.Add)
        np.testing.assert_raises(TypeError, galsim.Add, myImg)
        np.testing.assert_raises(TypeError, galsim.Add, [myImg])
        np.testing.assert_raises(TypeError, galsim.Add, [gauss1, myImg])
        np.testing.assert_raises(TypeError, galsim.Add, [gauss1, gauss1, myImg])
        np.testing.assert_raises(TypeError, galsim.Add, [gauss1, gauss1], real_space=False)
        np.testing.assert_raises(TypeError, galsim.Sum)
        np.testing.assert_raises(TypeError, galsim.Sum, myImg)
        np.testing.assert_raises(TypeError, galsim.Sum, [myImg])
        np.testing.assert_raises(TypeError, galsim.Sum, [gauss1, myImg])
        np.testing.assert_raises(TypeError, galsim.Sum, [gauss1, gauss1, myImg])
        np.testing.assert_raises(TypeError, galsim.Sum, [gauss1, gauss1], real_space=False)
    except ImportError:
        pass


@timer
def test_sub_neg():
    """Test that a - b is the same as a + (-b)."""
    a = galsim.Gaussian(fwhm=1)
    b = galsim.Kolmogorov(fwhm=1)

    c = a - b
    d = a + (-b)

    assert c == d

    im1 = c.drawImage()
    im2 = d.drawImage(im1.copy())

    np.testing.assert_equal(im1.array, im2.array)

@timer
def test_add_flux_scaling():
    """Test flux scaling for Add.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12

    # init with Gaussian and Exponential only (should be ok given last tests)
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = obj / 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")


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
    np.testing.assert_equal(inv_psf.centroid(), cen)
    np.testing.assert_almost_equal(inv_psf.getFlux(), 1./psf.flux)
    np.testing.assert_almost_equal(inv_psf.flux, 1./psf.flux)
    # This doesn't really have any meaning, but this is what we've assigned to a deconvolve maxSB.
    np.testing.assert_almost_equal(inv_psf.maxSB(), -psf.maxSB() / psf.flux**2)

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

    np.testing.assert_equal(inv_obj.centroid(), -obj.centroid())
    np.testing.assert_almost_equal(inv_obj.getFlux(), 1./obj.flux)
    np.testing.assert_almost_equal(inv_obj.flux, 1./obj.flux)
    np.testing.assert_almost_equal(inv_obj.maxSB(), -obj.maxSB() / obj.flux**2)

    check_basic(inv_obj, "Deconvolve(asym)", do_x=False)

    # Check picklability
    do_pickle(inv_obj)
    do_pickle(inv_obj.SBProfile)

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

    np.testing.assert_equal(transformed_inv_obj.centroid(), -transformed_obj.centroid())
    np.testing.assert_almost_equal(transformed_inv_obj.getFlux(), 1./transformed_obj.flux)
    np.testing.assert_almost_equal(transformed_inv_obj.flux, 1./transformed_obj.flux)
    np.testing.assert_almost_equal(transformed_inv_obj.maxSB(),
                                   -transformed_obj.maxSB() / transformed_obj.flux**2)

    check_basic(transformed_inv_obj, "transformed Deconvolve(asym)", do_x=False)

    # Check picklability
    do_pickle(transformed_inv_obj)
    do_pickle(transformed_inv_obj.SBProfile)

    # Should raise an exception for invalid arguments
    try:
        np.testing.assert_raises(TypeError, galsim.Deconvolve)
        np.testing.assert_raises(TypeError, galsim.Deconvolve, myImg1)
        np.testing.assert_raises(TypeError, galsim.Deconvolve, [psf])
        np.testing.assert_raises(TypeError, galsim.Deconvolve, psf, psf)
        np.testing.assert_raises(TypeError, galsim.Deconvolve, psf, real_space=False)
        np.testing.assert_raises(TypeError, galsim.Deconvolution)
        np.testing.assert_raises(TypeError, galsim.Deconvolution, myImg1)
        np.testing.assert_raises(TypeError, galsim.Deconvolution, [psf])
        np.testing.assert_raises(TypeError, galsim.Deconvolution, psf, psf)
        np.testing.assert_raises(TypeError, galsim.Deconvolution, psf, real_space=False)
    except ImportError:
        pass


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
    np.testing.assert_equal(conv2.centroid(), cen)
    np.testing.assert_almost_equal(conv2.getFlux(), psf.flux**2)
    np.testing.assert_almost_equal(conv2.flux, psf.flux**2)
    np.testing.assert_array_less(conv2.xValue(cen), conv2.maxSB())

    # Check picklability
    do_pickle(conv2.SBProfile, lambda x: (repr(x.getObj()), x.isRealSpace(), x.getGSParams()))
    do_pickle(conv2, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(conv2)
    do_pickle(conv2.SBProfile)

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
    np.testing.assert_equal(conv2.centroid(), cen)
    np.testing.assert_almost_equal(conv2.getFlux(), psf.flux**2)
    np.testing.assert_almost_equal(conv2.flux, psf.flux**2)
    np.testing.assert_array_less(conv2.xValue(cen), conv2.maxSB())

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

    cen = 2. * add.centroid()
    np.testing.assert_equal(autoconv.centroid(), cen)
    np.testing.assert_almost_equal(autoconv.getFlux(), add.flux**2)
    np.testing.assert_almost_equal(autoconv.flux, add.flux**2)
    np.testing.assert_array_less(autoconv.xValue(cen), autoconv.maxSB())

    check_basic(autoconv, "AutoConvolve(asym)")

    # Should raise an exception for invalid arguments
    try:
        np.testing.assert_raises(TypeError, galsim.AutoConvolve)
        np.testing.assert_raises(TypeError, galsim.AutoConvolve, myImg1)
        np.testing.assert_raises(TypeError, galsim.AutoConvolve, [psf])
        np.testing.assert_raises(TypeError, galsim.AutoConvolve, psf, psf)
        np.testing.assert_raises(TypeError, galsim.AutoConvolve, psf, realspace=False)
        np.testing.assert_raises(TypeError, galsim.AutoConvolution)
        np.testing.assert_raises(TypeError, galsim.AutoConvolution, myImg1)
        np.testing.assert_raises(TypeError, galsim.AutoConvolution, [psf])
        np.testing.assert_raises(TypeError, galsim.AutoConvolution, psf, psf)
        np.testing.assert_raises(TypeError, galsim.AutoConvolution, psf, realspace=False)
    except ImportError:
        pass


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
    do_pickle(corr.SBProfile, lambda x: (repr(x.getObj()), x.isRealSpace(), x.getGSParams()))
    do_pickle(corr, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(corr)
    do_pickle(corr.SBProfile)

    # Should raise an exception for invalid arguments
    try:
        np.testing.assert_raises(TypeError, galsim.AutoCorrelate)
        np.testing.assert_raises(TypeError, galsim.AutoCorrelate, myImg1)
        np.testing.assert_raises(TypeError, galsim.AutoCorrelate, [obj1])
        np.testing.assert_raises(TypeError, galsim.AutoCorrelate, obj1, obj2)
        np.testing.assert_raises(TypeError, galsim.AutoCorrelate, obj1, realspace=False)
        np.testing.assert_raises(TypeError, galsim.AutoCorrelation)
        np.testing.assert_raises(TypeError, galsim.AutoCorrelation, myImg1)
        np.testing.assert_raises(TypeError, galsim.AutoCorrelation, [obj1])
        np.testing.assert_raises(TypeError, galsim.AutoCorrelation, obj1, obj2)
        np.testing.assert_raises(TypeError, galsim.AutoCorrelation, obj1, realspace=False)
    except ImportError:
        pass

@timer
def test_ne():
    """ Check that inequality works as expected."""
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)
    gal1 = galsim.Gaussian(fwhm=1)
    gal2 = galsim.Gaussian(fwhm=2)

    # Sum.  Params are objs to add and potentially gsparams.
    gals = [galsim.Sum(gal1),
            galsim.Sum(gal1, gal2),
            galsim.Sum(gal2, gal1),  # Not! commutative.  (but is associative)
            galsim.Sum(galsim.Sum(gal1, gal2), gal2),
            galsim.Sum(gal1, gsparams=gsp)]
    all_obj_diff(gals)

    # Convolution.  Params are objs to convolve and potentially gsparams.
    # The following should test unequal
    gals = [galsim.Convolution(gal1),
            galsim.Convolution(gal1, gal2),
            galsim.Convolution(gal2, gal1),  # Not! commutative.
            galsim.Convolution(gal1, gal2, real_space=True),
            galsim.Convolution(galsim.Convolution(gal1, gal2), gal2),
            galsim.Convolution(gal1, galsim.Convolution(gal2, gal2)),  # Not! associative.
            galsim.Convolution(gal1, gsparams=gsp)]
    all_obj_diff(gals)

    # Deconvolution.  Only params here are obj to deconvolve and gsparams.
    gals = [galsim.Deconvolution(gal1),
            galsim.Deconvolution(gal2),
            galsim.Deconvolution(gal1, gsparams=gsp)]
    all_obj_diff(gals)

    # AutoConvolution.  Only params here are obj to deconvolve and gsparams.
    gals = [galsim.AutoConvolution(gal1),
            galsim.AutoConvolution(gal2),
            galsim.AutoConvolution(gal1, gsparams=gsp)]
    all_obj_diff(gals)

    # AutoCorrelation.  Only params here are obj to deconvolve and gsparams.
    gals = [galsim.AutoCorrelation(gal1),
            galsim.AutoCorrelation(gal2),
            galsim.AutoCorrelation(gal1, gsparams=gsp)]
    all_obj_diff(gals)


@timer
def test_fourier_sqrt():
    """Test that the FourierSqrt operator is the inverse of auto-convolution.
    """
    dx = 0.4
    myImg1 = galsim.ImageF(80,80, scale=dx)
    myImg1.setCenter(0,0)
    myImg2 = galsim.ImageF(80,80, scale=dx)
    myImg2.setCenter(0,0)

    # Test trivial case, where we could (but don't) analytically collapse the
    # chain of SBProfiles by recognizing that FourierSqrt is the inverse of
    # AutoConvolve.
    psf = galsim.Moffat(beta=3.8, fwhm=1.3, flux=5)
    psf.drawImage(myImg1, method='no_pixel')
    sqrt1 = galsim.FourierSqrt(psf)
    psf2 = galsim.AutoConvolve(sqrt1)
    np.testing.assert_almost_equal(psf.stepK(), psf2.stepK())
    psf2.drawImage(myImg2, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Moffat sqrt convolved with self disagrees with original")

    check_basic(sqrt1, "FourierSqrt", do_x=False)

    # Test non-trivial case where we compare (in Fourier space) sqrt(a*a + b*b + 2*a*b) against (a + b)
    a = galsim.Moffat(beta=3.8, fwhm=1.3, flux=5)
    a.shift(dx=0.5, dy=-0.3)  # need nonzero centroid to test centroid()
    b = galsim.Moffat(beta=2.5, fwhm=1.6, flux=3)
    check = galsim.Sum([a, b])
    sqrt = galsim.FourierSqrt(
        galsim.Sum([
            galsim.AutoConvolve(a),
            galsim.AutoConvolve(b),
            2*galsim.Convolve([a, b])
        ])
    )
    np.testing.assert_almost_equal(check.stepK(), sqrt.stepK())
    check.drawImage(myImg1, method='no_pixel')
    sqrt.drawImage(myImg2, method='no_pixel')
    np.testing.assert_almost_equal(check.centroid().x, sqrt.centroid().x)
    np.testing.assert_almost_equal(check.centroid().y, sqrt.centroid().y)
    np.testing.assert_almost_equal(check.getFlux(), sqrt.getFlux())
    np.testing.assert_almost_equal(check.xValue(check.centroid()), check.maxSB())
    print('check.maxSB = ',check.maxSB())
    print('sqrt.maxSB = ',sqrt.maxSB())
    # This isn't super accurate...
    np.testing.assert_allclose(check.maxSB(), sqrt.maxSB(), rtol=0.1)
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Fourier square root of expanded square disagrees with original")

    # Check picklability
    do_pickle(sqrt1.SBProfile, lambda x: (repr(x.getObj()), x.getGSParams()))
    do_pickle(sqrt1, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(sqrt1)
    do_pickle(sqrt1.SBProfile)

    # Should raise an exception for invalid arguments
    try:
        np.testing.assert_raises(TypeError, galsim.FourierSqrt)
        np.testing.assert_raises(TypeError, galsim.FourierSqrt, myImg1)
        np.testing.assert_raises(TypeError, galsim.FourierSqrt, [psf])
        np.testing.assert_raises(TypeError, galsim.FourierSqrt, psf, psf)
        np.testing.assert_raises(TypeError, galsim.FourierSqrt, psf, real_space=False)
        np.testing.assert_raises(TypeError, galsim.FourierSqrtProfile)
        np.testing.assert_raises(TypeError, galsim.FourierSqrtProfile, myImg1)
        np.testing.assert_raises(TypeError, galsim.FourierSqrtProfile, [psf])
        np.testing.assert_raises(TypeError, galsim.FourierSqrtProfile, psf, psf)
        np.testing.assert_raises(TypeError, galsim.FourierSqrtProfile, psf, real_space=False)
    except ImportError:
        pass

@timer
def test_sum_transform():
    """This test addresses a bug found by Ismael Serrano, #763, wherein some attributes
    got messed up for a Transform(Sum(Transform())) object.

    The bug was that we didn't bother to make a new SBProfile for a Sum (or Convolve) of
    a single object.  But if that was an SBTransform, then the next Transform operation
    combined the two Transforms, which messed up the repr.

    The fix is to always make an SBAdd or SBConvolve object even if the list of things to add
    or convolve only has one element.
    """
    gal0 = galsim.Exponential(scale_radius=0.34, flux=105.).shear(g1=-0.56,g2=0.15)

    for gal1 in [ galsim.Sum(gal0), galsim.Convolve(gal0) ]:
        gal2 = gal1.dilate(1)

        sgal1 = eval(str(gal1))
        rgal1 = eval(repr(gal1))
        sgal2 = eval(str(gal2))
        rgal2 = eval(repr(gal2))

        print('gal1 = ',repr(gal1))
        print('sgal1 = ',repr(sgal1))
        print('rgal1 = ',repr(rgal1))

        print('gal2 = ',repr(gal2))
        print('sgal2 = ',repr(sgal2))
        print('rgal2 = ',repr(rgal2))

        gal1_im = gal1.drawImage(nx=64, ny=64, scale=0.2)
        sgal1_im = sgal1.drawImage(nx=64, ny=64, scale=0.2)
        rgal1_im = rgal1.drawImage(nx=64, ny=64, scale=0.2)

        gal2_im = gal2.drawImage(nx=64, ny=64, scale=0.2)
        sgal2_im = sgal2.drawImage(nx=64, ny=64, scale=0.2)
        rgal2_im = rgal2.drawImage(nx=64, ny=64, scale=0.2)

        # Check that the objects are equivalent, even if they may be written differently.
        np.testing.assert_almost_equal(gal1_im.array, sgal1_im.array, decimal=8)
        np.testing.assert_almost_equal(gal1_im.array, rgal1_im.array, decimal=8)

        # These two used to fail.
        np.testing.assert_almost_equal(gal2_im.array, sgal2_im.array, decimal=8)
        np.testing.assert_almost_equal(gal2_im.array, rgal2_im.array, decimal=8)

        do_pickle(gal0)
        do_pickle(gal1)
        do_pickle(gal2)  # And this.

@timer
def test_compound_noise():
    """Test that noise propagation works properly for compount objects.
    """
    obj1 = galsim.Gaussian(sigma=1.7)
    obj2 = galsim.Gaussian(sigma=2.3)
    obj1.noise = galsim.UncorrelatedNoise(variance=0.3, scale=0.2)
    obj2.noise = galsim.UncorrelatedNoise(variance=0.5, scale=0.2)
    obj3 = galsim.Gaussian(sigma=2.9)

    # Sum adds the variance of the components
    sum2 = galsim.Sum([obj1,obj2])
    np.testing.assert_almost_equal(sum2.noise.getVariance(), 0.8,
            err_msg = "Sum of two objects did not add noise varinace")
    sum2 = galsim.Sum([obj1,obj3])
    np.testing.assert_almost_equal(sum2.noise.getVariance(), 0.3,
            err_msg = "Sum of two objects did not add noise varinace")
    sum2 = galsim.Sum([obj2,obj3])
    np.testing.assert_almost_equal(sum2.noise.getVariance(), 0.5,
            err_msg = "Sum of two objects did not add noise varinace")
    sum3 = galsim.Sum([obj1,obj2,obj3])
    np.testing.assert_almost_equal(sum3.noise.getVariance(), 0.8,
            err_msg = "Sum of three objects did not add noise varinace")
    sum3 = obj1 + obj2 + obj3
    np.testing.assert_almost_equal(sum3.noise.getVariance(), 0.8,
            err_msg = "Sum of three objects did not add noise varinace")

    # Adding noise objects with different WCSs will raise a warning.
    obj4 = galsim.Gaussian(sigma=3.3)
    obj4.noise = galsim.UncorrelatedNoise(variance=0.3, scale=0.8)
    try:
        np.testing.assert_warns(UserWarning, galsim.Sum, [obj1, obj4])
    except:
        pass

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
    try:
        conv2 = np.testing.assert_warns(UserWarning, galsim.Convolution, [obj1, obj2])
        conv3 = np.testing.assert_warns(UserWarning, galsim.Convolution, [obj1, obj2, obj3])
        # Other types don't propagate noise and give a warning about it.
        np.testing.assert_warns(UserWarning, galsim.Deconvolve, obj1)
        np.testing.assert_warns(UserWarning, galsim.AutoConvolve, obj1)
        np.testing.assert_warns(UserWarning, galsim.AutoCorrelate, obj1)
        np.testing.assert_warns(UserWarning, galsim.FourierSqrt, obj1)
    except:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conv2 = galsim.Convolution([obj1,obj2])
            conv3 = galsim.Convolution([obj1,obj2,obj3])

    obj2.noise = None  # Remove obj2 noise for the rest.
    noise = galsim.Convolve([obj1.noise._profile, obj2, obj2])
    variance = noise.drawImage(nx=1, ny=1, scale=1., method='sb')(1,1)
    np.testing.assert_almost_equal(conv2.noise.getVariance(), variance,
            err_msg = "Convolution of two objects did not correctly propagate noise varinace")
    noise = galsim.Convolve([obj1.noise._profile, obj2, obj2, obj3, obj3])
    variance = noise.drawImage(nx=1, ny=1, scale=1., method='sb')(1,1)
    np.testing.assert_almost_equal(conv3.noise.getVariance(), variance,
            err_msg = "Convolution of three objects did not correctly propagate noise varinace")


if __name__ == "__main__":
    test_convolve()
    test_convolve_flux_scaling()
    test_shearconvolve()
    test_realspace_convolve()
    test_realspace_distorted_convolve()
    test_realspace_shearconvolve()
    test_add()
    test_sub_neg()
    test_add_flux_scaling()
    test_deconvolve()
    test_autoconvolve()
    test_autocorrelate()
    test_ne()
    test_fourier_sqrt()
    test_sum_transform()
    test_compound_noise()
