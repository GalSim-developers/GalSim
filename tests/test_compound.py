# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

    # Test photon shooting.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        do_shoot(conv,myImg,"Moffat * Pixel")


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

    # Test photon shooting.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        do_shoot(conv,myImg,"sheared Gaussian * Pixel")


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

    # Test kvalues
    do_kvalue(conv,img,"Truncated Moffat convolved with Box")

    # Check picklability
    do_pickle(conv.SBProfile, lambda x: (repr(x.getObjs()), x.isRealSpace(), x.getGSParams()))
    do_pickle(conv, lambda x: x.drawImage(method='sb'))
    do_pickle(conv)
    do_pickle(conv.SBProfile)


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
    sum = galsim.Add(gauss1,gauss2)
    sum.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) disagrees with expected result")

    # Check with default_params
    sum = galsim.Add(gauss1,gauss2,gsparams=default_params)
    sum.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) with default_params disagrees with "
            "expected result")
    sum = galsim.Add(gauss1,gauss2,gsparams=galsim.GSParams())
    sum.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the sum:
    sum = gauss1 + gauss2
    sum.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject gauss1 + gauss2 disagrees with expected result")
    sum = gauss1
    sum += gauss2
    sum.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum = gauss1; sum += gauss2 disagrees with expected result")
    sum = galsim.Add([gauss1,gauss2])
    sum.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add([gauss1,gauss2]) disagrees with expected result")
    gauss1 = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = galsim.Gaussian(flux=1, sigma=3)
    sum = 0.75 * gauss1 + 0.25 * gauss2
    sum.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject 0.75 * gauss1 + 0.25 * gauss2 disagrees with expected result")
    sum = 0.75 * gauss1
    sum += 0.25 * gauss2
    sum.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum += 0.25 * gauss2 disagrees with expected result")

    # Test photon shooting.
    do_shoot(sum,myImg,"sum of 2 Gaussians")

    # Test kvalues
    do_kvalue(sum,myImg,"sum of 2 Gaussians")

    # Check picklability
    do_pickle(sum.SBProfile, lambda x: (repr(x.getObjs()), x.getGSParams()))
    do_pickle(sum, lambda x: x.drawImage(method='sb'))
    do_pickle(sum)
    do_pickle(sum.SBProfile)


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

    # Test photon shooting.
    do_shoot(conv2,myImg2,"AutoConvolve(Moffat)")

    # Also check AutoConvolve with an asymmetric profile.
    # (AutoCorrelate with this profile is done below...)
    obj1 = galsim.Gaussian(sigma=3., flux=4).shift(-0.2, -0.4)
    obj2 = galsim.Gaussian(sigma=6., flux=1.3).shift(0.3, 0.3)
    add = galsim.Add(obj1, obj2)
    conv = galsim.Convolve([add, add])
    conv.drawImage(myImg1, method='no_pixel')
    corr = galsim.AutoConvolve(add)
    corr.drawImage(myImg2, method='no_pixel')
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Asymmetric sum of Gaussians convolved with self disagrees with "+
            "AutoConvolve result")

    # Check picklability
    do_pickle(conv2.SBProfile, lambda x: (repr(x.getObj()), x.isRealSpace(), x.getGSParams()))
    do_pickle(conv2, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(conv2)
    do_pickle(conv2.SBProfile)


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

    # Test photon shooting.
    do_shoot(corr,myImg2,"AutoCorrelate")

    # Check picklability
    do_pickle(corr.SBProfile, lambda x: (repr(x.getObj()), x.isRealSpace(), x.getGSParams()))
    do_pickle(corr, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(corr)
    do_pickle(corr.SBProfile)


@timer
def test_ne():
    """ Check that inequality works as expected."""
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)
    gal1 = galsim.Gaussian(fwhm=1)
    gal2 = galsim.Gaussian(fwhm=2)

    # Sum.  Params are objs to add and potentially gsparams.
    gals = [galsim.Sum(gal1),
            galsim.Sum(gal1, gal2),
            galsim.Sum(gal2, gal1),  # Not! commutative.
            galsim.Sum(galsim.Sum(gal1, gal2), gal2),
            galsim.Sum(gal1, galsim.Sum(gal2, gal2)),  # Not! associative.
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


def test_fourier_sqrt():
    """Test that the FourierSqrt operator is the inverse of auto-convolution.
    """
    import time
    t1 = time.time()

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
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Fourier square root of expanded square disagrees with original")

    # Check picklability
    do_pickle(sqrt1.SBProfile, lambda x: (repr(x.getObj()), x.getGSParams()))
    do_pickle(sqrt1, lambda x: x.drawImage(method='no_pixel'))
    do_pickle(sqrt1)
    do_pickle(sqrt1.SBProfile)

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))

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
    test_autoconvolve()
    test_autocorrelate()
    test_ne()
    test_fourier_sqrt()
    test_sum_transform()
