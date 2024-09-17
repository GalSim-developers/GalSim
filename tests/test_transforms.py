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

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images.

# for flux normalization tests
test_flux = 1.8

# Some parameters used in the two unit tests test_integer_shift_fft and test_integer_shift_photon:
test_sigma = 1.8
test_hlr = 1.8
int_shift_x = 7
int_shift_y = 3
n_pix_x = 50
n_pix_y = 60
delta_sub = 30
image_decimal_precise = 15


@timer
def test_smallshear():
    """Test the application of a small shear to a Gaussian profile against a known result.
    """
    e1 = 0.02
    e2 = 0.02
    myShear = galsim.Shear(e1=e1, e2=e2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = gauss.shear(myShear)
    gauss2.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.max(), gauss2.max_sb, 5,
            err_msg="sheared profile max_sb did not match maximum pixel value")

    # Check with default_params
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=default_params)
    gauss = gauss.shear(myShear)
    gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear with default_params disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=galsim.GSParams())
    gauss = gauss._shear(myShear)
    gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear with GSParams() disagrees with expected result")

    check_basic(gauss, "sheared Gaussian")

    # Test photon shooting.
    do_shoot(gauss,myImg,"sheared Gaussian")

    # Test kvalues
    do_kvalue(gauss,myImg,"sheared Gaussian")

    # Check picklability
    check_pickle(gauss, lambda x: x.drawImage())
    check_pickle(gauss)

    # Check really small shear  (This mostly tests a branch in the str function.)
    check_pickle(galsim.Gaussian(sigma=2.3).shear(g1=1.e-13,g2=0))

    assert_raises(TypeError, gauss.shear)
    assert_raises(TypeError, gauss.shear, 0.3)
    assert_raises(TypeError, gauss.shear, 0.1, 0.3)
    assert_raises(TypeError, gauss.shear, g1=0.1, g2=0.1, invalid=0.3)
    assert_raises(TypeError, gauss.shear, myShear, invalid=0.3)


@timer
def test_largeshear():
    """Test the application of a large shear to a Sersic profile against a known result.
    """
    e1 = 0.0
    e2 = 0.5

    myShear = galsim.Shear(e1=e1, e2=e2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_largeshear.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc2 = devauc.shear(myShear)
    devauc2.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.max(), devauc2.max_sb, 5,
            err_msg="sheared profile max_sb did not match maximum pixel value")

    # Check with default_params
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1, gsparams=default_params)
    devauc = devauc.shear(myShear)
    devauc.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear with default_params disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    devauc = devauc._shear(myShear)
    devauc.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear with GSParams() disagrees with expected result")

    # Convolve with a small gaussian to smooth out the central peak.
    devauc2 = galsim.Convolve(devauc, galsim.Gaussian(sigma=0.3))
    check_basic(devauc2, "sheared DeVauc")

    # Test photon shooting.
    do_shoot(devauc2,myImg,"sheared DeVauc")

    # Test kvalues.
    # Testing a sheared devauc requires a rather large fft.  What we really care about
    # testing though is the accuracy of the shear function.  So just shear a Gaussian here.
    gauss = galsim.Gaussian(sigma=2.3)
    gauss = gauss.shear(myShear)
    do_kvalue(gauss,myImg, "sheared Gaussian")

    # Check picklability
    check_pickle(gauss, lambda x: x.drawImage())
    check_pickle(gauss)


@timer
def test_rotate():
    """Test the 45 degree rotation of a sheared Sersic profile against a known result.
    """
    myShear = galsim.Shear(e1=0.2, e2=0.0)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_ellip_rotated.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1)
    gal = gal.shear(myShear).rotate(45.0 * galsim.degrees)
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject rotate disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.max(), gal.max_sb, 5,
            err_msg="rotated profile max_sb did not match maximum pixel value")

    # Check with default_params
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1, gsparams=default_params)
    gal = gal.shear(myShear).rotate(45.0 * galsim.degrees)
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject rotate with default_params disagrees with expected "
            "result")
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    gal = gal._shear(myShear).rotate(45.0 * galsim.degrees)
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject rotate with GSParams() disagrees with expected result")

    # Convolve with a small gaussian to smooth out the central peak.
    gal2 = galsim.Convolve(gal, galsim.Gaussian(sigma=0.3))
    check_basic(gal2, "rotated sheared Sersic")

    # Test photon shooting.
    do_shoot(gal2,myImg,"rotated sheared Sersic")

    # Test kvalues
    do_kvalue(gal,myImg,"rotated sheared Sersic")

    # Check picklability
    check_pickle(gal, lambda x: x.drawImage())
    check_pickle(gal)

    assert_raises(TypeError, gal.rotate)
    assert_raises(TypeError, gal.rotate, 34)


@timer
def test_mag():
    """Test the magnification (size x 1.5) of an exponential profile against a known result.
    """
    re = 1.0
    r0 = re/1.67839
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_mag.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal = gal.dilate(1.5)
    gal *= 1.5**2 # Apply the flux magnification.
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject dilate disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.max(), gal.max_sb, 5,
            err_msg="dilated profile max_sb did not match maximum pixel value")

    # Check with default_params
    gal = galsim.Exponential(flux=1, scale_radius=r0, gsparams=default_params)
    gal = gal.dilate(1.5)
    gal *= 1.5**2
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject dilate with default_params disagrees with expected result")
    gal = galsim.Exponential(flux=1, scale_radius=r0, gsparams=galsim.GSParams())
    gal = gal.dilate(1.5)
    gal *= 1.5**2
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject dilate with GSParams() disagrees with expected result")

    # Use magnify
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal = gal.magnify(1.5**2) # area rescaling factor
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject magnify disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.max(), gal.max_sb, 5,
            err_msg="magnified profile max_sb did not match maximum pixel value")

    # Use lens
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal = gal.lens(0., 0., 1.5**2) # area rescaling factor
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject lens disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.max(), gal.max_sb, 5,
            err_msg="lensed profile max_sb did not match maximum pixel value")

    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal = gal.magnify(1.5**2) # area rescaling factor
    check_basic(gal, "dilated Exponential")

    # Test photon shooting.
    do_shoot(gal,myImg,"dilated Exponential")

    # Test kvalues
    do_kvalue(gal,myImg,"dilated Exponential")

    # Check picklability
    check_pickle(gal, lambda x: x.drawImage())
    check_pickle(gal)


@timer
def test_lens():
    """Test the lensing (shear, magnification) of a Sersic profile carried out 2 ways.
    """
    re = 1.0
    n = 3.
    g1 = 0.12
    g2 = -0.4
    mu = 1.2
    pix_scale = 0.1
    imsize = 100
    ser = galsim.Sersic(n, half_light_radius = re)

    ser1 = ser.shear(g1=g1, g2=g2).magnify(mu)
    im1 = galsim.ImageF(imsize, imsize, scale=pix_scale)
    im1 = ser1.drawImage(im1, method='no_pixel')

    ser2 = ser.lens(g1, g2, mu)
    im2 = galsim.ImageF(imsize, imsize, scale=pix_scale)
    im2 = ser2.drawImage(im2, method='no_pixel')
    np.testing.assert_array_almost_equal(im2.array, im1.array, 5,
        err_msg="Lensing of Sersic profile done in two different ways gives different answer")

    # _lens is equivalent in this case.
    ser3 = ser._lens(g1, g2, mu)
    im3 = galsim.ImageF(imsize, imsize, scale=pix_scale)
    im3 = ser3.drawImage(im3, method='no_pixel')
    np.testing.assert_array_almost_equal(im3.array, im1.array, 5,
        err_msg="Lensing of Sersic profile with _lens gives different answer")


@timer
def test_shift():
    """Test the translation of a Box profile against a known result.
    """
    dx = 0.2
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_shift.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    pixel = galsim.Pixel(scale=dx)
    pixel = pixel.shift(dx, -dx)
    pixel.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shift disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.max(), pixel.max_sb, 5,
            err_msg="shifted profile max_sb did not match maximum pixel value")

    # Check with default_params
    pixel = galsim.Pixel(scale=dx, gsparams=default_params)
    pixel = pixel.shift(dx, -dx)
    pixel.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shift with default_params disagrees with expected result")
    pixel = galsim.Pixel(scale=dx, gsparams=galsim.GSParams())
    pixel = pixel.shift(dx, -dx)
    pixel.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shift with GSParams() disagrees with expected result")

    check_basic(pixel, "shifted Box")

    # Test photon shooting.
    do_shoot(pixel,myImg,"shifted Box")

    # Test kvalues.
    # Testing a shifted box requires a ridiculously large fft.  What we really care about
    # testing though is the accuracy of the shift function.  So just shift a Gaussian here.
    gauss = galsim.Gaussian(sigma=2.3)
    gauss = gauss.shift(dx,-dx)
    do_kvalue(gauss,myImg, "shifted Gaussian")

    # Check picklability
    check_pickle(gauss, lambda x: x.drawImage())
    check_pickle(gauss)


@timer
def test_rescale():
    """Test the flux rescaling of a Sersic profile against a known result.
    """
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_doubleflux.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.withFlux(2).drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject withFlux disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.max(), sersic.withFlux(2).max_sb, 5,
            err_msg="rescaled profile max_sb did not match maximum pixel value")

    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic *= 2
    sersic.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject *= 2 disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic2 = sersic * 2
    sersic2.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject obj * 2 disagrees with expected result")
    sersic2 = 2 * sersic
    sersic2.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject 2 * obj disagrees with expected result")

    # Check with default_params
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=default_params)
    sersic *= 2
    sersic.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject *= 2 with default_params disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    sersic *= 2
    sersic.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject *= 2 with GSParams() disagrees with expected result")

    # Can also get a flux of 2 by drawing flux=1 twice with add_to_image=True
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    sersic.drawImage(myImg,scale=dx, method="sb",add_to_image=True,
                use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Drawing with add_to_image=True disagrees with expected result")

    # With lower folding_threshold and maxk_threshold, the calculated flux should come out right
    # so long as we also convolve by a pixel:
    gsp1 = galsim.GSParams(folding_threshold=1.e-3, maxk_threshold=5.e-4)
    sersic_acc = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=gsp1)
    myImg2 = sersic_acc.drawImage(scale=dx, use_true_center=False)
    print(myImg2.array.sum(), myImg2.added_flux)
    np.testing.assert_almost_equal(myImg2.array.sum(), 1., 3,
            err_msg="Drawing with gsp1 results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux, 1., 3,
            err_msg="Drawing with gsp1 returned wrong added_flux")
    myImg2 = sersic_acc.drawImage(myImg2, add_to_image=True, use_true_center=False)
    print(myImg2.array.sum(), myImg2.added_flux)
    np.testing.assert_almost_equal(myImg2.array.sum(), 2., 3,
            err_msg="Drawing with add_to_image=True results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux, 1., 3,
            err_msg="Drawing with add_to_image=True returned wrong added_flux")

    # Check that the flux works out when adding multiple times.
    # With a Gaussian, we can take the thresholds even lower and get another digit of accuracy.
    gsp2 = galsim.GSParams(folding_threshold=1.e-5, maxk_threshold=1.e-5)
    gauss = galsim.Gaussian(flux=1.e5, sigma=2., gsparams=gsp2)
    myImg2 = gauss.drawImage(scale=dx, use_true_center=False)
    print('image size = ',myImg2.array.shape)
    print(myImg2.array.sum(), myImg2.added_flux)
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 1., 4,
            err_msg="Drawing Gaussian results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian returns wrong added_flux")
    myImg2 = gauss.drawImage(myImg2, add_to_image=True, use_true_center=False)
    print(myImg2.array.sum(), myImg2.added_flux)
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 2., 4,
            err_msg="Drawing Gaussian with add_to_image=True results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian with add_to_image=True returns wrong added_flux")
    rng = galsim.BaseDeviate(12345)
    myImg2 = gauss.drawImage(myImg2, add_to_image=True, poisson_flux=False, rng=rng, method='phot')
    print(myImg2.array.sum(), myImg2.added_flux)
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 3., 4,
            err_msg="Drawing Gaussian with method=phot, add_to_image=True, poisson_flux=False "+
                    "results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian with method=phot, add_to_image=True, poisson_flux=False "+
                    "returned wrong added_flux")
    myImg2 = gauss.drawImage(myImg2, add_to_image=True, rng=rng, method='phot')
    print(myImg2.array.sum(), myImg2.added_flux)
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 4., 1,
            err_msg="Drawing Gaussian with method=phot, add_to_image=True, poisson_flux=True "+
                    "results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 1,
            err_msg="Drawing Gaussian with method=phot, add_to_image=True, poisson_flux=True "+
                    "returned wrong added_flux")
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 3.+myImg2.added_flux/1.e5, 4,
            err_msg="Drawing Gaussian with method=phot, add_to_image=True results in wrong flux "+
                    "according to the returned added_flux")

    # Can also get a flux of 2 using gain = 0.5
    sersic.drawImage(myImg, scale=dx, gain=0.5, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Drawing with gain=0.5 disagrees with expected result")
    myImg2 = sersic_acc.drawImage(scale=dx, gain=0.5, use_true_center=False)
    np.testing.assert_almost_equal(myImg2.array.sum(), 2., 3,
            err_msg="Drawing with gain=0.5 results in wrong flux")
    myImg2 = sersic_acc.drawImage(scale=dx, gain=4., use_true_center=False)
    np.testing.assert_almost_equal(myImg2.array.sum(), 0.25, 3,
            err_msg="Drawing with gain=4. results in wrong flux")
    # Check add_to_image in conjunction with gain
    sersic_acc.drawImage(myImg2, gain=4., add_to_image=True, use_true_center=False)
    np.testing.assert_almost_equal(myImg2.array.sum(), 0.5, 3,
            err_msg="Drawing with gain=4. results in wrong flux")

    # Convolve with a small gaussian to smooth out the central peak.
    sersic_smooth = galsim.Convolve(sersic2, galsim.Gaussian(sigma=0.3))
    check_basic(sersic_smooth, "scaled Sersic")

    # Test photon shooting.
    do_shoot(sersic_smooth,myImg,"scaled Sersic")

    # Test kvalues
    do_kvalue(sersic2,myImg, "scaled Sersic")

    # Check picklability
    check_pickle(sersic2, lambda x: x.drawImage())
    check_pickle(sersic2)


@timer
def test_integer_shift_fft():
    """Test if shift works correctly for integer shifts using drawImage method.
    """
    gal = galsim.Gaussian(sigma=test_sigma)
    psf = galsim.Airy(lam_over_diam=test_hlr)

    # shift galaxy only

    final=galsim.Convolve([gal, psf])
    img_center = galsim.ImageD(n_pix_x,n_pix_y)
    final.drawImage(img_center,scale=1)

    gal = gal.shift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    final.drawImage(img_shift,scale=1)

    sub_center = img_center.array[
        (n_pix_y - delta_sub) // 2 : (n_pix_y + delta_sub) // 2,
        (n_pix_x - delta_sub) // 2 : (n_pix_x + delta_sub) // 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) // 2  + int_shift_y : (n_pix_y + delta_sub) // 2  + int_shift_y,
        (n_pix_x - delta_sub) // 2  + int_shift_x : (n_pix_x + delta_sub) // 2  + int_shift_x]

    np.testing.assert_array_almost_equal(
        sub_center, sub_shift, decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with shifted Galaxy only")

    # shift PSF only

    gal = galsim.Gaussian(sigma=test_sigma)
    psf = psf.shift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    final.drawImage(img_shift,scale=1)

    sub_center = img_center.array[
        (n_pix_y - delta_sub) // 2 : (n_pix_y + delta_sub) // 2,
        (n_pix_x - delta_sub) // 2 : (n_pix_x + delta_sub) // 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) // 2  + int_shift_y : (n_pix_y + delta_sub) // 2  + int_shift_y,
        (n_pix_x - delta_sub) // 2  + int_shift_x : (n_pix_x + delta_sub) // 2  + int_shift_x]
    np.testing.assert_array_almost_equal(
        sub_center, sub_shift,  decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with only PSF shifted ")


@timer
def test_integer_shift_photon():
    """Test if shift works correctly for integer shifts using method=phot.
    """
    n_photons_low = 10
    seed = 10

    gal = galsim.Gaussian(sigma=test_sigma)
    psf = galsim.Airy(lam_over_diam=test_hlr)

    # shift galaxy only

    final=galsim.Convolve([gal, psf])
    img_center = galsim.ImageD(n_pix_x,n_pix_y)
    test_deviate = galsim.BaseDeviate(seed)
    final.drawImage(img_center,scale=1,rng=test_deviate,n_photons=n_photons_low, method='phot')

    gal = gal.shift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    test_deviate = galsim.BaseDeviate(seed)
    final.drawImage(img_shift,scale=1,rng=test_deviate,n_photons=n_photons_low, method='phot')

    sub_center = img_center.array[
        (n_pix_y - delta_sub) // 2 : (n_pix_y + delta_sub) // 2,
        (n_pix_x - delta_sub) // 2 : (n_pix_x + delta_sub) // 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) // 2  + int_shift_y : (n_pix_y + delta_sub) // 2  + int_shift_y,
        (n_pix_x - delta_sub) // 2  + int_shift_x : (n_pix_x + delta_sub) // 2  + int_shift_x]


    np.testing.assert_array_almost_equal(
        sub_center, sub_shift, decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with shifted Galaxy only")

    # shift PSF only

    gal = galsim.Gaussian(sigma=test_sigma)
    psf = psf.shift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    test_deviate = galsim.BaseDeviate(seed)
    final.drawImage(img_shift,scale=1,rng=test_deviate,n_photons=n_photons_low, method='phot')

    sub_center = img_center.array[
        (n_pix_y - delta_sub) // 2 : (n_pix_y + delta_sub) // 2,
        (n_pix_x - delta_sub) // 2 : (n_pix_x + delta_sub) // 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) // 2  + int_shift_y : (n_pix_y + delta_sub) // 2  + int_shift_y,
        (n_pix_x - delta_sub) // 2  + int_shift_x : (n_pix_x + delta_sub) // 2  + int_shift_x]
    np.testing.assert_array_almost_equal(
        sub_center, sub_shift,  decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with only PSF shifted ")


@timer
def test_flip(run_slow):
    """Test several ways to flip a profile
    """
    # The Shapelet profile has the advantage of being fast and not circularly symmetric, so
    # it is a good test of the actual code for doing the flips (in SBTransform).
    # But since the bug Rachel reported in #645 was actually in SBInterpolatedImage
    # (one calculation implicitly assumed dx > 0), it seems worthwhile to run through all the
    # classes to make sure we hit everything with negative steps for dx and dy.
    prof_list = [
        galsim.Shapelet(sigma=0.17, order=2,
                        bvec=[1.7, 0.01,0.03, 0.29, 0.33, -0.18]),
    ]
    if run_slow:
        image_dir = './real_comparison_images'
        catalog_file = 'test_catalog.fits'
        rgc = galsim.RealGalaxyCatalog(catalog_file, dir=image_dir)
        # Some of these are slow, so only do the Shapelet test as part of the normal unit tests.
        prof_list += [
            galsim.Airy(lam_over_diam=0.17, flux=1.7),
            galsim.Airy(lam_over_diam=0.17, obscuration=0.2, flux=1.7),
            # Box gets rendered with real-space convolution.  The default accuracy isn't quite
            # enough to get the flip to match at 6 decimal places.
            galsim.Box(0.17, 0.23, flux=1.7,
                       gsparams=galsim.GSParams(realspace_relerr=1.e-6)),
            # Without being convolved by anything with a reasonable k cutoff, this needs
            # a very large fft.
            galsim.DeVaucouleurs(half_light_radius=0.17, flux=1.7),
            # I don't really understand why this needs a lower maxk_threshold to work, but
            # without it, the k-space tests fail.
            galsim.Exponential(scale_radius=0.17, flux=1.7,
                               gsparams=galsim.GSParams(maxk_threshold=1.e-4)),
            galsim.Gaussian(sigma=0.17, flux=1.7),
            galsim.Kolmogorov(fwhm=0.17, flux=1.7),
            galsim.Moffat(beta=2.5, fwhm=0.17, flux=1.7),
            galsim.Moffat(beta=2.5, fwhm=0.17, flux=1.7, trunc=0.82),
            galsim.OpticalPSF(lam_over_diam=0.17, obscuration=0.2, nstruts=6,
                              coma1=0.2, coma2=0.5, defocus=-0.1, flux=1.7),
            # Like with Box, we need to increase the real-space convolution accuracy.
            # This time lowering both relerr and abserr.
            galsim.Pixel(0.23, flux=1.7,
                         gsparams=galsim.GSParams(realspace_relerr=1.e-6,
                                                  realspace_abserr=1.e-8)),
            # Note: RealGalaxy should not be rendered directly because of the deconvolution.
            # Here we convolve it by a Gaussian that is slightly larger than the original PSF.
            galsim.Convolve([ galsim.RealGalaxy(rgc, index=0, flux=1.7),  # "Real" RealGalaxy
                              galsim.Gaussian(sigma=0.08) ]),
            galsim.Convolve([ galsim.RealGalaxy(rgc, index=1, flux=1.7),  # "Fake" RealGalaxy
                              galsim.Gaussian(sigma=0.08) ]),             # (cf. test_real.py)
            galsim.Spergel(nu=-0.19, half_light_radius=0.17, flux=1.7),
            galsim.Spergel(nu=0., half_light_radius=0.17, flux=1.7),
            galsim.Spergel(nu=0.8, half_light_radius=0.17, flux=1.7),
            galsim.Sersic(n=2.3, half_light_radius=0.17, flux=1.7),
            galsim.Sersic(n=2.3, half_light_radius=0.17, flux=1.7, trunc=0.82),
            # The shifts here caught a bug in how SBTransform handled the recentering.
            # Two of the shifts (0.125 and 0.375) lead back to 0.0 happening on an integer
            # index, which now works correctly.
            galsim.Sum([ galsim.Gaussian(sigma=0.17, flux=1.7).shift(-0.2,0.125),
                         galsim.Exponential(scale_radius=0.23, flux=3.1).shift(0.375,0.23)]),
            galsim.TopHat(0.23, flux=1.7),
            # Box and Pixel use real-space convolution.  Convolve with a Gaussian to get fft.
            galsim.Convolve([ galsim.Box(0.17, 0.23, flux=1.7).shift(-0.2,0.1),
                              galsim.Gaussian(sigma=0.09) ]),
            galsim.Convolve([ galsim.TopHat(0.17, flux=1.7).shift(-0.275,0.125),
                              galsim.Gaussian(sigma=0.09) ]),
            # Test something really crazy with several layers worth of transformations
            galsim.Convolve([
                galsim.Sum([
                    galsim.Gaussian(sigma=0.17, flux=1.7).shear(g1=0.1,g2=0.2).shift(2,3),
                    galsim.Kolmogorov(fwhm=0.33, flux=3.9).transform(0.31,0.19,-0.23,0.33) * 88.,
                    galsim.Box(0.11, 0.44, flux=4).rotate(33 * galsim.degrees) / 1.9
                ]).shift(-0.3,1),
                galsim.AutoConvolve(galsim.TopHat(0.5).shear(g1=0.3,g2=0)).rotate(3*galsim.degrees),
                (galsim.AutoCorrelate(galsim.Box(0.2, 0.3)) * 11).shift(3,2).shift(2,-3) * 0.31
            ]).shift(0,0).transform(0,-1,-1,0).shift(-1,1)
        ]

    s = galsim.Shear(g1=0.11, g2=-0.21)
    s1 = galsim.Shear(g1=0.11, g2=0.21)  # Appropriate for the flips around x and y axes
    s2 = galsim.Shear(g1=-0.11, g2=-0.21)  # Appropriate for the flip around x=y

    # Also use shears with just a g1 to get dx != dy, but dxy, dyx = 0.
    q = galsim.Shear(g1=0.11, g2=0.)
    q1 = galsim.Shear(g1=0.11, g2=0.)  # Appropriate for the flips around x and y axes
    q2 = galsim.Shear(g1=-0.11, g2=0.)  # Appropriate for the flip around x=y

    decimal=6  # Oddly, these aren't as precise as I would have expected.
               # Even when we only go to this many digits of accuracy, the Exponential needed
               # a lower than default value for maxk_threshold.
    im = galsim.ImageD(16,16, scale=0.05)

    for prof in prof_list:
        print('prof = ',prof)

        # Not all profiles are expected to have a max_sb value close to the maximum pixel value,
        # so mark the ones where we don't want to require this to be true.
        close_maxsb = True
        name = str(prof)
        if ('DeVauc' in name or 'Sersic' in name or 'Spergel' in name or
            'Optical' in name or 'shift' in name):
            close_maxsb = False

        # Make sure we hit all 4 fill functions.
        # image_x uses fillXValue with izero, jzero
        # image_x1 uses fillXValue with izero, jzero, and unequal dx,dy
        # image_x2 uses fillXValue with dxy, dyx
        # image_k uses fillKValue with izero, jzero
        # image_k1 uses fillKValue with izero, jzero, and unequal dx,dy
        # image_k2 uses fillKValue with dxy, dyx
        image_x = prof.drawImage(image=im.copy(), method='no_pixel')
        image_x1 = prof.shear(q).drawImage(image=im.copy(), method='no_pixel')
        image_x2 = prof.shear(s).drawImage(image=im.copy(), method='no_pixel')
        image_k = prof.drawImage(image=im.copy())
        image_k1 = prof.shear(q).drawImage(image=im.copy())
        image_k2 = prof.shear(s).drawImage(image=im.copy())

        if close_maxsb:
            np.testing.assert_allclose(
                    image_x.array.max(), prof.max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")
            np.testing.assert_allclose(
                    image_x1.array.max(), prof.shear(q).max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")
            np.testing.assert_allclose(
                    image_x2.array.max(), prof.shear(s).max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")

        # Flip around y axis (i.e. x -> -x)
        flip1 = prof.transform(-1, 0, 0, 1)
        image2_x = flip1.drawImage(image=im.copy(), method='no_pixel')
        np.testing.assert_array_almost_equal(
            image_x.array, image2_x.array[:,::-1], decimal=decimal,
            err_msg="Flipping image around y-axis failed x test")
        image2_x1 = flip1.shear(q1).drawImage(image=im.copy(), method='no_pixel')
        np.testing.assert_array_almost_equal(
            image_x1.array, image2_x1.array[:,::-1], decimal=decimal,
            err_msg="Flipping image around y-axis failed x1 test")
        image2_x2 = flip1.shear(s1).drawImage(image=im.copy(), method='no_pixel')
        np.testing.assert_array_almost_equal(
            image_x2.array, image2_x2.array[:,::-1], decimal=decimal,
            err_msg="Flipping image around y-axis failed x2 test")
        image2_k = flip1.drawImage(image=im.copy())
        np.testing.assert_array_almost_equal(
            image_k.array, image2_k.array[:,::-1], decimal=decimal,
            err_msg="Flipping image around y-axis failed k test")
        image2_k1 = flip1.shear(q1).drawImage(image=im.copy())
        np.testing.assert_array_almost_equal(
            image_k1.array, image2_k1.array[:,::-1], decimal=decimal,
            err_msg="Flipping image around y-axis failed k1 test")
        image2_k2 = flip1.shear(s1).drawImage(image=im.copy())
        np.testing.assert_array_almost_equal(
            image_k2.array, image2_k2.array[:,::-1], decimal=decimal,
            err_msg="Flipping image around y-axis failed k2 test")

        if close_maxsb:
            np.testing.assert_allclose(
                    image2_x.array.max(), flip1.max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")
            np.testing.assert_allclose(
                    image2_x1.array.max(), flip1.shear(q).max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")
            np.testing.assert_allclose(
                    image2_x2.array.max(), flip1.shear(s).max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")

        # Flip around x axis (i.e. y -> -y)
        flip2 = prof.transform(1, 0, 0, -1)
        image2_x = flip2.drawImage(image=im.copy(), method='no_pixel')
        np.testing.assert_array_almost_equal(
            image_x.array, image2_x.array[::-1,:], decimal=decimal,
            err_msg="Flipping image around x-axis failed x test")
        image2_x1 = flip2.shear(q1).drawImage(image=im.copy(), method='no_pixel')
        np.testing.assert_array_almost_equal(
            image_x1.array, image2_x1.array[::-1,:], decimal=decimal,
            err_msg="Flipping image around x-axis failed x1 test")
        image2_x2 = flip2.shear(s1).drawImage(image=im.copy(), method='no_pixel')
        np.testing.assert_array_almost_equal(
            image_x2.array, image2_x2.array[::-1,:], decimal=decimal,
            err_msg="Flipping image around x-axis failed x2 test")
        image2_k = flip2.drawImage(image=im.copy())
        np.testing.assert_array_almost_equal(
            image_k.array, image2_k.array[::-1,:], decimal=decimal,
            err_msg="Flipping image around x-axis failed k test")
        image2_k1 = flip2.shear(q1).drawImage(image=im.copy())
        np.testing.assert_array_almost_equal(
            image_k1.array, image2_k1.array[::-1,:], decimal=decimal,
            err_msg="Flipping image around x-axis failed k1 test")
        image2_k2 = flip2.shear(s1).drawImage(image=im.copy())
        np.testing.assert_array_almost_equal(
            image_k2.array, image2_k2.array[::-1,:], decimal=decimal,
            err_msg="Flipping image around x-axis failed k2 test")

        if close_maxsb:
            np.testing.assert_allclose(
                    image2_x.array.max(), flip2.max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")
            np.testing.assert_allclose(
                    image2_x1.array.max(), flip2.shear(q).max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")
            np.testing.assert_allclose(
                    image2_x2.array.max(), flip2.shear(s).max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")

        # Flip around x=y (i.e. y -> x, x -> y)
        flip3 = prof.transform(0, 1, 1, 0)
        image2_x = flip3.drawImage(image=im.copy(), method='no_pixel')
        np.testing.assert_array_almost_equal(
            image_x.array, np.transpose(image2_x.array), decimal=decimal,
            err_msg="Flipping image around x=y failed x test")
        image2_x1 = flip3.shear(q2).drawImage(image=im.copy(), method='no_pixel')
        np.testing.assert_array_almost_equal(
            image_x1.array, np.transpose(image2_x1.array), decimal=decimal,
            err_msg="Flipping image around x=y failed x1 test")
        image2_x2 = flip3.shear(s2).drawImage(image=im.copy(), method='no_pixel')
        np.testing.assert_array_almost_equal(
            image_x2.array, np.transpose(image2_x2.array), decimal=decimal,
            err_msg="Flipping image around x=y failed x2 test")
        image2_k = flip3.drawImage(image=im.copy())
        np.testing.assert_array_almost_equal(
            image_k.array, np.transpose(image2_k.array), decimal=decimal,
            err_msg="Flipping image around x=y failed k test")
        image2_k1 = flip3.shear(q2).drawImage(image=im.copy())
        np.testing.assert_array_almost_equal(
            image_k1.array, np.transpose(image2_k1.array), decimal=decimal,
            err_msg="Flipping image around x=y failed k1 test")
        image2_k2 = flip3.shear(s2).drawImage(image=im.copy())
        np.testing.assert_array_almost_equal(
            image_k2.array, np.transpose(image2_k2.array), decimal=decimal,
            err_msg="Flipping image around x=y failed k2 test")

        if close_maxsb:
            np.testing.assert_allclose(
                    image2_x.array.max(), flip3.max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")
            np.testing.assert_allclose(
                    image2_x1.array.max(), flip3.shear(q).max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")
            np.testing.assert_allclose(
                    image2_x2.array.max(), flip3.shear(s).max_sb*im.scale**2, rtol=0.2,
                    err_msg="max_sb did not match maximum pixel value")

        check_pickle(prof, lambda x: x.drawImage(image=im.copy(), method='no_pixel'))
        check_pickle(flip1, lambda x: x.drawImage(image=im.copy(), method='no_pixel'))
        check_pickle(flip2, lambda x: x.drawImage(image=im.copy(), method='no_pixel'))
        check_pickle(flip3, lambda x: x.drawImage(image=im.copy(), method='no_pixel'))
        check_pickle(prof)
        check_pickle(flip1)
        check_pickle(flip2)
        check_pickle(flip3)


@timer
def test_ne():
    """ Check that inequality works as expected."""
    gal1 = galsim.Gaussian(fwhm=1)
    gal2 = galsim.Gaussian(fwhm=2)
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Transforms using the callables below will produce galsim.ChromaticTransformation objects
    # instead of galsim.Transformation objects, but they should still compare unequal so we go ahead
    # and test them too.
    jac = lambda w: [[w, 0], [0, 1]]
    offset = lambda w: (0, w)
    flux_ratio = lambda w: w

    objs = [galsim.Transform(gal1),
            galsim.Transform(gal2),
            galsim.Transform(gal1, jac=(1, 0.5, 0.5, 1)),
            galsim.Transform(gal1, jac=(1, 1, 1, 0)),
            galsim.Transform(gal1, jac=jac),
            galsim.Transform(gal1, offset=galsim.PositionD(2, 2)),
            galsim.Transform(gal1, offset=offset),
            galsim.Transform(gal1, flux_ratio=1.1),
            galsim.Transform(gal1, flux_ratio=flux_ratio),
            galsim.Transform(gal1, gsparams=gsp),
            galsim.Transform(gal1, gsparams=gsp, propagate_gsparams=False)]
    check_all_diff(objs)

    # A degenerate jacobian will raise an exception.
    with assert_raises(galsim.GalSimError):
        degen = galsim.Transform(gal1, jac=(1, 1, 1, 1))

    assert_raises(TypeError, galsim.Transform, jac)


@timer
def test_compound():
    """Check that transformations of transformations work the same whether they are compounded
    automatically or not.
    """
    gal1 = galsim.Gaussian(fwhm=1.7, flux=2.3)
    gal2 = gal1.shear(g1=0.21, g2=0.12).rotate(33 * galsim.degrees).shift(0.1,0.4) * 1.1
    gal3 = gal2.shear(g1=0.18, g2=0.09).rotate(12 * galsim.degrees).shift(-0.3,0.5) * 8.9
    # These should have compounded automatically into a single transformation
    print('gal3 = ',gal3)
    print('gal3.original = ',gal3.original)
    assert gal3.original == gal1

    gal4 = gal2 + 0. * gal1  # Equivalent to gal2, but the sum kills the automatic compounding.
    gal5 = gal4.shear(g1=0.18, g2=0.09).rotate(12 * galsim.degrees).shift(-0.3,0.5) * 8.9
    print('gal5 = ',gal5)
    print('gal5.original = ',gal5.original)
    assert gal5.original != gal1
    assert gal5.original == gal4

    # Despite that, the gal3 and gal5 should draw the same in both real and k-space
    im3_d = galsim.ImageD(8,8)
    im5_d = galsim.ImageD(8,8)
    im3_f = galsim.ImageF(8,8)
    im5_f = galsim.ImageF(8,8)
    im3_cd = galsim.ImageCD(8,8)
    im5_cd = galsim.ImageCD(8,8)
    im3_cf = galsim.ImageCF(8,8)
    im5_cf = galsim.ImageCF(8,8)

    # Note: these are not equal.  gal5 lost track of the overall transformation and thinks it
    # needs a bit larger maxk, smaller stepk.  ~10% different.
    print('gal3.stepk = ',gal3.stepk)
    print('gal5.stepk = ',gal5.stepk)
    print('gal3.maxk = ',gal3.maxk)
    print('gal5.maxk = ',gal5.maxk)

    gal3.drawImage(image=im3_d, method='sb', scale=0.2)
    gal5.drawImage(image=im5_d, method='sb', scale=0.2)
    np.testing.assert_almost_equal(im3_d[1,1], gal3.xValue(-0.7,-0.7), decimal=12)
    np.testing.assert_almost_equal(im5_d[1,1], gal3.xValue(-0.7,-0.7), decimal=12)
    np.testing.assert_almost_equal(im3_d.array[0,0], gal3.xValue(-0.7,-0.7), decimal=12)
    np.testing.assert_almost_equal(im5_d.array[0,0], gal3.xValue(-0.7,-0.7), decimal=12)
    np.testing.assert_array_almost_equal(im3_d.array, im5_d.array, decimal=12)

    gal3.drawImage(image=im3_f, method='sb', scale=0.2)
    gal5.drawImage(image=im5_f, method='sb', scale=0.2)
    np.testing.assert_almost_equal(im3_f[1,1], gal3.xValue(-0.7,-0.7), decimal=4)
    np.testing.assert_almost_equal(im5_f[1,1], gal3.xValue(-0.7,-0.7), decimal=4)
    np.testing.assert_almost_equal(im3_f.array, im5_f.array, decimal=4)
    np.testing.assert_almost_equal(im3_f.array, im3_d.array, decimal=4)
    np.testing.assert_almost_equal(im5_f.array, im5_d.array, decimal=4)

    gal3.drawKImage(image=im3_cd, scale=0.5)
    gal5.drawKImage(image=im5_cd, scale=0.5)
    np.testing.assert_almost_equal(im3_cd[-4,-4], gal3.kValue(-2.,-2.), decimal=12)
    np.testing.assert_almost_equal(im5_cd[-4,-4], gal3.kValue(-2.,-2.), decimal=12)
    np.testing.assert_almost_equal(im3_cd.array[0,0], gal3.kValue(-2.,-2.), decimal=12)
    np.testing.assert_almost_equal(im5_cd.array[0,0], gal3.kValue(-2.,-2.), decimal=12)
    np.testing.assert_array_almost_equal(im3_cd.array, im5_cd.array, decimal=12)

    gal3.drawKImage(image=im3_cf, scale=0.5)
    gal5.drawKImage(image=im5_cf, scale=0.5)
    np.testing.assert_almost_equal(im3_cf[-4,-4], gal3.kValue(-2.,-2.), decimal=3)
    np.testing.assert_almost_equal(im5_cf[-4,-4], gal3.kValue(-2.,-2.), decimal=3)
    np.testing.assert_array_almost_equal(im3_cf.array, im5_cf.array, decimal=3)
    np.testing.assert_array_almost_equal(im3_cf.array, im3_cd.array, decimal=3)
    np.testing.assert_array_almost_equal(im5_cf.array, im5_cd.array, decimal=3)


@timer
def test_gsparams():
    """Test withGSParams with some non-default gsparams
    """
    obj = galsim.Exponential(half_light_radius=1.7)
    gsp = galsim.GSParams(folding_threshold=1.e-4, maxk_threshold=1.e-4, maximum_fft_size=1.e4)
    gsp2 = galsim.GSParams(folding_threshold=1.e-2, maxk_threshold=1.e-2)

    tr = obj.shear(g1=0.2, g2=0.3)
    jac = galsim.Shear(g1=0.2, g2=0.3).getMatrix()
    tr1 = tr.withGSParams(gsp)
    assert tr.gsparams == galsim.GSParams()
    assert tr1.gsparams == gsp
    assert tr1.original.gsparams == gsp

    tr2 = galsim.Transformation(obj.withGSParams(gsp), jac=jac)
    tr3 = galsim.Transformation(galsim.Exponential(half_light_radius=1.7, gsparams=gsp), jac=jac)
    tr4 = galsim.Transform(obj, jac=jac, gsparams=gsp)
    assert tr != tr1
    assert tr1 == tr2
    assert tr1 == tr3
    assert tr1 == tr4
    print('stepk = ',tr.stepk, tr1.stepk)
    assert tr1.stepk < tr.stepk
    print('maxk = ',tr.maxk, tr1.maxk)
    assert tr1.maxk > tr.maxk

    tr5 = galsim.Transform(obj, jac=jac, gsparams=gsp, propagate_gsparams=False)
    assert tr5 != tr4
    assert tr5.gsparams == gsp
    assert tr5.original.gsparams == galsim.GSParams()

    tr6 = tr5.withGSParams(gsp2)
    assert tr6 != tr5
    assert tr6.gsparams == gsp2
    assert tr6.original.gsparams == galsim.GSParams()

    tr7 = tr5.withGSParams(folding_threshold=1.e-2, maxk_threshold=1.e-2)
    assert tr7 != tr5
    assert tr7 != tr6
    assert tr7.gsparams == gsp2.withParams(maximum_fft_size=1.e4)


if __name__ == "__main__":
    runtests(__file__)
