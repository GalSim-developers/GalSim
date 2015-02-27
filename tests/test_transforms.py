# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

from galsim_test_helpers import *

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images. 

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# for flux normalization tests
test_flux = 1.8

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

# Some parameters used in the two unit tests test_integer_shift_fft and test_integer_shift_photon:
test_sigma = 1.8
test_hlr = 1.8
int_shift_x = 7
int_shift_y = 3
n_pix_x = 50
n_pix_y = 60
delta_sub = 30
image_decimal_precise = 15


def test_smallshear():
    """Test the application of a small shear to a Gaussian SBProfile against a known result.
    """
    import time
    t1 = time.time()
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
 
    # Check with default_params
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=default_params)
    gauss = gauss.shear(myShear)
    gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear with default_params disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=galsim.GSParams())
    gauss = gauss.shear(myShear)
    gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear with GSParams() disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(gauss,myImg,"sheared Gaussian")

    # Test kvalues
    do_kvalue(gauss,myImg,"sheared Gaussian")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_largeshear():
    """Test the application of a large shear to a Sersic SBProfile against a known result.
    """
    import time
    t1 = time.time()
    e1 = 0.0
    e2 = 0.5

    myShear = galsim.Shear(e1=e1, e2=e2)
    # test the SBProfile version using shear
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

    # Check with default_params
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1, gsparams=default_params)
    devauc = devauc.shear(myShear)
    devauc.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear with default_params disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    devauc = devauc.shear(myShear)
    devauc.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject shear with GSParams() disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    devauc2 = galsim.Convolve(devauc, galsim.Gaussian(sigma=0.3))
    do_shoot(devauc2,myImg,"sheared DeVauc")

    # Test kvalues.
    # Testing a sheared devauc requires a rather large fft.  What we really care about 
    # testing though is the accuracy of the shear function.  So just shear a Gaussian here.
    gauss = galsim.Gaussian(sigma=2.3)
    gauss = gauss.shear(myShear)
    do_kvalue(gauss,myImg, "sheared Gaussian")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

 

def test_rotate():
    """Test the 45 degree rotation of a sheared Sersic profile against a known result.
    """
    import time
    t1 = time.time()
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

    # Check with default_params
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1, gsparams=default_params)
    gal = gal.shear(myShear).rotate(45.0 * galsim.degrees)
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject rotate with default_params disagrees with expected "
            "result")
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    gal = gal.shear(myShear).rotate(45.0 * galsim.degrees)
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject rotate with GSParams() disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    gal2 = galsim.Convolve(gal, galsim.Gaussian(sigma=0.3))
    do_shoot(gal2,myImg,"rotated sheared Sersic")

    # Test kvalues
    do_kvalue(gal,myImg,"rotated sheared Sersic")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_mag():
    """Test the magnification (size x 1.5) of an exponential profile against a known result.
    """
    import time
    t1 = time.time()
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

    # Use lens
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal = gal.lens(0., 0., 1.5**2) # area rescaling factor
    gal.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject lens disagrees with expected result")

    # Test photon shooting.
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal = gal.magnify(1.5**2) # area rescaling factor
    do_shoot(gal,myImg,"dilated Exponential")

    # Test kvalues
    do_kvalue(gal,myImg,"dilated Exponential")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_lens():
    """Test the lensing (shear, magnification) of a Sersic profile carried out 2 ways.
    """
    import time
    t1 = time.time()
    re = 1.0
    n = 3.
    g1 = 0.12
    g2 = -0.4
    mu = 1.2
    pix_scale = 0.1
    imsize = 100
    ser = galsim.Sersic(n, half_light_radius = re)
    ser2 = ser.lens(g1, g2, mu)
    ser = ser.shear(g1=g1, g2=g2).magnify(mu)
    im = galsim.ImageF(imsize, imsize, scale=pix_scale)
    im = ser.drawImage(im, method='no_pixel')
    im2 = galsim.ImageF(imsize, imsize, scale=pix_scale)
    im2 = ser2.drawImage(im2, method='no_pixel')
    np.testing.assert_array_almost_equal(im.array, im2.array, 5,
        err_msg="Lensing of Sersic profile done in two different ways gives different answer")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shift():
    """Test the translation of a Box profile against a known result.
    """
    import time
    t1 = time.time()
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
 
    # Test photon shooting.
    do_shoot(pixel,myImg,"shifted Box")

    # Test kvalues.
    # Testing a shifted box requires a ridiculously large fft.  What we really care about 
    # testing though is the accuracy of the shift function.  So just shift a Gaussian here.
    gauss = galsim.Gaussian(sigma=2.3)
    gauss = gauss.shift(dx,-dx)
    do_kvalue(gauss,myImg, "shifted Gaussian")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_rescale():
    """Test the flux rescaling of a Sersic profile against a known result.
    """
    import time
    t1 = time.time()
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_doubleflux.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.withFlux(2).drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject withFlux disagrees with expected result")
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
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum(), 1., 3,
            err_msg="Drawing with gsp1 results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux, 1., 3,
            err_msg="Drawing with gsp1 returned wrong added_flux")
    myImg2 = sersic_acc.drawImage(myImg2, add_to_image=True, use_true_center=False)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum(), 2., 3,
            err_msg="Drawing with add_to_image=True results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux, 1., 3,
            err_msg="Drawing with add_to_image=True returned wrong added_flux")

    # Check that the flux works out when adding multiple times.
    # With a Gaussian, we can take the thresholds even lower and get another digit of accuracy.
    gsp2 = galsim.GSParams(folding_threshold=1.e-5, maxk_threshold=1.e-5)
    gauss = galsim.Gaussian(flux=1.e5, sigma=2., gsparams=gsp2)
    myImg2 = gauss.drawImage(scale=dx, use_true_center=False)
    print 'image size = ',myImg2.array.shape
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 1., 4,
            err_msg="Drawing Gaussian results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian returns wrong added_flux")
    myImg2 = gauss.drawImage(myImg2, add_to_image=True, use_true_center=False)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 2., 4,
            err_msg="Drawing Gaussian with add_to_image=True results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian with add_to_image=True returns wrong added_flux")
    rng = galsim.BaseDeviate(12345)
    myImg2 = gauss.drawImage(myImg2, add_to_image=True, poisson_flux=False, rng=rng, method='phot')
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 3., 4,
            err_msg="Drawing Gaussian with method=phot, add_to_image=True, poisson_flux=False "+
                    "results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian with method=phot, add_to_image=True, poisson_flux=False "+
                    "returned wrong added_flux")
    myImg2 = gauss.drawImage(myImg2, add_to_image=True, rng=rng, method='phot')
    print myImg2.array.sum(), myImg2.added_flux
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
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic_smooth = galsim.Convolve(sersic2, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic_smooth,myImg,"scaled Sersic")

    # Test kvalues
    do_kvalue(sersic2,myImg, "scaled Sersic")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_integer_shift_fft():
    """Test if shift works correctly for integer shifts using drawImage method.
    """
    import time
    t1 = time.time()

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
        (n_pix_y - delta_sub) / 2 : (n_pix_y + delta_sub) / 2,
        (n_pix_x - delta_sub) / 2 : (n_pix_x + delta_sub) / 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) / 2  + int_shift_y : (n_pix_y + delta_sub) / 2  + int_shift_y,
        (n_pix_x - delta_sub) / 2  + int_shift_x : (n_pix_x + delta_sub) / 2  + int_shift_x]

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
        (n_pix_y - delta_sub) / 2 : (n_pix_y + delta_sub) / 2,
        (n_pix_x - delta_sub) / 2 : (n_pix_x + delta_sub) / 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) / 2  + int_shift_y : (n_pix_y + delta_sub) / 2  + int_shift_y,
        (n_pix_x - delta_sub) / 2  + int_shift_x : (n_pix_x + delta_sub) / 2  + int_shift_x]
    np.testing.assert_array_almost_equal(
        sub_center, sub_shift,  decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with only PSF shifted ")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_integer_shift_photon():
    """Test if shift works correctly for integer shifts using method=phot.
    """
    import time
    t1 = time.time()

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
        (n_pix_y - delta_sub) / 2 : (n_pix_y + delta_sub) / 2,
        (n_pix_x - delta_sub) / 2 : (n_pix_x + delta_sub) / 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) / 2  + int_shift_y : (n_pix_y + delta_sub) / 2  + int_shift_y,
        (n_pix_x - delta_sub) / 2  + int_shift_x : (n_pix_x + delta_sub) / 2  + int_shift_x]


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
        (n_pix_y - delta_sub) / 2 : (n_pix_y + delta_sub) / 2,
        (n_pix_x - delta_sub) / 2 : (n_pix_x + delta_sub) / 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) / 2  + int_shift_y : (n_pix_y + delta_sub) / 2  + int_shift_y,
        (n_pix_x - delta_sub) / 2  + int_shift_x : (n_pix_x + delta_sub) / 2  + int_shift_x]
    np.testing.assert_array_almost_equal(
        sub_center, sub_shift,  decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with only PSF shifted ")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_smallshear()
    test_largeshear()
    test_rotate()
    test_mag()
    test_lens()
    test_shift()
    test_rescale()
    test_integer_shift_fft()
    test_integer_shift_photon()
