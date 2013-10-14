# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
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
        alias_threshold = 5.e-3,
        maxk_threshold = 1.e-3,
        kvalue_accuracy = 1.e-5,
        xvalue_accuracy = 1.e-5,
        shoot_accuracy = 1.e-5,
        realspace_relerr = 1.e-3,
        realspace_abserr = 1.e-6,
        integration_relerr = 1.e-5,
        integration_abserr = 1.e-7)

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
    # test the SBProfile version using applyShear
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    mySBP.applyShear(myShear._shear)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Small-shear Gaussian profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.applyShear(myShear)
    gauss.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = gauss.createSheared(myShear)
    gauss2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createSheared disagrees with expected result")
 
    # Check with default_params
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=default_params)
    gauss.applyShear(myShear)
    gauss.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear with default_params disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=galsim.GSParams())
    gauss.applyShear(myShear)
    gauss.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear with GSParams() disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(gauss,myImg,"sheared Gaussian")

    # Test kvalues
    do_kvalue(gauss,"sheared Gaussian")

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
    # test the SBProfile version using applyShear
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_largeshear.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP = galsim.SBDeVaucouleurs(flux=1, half_light_radius=1)
    mySBP.applyShear(myShear._shear)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Large-shear DeVaucouleurs profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc.applyShear(myShear)
    devauc.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc2 = devauc.createSheared(myShear)
    devauc2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createSheared disagrees with expected result")

    # Check with default_params
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1, gsparams=default_params)
    devauc.applyShear(myShear)
    devauc.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear with default_params disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    devauc.applyShear(myShear)
    devauc.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear with GSParams() disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    devauc2 = galsim.Convolve(devauc, galsim.Gaussian(sigma=0.3))
    do_shoot(devauc2,myImg,"sheared DeVauc")

    # Test kvalues.
    # Testing a sheared devauc requires a rather large fft.  What we really care about 
    # testing though is the accuracy of the applyShear function.  So just shear a Gaussian here.
    gauss = galsim.Gaussian(sigma=2.3)
    gauss.applyShear(myShear)
    do_kvalue(gauss, "sheared Gaussian")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

 

def test_rotate():
    """Test the 45 degree rotation of a sheared Sersic profile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBSersic(n=2.5, flux=1, half_light_radius=1)
    myShear = galsim.Shear(e1=0.2, e2=0.0)
    mySBP.applyShear(myShear._shear)
    mySBP.applyRotation(45.0 * galsim.degrees)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_ellip_rotated.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="45-degree rotated elliptical Gaussian disagrees with expected result")

    # Repeat with the GSObject version of this:
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1)
    gal.applyShear(myShear)
    gal.applyRotation(45.0 * galsim.degrees)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation disagrees with expected result")

    # Check with default_params
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1, gsparams=default_params)
    gal.applyShear(myShear)
    gal.applyRotation(45.0 * galsim.degrees)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation with default_params disagrees with expected "
            "result")
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    gal.applyShear(myShear)
    gal.applyRotation(45.0 * galsim.degrees)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation with GSParams() disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    gal2 = galsim.Convolve(gal, galsim.Gaussian(sigma=0.3))
    do_shoot(gal2,myImg,"rotated sheared Sersic")

    # Test kvalues
    do_kvalue(gal,"rotated sheared Sersic")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_mag():
    """Test the magnification (size x 1.5) of an exponential profile against a known result.
    """
    import time
    t1 = time.time()
    re = 1.0
    r0 = re/1.67839
    mySBP = galsim.SBExponential(flux=1, scale_radius=r0)
    mySBP.applyExpansion(1.5)
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_mag.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Magnification (x1.5) of exponential SBProfile disagrees with expected result")

    # Use applyDilation
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyDilation(1.5)
    gal.scaleFlux(1.5**2) # Apply the flux magnification.
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDilation disagrees with expected result")
 
    # Check with default_params
    gal = galsim.Exponential(flux=1, scale_radius=r0, gsparams=default_params)
    gal.applyDilation(1.5)
    gal.scaleFlux(1.5**2)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation with default_params disagrees with "
            "expected result")
    gal = galsim.Exponential(flux=1, scale_radius=r0, gsparams=galsim.GSParams())
    gal.applyDilation(1.5)
    gal.scaleFlux(1.5**2)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation with GSParams() disagrees with "
            "expected result")

    # Use applyMagnification
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyMagnification(1.5**2) # area rescaling factor
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyMagnification disagrees with expected result")

    # Use applyLensing
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyLensing(0., 0., 1.5**2) # area rescaling factor
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyLensing disagrees with expected result")

    # Use createDilated
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal2 = gal.createDilated(1.5)
    gal2.scaleFlux(1.5**2) # Apply the flux magnification.
    gal2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createDilated disagrees with expected result")
 
    # Use createMagnified
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal2 = gal.createMagnified(1.5**2) # area rescaling factor
    gal2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createMagnified disagrees with expected result")
 
    # Use createLensed
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal2 = gal.createLensed(0., 0., 1.5**2) # area rescaling factor
    gal2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createLensed disagrees with expected result")
 
    # Test photon shooting.
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyMagnification(1.5**2) # area rescaling factor
    do_shoot(gal,myImg,"dilated Exponential")

    # Test kvalues
    do_kvalue(gal,"dilated Exponential")

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
    ser2 = ser.createLensed(g1, g2, mu)
    ser.applyShear(g1=g1, g2=g2)
    ser.applyMagnification(mu)
    im = galsim.ImageF(imsize, imsize, scale=pix_scale)
    im = ser.draw(im.view())
    im2 = galsim.ImageF(imsize, imsize, scale=pix_scale)
    im2 = ser2.draw(im2.view())
    np.testing.assert_array_almost_equal(im.array, im2.array, 5,
        err_msg="Lensing of Sersic profile done in two different ways gives different answer")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shift():
    """Test the translation of a Box profile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBBox(xw=0.2, yw=0.2, flux=1)
    mySBP.applyShift(0.2, -0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_shift.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Shifted box profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    pixel = galsim.Pixel(xw=0.2, yw=0.2)
    pixel.applyShift(0.2, -0.2)
    pixel.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShift disagrees with expected result")
 
    # Check with default_params
    pixel = galsim.Pixel(xw=0.2, yw=0.2, gsparams=default_params)
    pixel.applyShift(0.2, -0.2)
    pixel.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShift with default_params disagrees with expected result")
    pixel = galsim.Pixel(xw=0.2, yw=0.2, gsparams=galsim.GSParams())
    pixel.applyShift(0.2, -0.2)
    pixel.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShift with GSParams() disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(pixel,myImg,"shifted Box")

    # Test kvalues.
    # Testing a shifted box requires a ridiculously large fft.  What we really care about 
    # testing though is the accuracy of the applyShift function.  So just shift a Gaussian here.
    gauss = galsim.Gaussian(sigma=2.3)
    gauss.applyShift(0.2,-0.2)
    do_kvalue(gauss, "shifted Gaussian")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_rescale():
    """Test the flux rescaling of a Sersic profile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBSersic(n=3, flux=1, half_light_radius=1)
    mySBP.setFlux(2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_doubleflux.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Flux-rescale sersic profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.setFlux(2)
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject setFlux disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic *= 2
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject *= 2 disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic2 = sersic * 2
    sersic2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject obj * 2 disagrees with expected result")
    sersic2 = 2 * sersic
    sersic2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject 2 * obj disagrees with expected result")

    # Check with default_params
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=default_params)
    sersic *= 2
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject *= 2 with default_params disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    sersic *= 2
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject *= 2 with GSParams() disagrees with expected result")
 
    # Can also get a flux of 2 by drawing flux=1 twice with add_to_image=True
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    sersic.draw(myImg,dx=0.2, normalization="surface brightness",add_to_image=True,
                use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Drawing with add_to_image=True disagrees with expected result")

    # With lower alias_threshold and maxk_threshold, the calculated flux should come out right 
    # so long as we also convolve by a pixel:
    gsp1 = galsim.GSParams(alias_threshold=1.e-3, maxk_threshold=5.e-4)
    sersic_acc = galsim.Convolve([
            galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=gsp1),
            galsim.Pixel(xw=0.2, gsparams=gsp1)  
            ])
    myImg2 = sersic_acc.draw(dx=0.2, use_true_center=False)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum(), 1., 3,
            err_msg="Drawing with gsp1 results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux, 1., 3,
            err_msg="Drawing with gsp1 returned wrong added_flux")
    myImg2 = sersic_acc.draw(myImg2, add_to_image=True, use_true_center=False)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum(), 2., 3,
            err_msg="Drawing with add_to_image=True results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux, 1., 3,
            err_msg="Drawing with add_to_image=True returned wrong added_flux")

    # Check that the flux works out when adding multiple times.
    # With a Gaussian, we can take the thresholds even lower and get another digit of accuracy.
    gsp2 = galsim.GSParams(alias_threshold=1.e-5, maxk_threshold=1.e-5)
    gauss = galsim.Gaussian(flux=1.e5, sigma=2., gsparams=gsp2)
    gauss2 = galsim.Convolve([gauss, galsim.Pixel(xw=0.2, gsparams=gsp2)])
    myImg2 = gauss2.draw(dx=0.2, use_true_center=False)
    print 'image size = ',myImg2.array.shape
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 1., 4,
            err_msg="Drawing Gaussian results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian returns wrong added_flux")
    myImg2 = gauss2.draw(myImg2, add_to_image=True, use_true_center=False)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 2., 4,
            err_msg="Drawing Gaussian with add_to_image=True results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian with add_to_image=True returns wrong added_flux")
    rng = galsim.BaseDeviate(12345)
    myImg2 = gauss.drawShoot(myImg2, add_to_image=True, poisson_flux=False, rng=rng)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 3., 4,
            err_msg="Drawing Gaussian with drawShoot, add_to_image=True, poisson_flux=False "+
                    "results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian with drawShoot, add_to_image=True, poisson_flux=False "+
                    "returned wrong added_flux")
    myImg2 = gauss.drawShoot(myImg2, add_to_image=True, rng=rng)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 4., 1,
            err_msg="Drawing Gaussian with drawShoot, add_to_image=True, poisson_flux=True "+
                    "results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 1,
            err_msg="Drawing Gaussian with drawShoot, add_to_image=True, poisson_flux=True "+
                    "returned wrong added_flux")
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 3.+myImg2.added_flux/1.e5, 4,
            err_msg="Drawing Gaussian with drawShoot, add_to_image=True results in wrong flux "+
                    "according to the returned added_flux")

    # Can also get a flux of 2 using gain = 0.5
    sersic.draw(myImg, dx=0.2, gain=0.5, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Drawing with gain=0.5 disagrees with expected result")
    myImg2 = sersic_acc.draw(dx=0.2, gain=0.5, use_true_center=False)
    np.testing.assert_almost_equal(myImg2.array.sum(), 2., 3,
            err_msg="Drawing with gain=0.5 results in wrong flux")
    myImg2 = sersic_acc.draw(dx=0.2, gain=4., use_true_center=False)
    np.testing.assert_almost_equal(myImg2.array.sum(), 0.25, 3,
            err_msg="Drawing with gain=4. results in wrong flux")
    # Check add_to_image in conjunction with gain
    sersic_acc.draw(myImg2, gain=4., add_to_image=True, use_true_center=False)
    np.testing.assert_almost_equal(myImg2.array.sum(), 0.5, 3,
            err_msg="Drawing with gain=4. results in wrong flux")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic_smooth = galsim.Convolve(sersic2, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic_smooth,myImg,"scaled Sersic")

    # Test kvalues
    do_kvalue(sersic2, "scaled Sersic")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_integer_shift_fft():
    """Test if applyShift works correctly for integer shifts using draw method.
    """
    import time
    t1 = time.time()

    gal = galsim.Gaussian(sigma=test_sigma)
    pix = galsim.Pixel(1.)
    psf = galsim.Airy(lam_over_diam=test_hlr)

    # shift galaxy only
 
    final=galsim.Convolve([gal, psf, pix])
    img_center = galsim.ImageD(n_pix_x,n_pix_y)
    final.draw(img_center,dx=1)

    gal.applyShift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf, pix])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    final.draw(img_shift,dx=1)

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
    psf.applyShift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf, pix])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    final.draw(img_shift,dx=1)

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
    """Test if applyShift works correctly for integer shifts using drawShoot method.
    """
    import time
    t1 = time.time()

    n_photons_low = 10
    seed = 10

    gal = galsim.Gaussian(sigma=test_sigma)
    pix = galsim.Pixel(1.)
    psf = galsim.Airy(lam_over_diam=test_hlr)

    # shift galaxy only
 
    final=galsim.Convolve([gal, psf, pix])
    img_center = galsim.ImageD(n_pix_x,n_pix_y)
    test_deviate = galsim.BaseDeviate(seed)
    final.drawShoot(img_center,dx=1,rng=test_deviate,n_photons=n_photons_low)

    gal.applyShift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf, pix])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    test_deviate = galsim.BaseDeviate(seed)
    final.drawShoot(img_shift,dx=1,rng=test_deviate,n_photons=n_photons_low)
    
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
    psf.applyShift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf, pix])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    test_deviate = galsim.BaseDeviate(seed)
    final.drawShoot(img_shift,dx=1,rng=test_deviate,n_photons=n_photons_low)

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
