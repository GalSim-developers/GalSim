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

# Some standard values for testing
test_flux = 1.8
test_hlr = 1.8
test_sigma = 1.8
test_scale = 1.8


def test_convolve():
    """Test the convolution of a Moffat and a Box SBProfile against a known result.
    """
    import time
    t1 = time.time()
    # Code was formerly:
    # mySBP = galsim.SBMoffat(beta=1.5, truncationFWHM=4, flux=1, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in 
    # physical units.  However, the same profile can be constructed using 
    # fwhm=1.0927449310213702,
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    fwhm_backwards_compatible = 1.0927449310213702
    mySBP = galsim.SBMoffat(beta=1.5, fwhm=fwhm_backwards_compatible, 
                            trunc=4*fwhm_backwards_compatible, flux=1)
    mySBP2 = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    myConv = galsim.SBConvolve([mySBP,mySBP2])
    # Using an exact Maple calculation for the comparison.  Only accurate to 4 decimal places.
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_pixel.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    myConv.draw(myImg.view())
    printval(myImg, savedImg)
 
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 4,
            err_msg="Moffat convolved with Box SBProfile disagrees with expected result")

    # Repeat with the GSObject version of this:
    psf = galsim.Moffat(beta=1.5, fwhm=fwhm_backwards_compatible, trunc=4*fwhm_backwards_compatible,
                        flux=1)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    # Note: Since both of these have hard edges, GalSim wants to do this with real_space=True.
    # Here we are intentionally tesing the Fourier convolution, so we want to suppress the
    # warning that GalSim emits.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # We'll do the real space convolution below
        conv = galsim.Convolve([psf,pixel],real_space=False)
        conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")

        # Other ways to do the convolution:
        conv = galsim.Convolve(psf,pixel,real_space=False)
        conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")

        # Check with default_params
        conv = galsim.Convolve([psf,pixel],real_space=False,gsparams=default_params)
        conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve([psf,pixel]) with default_params disagrees with"
                "expected result")
        conv = galsim.Convolve([psf,pixel],real_space=False,gsparams=galsim.GSParams())
        conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve([psf,pixel]) with GSParams() disagrees with"
                "expected result")
 
    # Test photon shooting.
    do_shoot(conv,myImg,"Moffat * Pixel")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_convolve_flux_scaling():
    """Test flux scaling for Convolve.
    """
    import time
    t1 = time.time()

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
    # First test that original obj is unharmed... (also tests that .copy() is working)
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
    # First test that original obj is unharmed... (also tests that .copy() is working)
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
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



def test_shearconvolve():
    """Test the convolution of a sheared Gaussian and a Box SBProfile against a known result.
    """
    import time
    t1 = time.time()
    e1 = 0.04
    e2 = 0.0
    myShear = galsim.Shear(e1=e1, e2=e2)
    # test at SBProfile level using applyShear
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    mySBP.applyShear(myShear._shear)
    mySBP2 = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    myConv = galsim.SBConvolve([mySBP,mySBP2])
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear_convolve_box.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    myConv.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Sheared Gaussian convolved with Box SBProfile disagrees with expected result")

    # Repeat with the GSObject version of this:
    psf = galsim.Gaussian(flux=1, sigma=1)
    psf2 = psf.createSheared(e1=e1, e2=e2)
    psf.applyShear(e1=e1, e2=e2)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel])
    conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    conv2 = galsim.Convolve([psf2,pixel])
    conv2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")

    # Check with default_params
    conv = galsim.Convolve([psf,pixel],gsparams=default_params)
    conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with default_params disagrees with "
            "expected result")
    conv = galsim.Convolve([psf,pixel],gsparams=galsim.GSParams())
    conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with GSParams() disagrees with "
            "expected result")
 
    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel)
    conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(conv,myImg,"sheared Gaussian * Pixel")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_realspace_convolve():
    """Test the real-space convolution of a Moffat and a Box SBProfile against a known result.
    """
    import time
    t1 = time.time()
    # Code was formerly:
    # mySBP = galsim.SBMoffat(beta=1.5, truncationFWHM=4, flux=1, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in 
    # physical units.  However, the same profile can be constructed using 
    # fwhm=1.0927449310213702,
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    fwhm_backwards_compatible = 1.0927449310213702
    #psf = galsim.SBMoffat(beta=1.5, fwhm=fwhm_backwards_compatible, 
                          #trunc=4*fwhm_backwards_compatible, flux=1)
    psf = galsim.SBMoffat(beta=1.5, half_light_radius=1,
                          trunc=4*fwhm_backwards_compatible, flux=1)
    pixel = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.SBConvolve([psf,pixel],real_space=True)
    # Note: Using an image created from Maple "exact" calculations.
    saved_img = galsim.fits.read(os.path.join(imgdir, "moffat_pixel.fits"))
    img = galsim.ImageF(saved_img.bounds, scale=0.2)
    conv.draw(img.view())
    printval(img, saved_img)
    arg = abs(saved_img.array-img.array).argmax()
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Moffat convolved with Box SBProfile disagrees with expected result")

    # Repeat with the GSObject version of this:
    psf = galsim.Moffat(beta=1.5, half_light_radius=1,
                        trunc=4*fwhm_backwards_compatible, flux=1)
    #psf = galsim.Moffat(beta=1.5, fwhm=fwhm_backwards_compatible,
                        #trunc=4*fwhm_backwards_compatible, flux=1)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel],real_space=True)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")

    # Check with default_params
    conv = galsim.Convolve([psf,pixel],real_space=True,gsparams=default_params)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with default_params disagrees with "
            "expected result")
    conv = galsim.Convolve([psf,pixel],real_space=True,gsparams=galsim.GSParams())
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel,real_space=True)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")

    # The real-space convolution algorithm is not (trivially) independent of the order of
    # the two things being convolved.  So check the opposite order.
    conv = galsim.Convolve([pixel,psf],real_space=True)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([pixel,psf]) disagrees with expected result")

    # Test kvalues
    do_kvalue(conv,"Truncated Moffat convolved with Box")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
 

def test_realspace_distorted_convolve():
    """
    The same as above, but both the Moffat and the Box are sheared, rotated and shifted
    to stress test the code that deals with this for real-space convolutions that wouldn't
    be tested otherwise.
    """
    import time
    t1 = time.time()
    fwhm_backwards_compatible = 1.0927449310213702
    psf = galsim.SBMoffat(beta=1.5, half_light_radius=1,
                          trunc=4*fwhm_backwards_compatible, flux=1)
    #psf = galsim.SBMoffat(beta=1.5, fwhm=fwhm_backwards_compatible, 
                          #trunc=4*fwhm_backwards_compatible, flux=1)  
    psf.applyShear(galsim.Shear(g1=0.11,g2=0.17)._shear)
    psf.applyRotation(13 * galsim.degrees)
    pixel = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    pixel.applyShear(galsim.Shear(g1=0.2,g2=0.0)._shear)
    pixel.applyRotation(80 * galsim.degrees)
    pixel.applyShift(0.13,0.27)
    conv = galsim.SBConvolve([psf,pixel],real_space=True)

    # Note: Using an image created from Maple "exact" calculations.
    saved_img = galsim.fits.read(os.path.join(imgdir, "moffat_pixel_distorted.fits"))
    img = galsim.ImageF(saved_img.bounds, scale=0.2)
    conv.draw(img.view())
    printval(img, saved_img)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="distorted Moffat convolved with distorted Box disagrees with expected result")

    # Repeat with the GSObject version of this:
    psf = galsim.Moffat(beta=1.5, half_light_radius=1,
                        trunc=4*fwhm_backwards_compatible, flux=1)
    #psf = galsim.Moffat(beta=1.5, fwhm=fwhm_backwards_compatible,
                        #trunc=4*fwhm_backwards_compatible, flux=1)
    psf.applyShear(g1=0.11,g2=0.17)
    psf.applyRotation(13 * galsim.degrees)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    pixel.applyShear(g1=0.2,g2=0.0)
    pixel.applyRotation(80 * galsim.degrees)
    pixel.applyShift(0.13,0.27)
    # NB: real-space is chosen automatically
    conv = galsim.Convolve([psf,pixel])
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([psf,pixel]) (distorted) disagrees with expected result")

    # Check with default_params
    conv = galsim.Convolve([psf,pixel],gsparams=default_params)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([psf,pixel]) (distorted) with default_params disagrees with "
            "expected result")
    conv = galsim.Convolve([psf,pixel],gsparams=galsim.GSParams())
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([psf,pixel]) (distorted) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve(psf,pixel) (distorted) disagrees with expected result")

    # The real-space convolution algorithm is not (trivially) independent of the order of
    # the two things being convolved.  So check the opposite order.
    conv = galsim.Convolve([pixel,psf])
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([pixel,psf]) (distorted) disagrees with expected result")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
 
def test_realspace_shearconvolve():
    """Test the real-space convolution of a sheared Gaussian and a Box SBProfile against a 
       known result.
    """
    import time
    t1 = time.time()
    psf = galsim.SBGaussian(flux=1, sigma=1)
    e1 = 0.04
    e2 = 0.0
    myShear = galsim.Shear(e1=e1, e2=e2)
    psf.applyShear(myShear._shear)
    pix = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.SBConvolve([psf,pix],real_space=True)
    saved_img = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear_convolve_box.fits"))
    img = galsim.ImageF(saved_img.bounds, scale=0.2)
    conv.draw(img.view())
    printval(img, saved_img)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Sheared Gaussian convolved with Box SBProfile disagrees with expected result")

    # Repeat with the GSObject version of this:
    psf = galsim.Gaussian(flux=1, sigma=1)
    psf.applyShear(e1=e1,e2=e2)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel],real_space=True)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")

    # Check with default_params
    conv = galsim.Convolve([psf,pixel],real_space=True,gsparams=default_params)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with default_params disagrees with "
            "expected result")
    conv = galsim.Convolve([psf,pixel],real_space=True,gsparams=galsim.GSParams())
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel,real_space=True)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")

    # The real-space convolution algorithm is not (trivially) independent of the order of
    # the two things being convolved.  So check the opposite order.
    conv = galsim.Convolve([pixel,psf],real_space=True)
    conv.draw(img,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([pixel,psf]) disagrees with expected result")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_add():
    """Test the addition of two rescaled Gaussian profiles against a known double Gaussian result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBGaussian(flux=0.75, sigma=1)
    mySBP2 = galsim.SBGaussian(flux=0.25, sigma=3)
    myAdd = galsim.SBAdd([mySBP, mySBP2])
    savedImg = galsim.fits.read(os.path.join(imgdir, "double_gaussian.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    myAdd.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Addition of two rescaled Gaussian profiles disagrees with expected result")

    # Repeat with the GSObject version of this:
    gauss1 = galsim.Gaussian(flux=0.75, sigma=1)
    gauss2 = galsim.Gaussian(flux=0.25, sigma=3)
    sum = galsim.Add(gauss1,gauss2)
    sum.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) disagrees with expected result")

    # Check with default_params
    sum = galsim.Add(gauss1,gauss2,gsparams=default_params)
    sum.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) with default_params disagrees with "
            "expected result")
    sum = galsim.Add(gauss1,gauss2,gsparams=galsim.GSParams())
    sum.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the sum:
    sum = gauss1 + gauss2
    sum.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject gauss1 + gauss2 disagrees with expected result")
    sum = gauss1.copy()
    sum += gauss2
    sum.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum = gauss1; sum += gauss2 disagrees with expected result")
    sum = galsim.Add([gauss1,gauss2])
    sum.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add([gauss1,gauss2]) disagrees with expected result")
    gauss1 = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = galsim.Gaussian(flux=1, sigma=3)
    sum = 0.75 * gauss1 + 0.25 * gauss2
    sum.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject 0.75 * gauss1 + 0.25 * gauss2 disagrees with expected result")
    sum = 0.75 * gauss1
    sum += 0.25 * gauss2
    sum.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum += 0.25 * gauss2 disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(sum,myImg,"sum of 2 Gaussians")

    # Test kvalues
    do_kvalue(sum,"sum of 2 Gaussians")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_add_flux_scaling():
    """Test flux scaling for Add.
    """
    import time
    t1 = time.time()

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
    # First test that original obj is unharmed... (also tests that .copy() is working)
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
    # First test that original obj is unharmed... (also tests that .copy() is working)
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
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_autoconvolve():
    """Test that auto-convolution works the same as convolution with itself.
    """
    import time
    t1 = time.time()

    mySBP = galsim.SBMoffat(beta=3.8, fwhm=1.3, flux=5)
    myConv = galsim.SBConvolve([mySBP,mySBP])
    myImg1 = galsim.ImageF(80,80, scale=0.4)
    myConv.draw(myImg1.view())
    myAutoConv = galsim.SBAutoConvolve(mySBP)
    myImg2 = galsim.ImageF(80,80, scale=0.4)
    myAutoConv.draw(myImg2.view())
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Moffat convolved with self disagrees with SBAutoConvolve result")

    # Repeat with the GSObject version of this:
    psf = galsim.Moffat(beta=3.8, fwhm=1.3, flux=5)
    conv = galsim.Convolve([psf,psf])
    conv.draw(myImg1)
    conv2 = galsim.AutoConvolve(psf)
    conv2.draw(myImg2)
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Moffat convolved with self disagrees with AutoConvolve result")

    # Check with default_params
    conv = galsim.AutoConvolve(psf, gsparams=default_params)
    conv.draw(myImg1)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Using AutoConvolve with default_params disagrees with expected result")
    conv = galsim.AutoConvolve(psf, gsparams=galsim.GSParams())
    conv.draw(myImg1)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Using AutoConvolve with GSParams() disagrees with expected result")

    # For a symmetric profile, AutoCorrelate is the same thing:
    conv2 = galsim.AutoCorrelate(psf)
    conv2.draw(myImg2)
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Moffat convolved with self disagrees with AutoCorrelate result")

    # And check AutoCorrelate with gsparams:
    conv2 = galsim.AutoCorrelate(psf, gsparams=default_params)
    conv2.draw(myImg1)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Using AutoCorrelate with default_params disagrees with expected result")
    conv2 = galsim.AutoCorrelate(psf, gsparams=galsim.GSParams())
    conv2.draw(myImg1)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Using AutoCorrelate with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(conv2,myImg2,"AutoConvolve(Moffat)")

    # Also check AutoConvolve with an asymmetric profile.
    # (AutoCorrelate with this profile is done below...)
    obj1 = galsim.Gaussian(sigma=3., flux=4)
    obj1.applyShift(-0.2, -0.4)
    obj2 = galsim.Gaussian(sigma=6., flux=1.3)
    obj2.applyShift(0.3, 0.3)
    add = galsim.Add(obj1, obj2)
    conv = galsim.Convolve([add, add])
    conv.draw(myImg1)
    corr = galsim.AutoConvolve(add)
    corr.draw(myImg2)
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Asymmetric sum of Gaussians convolved with self disagrees with "+
            "AutoConvolve result")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_autocorrelate():
    """Test that auto-correlation works the same as convolution with the mirror image of itself.

    (See the Signal processing Section of http://en.wikipedia.org/wiki/Autocorrelation)
    """
    import time
    t1 = time.time()
    # Use a function that is NOT two-fold rotationally symmetric, e.g. two different flux Gaussians
    myGauss1 = galsim.SBGaussian(sigma=3., flux=4)
    myGauss1.applyShift(-0.2, -0.4)
    myGauss2 = (galsim.SBGaussian(sigma=6., flux=1.3))
    myGauss2.applyShift(0.3, 0.3)
    mySBP1 = galsim.SBAdd([myGauss1, myGauss2])
    mySBP2 = galsim.SBAdd([myGauss1, myGauss2])
    # Here we rotate by 180 degrees to create mirror image
    mySBP2.applyRotation(180. * galsim.degrees)
    myConv = galsim.SBConvolve([mySBP1, mySBP2])
    myImg1 = galsim.ImageF(80,80, scale=0.7)
    myConv.draw(myImg1.view())
    myAutoCorr = galsim.SBAutoCorrelate(mySBP1)
    myImg2 = galsim.ImageF(80,80, scale=0.7)
    myAutoCorr.draw(myImg2.view())
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Asymmetric sum of Gaussians convolved with mirror of self disagrees with "+
            "SBAutoCorrelate result")

    # Repeat with the GSObject version of this:
    obj1 = galsim.Gaussian(sigma=3., flux=4)
    obj1.applyShift(-0.2, -0.4)
    obj2 = galsim.Gaussian(sigma=6., flux=1.3)
    obj2.applyShift(0.3, 0.3)
    add1 = galsim.Add(obj1, obj2)
    add2 = (galsim.Add(obj1, obj2)).createRotated(180. * galsim.degrees)
    conv = galsim.Convolve([add1, add2])
    conv.draw(myImg1)
    corr = galsim.AutoCorrelate(add1)
    corr.draw(myImg2)
    printval(myImg1, myImg2)
    np.testing.assert_array_almost_equal(
            myImg1.array, myImg2.array, 4,
            err_msg="Asymmetric sum of Gaussians convolved with mirror of self disagrees with "+
            "AutoCorrelate result")

    # Test photon shooting.
    do_shoot(corr,myImg2,"AutoCorrelate")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



if __name__ == "__main__":
    test_convolve()
    test_convolve_flux_scaling()
    test_shearconvolve()
    test_realspace_convolve()
    test_realspace_distorted_convolve()
    test_realspace_shearconvolve()
    test_add()
    test_add_flux_scaling()
    test_autoconvolve()
    test_autocorrelate()
