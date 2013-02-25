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

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images. 

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def printval(image1, image2):
    print "New, saved array sizes: ", np.shape(image1.array), np.shape(image2.array)
    print "Sum of values: ", np.sum(image1.array), np.sum(image2.array)
    print "Minimum image value: ", np.min(image1.array), np.min(image2.array)
    print "Maximum image value: ", np.max(image1.array), np.max(image2.array)
    print "Peak location: ", image1.array.argmax(), image2.array.argmax()
    print "Moments Mx, My, Mxx, Myy, Mxy for new array: "
    getmoments(image1)
    print "Moments Mx, My, Mxx, Myy, Mxy for saved array: "
    getmoments(image2)

def getmoments(image1):
    xgrid, ygrid = np.meshgrid(np.arange(np.shape(image1.array)[0]) + image1.getXMin(), 
                               np.arange(np.shape(image1.array)[1]) + image1.getYMin())
    mx = np.mean(xgrid * image1.array) / np.mean(image1.array)
    my = np.mean(ygrid * image1.array) / np.mean(image1.array)
    mxx = np.mean(((xgrid-mx)**2) * image1.array) / np.mean(image1.array)
    myy = np.mean(((ygrid-my)**2) * image1.array) / np.mean(image1.array)
    mxy = np.mean((xgrid-mx) * (ygrid-my) * image1.array) / np.mean(image1.array)
    print "    ", mx-image1.getXMin(), my-image1.getYMin(), mxx, myy, mxy

def funcname():
    import inspect
    return inspect.stack()[1][3]

# define a series of tests

def test_shapelet():
    """Test the basic properties of the Shapelet class
    """
    import time
    t1 = time.time()

    ftypes = [np.float32, np.float64]
    dx = 0.2

    # First, a Shapelet with only b_00 = 1 should be identically a Gaussian
    for array_type in ftypes:
        im1 = galsim.Image[array_type](64,64)
        im2 = galsim.Image[array_type](64,64)
        for flux in [1., 20., 0.04]:
            for sigma in [1., 0.6, 2.4]:
                gauss = galsim.Gaussian(flux=flux, sigma=sigma)
                gauss.draw(im1, dx=0.2)
                for order in [0, 2, 8]:
                    bvec = np.zeros(galsim.LVectorSize(order))
                    bvec[0] = flux
                    shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
                    shapelet.draw(im2, dx=0.2)
                    np.testing.assert_array_almost_equal(
                            im1.array, im2.array, 5,
                            err_msg="Shapelet with (only) b00=1 disagrees with Gaussian result"
                            "for type=%s, flux=%f, sigma=%f, order=%d"%(
                                str(array_type),flux,sigma,order))

    # Test expected behavior for a non-trivial shapelet profile.
    pix = galsim.Pixel(dx)
    for array_type in ftypes:
        im = galsim.Image[array_type](128,128)
        for sigma in [1., 0.6, 2.4]:
            for order in [0, 2, 8]:
                shapelet = galsim.Shapelet(sigma=sigma, order=order)
                shapelet.setNM(0,0,1.)
                for n in range(1,order+1):
                    if n%2 == 0 and n > 2:
                        shapelet.setNM(n,0,0.2/(n*n))
                    if n >= 2:
                        shapelet.setNM(n,2,0.1/n)
                print 'shapelet vector = ',shapelet.getBVec()

                # Test normalization  (This is normally part of do_shoot.  When we eventually 
                # implement photon shooting, we should go back to the normal do_shoot call, 
                # and remove this section.)
                for test_flux in [1., 20., 0.04]:
                    shapelet.setFlux(test_flux)
                    print 'shapelet flux = ',shapelet.getFlux()
                    shapelet.draw(im, dx=dx)

                    # Need to convolve with a pixel if we want the flux to come out right.
                    conv = galsim.Convolve([pix,shapelet])

                    conv.draw(im, normalization="surface brightness")
                    print 'img.sum = ',im.array.sum(),'  cf. ',test_flux/(dx*dx)
                    np.testing.assert_almost_equal(im.array.sum() * dx*dx / test_flux, 1., 4,
                            err_msg="Surface brightness normalization for Shapelet "
                            "disagrees with expected result")

                    conv.draw(im, normalization="flux")
                    print 'im.sum = ',im.array.sum(),'  cf. ',test_flux
                    np.testing.assert_almost_equal(im.array.sum() / test_flux, 1., 4,
                            err_msg="Flux normalization for Shapelet "
                            "disagrees with expected result")


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_properties():
    """Test some basic properties of the SBGaussian profile.
    """
    import time
    t1 = time.time()
    psf = galsim.SBGaussian(flux=1, sigma=1)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_equal(psf.maxK(), 3.7169221888498383)
    np.testing.assert_almost_equal(psf.stepK(), 0.78539816339744828)
    np.testing.assert_equal(psf.kValue(cen), 1+0j)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        psfFlux = galsim.SBGaussian(flux=inFlux, sigma=2.)
        outFlux = psfFlux.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.15915494309189535)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shapelet_smallshear():
    """Test the application of a small shear to a Gaussian SBProfile against a known result.
    """
    import time
    t1 = time.time()
    e1 = 0.02
    e2 = 0.02
    myShear = galsim.Shear(e1=e1, e2=e2)
    myEllipse = galsim.Ellipse(e1=e1, e2=e2)
    # test the SBProfile version using applyShear
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    mySBP.applyShear(myShear._shear)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Small-shear Gaussian profile disagrees with expected result")
    # test the SBProfile version using applyTransformation
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    mySBP.applyTransformation(myEllipse._ellipse)
    myImg.setZero()
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Small-shear Gaussian profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.applyShear(myShear)
    gauss.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = gauss.createSheared(myShear)
    gauss2.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createSheared disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.applyTransformation(myEllipse)
    gauss.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = gauss.createTransformed(myEllipse)
    gauss2.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createTransformed disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(gauss,myImg,"sheared Gaussian")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_largeshear():
    """Test the application of a large shear to a Sersic SBProfile against a known result.
    """
    import time
    t1 = time.time()
    e1 = 0.0
    e2 = 0.5

    myShear = galsim.Shear(e1=e1, e2=e2)
    myEllipse = galsim.Ellipse(e1=e1, e2=e2)
    # test the SBProfile version using applyShear
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_largeshear.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP = galsim.SBDeVaucouleurs(flux=1, half_light_radius=1)
    mySBP.applyShear(myShear._shear)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Large-shear DeVaucouleurs profile disagrees with expected result")
    # test the SBProfile version using applyTransformation
    mySBP = galsim.SBDeVaucouleurs(flux=1, half_light_radius=1)
    mySBP.applyTransformation(myEllipse._ellipse)
    myImg.setZero()
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Large-shear DeVaucouleurs profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc.applyShear(myShear)
    devauc.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc2 = devauc.createSheared(myShear)
    devauc2.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createSheared disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc.applyTransformation(myEllipse)
    devauc.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc2 = devauc.createTransformed(myEllipse)
    devauc2.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createTransformed disagrees with expected result")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    devauc2 = galsim.Convolve(devauc, galsim.Gaussian(sigma=0.3))
    do_shoot(devauc2,myImg,"sheared DeVauc")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

 
def test_shapelet_convolve():
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
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
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
        conv.draw(myImg,dx=0.2, normalization="surface brightness")
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")

        # Other ways to do the convolution:
        conv = galsim.Convolve(psf,pixel,real_space=False)
        conv.draw(myImg,dx=0.2, normalization="surface brightness")
        np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 4,
                err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(conv,myImg,"Moffat * Pixel")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_shearconvolve():
    """Test the convolution of a sheared Gaussian and a Box SBProfile against a known result.
    """
    import time
    t1 = time.time()
    e1 = 0.04
    e2 = 0.0
    myShear = galsim.Shear(e1=e1, e2=e2)
    myEllipse = galsim.Ellipse(e1=e1, e2=e2)
    # test at SBProfile level using applyShear
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    mySBP.applyShear(myShear._shear)
    mySBP2 = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    myConv = galsim.SBConvolve([mySBP,mySBP2])
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear_convolve_box.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    myConv.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Sheared Gaussian convolved with Box SBProfile disagrees with expected result")

    # test at SBProfile level using applyTransformation
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    mySBP.applyTransformation(myEllipse._ellipse)
    mySBP2 = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    myConv = galsim.SBConvolve([mySBP,mySBP2])
    myImg.setZero()
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
    conv.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    conv2 = galsim.Convolve([psf2,pixel])
    conv2.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    psf = galsim.Gaussian(flux=1, sigma=1)
    psf2 = psf.createTransformed(myEllipse)
    psf.applyTransformation(myEllipse)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel])
    conv2 = galsim.Convolve([psf2,pixel])
    conv.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    conv2.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")

    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel)
    conv.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(conv,myImg,"sheared Gaussian * Pixel")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_rotate():
    """Test the 45 degree rotation of a sheared Sersic profile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBSersic(n=2.5, flux=1, half_light_radius=1)
    myShear = galsim.Shear(e1=0.2, e2=0.0)
    myEllipse = galsim.Ellipse(e1=0.2, e2=0.0)
    mySBP.applyTransformation(myEllipse._ellipse)
    mySBP.applyRotation(45.0 * galsim.degrees)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_ellip_rotated.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="45-degree rotated elliptical Gaussian disagrees with expected result")

    # Repeat with the GSObject version of this:
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1)
    gal.applyTransformation(myEllipse);
    gal.applyRotation(45.0 * galsim.degrees)
    gal.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    gal2 = galsim.Convolve(gal, galsim.Gaussian(sigma=0.3))
    do_shoot(gal2,myImg,"rotated sheared Sersic")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_mag():
    """Test the magnification (size x 1.5) of an exponential profile against a known result.
    """
    import time
    t1 = time.time()
    re = 1.0
    r0 = re/1.67839
    mySBP = galsim.SBExponential(flux=1, scale_radius=r0)
    myEll = galsim.Ellipse(np.log(1.5))
    mySBP.applyTransformation(myEll._ellipse)
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_mag.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Magnification (x1.5) of exponential SBProfile disagrees with expected result")

    # Repeat with the GSObject version of this:
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyTransformation(myEll)
    gal.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation disagrees with expected result")

    # Use applyDilation
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyDilation(1.5)
    gal.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    gal.scaleFlux(1.5**2) # Apply the flux magnification.
    gal.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDilation disagrees with expected result")
 
    # Use applyMagnification
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyMagnification(1.5)
    gal.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyMagnification disagrees with expected result")

    # Use createDilated
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal2 = gal.createDilated(1.5)
    gal2.scaleFlux(1.5**2) # Apply the flux magnification.
    gal2.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createDilated disagrees with expected result")
 
    # Use createMagnified
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal2 = gal.createMagnified(1.5)
    gal2.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createMagnified disagrees with expected result")
 
    # Test photon shooting.
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyMagnification(1.5)
    do_shoot(gal,myImg,"dilated Exponential")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_add():
    """Test the addition of two rescaled Gaussian profiles against a known double Gaussian result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBGaussian(flux=0.75, sigma=1)
    mySBP2 = galsim.SBGaussian(flux=0.25, sigma=3)
    myAdd = galsim.SBAdd(mySBP, mySBP2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "double_gaussian.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    myAdd.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Addition of two rescaled Gaussian profiles disagrees with expected result")

    # Repeat with the GSObject version of this:
    gauss1 = galsim.Gaussian(flux=0.75, sigma=1)
    gauss2 = galsim.Gaussian(flux=0.25, sigma=3)
    sum = galsim.Add(gauss1,gauss2)
    sum.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) disagrees with expected result")

    # Other ways to do the sum:
    sum = gauss1 + gauss2
    sum.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject gauss1 + gauss2 disagrees with expected result")
    sum = gauss1.copy()
    sum += gauss2
    sum.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum = gauss1; sum += gauss2 disagrees with expected result")
    sum = galsim.Add([gauss1,gauss2])
    sum.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add([gauss1,gauss2]) disagrees with expected result")
    gauss1 = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = galsim.Gaussian(flux=1, sigma=3)
    sum = 0.75 * gauss1 + 0.25 * gauss2
    sum.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject 0.75 * gauss1 + 0.25 * gauss2 disagrees with expected result")
    sum = 0.75 * gauss1
    sum += 0.25 * gauss2
    sum.draw(myImg,dx=0.2, normalization="surface brightness")
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum += 0.25 * gauss2 disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(sum,myImg,"sum of 2 Gaussians")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_shift():
    """Test the translation of a Box profile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBBox(xw=0.2, yw=0.2, flux=1)
    mySBP.applyShift(0.2, -0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_shift.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Shifted box profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    pixel = galsim.Pixel(xw=0.2, yw=0.2)
    pixel.applyShift(0.2, -0.2)
    pixel.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShift disagrees with expected result")
    pixel = galsim.Pixel(xw=0.2, yw=0.2)
    pixel.applyTransformation(galsim.Ellipse(galsim.PositionD(0.2, -0.2)))
    pixel.draw(myImg,dx=0.2, normalization="surface brightness")
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(pixel,myImg,"shifted Box")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_shapelet()
    #test_shapelet_properties()
    #test_shapelet_radii()
    #test_shapelet_smallshear()
    #test_shapelet_largeshear()
    #test_shapelet_convolve()
    #test_shapelet_shearconvolve()
    #test_shapelet_rotate()
    #test_shapelet_mag()
    #test_shapelet_add()
    #test_shapelet_shift()
