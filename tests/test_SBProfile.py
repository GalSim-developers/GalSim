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

# For photon shooting, we calculate the number of photons to use based on the target
# accuracy we are shooting for.  (Pun intended.)
# For each pixel,
# uncertainty = sqrt(N_pix) * flux_photon = sqrt(N_tot * flux_pix / flux_tot) * flux_tot / N_tot
#             = sqrt(flux_pix) * sqrt(flux_tot) / sqrt(N_tot)
# This is largest for the brightest pixel.  So we use:
# N = flux_max * flux_tot / photon_shoot_accuracy^2
photon_shoot_accuracy = 2.e-3
# The number of decimal places at which to test the photon shooting
photon_decimal_test = 2

# for radius tests - specify half-light-radius, FHWM, sigma to be compared with high-res image (with
# pixel scale chosen iteratively until convergence is achieved, beginning with test_dx)
test_hlr = 1.8
test_fwhm = 1.8
test_sigma = 1.8
test_scale = 1.8
test_sersic_n = [1.5, 2.5]
test_sersic_trunc = [0., 8.5]

# for flux normalization tests
test_flux = 1.8

# Use a deterministic random number generator so we don't fail tests because of rare flukes
# in the random numbers.
glob_ud = galsim.UniformDeviate(12345)

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


# define some functions to carry out computations that are carried out by several of the tests

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
    #xcen = image2.array.shape[0]/2
    #ycen = image2.array.shape[1]/2
    #print "new image.center = ",image1.array[xcen-3:xcen+4,ycen-3:ycen+4]
    #print "saved image.center = ",image2.array[xcen-3:xcen+4,ycen-3:ycen+4]

def getmoments(image1):
    xgrid, ygrid = np.meshgrid(np.arange(np.shape(image1.array)[0]) + image1.getXMin(), 
                               np.arange(np.shape(image1.array)[1]) + image1.getYMin())
    mx = np.mean(xgrid * image1.array) / np.mean(image1.array)
    my = np.mean(ygrid * image1.array) / np.mean(image1.array)
    mxx = np.mean(((xgrid-mx)**2) * image1.array) / np.mean(image1.array)
    myy = np.mean(((ygrid-my)**2) * image1.array) / np.mean(image1.array)
    mxy = np.mean((xgrid-mx) * (ygrid-my) * image1.array) / np.mean(image1.array)
    print "    ", mx-image1.getXMin(), my-image1.getYMin(), mxx, myy, mxy

def convertToShear(e1,e2):
    # Convert a distortion (e1,e2) to a shear (g1,g2)
    import math
    e = math.sqrt(e1*e1 + e2*e2)
    g = math.tanh( 0.5 * math.atanh(e) )
    g1 = e1 * (g/e)
    g2 = e2 * (g/e)
    return (g1,g2)

def do_shoot(prof, img, name):
    print 'Start do_shoot'
    # Test photon shooting for a particular profile (given as prof). 
    # Since shooting implicitly convolves with the pixel, we need to compare it to 
    # the given profile convolved with a pixel.
    pix = galsim.Pixel(xw=img.getScale())
    compar = galsim.Convolve(prof,pix)
    compar.draw(img)
    flux_max = img.array.max()
    print 'prof.getFlux = ',prof.getFlux()
    print 'compar.getFlux = ',compar.getFlux()
    print 'flux_max = ',flux_max
    flux_tot = img.array.sum()
    print 'flux_tot = ',flux_tot
    if flux_max > 1.:
        # Since the number of photons required for a given accuracy level (in terms of 
        # number of decimal places), we rescale the comparison by the flux of the 
        # brightest pixel.
        compar /= flux_max
        img /= flux_max
        prof /= flux_max
        # The formula for number of photons needed is:
        # nphot = flux_max * flux_tot / photon_shoot_accuracy**2
        # But since we rescaled the image by 1/flux_max, it becomes
        nphot = flux_tot / flux_max / photon_shoot_accuracy**2
    elif flux_max < 0.1:
        # If the max is very small, at least bring it up to 0.1, so we are testing something.
        scale = 0.1 / flux_max;
        print 'scale = ',scale
        compar *= scale
        img *= scale
        prof *= scale
        nphot = flux_max * flux_tot * scale * scale / photon_shoot_accuracy**2
    else:
        nphot = flux_max * flux_tot / photon_shoot_accuracy**2
    print 'prof.getFlux => ',prof.getFlux()
    print 'compar.getFlux => ',compar.getFlux()
    print 'img.sum => ',img.array.sum()
    print 'img.max => ',img.array.max()
    print 'nphot = ',nphot
    img2 = img.copy()
    prof.drawShoot(img2, n_photons=nphot, poisson_flux=False, rng=glob_ud)
    print 'img2.sum => ',img2.array.sum()
    np.testing.assert_array_almost_equal(
            img2.array, img.array, photon_decimal_test,
            err_msg="Photon shooting for %s disagrees with expected result"%name)

    # Test normalization
    dx = img.getScale()
    # Test with a large image to make sure we capture enough of the flux
    # even for slow convergers like Airy (which needs a _very_ large image) or Sersic.
    if 'Airy' in name:
        img = galsim.ImageD(2048,2048)
    elif 'Sersic' in name or 'DeVauc' in name:
        img = galsim.ImageD(512,512)
    else:
        img = galsim.ImageD(128,128)
    img.setScale(dx)
    compar.setFlux(test_flux)
    compar.draw(img, normalization="surface brightness")
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux/(dx*dx)
    np.testing.assert_almost_equal(img.array.sum() * dx*dx, test_flux, 4,
            err_msg="Surface brightness normalization for %s disagrees with expected result"%name)
    compar.draw(img, normalization="flux")
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux
    np.testing.assert_almost_equal(img.array.sum(), test_flux, 4,
            err_msg="Flux normalization for %s disagrees with expected result"%name)

    prof.setFlux(test_flux)
    scale = test_flux / flux_tot # from above
    nphot *= scale * scale
    print 'nphot -> ',nphot
    if 'InterpolatedImage' in name:
        nphot *= 10
        print 'nphot -> ',nphot
    prof.drawShoot(img, n_photons=nphot, normalization="surface brightness", poisson_flux=False,
                   rng=glob_ud)
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux/(dx*dx)
    np.testing.assert_almost_equal(img.array.sum() * dx*dx, test_flux, photon_decimal_test,
            err_msg="Photon shooting SB normalization for %s disagrees with expected result"%name)
    prof.drawShoot(img, n_photons=nphot, normalization="flux", poisson_flux=False,
                   rng=glob_ud)
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux
    np.testing.assert_almost_equal(img.array.sum(), test_flux, photon_decimal_test,
            err_msg="Photon shooting flux normalization for %s disagrees with expected result"%name)


def radial_integrate(prof, minr, maxr, dr):
    """A simple helper that calculates int 2pi r f(r) dr, from rmin to rmax
       for an axially symmetric profile.
    """
    import math
    assert prof.isAxisymmetric()
    r = minr
    sum = 0.
    while r < maxr:
        sum += r * prof.xValue(galsim.PositionD(r,0)) 
        r += dr
    sum *= 2. * math.pi * dr
    return sum
 
def funcname():
    import inspect
    return inspect.stack()[1][3]

# define a series of tests

def test_gaussian():
    """Test the generation of a specific Gaussian profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_1.fits"))
    savedImg.setCenter(0,0)
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setCenter(0,0)
    dx = 0.2
    myImg.setScale(dx)
    tot = mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Gaussian profile disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.sum() *dx**2, tot, 5,
            err_msg="Gaussian profile SBProfile::draw returned wrong tot")

    # Repeat with the GSObject version of this:
    gauss = galsim.Gaussian(flux=1, sigma=1)
    # Reference images were made with old centering, which is equivalent to use_true_center=False.
    myImg = gauss.draw(myImg, dx=dx, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian disagrees with expected result")
    np.testing.assert_almost_equal(
            myImg.array.sum() *dx**2, myImg.added_flux, 5,
            err_msg="Gaussian profile GSObject::draw returned wrong added_flux")

    # Check a non-square image
    print myImg.bounds
    recImg = galsim.ImageF(45,66)
    recImg.setCenter(0,0)
    recImg = gauss.draw(recImg, dx=dx, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            recImg[savedImg.bounds].array, savedImg.array, 5,
            err_msg="Drawing Gaussian on non-square image disagrees with expected result")
    np.testing.assert_almost_equal(
            recImg.array.sum() *dx**2, recImg.added_flux, 5,
            err_msg="Gaussian profile GSObject::draw on non-square image returned wrong added_flux")

    # Check with default_params
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=default_params)
    gauss.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian with default_params disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1, gsparams=galsim.GSParams())
    gauss.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(gauss,myImg,"Gaussian")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_gaussian_properties():
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

def test_gaussian_radii():
    """Test initialization of Gaussian with different types of radius specification.
    """
    import time
    t1 = time.time()
    import math
    # Test constructor using half-light-radius:
    test_gal = galsim.Gaussian(flux = 1., half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Gaussian constructor with half-light radius")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Gaussian initialized with half-light radius")

    # test that getSigma() method provides correct sigma
    got_sigma = test_gal.getSigma()
    test_sigma_ratio = (test_gal.xValue(galsim.PositionD(got_sigma, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'sigma ratio = ', test_sigma_ratio
    np.testing.assert_almost_equal(
            test_sigma_ratio, math.exp(-0.5), decimal=4,
            err_msg="Error in sigma for Gaussian initialized with half-light radius")

    # Test constructor using sigma:
    test_gal = galsim.Gaussian(flux = 1., sigma = test_sigma)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_sigma,0)) / center
    print 'sigma ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, np.exp(-0.5), decimal=4,
            err_msg="Error in Gaussian constructor with sigma")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with sigma) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Gaussian initialized with sigma.")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Gaussian initialized with sigma.")

    # Test constructor using FWHM:
    test_gal = galsim.Gaussian(flux = 1., fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Gaussian constructor with fwhm")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with fwhm) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Gaussian initialized with FWHM.")

    # test that getSigma() method provides correct sigma
    got_sigma = test_gal.getSigma()
    test_sigma_ratio = (test_gal.xValue(galsim.PositionD(got_sigma, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'sigma ratio = ', test_sigma_ratio
    np.testing.assert_almost_equal(
            test_sigma_ratio, math.exp(-0.5), decimal=4,
            err_msg="Error in sigma for Gaussian initialized with FWHM.")

    # Check that the getters don't work after modifying the original.
    # Note: I test all the modifiers here.  For the rest of the profile types, I'll
    # just confirm that it is true of applyShear.  I don't think that has any chance
    # of missing anything.
    test_gal_flux1 = test_gal.copy()
    print 'fwhm = ',test_gal_flux1.getFWHM()
    print 'hlr = ',test_gal_flux1.getHalfLightRadius()
    print 'sigma = ',test_gal_flux1.getSigma()
    test_gal_flux1.setFlux(3.)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux1, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux1, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux1, "getSigma")
    except ImportError:
        # assert_raises requires nose, which we don't want to force people to install.
        # So if they are running this without nose, we just skip these tests.
        pass

    test_gal_flux2 = test_gal.copy()
    print 'fwhm = ',test_gal_flux2.getFWHM()
    print 'hlr = ',test_gal_flux2.getHalfLightRadius()
    print 'sigma = ',test_gal_flux2.getSigma()
    test_gal_flux2.setFlux(3.)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux2, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux2, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_flux2, "getSigma")
    except ImportError:
        pass

    test_gal_shear = test_gal.copy()
    print 'fwhm = ',test_gal_shear.getFWHM()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'sigma = ',test_gal_shear.getSigma()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getSigma")
    except ImportError:
        pass

    test_gal_rot = test_gal.copy()
    print 'fwhm = ',test_gal_rot.getFWHM()
    print 'hlr = ',test_gal_rot.getHalfLightRadius()
    print 'sigma = ',test_gal_rot.getSigma()
    test_gal_rot.applyRotation(theta = 0.5 * galsim.radians)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_rot, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_rot, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_rot, "getSigma")
    except ImportError:
        pass

    test_gal_shift = test_gal.copy()
    print 'fwhm = ',test_gal_shift.getFWHM()
    print 'hlr = ',test_gal_shift.getHalfLightRadius()
    print 'sigma = ',test_gal_shift.getSigma()
    test_gal_shift.applyShift(dx=0.11, dy=0.04)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shift, "getFWHM")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shift, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shift, "getSigma")
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_exponential():
    """Test the generation of a specific exp profile using SBProfile against a known result. 
    """
    import time
    t1 = time.time()
    re = 1.0
    # Note the factor below should really be 1.6783469900166605, but the value of 1.67839 is
    # retained here as it was used by SBParse to generate the original known result (this changed
    # in commit b77eb05ab42ecd31bc8ca03f1c0ae4ee0bc0a78b.
    # The value of this test for regression purposes is not harmed by retaining the old scaling, it
    # just means that the half light radius chosen for the test is not really 1, but 0.999974...
    r0 = re/1.67839
    mySBP = galsim.SBExponential(flux=1., scale_radius=r0)
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Exponential profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    expon = galsim.Exponential(flux=1., scale_radius=r0)
    expon.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential disagrees with expected result")

    # Check with default_params
    expon = galsim.Exponential(flux=1., scale_radius=r0, gsparams=default_params)
    expon.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential with default_params disagrees with expected result")
    expon = galsim.Exponential(flux=1., scale_radius=r0, gsparams=galsim.GSParams())
    expon.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(expon,myImg,"Exponential")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_exponential_radii():
    """Test initialization of Exponential with different types of radius specification.
    """
    import time
    t1 = time.time() 
    import math
    # Test constructor using half-light-radius:
    test_gal = galsim.Exponential(flux = 1., half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Exponential constructor with half-light radius")

    # then test scale getter
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_gal.getScaleRadius(),0)) / center
    print 'scale ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, np.exp(-1.0), decimal=4,
            err_msg="Error in getScaleRadius for Exponential constructed with half light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Exponential(flux = 1., scale_radius = test_scale)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale,0)) / center
    print 'scale ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, np.exp(-1.0), decimal=4,
            err_msg="Error in Exponential constructor with scale")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Exponential initialized with scale_radius.")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.copy()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'scale = ',test_gal_shear.getScaleRadius()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getScaleRadius")
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sersic():
    """Test the generation of a specific Sersic profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()

    # Test SBSersic
    mySBP = galsim.SBSersic(n=3, flux=1, half_light_radius=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Sersic profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic disagrees with expected result")

    # Check with default_params
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=default_params)
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic with default_params disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic with GSParams() disagrees with expected result")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,"Sersic")

    # Now repeat everything using a truncation.  (Above had no truncation.)

    # Test Truncated SBSersic
    mySBP = galsim.SBSersic(n=3, flux=1, half_light_radius=1, trunc=10)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1_10.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Truncated Sersic profile disagrees with expected result")

    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1, trunc=10)
    sersic.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using truncated GSObject Sersic disagrees with expected result")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,"Truncated Sersic")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sersic_radii():
    """Test initialization of Sersic with different types of radius specification.
    """
    import time
    t1 = time.time()
    import math
    for n in test_sersic_n:
        for trunc in test_sersic_trunc:
            # Test constructor using half-light-radius: (only option for sersic)
            test_gal = galsim.Sersic(n=n, half_light_radius=test_hlr, trunc=trunc, flux=1.)
            hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
            print 'hlr_sum = ',hlr_sum
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5, decimal=4,
                    err_msg="Error in Sersic constructor with half-light radius, n=%d, trunc=%d"\
                             %(n,trunc))

            # Test with flux_untruncated=True (above unit tests for flux_untruncated=False)
            test_gal = galsim.Sersic(n=n, half_light_radius=test_hlr, trunc=trunc, flux=1.,
                                     flux_untruncated=True)
            hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
            print 'hlr_sum (truncated and flux_untruncated) = ',hlr_sum
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5, decimal=4,
                    err_msg="Error in Sersic constructor with flux_untruncated, n=%d, trunc=%d"\
                             %(n,trunc))

            # Check that the getters don't work after modifying the original.
            test_gal_shear = test_gal.copy()
            print 'n = ',test_gal_shear.getN()
            print 'hlr = ',test_gal_shear.getHalfLightRadius()
            test_gal_shear.applyShear(g1=0.3, g2=0.1)
            try:
                np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getN");
                np.testing.assert_raises(AttributeError, getattr, test_gal_shear, 
                                         "getHalfLightRadius")
            except ImportError:
                pass

    for n in test_sersic_n:
        for trunc in test_sersic_trunc:
            # Test flux_untruncated scale and normalization
            test_gal = galsim.Sersic(n=n, half_light_radius=test_hlr, flux=1.)

            test_gal2 = galsim.Sersic(n=n, half_light_radius=test_hlr, trunc=trunc, flux=1.,
                                      flux_untruncated=True)
            center = test_gal.xValue(galsim.PositionD(0,0))
            center2 = test_gal2.xValue(galsim.PositionD(0,0))
            ratio = center / center2
            print 'peak value = ', center, center2
            print 'hlr = ', test_gal.getHalfLightRadius(), test_gal2.getHalfLightRadius()
            np.testing.assert_almost_equal(ratio, 1., 9,
                                           "Error in Sersic flux_untruncated=True normalization")

            # Test true HLR with flux_untruncated=True
            true_hlr = test_gal2.getHalfLightRadius()
            hlr_sum = radial_integrate(test_gal, 0., true_hlr, 1.e-4)
            true_flux = test_gal2.getFlux()
            print 'true hlr_sum = ',hlr_sum
            np.testing.assert_almost_equal(
                    hlr_sum, 0.5*true_flux, decimal=4,
                    err_msg="Error in true half-light radius with flux_untruncated, n=%d, trunc=%d"\
                             %(n,trunc))

    # Repeat the above for an explicit DeVaucouleurs.  (Same as n=4, but special name.)
    for trunc in test_sersic_trunc:
        # Test constuctor
        test_gal = galsim.DeVaucouleurs(half_light_radius=test_hlr, trunc=trunc, flux=1.)
        hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
        print 'hlr_sum = ',hlr_sum
        np.testing.assert_almost_equal(
                hlr_sum, 0.5, decimal=4,
                err_msg="Error in DeVaucouleurs constructor with half-light radius, trunc=%d"\
                         %trunc)

        # Check that the getters don't work after modifying the original.
        test_gal_shear = test_gal.copy()
        print 'hlr = ',test_gal_shear.getHalfLightRadius()
        test_gal_shear.applyShear(g1=0.3, g2=0.1)
        try:
            np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        except ImportError:
            pass

    # Test flux_untruncated scale and normalization
    test_gal = galsim.DeVaucouleurs(half_light_radius=test_hlr, trunc=0., flux=1.)
    test_gal2 = galsim.DeVaucouleurs(half_light_radius=test_hlr, trunc=trunc, flux=1.,
                                     flux_untruncated=True)
    center = test_gal.xValue(galsim.PositionD(0,0))
    center2 = test_gal2.xValue(galsim.PositionD(0,0))
    ratio = center / center2
    print 'peak value = ', center, center2
    print 'hlr = ', test_gal.getHalfLightRadius(), test_gal2.getHalfLightRadius()
    np.testing.assert_almost_equal(ratio, 1., 9,
                                   "Error in DeVaucouleurs flux_untruncated=True normalization")

    # Test true HLR with flux_untruncated=True
    true_hlr = test_gal2.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., true_hlr, 1.e-4)
    true_flux = test_gal2.getFlux()
    print 'true hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5*true_flux, decimal=4,
            err_msg="Error in DeVaucouleurs true half-light radius with flux_untruncated, trunc=%d"\
                     %(trunc))

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_airy():
    """Test the generation of a specific Airy profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBAiry(lam_over_diam=1./0.8, obscuration=0.1, flux=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "airy_.8_.1.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Airy profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1)
    airy.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy disagrees with expected result")

    # Check with default_params
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1, gsparams=default_params)
    airy.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy with default_params disagrees with expected result")
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1, gsparams=galsim.GSParams())
    airy.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy with GSParams() disagrees with expected result")

    # Test photon shooting.
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.0, flux=1)
    do_shoot(airy,myImg,"Airy obscuration=0.0")
    airy = galsim.Airy(lam_over_diam=1./0.8, obscuration=0.1, flux=1)
    do_shoot(airy,myImg,"Airy obscuration=0.1")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_airy_radii():
    """Test Airy half light radius and FWHM correctly set and match image.
    """
    import time
    t1 = time.time() 
    import math
    # Test constructor using lam_over_diam: (only option for Airy)
    test_gal = galsim.Airy(lam_over_diam= 1./0.8, flux=1.)
    # test half-light-radius getter
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Airy half-light radius")

    # test FWHM getter
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(.5 * test_gal.getFWHM(),0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in getFWHM() for Airy.")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.copy()
    print 'fwhm = ',test_gal_shear.getFWHM()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'lod = ',test_gal_shear.getLamOverD()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getFWHM");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getLamOverD")
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_box():
    """Test the generation of a specific box profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBBox(xw=1, yw=1, flux=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Box profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    pixel = galsim.Pixel(xw=1, yw=1, flux=1)
    pixel.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel disagrees with expected result")

    # Check with default_params
    pixel = galsim.Pixel(xw=1, yw=1, flux=1, gsparams=default_params)
    pixel.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel with default_params disagrees with expected result")
    pixel = galsim.Pixel(xw=1, yw=1, flux=1, gsparams=galsim.GSParams())
    pixel.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(pixel,myImg,"Pixel")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_moffat():
    """Test the generation of a specific Moffat profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    # Code was formerly:
    # mySBP = galsim.SBMoffat(beta=2, truncationFWHM=5, flux=1, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in 
    # physical units.  However, the same profile can be constructed using 
    # fwhm=1.3178976627539716
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    fwhm_backwards_compatible = 1.3178976627539716
    mySBP = galsim.SBMoffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_2_5.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Moffat profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    moffat = galsim.Moffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1)
    moffat.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat disagrees with expected result")

    # Check with default_params
    moffat = galsim.Moffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1, 
                           gsparams=default_params)
    moffat.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat with default_params disagrees with expected result")
    moffat = galsim.Moffat(beta=2, half_light_radius=1, trunc=5*fwhm_backwards_compatible, flux=1, 
                           gsparams=galsim.GSParams())
    moffat.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat with GSParams() disagrees with expected result")

    # Test photon shooting.
    do_shoot(moffat,myImg,"Moffat")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_moffat_properties():
    """Test some basic properties of the SBMoffat profile.
    """
    import time
    t1 = time.time()
    # Code was formerly:
    # mySBP = galsim.SBMoffat(beta=2.0, truncationFWHM=2, flux=1.8, half_light_radius=1)
    #
    # ...but this is no longer quite so simple since we changed the handling of trunc to be in 
    # physical units.  However, the same profile can be constructed using 
    # fwhm=1.4686232496771867, 
    # as calculated by interval bisection in devutils/external/calculate_moffat_radii.py
    fwhm_backwards_compatible = 1.4686232496771867
    psf = galsim.SBMoffat(beta=2.0, fwhm=fwhm_backwards_compatible,
                          trunc=2*fwhm_backwards_compatible, flux=1.8)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxK(), 11.569262763913111)
    np.testing.assert_almost_equal(psf.stepK(), 1.0695706520648969)
    np.testing.assert_almost_equal(psf.kValue(cen), 1.8+0j)
    np.testing.assert_almost_equal(psf.getHalfLightRadius(), 1.0)
    np.testing.assert_almost_equal(psf.getFWHM(), fwhm_backwards_compatible)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.50654651638242509)

    # Now create the same profile using the half_light_radius:
    psf = galsim.SBMoffat(beta=2.0, half_light_radius=1.,
            trunc=2*fwhm_backwards_compatible, flux=1.8)
    np.testing.assert_equal(psf.centroid(), cen)
    np.testing.assert_almost_equal(psf.maxK(), 11.569262763913111)
    np.testing.assert_almost_equal(psf.stepK(), 1.0695706520648969)
    np.testing.assert_almost_equal(psf.kValue(cen), 1.8+0j)
    np.testing.assert_almost_equal(psf.getHalfLightRadius(), 1.0)
    np.testing.assert_almost_equal(psf.getFWHM(), fwhm_backwards_compatible)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.50654651638242509)

    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        psfFlux = galsim.SBMoffat(2.0, fwhm=fwhm_backwards_compatible,
                                  trunc=2*fwhm_backwards_compatible, flux=inFlux)
        outFlux = psfFlux.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_moffat_radii():
    """Test initialization of Moffat with different types of radius specification.
    """
    import time 
    t1 = time.time()
    import math
    # Test constructor using half-light-radius:
    test_beta = 2.
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with half-light radius")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with half-light radius")

    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with half-light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, scale_radius = test_scale)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale,0)) / center
    print 'scale ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, pow(2,-test_beta), decimal=4,
            err_msg="Error in Moffat constructor with scale")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Moffat initialized with scale radius.")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with scale radius")

    # Test constructor using FWHM:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with fwhm")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with FWHM) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Moffat initialized with FWHM.")
    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with scale radius")

    # Now repeat everything using a severe truncation.  (Above had no truncation.)

    # Test constructor using half-light-radius:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, half_light_radius = test_hlr,
                             trunc=2*test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with half-light radius")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with half-light radius")

    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with half-light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Moffat(flux=1., beta=test_beta, trunc=2*test_scale,
                             scale_radius=test_scale)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale,0)) / center
    print 'scale ratio = ', ratio
    np.testing.assert_almost_equal(
            ratio, pow(2,-test_beta), decimal=4,
            err_msg="Error in Moffat constructor with scale")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (truncated profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for truncated Moffat "+
                    "initialized with scale radius.")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for truncated Moffat initialized with scale radius")

    # Test constructor using FWHM:
    test_gal = galsim.Moffat(flux=1., beta=test_beta, trunc=2.*test_fwhm,
                             fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ', ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with fwhm")

    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (truncated profile initialized with FWHM) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for truncated Moffat initialized with FWHM.")

    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(
            test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for truncated Moffat initialized with scale radius")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.copy()
    print 'beta = ',test_gal_shear.getBeta()
    print 'fwhm = ',test_gal_shear.getFWHM()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'scale = ',test_gal_shear.getScaleRadius()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getBeta");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getFWHM");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius")
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getScaleRadius");
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_kolmogorov():
    """Test the generation of a specific Kolmogorov profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBKolmogorov(lam_over_r0=1.5, flux=1.8)
    # This savedImg was created from the SBKolmogorov implementation in
    # commit c8efd74d1930157b1b1ffc0bfcfb5e1bf6fe3201
    # It would be nice to get an independent calculation here...
    #savedImg = galsim.ImageF(128,128)
    #mySBP.draw(image=savedImg, dx=0.2)
    #savedImg.write(os.path.join(imgdir, "kolmogorov.fits"))
    savedImg = galsim.fits.read(os.path.join(imgdir, "kolmogorov.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
    mySBP.draw(myImg.view())
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Kolmogorov profile disagrees with expected result") 

    # Repeat with the GSObject version of this:
    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=1.8)
    kolm.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Kolmogorov disagrees with expected result")

    # Check with default_params
    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=1.8, gsparams=default_params)
    kolm.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Kolmogorov with default_params disagrees with expected result")
    kolm = galsim.Kolmogorov(lam_over_r0=1.5, flux=1.8, gsparams=galsim.GSParams())
    kolm.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Kolmogorov with GSParams() disagrees with expected result")

    # Test equivalence when convolved by an effective delta function
    # This tests the equivalence between xValue and kValue calculations.
    delta = galsim.Gaussian(sigma=1.e-8)
    conv = galsim.Convolve([kolm,delta])
    conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 3,
            err_msg="Kolmogorov * delta disagrees with expected result")

    # Test photon shooting.
    do_shoot(kolm,myImg,"Kolmogorov")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_kolmogorov_properties():
    """Test some basic properties of the Kolmogorov profile.
    """
    import time
    t1 = time.time()

    lor = 1.5
    flux = 1.8
    psf = galsim.Kolmogorov(lam_over_r0=lor, flux=flux)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxK(), 8.6440505245909858, 9)
    np.testing.assert_almost_equal(psf.stepK(), 0.3437479193077736, 9)
    np.testing.assert_almost_equal(psf.kValue(cen), flux+0j)
    np.testing.assert_almost_equal(psf.getLamOverR0(), lor)
    np.testing.assert_almost_equal(psf.getHalfLightRadius(), lor * 0.554811)
    np.testing.assert_almost_equal(psf.getFWHM(), lor * 0.975865)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.6283160485127478)

    # Check input flux vs output flux
    lors = [1, 0.5, 2, 5]
    for lor in lors:
        psf = galsim.Kolmogorov(lam_over_r0=lor, flux=flux)
        out_flux = psf.getFlux()
        np.testing.assert_almost_equal(out_flux, flux,
                                       err_msg="Flux of Kolmogorov (getFlux) is incorrect.")

        # Also check the realized flux in a drawn image
        dx = lor / 10.
        img = galsim.ImageF(256,256)
        pix = galsim.Pixel(dx)
        conv = galsim.Convolve([psf,pix])
        conv.draw(image=img, dx=dx)
        out_flux = img.array.sum()
        np.testing.assert_almost_equal(out_flux, flux, 3,
                                       err_msg="Flux of Kolmogorov (image array) is incorrect.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_kolmogorov_radii():
    """Test initialization of Kolmogorov with different types of radius specification.
    """
    import time 
    t1 = time.time()
    import math
    # Test constructor using lambda/r0
    lors = [1, 0.5, 2, 5]
    for lor in lors:
        print 'lor = ',lor
        test_gal = galsim.Kolmogorov(flux=1., lam_over_r0=lor)

        np.testing.assert_almost_equal(
                lor, test_gal.getLamOverR0(), decimal=9,
                err_msg="Error in Kolmogorov, lor != getLamOverR0")

        # test that getFWHM() method provides correct FWHM
        got_fwhm = test_gal.getFWHM()
        print 'got_fwhm = ',got_fwhm
        test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
        print 'fwhm ratio = ', test_fwhm_ratio
        np.testing.assert_almost_equal(
                test_fwhm_ratio, 0.5, decimal=4,
                err_msg="Error in FWHM for Kolmogorov initialized with half-light radius")

        # then test that image indeed has the correct HLR properties when radially integrated
        got_hlr = test_gal.getHalfLightRadius()
        print 'got_hlr = ',got_hlr
        hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
        print 'hlr_sum = ',hlr_sum
        np.testing.assert_almost_equal(
                hlr_sum, 0.5, decimal=3,
                err_msg="Error in half light radius for Kolmogorov initialized with lam_over_r0.")

    # Test constructor using half-light-radius:
    test_gal = galsim.Kolmogorov(flux=1., half_light_radius = test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=3,
            err_msg="Error in Kolmogorov constructor with half-light radius")

    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    print 'got_fwhm = ',got_fwhm
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                    test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(
            test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Kolmogorov initialized with half-light radius")

    # Test constructor using FWHM:
    test_gal = galsim.Kolmogorov(flux=1., fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(
            ratio, 0.5, decimal=4,
            err_msg="Error in Kolmogorov constructor with fwhm")

    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    print 'got_hlr = ',got_hlr
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with fwhm) = ',hlr_sum
    np.testing.assert_almost_equal(
            hlr_sum, 0.5, decimal=3,
            err_msg="Error in half light radius for Gaussian initialized with FWHM.")

    # Check that the getters don't work after modifying the original.
    test_gal_shear = test_gal.copy()
    print 'fwhm = ',test_gal_shear.getFWHM()
    print 'hlr = ',test_gal_shear.getHalfLightRadius()
    print 'lor = ',test_gal_shear.getLamOverR0()
    test_gal_shear.applyShear(g1=0.3, g2=0.1)
    try:
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getFWHM");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getHalfLightRadius");
        np.testing.assert_raises(AttributeError, getattr, test_gal_shear, "getLamOverR0");
    except ImportError:
        pass

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_smallshear():
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
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.applyTransformation(myEllipse)
    gauss.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation disagrees with expected result")
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = gauss.createTransformed(myEllipse)
    gauss2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createTransformed disagrees with expected result")
 
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
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc.applyTransformation(myEllipse)
    devauc.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc2 = devauc.createTransformed(myEllipse)
    devauc2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject createTransformed disagrees with expected result")

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
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

 
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


def test_shearconvolve():
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
    conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    conv2 = galsim.Convolve([psf2,pixel])
    conv2.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    psf = galsim.Gaussian(flux=1, sigma=1)
    psf2 = psf.createTransformed(myEllipse)
    psf.applyTransformation(myEllipse)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel])
    conv2 = galsim.Convolve([psf2,pixel])
    conv.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
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
    img = galsim.ImageF(saved_img.bounds)
    img.setScale(0.2)
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
    img = galsim.ImageF(saved_img.bounds)
    img.setScale(0.2)
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
    myEllipse = galsim.Ellipse(e1=e1, e2=e2)
    psf.applyTransformation(myEllipse._ellipse)
    pix = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.SBConvolve([psf,pix],real_space=True)
    saved_img = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear_convolve_box.fits"))
    img = galsim.ImageF(saved_img.bounds)
    img.setScale(0.2)
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

def test_rotate():
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
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation disagrees with expected result")

    # Check with default_params
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1, gsparams=default_params)
    gal.applyTransformation(myEllipse);
    gal.applyRotation(45.0 * galsim.degrees)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation with default_params disagrees with expected "
            "result")
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1, gsparams=galsim.GSParams())
    gal.applyTransformation(myEllipse);
    gal.applyRotation(45.0 * galsim.degrees)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation with GSParams() disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    gal2 = galsim.Convolve(gal, galsim.Gaussian(sigma=0.3))
    do_shoot(gal2,myImg,"rotated sheared Sersic")
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
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation disagrees with expected result")

    # Check with default_params
    gal = galsim.Exponential(flux=1, scale_radius=r0, gsparams=default_params)
    gal.applyTransformation(myEll)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation with default_params disagrees with "
            "expected result")
    gal = galsim.Exponential(flux=1, scale_radius=r0, gsparams=galsim.GSParams())
    gal.applyTransformation(myEll)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation with GSParams() disagrees with "
            "expected result")

    # Use applyDilation
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyDilation(1.5)
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    gal.scaleFlux(1.5**2) # Apply the flux magnification.
    gal.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDilation disagrees with expected result")
 
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
    im = galsim.ImageF(imsize, imsize)
    im.setScale(pix_scale)
    im = ser.draw(im.view())
    im2 = galsim.ImageF(imsize, imsize)
    im2.setScale(pix_scale)
    im2 = ser2.draw(im2.view())
    np.testing.assert_array_almost_equal(im.array, im2.array, 5,
        err_msg="Lensing of Sersic profile done in two different ways gives different answer")

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
    pixel.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShift disagrees with expected result")
    pixel = galsim.Pixel(xw=0.2, yw=0.2)
    pixel.applyTransformation(galsim.Ellipse(galsim.PositionD(0.2, -0.2)))
    pixel.draw(myImg,dx=0.2, normalization="surface brightness", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyTransformation disagrees with expected result")
 
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
    myImg = galsim.ImageF(savedImg.bounds)
    myImg.setScale(0.2)
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
    # With wmult = 3, the calculated flux should come out right so long as we also 
    # convolve by a pixel:
    sersic3 = galsim.Convolve([sersic, galsim.Pixel(xw=0.2)])
    myImg2 = sersic3.draw(dx=0.2, wmult=3, use_true_center=False)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum(), 1., 3,
            err_msg="Drawing with wmult=3 results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux, 1., 3,
            err_msg="Drawing with wmult=3 returned wrong added_flux")
    myImg2 = sersic3.draw(myImg2, add_to_image=True, use_true_center=False)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum(), 2., 3,
            err_msg="Drawing with add_to_image=True results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux, 1., 3,
            err_msg="Drawing with add_to_image=True returned wrong added_flux")

    # Check that the flux works out when adding multiple times.
    gauss = galsim.Gaussian(flux=1.e5, sigma=2.)
    gauss2 = galsim.Convolve([gauss, galsim.Pixel(xw=0.2)])
    myImg2 = gauss2.draw(dx=0.2, wmult=3, use_true_center=False)
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
    myImg2 = gauss.drawShoot(myImg2, add_to_image=True, poisson_flux=False,
                                  rng=glob_ud)
    print myImg2.array.sum(), myImg2.added_flux
    np.testing.assert_almost_equal(myImg2.array.sum()/1.e5, 3., 4,
            err_msg="Drawing Gaussian with drawShoot, add_to_image=True, poisson_flux=False "+
                    "results in wrong flux")
    np.testing.assert_almost_equal(myImg2.added_flux/1.e5, 1., 4,
            err_msg="Drawing Gaussian with drawShoot, add_to_image=True, poisson_flux=False "+
                    "returned wrong added_flux")
    myImg2 = gauss.drawShoot(myImg2, add_to_image=True, rng=glob_ud)
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
    myImg2 = sersic3.draw(dx=0.2, wmult=3, gain=0.5, use_true_center=False)
    np.testing.assert_almost_equal(myImg2.array.sum(), 2., 3,
            err_msg="Drawing with gain=0.5 results in wrong flux")
    myImg2 = sersic3.draw(dx=0.2, wmult=3, gain=4., use_true_center=False)
    np.testing.assert_almost_equal(myImg2.array.sum(), 0.25, 3,
            err_msg="Drawing with gain=4. results in wrong flux")
    # Check add_to_image in conjunction with gain
    sersic3.draw(myImg2, gain=4., add_to_image=True, use_true_center=False)
    np.testing.assert_almost_equal(myImg2.array.sum(), 0.5, 3,
            err_msg="Drawing with gain=4. results in wrong flux")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic3 = galsim.Convolve(sersic2, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic3,myImg,"scaled Sersic")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbinterpolatedimage():
    """Test that we can make SBInterpolatedImages from Images of various types, and convert back.
    """
    import time
    t1 = time.time()
    # for each type, try to make an SBInterpolatedImage, and check that when we draw an image from
    # that SBInterpolatedImage that it is the same as the original
    lan3 = galsim.Lanczos(3, True, 1.E-4)
    lan3_2d = galsim.InterpolantXY(lan3)

    ftypes = [np.float32, np.float64]
    ref_array = np.array([
        [0.01, 0.08, 0.07, 0.02],
        [0.13, 0.38, 0.52, 0.06],
        [0.09, 0.41, 0.44, 0.09],
        [0.04, 0.11, 0.10, 0.01] ]) 

    for array_type in ftypes:
        image_in = galsim.ImageView[array_type](ref_array.astype(array_type))
        np.testing.assert_array_equal(
                ref_array.astype(array_type),image_in.array,
                err_msg="Array from input Image differs from reference array for type %s"%
                        array_type)
        sbinterp = galsim.SBInterpolatedImage(image_in, lan3_2d, dx=1.0)
        test_array = np.zeros(ref_array.shape, dtype=array_type)
        image_out = galsim.ImageView[array_type](test_array)
        image_out.setScale(1.0)
        sbinterp.draw(image_out.view())
        np.testing.assert_array_equal(
                ref_array.astype(array_type),image_out.array,
                err_msg="Array from output Image differs from reference array for type %s"%
                        array_type)
 
        # Lanczos doesn't quite get the flux right.  Wrong at the 5th decimal place.
        # Gary says that's expected -- Lanczos isn't technically flux conserving.  
        # He applied the 1st order correction to the flux, but expect to be wrong at around
        # the 10^-5 level.
        # Anyway, Quintic seems to be accurate enough.
        quint = galsim.Quintic(1.e-4)
        quint_2d = galsim.InterpolantXY(quint)
        sbinterp = galsim.SBInterpolatedImage(image_in, quint_2d, dx=1.0)
        sbinterp.setFlux(1.)
        do_shoot(galsim.GSObject(sbinterp),image_out,"InterpolatedImage")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_draw():
    """Test the various optional parameters to the draw function.
       In particular test the parameters image, dx, and wmult in various combinations.
    """
    import time
    t1 = time.time()

    # We use a simple Exponential for our object:
    obj = galsim.Exponential(flux=test_flux, scale_radius=2)

    # Setup the test to see if this was drawn correctly on the image:
    def CalculateScale(im):
        # We just determine the scale radius of the drawn exponential by calculating 
        # the second moments of the image.
        # int r^2 exp(-r/s) 2pir dr = 12 s^4 pi
        # int exp(-r/s) 2pir dr = 2 s^2 pi
        x, y = np.meshgrid(np.arange(np.shape(im.array)[0]), np.arange(np.shape(im.array)[1]))
        flux = im.array.sum()
        mx = (x * im.array).sum() / flux
        my = (y * im.array).sum() / flux
        mxx = (((x-mx)**2) * im.array).sum() / flux
        myy = (((y-my)**2) * im.array).sum() / flux
        mxy = ((x-mx) * (y-my) * im.array).sum() / flux
        print flux,mx,my,mxx,myy,mxy
        s2 = mxx+myy
        np.testing.assert_almost_equal((mxx-myy)/s2, 0, 5, "Found e1 != 0 for Exponential draw")
        np.testing.assert_almost_equal(2*mxy/s2, 0, 5, "Found e2 != 0 for Exponential draw")
        return np.sqrt(s2/6) * im.scale
 

    # First test draw() with no kwargs.  It should:
    #   - create a new image
    #   - return the new image
    #   - set the scale to obj.nyquistDx()
    #   - set the size large enough to contain 99.5% of the flux
    im1 = obj.draw()
    dx_nyq = obj.nyquistDx()
    np.testing.assert_almost_equal(im1.scale, dx_nyq, 9,
                                   "obj.draw() produced image with wrong scale")
    assert im1.bounds == galsim.BoundsI(1,48,1,48),(
            "obj.draw() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im1), 2, 1,
                                   "Measured wrong scale after obj.draw()")

    # The flux is only really expected to come out right if the object has been
    # convoled with a pixel:
    obj2 = galsim.Convolve([ obj, galsim.Pixel(im1.scale) ])
    im2 = obj2.draw()
    dx_nyq = obj2.nyquistDx()
    np.testing.assert_almost_equal(im2.scale, dx_nyq, 9,
                                   "obj2.draw() produced image with wrong scale")
    np.testing.assert_almost_equal(im2.array.astype(float).sum(), test_flux, 2,
                                   "obj2.draw() produced image with wrong flux")
    assert im2.bounds == galsim.BoundsI(1,48,1,48),(
            "obj2.draw() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im2), 2, 1,
                                   "Measured wrong scale after obj2.draw()")

    # Test if we provide an image argument.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj2.nyquistDx()
    #   - zero out any existing data
    im3 = galsim.ImageD(48,48)
    im4 = obj2.draw(im3)
    np.testing.assert_almost_equal(im3.scale, dx_nyq, 9,
                                   "obj2.draw(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3.array.sum(), test_flux, 2,
                                   "obj2.draw(im3) produced image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), im2.array.astype(float).sum(), 6,
                                   "obj2.draw(im3) produced image with different flux than im2")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj2.draw(im3)")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "im4 = obj2.draw(im3) produced im4 != im3")
    im3.fill(9.8)
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "im4 = obj2.draw(im3) produced im4 is not im3")
    im4 = obj2.draw(im3)
    np.testing.assert_almost_equal(im3.array.sum(), im2.array.astype(float).sum(), 6,
                                   "obj2.draw(im3) doesn't zero out existing data")
    
    # Test if we provide an image with undefined bounds.  It should:
    #   - resize the provided image
    #   - also return that image
    #   - set the scale to obj2.nyquistDx()
    im5 = galsim.ImageD()
    obj2.draw(im5)
    np.testing.assert_almost_equal(im5.scale, dx_nyq, 9,
                                   "obj2.draw(im5) produced image with wrong scale")
    np.testing.assert_almost_equal(im5.array.sum(), test_flux, 2,
                                   "obj2.draw(im5) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im5), 2, 1,
                                   "Measured wrong scale after obj2.draw(im5)")
    np.testing.assert_almost_equal(im5.array.sum(), im2.array.astype(float).sum(), 6,
                                   "obj2.draw(im5) produced image with different flux than im2")
    assert im5.bounds == galsim.BoundsI(1,48,1,48),(
            "obj2.draw(im5) produced image with wrong bounds")

    # Test if we provide wmult.  It should:
    #   - create a new image that is wmult times larger in each direction.
    #   - return the new image
    #   - set the scale to obj2.nyquistDx()
    im6 = obj2.draw(wmult=4.)
    np.testing.assert_almost_equal(im6.scale, dx_nyq, 9,
                                   "obj2.draw(wmult) produced image with wrong scale")
    # Can assert accuracy to 5 decimal places now, since we're capturing much more
    # of the flux on the image.
    np.testing.assert_almost_equal(im6.array.astype(float).sum(), test_flux, 5,
                                   "obj2.draw(wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im6), 2, 2,
                                   "Measured wrong scale after obj2.draw(wmult)")
    assert im6.bounds == galsim.BoundsI(1,190,1,190),(
            "obj2.draw(wmult) produced image with wrong bounds")

    # Test if we provide an image argument and wmult.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj2.nyquistDx()
    #   - zero out any existing data
    #   - the calculation of the convolution should be slightly more accurate than for im3
    im3.setZero()
    im5.setZero()
    obj2.draw(im3, wmult=4.)
    obj2.draw(im5)
    np.testing.assert_almost_equal(im3.scale, dx_nyq, 9,
                                   "obj2.draw(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3.array.sum(), test_flux, 2,
                                   "obj2.draw(im3,wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj2.draw(im3,wmult)")
    assert ((im3.array-im5.array)**2).sum() > 0, (
            "obj2.draw(im3,wmult) produced the same image as without wmult")
    
    # Test if we provide a dx to use.  It should:
    #   - create a new image using that dx for the scale
    #   - return the new image
    #   - set the size large enough to contain 99.5% of the flux
    im7 = obj2.draw(dx=0.51)
    np.testing.assert_almost_equal(im7.scale, 0.51, 9,
                                   "obj2.draw(dx) produced image with wrong scale")
    np.testing.assert_almost_equal(im7.array.astype(float).sum(), test_flux, 2,
                                   "obj2.draw(dx) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im7), 2, 1,
                                   "Measured wrong scale after obj2.draw(dx)")
    assert im7.bounds == galsim.BoundsI(1,60,1,60),(
            "obj2.draw(dx) produced image with wrong bounds")

    # Test with dx and wmult.  It should:
    #   - create a new image using that dx for the scale
    #   - set the size a factor of wmult times larger in each direction.
    #   - return the new image
    im8 = obj2.draw(dx=0.51, wmult=4.)
    np.testing.assert_almost_equal(im8.scale, 0.51, 9,
                                   "obj2.draw(dx,wmult) produced image with wrong scale")
    np.testing.assert_almost_equal(im8.array.astype(float).sum(), test_flux, 5,
                                   "obj2.draw(dx,wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im8), 2, 2,
                                   "Measured wrong scale after obj2.draw(dx,wmult)")
    assert im8.bounds == galsim.BoundsI(1,234,1,234),(
            "obj2.draw(dx,wmult) produced image with wrong bounds")

    # Test if we provide an image with a defined scale.  It should:
    #   - write to the existing image
    #   - use the image's scale 
    im9 = galsim.ImageD(200,200)
    im9.setScale(0.51)
    obj2.draw(im9)
    np.testing.assert_almost_equal(im9.scale, 0.51, 9,
                                   "obj2.draw(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 5,
                                   "obj2.draw(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9)")

    # Test if we provide an image with a defined scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj2.nyquistDx()
    im9.setScale(-0.51)
    im9.setZero()
    obj2.draw(im9)
    np.testing.assert_almost_equal(im9.scale, dx_nyq, 9,
                                   "obj2.draw(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 5,
                                   "obj2.draw(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9)")
    im9.setScale(0)
    im9.setZero()
    obj2.draw(im9)
    np.testing.assert_almost_equal(im9.scale, dx_nyq, 9,
                                   "obj2.draw(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 5,
                                   "obj2.draw(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9)")
    

    # Test if we provide an image and dx.  It should:
    #   - write to the existing image
    #   - use the provided dx
    #   - write the new dx value to the image's scale
    im9.setScale(0.73)
    im9.setZero()
    obj2.draw(im9, dx=0.51)
    np.testing.assert_almost_equal(im9.scale, 0.51, 9,
                                   "obj2.draw(im9,dx) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 5,
                                   "obj2.draw(im9,dx) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9,dx)")

    # Test if we provide an image and dx <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj2.nyquistDx()
    im9.setScale(0.73)
    im9.setZero()
    obj2.draw(im9, dx=-0.51)
    np.testing.assert_almost_equal(im9.scale, dx_nyq, 9,
                                   "obj2.draw(im9,dx<0) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 5,
                                   "obj2.draw(im9,dx<0) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9,dx<0)")
    im9.setScale(0.73)
    im9.setZero()
    obj2.draw(im9, dx=0)
    np.testing.assert_almost_equal(im9.scale, dx_nyq, 9,
                                   "obj2.draw(im9,dx=0) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 5,
                                   "obj2.draw(im9,dx=0) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9,dx=0)")
    
    
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_autoconvolve():
    """Test that auto-convolution works the same as convolution with itself.
    """
    import time
    t1 = time.time()

    mySBP = galsim.SBMoffat(beta=3.8, fwhm=1.3, flux=5)
    myConv = galsim.SBConvolve([mySBP,mySBP])
    myImg1 = galsim.ImageF(80,80)
    myImg1.setScale(0.4)
    myConv.draw(myImg1.view())
    myAutoConv = galsim.SBAutoConvolve(mySBP)
    myImg2 = galsim.ImageF(80,80)
    myImg2.setScale(0.4)
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
    myImg1 = galsim.ImageF(80,80)
    myImg1.setScale(0.7)
    myConv.draw(myImg1.view())
    myAutoCorr = galsim.SBAutoCorrelate(mySBP1)
    myImg2 = galsim.ImageF(80,80)
    myImg2.setScale(0.7)
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
    test_gaussian()
    test_gaussian_properties()
    test_gaussian_radii()
    test_exponential()
    test_exponential_radii()
    test_sersic()
    test_sersic_radii()
    test_airy()
    test_airy_radii()
    test_box()
    test_moffat()
    test_moffat_properties()
    test_moffat_radii()
    test_kolmogorov()
    test_kolmogorov_properties()
    test_kolmogorov_radii()
    test_smallshear()
    test_largeshear()
    test_convolve()
    test_shearconvolve()
    test_realspace_convolve()
    test_realspace_distorted_convolve()
    test_realspace_shearconvolve()
    test_rotate()
    test_mag()
    test_lens()
    test_add()
    test_shift()
    test_rescale()
    test_sbinterpolatedimage()
    test_draw()
    test_autoconvolve()
    test_autocorrelate()
