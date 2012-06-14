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

# Setup info for image tests
testshape = (4, 4)  # shape of image arrays for all tests
ntypes = 4
types = [np.int16, np.int32, np.float32, np.float64]
ftypes = [np.float32, np.float64]
tchar = ['S', 'I', 'F', 'D']
ftchar = ['F', 'D']

ref_array = np.array([[00, 10, 20, 30], [01, 11, 21, 31], [02, 12, 22, 32],
                      [03, 13, 23, 33]]).astype(types[0])

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

def do_shoot(prof, img, dx, name):
    print 'Start do_shoot'
    # Test photon shooting for a particular profile (given as prof). 
    # Since shooting implicitly convolves with the pixel, we need to compare it to 
    # the given profile convolved with a pixel.
    pix = galsim.Pixel(xw=dx)
    compar = galsim.Convolve(prof,pix)
    compar.draw(img,dx=dx)
    flux_max = img.array.max()
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
    else:
        nphot = flux_max * flux_tot / photon_shoot_accuracy**2
    print 'nphot = ',nphot
    img2 = img.copy()
    prof.drawShoot(img2,nphot)
    np.testing.assert_array_almost_equal(
            img2.array, img.array, photon_decimal_test,
            err_msg="Photon shooting for %s disagrees with expected result"%name)

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

def test_sbprofile_gaussian():
    """Test the generation of a specific Gaussian profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
            err_msg="Gaussian profile disagrees with expected result")
    # Repeat with the GSObject version of this:
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian disagrees with expected result")

    # Test photon shooting.
    do_shoot(gauss,myImg,0.2,"Gaussian")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_gaussian_properties():
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
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in Gaussian constructor with half-light radius")
    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Gaussian initialized with half-light radius")
    # test that getSigma() method provides correct sigma
    got_sigma = test_gal.getSigma()
    test_sigma_ratio = (test_gal.xValue(galsim.PositionD(got_sigma, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'sigma ratio = ', test_sigma_ratio
    np.testing.assert_almost_equal(test_sigma_ratio, math.exp(-0.5), decimal=4,
            err_msg="Error in sigma for Gaussian initialized with half-light radius")

    # Test constructor using sigma:
    test_gal = galsim.Gaussian(flux = 1., sigma = test_sigma)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_sigma,0)) / center
    print 'sigma ratio = ',ratio
    np.testing.assert_almost_equal(ratio, np.exp(-0.5), decimal=4,
            err_msg="Error in Gaussian constructor with sigma")
    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with sigma) = ',hlr_sum
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Gaussian initialized with sigma.")
    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Gaussian initialized with sigma.")

    # Test constructor using FWHM:
    test_gal = galsim.Gaussian(flux = 1., fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(ratio, 0.5, decimal=4,
            err_msg="Error in Gaussian constructor with fwhm")
    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with fwhm) = ',hlr_sum
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Gaussian initialized with FWHM.")
    # test that getSigma() method provides correct sigma
    got_sigma = test_gal.getSigma()
    test_sigma_ratio = (test_gal.xValue(galsim.PositionD(got_sigma, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'sigma ratio = ', test_sigma_ratio
    np.testing.assert_almost_equal(test_sigma_ratio, math.exp(-0.5), decimal=4,
            err_msg="Error in sigma for Gaussian initialized with FWHM.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sbprofile_exponential():
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
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
            err_msg="Exponential profile disagrees with expected result") 
    # Repeat with the GSObject version of this:
    expon = galsim.Exponential(flux=1., scale_radius=r0)
    expon.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential disagrees with expected result")

    # Test photon shooting.
    do_shoot(expon,myImg,0.2,"Exponential")
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
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in Exponential constructor with half-light radius")
    # then test scale getter
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_gal.getScaleRadius(),0)) / center
    print 'scale ratio = ',ratio
    np.testing.assert_almost_equal(ratio, np.exp(-1.0), decimal=4,
            err_msg="Error in getScaleRadius for Exponential constructed with half light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Exponential(flux = 1., scale_radius = test_scale)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale,0)) / center
    print 'scale ratio = ',ratio
    np.testing.assert_almost_equal(ratio, np.exp(-1.0), decimal=4,
            err_msg="Error in Exponential constructor with scale")
    # then test that image indeed has the correct HLR properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Exponential initialized with scale_radius.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sbprofile_sersic():
    """Test the generation of a specific Sersic profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBSersic(n=3, flux=1, half_light_radius=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
            err_msg="Sersic profile disagrees with expected result")
    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic disagrees with expected result")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,0.2,"Sersic")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sersic_radii():
    """Test initialization of Sersic with different types of radius specification.
    """
    import time
    t1 = time.time()
    import math
    for n in test_sersic_n:
        # Test constructor using half-light-radius: (only option for sersic)
        test_gal = galsim.Sersic(n=n, half_light_radius=test_hlr, flux=1.)
        hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
        print 'hlr_sum = ',hlr_sum
        np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
                err_msg="Error in Sersic constructor with half-light radius, n = %d"%n)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sbprofile_airy():
    """Test the generation of a specific Airy profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBAiry(lam_over_D=1./0.8, obscuration=0.1, flux=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "airy_.8_.1.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
            err_msg="Airy profile disagrees with expected result") 
    # Repeat with the GSObject version of this:
    airy = galsim.Airy(lam_over_D=1./0.8, obscuration=0.1, flux=1)
    airy.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy disagrees with expected result")

    # Test photon shooting.
    do_shoot(airy,myImg,0.2,"Airy")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_airy_radii():
    """Test Airy half light radius and FWHM correctly set and match image.
    """
    import time
    t1 = time.time() 
    import math
    # Test constructor using lam_over_D: (only option for Airy)
    test_gal = galsim.Airy(lam_over_D= 1./0.8, flux=1.)
    # test half-light-radius getter
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
                                   err_msg="Error in Airy half-light radius")
    # test FWHM getter
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(.5 * test_gal.getFWHM(),0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(ratio, 0.5, decimal=4,
                                   err_msg="Error in getFWHM() for Airy.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sbprofile_box():
    """Test the generation of a specific box profile using SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBBox(xw=1, yw=1, flux=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
            err_msg="Box profile disagrees with expected result") 
    # Repeat with the GSObject version of this:
    pixel = galsim.Pixel(xw=1, yw=1, flux=1)
    pixel.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel disagrees with expected result")

    # Test photon shooting.
    do_shoot(pixel,myImg,0.2,"Pixel")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_moffat():
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
    #mySBP = galsim.SBMoffat(beta=2, fwhm=fwhm_backwards_compatible,
                            #trunc=5*fwhm_backwards_compatible, flux=1)
    mySBP = galsim.SBMoffat(beta=2, half_light_radius=1,
                            trunc=5*fwhm_backwards_compatible, flux=1)
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_2_5.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
            err_msg="Moffat profile disagrees with expected result") 
    # Repeat with the GSObject version of this:
    moffat = galsim.Moffat(beta=2, half_light_radius=1,
                           trunc=5*fwhm_backwards_compatible, flux=1)
    #moffat = galsim.Moffat(beta=2, fwhm=fwhm_backwards_compatible,
                           #trunc=5*fwhm_backwards_compatible, flux=1)
    moffat.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat disagrees with expected result")

    # Test photon shooting.
    do_shoot(moffat,myImg,0.2,"Moffat")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_moffat_properties():
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
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with half-light radius")
    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with half-light radius")
    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with half-light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, scale_radius = test_scale)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale,0)) / center
    print 'scale ratio = ',ratio
    np.testing.assert_almost_equal(ratio, pow(2,-test_beta), decimal=4,
            err_msg="Error in Moffat constructor with scale")
    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Moffat initialized with scale radius.")
    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with scale radius")

    # Test constructor using FWHM:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ',ratio
    np.testing.assert_almost_equal(ratio, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with fwhm")
    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (profile initialized with FWHM) = ',hlr_sum
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for Moffat initialized with FWHM.")
    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with scale radius")

    # Now repeat everything using a severe trunctation.  (Above had no truncation.)

    # Test constructor using half-light-radius:
    test_gal = galsim.Moffat(flux = 1., beta=test_beta, half_light_radius = test_hlr,
                             trunc=2*test_hlr)
    hlr_sum = radial_integrate(test_gal, 0., test_hlr, 1.e-4)
    print 'hlr_sum = ',hlr_sum
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with half-light radius")
    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for Moffat initialized with half-light radius")
    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for Moffat initialized with half-light radius")

    # Test constructor using scale radius:
    test_gal = galsim.Moffat(flux=1., beta=test_beta, trunc=2*test_scale,
                             scale_radius=test_scale)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_scale,0)) / center
    print 'scale ratio = ', ratio
    np.testing.assert_almost_equal(ratio, pow(2,-test_beta), decimal=4,
                                   err_msg="Error in Moffat constructor with scale")
    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (truncated profile initialized with scale_radius) = ',hlr_sum
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for truncated Moffat "+
            "initialized with scale radius.")
    # test that getFWHM() method provides correct FWHM
    got_fwhm = test_gal.getFWHM()
    test_fwhm_ratio = (test_gal.xValue(galsim.PositionD(.5 * got_fwhm, 0.)) / 
                       test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'fwhm ratio = ', test_fwhm_ratio
    np.testing.assert_almost_equal(test_fwhm_ratio, 0.5, decimal=4,
            err_msg="Error in FWHM for truncated Moffat initialized with scale radius")

    # Test constructor using FWHM:
    test_gal = galsim.Moffat(flux=1., beta=test_beta, trunc=2.*test_fwhm,
                             fwhm = test_fwhm)
    center = test_gal.xValue(galsim.PositionD(0,0))
    ratio = test_gal.xValue(galsim.PositionD(test_fwhm/2.,0)) / center
    print 'fwhm ratio = ', ratio
    np.testing.assert_almost_equal(ratio, 0.5, decimal=4,
            err_msg="Error in Moffat constructor with fwhm")
    # then test that image indeed has the matching properties when radially integrated
    got_hlr = test_gal.getHalfLightRadius()
    hlr_sum = radial_integrate(test_gal, 0., got_hlr, 1.e-4)
    print 'hlr_sum (truncated profile initialized with FWHM) = ',hlr_sum
    np.testing.assert_almost_equal(hlr_sum, 0.5, decimal=4,
            err_msg="Error in half light radius for truncated Moffat initialized with FWHM.")
    # test that getScaleRadius() method provides correct scale
    got_scale = test_gal.getScaleRadius()
    test_scale_ratio = (test_gal.xValue(galsim.PositionD(got_scale, 0.)) / 
                        test_gal.xValue(galsim.PositionD(0., 0.)))
    print 'scale ratio = ', test_scale_ratio
    np.testing.assert_almost_equal(test_scale_ratio, 2.**(-test_beta), decimal=4,
            err_msg="Error in scale radius for truncated Moffat initialized with scale radius")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sbprofile_smallshear():
    """Test the application of a small shear to a Gaussian SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    e1 = 0.02
    e2 = 0.02
    mySBP.applyDistortion(galsim.Ellipse(e1,e2))
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Small-shear Gaussian profile disagrees with expected result")
    # Repeat with the GSObject version of this:
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.applyDistortion(galsim.Ellipse(e1,e2))
    gauss.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDistortion disagrees with expected result")
    # The GSObject applyShear uses gamma version of shear, rather than distortion,
    # which is what SBProfile (confusingly) uses.  So figure out the corresponding gamma:
    g1,g2 = convertToShear(e1,e2)
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.applyShear(g1,g2)
    gauss.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(gauss,myImg,0.2,"sheared Gaussian")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_largeshear():
    """Test the application of a large shear to a Sersic SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBDeVaucouleurs(flux=1, half_light_radius=1)
    e1 = 0.0
    e2 = 0.5
    mySBP.applyDistortion(galsim.Ellipse(e1,e2))
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_largeshear.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Large-shear DeVauc profile disagrees with expected result")
    # Repeat with the GSObject version of this:
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc.applyDistortion(galsim.Ellipse(e1,e2))
    devauc.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDistortion disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    g1,g2 = convertToShear(e1,e2)
    devauc.applyShear(g1,g2)
    devauc.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    devauc2 = galsim.Convolve(devauc, galsim.Gaussian(sigma=0.3))
    do_shoot(devauc2,myImg,0.2,"sheared DeVauc")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

 
def test_sbprofile_convolve():
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
    myConv.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 4,
        err_msg="Moffat convolved with Box SBProfile disagrees with expected result")
    # Repeat with the GSObject version of this:
    psf = galsim.Moffat(beta=1.5, fwhm=fwhm_backwards_compatible, trunc=4*fwhm_backwards_compatible,
                        flux=1)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    # We'll do the real space convolution below
    conv = galsim.Convolve([psf,pixel],real_space=False)
    conv.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 4,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel,real_space=False)
    conv.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 4,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(conv,myImg,0.2,"Moffat * Pixel")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_shearconvolve():
    """Test the convolution of a sheared Gaussian and a Box SBProfile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    e1 = 0.04
    e2 = 0.0
    mySBP.applyDistortion(galsim.Ellipse(e1,e2))
    mySBP2 = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    myConv = galsim.SBConvolve([mySBP,mySBP2])
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear_convolve_box.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myConv.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Sheared Gaussian convolved with Box SBProfile disagrees with expected result")
    # Repeat with the GSObject version of this:
    psf = galsim.Gaussian(flux=1, sigma=1)
    g1,g2 = convertToShear(e1,e2)
    psf.applyShear(g1,g2)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel])
    conv.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel)
    conv.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(conv,myImg,0.2,"sheared Gaussian * Pixel")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_realspace_convolve():
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
    conv.draw(img,dx=0.2)
    printval(img, saved_img)
    arg = abs(saved_img.array-img.array).argmax()
    np.testing.assert_array_almost_equal(img.array, saved_img.array, 5,
        err_msg="Moffat convolved with Box SBProfile disagrees with expected result")
    # Repeat with the GSObject version of this:
    psf = galsim.Moffat(beta=1.5, half_light_radius=1,
                        trunc=4*fwhm_backwards_compatible, flux=1)
    #psf = galsim.Moffat(beta=1.5, fwhm=fwhm_backwards_compatible,
                        #trunc=4*fwhm_backwards_compatible, flux=1)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel],real_space=True)
    conv.draw(img,dx=0.2)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel,real_space=True)
    conv.draw(img,dx=0.2)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
    # The real-space convolution algorithm is not (trivially) independent of the order of
    # the two things being convolved.  So check the opposite order.
    conv = galsim.Convolve([pixel,psf],real_space=True)
    conv.draw(img,dx=0.2)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([pixel,psf]) disagrees with expected result")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
 

def test_sbprofile_realspace_distorted_convolve():
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
    psf.applyShear(0.11,0.17)
    psf.applyRotation(13 * galsim.degrees)
    pixel = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    pixel.applyShear(0.2,0.0)
    pixel.applyRotation(80 * galsim.degrees)
    pixel.applyShift(0.13,0.27)
    conv = galsim.SBConvolve([psf,pixel],real_space=True)

    # Note: Using an image created from Maple "exact" calculations.
    saved_img = galsim.fits.read(os.path.join(imgdir, "moffat_pixel_distorted.fits"))
    img = galsim.ImageF(saved_img.bounds)
    conv.draw(img,dx=0.2)
    printval(img, saved_img)
    np.testing.assert_array_almost_equal(img.array, saved_img.array, 5,
        err_msg="distorted Moffat convolved with distorted Box disagrees with expected result")

    # Repeat with the GSObject version of this:
    psf = galsim.Moffat(beta=1.5, half_light_radius=1,
                        trunc=4*fwhm_backwards_compatible, flux=1)
    #psf = galsim.Moffat(beta=1.5, fwhm=fwhm_backwards_compatible,
                        #trunc=4*fwhm_backwards_compatible, flux=1)
    psf.applyShear(0.11,0.17)
    psf.applyRotation(13 * galsim.degrees)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    pixel.applyShear(0.2,0.0)
    pixel.applyRotation(80 * galsim.degrees)
    pixel.applyShift(0.13,0.27)
    # NB: real-space is chosen automatically
    conv = galsim.Convolve([psf,pixel])
    conv.draw(img,dx=0.2)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([psf,pixel]) (distorted) disagrees with expected result")
    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel)
    conv.draw(img,dx=0.2)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve(psf,pixel) (distorted) disagrees with expected result")
    # The real-space convolution algorithm is not (trivially) independent of the order of
    # the two things being convolved.  So check the opposite order.
    conv = galsim.Convolve([pixel,psf])
    conv.draw(img,dx=0.2)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using Convolve([pixel,psf]) (distorted) disagrees with expected result")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
 
def test_sbprofile_realspace_shearconvolve():
    """Test the real-space convolution of a sheared Gaussian and a Box SBProfile against a 
       known result.
    """
    import time
    t1 = time.time()
    psf = galsim.SBGaussian(flux=1, sigma=1)
    e1 = 0.04
    e2 = 0.0
    psf.applyDistortion(galsim.Ellipse(e1,e2))
    pix = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.SBConvolve([psf,pix],real_space=True)
    saved_img = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear_convolve_box.fits"))
    img = galsim.ImageF(saved_img.bounds)
    conv.draw(img,dx=0.2)
    printval(img, saved_img)
    np.testing.assert_array_almost_equal(img.array, saved_img.array, 5,
        err_msg="Sheared Gaussian convolved with Box SBProfile disagrees with expected result")
    # Repeat with the GSObject version of this:
    psf = galsim.Gaussian(flux=1, sigma=1)
    g1,g2 = convertToShear(e1,e2)
    psf.applyShear(g1,g2)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel],real_space=True)
    conv.draw(img,dx=0.2)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel,real_space=True)
    conv.draw(img,dx=0.2)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
    # The real-space convolution algorithm is not (trivially) independent of the order of
    # the two things being convolved.  So check the opposite order.
    conv = galsim.Convolve([pixel,psf],real_space=True)
    conv.draw(img,dx=0.2)
    np.testing.assert_array_almost_equal(
            img.array, saved_img.array, 5,
            err_msg="Using GSObject Convolve([pixel,psf]) disagrees with expected result")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sbprofile_rotate():
    """Test the 45 degree rotation of a sheared Sersic profile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBSersic(n=2.5, flux=1, half_light_radius=1)
    mySBP.applyDistortion(galsim.Ellipse(0.2, 0.0))
    mySBP.applyRotation(45.0 * galsim.degrees)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_ellip_rotated.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="45-degree rotated elliptical Gaussian disagrees with expected result")
    # Repeat with the GSObject version of this:
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1)
    gal.applyDistortion(galsim.Ellipse(0.2,0.0));
    gal.applyRotation(45.0 * galsim.degrees)
    gal.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    gal2 = galsim.Convolve(gal, galsim.Gaussian(sigma=0.3))
    do_shoot(gal2,myImg,0.2,"rotated sheared Sersic")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_mag():
    """Test the magnification (size x 1.5) of an exponential profile against a known result.
    """
    import time
    t1 = time.time()
    re = 1.0
    r0 = re/1.67839
    mySBP = galsim.SBExponential(flux=1, scale_radius=r0)
    myEll = galsim.Ellipse(0., 0., np.log(1.5))
    mySBP.applyDistortion(myEll)
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_mag.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Magnification (x1.5) of exponential SBProfile disagrees with expected result")
    # Repeat with the GSObject version of this:
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyDistortion(myEll)
    gal.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDistortion disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(gal,myImg,0.2,"dilated Exponential")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_add():
    """Test the addition of two rescaled Gaussian profiles against a known double Gaussian result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBGaussian(flux=0.75, sigma=1)
    mySBP2 = galsim.SBGaussian(flux=0.25, sigma=3)
    myAdd = galsim.SBAdd(mySBP, mySBP2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "double_gaussian.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    myAdd.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Addition of two rescaled Gaussian profiles disagrees with expected result")
    # Repeat with the GSObject version of this:
    gauss1 = galsim.Gaussian(flux=0.75, sigma=1)
    gauss2 = galsim.Gaussian(flux=0.25, sigma=3)
    sum = galsim.Add(gauss1,gauss2)
    sum.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject Add(gauss1,gauss2) disagrees with expected result")
    # Other ways to do the sum:
    sum = gauss1 + gauss2
    sum.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject gauss1 + gauss2 disagrees with expected result")
    sum = gauss1.copy()
    sum += gauss2
    sum.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject sum = gauss1; sum += gauss2 disagrees with expected result")
    sum = galsim.Add([gauss1,gauss2])
    sum.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject Add([gauss1,gauss2]) disagrees with expected result")
    gauss1 = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = galsim.Gaussian(flux=1, sigma=3)
    sum = 0.75 * gauss1 + 0.25 * gauss2
    sum.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject 0.75 * gauss1 + 0.25 * gauss2 disagrees with expected result")
    sum = 0.75 * gauss1
    sum += 0.25 * gauss2
    sum.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject sum += 0.25 * gauss2 disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(sum,myImg,0.2,"sum of 2 Gaussians")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_shift():
    """Test the translation of a Box profile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBBox(xw=0.2, yw=0.2, flux=1)
    mySBP.applyShift(0.2, -0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_shift.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Shifted box profile disagrees with expected result")
    # Repeat with the GSObject version of this:
    pixel = galsim.Pixel(xw=0.2, yw=0.2)
    pixel.applyShift(0.2, -0.2)
    pixel.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShiift disagrees with expected result")
    print 'After applyShift test'
 
    # Test photon shooting.
    # Since photon shooting compares to a DFT convolution with a box, we can't 
    # do the shift test for a box here.  (Box * Box is very inaccurate with DFT)
    # So instead we give it a shifted Gaussian to compare with.
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.applyShift(0.4,-0.3)
    myImg = gauss.draw(dx=0.2)
    do_shoot(gauss,myImg,0.2,"shifted Gaussian")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_rescale():
    """Test the flux rescaling of a Sersic profile against a known result.
    """
    import time
    t1 = time.time()
    mySBP = galsim.SBSersic(n=3, flux=1, half_light_radius=1)
    mySBP.setFlux(2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_doubleflux.fits"))
    myImg = galsim.ImageF(savedImg.bounds)
    mySBP.draw(myImg,dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Flux-rescale sersic profile disagrees with expected result")
    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.setFlux(2)
    sersic.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject setFlux disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic *= 2
    sersic.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject *= 2 disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic2 = sersic * 2
    sersic2.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject obj * 2 disagrees with expected result")
    sersic2 = 2 * sersic
    sersic2.draw(myImg,dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject 2 * obj disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic3 = galsim.Convolve(sersic2, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic3,myImg,0.2,"scaled Sersic")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_sbprofile_sbinterpolatedimage():
    """Test that we can make SBInterpolatedImages from Images of various types, and convert back.
    """
    import time
    t1 = time.time()
    # for each type, try to make an SBInterpolatedImage, and check that when we draw an image from
    # that SBInterpolatedImage that it is the same as the original
    l3 = galsim.Lanczos(3, True, 1.0E-4)
    l32d = galsim.InterpolantXY(l3)
    for array_type in ftypes:
        image_in = galsim.ImageView[array_type](ref_array.astype(array_type))
        np.testing.assert_array_equal(
            ref_array.astype(array_type),image_in.array,
            err_msg="Array from input Image differs from reference array for type %s"%array_type)
        sbinterp = galsim.SBInterpolatedImage(image_in, l32d, dx=1.0)
        test_array = np.zeros(testshape, dtype=array_type)
        image_out = galsim.ImageView[array_type](test_array)
        sbinterp.draw(image_out, dx=1.0)
        np.testing.assert_array_equal(
            ref_array.astype(array_type),image_out.array,
            err_msg="Array from output Image differs from reference array for type %s"%array_type)
 
        # Since SBInterp is an SBProfile, rather than a GSObject, we can't just
        # use the do_shoot function we've been using for the others, since a few things
        # need to be a little different.
        flux_max = image_out.array.max()
        print 'flux_max = ',flux_max
        flux_tot = image_out.array.sum()
        print 'flux_tot = ',flux_tot

        sbinterp.scaleFlux(1. / flux_max)
        nphot = flux_tot / flux_max / photon_shoot_accuracy**2
        print 'nphot = ',nphot
        ud = galsim.UniformDeviate()
        sbinterp.drawShoot(image_out,int(nphot),ud)

        # Compare this to a convolution of the sbinter with a pixel
        pix = galsim.SBBox(xw=1.0, yw=1.0, flux=1.)
        conv = galsim.SBConvolve([sbinterp,pix])
        image_comp = image_out.copy()
        conv.draw(image_comp, dx=1.0)
        np.testing.assert_array_almost_equal(
                image_comp.array, image_out.array, photon_decimal_test,
                err_msg="Photon shooting for interpolated image disagrees with expected result")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



if __name__ == "__main__":
    test_sbprofile_gaussian()
    test_sbprofile_gaussian_properties()
    test_gaussian_radii()
    test_sbprofile_exponential()
    test_exponential_radii()
    test_sbprofile_sersic()
    test_sersic_radii()
    test_sbprofile_airy()
    test_airy_radii()
    test_sbprofile_box()
    test_sbprofile_moffat()
    test_sbprofile_moffat_properties()
    test_moffat_radii()
    test_sbprofile_smallshear()
    test_sbprofile_largeshear()
    test_sbprofile_convolve()
    test_sbprofile_shearconvolve()
    test_sbprofile_realspace_convolve()
    test_sbprofile_realspace_distorted_convolve()
    test_sbprofile_realspace_shearconvolve()
    test_sbprofile_rotate()
    test_sbprofile_mag()
    test_sbprofile_add()
    test_sbprofile_shift()
    test_sbprofile_rescale()
    test_sbprofile_sbinterpolatedimage()
