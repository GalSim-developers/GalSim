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
test_hlr = 1.0
test_fwhm = 1.0
test_sigma = 1.0
test_scale = 1.0
test_dx = 0.2
test_sersic_n = [1.5, 2.5]
target_precision = 0.004 # convergence criterion governing choice of pixel scale
init_ratio = 1000.0 # a junk value to start with
convergence_value = init_ratio # a junk value to start with, should be >> target_precision

# define some functions to carry out computations that are carried out by several of the tests

def getRGrid(image1):
    # function to get the value of radius from the image center at the position of each pixel
    xgrid, ygrid = np.meshgrid(np.arange(np.shape(image1.array)[0]) + image1.getXMin(),
                               np.arange(np.shape(image1.array)[1]) + image1.getYMin())
    xcent = np.mean(xgrid * image1.array) / np.mean(image1.array)
    ycent = np.mean(ygrid * image1.array) / np.mean(image1.array)
    rgrid = np.sqrt((xgrid-xcent)**2 + (ygrid-ycent)**2)
    return rgrid

def getIntegratedFlux(image1, radius):
    # integrate to compute the flux in an image within some chosen radius [units: pixels], in a
    # clunky but transparent way -- will only be reasonably accurate for high-resolution images.
    return np.sum(image1.array[np.where(getRGrid(image1) < radius)])

def getIntensityAtRadius(image1, radius):
    # get the intensity in an image at some chosen radius [units: pixels] from the center, in a
    # clunky yet transparent way -- will only be reasonably accurate for high-resolution images, not
    # right at the center.
    rgrid = getRGrid(image1)
    rvec = np.arange(1., np.max(rgrid), 1.)
    Ivec = 0 * rvec
    rgrid_nearest = (np.round(rgrid)).astype(np.integer)
    ind_below = np.max(np.where(rvec < radius))
    ind_above = np.min(np.where(rvec >= radius))
    newvec = image1.array[np.where(rgrid_nearest == ind_below)]
    Ibelow = np.sum(newvec)/len(newvec)
    newvec = image1.array[np.where(rgrid_nearest == ind_above)]
    Iabove = np.sum(newvec)/len(newvec)
    delta = (radius - rvec[ind_below])/(rvec[ind_above]-rvec[ind_below])
    return (delta*Iabove + (1.0-delta)*Ibelow)

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
    #img.write("junk1.fits")
    prof.drawShoot(img2,nphot)
    #img2.write("junk2.fits")
    np.testing.assert_array_almost_equal(
            img2.array, img.array, photon_decimal_test,
            err_msg="Photon shooting for %s disagrees with expected result"%name)


# define a series of tests

def test_sbprofile_gaussian():
    """Test the generation of a specific Gaussian profile using SBProfile against a known result.
    """
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_1.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
            err_msg="Gaussian profile disagrees with expected result")
    # Repeat with the GSObject version of this:
    gauss = galsim.Gaussian(flux=1, sigma=1)
    myImg = gauss.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Gaussian disagrees with expected result")

    # Test photon shooting.
    do_shoot(gauss,myImg,0.2,"Gaussian")


def test_sbprofile_gaussian_properties():
    """Test some basic properties of the SBGaussian profile.
    """
    psf = galsim.SBGaussian(flux=1, sigma=1)
    # Check that we are centered on (0, 0)
    cen = galsim._galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_equal(psf.maxK(), 4.0)
    np.testing.assert_almost_equal(psf.stepK(), 0.78539816339744828)
    np.testing.assert_equal(psf.kValue(cen), 1+0j)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        psfFlux = galsim.SBGaussian(flux=inFlux, sigma=2.)
        outFlux = psfFlux.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.15915494309189535)

def test_gaussian_radii():
    """Test initialization of Gaussian with different types of radius specification.
    """
    # first test half-light-radius
    my_test_dx = test_dx
    my_prev_ratio = init_ratio
    my_convergence_value = convergence_value
    while (my_convergence_value > target_precision):
        test_gal = galsim.Gaussian(flux = 1., half_light_radius = test_hlr)
        test_gal_image = test_gal.draw(dx = my_test_dx)
        my_ratio = getIntegratedFlux(test_gal_image, test_hlr/my_test_dx)/np.sum(test_gal_image.array)
        my_convergence_value = np.fabs((my_ratio - my_prev_ratio)/my_prev_ratio)
        my_prev_ratio = my_ratio
        my_test_dx /= 2.0
    np.testing.assert_almost_equal(my_ratio, 0.5, decimal = 2,
                                   err_msg="Error in Gaussian constructor with half-light radius")
    # then test sigma
    my_test_dx = test_dx
    my_prev_ratio = init_ratio
    my_convergence_value = convergence_value
    while (my_convergence_value > target_precision):
        test_gal = galsim.Gaussian(flux = 1., sigma = test_sigma)
        test_gal_image = test_gal.draw(dx = my_test_dx)
        my_ratio = getIntensityAtRadius(test_gal_image, test_sigma/my_test_dx)/np.max(test_gal_image.array)
        my_convergence_value = np.fabs((my_ratio - my_prev_ratio)/my_prev_ratio)
        my_prev_ratio = my_ratio
        my_test_dx /= 2.0
    np.testing.assert_almost_equal(my_ratio, np.exp(-0.5), decimal = 2,
                                   err_msg="Error in Gaussian constructor with sigma")
    # then test FWHM
    my_test_dx = test_dx
    my_prev_ratio = init_ratio
    my_convergence_value = convergence_value
    while (my_convergence_value > target_precision):
        test_gal = galsim.Gaussian(flux = 1., fwhm = test_fwhm)
        test_gal_image = test_gal.draw(dx = my_test_dx)
        my_ratio = getIntensityAtRadius(test_gal_image, 0.5*test_fwhm/my_test_dx)/np.max(test_gal_image.array)
        my_convergence_value = np.fabs((my_ratio - my_prev_ratio)/my_prev_ratio)
        my_prev_ratio = my_ratio
        my_test_dx /= 2.0
    np.testing.assert_almost_equal(my_ratio, 0.5, decimal = 2,
                                   err_msg="Error in Gaussian constructor with FWHM")

def test_sbprofile_exponential():
    """Test the generation of a specific exp profile using SBProfile against a known result. 
    """
    re = 1.0
    r0 = re/1.67839
    mySBP = galsim.SBExponential(flux=1., scale_radius=r0)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_1.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
                                         err_msg="Exponential profile disagrees with expected"
                                         +" result") 
    # Repeat with the GSObject version of this:
    expon = galsim.Exponential(flux=1., scale_radius=r0)
    myImg = expon.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential disagrees with expected result")

    # Test photon shooting.
    do_shoot(expon,myImg,0.2,"Exponential")


def test_exponential_radii():
    """Test initialization of Exponential with different types of radius specification.
    """
    # first test half-light-radius
    my_test_dx = test_dx
    my_prev_ratio = init_ratio
    my_convergence_value = convergence_value
    while (my_convergence_value > target_precision):
        test_gal = galsim.Exponential(flux = 1., half_light_radius = test_hlr)
        test_gal_image = test_gal.draw(dx = my_test_dx)
        my_ratio = getIntegratedFlux(test_gal_image, test_hlr/my_test_dx)/np.sum(test_gal_image.array)
        my_convergence_value = np.fabs((my_ratio - my_prev_ratio)/my_prev_ratio)
        my_prev_ratio = my_ratio
        my_test_dx /= 2.0
    np.testing.assert_almost_equal(my_ratio, 0.5, decimal = 2,
                                   err_msg="Error in Exponential constructor with half-light radius")
    # then test scale
    my_test_dx = test_dx
    my_prev_ratio = init_ratio
    my_convergence_value = convergence_value
    while (my_convergence_value > target_precision):
        test_gal = galsim.Exponential(flux = 1., scale_radius = test_scale)
        test_gal_image = test_gal.draw(dx = my_test_dx)
        my_ratio = getIntensityAtRadius(test_gal_image, test_scale/my_test_dx)/np.max(test_gal_image.array)
        my_convergence_value = np.fabs((my_ratio - my_prev_ratio)/my_prev_ratio)
        my_prev_ratio = my_ratio
        my_test_dx /= 2.0
    np.testing.assert_almost_equal(my_ratio, np.exp(-1.0), decimal = 2,
                                   err_msg="Error in Exponential constructor with scale radius")

def test_sbprofile_sersic():
    """Test the generation of a specific Sersic profile using SBProfile against a known result.
    """
    mySBP = galsim.SBSersic(n=3, flux=1, half_light_radius=1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
                                         err_msg="Sersic profile disagrees with expected result")   
    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    myImg = sersic.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic disagrees with expected result")

    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic2 = galsim.Convolve(sersic, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic2,myImg,0.2,"Sersic")


def test_sersic_radii():
    """Test initialization of Sersic with different types of radius specification.
    """
    # test half-light-radius
    for sersicn in test_sersic_n:
        my_test_dx = test_dx
        my_prev_ratio = init_ratio
        my_convergence_value = convergence_value
        while (my_convergence_value > target_precision):
            test_gal = galsim.Sersic(sersicn, flux = 1., half_light_radius = test_hlr)
            test_gal_image = test_gal.draw(dx = my_test_dx)
            my_ratio = getIntegratedFlux(test_gal_image, test_hlr/my_test_dx)/np.sum(test_gal_image.array)
            my_convergence_value = np.fabs((my_ratio - my_prev_ratio)/my_prev_ratio)
            my_prev_ratio = my_ratio
            my_test_dx /= 2.0
        np.testing.assert_almost_equal(my_ratio, 0.5, decimal = 2,
                                       err_msg="Error in Sersic constructor with half-light radius")

def test_sbprofile_airy():
    """Test the generation of a specific Airy profile using SBProfile against a known result.
    """
    mySBP = galsim.SBAiry(D=0.8, obscuration=0.1, flux=1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "airy_.8_.1.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
                                         err_msg="Airy profile disagrees with expected result") 
    # Repeat with the GSObject version of this:
    airy = galsim.Airy(D=0.8, obscuration=0.1, flux=1)
    myImg = airy.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy disagrees with expected result")

    # Test photon shooting.
    do_shoot(airy,myImg,0.2,"Airy")


def test_sbprofile_box():
    """Test the generation of a specific box profile using SBProfile against a known result.
    """
    # MJ: Could use yw=0, which means use yw=xw, but this is not as intuitive as just
    #     making xw and yw both = 1 (or both = pixel_scale normally).
    mySBP = galsim.SBBox(xw=1, yw=1, flux=1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_1.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
                                         err_msg="Box profile disagrees with expected result") 
    # Repeat with the GSObject version of this:
    pixel = galsim.Pixel(xw=1, yw=1, flux=1)
    myImg = pixel.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel disagrees with expected result")

    # Test photon shooting.
    do_shoot(pixel,myImg,0.2,"Pixel")


def test_sbprofile_moffat():
    """Test the generation of a specific Moffat profile using SBProfile against a known result.
    """
    mySBP = galsim.SBMoffat(beta=2, truncationFWHM=5, flux=1, half_light_radius=1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_2_5.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
                                         err_msg="Moffat profile disagrees with expected result") 
    # Repeat with the GSObject version of this:
    moffat = galsim.Moffat(beta=2, truncationFWHM=5, flux=1, half_light_radius=1)
    myImg = moffat.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat disagrees with expected result")

    # Test photon shooting.
    do_shoot(moffat,myImg,0.2,"Moffat")


def test_sbprofile_moffat_properties():
    """Test some basic properties of the SBMoffat profile.
    """
    psf = galsim.SBMoffat(beta=2.0, truncationFWHM=2, flux=1, half_light_radius=1)
    # Check that we are centered on (0, 0)
    cen = galsim._galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxK(), 34.226260866076707)
    np.testing.assert_almost_equal(psf.stepK(), 0.08604618622618046)
    np.testing.assert_equal(psf.kValue(cen), 1+0j)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        psfFlux = galsim.SBMoffat(2.0, truncationFWHM=2, flux=inFlux, half_light_radius=1)
        outFlux = psfFlux.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.28141470275895519)
    
def test_moffat_radii():
    """Test initialization of Moffat with different types of radius specification.
    """
    test_beta = 2.
    # first test half-light-radius
    my_test_dx = test_dx
    my_prev_ratio = init_ratio
    my_convergence_value = convergence_value
    while (my_convergence_value > target_precision):
        test_gal = galsim.Moffat(beta=test_beta, truncationFWHM=5, flux = 1., half_light_radius = test_hlr)
        test_gal_image = test_gal.draw(dx = my_test_dx)
        my_ratio = getIntegratedFlux(test_gal_image, test_hlr/my_test_dx)/np.sum(test_gal_image.array)
        my_convergence_value = np.fabs((my_ratio - my_prev_ratio)/my_prev_ratio)
        my_prev_ratio = my_ratio
        my_test_dx /= 2.0
    np.testing.assert_almost_equal(my_ratio, 0.5, decimal = 2,
                                   err_msg="Error in Moffat constructor with half-light radius")
    # then test scale -- later!  this method takes too long
    # then test FWHM -- later!  this method takes too long

def test_sbprofile_smallshear():
    """Test the application of a small shear to a Gaussian SBProfile against a known result.
    """
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    e1 = 0.02
    e2 = 0.02
    mySBP_shear = mySBP.shear(e1,e2)
    myImg = mySBP_shear.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Small-shear Gaussian profile disagrees with expected result")  
    # Repeat with the GSObject version of this:
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.applyDistortion(galsim.Ellipse(e1,e2))
    myImg = gauss.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDistortion disagrees with expected result")
    # The GSObject applyShear uses gamma version of shear, rather than distortion,
    # which is what SBProfile (confusingly) uses.  So figure out the corresponding gamma:
    g1,g2 = convertToShear(e1,e2)
    gauss = galsim.Gaussian(flux=1, sigma=1)
    gauss.applyShear(g1,g2)
    myImg = gauss.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear disagrees with expected result")
    
    # Test photon shooting.
    do_shoot(gauss,myImg,0.2,"sheared Gaussian")


def test_sbprofile_largeshear():
    """Test the application of a large shear to a Sersic SBProfile against a known result.
    """
    mySBP = galsim.SBDeVaucouleurs(flux=1, half_light_radius=1)
    e1 = 0.0
    e2 = 0.5
    mySBP_shear = mySBP.shear(e1,e2)
    myImg = mySBP_shear.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_largeshear.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Large-shear DeVauc profile disagrees with expected result")  
    # Repeat with the GSObject version of this:
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    devauc.applyDistortion(galsim.Ellipse(e1,e2))
    myImg = devauc.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDistortion disagrees with expected result")
    devauc = galsim.DeVaucouleurs(flux=1, half_light_radius=1)
    g1,g2 = convertToShear(e1,e2)
    devauc.applyShear(g1,g2)
    myImg = devauc.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear disagrees with expected result")
    
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    devauc2 = galsim.Convolve(devauc, galsim.Gaussian(sigma=0.3))
    do_shoot(devauc2,myImg,0.2,"sheared DeVauc")

 
def test_sbprofile_convolve():
    """Test the convolution of a Moffat and a Box SBProfile against a known result.
    """
    mySBP = galsim.SBMoffat(beta=1.5, truncationFWHM=4, flux=1, half_light_radius=1)
    mySBP2 = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    myConv = galsim.SBConvolve(mySBP)
    myConv.add(mySBP2)
    myImg = myConv.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_convolve_box.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Moffat convolved with Box SBProfile disagrees with expected result")  
    # Repeat with the GSObject version of this:
    psf = galsim.Moffat(beta=1.5, truncationFWHM=4, flux=1, half_light_radius=1)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel])
    myImg = conv.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel)
    myImg = conv.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
    conv = galsim.Convolve(psf)
    conv.add(pixel)
    myImg = conv.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve(psf) with add(pixel) disagrees with expected result")
    conv = galsim.Convolve()
    conv.add(psf)
    conv.add(pixel)
    myImg = conv.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve() with add both disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(conv,myImg,0.2,"Moffat * Pixel")


def test_sbprofile_shearconvolve():
    """Test the convolution of a sheared Gaussian and a Box SBProfile against a known result.
    """
    mySBP = galsim.SBGaussian(flux=1, sigma=1)
    e1 = 0.04
    e2 = 0.0
    mySBP_shear = mySBP.shear(e1,e2)
    mySBP2 = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    myConv = galsim.SBConvolve(mySBP_shear)
    myConv.add(mySBP2)
    myImg = myConv.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "gauss_smallshear_convolve_box.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Sheared Gaussian convolved with Box SBProfile disagrees with expected result")  
    # Repeat with the GSObject version of this:
    psf = galsim.Gaussian(flux=1, sigma=1)
    g1,g2 = convertToShear(e1,e2)
    psf.applyShear(g1,g2)
    pixel = galsim.Pixel(xw=0.2, yw=0.2, flux=1.)
    conv = galsim.Convolve([psf,pixel])
    myImg = conv.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result")
    # Other ways to do the convolution:
    conv = galsim.Convolve(psf,pixel)
    myImg = conv.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve(psf,pixel) disagrees with expected result")
    conv = galsim.Convolve(psf)
    conv.add(pixel)
    myImg = conv.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve(psf) with add(pixel) disagrees with expected result")
    conv = galsim.Convolve()
    conv.add(pixel)
    conv.add(psf)
    myImg = conv.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Convolve() with add both disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(conv,myImg,0.2,"sheared Gaussian * Pixel")


def test_sbprofile_rotate():
    """Test the 45 degree rotation of a sheared Sersic profile against a known result.
    """
    mySBP = galsim.SBSersic(n=2.5, flux=1, half_light_radius=1)
    mySBP_shear = mySBP.shear(0.2, 0.0)
    mySBP_shear_rotate = mySBP_shear.rotate(45.0 * galsim.degrees)
    myImg = mySBP_shear_rotate.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_ellip_rotated.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="45-degree rotated elliptical Gaussian disagrees with expected result")  
    # Repeat with the GSObject version of this:
    gal = galsim.Sersic(n=2.5, flux=1, half_light_radius=1)
    gal.applyDistortion(galsim.Ellipse(0.2,0.0));
    gal.applyRotation(45.0 * galsim.degrees)
    myImg = gal.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    gal2 = galsim.Convolve(gal, galsim.Gaussian(sigma=0.3))
    do_shoot(gal2,myImg,0.2,"rotated sheared Sersic")


def test_sbprofile_mag():
    """Test the magnification (size x 1.5) of an exponential profile against a known result.
    """
    re = 1.0
    r0 = re/1.67839
    mySBP = galsim.SBExponential(flux=1, scale_radius=r0)
    myEll = galsim.Ellipse(0., 0., np.log(1.5))
    mySBP_mag = mySBP.distort(myEll)
    myImg = mySBP_mag.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_mag.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Magnification (x1.5) of exponential SBProfile disagrees with expected result")   
    # Repeat with the GSObject version of this:
    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal.applyDistortion(myEll)
    myImg = gal.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDistortion disagrees with expected result")
 
    # Test photon shooting.
    do_shoot(gal,myImg,0.2,"dilated Exponential")


def test_sbprofile_add():
    """Test the addition of two rescaled Gaussian profiles against a known double Gaussian result.
    """
    mySBP = galsim.SBGaussian(flux=0.75, sigma=1)
    mySBP2 = galsim.SBGaussian(flux=0.25, sigma=3)
    myAdd = galsim.SBAdd(mySBP, mySBP2)
    myImg = myAdd.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "double_gaussian.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Addition of two rescaled Gaussian profiles disagrees with expected result")   
    # Repeat with the GSObject version of this:
    gauss1 = galsim.Gaussian(flux=0.75, sigma=1)
    gauss2 = galsim.Gaussian(flux=0.25, sigma=3)
    sum = galsim.Add(gauss1,gauss2)
    myImg = sum.draw(dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject Add(gauss1,gauss2) disagrees with expected result")   
    # Other ways to do the sum:
    sum = gauss1 + gauss2
    myImg = sum.draw(dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject gauss1 + gauss2 disagrees with expected result")   
    sum = gauss1.copy()
    sum += gauss2
    myImg = sum.draw(dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject sum = gauss1; sum += gauss2 disagrees with expected result")   
    sum = galsim.Add([gauss1,gauss2])
    myImg = sum.draw(dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject Add([gauss1,gauss2]) disagrees with expected result")   
    sum = galsim.Add(gauss1)
    sum.add(gauss2)
    myImg = sum.draw(dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject Add(gauss1) with add(gauss2) disagrees with expected result")   
    sum = galsim.Add()
    sum.add(gauss1)
    sum.add(gauss2)
    myImg = sum.draw(dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject Add() with add both disagrees with expected result")   
    gauss1 = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = galsim.Gaussian(flux=1, sigma=3)
    sum = 0.75 * gauss1 + 0.25 * gauss2
    myImg = sum.draw(dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject 0.75 * gauss1 + 0.25 * gauss2 disagrees with expected result")   
    sum = 0.75 * gauss1
    sum += 0.25 * gauss2
    myImg = sum.draw(dx=0.2)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Using GSObject sum += 0.25 * gauss2 disagrees with expected result")   
 
    # Test photon shooting.
    do_shoot(sum,myImg,0.2,"sum of 2 Gaussians")


def test_sbprofile_shift():
    """Test the translation of a Box profile against a known result.
    """
    mySBP = galsim.SBBox(xw=0.2, yw=0.2, flux=1)
    mySBP_shift = mySBP.shift(0.2, -0.2)
    myImg = mySBP_shift.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_shift.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Shifted box profile disagrees with expected result")   
    # Repeat with the GSObject version of this:
    pixel = galsim.Pixel(xw=0.2, yw=0.2)
    pixel.applyShift(0.2, -0.2)
    myImg = pixel.draw(dx=0.2)
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


def test_sbprofile_rescale():
    """Test the flux rescaling of a Sersic profile against a known result.
    """
    mySBP = galsim.SBSersic(n=3, flux=1, half_light_radius=1)
    mySBP.setFlux(2)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_doubleflux.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Flux-rescale sersic profile disagrees with expected result")   
    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic.setFlux(2)
    myImg = sersic.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject setFlux disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic *= 2
    myImg = sersic.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject *= 2 disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, half_light_radius=1)
    sersic2 = sersic * 2
    myImg = sersic2.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject obj * 2 disagrees with expected result")
    sersic2 = 2 * sersic
    myImg = sersic2.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject 2 * obj disagrees with expected result")
 
    # Test photon shooting.
    # Convolve with a small gaussian to smooth out the central peak.
    sersic3 = galsim.Convolve(sersic2, galsim.Gaussian(sigma=0.3))
    do_shoot(sersic3,myImg,0.2,"scaled Sersic")


def test_sbprofile_sbinterpolatedimage():
    """Test that we can make SBInterpolatedImages from Images of various types, and convert back.
    """
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

        sbinterp.setFlux(sbinterp.getFlux() / flux_max)
        nphot = flux_tot / flux_max / photon_shoot_accuracy**2
        print 'nphot = ',nphot
        ud = galsim.UniformDeviate()
        sbinterp.drawShoot(image_out,int(nphot),ud)

        # Compare this to a convolution of the sbinter with a pixel
        pix = galsim.SBBox(xw=1.0, yw=1.0, flux=1.)
        conv = galsim.SBConvolve(sbinterp)
        conv.add(pix)
        image_comp = image_out.copy()
        conv.draw(image_comp, dx=1.0)
        np.testing.assert_array_almost_equal(
                image_comp.array, image_out.array, photon_decimal_test,
                err_msg="Photon shooting for interpolated image disagrees with expected result")



if __name__ == "__main__":
    test_sbprofile_gaussian()
    test_sbprofile_gaussian_properties()
    test_gaussian_radii()
    test_sbprofile_exponential()
    test_exponential_radii()
    test_sbprofile_sersic()
    test_sersic_radii()
    test_sbprofile_airy()
    test_sbprofile_box()
    test_sbprofile_moffat()
    test_sbprofile_moffat_properties()
    test_moffat_radii()
    test_sbprofile_smallshear()
    test_sbprofile_largeshear()
    test_sbprofile_convolve()
    test_sbprofile_shearconvolve()
    test_sbprofile_rotate()
    test_sbprofile_mag()
    test_sbprofile_add()
    test_sbprofile_shift()
    test_sbprofile_rescale()
    test_sbprofile_sbinterpolatedimage()
