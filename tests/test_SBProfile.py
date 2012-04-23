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

# define a series of tests

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


def test_sbprofile_gaussian_properties():
    """Test some basic properties of the SBGaussian profile.
    """
    psf = galsim.SBGaussian()
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

def test_sbprofile_exponential():
    """Test the generation of a specific exp profile using SBProfile against a known result. 
    """
    re = 1.0
    r0 = re/1.67839
    mySBP = galsim.SBExponential(flux=1., r0=r0)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_1.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
                                         err_msg="Exponential profile disagrees with expected"
                                         +" result") 
    # Repeat with the GSObject version of this:
    expon = galsim.Exponential(flux=1, r0=r0)
    myImg = expon.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Exponential disagrees with expected result")

def test_sbprofile_sersic():
    """Test the generation of a specific Sersic profile using SBProfile against a known result.
    """
    mySBP = galsim.SBSersic(n=3, flux=1, re=1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_3_1.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
                                         err_msg="Sersic profile disagrees with expected result")   
    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, re=1)
    myImg = sersic.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Sersic disagrees with expected result")

def test_sbprofile_airy():
    """Test the generation of a specific Airy profile using SBProfile against a known result.
    """
    mySBP = galsim.SBAiry(D=0.8, obs=0.1, flux=1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "airy_.8_.1.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
                                         err_msg="Airy profile disagrees with expected result") 
    # Repeat with the GSObject version of this:
    airy = galsim.Airy(D=0.8, obs=0.1, flux=1)
    myImg = airy.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Airy disagrees with expected result")

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

def test_sbprofile_moffat():
    """Test the generation of a specific Moffat profile using SBProfile against a known result.
    """
    mySBP = galsim.SBMoffat(beta=2, truncationFWHM=5, flux=1, re=1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_2_5.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
                                         err_msg="Moffat profile disagrees with expected result") 
    # Repeat with the GSObject version of this:
    moffat = galsim.Moffat(beta=2, truncationFWHM=5, flux=1, re=1)
    myImg = moffat.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Moffat disagrees with expected result")

def test_sbprofile_moffat_properties():
    """Test some basic properties of the SBMoffat profile.
    """
    psf = galsim.SBMoffat(2.0)
    # Check that we are centered on (0, 0)
    cen = galsim._galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxK(), 34.226259129031952)
    np.testing.assert_almost_equal(psf.stepK(), 0.08604618622618046)
    np.testing.assert_equal(psf.kValue(cen), 1+0j)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        psfFlux = galsim.SBMoffat(2.0, flux=inFlux)
        outFlux = psfFlux.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.28141470275895519)
    
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
    
def test_sbprofile_largeshear():
    """Test the application of a large shear to a Sersic SBProfile against a known result.
    """
    mySBP = galsim.SBSersic(n=4, flux=1, re=1)
    e1 = 0.0
    e2 = 0.5
    mySBP_shear = mySBP.shear(e1,e2)
    myImg = mySBP_shear.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_largeshear.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Large-shear Sersic profile disagrees with expected result")  
    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=4, flux=1, re=1)
    sersic.applyDistortion(galsim.Ellipse(e1,e2))
    myImg = sersic.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDistortion disagrees with expected result")
    sersic = galsim.Sersic(n=4, flux=1, re=1)
    g1,g2 = convertToShear(e1,e2)
    sersic.applyShear(g1,g2)
    myImg = sersic.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyShear disagrees with expected result")
    
def test_sbprofile_convolve():
    """Test the convolution of a Moffat and a Box SBProfile against a known result.
    """
    mySBP = galsim.SBMoffat(beta=1.5, truncationFWHM=4, flux=1, re=1)
    mySBP2 = galsim.SBBox(xw=0.2, yw=0.2, flux=1.)
    myConv = galsim.SBConvolve(mySBP)
    myConv.add(mySBP2)
    myImg = myConv.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "moffat_convolve_box.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Moffat convolved with Box SBProfile disagrees with expected result")  
    # Repeat with the GSObject version of this:
    psf = galsim.Moffat(beta=1.5, truncationFWHM=4, flux=1, re=1)
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

def test_sbprofile_rotate():
    """Test the 45 degree rotation of a sheared Sersic profile against a known result.
    """
    mySBP = galsim.SBSersic(n=2.5, flux=1, re=1)
    mySBP_shear = mySBP.shear(0.2, 0.0)
    # TODO: I think this is rotating by 45 radians, so need to think about if this is the 
    #       syntax we want for rotate.  Clearly not what the creator of this test expected.
    mySBP_shear_rotate = mySBP_shear.rotate(45.0)
    myImg = mySBP_shear_rotate.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_ellip_rotated.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="45-degree rotated elliptical Gaussian disagrees with expected result")  
    # Repeat with the GSObject version of this:
    gal = galsim.Sersic(n=2.5, flux=1, re=1)
    gal.applyDistortion(galsim.Ellipse(0.2,0.0));
    gal.applyRotation(45.0)
    myImg = gal.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyRotation disagrees with expected result")

def test_sbprofile_mag():
    """Test the magnification (size x 1.5) of an exponential profile against a known result.
    """
    re = 1.0
    r0 = re/1.67839
    mySBP = galsim.SBExponential(flux=1, r0=r0)
    myEll = galsim.Ellipse(0., 0., np.log(1.5))
    mySBP_mag = mySBP.distort(myEll)
    myImg = mySBP_mag.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "exp_mag.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Magnification (x1.5) of exponential SBProfile disagrees with expected result")   
    # Repeat with the GSObject version of this:
    gal = galsim.Exponential(flux=1, r0=r0)
    gal.applyDistortion(myEll)
    myImg = gal.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject applyDistortion disagrees with expected result")

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

def test_sbprofile_rescale():
    """Test the flux rescaling of a Sersic profile against a known result.
    """
    mySBP = galsim.SBSersic(n=3, flux=1, re=1)
    mySBP.setFlux(2)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "sersic_doubleflux.fits"))
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Flux-rescale sersic profile disagrees with expected result")   
    # Repeat with the GSObject version of this:
    sersic = galsim.Sersic(n=3, flux=1, re=1)
    sersic.setFlux(2)
    myImg = sersic.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject setFlux disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, re=1)
    sersic *= 2
    myImg = sersic.draw(dx=0.2)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject *= 2 disagrees with expected result")
    sersic = galsim.Sersic(n=3, flux=1, re=1)
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

def test_sbprofile_sbinterpolatedimage():
    """Test that we can make SBInterpolatedImages from Images of various types, and convert back.
    """
    # for each type, try to make an SBInterpolatedImage, and check that when we draw an image from
    # that SBInterpolatedImage that it is the same as the original
    l3 = galsim.Lanczos(3, True, 1.0E-4)
    l32d = galsim.InterpolantXY(l3)
    for array_type in ftypes:
        image_in = galsim.Image[array_type](ref_array.astype(array_type))
        np.testing.assert_array_equal(
            ref_array.astype(array_type),image_in.array,
            err_msg="Array from input Image differs from reference array for type %s"%array_type)
        sbinterp = galsim.SBInterpolatedImage(image_in, l32d, dx=1.0)
        test_array = np.zeros(testshape, dtype=array_type)
        image_out = galsim.Image[array_type](test_array)
        sbinterp.draw(image_out, dx=1.0)
        np.testing.assert_array_equal(
            ref_array.astype(array_type),image_out.array,
            err_msg="Array from output Image differs from reference array for type %s"%array_type)

if __name__ == "__main__":
    test_sbprofile_gaussian()
    test_sbprofile_gaussian_properties()
    test_sbprofile_exponential()
    test_sbprofile_sersic()
    test_sbprofile_airy()
    test_sbprofile_box()
    test_sbprofile_moffat()
    test_sbprofile_moffat_properties()
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
