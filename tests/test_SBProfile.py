import galsim
import numpy as np
import os

imgdir = os.path.join(".","SBProfile_comparison_images") # Directory containing the reference images.

# define a series of tests

def printval(image1, image2):
    print "New, saved array sizes: ",np.shape(image1.array),np.shape(image2.array)
    print "Sum of values: ",np.sum(image1.array),np.sum(image2.array)
    print "Minimum image value: ",np.min(image1.array),np.min(image2.array)
    print "Maximum image value: ",np.max(image1.array),np.max(image2.array)
    print "Peak location: ",image1.array.argmax(),image2.array.argmax()

def test_sbprofile_gaussian():
    """Test the generation of a specific Gaussian profile using SBProfile against a known result.
    """
    mySBP = galsim.SBGaussian(1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"gauss_1.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
                                         err_msg="Gaussian profile disagrees with expected result")   

def test_sbprofile_exponential():
    """Test the generation of a specific exponential profile using SBProfile against a known result. 
    """
    mySBP = galsim.SBExponential(1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"exp_1.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
                                         err_msg="Exponential profile disagrees with expected"
                                         +" result") 

def test_sbprofile_sersic():
    """Test the generation of a specific Sersic profile using SBProfile against a known result.
    """
    mySBP = galsim.SBSersic(3,1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"sersic_3_1.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
                                         err_msg="Sersic profile disagrees with expected result")   

def test_sbprofile_airy():
    """Test the generation of a specific Airy profile using SBProfile against a known result.
    """
    mySBP = galsim.SBAiry(0.8,0.1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"airy_.8_.1.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
                                         err_msg="Airy profile disagrees with expected result") 

def test_sbprofile_box():
    """Test the generation of a specific box profile using SBProfile against a known result.
    """
    mySBP = galsim.SBBox(1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"box_1.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
                                         err_msg="Box profile disagrees with expected result") 

def test_sbprofile_moffat():
    """Test the generation of a specific Moffat profile using SBProfile against a known result.
    """
    mySBP = galsim.SBMoffat(2,5,1)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"moffat_2_5.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
                                         err_msg="Moffat profile disagrees with expected result") 

def test_sbprofile_smallshear():
    """Test the application of a small shear to a Gaussian SBProfile against a known result.
    """
    mySBP = galsim.SBGaussian(1)
    mySBP.shear(0.02,0.02)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"gauss_smallshear.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
        err_msg="Small-shear Gaussian profile disagrees with expected result")  
    
def test_sbprofile_largeshear():
    """Test the application of a large shear to a Sersic SBProfile against a known result.
    """
    mySBP = galsim.SBSersic(4,1)
    mySBP.shear(0.0,0.5)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"sersic_largeshear.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
        err_msg="Large-shear Sersic profile disagrees with expected result")  
    
def test_sbprofile_convolve():
    """Test the convolution of a Moffat and a Box SBProfile against a known result.
    """
    mySBP = galsim.SBMoffat(1.5,4,1)
    mySBP2 = galsim.SBBox(0.2)
    myConv = galsim.SBConvolve(mySBP)
    myConv.add(mySBP2)
    myImg = myConv.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"moffat_convolve_box.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
        err_msg="Moffat convolved with Box SBProfile disagrees with expected result")  

def test_sbprofile_shearconvolve():
    """Test the convolution of a sheared Gaussian and a Box SBProfile against a known result.
    """
    mySBP = galsim.SBGaussian(1)
    mySBP.shear(0.04,0.0)
    mySBP2 = galsim.SBBox(0.2)
    myConv = galsim.SBConvolve(mySBP)
    myConv.add(mySBP2)
    myImg = myConv.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"gauss_smallshear_convolve_box.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
        err_msg="Sheared Gaussian convolved with Box SBProfile disagrees with expected result")  

def test_sbprofile_rotate():
    """Test the 45 degree rotation of a sheared Sersic profile against a known result.
    """
    mySBP = galsim.SBSersic(2.5,1)
    mySBP.shear(0.2,0.0)
    mySBP.rotate(45.0)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"sersic_ellip_rotated.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
        err_msg="45-degree rotated elliptical Gaussian disagrees with expected result")  

def test_sbprofile_mag():
    """Test the magnification (size x 1.5) of an exponential profile against a known result.
    """
    mySBP = galsim.SBExponential(1)
    myEll = galsim.Ellipse(0.,0.,np.log(1.5))
    mySBP.distort(myEll)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"exp_mag.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
        err_msg="Magnification (x1.5) of exponential SBProfile disagrees with expected result")   

def test_sbprofile_add():
    """Test the addition of two rescaled Gaussian profiles against a known double Gaussian result.
    """
    mySBP = galsim.SBGaussian(1)
    mySBP.setFlux(0.75)
    mySBP2 = galsim.SBGaussian(3)
    mySBP2.setFlux(0.25)
    myAdd = galsim.SBAdd(mySBP)
    myAdd.add(mySBP2)
    myImg = myAdd.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"double_gaussian.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
        err_msg="Addition of two rescaled Gaussian profiles disagrees with expected result")   

def test_sbprofile_shift():
    """Test the translation of a Box profile against a known result.
    """
    mySBP = galsim.SBBox(0.2)
    mySBP.shift(0.2,-0.2)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"box_shift.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
        err_msg="Shifted box profile disagrees with expected result")   

def test_sbprofile_rescale():
    """Test the flux rescaling of a Sersic profile against a known result.
    """
    mySBP = galsim.SBSersic(3,1)
    mySBP.setFlux(2)
    myImg = mySBP.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir,"sersic_doubleflux.fits"))
    printval(myImg,savedImg)
    np.testing.assert_array_almost_equal(myImg.array,savedImg.array,5,
        err_msg="Flux-rescale sersic profile disagrees with expected result")   




