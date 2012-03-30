import galsim
import numpy as np
import os

imgdir = os.path.join(".","SBProfile_comparison_images") # Directory containing the reference images.

# define a series of tests
def test_sbprofile_gaussian():
    """Test the generation of a specific Gaussian profile using SBProfile against a known result.
    """
    myGauss = galsim.SBGaussian(1)
    myImg = myGauss.draw(dx=0.2)
    testImg = galsim.fits.read(os.path.join(imgdir,"gauss_1.fits"))
    np.testing.assert_array_almost_equal(myImg.array,testImg.array,5,
                                         err_msg="Gaussian profile disagrees with expected result") 

def test_sbprofile_exponential():
    """Test the generation of a specific exponential profile using SBProfile against a known result.
    """
    myExp = galsim.SBExponential(1)
    myImg = myExp.draw(dx=0.2)
    testImg = galsim.fits.read(os.path.join(imgdir,"exp_1.fits"))
    np.testing.assert_array_almost_equal(myImg.array,testImg.array,5,
                                         err_msg="Exponential profile disagrees with expected result") 

def test_sbprofile_sersic():
    """Test the generation of a specific Sersic profile using SBProfile against a known result.
    """
    mySersic = galsim.SBSersic(3,1)
    myImg = mySersic.draw(dx=0.2)
    testImg = galsim.fits.read(os.path.join(imgdir,"sersic_3_1.fits"))
    np.testing.assert_array_almost_equal(myImg.array,testImg.array,5,
                                         err_msg="Sersic profile disagrees with expected result") 

def test_sbprofile_airy():
    """Test the generation of a specific Airy profile using SBProfile against a known result.
    """
    myAiry = galsim.SBAiry(0.8,0.1)
    myImg = myAiry.draw(dx=0.2)
    testImg = galsim.fits.read(os.path.join(imgdir,"airy_.8_.1.fits"))
    np.testing.assert_array_almost_equal(myImg.array,testImg.array,5,
                                         err_msg="Airy profile disagrees with expected result") 

def test_sbprofile_box():
    """Test the generation of a specific box profile using SBProfile against a known result.
    """
    myBox = galsim.SBBox(1)
    myImg = myBox.draw(dx=0.2)
    testImg = galsim.fits.read(os.path.join(imgdir,"box_1.fits"))
    np.testing.assert_array_almost_equal(myImg.array,testImg.array,5,
                                         err_msg="Box profile disagrees with expected result") 

def test_sbprofile_moffat():
    """Test the generation of a specific Moffat profile using SBProfile against a known result.
    """
    myMoffat = galsim.SBMoffat(2,5,1)
    myImg = myMoffat.draw(dx=0.2)
    testImg = galsim.fits.read(os.path.join(imgdir,"moffat_2_5.fits"))
    np.testing.assert_array_almost_equal(myImg.array,testImg.array,5,
                                         err_msg="Moffat profile disagrees with expected result") 




