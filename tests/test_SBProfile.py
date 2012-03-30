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




