import os

import numpy as np

import galsim
import galsim.atmosphere

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images. 

def test_doublegaussian_vs_sbadd():
    """Test that profiles from galsim.atmosphere.DoubleGaussian equal those from SBGaussian/SBAdd.
    """
    for flux1 in np.linspace(0.2, 3, 3):
        for sigma1 in np.linspace(0.2, 3, 3):
            for flux2 in np.linspace(0.2, 3, 3):
                for sigma2 in np.linspace(0.2, 3, 3):
                    dbl1 = galsim.atmosphere.DoubleGaussian(flux1, sigma1, flux2, sigma2)
                    g1 = galsim.SBGaussian(flux1, sigma1)
                    g2 = galsim.SBGaussian(flux2, sigma2)
                    dbl2 = galsim.SBAdd(g1, g2)
                    np.testing.assert_almost_equal(dbl1.draw().array, dbl2.draw().array)

def test_doublegaussian_vs_refimg():
    """Test a specific double Gaussian from galsim.atmosphere.DoubleGaussian against a saved result.
    """
    dblg = galsim.atmosphere.DoubleGaussian(0.75, 1, 0.25, 3)
    myImg = dblg.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "double_gaussian.fits"))
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Two Gaussian reference image disagrees with DoubleGaussian class")   
