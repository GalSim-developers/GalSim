import os
import sys

import numpy as np

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

import galsim.atmosphere

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images. 

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_doublegaussian_vs_sbadd():
    """Test that profiles from galsim.atmosphere.DoubleGaussian equal those from SBGaussian/SBAdd.
    """
    import time
    t1 = time.time()
    for flux1 in np.linspace(0.2, 3, 3):
        for sigma1 in np.linspace(0.2, 3, 3):
            for flux2 in np.linspace(0.2, 3, 3):
                for sigma2 in np.linspace(0.2, 3, 3):
                    dbl1 = galsim.atmosphere.DoubleGaussian(flux1, flux2, sigma1=sigma1, sigma2=sigma2)
                    g1 = galsim.SBGaussian(flux1, sigma=sigma1)
                    g2 = galsim.SBGaussian(flux2, sigma=sigma2)
                    dbl2 = galsim.SBAdd(g1, g2)
                    np.testing.assert_almost_equal(dbl1.draw().array, dbl2.draw().array)
    for flux1 in np.linspace(0.2, 3, 3):
        for fwhm1 in np.linspace(0.2, 3, 3):
            for flux2 in np.linspace(0.2, 3, 3):
                for fwhm2 in np.linspace(0.2, 3, 3):
                    dbl1 = galsim.atmosphere.DoubleGaussian(flux1, flux2, fwhm1=fwhm1, fwhm2=fwhm2)
                    g1 = galsim.SBGaussian(flux1, fwhm=fwhm1)
                    g2 = galsim.SBGaussian(flux2, fwhm=fwhm2)
                    dbl2 = galsim.SBAdd(g1, g2)
                    np.testing.assert_almost_equal(dbl1.draw().array, dbl2.draw().array)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_doublegaussian_vs_refimg():
    """Test a specific double Gaussian from galsim.atmosphere.DoubleGaussian against a saved result.
    """
    import time
    t1 = time.time()
    dblg = galsim.atmosphere.DoubleGaussian(0.75, 0.25, sigma1=1., sigma2=3.)
    myImg = dblg.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "double_gaussian.fits"))
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Two Gaussian reference image disagrees with DoubleGaussian class")   
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_doublegaussian_vs_sbadd()
    test_doublegaussian_vs_refimg()
