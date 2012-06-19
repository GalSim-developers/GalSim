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
                    dbl1 = galsim.atmosphere.DoubleGaussian(sigma1=sigma1, sigma2=sigma2,
                                                            flux1=flux1, flux2=flux2)
                    g1 = galsim.SBGaussian(sigma=sigma1, flux=flux1)
                    g2 = galsim.SBGaussian(sigma=sigma2, flux=flux2)
                    dbl2 = galsim.SBAdd(g1, g2)
                    np.testing.assert_almost_equal(dbl1.draw().array, dbl2.draw().array)
    for flux1 in np.linspace(0.2, 3, 3):
        for fwhm1 in np.linspace(0.2, 3, 3):
            for flux2 in np.linspace(0.2, 3, 3):
                for fwhm2 in np.linspace(0.2, 3, 3):
                    dbl1 = galsim.atmosphere.DoubleGaussian(fwhm1=fwhm1, fwhm2=fwhm2,
                                                            flux1=flux1, flux2=flux2) 
                    g1 = galsim.SBGaussian(fwhm=fwhm1, flux=flux1)
                    g2 = galsim.SBGaussian(fwhm=fwhm2, flux=flux2)
                    dbl2 = galsim.SBAdd(g1, g2)
                    np.testing.assert_almost_equal(dbl1.draw().array, dbl2.draw().array)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_doublegaussian_vs_refimg():
    """Test a specific double Gaussian from galsim.atmosphere.DoubleGaussian against a saved result.
    """
    import time
    t1 = time.time()
    dblg = galsim.atmosphere.DoubleGaussian(sigma1=1., sigma2=3., flux1=0.75, flux2=0.25)
    myImg = dblg.draw(dx=0.2)
    savedImg = galsim.fits.read(os.path.join(imgdir, "double_gaussian.fits"))
    np.testing.assert_array_almost_equal(myImg.array, savedImg.array, 5,
        err_msg="Two Gaussian reference image disagrees with DoubleGaussian class")   
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_AtmosphericPSF_properties():
    """Test some basic properties of a known Atmospheric PSF.
    """
    import time
    t1 = time.time()
    apsf = galsim.AtmosphericPSF(lam_over_r0=1.5)
    # Check that we are centered on (0, 0)
    cen = galsim._galsim.PositionD(0, 0)
    np.testing.assert_array_almost_equal(
            [apsf.centroid().x, apsf.centroid().y], [cen.x, cen.y], 10,
            err_msg="Atmospheric PSF not centered on (0, 0)")
    # Check Fourier properties
    np.testing.assert_almost_equal(apsf.maxK(), 24.051209331580893, 9,
                                   err_msg="Atmospheric PSF .maxk() does not return known value.")
    np.testing.assert_almost_equal(apsf.stepK(), 0.15331483362499429, 9,
                                   err_msg="Atmospheric PSF .stepk() does not return known value.")
    np.testing.assert_almost_equal(apsf.kValue(cen), 1+0j, 4,
                                   err_msg="Atmospheric PSF k value at (0, 0) is not 1+0j.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_AtmosphericPSF_flux():
    """Test that the flux of the atmospheric PSF is normalized to unity.
    """
    import time
    t1 = time.time()
    lors = np.linspace(0.5, 2., 5) # Different lambda_over_r0 values
    for lor in lors:
        apsf = galsim.AtmosphericPSF(lam_over_r0=lor)
        np.testing.assert_almost_equal(apsf.getFlux(), 1., 6, 
                                       err_msg="Flux of atmospheric PSF (ImageViewD) is not 1.")
        # .draw() throws a warning if it doesn't get a float. This includes np.float64. Convert to
        # have the test pass.
        dx = float(lor / 10.)
        img_array = apsf.draw(dx=dx).array
        np.testing.assert_almost_equal(img_array.sum() * dx**2, 1., 3,
                                       err_msg="Flux of atmospheric PSF (image array) is not 1.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
        
def test_AtmosphericPSF_fwhm():
    """Test that the FWHM of the atmospheric PSF corresponds to the one expected from the
    lambda / r0 input."""
    import time
    t1 = time.time()
    lors = np.linspace(0.5, 2., 5) # Different lambda_over_r0 values
    for lor in lors:
        apsf = galsim.AtmosphericPSF(lam_over_r0=lor)
        # .draw() throws a warning if it doesn't get a float. This includes np.float64. Convert to
        # have the test pass.
        dx_scale = 10
        dx = float(lor / dx_scale)
        psf_array = apsf.draw(dx=dx).array
        nx, ny = psf_array.shape
        profile = psf_array[nx / 2, ny / 2:]
        # Now get the last array index where the profile value exceeds half the peak value as a 
        # rough estimator of the HWHM.
        hwhm_index = np.where(profile > profile.max() / 2.)[0][-1]
        np.testing.assert_equal(hwhm_index, dx_scale / 2, 
                                err_msg="Kolmogorov PSF does not have the expected FWHM.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
        
if __name__ == "__main__":
    test_doublegaussian_vs_sbadd()
    test_doublegaussian_vs_refimg()
    test_AtmosphericPSF_flux()
    test_AtmosphericPSF_properties()
    test_AtmosphericPSF_fwhm()
