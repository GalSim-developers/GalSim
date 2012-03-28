import numpy as np

import galsim
import galsim.atmosphere

def test_gaussian():
    """Test basic properties of the standard SBGaussian returned by atmosphere.gaussian.
    """
    psf = galsim.atmosphere.gaussian()
    # Check that we are centered on (0, 0)
    cen = galsim._galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_equal(psf.maxK(), 4.0)
    np.testing.assert_almost_equal(psf.stepK(), 0.78539816339744828)
    # Check flux
    for inFlux in np.logspace(-2, 2, 10):
        psfFlux = galsim.atmosphere.moffat(2.0, flux=inFlux)
        outFlux = psfFlux.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.15915494309189535)
    np.testing.assert_equal(psf.kValue(cen), 1+0j)

def test_moffat():
    """Test basic properties of the SBMoffat returend by atmosphere.moffat.
    """
    psf = galsim.atmosphere.moffat(2.0)
    # Check that we are centered on (0, 0)
    cen = galsim._galsim.PositionD(0, 0)
    np.testing.assert_equal(psf.centroid(), cen)
    # Check Fourier properties
    np.testing.assert_almost_equal(psf.maxK(), 34.226259129031952)
    np.testing.assert_almost_equal(psf.stepK(), 0.08604618622618046)
    # Check flux
    for inFlux in np.logspace(-2, 2, 10):
        psfFlux = galsim.atmosphere.moffat(2.0, flux=inFlux)
        outFlux = psfFlux.getFlux()
        np.testing.assert_almost_equal(outFlux, inFlux)
    np.testing.assert_almost_equal(psf.xValue(cen), 0.28141470275895519)
    np.testing.assert_equal(psf.kValue(cen), 1+0j)
    
