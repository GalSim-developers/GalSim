import os
import sys
import pyfits
import numpy as np
import math

"""Unit tests for the PSF correction and shear estimation routines.

There are two types of tests: tests that use Gaussian profiles, for which the ideal results are
known; and tests that use real galaxies in SDSS for which results were tabulated using the same code
before it was integrated into GalSim (so we can make sure we are not breaking anything as we modify
the code).
"""

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# define a range of input parameters for the Gaussians that we are testing
gaussian_sig_values = [1.0, 2.0, 3.0]
shear_values = [0.01, 0.03, 0.05]
pixel_scale = 0.2
decimal = 2 # decimal place at which to require equality in sizes
decimal_shape = 3 # decimal place at which to require equality in shapes

def test_moments_basic():
    """Test that we can properly recover adaptive moments for Gaussians."""
    for sig in gaussian_sig_values:
        for g1 in shear_values:
            distortion_1 = np.tanh(2.0*math.atanh(g1))
            for g2 in shear_values:
                distortion_2 = np.tanh(2.0*math.atanh(g2))
                gal = galsim.Gaussian(flux = 1.0, sigma = sig)
                gal.applyShear(g1, g2)
                gal_image = gal.draw(dx = pixel_scale)
                result = gal_image.FindAdaptiveMom()
                # make sure we find the right Gaussian sigma
                np.testing.assert_almost_equal(np.fabs(result.moments_sigma-sig/pixel_scale), 0.0,
                                               err_msg = "- incorrect dsigma", decimal = decimal)
                # make sure we find the right e
                np.testing.assert_almost_equal(result.observed_shape.getE1(),
                                               distortion_1, err_msg = "- incorrect e1",
                                               decimal = decimal_shape)
                np.testing.assert_almost_equal(result.observed_shape.getE2(),
                                               distortion_2, err_msg = "- incorrect e2",
                                               decimal = decimal_shape)

def test_shearest_basic():
    """Test that we can recover shears for Gaussian galaxies and PSFs."""

def test_shearest_precomputed():
    """Test that we can recover shears the same as before the code was put into GalSim."""

if __name__ == "__main__":
    test_moments_basic()
    test_shearest_basic()
    test_shearest_precomputed()
