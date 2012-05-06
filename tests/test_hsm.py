import os
import sys
import pyfits
import numpy as np

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

def test_moments_basic():
    """Test that we can properly recover adaptive moments for Gaussians."""

def test_shearest_basic():
    """Test that we can recover shears for Gaussian galaxies and PSFs."""

def test_shearest_precomputed():
    """Test that we can recover shears the same as before the code was put into GalSim."""

if __name__ == "__main__":
    test_moments_basic()
    test_shearest_basic()
    test_shearest_precomputed()
