import numpy as np
import os
import sys

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# set up any necessary info for tests

def test_real_galaxy_readin():
    """Read in real galaxy as RealGalaxy, make sure results agree with direct readin"""
    # read in RealGalaxy stuff using proper base classes

    # read in images directly from file

    # make sure original and PSF image are as expected


def test_real_galaxy_ideal():
    """Test accuracy of various calculations with fake Gaussian RealGalaxy vs. ideal expectations"""
    # read in faked Gaussian RealGalaxy from file

    # convolve with a range of Gaussians, with and without shear;
    # compare with expected (fixed image size)

def test_real_galaxy_saved():
    """Test accuracy of various calculations with real RealGalaxy vs. stored SHERA result"""
    # read in real RealGalaxy from file

    # read in expected result for some shear

    # read try to simulate the same galaxy with GalSim

    # require results to agree at fairly high significance

    # check that if we rotate before convolving with round PSF, results agree with convolving with
    # round PSF then applying the same rotation

if __name__ == "__main__":
    test_real_galaxy_readin()
    test_real_galaxy_ideal()
    test_real_galaxy_saved()
