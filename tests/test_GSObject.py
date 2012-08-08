import os
import sys
import numpy as np
# import galsim even if path not yet added to PYTHONPATH env variable (e.g. by full install)
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

imgdir = os.path.join(".", "SBProfile_comparison_images")

# Test values taken from test_SBProfile.py... and modified slightly.
# for radius tests - specify half-light-radius, FHWM, sigma to be compared with high-res image (with
# pixel scale chosen iteratively until convergence is achieved, beginning with test_dx)
test_hlr = 1.9
test_fwhm = 1.9
test_sigma = 1.9
test_scale = 1.9
test_sersic_n = [1.4, 2.6]

# for flux normalization tests
test_flux = 1.9




