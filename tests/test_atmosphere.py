# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
import os
import sys

import numpy as np

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images. 

def funcname():
    import inspect
    return inspect.stack()[1][3]

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
    np.testing.assert_almost_equal(apsf.maxK(), 5.8341564391716183, 9,
                                   err_msg="Atmospheric PSF .maxk() does not return known value.")
    np.testing.assert_almost_equal(apsf.stepK(), 1.0275679547331542, 9,
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
        print 'apsf.getFlux = ',apsf.getFlux()
        np.testing.assert_almost_equal(apsf.getFlux(), 1., 6, 
                                       err_msg="Flux of atmospheric PSF (ImageViewD) is not 1.")
        # .draw() throws a warning if it doesn't get a float. This includes np.float64. Convert to
        # have the test pass.
        dx = float(lor / 10.)
        img = galsim.ImageF(256,256)
        img_array = apsf.draw(image=img, dx=dx).array
        np.testing.assert_almost_equal(img_array.sum(), 1., 3,
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
        # Need use_true_center=False, since we want the maximum to actually be drawn in one 
        # of the pixels, rather than between the central 4 pixels.
        psf_array = apsf.draw(dx=dx, use_true_center=False).array
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
    test_AtmosphericPSF_flux()
    test_AtmosphericPSF_properties()
    test_AtmosphericPSF_fwhm()
