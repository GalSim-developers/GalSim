# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from __future__ import print_function
import numpy as np
import os
import sys

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

@timer
def test_photon_angles():
    """Test the photon_array function
    """
    # Make a photon array
    ud = galsim.UniformDeviate()
    gal = galsim.Sersic(n=4, half_light_radius=1)
    photon_array = gal.SBProfile.shoot(100000, ud)

    # Add the directions (not currently working)
    fratio = 1.2
    obscuration = 0.2
    seed = 12345
    photon_array.assignPhotonAngles(fratio, obscuration, seed)

    dxdz = photon_array.getDXDZArray()
    dydz = photon_array.getDYDZArray()

    # Compare these slopes with the expected distributions (uniform in azimuth 
    # over all azimiths and uniform in sin(inclination) over the range of
    # allowed inclinations
    phi = np.arctan2(dydz,dxdz)
    sintheta = np.sqrt(1. - np.square(dxdz) - np.square(dydz))

    phi_histo, phi_bins = np.histogram(phi, bins=100)
    sintheta_histo, sintheta_bins = np.histogram(sintheta, bins=100)

    phi_ref = phi_histo*0 + float(np.sum(phi_histo))/phi_histo.size
    sintheta_ref = sintheta_histo*0 + float(np.sum(sintheta_histo)
                   )/sintheta_histo.size

    chisqr_phi = np.sum(np.square(phi_histo - phi_ref)/phi_ref) / phi_histo.size
    chisqr_sintheta = np.sum(np.square(sintheta_histo - sintheta_ref) /
                      sintheta_ref) / sintheta_histo.size

    
    # In the assert_almost_equal tests below, the expected values are defined
    # for the particular set of directions generated for the seed value specified
    # above
    np.testing.assert_almost_equal(chisqr_phi, 1.05562, 5, \
        "Distribution in azimuth is not uniform")
    np.testing.assert_almost_equal(chisqr_sintheta, 0.95920, 5, \
        "Distribution in sin(inclination) is not uniform")

    # Also check that the values are within the ranges expected
    np.testing.assert_almost_equal(np.min(phi) + np.pi, 0.000258, 5, \
        "Azimuth angle range extends to too-small angles")
    np.testing.assert_almost_equal(np.max(phi) - np.pi, -6.474875e-05, 5, \
        "Inclination angle range extends to too-large angles")

    fov_angle = np.arctan(0.5 / fratio)
    obscuration_angle = obscuration * fov_angle

    np.testing.assert_almost_equal(np.min(sintheta) - np.sin(obscuration_angle), \
        0., 5, "Inclination angle range extends to too-small angles")
    np.testing.assert_almost_equal(np.max(sintheta) - np.sin(fov_angle), \
        0., 5, "Inclination angle range extends to too-large angles")

if __name__ == "__main__":
    test_photon_angles()
