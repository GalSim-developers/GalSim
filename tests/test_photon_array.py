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
    seed = 12345
    ud = galsim.UniformDeviate(seed)
    gal = galsim.Sersic(n=4, half_light_radius=1)
    photon_array = gal.SBProfile.shoot(100000, ud)

    # Try some invalid inputs
    try:
        np.testing.assert_raises(ValueError, galsim.FRatioAngles, fratio=-0.3)
        np.testing.assert_raises(ValueError, galsim.FRatioAngles, fratio=1.2, obscuration=-0.3)
        np.testing.assert_raises(ValueError, galsim.FRatioAngles, fratio=1.2, obscuration=1.0)
        np.testing.assert_raises(ValueError, galsim.FRatioAngles, fratio=1.2, obscuration=1.9)
    except ImportError:
        pass

    # Add the directions (N.B. using the same seed as for generating the photon array
    # above.  The fact that it is the same does not matter here; the testing routine
    # only needs to have a definite seed value so the consistency of the results with
    # expectations can be evaluated precisely
    fratio = 1.2
    obscuration = 0.2
    assigner = galsim.FRatioAngles(fratio, obscuration, seed)
    assigner.applyTo(photon_array)

    dxdz = photon_array.getDXDZArray()
    dydz = photon_array.getDYDZArray()

    # Compare these slopes with the expected distributions (uniform in azimuth 
    # over all azimiths and uniform in sin(inclination) over the range of
    # allowed inclinations
    phi = np.arctan2(dydz,dxdz)
    tantheta = np.sqrt(np.square(dxdz) + np.square(dydz))
    sintheta = np.sin(np.arctan(tantheta))

    phi_histo, phi_bins = np.histogram(phi, bins=100)
    sintheta_histo, sintheta_bins = np.histogram(sintheta, bins=100)

    phi_ref = float(np.sum(phi_histo))/phi_histo.size
    sintheta_ref = float(np.sum(sintheta_histo))/sintheta_histo.size

    chisqr_phi = np.sum(np.square(phi_histo - phi_ref)/phi_ref) / phi_histo.size
    chisqr_sintheta = np.sum(np.square(sintheta_histo - sintheta_ref) /
                      sintheta_ref) / sintheta_histo.size
    assert 0.9 < chisqr_phi < 1.1, "Distribution in azimuth is not nearly uniform"
    assert 0.9 < chisqr_sintheta < 1.1, "Distribution in sin(inclination) is not nearly uniform"

    # Also check that the values are within the ranges expected
    # (The test on phi really can't fail, because it is only testing the range of the
    # arctan2 function.)
    np.testing.assert_array_less(-phi, np.pi, "Azimuth angles outside possible range")
    np.testing.assert_array_less(phi, np.pi, "Azimuth angles outside possible range")

    fov_angle = np.arctan(0.5 / fratio)
    obscuration_angle = obscuration * fov_angle
    np.testing.assert_array_less(-sintheta, -np.sin(obscuration_angle), \
        "Inclination angles outside possible range")
    np.testing.assert_array_less(sintheta, np.sin(fov_angle), \
        "Inclination angles outside possible range")

if __name__ == "__main__":
    test_photon_angles()
