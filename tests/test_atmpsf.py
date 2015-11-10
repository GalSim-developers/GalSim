# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
import numpy as np


try:
    import galsim
except ImportError:
    import os
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


def test_atmpsf_kwargs():
    """ Test that different ways of initializing AtmsphericPSF don't crash.
    """

    # First the simple cases, where the size is set via a scalar

    lam = 500.0  # nm
    r0 = 0.2  # m
    psf0 = galsim.AtmosphericPSF(lam=lam, r0=r0)
    psf0.drawImage()

    lam_over_r0 = lam*1e-9/r0 * galsim.radians/galsim.arcsec
    psf1 = galsim.AtmosphericPSF(lam_over_r0=lam_over_r0)
    psf1.drawImage()

    fwhm = galsim.Kolmogorov(lam_over_r0).getFWHM()
    psf2 = galsim.AtmosphericPSF(fwhm=fwhm)
    psf2.drawImage()

    # Now try multiple phase screens
    weights = [0.1, 0.2, 0.3, 0.4]
    psf3 = galsim.AtmosphericPSF(lam=lam, r0=r0, weights=weights)
    psf3.drawImage()

    velocities = [1.0, 2.0, 3.0, 4.0]
    psf4 = galsim.AtmosphericPSF(lam=lam, r0=r0, velocity=velocities)
    psf4.drawImage()

    psf5 = galsim.AtmosphericPSF(lam_over_r0=lam_over_r0, weights=weights)
    psf5.drawImage()

    psf6 = galsim.AtmosphericPSF(fwhm=fwhm, weights=weights)
    psf6.drawImage()

    # Try specifying r0s directly
    r0s = [0.6, 0.6, 0.6, 0.6]
    psf7 = galsim.AtmosphericPSF(lam=lam, r0=r0s)
    psf7.drawImage()

    # Try multiple broadcasts
    directions = [d*galsim.degrees for d in (0, 10, 20, 30)]
    psf8 = galsim.AtmosphericPSF(lam=lam, r0=r0, direction=directions, velocity=velocities,
                                 weights=weights)
    psf8.drawImage()

    alpha_mags = [0.999, 0.998, 0.997, 0.996]
    psf9 = galsim.AtmosphericPSF(lam=lam, r0=r0s, direction=directions, velocity=velocities,
                                 alpha_mag=alpha_mags)
    psf9.drawImage()

    # Now try some things that *should* fail.
    try:
        # Can't specify both r0 and weights as lists.
        np.testing.assert_raises(ValueError, galsim.AtmosphericPSF,
                                 lam=lam, r0=r0s, weights=weights)
        # Can't specify both lam_over_r0 and weights as lists.
        np.testing.assert_raises(ValueError, galsim.AtmosphericPSF,
                                 lam_over_r0=[lam/r for r in r0s], weights=weights)

    except ImportError:
        print 'The assert_raises tests require nose'


if __name__ == "__main__":
    test_atmpsf_kwargs()
