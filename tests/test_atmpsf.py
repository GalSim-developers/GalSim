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


def test_multi_atmpsf_reset():
    rng = galsim.BaseDeviate(1234)
    atm = galsim.Atmosphere(altitude=10.0, frozen=False, rng=rng)
    theta_x = [0.0 * galsim.arcmin, 0.1 * galsim.arcmin]
    theta_y = [0.0 * galsim.arcmin, 0.1 * galsim.arcmin]
    psfs = atm.getPSFs(theta_x=theta_x, theta_y=theta_y, exptime=0.06, diam=4.0)

    atm.reset()
    psf0 = atm.getPSF(theta_x=theta_x[0], theta_y=theta_y[0], exptime=0.06, diam=4.0)
    atm.reset()
    psf1 = atm.getPSF(theta_x=theta_x[1], theta_y=theta_y[1], exptime=0.06, diam=4.0)

    np.testing.assert_array_equal(
        psfs[0].img, psf0.img,
        "AtmosphericPSF generated singly differs from AtmosphericPSF generated multiply")
    np.testing.assert_array_equal(
        psfs[1].img, psf1.img,
        "AtmosphericPSF generated singly differs from AtmosphericPSF generated multiply")

if __name__ == "__main__":
    test_multi_atmpsf_reset()
