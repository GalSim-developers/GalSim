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
def test_silicon():
    """Test the basic construction and use of the SiloconSensor class.
    """

    obj = galsim.Exponential(flux=353, scale_radius=1.3)

    # We'll draw the same object using SiliconSensor, Sensor, and the default (sensor=None)
    im1 = galsim.ImageD(64, 64, scale=0.2)  # Will use sensor=silicon
    im2 = galsim.ImageD(64, 64, scale=0.2)  # Will use sensor=simple
    im3 = galsim.ImageD(64, 64, scale=0.2)  # Will use sensor=None

    rng1 = galsim.BaseDeviate(5678)
    rng2 = galsim.BaseDeviate(5678)
    rng3 = galsim.BaseDeviate(5678)

    # TODO: This file needs a better location.  If it's generic, we can put it in shared
    # which will install into an accessibly location when GalSim is installed.  If it's
    # too specific to be broadly useful, then we should switch to setting specific parameters
    # via constructor arguments, rather than use a file at all.  (Should probably enable this
    # anyway...)
    silicon = galsim.SiliconSensor('../devel/poisson/BF_256_9x9_0_Vertices', rng=rng1)
    simple = galsim.Sensor()

    # Start with photon shooting, since that's more straightforward.
    nphot = 10000
    obj.drawImage(im1, method='phot', n_photons=nphot, sensor=silicon, rng=rng1)
    obj.drawImage(im2, method='phot', n_photons=nphot, sensor=simple, rng=rng2)
    obj.drawImage(im3, method='phot', n_photons=nphot, rng=rng3)

    print('im1: sum = ',im1.array.sum(),' flux = ',obj.flux)
    print('im2: sum = ',im2.array.sum(),' flux = ',obj.flux)
    print('im3: sum = ',im3.array.sum(),' flux = ',obj.flux)

    # First, im2 and im3 should be exactly equal.
    np.testing.assert_array_equal(im2.array, im3.array)

    # im1 should be similar, but not equal
    np.testing.assert_almost_equal(im1.array/obj.flux, im2.array/obj.flux, decimal=3)

    # TODO: Obviously, the above test is sort of the opposite of what we care about.
    # We need some tests that this is doing the right thing.  Not just nearly equivalent
    # to the default Sensor.


if __name__ == "__main__":
    test_silicon()
