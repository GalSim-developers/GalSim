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

    # Note: Use something quite small in terms of npixels so the B/F effect kicks in without
    # requiring a ridiculous number of photons
    obj = galsim.Gaussian(flux=3539, sigma=0.3)

    # We'll draw the same object using SiliconSensor, Sensor, and the default (sensor=None)
    im1 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=silicon
    im2 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=simple
    im3 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=None

    rng1 = galsim.BaseDeviate(5678)
    rng2 = galsim.BaseDeviate(5678)
    rng3 = galsim.BaseDeviate(5678)

    # TODO: This file needs a better location (and probably name).
    # If it's generic, we can put it in shared, which will install into an accessibly location
    # when GalSim is installed.  If it's too specific to be broadly useful, then we should switch
    # to setting specific parameters via constructor arguments, rather than use a file at all.
    # (Should probably enable this feature anyway...)
    silicon = galsim.SiliconSensor('../devel/poisson/BF_256_9x9_0_Vertices', rng=rng1)
    simple = galsim.Sensor()

    # Start with photon shooting, since that's more straightforward.
    obj.drawImage(im1, method='phot', poisson_flux=False, sensor=silicon, rng=rng1)
    obj.drawImage(im2, method='phot', poisson_flux=False, sensor=simple, rng=rng2)
    obj.drawImage(im3, method='phot', poisson_flux=False, rng=rng3)

    # First, im2 and im3 should be exactly equal.
    np.testing.assert_array_equal(im2.array, im3.array)

    # im1 should be similar, but not equal
    np.testing.assert_almost_equal(im1.array/obj.flux, im2.array/obj.flux, decimal=2)

    # Now use a different seed for 3 to see how much of the variation is just from randomness.
    rng3.seed(234241)
    obj.drawImage(im3, method='phot', poisson_flux=False, rng=rng3)

    r1 = im1.calculateMomentRadius(flux=obj.flux)
    r2 = im2.calculateMomentRadius(flux=obj.flux)
    r3 = im3.calculateMomentRadius(flux=obj.flux)
    print('Flux = %.0f:  sum        peak         radius'%obj.flux)
    print('im1:         %.1f     %.2f       %f'%(im1.array.sum(),im1.array.max(), r1))
    print('im2:         %.1f     %.2f       %f'%(im2.array.sum(),im2.array.max(), r2))
    print('im3:         %.1f     %.2f       %f'%(im3.array.sum(),im3.array.max(), r3))

    # Fluxes should all equal obj.flux
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux)

    # Sizes are all about equal since flux is not large enough for B/F to be significant
    # Variance of Irr for Gaussian with Poisson noise is
    # Var(Irr) = Sum(I r^4) = 4Irr [using Gaussian kurtosis = 8sigma^2, Irr = 2sigma^2]
    # r = sqrt(Irr/flux), so sigma(r) = 1/2 r sqrt(Var(Irr))/Irr = 1/sqrt(flux)
    # Use 2sigma for below checks.
    rtol = 2. / np.sqrt(obj.flux) * im1.scale
    np.testing.assert_allclose(r1, r2, atol=rtol)
    np.testing.assert_allclose(r2, r3, atol=rtol)

    # Repeat with 20X more photons where the brighter-fatter effect should kick in more.
    obj *= 20
    rtol = 2. / np.sqrt(obj.flux) * im1.scale
    obj.drawImage(im1, method='phot', poisson_flux=False, sensor=silicon, rng=rng1)
    obj.drawImage(im2, method='phot', poisson_flux=False, sensor=simple, rng=rng2)
    obj.drawImage(im3, method='phot', poisson_flux=False, rng=rng3)

    r1 = im1.calculateMomentRadius(flux=obj.flux)
    r2 = im2.calculateMomentRadius(flux=obj.flux)
    r3 = im3.calculateMomentRadius(flux=obj.flux)
    print('Flux = %.0f:  sum        peak          radius'%obj.flux)
    print('im1:         %.1f     %.2f       %f'%(im1.array.sum(),im1.array.max(), r1))
    print('im2:         %.1f     %.2f       %f'%(im2.array.sum(),im2.array.max(), r2))
    print('im3:         %.1f     %.2f       %f'%(im3.array.sum(),im3.array.max(), r3))

    # Fluxes should still be fine.
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux)

    # Sizes for 2,3 should be about equal, but 1 should be larger.
    print('check |r2-r3| = %f <? %f'%(np.abs(r2-r3), rtol))
    np.testing.assert_allclose(r2, r3, atol=rtol)
    print('check |r1-r3| = %f >? %f'%(np.abs(r1-r3), rtol))
    assert r1-r3 > rtol

    # Check that it is really responding to flux, not number of photons.
    # Using fewer shot photons will mean each one encapsulates several electrons at once.
    obj.drawImage(im1, method='phot', n_photons=3000, poisson_flux=False, sensor=silicon, rng=rng1)
    obj.drawImage(im2, method='phot', n_photons=3000, poisson_flux=False, sensor=simple, rng=rng2)
    obj.drawImage(im3, method='phot', n_photons=3000, poisson_flux=False, rng=rng3)

    r1 = im1.calculateMomentRadius(flux=obj.flux)
    r2 = im2.calculateMomentRadius(flux=obj.flux)
    r3 = im3.calculateMomentRadius(flux=obj.flux)
    print('Flux = %.0f:  sum        peak          radius'%obj.flux)
    print('im1:         %.1f     %.2f       %f'%(im1.array.sum(),im1.array.max(), r1))
    print('im2:         %.1f     %.2f       %f'%(im2.array.sum(),im2.array.max(), r2))
    print('im3:         %.1f     %.2f       %f'%(im3.array.sum(),im3.array.max(), r3))

    np.testing.assert_almost_equal(im1.array.sum(), obj.flux)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux)

    print('check |r2-r3| = %f <? %f'%(np.abs(r2-r3), rtol))
    np.testing.assert_allclose(r2, r3, atol=rtol)  # It's actually noisier now, but this passes.
    print('check |r1-r3| = %f >? %f'%(np.abs(r1-r3), rtol))
    assert r1-r3 > rtol

    # TODO: The above tests are not really sufficient.
    # We need some more sophisticated tests that this is doing the right thing, not just
    # showing some evidence for B/F.




if __name__ == "__main__":
    test_silicon()
