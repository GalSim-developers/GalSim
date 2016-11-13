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

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

bppath = os.path.join(galsim.meta_data.share_dir, "bandpasses")
sedpath = os.path.join(galsim.meta_data.share_dir, "SEDs")


@timer
def test_photon_array():
    """Test the basic methods of PhotonArray class
    """
    nphotons = 1000

    # First create from scratch
    photon_array = galsim.PhotonArray(nphotons)
    assert len(photon_array.x) == nphotons
    assert len(photon_array.y) == nphotons
    assert len(photon_array.flux) == nphotons
    assert not photon_array.hasAllocatedWavelengths()
    assert not photon_array.hasAllocatedAngles()

    # Initial values should all be 0
    np.testing.assert_array_equal(photon_array.x, 0.)
    np.testing.assert_array_equal(photon_array.y, 0.)
    np.testing.assert_array_equal(photon_array.flux, 0.)

    # Check picklability
    do_pickle(photon_array)

    # Check assignment via numpy [:]
    photon_array.x[:] = 5
    photon_array.y[:] = 17
    photon_array.flux[:] = 23
    np.testing.assert_array_equal(photon_array.x, 5.)
    np.testing.assert_array_equal(photon_array.y, 17.)
    np.testing.assert_array_equal(photon_array.flux, 23.)

    # Check assignment directly to the attributes
    photon_array.x = 25
    photon_array.y = 37
    photon_array.flux = 53
    np.testing.assert_array_equal(photon_array.x, 25.)
    np.testing.assert_array_equal(photon_array.y, 37.)
    np.testing.assert_array_equal(photon_array.flux, 53.)

    # Now create from shooting a profile
    obj = galsim.Exponential(flux=1.7, scale_radius=2.3)
    rng = galsim.UniformDeviate(1234)
    photon_array = obj.SBProfile.shoot(nphotons, rng)
    orig_x = photon_array.x.copy()
    orig_y = photon_array.y.copy()
    orig_flux = photon_array.flux.copy()
    assert len(photon_array.x) == nphotons
    assert len(photon_array.y) == nphotons
    assert len(photon_array.flux) == nphotons
    assert not photon_array.hasAllocatedWavelengths()
    assert not photon_array.hasAllocatedAngles()

    # Check arithmetic ops
    photon_array.x *= 5
    photon_array.y += 17
    photon_array.flux /= 23
    np.testing.assert_almost_equal(photon_array.x, orig_x * 5.)
    np.testing.assert_almost_equal(photon_array.y, orig_y + 17.)
    np.testing.assert_almost_equal(photon_array.flux, orig_flux / 23.)

    # Check picklability again with non-zero values
    do_pickle(photon_array)

    # Now assign to the optional arrays
    photon_array.dxdz = 0.17
    assert photon_array.hasAllocatedAngles()
    assert not photon_array.hasAllocatedWavelengths()
    np.testing.assert_array_equal(photon_array.dxdz, 0.17)
    np.testing.assert_array_equal(photon_array.dydz, 0.)

    photon_array.dydz = 0.59
    np.testing.assert_array_equal(photon_array.dxdz, 0.17)
    np.testing.assert_array_equal(photon_array.dydz, 0.59)

    # Start over to check that assigning to wavelength leaves dxdz, dydz alone.
    photon_array = obj.SBProfile.shoot(nphotons, rng)
    photon_array.wavelength = 500.
    assert photon_array.hasAllocatedWavelengths()
    assert not photon_array.hasAllocatedAngles()
    np.testing.assert_array_equal(photon_array.wavelength, 500)

    photon_array.dxdz = 0.23
    photon_array.dydz = 0.88
    photon_array.wavelength = 912.
    assert photon_array.hasAllocatedWavelengths()
    assert photon_array.hasAllocatedAngles()
    np.testing.assert_array_equal(photon_array.dxdz, 0.23)
    np.testing.assert_array_equal(photon_array.dydz, 0.88)
    np.testing.assert_array_equal(photon_array.wavelength, 912)

    # Check picklability again with non-zero values for everything
    do_pickle(photon_array)

@timer
def test_wavelength_sampler():
    nphotons = 1000
    obj = galsim.Exponential(flux=1.7, scale_radius=2.3)
    rng = galsim.UniformDeviate(1234)

    photon_array = obj.SBProfile.shoot(nphotons, rng)

    sed = galsim.SED(os.path.join(sedpath, 'CWW_E_ext.sed'), 'A', 'flambda').thin()
    bandpass = galsim.Bandpass(os.path.join(bppath, 'LSST_r.dat'), 'nm').thin()

    sampler = galsim.WavelengthSampler(sed, bandpass, rng)
    sampler.applyTo(photon_array)

    # Note: the underlying functionality of the sampleWavelengths function is tested
    # in test_sed.py.  So here we are really just testing that the wrapper class is
    # properly writing to the photon_array.wavelengths array.

    assert photon_array.hasAllocatedWavelengths()
    assert not photon_array.hasAllocatedAngles()

    print('mean wavelength = ',np.mean(photon_array.wavelength))
    print('min wavelength = ',np.min(photon_array.wavelength))
    print('max wavelength = ',np.max(photon_array.wavelength))

    assert np.min(photon_array.wavelength) > bandpass.blue_limit
    assert np.max(photon_array.wavelength) < bandpass.red_limit

    # This is a regression test
    np.testing.assert_almost_equal(np.mean(photon_array.wavelength), 622.755128, decimal=3)

    # If we use a flat SED (in photons/nm), then the mean sampled wavelength should very closely
    # match the bandpass effective wavelength.
    photon_array2 = galsim.PhotonArray(100000)
    sed2 = galsim.SED('1', 'nm', 'fphotons')
    sampler2 = galsim.WavelengthSampler(sed2, bandpass, rng)
    sampler2.applyTo(photon_array2)
    np.testing.assert_allclose(np.mean(photon_array2.wavelength),
                               bandpass.effective_wavelength,
                               rtol=0, atol=0.2,  # 2 Angstrom accuracy is pretty good
                               err_msg="Mean sampled wavelength not close to effective_wavelength")

    # Test that using this as a surface op works properly.

    # First do the shooting and clipping manually.
    im1 = galsim.Image(64,64,scale=1)
    im1.setCenter(0,0)
    photon_array.flux[photon_array.wavelength < 600] = 0.
    photon_array.addTo(im1.image.view())

    # Make a dummy surface op that clips any photons with lambda < 600
    class Clip600(object):
        def applyTo(self, photon_array):
            photon_array.flux[photon_array.wavelength < 600] = 0.

    # Use (a new) sampler and clip600 as surface_ops in drawImage
    im2 = galsim.Image(64,64,scale=1)
    im2.setCenter(0,0)
    clip600 = Clip600()
    rng2 = galsim.BaseDeviate(1234)
    sampler2 = galsim.WavelengthSampler(sed, bandpass, rng2)
    obj.drawImage(im2, method='phot', n_photons=nphotons, use_true_center=False,
                  surface_ops=[sampler2,clip600], rng=rng2)
    print('sum = ',im1.array.sum(),im2.array.sum())
    np.testing.assert_array_equal(im1.array, im2.array)


if __name__ == '__main__':
    test_photon_array()
    test_wavelength_sampler()
