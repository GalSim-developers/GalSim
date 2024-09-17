# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
import os
import sys
import time
from unittest import mock

import galsim
from galsim_test_helpers import *


@timer
def test_simple():
    """Test the default Sensor class that acts basically like not passing any sensor object.
    """

    # Start with photon shooting, since that's the most typical way that sensors are used.
    obj = galsim.Gaussian(flux=10000, sigma=1.3)

    # We'll draw the same object using SiliconSensor, Sensor, and the default (sensor=None)
    im1 = galsim.ImageD(64, 64, scale=0.3)  # Refefence image with sensor=None
    im2 = galsim.ImageD(64, 64, scale=0.3)  # Use sensor=simple
    im3 = galsim.ImageD(10, 10, scale=0.3)  # Small image, where some flux falls off the edge

    rng1 = galsim.BaseDeviate(5678)
    rng2 = galsim.BaseDeviate(5678)
    rng3 = galsim.BaseDeviate(5678)

    simple = galsim.Sensor()

    # Start with photon shooting, since that's more straightforward.
    obj.drawImage(im1, method='phot', poisson_flux=False, rng=rng1)
    obj.drawImage(im2, method='phot', poisson_flux=False, sensor=simple, rng=rng2)
    obj.drawImage(im3, method='phot', poisson_flux=False, rng=rng3)

    # Should be exactly equal
    np.testing.assert_array_equal(im2.array, im1.array)
    np.testing.assert_array_equal(im2.array[27:37,27:37], im3.array)

    # Fluxes on im1,im2 should all equal obj.flux
    print('added_flux = ',im1.added_flux, im2.added_flux, im3.added_flux)
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux, decimal=6)

    # Some flux missing from im3
    assert im3.added_flux < 0.8 * obj.flux
    np.testing.assert_almost_equal(im3.array.sum(), im3.added_flux, decimal=6)

    # Now test fft drawing, which is more complicated with possible temporaries and subsampling.
    im1 = galsim.ImageD(64, 64, scale=0.3)  # Reference image with sensor=None
    im2 = galsim.ImageD(64, 64, scale=0.3)  # Use sensor=simple
    im3 = galsim.ImageD(64, 64, scale=0.3)  # Use sensor=simple, no subsampling
    im4 = galsim.ImageCD(64, 64, scale=0.3) # Equivalent to image2, but checks using a temporary.
                                            # Also check add_to_image=True with im5.
    im5 = galsim.ImageD(64, 64, scale=0.3)  # Check manually convolving by the pixel.
    im6 = galsim.ImageD(64, 64, scale=0.3)  # Check manually convolving by the pixel, n_subsample=1

    # The rng shouldn't matter anymore for these, so just use the default rng=None
    obj.drawImage(im1, method='fft')
    obj.drawImage(im2, method='fft', sensor=simple)
    obj.drawImage(im3, method='fft', sensor=simple, n_subsample=1)
    obj.drawImage(im4, method='fft', sensor=simple, add_to_image=True)

    obj_with_pixel = galsim.Convolve(obj, galsim.Pixel(0.3))
    obj_with_pixel.drawImage(im5, method='no_pixel', sensor=simple)
    obj_with_pixel.drawImage(im6, method='no_pixel', sensor=simple, n_subsample=1)

    # Fluxes should all equal obj.flux
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux, decimal=3)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux, decimal=3)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux, decimal=3)
    np.testing.assert_almost_equal(im4.array.sum(), obj.flux, decimal=3)
    np.testing.assert_almost_equal(im5.array.sum(), obj.flux, decimal=3)
    np.testing.assert_almost_equal(im6.array.sum(), obj.flux, decimal=3)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux, decimal=3)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux, decimal=3)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux, decimal=3)
    np.testing.assert_almost_equal(im4.added_flux, obj.flux, decimal=3)
    np.testing.assert_almost_equal(im5.added_flux, obj.flux, decimal=3)
    np.testing.assert_almost_equal(im6.added_flux, obj.flux, decimal=3)

    # im1 and im2 are not precisely equal, since im2 was made with subsampling and then binning,
    # but with a largish object relative to the pixel, it's very close. (cf. similar test below
    # in test_silicon_fft, where the agreement is not so good.)
    print('max diff between im1, im2 with fft = ',np.max(np.abs(im2.array-im1.array)))
    np.testing.assert_almost_equal(im2.array, im1.array, decimal=10)

    # With no subsampling it should be nearly perfect (although this would be expected to be worse
    # when done with a real Sensor model).
    print('max diff without subsampling = ',np.max(np.abs(im3.array-im1.array)))
    np.testing.assert_almost_equal(im3.array, im1.array, decimal=12)

    # Using a temporary (and add_to_image) shouldn't affect anything for the D -> CD case.
    print('max diff with temporary = ',np.max(np.abs(im4.array-im2.array)))
    np.testing.assert_almost_equal(im4.array.real, im2.array, decimal=12)

    # Manual convolution should be identical to what 'fft' does automatically.
    print('max diff with manual pixel conv = ',np.max(np.abs(im5.array-im2.array)))
    #np.testing.assert_almost_equal(im5.array, im2.array, decimal=12)
    print('max diff with manual pixel conv, no subsampling = ',np.max(np.abs(im6.array-im3.array)))
    np.testing.assert_almost_equal(im6.array, im3.array, decimal=12)

    check_pickle(simple)


@timer
def test_silicon():
    """Test the basic construction and use of the SiliconSensor class.
    """

    # Note: Use something quite small in terms of npixels so the B/F effect kicks in without
    # requiring a ridiculous number of photons
    obj = galsim.Gaussian(flux=10000, sigma=0.3)

    # We'll draw the same object using SiliconSensor, Sensor, and the default (sensor=None)
    im1 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=silicon
    im2 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=simple
    im3 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=None
    im4 = galsim.ImageD(4, 4, scale=0.3)    # Smaller image, some flux will be missing.
    im5 = im1.copy()[galsim.BoundsI(17,20,33,36)]  # Same size as im4, but not contiguous.
    im6 = im1.copy()[im5.bounds].rot_180()         # Not unit step in either direction.
    im6.scale = 0.3

    rng1 = galsim.BaseDeviate(5678)
    rng2 = galsim.BaseDeviate(5678)
    rng3 = galsim.BaseDeviate(5678)
    rng4 = galsim.BaseDeviate(5678)
    rng5 = galsim.BaseDeviate(5678)
    rng6 = galsim.BaseDeviate(5678)

    silicon = galsim.SiliconSensor(rng=rng1, diffusion_factor=0.0)
    simple = galsim.Sensor()

    # Start with photon shooting, since that's more straightforward.
    obj.drawImage(im1, method='phot', poisson_flux=False, sensor=silicon, rng=rng1)
    obj.drawImage(im2, method='phot', poisson_flux=False, sensor=simple, rng=rng2)
    obj.drawImage(im3, method='phot', poisson_flux=False, rng=rng3)
    obj.drawImage(im4, method='phot', poisson_flux=False, sensor=silicon, rng=rng4)
    obj.drawImage(im5, method='phot', poisson_flux=False, sensor=silicon, rng=rng5)
    obj.drawImage(im6, method='phot', poisson_flux=False, sensor=silicon, rng=rng6)

    # First, im2 and im3 should be exactly equal.
    np.testing.assert_array_equal(im2.array, im3.array)

    # im1 should be similar, but not equal
    np.testing.assert_almost_equal(im1.array/obj.flux, im2.array/obj.flux, decimal=2)

    # im4 should match im1 where they overlap
    np.testing.assert_array_equal(im1.array[30:34,30:34], im4.array)

    # im5 and im6 should exactly match im4.
    np.testing.assert_array_equal(im5.array, im4.array)
    np.testing.assert_array_equal(im6.array, im4.array)

    # Fluxes on im1,im2 should equal obj.flux
    print('added_flux = ',im1.added_flux, im2.added_flux, im4.added_flux)
    np.testing.assert_almost_equal(im1.array.sum(), im1.added_flux, decimal=6)
    np.testing.assert_almost_equal(im2.array.sum(), im2.added_flux, decimal=6)
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux, decimal=6)

    # im4 should be missing flux, but report that fact correctly.
    np.testing.assert_almost_equal(im4.array.sum(), im4.added_flux, decimal=6)
    assert im4.added_flux < 0.95 * obj.flux

    # Now use a different seed for 3 to see how much of the variation is just from randomness.
    rng3.seed(234241)
    obj.drawImage(im3, method='phot', poisson_flux=False, rng=rng3)

    r1 = im1.calculateMomentRadius(flux=obj.flux)
    r2 = im2.calculateMomentRadius(flux=obj.flux)
    r3 = im3.calculateMomentRadius(flux=obj.flux)
    print('Flux = %.0f:  sum        peak          radius'%obj.flux)
    print('im1:         %.1f     %.2f       %f'%(im1.array.sum(),im1.array.max(), r1))
    print('im2:         %.1f     %.2f       %f'%(im2.array.sum(),im2.array.max(), r2))
    print('im3:         %.1f     %.2f       %f'%(im3.array.sum(),im3.array.max(), r3))

    # Fluxes should all equal obj.flux
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux, decimal=6)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux, decimal=6)

    # Sizes are all about equal since flux is not large enough for B/F to be significant
    # Variance of Irr for Gaussian with Poisson noise is
    # Var(Irr) = Sum(I r^4) = 4Irr [using Gaussian kurtosis = 8sigma^2, Irr = 2sigma^2]
    # r = sqrt(Irr/flux), so sigma(r) = 1/2 r sqrt(Var(Irr))/Irr = 1/sqrt(flux)
    # Use 2sigma for below checks.
    sigma_r = 1. / np.sqrt(obj.flux) * im1.scale
    np.testing.assert_allclose(r1, r2, atol=2.*sigma_r)
    np.testing.assert_allclose(r2, r3, atol=2.*sigma_r)

    # Repeat with 100X more photons where the brighter-fatter effect should kick in more.
    obj *= 100
    rng1 = galsim.BaseDeviate(5678)
    rng2 = galsim.BaseDeviate(5678)
    rng3 = galsim.BaseDeviate(5678)

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
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux, decimal=6)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux, decimal=6)

    # Sizes for 2,3 should be about equal, but 1 should be larger.
    sigma_r = 1. / np.sqrt(obj.flux) * im1.scale
    print('check |r2-r3| = %f <? %f'%(np.abs(r2-r3), 2.*sigma_r))
    np.testing.assert_allclose(r2, r3, atol=2.*sigma_r)
    print('check r1 - r3 = %f > %f due to brighter-fatter'%(r1-r2,sigma_r))
    assert r1 - r3 > 2*sigma_r

    # Check that it is really responding to flux, not number of photons.
    # Using fewer shot photons will mean each one encapsulates several electrons at once.
    obj.drawImage(im1, method='phot', n_photons=1000, poisson_flux=False, sensor=silicon,
                  rng=rng1)
    obj.drawImage(im2, method='phot', n_photons=1000, poisson_flux=False, sensor=simple,
                  rng=rng2)
    obj.drawImage(im3, method='phot', n_photons=1000, poisson_flux=False, rng=rng3)

    r1 = im1.calculateMomentRadius(flux=obj.flux)
    r2 = im2.calculateMomentRadius(flux=obj.flux)
    r3 = im3.calculateMomentRadius(flux=obj.flux)
    print('Flux = %.0f:  sum        peak          radius'%obj.flux)
    print('im1:         %.1f     %.2f       %f'%(im1.array.sum(),im1.array.max(), r1))
    print('im2:         %.1f     %.2f       %f'%(im2.array.sum(),im2.array.max(), r2))
    print('im3:         %.1f     %.2f       %f'%(im3.array.sum(),im3.array.max(), r3))

    np.testing.assert_almost_equal(im1.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux, decimal=6)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux, decimal=6)

    print('check r1 - r3 = %f > %f due to brighter-fatter'%(r1-r2,sigma_r))
    assert r1 - r3 > 2*sigma_r

    # Can also get the stronger BF effect with the strength parameter.
    obj /= 100  # Back to what it originally was.
    rng1 = galsim.BaseDeviate(5678)
    rng2 = galsim.BaseDeviate(5678)
    rng3 = galsim.BaseDeviate(5678)

    silicon = galsim.SiliconSensor(name='lsst_itl_8', strength=100., rng=rng1, diffusion_factor=0.0)
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
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux, decimal=6)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux, decimal=6)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux, decimal=6)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux, decimal=6)

    # Sizes for 2,3 should be about equal, but 1 should be larger.
    sigma_r = 1. / np.sqrt(obj.flux) * im1.scale
    print('check |r2-r3| = %f <? %f'%(np.abs(r2-r3), 2.*sigma_r))
    np.testing.assert_allclose(r2, r3, atol=2.*sigma_r)
    print('check r1 - r3 = %f > %f due to brighter-fatter'%(r1-r2,sigma_r))
    assert r1 - r3 > 2*sigma_r / 100

    # Check the construction with an explicit name
    s0 = galsim.SiliconSensor(rng=rng1)
    name = os.path.join(galsim.meta_data.share_dir, 'sensors', 'lsst_itl_50_8')
    s1 = galsim.SiliconSensor(name=name, strength=1.0, rng=rng1, diffusion_factor=1.0, qdist=3,
                              nrecalc=10000)
    assert s0 == s1
    s1 = galsim.SiliconSensor(name, 1.0, rng1, 1.0, 3, 10000)
    assert s0 == s1
    s2 = galsim.SiliconSensor(rng=rng1, name='lsst_itl_50_8')
    assert s0 == s2
    s3 = galsim.SiliconSensor(rng=rng1, strength=10.)
    s4 = galsim.SiliconSensor(rng=rng1, diffusion_factor=2.0)
    s5 = galsim.SiliconSensor(rng=rng1, qdist=4)
    s6 = galsim.SiliconSensor(rng=rng1, nrecalc=12345)
    s7 = galsim.SiliconSensor(name=name, strength=1.5, rng=rng1, diffusion_factor=1.3, qdist=4,
                              nrecalc=12345)
    for s in [ s3, s4, s5, s6, s7 ]:
        assert silicon != s
        assert s != s0

    check_pickle(s0)
    check_pickle(s1)
    check_pickle(s7)

    assert_raises(OSError, galsim.SiliconSensor, name='junk')
    assert_raises(OSError, galsim.SiliconSensor, name='output')
    assert_raises(TypeError, galsim.SiliconSensor, rng=3.4)
    assert_raises(TypeError, galsim.SiliconSensor, 'lsst_itl_8', rng1)

    # Invalid to accumulate onto undefined image.
    photons = galsim.PhotonArray(3)
    image = galsim.ImageD()
    with assert_raises(galsim.GalSimUndefinedBoundsError):
        simple.accumulate(photons, image)
    with assert_raises(galsim.GalSimUndefinedBoundsError):
        silicon.accumulate(photons, image)


@timer
def test_silicon_fft():
    """Test that drawing with method='fft' also works for SiliconSensor.
    """
    # Lower this somewhat so we get more accurate fluxes from the FFT.
    # (Still only accurate to 3 d.p. though.)
    gsparams = galsim.GSParams(maxk_threshold=1.e-5)
    obj = galsim.Gaussian(flux=3539, sigma=0.3, gsparams=gsparams)

    # We'll draw the same object using SiliconSensor, Sensor, and the default (sensor=None)
    im1 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=silicon
    im2 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=simple
    im3 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=None

    rng = galsim.BaseDeviate(5678)
    silicon = galsim.SiliconSensor(rng=rng, diffusion_factor=0.0)
    simple = galsim.Sensor()

    obj.drawImage(im1, method='fft', sensor=silicon, rng=rng)
    obj.drawImage(im2, method='fft', sensor=simple, rng=rng)
    obj.drawImage(im3, method='fft')

    printval(im1,im2)

    r1 = im1.calculateMomentRadius(flux=obj.flux)
    r2 = im2.calculateMomentRadius(flux=obj.flux)
    r3 = im3.calculateMomentRadius(flux=obj.flux)
    print('Flux = %.0f:  sum        peak         radius'%obj.flux)
    print('im1:         %.1f     %.2f       %f'%(im1.array.sum(),im1.array.max(), r1))
    print('im2:         %.1f     %.2f       %f'%(im2.array.sum(),im2.array.max(), r2))
    print('im3:         %.1f     %.2f       %f'%(im3.array.sum(),im3.array.max(), r3))

    # First, im2 and im3 should be almost exactly equal.  Not precisely, since im2 was made with
    # subsampling and then binning, so the FFT ringing is different (im3 is worse in this regard,
    # since it used convolution with a larger pixel).  So 3 digits is all we manage to get here.
    np.testing.assert_almost_equal(im2.array, im3.array, decimal=3)

    # im1 should be similar, but not equal
    np.testing.assert_almost_equal(im1.array/obj.flux, im2.array/obj.flux, decimal=2)

    # Fluxes should all equal obj.flux
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux, decimal=3)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux, decimal=3)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux, decimal=3)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux, decimal=3)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux, decimal=3)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux, decimal=3)

    # Sizes are all about equal since flux is not large enough for B/F to be significant
    sigma_r = 1./np.sqrt(obj.flux) * im1.scale
    np.testing.assert_allclose(r1, r2, atol=2.*sigma_r)
    np.testing.assert_allclose(r2, r3, atol=2.*sigma_r)

    # Repeat with 20X more photons where the brighter-fatter effect should kick in more.
    # (Also not in true_center, to hit a different branch about offset in the drawImage code.)
    obj *= 200
    obj.drawImage(im1, method='fft', sensor=silicon, rng=rng, use_true_center=False)
    obj.drawImage(im2, method='fft', sensor=simple, rng=rng, use_true_center=False)
    obj.drawImage(im3, method='fft', use_true_center=False)

    r1 = im1.calculateMomentRadius(flux=obj.flux)
    r2 = im2.calculateMomentRadius(flux=obj.flux)
    r3 = im3.calculateMomentRadius(flux=obj.flux)
    print('Flux = %.0f:  sum        peak          radius'%obj.flux)
    print('im1:         %.1f     %.2f       %f'%(im1.array.sum(),im1.array.max(), r1))
    print('im2:         %.1f     %.2f       %f'%(im2.array.sum(),im2.array.max(), r2))
    print('im3:         %.1f     %.2f       %f'%(im3.array.sum(),im3.array.max(), r3))

    # Fluxes should still be fine.
    np.testing.assert_almost_equal(im1.array.sum(), obj.flux, decimal=1)
    np.testing.assert_almost_equal(im2.array.sum(), obj.flux, decimal=1)
    np.testing.assert_almost_equal(im3.array.sum(), obj.flux, decimal=1)
    np.testing.assert_almost_equal(im1.added_flux, obj.flux, decimal=1)
    np.testing.assert_almost_equal(im2.added_flux, obj.flux, decimal=1)
    np.testing.assert_almost_equal(im3.added_flux, obj.flux, decimal=1)

    # Sizes for 2,3 should be about equal, but 1 should be larger.
    sigma_r = 1./np.sqrt(obj.flux) * im1.scale
    print('check |r2-r3| = %f <? %f'%(np.abs(r2-r3), 2.*sigma_r))
    np.testing.assert_allclose(r2, r3, atol=2.*sigma_r)
    print('check |r1-r3| = %f >? %f'%(np.abs(r1-r3), 2.*sigma_r))
    assert r1-r3 > 2.*sigma_r

@timer
def test_silicon_area():
    """Test the Silicon class calculate_pixel_areas() function.
    """
    # Adding this test to compare to Poisson simulation
    # Craig Lage - 27Apr18
    # Draw a very small spot with 80K electrons
    # This puts all of the charge in the central pixel to
    # match how the Poisson simulations are done
    obj = galsim.Gaussian(flux=8.0E4, sigma=0.02)
    im = obj.drawImage(nx=9, ny=9, scale=0.3, dtype=float)
    im.setCenter(0,0)

    rng = galsim.BaseDeviate(5678)
    # We compare to an explicit Poisson simulation that was equivalent to the model
    # save as lsst_itl_8, so use that here, not the default lsst_itl_50_8.
    silicon = galsim.SiliconSensor(name='lsst_itl_8', rng=rng)
    area_image = silicon.calculate_pixel_areas(im)
    # Get the area data from the Poisson simulation
    area_filename = silicon.vertex_file.split('/')[-1].strip('.dat')+'_areas.dat'
    area_dir = os.path.join(os.getcwd(),'sensor_validation/')
    area_data = np.loadtxt(area_dir+area_filename, skiprows = 1)
    # Now test that they are almost equal
    for line in area_data:
        nx = int(line[0]-4)
        ny = int(line[1]-4)
        poisson_area = line[2]
        galsim_area = area_image[nx,ny]
        #print(nx,ny,poisson_area,galsim_area)
        np.testing.assert_almost_equal(poisson_area/100.0, galsim_area, decimal=3)

    # NB. All regression values here and below are taken from GalSim version 2.3, computed with
    # lsst_itl_32, which is taken as truth for this purpose.
    # The rtol values are then chosen such that the lsst_itl_8 model passes the tests.
    print(area_image(0,0))
    print(area_image(1,0))
    print(area_image(-1,0))
    print(area_image(0,1))
    print(area_image(0,-1))
    np.testing.assert_allclose(area_image(0,0), 0.9291962869080659, rtol=1.e-3)
    np.testing.assert_allclose(area_image(1,0), 1.0056389435560653, rtol=1.e-3)
    np.testing.assert_allclose(area_image(-1,0), 1.0056399961006395, rtol=1.e-3)
    np.testing.assert_allclose(area_image(0,1), 1.0121748816976643, rtol=3.e-4)
    np.testing.assert_allclose(area_image(0,-1), 1.012139089334273, rtol=3.e-4)

    # Repeat with transpose=True to check that things are transposed properly.
    siliconT = galsim.SiliconSensor(name='lsst_itl_8', rng=rng, transpose=True)
    area_imageT = siliconT.calculate_pixel_areas(im)
    # This actually comes out exactly equal, but only test at single precision.
    np.testing.assert_allclose(area_imageT.array, area_image.array.T, rtol=1.e-8)

    # Draw a smallish but very bright Gaussian image
    obj = galsim.Gaussian(flux=5.e5, sigma=0.2)
    im = obj.drawImage(nx=17, ny=17, scale=0.3, dtype=float)
    im.setCenter(0,0)

    print('im min = ',im.array.min())
    print('im max = ',im.array.max())
    print('im(0,0) = ',im(0,0))
    assert im(0,0) == im.array.max()
    np.testing.assert_almost_equal(im(0,0), 149462.06966413918)

    rng = galsim.BaseDeviate(5678)
    silicon = galsim.SiliconSensor(rng=rng, name='lsst_itl_8', diffusion_factor=0.0)
    area_image = silicon.calculate_pixel_areas(im)
    print('area min = ',area_image.array.min())
    print('area max = ',area_image.array.max())
    print('area(0,0) = ',area_image(0,0))
    assert area_image(0,0) == area_image.array.min()
    np.testing.assert_allclose(area_image(0,0), 0.8977146150336172, rtol=2.e-3)
    np.testing.assert_allclose(area_image.array.max(), 1.0108970014387908, rtol=5.e-4)

    # The Silicon code is asymmetric.  Charge flows more easily along columns than rows.
    # It's not completely intuitive, since there are competing effects in play, but the net
    # result on the areas for this image is that the pixels above and below the central pixel
    # are slightly larger than the ones to the left and right.
    print('+- 1 along column: ',area_image(0,1),area_image(0,-1))
    print('+- 1 along row:    ',area_image(1,0),area_image(-1,0))
    np.testing.assert_allclose((area_image(0,1) + area_image(0,-1))/2., 0.9790312068001017,
                               rtol=4.e-4)
    np.testing.assert_allclose((area_image(1,0) + area_image(-1,0))/2., 0.9700828503223906,
                               rtol=2.e-3)

    # Just to confirm that the bigger effect really is along the column directions, draw the
    # object with the silicon sensor in play.
    im2 = obj.drawImage(nx=17, ny=17, scale=0.3, method='phot', sensor=silicon, rng=rng)
    im2.setCenter(0,0)
    print('im min = ',im2.array.min())
    print('im max = ',im2.array.max())
    print('im(0,0) = ',im2(0,0))
    print('+- 1 along column: ',im2(0,1),im2(0,-1))
    print('+- 1 along row:    ',im2(1,0),im2(-1,0))
    assert im2(0,0) == im2.array.max()
    assert im2(0,1) + im2(0,-1) > im2(1,0) + im2(-1,0)
    np.testing.assert_allclose(im2(0,0), 143345, rtol=3e-4)
    np.testing.assert_allclose((im2(0,1) + im2(0,-1))/2., 59280.5, rtol=1e-3)
    np.testing.assert_allclose((im2(1,0) + im2(-1,0))/2., 59030.0, rtol=1e-3)

    # Repeat with transpose=True to check that things are transposed properly.
    siliconT = galsim.SiliconSensor(rng=rng, name='lsst_itl_8', transpose=True, diffusion_factor=0.0)
    area_imageT = siliconT.calculate_pixel_areas(im)
    print('with transpose=True:')
    print('area min = ',area_imageT.array.min())
    print('area max = ',area_imageT.array.max())
    print('area(0,0) = ',area_imageT(0,0))
    print('+- 1 along column: ',area_imageT(0,1),area_imageT(0,-1))
    print('+- 1 along row:    ',area_imageT(1,0),area_imageT(-1,0))
    np.testing.assert_allclose(area_imageT(0,0), 0.897714615033617, rtol=2.e-3)
    np.testing.assert_allclose((area_imageT(0,1) + area_imageT(0,-1))/2., 0.9700828503223906,
                               rtol=2.e-3)
    np.testing.assert_allclose((area_imageT(1,0) + area_imageT(-1,0))/2., 0.9790312068001015,
                               rtol=4.e-4)
    # This actually comes out exactly equal, but only test at single precision.
    np.testing.assert_allclose(area_imageT.array, area_image.array.T, rtol=1.e-8)

    im2T = obj.drawImage(nx=17, ny=17, scale=0.3, method='phot', sensor=siliconT, rng=rng)
    im2T.setCenter(0,0)
    print('im min = ',im2T.array.min())
    print('im max = ',im2T.array.max())
    print('im(0,0) = ',im2T(0,0))
    print('+- 1 along column: ',im2T(0,1),im2T(0,-1))
    print('+- 1 along row:    ',im2T(1,0),im2T(-1,0))
    assert im2T(0,0) == im2T.array.max()
    assert im2T(0,1) + im2T(0,-1) < im2T(1,0) + im2T(-1,0)
    # Actual values are different, since rng is in different state. But qualitatively transposed.
    np.testing.assert_allclose(im2T(0,0), 142617, rtol=3e-4)
    np.testing.assert_allclose((im2T(0,1) + im2T(0,-1))/2., 59179.0, rtol=1e-3)
    np.testing.assert_allclose((im2T(1,0) + im2T(-1,0))/2., 59351.0, rtol=1e-3)

    check_pickle(siliconT)

    # Finally, the regular Sensor pixel areas are just 1.
    simple = galsim.Sensor()
    assert simple.calculate_pixel_areas(im) == 1.


@timer
def test_sensor_wavelengths_and_angles(run_slow):

    print('Starting test_wavelengths_and_angles')
    sys.stdout.flush()

    bppath = os.path.join(galsim.meta_data.share_dir, "bandpasses")
    sedpath = os.path.join(galsim.meta_data.share_dir, "SEDs")
    sed = galsim.SED(os.path.join(sedpath, 'CWW_E_ext.sed'), 'nm', 'flambda').thin()

    # Add the directions (not currently working?? seems to work - CL)
    fratio = 1.2
    obscuration = 0.2
    seed = 12345
    assigner = galsim.FRatioAngles(fratio, obscuration)
    obj = galsim.Gaussian(flux=3539, sigma=0.3)

    if run_slow:
        bands = ['r', 'i', 'z', 'y']
    else:
        bands = ['i'] # Only test the i band for pytest

    for band in bands:
        bandpass = galsim.Bandpass(os.path.join(bppath, 'LSST_%s.dat'%band), 'nm').thin()
        sampler = galsim.WavelengthSampler(sed, bandpass)
        rng = galsim.BaseDeviate(5678)
        silicon = galsim.SiliconSensor(rng=rng, diffusion_factor=0.0)

        # We'll draw the same object using SiliconSensor
        im1 = galsim.ImageF(64, 64, scale=0.3)  # Will use sensor=silicon, no wavelengths
        im2 = galsim.ImageF(64, 64, scale=0.3)  # Will use sensor=silicon, with wavelengths
        im3 = galsim.ImageF(64, 64, scale=0.3)  # Will use sensor=silicon, with angles
        im4 = galsim.ImageF(64, 64, scale=0.3)  # Will use sensor=silicon, with wavelengths, angles
        big_im = galsim.ImageF(164, 164, scale=0.3)
        im5 = big_im[galsim.BoundsI(30,93,100,163)] # Non-contiguous
        im6 = big_im.copy()[im5.bounds].rot_180()   # Non-unit steps
        im6.scale = 0.3

        rng1 = galsim.BaseDeviate(5678)
        rng2 = galsim.BaseDeviate(5678)
        rng3 = galsim.BaseDeviate(5678)
        rng4 = galsim.BaseDeviate(5678)
        rng5 = galsim.BaseDeviate(5678)
        rng6 = galsim.BaseDeviate(5678)

        # Use photon shooting first
        obj.drawImage(im1, method='phot', sensor=silicon, rng=rng1)
        obj.drawImage(im2, method='phot', sensor=silicon, photon_ops=[sampler], rng=rng2)
        obj.drawImage(im3, method='phot', sensor=silicon, photon_ops=[assigner], rng=rng3)
        silicon.updateRNG(rng4)
        obj.drawImage(im4, method='phot', sensor=silicon, photon_ops=[sampler, assigner], rng=rng4)
        silicon.updateRNG(rng5)
        obj.drawImage(im5, method='phot', sensor=silicon, photon_ops=[sampler, assigner], rng=rng5)
        silicon.updateRNG(rng6)
        obj.drawImage(im6, method='phot', sensor=silicon, photon_ops=[sampler, assigner], rng=rng6)
        np.testing.assert_array_equal(im5.array, im4.array)
        np.testing.assert_array_equal(im6.array, im4.array)

        r1 = im1.calculateMomentRadius(flux=obj.flux)
        r2 = im2.calculateMomentRadius(flux=obj.flux)
        r3 = im3.calculateMomentRadius(flux=obj.flux)
        r4 = im4.calculateMomentRadius(flux=obj.flux)
        print('Testing Wavelength and Angle sampling - %s band'%band)
        print('Flux = %.0f:                sum        peak          radius'%obj.flux)
        print('No lamb, no angles:         %.1f     %.2f       %f'%(
                im1.array.sum(),im1.array.max(), r1))
        print('W/ lamb, no angles:         %.1f     %.2f       %f'%(
                im2.array.sum(),im2.array.max(), r2))
        print('No lamb, w/ angles:         %.1f     %.2f       %f'%(
                im3.array.sum(),im3.array.max(), r3))
        print('W/ lamb, w/ angles:         %.1f     %.2f       %f'%(
                im4.array.sum(),im4.array.max(), r4))

        # r4 should be greater than r1 with wavelengths and angles turned on.
        sigma_r = 1. / np.sqrt(obj.flux) * im1.scale
        print('check r4 > r1 due to added wavelengths and angles')
        print('r1 = %f, r4 = %f, 2*sigma_r = %f'%(r1,r4,2.*sigma_r))
        assert r4 > r1

        # Now check fft
        obj.drawImage(im1, method='fft', sensor=silicon, rng=rng1)
        obj.drawImage(im2, method='fft', sensor=silicon, photon_ops=[sampler], rng=rng2)
        obj.drawImage(im3, method='fft', sensor=silicon, photon_ops=[assigner], rng=rng3)
        obj.drawImage(im4, method='fft', sensor=silicon, photon_ops=[sampler, assigner], rng=rng4)

        r1 = im1.calculateMomentRadius(flux=obj.flux)
        r2 = im2.calculateMomentRadius(flux=obj.flux)
        r3 = im3.calculateMomentRadius(flux=obj.flux)
        r4 = im4.calculateMomentRadius(flux=obj.flux)
        print('Testing Wavelength and Angle sampling - %s band'%band)
        print('Flux = %.0f:                sum        peak          radius'%obj.flux)
        print('No lamb, no angles:         %.1f     %.2f       %f'%(
                im1.array.sum(),im1.array.max(), r1))
        print('W/ lamb, no angles:         %.1f     %.2f       %f'%(
                im2.array.sum(),im2.array.max(), r2))
        print('No lamb, w/ angles:         %.1f     %.2f       %f'%(
                im3.array.sum(),im3.array.max(), r3))
        print('W/ lamb, w/ angles:         %.1f     %.2f       %f'%(
                im4.array.sum(),im4.array.max(), r4))

        # r4 should be greater than r1 with wavelengths and angles turned on.
        sigma_r = 1. / np.sqrt(obj.flux) * im1.scale
        print('check r4 > r1 due to added wavelengths and angles')
        print('r1 = %f, r4 = %f, 2*sigma_r = %f'%(r1,r4,2.*sigma_r))
        assert r4 > r1

@timer
def test_bad_wavelengths():
    """If a wavelength is outside the valid range of our tabulated absorption length table,
    make sure it does something reasonable, rather than abort.
    """
    rng = galsim.BaseDeviate(1234)

    nphot = 7
    x = rng.np.uniform(10,20, size=nphot)
    y = rng.np.uniform(10,20, size=nphot)
    flux = np.ones(nphot)
    dxdz = rng.np.uniform(0,0.5, size=nphot)
    dydz = rng.np.uniform(0,0.5, size=nphot)

    # The original bug that triggered this test involved photons with wavelength=0.
    # That's not physically possible, but now this works, using the min or max wavelength
    # in the lookup table for any photons that are too blue or too red respectively.
    # Note: the valid range is [255, 1450]
    wave = [0.0, 55, 255., 800., 1450., 4000., np.inf]
    photons = galsim.PhotonArray(nphot, x=x, y=y, flux=flux, dxdz=dxdz, dydz=dydz, wavelength=wave)

    image = galsim.Image(32,32)
    sensor = galsim.SiliconSensor(name='lsst_itl_50_8', rng=rng)
    sensor.accumulate(photons, image)

    # The real test is just that that didn't throw an exception.
    # But check that all the photons were put somewhere.
    assert np.sum(image.array) == nphot

    # The original code that triggered this was actually putting flux=0 for the bad wavelengths,
    # so let's do that too.
    flux = [0, 0, 1, 1, 1, 0, 0]
    photons = galsim.PhotonArray(nphot, x=x, y=y, flux=flux, dxdz=dxdz, dydz=dydz, wavelength=wave)
    image.setZero()
    sensor.accumulate(photons, image)
    assert np.sum(image.array) == np.sum(flux)


@timer
def test_bf_slopes():
    """Test the brighter-fatter slopes
    with both the B-F effect and diffusion turned on and off.
    """
    from scipy.stats import linregress

    simple = galsim.Sensor()

    init_flux = 400000
    obj = galsim.Gaussian(flux=init_flux, sigma=0.3)

    num_fluxes = 5
    x_moments = np.zeros([num_fluxes, 3])
    y_moments = np.zeros([num_fluxes, 3])
    fluxes = np.zeros([num_fluxes])

    for fluxmult in range(num_fluxes):
        rng1 = galsim.BaseDeviate(5678)
        rng2 = galsim.BaseDeviate(5678)
        rng3 = galsim.BaseDeviate(5678)
        # silicon1 has diffusion turned off, silicon2 has it turned on.
        silicon1 = galsim.SiliconSensor(rng=rng1, diffusion_factor=0.0)
        silicon2 = galsim.SiliconSensor(rng=rng2)

        # We'll draw the same object using SiliconSensor, Sensor, and the default (sensor=None)
        im1 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=silicon1 (diffusion off)
        im2 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=silicon2 (diffusion on)
        im3 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=simple

        obj1 = obj * (fluxmult + 1)

        obj1.drawImage(im1, method='phot', poisson_flux=False, sensor=silicon1, rng=rng1)
        obj1.drawImage(im2, method='phot', poisson_flux=False, sensor=silicon2, rng=rng2)
        obj1.drawImage(im3, method='phot', poisson_flux=False, sensor=simple, rng=rng3)

        print('Moments Mx, My, Mxx, Myy, Mxy for im1, flux = %f:'%obj1.flux)
        mom = galsim.utilities.unweighted_moments(im1)
        x_moments[fluxmult,0] = mom['Mxx']
        y_moments[fluxmult,0] = mom['Myy']
        print('Moments Mx, My, Mxx, Myy, Mxy for im2, flux = %f:'%obj1.flux)
        mom = galsim.utilities.unweighted_moments(im2)
        x_moments[fluxmult,1] = mom['Mxx']
        y_moments[fluxmult,1] = mom['Myy']
        print('Moments Mx, My, Mxx, Myy, Mxy for im3, flux = %f:'%obj1.flux)
        mom = galsim.utilities.unweighted_moments(im3)
        x_moments[fluxmult,2] = mom['Mxx']
        y_moments[fluxmult,2] = mom['Myy']
        fluxes[fluxmult] = im1.array.max()
    print('fluxes = ',fluxes)
    print('x_moments = ',x_moments[:,0])
    print('y_moments = ',y_moments[:,0])

    x_slope, intercept, r_value, p_value, std_err = linregress(fluxes,x_moments[:,0])
    y_slope, intercept, r_value, p_value, std_err = linregress(fluxes,y_moments[:,0])
    x_slope *= 50000.0 * 100.0
    y_slope *= 50000.0 * 100.0
    print('With BF turned on, diffusion off, x_slope = %.3f, y_slope = %.3f %% per 50K e-'%(
            x_slope, y_slope ))
    assert x_slope > 0.3
    assert y_slope > 0.3
    x_slope, intercept, r_value, p_value, std_err = linregress(fluxes,x_moments[:,1])
    y_slope, intercept, r_value, p_value, std_err = linregress(fluxes,y_moments[:,1])
    x_slope *= 50000.0 * 100.0
    y_slope *= 50000.0 * 100.0
    print('With BF turned on, diffusion on, x_slope = %.3f, y_slope = %.3f %% per 50K e-'%(
            x_slope, y_slope ))
    assert x_slope > 0.3
    assert y_slope > 0.3
    x_slope, intercept, r_value, p_value, std_err = linregress(fluxes,x_moments[:,2])
    y_slope, intercept, r_value, p_value, std_err = linregress(fluxes,y_moments[:,2])
    x_slope *= 50000.0 * 100.0
    y_slope *= 50000.0 * 100.0
    print('With BF turned off, x_slope = %.3f, y_slope = %.3f %% per 50K e-'%(x_slope, y_slope ))
    assert abs(x_slope) < 0.3
    assert abs(y_slope) < 0.3

def treering_function(r):
    return 0.5 * np.cos(r / 250. * 2.0 * np.pi)

@timer
def test_treerings():
    """Test the addition of tree rings.
    compare image positions with the simple sensor to
    a SiliconSensor with no tree rings and six
    different additions of tree rings.
    """
    # Set up the different sensors.
    treering_amplitude = 0.5
    rng1 = galsim.BaseDeviate(5678)
    rng2 = galsim.BaseDeviate(5678)
    rng3 = galsim.BaseDeviate(5678)
    rng4 = galsim.BaseDeviate(5678)
    rng5 = galsim.BaseDeviate(5678)
    rng6 = galsim.BaseDeviate(5678)
    rng7 = galsim.BaseDeviate(5678)
    sensor0 = galsim.Sensor()
    sensor1 = galsim.SiliconSensor(rng=rng1)
    tr2 = galsim.SiliconSensor.simple_treerings(treering_amplitude, 250.)
    sensor2 = galsim.SiliconSensor(rng=rng2, treering_func=tr2,
                                   treering_center=galsim.PositionD(-1000.0,0.0))
    sensor3 = galsim.SiliconSensor(rng=rng3, treering_func=tr2,
                                   treering_center=galsim.PositionD(1000.0,0.0))
    sensor4 = galsim.SiliconSensor(rng=rng4, treering_func=tr2,
                                   treering_center=galsim.PositionD(0.0,-1000.0))
    tr5 = galsim.SiliconSensor.simple_treerings(treering_amplitude, 250., r_max=2000, dr=1.)
    sensor5 = galsim.SiliconSensor(rng=rng5, treering_func=tr5,
                                   treering_center=galsim.PositionD(0.0,1000.0))

    # Now test the ability to read in a user-specified function
    tr6 = galsim.LookupTable.from_func(treering_function, x_min=0.0, x_max=2000)
    sensor6 = galsim.SiliconSensor(rng=rng6, treering_func=tr6,
                                   treering_center=galsim.PositionD(-1000.0,0.0))

    # Now test the ability to read in a lookup table
    tr7 = galsim.LookupTable.from_file('tree_ring_lookup.dat', amplitude=treering_amplitude)
    sensor7 = galsim.SiliconSensor(rng=rng7, treering_func=tr7,
                                   treering_center=galsim.PositionD(-1000.0,0.0))

    sensors = [sensor0, sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7]
    names = ['Sensor()',
             'SiliconSensor, no TreeRings',
             'SiliconSensor, TreeRingCenter= (-1000,0)',
             'SiliconSensor, TreeRingCenter= (1000,0)',
             'SiliconSensor, TreeRingCenter= (0,-1000)',
             'SiliconSensor, TreeRingCenter= (0,1000)',
             'SiliconSensor with specified function, TreeRingCenter= (-1000,0)',
             'SiliconSensor with lookup table, TreeRingCenter= (-1000,0)']
    centers = [None, None,
               (-1000,0),
               (1000,0),
               (0,-1000),
               (0,1000),
               (-1000,0),
               (-1000,0)]

    init_flux = 200000
    obj = galsim.Gaussian(flux=init_flux, sigma=0.3)

    im = galsim.ImageD(10,10, scale=0.3)
    obj.drawImage(im)
    ref_mom = galsim.utilities.unweighted_moments(im)
    print('Reference Moments Mx, My = (%f, %f):'%(ref_mom['Mx'], ref_mom['My']))

    for sensor, name, center in zip(sensors, names, centers):
        im = galsim.ImageD(10,10, scale=0.3)
        obj.drawImage(im, method='phot', sensor=sensor)
        mom = galsim.utilities.unweighted_moments(im)
        print('%s, Moments Mx, My = (%f, %f):'%(name, mom['Mx'], mom['My']))
        if center is None:
            np.testing.assert_almost_equal(mom['Mx'], ref_mom['Mx'], decimal = 1)
            np.testing.assert_almost_equal(mom['My'], ref_mom['My'], decimal = 1)
        else:
            np.testing.assert_almost_equal(ref_mom['Mx'] + treering_amplitude * center[0] / 1000,
                                           mom['Mx'], decimal=1)
            np.testing.assert_almost_equal(ref_mom['My'] + treering_amplitude * center[1] / 1000,
                                           mom['My'], decimal=1)

    assert_raises(TypeError, galsim.SiliconSensor, treering_func=lambda x:np.cos(x))
    assert_raises(TypeError, galsim.SiliconSensor, treering_func=tr7, treering_center=(3,4))

    # With tree rings, the pixel areas are non-trivial even with no flux.
    areas0 = sensor5.calculate_pixel_areas(im)
    print('min/max area0 = ',np.min(areas0.array),np.max(areas0.array))
    im.fill(0)
    areas1 = sensor5.calculate_pixel_areas(im)
    print('min/max area1 = ',np.min(areas1.array),np.max(areas1.array))
    areas2 = sensor5.calculate_pixel_areas(im, use_flux=False)
    np.testing.assert_allclose(areas1.array, areas2.array, rtol=2.e-8)
    print('min/max area2 = ',np.min(areas2.array),np.max(areas2.array))
    # But the areas with flux have a larger range (in both directions) because of the BFE
    assert np.min(areas0.array) < np.min(areas2.array)
    assert np.max(areas0.array) > np.max(areas2.array)

    # Check that pixel areas don't crash when tree ring center is at a pixel center (so r=0).
    sensor8 = galsim.SiliconSensor(rng=rng4, treering_func=tr2,
                                   treering_center=galsim.PositionD(1,1))
    im.fill(0)
    areas8 = sensor8.calculate_pixel_areas(im)
    print('min/max area1 = ',np.min(areas8.array),np.max(areas8.array))
    # Mostly checking that there aren't any nan's here from division by 0.
    assert np.min(areas8.array) > 0

    # Also check that things behave sensibly if the stamp is outside the arg range of
    # the treering function
    # tr6 has a max radius of 2000.
    im.setCenter(2000,2000)
    obj.drawImage(im, method='phot', sensor=sensor6, rng=rng6)
    print('im.sum = ',im.array.sum(), im.added_flux)
    np.testing.assert_allclose(im.array.sum(), im.added_flux)
    np.testing.assert_allclose(im.array.sum(), init_flux, rtol=1.e-2)

    areas6a = sensor6.calculate_pixel_areas(im, use_flux=True)
    assert np.min(areas6a.array) > 0
    print('areas6a = ',areas6a.array)
    areas6b = sensor6.calculate_pixel_areas(im, use_flux=False)
    assert np.min(areas6b.array) > 0
    # When the stamp is outside the range of the treering function, the areas will be
    # identical for the use_flux=False case.
    print('areas6b = ',areas6b.array)
    print('min, max = ',areas6b.array.min(), areas6b.array.max())
    assert np.all(areas6b.array == areas6b.array[0,0])
    # But not when use_flux=True
    assert not np.all(areas6a.array == areas6a.array[0,0])
    print('mean, std when use_flux=True: ', np.mean(areas6a.array), np.std(areas6a.array))
    np.testing.assert_allclose(np.mean(areas6a.array), np.mean(areas6b.array),
                               rtol=np.std(areas6a.array)/np.sqrt(np.prod(areas6b.array.shape))*3)


@timer
def test_resume(run_slow):
    """Test that the resume option for accumulate works properly.
    """
    # Note: This test is based on a script devel/lsst/treering_skybg_check.py

    rng = galsim.UniformDeviate(314159)

    if run_slow:
        flux_per_pixel = 40
        nx = 200
        ny = 200
        block_size = int(1.2e5)
        nrecalc = 1.e6
    else:
        flux_per_pixel = 40
        nx = 20
        ny = 20
        block_size = int(1.3e3)
        nrecalc = 1.e4

    expected_num_photons = nx * ny * flux_per_pixel
    pd = galsim.PoissonDeviate(rng, mean=expected_num_photons)
    num_photons = int(pd())  # Poisson realization of the given expected number of photons.
    #nrecalc = num_photons / 2  # Only recalc once.
    flux_per_photon = 1
    print('num_photons = ',num_photons,' .. expected = ',expected_num_photons)

    # Use treerings to make sure that aspect of the setup is preserved properly on resume
    treering_func = galsim.SiliconSensor.simple_treerings(0.5, 250.)
    treering_center = galsim.PositionD(-1000,0)
    sensor1 = galsim.SiliconSensor(rng=rng.duplicate(), nrecalc=nrecalc,
                                   treering_func=treering_func, treering_center=treering_center)
    sensor2 = galsim.SiliconSensor(rng=rng.duplicate(), nrecalc=nrecalc,
                                   treering_func=treering_func, treering_center=treering_center)
    sensor3 = galsim.SiliconSensor(rng=rng.duplicate(), nrecalc=nrecalc,
                                   treering_func=treering_func, treering_center=treering_center)

    waves = galsim.WavelengthSampler(sed = galsim.SED('1', 'nm', 'fphotons'),
                                     bandpass = galsim.Bandpass('LSST_r.dat', 'nm'))
    angles = galsim.FRatioAngles(1.2, 0.4)

    im1 = galsim.ImageF(nx,ny)  # Will not use resume
    im2 = galsim.ImageF(nx,ny)  # Will use resume
    im3 = galsim.ImageF(nx,ny)  # Will run all photons in one pass

    t_resume = 0
    t_no_resume = 0

    all_photons = galsim.PhotonArray(num_photons)
    all_photons.allocateWavelengths()
    all_photons.allocateAngles()
    n_added = 0

    first = True
    while num_photons > 0:
        print(num_photons,'photons left. image min/max =',im1.array.min(),im1.array.max())
        nphot = min(block_size, num_photons)
        num_photons -= nphot

        t0 = time.time()
        photons = galsim.PhotonArray(int(nphot))
        rng.generate(photons.x) # 0..1 so far
        photons.x *= nx
        photons.x += 0.5  # Now from xmin-0.5 .. xmax+0.5
        rng.generate(photons.y)
        photons.y *= ny
        photons.y += 0.5
        photons.flux = flux_per_photon
        waves.applyTo(photons, rng=rng)
        angles.applyTo(photons, rng=rng)

        all_photons.x[n_added:n_added+nphot] = photons.x
        all_photons.y[n_added:n_added+nphot] = photons.y
        all_photons.flux[n_added:n_added+nphot] = photons.flux
        all_photons.dxdz[n_added:n_added+nphot] = photons.dxdz
        all_photons.dydz[n_added:n_added+nphot] = photons.dydz
        all_photons.wavelength[n_added:n_added+nphot] = photons.wavelength
        n_added += nphot

        t1 = time.time()
        sensor1.accumulate(photons, im1)

        t2 = time.time()
        sensor2.accumulate(photons, im2, resume = not first)
        first = False
        t3 = time.time()
        print('Times = ',t1-t0,t2-t1,t3-t2)
        t_resume += t3-t2
        t_no_resume += t2-t1

    print('max diff = ',np.max(np.abs(im1.array - im2.array)))
    print('max rel diff = ',np.max(np.abs(im1.array - im2.array)/np.abs(im2.array)))
    np.testing.assert_almost_equal(im2.array/expected_num_photons, im1.array/expected_num_photons,
                                   decimal=5)
    print('Time with resume = ',t_resume)
    print('Time without resume = ',t_no_resume)
    assert t_resume < t_no_resume

    # The resume path should be exactly the same as doing all the photons at once.
    sensor3.accumulate(all_photons, im3)
    np.testing.assert_array_equal(im2.array, im3.array)

    # If resume is used either with the wrong image or on the first call to accumulate, then
    # this should raise an exception.
    assert_raises(RuntimeError, sensor3.accumulate, all_photons, im1, resume=True)
    sensor4 = galsim.SiliconSensor(rng=rng.duplicate(), nrecalc=nrecalc,
                                   treering_func=treering_func, treering_center=treering_center)
    assert_raises(RuntimeError, sensor4.accumulate, all_photons, im1, resume=True)

@timer
def test_flat(run_slow):
    """Test building a flat field image using the Silicon class.
    """
    # Note: This test is based on a script devel/lsst/treering_flat.py

    if run_slow:
        nx = 200
        ny = 200
        nflats = 20
        niter = 50
        toler = 0.01
    else:
        nx = 50
        ny = 50
        nflats = 3
        niter = 20  # Seem to really need 20 or more iterations to get covariances close.
        toler = 0.05

    counts_total = 80.e3
    counts_per_iter = counts_total / niter

    # Silicon sensor with tree rings
    seed = 31415
    rng = galsim.UniformDeviate(seed)
    treering_func = galsim.SiliconSensor.simple_treerings(0.26, 47)
    treering_center = galsim.PositionD(0,0)
    sensor = galsim.SiliconSensor(rng=rng,
                                   treering_func=treering_func, treering_center=treering_center)

    # Use a non-trivial WCS to make sure that works properly.
    wcs = galsim.FitsWCS('fits_files/tnx.fits')
    # We add on a border of 2 pixels, since the outer row/col get a little messed up by photons
    # falling off the edge, but not coming on from the other direction.
    # We do 2 rows/cols rather than just 1 to be safe, since I think diffusion can probably go
    # 2 pixels, even though the deficit is only really evident on the outer pixel.
    nborder = 2
    base_image = galsim.ImageF(nx+2*nborder, ny+2*nborder, wcs=wcs)
    base_image.wcs.makeSkyImage(base_image, sky_level=1.)

    # Rescale so that the mean sky level per pixel is skyCounts
    mean_pixel_area = base_image.array.mean()
    sky_level_per_iter = counts_per_iter / mean_pixel_area  # in ADU/arcsec^2 now.
    base_image *= sky_level_per_iter

    # The base_image now has the right level to account for the WCS distortion, but not any sensor
    # effects.
    # This is the noise-free level that we want to add each iteration modulated by the sensor.

    noise = galsim.PoissonNoise(rng)
    flats = []

    for n in range(nflats):
        print('n = ',n)
        # image is the image that we will build up in steps.
        image = galsim.ImageF(nx+2*nborder, ny+2*nborder, wcs=wcs)

        for i in range(niter):
            # temp is the additional flux we will add to the image in this iteration.
            # Start with the right area due to the sensor effects.
            temp = sensor.calculate_pixel_areas(image)
            temp /= temp.array.mean()

            # Multiply by the base image to get the right mean level and wcs effects
            temp *= base_image

            # Finally, add noise.  What we have here so far is the expectation value in each pixel.
            # We need to realize this according to Poisson statistics with these means.
            temp.addNoise(noise)

            # Add this to the image we are building up.
            image += temp

        # Cut off the outer border where things don't work quite right.
        image = image.subImage(galsim.BoundsI(1+nborder,nx+nborder,1+nborder,ny+nborder))
        flats.append(image.array)

    # These are somewhat noisy, so compute for all pairs and average them.
    mean = var = cov01 = cov10 = cov11a = cov11b = cov02 = cov20 = 0
    n = len(flats)
    npairs = 0

    for i in range(n):
        flati = flats[i]
        print('mean ',i,' = ',flati.mean())
        mean += flati.mean()
        for j in range(i+1,n):
            flatj = flats[j]
            diff = flati - flatj
            var += diff.var()/2
            cov01 += np.mean(diff[1:,:] * diff[:-1,:])
            cov10 += np.mean(diff[:,1:] * diff[:,:-1])
            cov11a += np.mean(diff[1:,1:] * diff[:-1,:-1])
            cov11b += np.mean(diff[1:,:-1] * diff[:-1,1:])
            cov02 += np.mean(diff[2:,:] * diff[:-2,:])
            cov20 += np.mean(diff[:,2:] * diff[:,:-2])
            npairs += 1
    mean /= n
    var /= npairs
    cov01 /= npairs
    cov10 /= npairs
    cov11a /= npairs
    cov11b /= npairs
    cov02 /= npairs
    cov20 /= npairs

    print('var(diff)/2 = ',var, 0.93*counts_total)
    print('cov01 = ',cov01, 0.03*counts_total)   # Note: I don't actually know if these are
    print('cov10 = ',cov10, 0.015*counts_total)  # the right covariances...
    print('cov11a = ',cov11a, cov11a/counts_total)
    print('cov11b = ',cov11b, cov11b/counts_total)
    print('cov02 = ',cov02, cov02/counts_total)
    print('cov20 = ',cov20, cov20/counts_total)

    # Mean should be close to target counts
    np.testing.assert_allclose(mean, counts_total, rtol=toler)
    # Variance is a bit less than the mean due to B/F.
    np.testing.assert_allclose(var, 0.93 * counts_total, rtol=1.5*toler)
    # 01 and 10 covariances are significant.
    np.testing.assert_allclose(cov01, 0.03 * counts_total, rtol=30*toler)
    np.testing.assert_allclose(cov10, 0.015 * counts_total, rtol=65*toler)
    # The rest are small
    np.testing.assert_allclose(cov11a / counts_total, 0., atol=2*toler)
    np.testing.assert_allclose(cov11b / counts_total, 0., atol=2*toler)
    np.testing.assert_allclose(cov20 / counts_total, 0., atol=2*toler)
    np.testing.assert_allclose(cov02 / counts_total, 0., atol=2*toler)

@timer
def test_omp():
    """Test setting the number of omp threads.
    """
    import multiprocessing

    # If num_threads <= 0 or None, get num from cpu_count
    cpus = multiprocessing.cpu_count()
    assert galsim.set_omp_threads(0) > 0
    assert galsim.set_omp_threads(0) <= cpus
    assert galsim.set_omp_threads(None) > 0
    assert galsim.set_omp_threads(None) <= cpus

    # If num_threads == 1, it should always set to 1
    assert galsim.set_omp_threads(1) == 1
    assert galsim.get_omp_threads() == 1

    # If num_threads > 1, it could be 1 or up to the input num_threads
    num_threads = galsim.set_omp_threads(2)
    assert galsim.set_omp_threads(2) >= 1
    assert galsim.set_omp_threads(2) <= 2
    assert galsim.get_omp_threads() == num_threads
    num_threads = galsim.set_omp_threads(20)
    assert galsim.set_omp_threads(20) >= 1
    assert galsim.set_omp_threads(20) <= 20
    assert galsim.get_omp_threads() == num_threads

    # Repeat and check that appropriate messages are emitted
    with CaptureLog() as cl:
        num_threads = galsim.set_omp_threads(0, logger=cl.logger)
    assert "multiprocessing.cpu_count() = " in cl.output
    assert "Telling OpenMP to use %s threads"%cpus in cl.output

    with CaptureLog() as cl:
        galsim.set_omp_threads(None, logger=cl.logger)
    assert "multiprocessing.cpu_count() = " in cl.output
    assert "Telling OpenMP to use %s threads"%cpus in cl.output

    with CaptureLog() as cl:
        galsim.set_omp_threads(1, logger=cl.logger)
    assert "multiprocessing.cpu_count() = " not in cl.output
    assert "Telling OpenMP to use 1 threads" in cl.output
    assert "Using %s threads"%num_threads not in cl.output
    assert "Unable to use multiple threads" not in cl.output

    with CaptureLog() as cl:
        galsim.set_omp_threads(2, logger=cl.logger)
    assert "multiprocessing.cpu_count() = " not in cl.output
    assert "Telling OpenMP to use 2 threads" in cl.output

    # It's hard to tell what happens in the next step, since we can't control what
    # galsim._galsim.SetOMPThreads does.  It depends on whether OpenMP is enabled and
    # how many cores are available.  So let's mock it up.
    with mock.patch('galsim.utilities._galsim') as _galsim:
        # First mock with OpenMP enables and able to use lots of threads
        _galsim.SetOMPThreads = lambda x: x
        assert galsim.set_omp_threads(20) == 20
        with CaptureLog() as cl:
            galsim.set_omp_threads(20, logger=cl.logger)
        assert "OpenMP reports that it will use 20 threads" in cl.output
        assert "Using 20 threads" in cl.output

        # Next only 4 threads available
        _galsim.SetOMPThreads = lambda x: 4 if x > 4 else x
        print(galsim.set_omp_threads(20))
        assert galsim.set_omp_threads(20) == 4
        with CaptureLog() as cl:
            galsim.set_omp_threads(20, logger=cl.logger)
        assert "OpenMP reports that it will use 4 threads" in cl.output
        assert "Using 4 threads" in cl.output

        assert galsim.set_omp_threads(2) == 2
        with CaptureLog() as cl:
            galsim.set_omp_threads(2, logger=cl.logger)
        assert "OpenMP reports that it will use 2 threads" in cl.output

        # Finally, no OpenMP
        _galsim.SetOMPThreads = lambda x: 1
        assert galsim.set_omp_threads(20) == 1
        with CaptureLog() as cl:
            galsim.set_omp_threads(20, logger=cl.logger)
        assert "OpenMP reports that it will use 1 threads" in cl.output
        assert "Unable to use multiple threads" in cl.output

    # This is really just for coverage.  Check that OMP_PROC_BIND gets set properly.
    with mock.patch('os.environ', {}):
        assert os.environ.get('OMP_PROC_BIND') is None
        galsim.get_omp_threads()
        assert os.environ.get('OMP_PROC_BIND') == 'false'

    with mock.patch('os.environ', {}):
        assert os.environ.get('OMP_PROC_BIND') is None
        galsim.set_omp_threads(4)
        assert os.environ.get('OMP_PROC_BIND') == 'false'


@timer
def test_big_then_small():
    # After the initial implementation of the GPU version of Silicon, it was possible to get
    # a segmentation fault if drawing onto a large image followed by a small image.
    # The problem was that delta is resized downwards, but the allocated memory is not
    # deallocated.  And rather than use a correct calculation for the number of pixels,
    # the number of pixels to copy to the target was incorrectly set from the maxPtr-Data
    # of the allocated memory in _delta.  This would write past the end of the target image and
    # potentially cause a seg fault.

    obj = galsim.Gaussian(flux=100000, sigma=20)

    im1 = galsim.ImageD(512, 512, scale=0.3)
    im1.setCenter(0,0)
    im2 = galsim.ImageD(32, 32, scale=0.3)
    im2.setCenter(0,0)

    rng1 = galsim.BaseDeviate(1234)
    rng2 = galsim.BaseDeviate(1234)
    silicon = galsim.SiliconSensor(diffusion_factor=0.0)

    obj.drawImage(im1, method='phot', poisson_flux=False, sensor=silicon, rng=rng1)
    obj.drawImage(im2, method='phot', poisson_flux=False, sensor=silicon, rng=rng2)

    # The main test is that it didn't seg fault.
    # But to make sure it did something reasonable, the two images should match where
    # they overlap.
    np.testing.assert_array_equal(im1[im2.bounds].array, im2.array)


if __name__ == "__main__":
    runtests(__file__)
