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
    """Test the basic construction and use of the SiliconSensor class.
    """

    # Note: Use something quite small in terms of npixels so the B/F effect kicks in without
    # requiring a ridiculous number of photons
    obj = galsim.Gaussian(flux=10000, sigma=0.3)

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
    silicon = galsim.SiliconSensor(rng=rng1, diffusion_factor=0.0)
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

    # First, im2 and im3 should be essentially exactly equal.
    np.testing.assert_almost_equal(im2.array, im3.array, decimal=10)

    # im1 should be similar, but not equal
    np.testing.assert_almost_equal(im1.array/obj.flux, im2.array/obj.flux, decimal=2)

    # Fluxes should all equal obj.flux
    print('im1.array = ',im1.array)
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
    obj *= 200
    obj.drawImage(im1, method='fft', sensor=silicon, rng=rng)
    obj.drawImage(im2, method='fft', sensor=simple, rng=rng)
    obj.drawImage(im3, method='fft')

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
def test_sensor_wavelengths_and_angles():

    print('Starting test_wavelengths_and_angles')
    sys.stdout.flush()

    bppath = os.path.abspath(os.path.join(path, "../examples/data/"))
    sedpath = os.path.abspath(os.path.join(path, "../share/"))
    sed = galsim.SED(os.path.join(sedpath, 'CWW_E_ext.sed'), 'nm', 'flambda').thin()

    # Add the directions (not currently working?? seems to work - CL)
    fratio = 1.2
    obscuration = 0.2
    seed = 12345
    assigner = galsim.FRatioAngles(fratio, obscuration, seed)
    obj = galsim.Gaussian(flux=3539, sigma=0.3)

    if __name__ == "__main__":
        bands = ['r', 'i', 'z', 'y']
    else:
        bands = ['i'] # Only test the i band for nosetests

    for band in bands:
        bandpass = galsim.Bandpass(os.path.join(bppath, 'LSST_%s.dat'%band), 'nm').thin()
        rng3 = galsim.BaseDeviate(1234)
        sampler = galsim.WavelengthSampler(sed, bandpass, rng3)
        rng4 = galsim.BaseDeviate(5678)
        silicon = galsim.SiliconSensor(rng=rng4, diffusion_factor=0.0)    

        # We'll draw the same object using SiliconSensor
        im1 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=silicon, no wavelengths
        im2 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=silicon, with wavelengths
        im3 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=silicon, with angles
        im4 = galsim.ImageD(64, 64, scale=0.3)  # Will use sensor=silicon, with wavelengths, angles

        rng1 = galsim.BaseDeviate(5678)
        rng2 = galsim.BaseDeviate(5678)
        rng3 = galsim.BaseDeviate(5678)
        rng4 = galsim.BaseDeviate(5678)

        # Use photon shooting first
        obj.drawImage(im1, method='phot', sensor=silicon, rng=rng1)
        obj.drawImage(im2, method='phot', sensor=silicon, surface_ops=[sampler], rng=rng2)
        obj.drawImage(im3, method='phot', sensor=silicon, surface_ops=[assigner], rng=rng3)
        obj.drawImage(im4, method='phot', sensor=silicon, surface_ops=[sampler, assigner], rng=rng4)

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
        obj.drawImage(im2, method='fft', sensor=silicon, surface_ops=[sampler], rng=rng2)
        obj.drawImage(im3, method='fft', sensor=silicon, surface_ops=[assigner], rng=rng3)
        obj.drawImage(im4, method='fft', sensor=silicon, surface_ops=[sampler, assigner], rng=rng4)

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
def test_bf_slopes():
    """Test the brighter-fatter slopes
    with both the B-F effect and diffusion turned on and off.
    """
    from scipy import stats
    simple = galsim.Sensor()

    init_flux = 200000
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
    x_slope, intercept, r_value, p_value, std_err = stats.linregress(fluxes,x_moments[:,0])
    y_slope, intercept, r_value, p_value, std_err = stats.linregress(fluxes,y_moments[:,0])
    x_slope *= 50000.0 * 100.0
    y_slope *= 50000.0 * 100.0
    print('With BF turned on, diffusion off, x_slope = %.3f, y_slope = %.3f %% per 50K e-'%(
            x_slope, y_slope ))
    assert x_slope > 0.5
    assert y_slope > 0.5
    x_slope, intercept, r_value, p_value, std_err = stats.linregress(fluxes,x_moments[:,1])
    y_slope, intercept, r_value, p_value, std_err = stats.linregress(fluxes,y_moments[:,1])
    x_slope *= 50000.0 * 100.0
    y_slope *= 50000.0 * 100.0
    print('With BF turned on, diffusion on, x_slope = %.3f, y_slope = %.3f %% per 50K e-'%(
            x_slope, y_slope ))
    assert x_slope > 0.5
    assert y_slope > 0.5
    x_slope, intercept, r_value, p_value, std_err = stats.linregress(fluxes,x_moments[:,2])
    y_slope, intercept, r_value, p_value, std_err = stats.linregress(fluxes,y_moments[:,2])
    x_slope *= 50000.0 * 100.0
    y_slope *= 50000.0 * 100.0
    print('With BF turned off, x_slope = %.3f, y_slope = %.3f %% per 50K e-'%(x_slope, y_slope ))
    assert abs(x_slope) < 0.5
    assert abs(y_slope) < 0.5


if __name__ == "__main__":
    test_silicon()
    test_sensor_wavelengths_and_angles()
    test_silicon_fft()
    test_bf_slopes()
