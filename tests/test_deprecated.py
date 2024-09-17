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

import os
import numpy as np

import galsim
from galsim_test_helpers import *


def check_dep(f, *args, **kwargs):
    """Check that some function raises a GalSimDeprecationWarning as a warning, but not an error.
    """
    # Check that f() raises a warning, but not an error.
    with assert_warns(galsim.GalSimDeprecationWarning):
        res = f(*args, **kwargs)
    return res


@timer
def test_gsparams():
    check_dep(galsim.GSParams, allowed_flux_variation=0.90)
    check_dep(galsim.GSParams, range_division_for_extrema=50)
    check_dep(galsim.GSParams, small_fraction_of_flux=1.e-6)


@timer
def test_phase_psf():
    atm = galsim.Atmosphere(screen_size=10.0, altitude=0, r0_500=0.15, suppress_warning=True)
    psf = atm.makePSF(exptime=0.02, time_step=0.01, diam=1.1, lam=1000.0)
    check_dep(galsim.PhaseScreenPSF.__getattribute__, psf, "img")
    check_dep(galsim.PhaseScreenPSF.__getattribute__, psf, "finalized")

@timer
def test_interpolant():
    d = check_dep(galsim.Delta, tol=1.e-2)
    assert d.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, d, 'tol') == d.gsparams.kvalue_accuracy
    n = check_dep(galsim.Nearest, tol=1.e-2)
    assert n.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, n, 'tol') == n.gsparams.kvalue_accuracy
    s = check_dep(galsim.SincInterpolant, tol=1.e-2)
    assert s.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, s, 'tol') == s.gsparams.kvalue_accuracy
    l = check_dep(galsim.Linear, tol=1.e-2)
    assert l.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, l, 'tol') == l.gsparams.kvalue_accuracy
    c = check_dep(galsim.Cubic, tol=1.e-2)
    assert c.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, c, 'tol') == c.gsparams.kvalue_accuracy
    q = check_dep(galsim.Quintic, tol=1.e-2)
    assert q.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, q, 'tol') == q.gsparams.kvalue_accuracy
    l3 = check_dep(galsim.Lanczos, 3, tol=1.e-2)
    assert l3.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, l3, 'tol') == l3.gsparams.kvalue_accuracy
    ldc = check_dep(galsim.Lanczos, 3, False, tol=1.e-2)
    assert ldc.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, ldc, 'tol') == ldc.gsparams.kvalue_accuracy
    l8 = check_dep(galsim.Lanczos, 8, tol=1.e-2)
    assert l8.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, l8, 'tol') == l8.gsparams.kvalue_accuracy
    l11 = check_dep(galsim.Interpolant.from_name, 'lanczos11', tol=1.e-2)
    assert l11.gsparams.kvalue_accuracy == 1.e-2
    assert check_dep(getattr, l11, 'tol') == l11.gsparams.kvalue_accuracy

@timer
def test_noise():
    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = 'real_galaxy_catalog_23.5_example.fits'
    real_cat = galsim.RealGalaxyCatalog(
        dir=real_gal_dir, file_name=real_gal_cat, preload=True)

    test_seed=987654
    test_index = 17
    cf_1 = real_cat.getNoise(test_index, rng=galsim.BaseDeviate(test_seed))
    im_2, pix_scale_2, var_2 = real_cat.getNoiseProperties(test_index)
    # Check the variance:
    var_1 = cf_1.getVariance()
    assert var_1==var_2,'Inconsistent noise variance from getNoise and getNoiseProperties'
    # Check the image:
    ii = galsim.InterpolatedImage(im_2, normalization='sb', calculate_stepk=False,
                                  calculate_maxk=False, x_interpolant='linear')
    cf_2 = check_dep(galsim.correlatednoise._BaseCorrelatedNoise,
                     galsim.BaseDeviate(test_seed), ii, im_2.wcs)
    cf_2 = cf_2.withVariance(var_2)
    assert cf_1==cf_2,'Inconsistent noise properties from getNoise and getNoiseProperties'

@timer
def test_randwalk_defaults():
    """
    Create a random walk galaxy and test that the getters work for
    default inputs
    """

    # try constructing with mostly defaults
    npoints=100
    hlr = 8.0
    rng = galsim.BaseDeviate(1234)
    rw=check_dep(galsim.RandomWalk, npoints, half_light_radius=hlr, rng=rng)

    assert rw.npoints==npoints,"expected npoints==%d, got %d" % (npoints, rw.npoints)
    assert rw.input_half_light_radius==hlr,\
        "expected hlr==%g, got %g" % (hlr, rw.input_half_light_radius)

    nobj=len(rw.points)
    assert nobj == npoints,"expected %d objects, got %d" % (npoints, nobj)

    pts=rw.points
    assert pts.shape == (npoints,2),"expected (%d,2) shape for points, got %s" % (npoints, pts.shape)
    np.testing.assert_almost_equal(rw.centroid.x, np.mean(pts[:,0]))
    np.testing.assert_almost_equal(rw.centroid.y, np.mean(pts[:,1]))

    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    rng2 = galsim.BaseDeviate(1234)
    rw2 = check_dep(galsim.RandomWalk, npoints, half_light_radius=hlr, rng=rng2, gsparams=gsp)
    assert rw2 != rw
    assert rw2 == rw.withGSParams(gsp)

    # Check that they produce identical images.
    psf = galsim.Gaussian(sigma=0.8)
    conv1 = galsim.Convolve(rw.withGSParams(gsp), psf)
    conv2 = galsim.Convolve(rw2, psf)
    im1 = conv1.drawImage()
    im2 = conv2.drawImage()
    assert im1 == im2

    # Check that image is not sensitive to use of rng by other objects.
    rng3 = galsim.BaseDeviate(1234)
    rw3=check_dep(galsim.RandomWalk, npoints, half_light_radius=hlr, rng=rng3)
    rng3.discard(523)
    conv1 = galsim.Convolve(rw, psf)
    conv3 = galsim.Convolve(rw3, psf)
    im1 = conv1.drawImage()
    im3 = conv2.drawImage()
    assert im1 == im3

    # Run some basic tests of correctness
    check_basic(conv1, "RandomWalk")
    im = galsim.ImageD(64,64, scale=0.5)
    do_shoot(conv1, im, "RandomWalk")
    do_kvalue(conv1, im, "RandomWalk")
    check_pickle(rw)
    check_pickle(conv1)
    check_pickle(conv1, lambda x: x.drawImage(scale=1))


@timer
def test_randwalk_repr():
    """
    test the repr and str work, and that a new object can be created
    using eval
    """

    npoints=100
    hlr = 8.0
    flux=1
    rw1=check_dep(galsim.RandomWalk,
        npoints,
        half_light_radius=hlr,
        flux=flux,
    )
    rw2=check_dep(galsim.RandomWalk,
        npoints,
        profile=galsim.Exponential(half_light_radius=hlr, flux=flux),
    )

    for rw in (rw1, rw2):


        # just make sure str() works, don't require eval to give
        # a consistent object back
        st=str(rw)

        # require eval(repr(rw)) to give a consistent object back

        new_rw = eval(repr(rw))

        assert new_rw.npoints == rw.npoints,\
            "expected npoints=%d got %d" % (rw.npoints,new_rw.npoints)

        mess="expected input_half_light_radius=%.16g got %.16g"
        assert new_rw.input_half_light_radius == rw.input_half_light_radius,\
            mess % (rw.input_half_light_radius,new_rw.input_half_light_radius)
        assert new_rw.flux == rw.flux,\
            "expected flux=%.16g got %.16g" % (rw.flux,new_rw.flux)

@timer
def test_randwalk_config():
    """
    test we get the same object using a configuration and the
    explicit constructor
    """

    hlr=2.0
    flux=np.pi
    gal_config1 = {
        'type':'RandomWalk',
        'npoints':100,
        'half_light_radius':hlr,
        'flux':flux,
    }
    gal_config2 = {
        'type':'RandomWalk',
        'npoints':150,
        'profile': {
            'type': 'Exponential',
            'half_light_radius': hlr,
            'flux': flux,
        }
    }

    for gal_config in (gal_config1, gal_config2):
        config={
            'gal':gal_config,
            'rng':galsim.BaseDeviate(31415),
        }

        rwc = check_dep(galsim.config.BuildGSObject, config, 'gal')[0]
        print(repr(rwc._profile))

        rw = check_dep(galsim.RandomWalk,
            gal_config['npoints'],
            half_light_radius=hlr,
            flux=flux,
        )

        assert rw.npoints==rwc.npoints,\
            "expected npoints==%d, got %d" % (rw.npoints, rwc.npoints)

        assert rw.input_half_light_radius==rwc.input_half_light_radius,\
            "expected hlr==%g, got %g" % (rw.input_half_light_radius, rw.input_half_light_radius)

        nobj=len(rw.points)
        nobjc=len(rwc.points)
        assert nobj==nobjc,"expected %d objects, got %d" % (nobj,nobjc)

        pts=rw.points
        ptsc=rwc.points
        assert (pts.shape == ptsc.shape),\
                "expected %s shape for points, got %s" % (pts.shape,ptsc.shape)

@timer
def test_withOrigin():
    from test_wcs import Cubic

    # First EuclideantWCS types:

    wcs_list = [ galsim.OffsetWCS(0.3, galsim.PositionD(1,1), galsim.PositionD(10,23)),
                 galsim.OffsetShearWCS(0.23, galsim.Shear(g1=0.1,g2=0.3), galsim.PositionD(12,43)),
                 galsim.AffineTransform(0.01,0.26,-0.26,0.02, galsim.PositionD(12,43)),
                 galsim.UVFunction(ufunc = lambda x,y: 0.2*x, vfunc = lambda x,y: 0.2*y),
                 galsim.UVFunction(ufunc = lambda x,y: 0.2*x, vfunc = lambda x,y: 0.2*y,
                                   xfunc = lambda u,v: u / scale, yfunc = lambda u,v: v / scale),
                 galsim.UVFunction(ufunc='0.2*x + 0.03*y', vfunc='0.01*x + 0.2*y'),
               ]

    color = 0.3
    for wcs in wcs_list:
        # Original version of the shiftOrigin tests in do_nonlocal_wcs using deprecated name.
        new_origin = galsim.PositionI(123,321)
        wcs3 = check_dep(wcs.withOrigin, new_origin)
        assert wcs != wcs3, name+' is not != wcs.withOrigin(pos)'
        wcs4 = wcs.local(wcs.origin, color=color)
        assert wcs != wcs4, name+' is not != wcs.local()'
        assert wcs4 != wcs, name+' is not != wcs.local() (reverse)'
        world_origin = wcs.toWorld(wcs.origin, color=color)
        if wcs.isUniform():
            if wcs.world_origin == galsim.PositionD(0,0):
                wcs2 = wcs.local(wcs.origin, color=color).withOrigin(wcs.origin)
                assert wcs == wcs2, name+' is not equal after wcs.local().withOrigin(origin)'
            wcs2 = wcs.local(wcs.origin, color=color).withOrigin(wcs.origin, wcs.world_origin)
            assert wcs == wcs2, name+' not equal after wcs.local().withOrigin(origin,world_origin)'
        world_pos1 = wcs.toWorld(galsim.PositionD(0,0), color=color)
        wcs3 = check_dep(wcs.withOrigin, new_origin)
        world_pos2 = wcs3.toWorld(new_origin, color=color)
        np.testing.assert_almost_equal(
                world_pos2.x, world_pos1.x, 7,
                'withOrigin(new_origin) returned wrong world position')
        np.testing.assert_almost_equal(
                world_pos2.y, world_pos1.y, 7,
                'withOrigin(new_origin) returned wrong world position')
        new_world_origin = galsim.PositionD(5352.7, 9234.3)
        wcs5 = check_dep(wcs.withOrigin, new_origin, new_world_origin, color=color)
        world_pos3 = wcs5.toWorld(new_origin, color=color)
        np.testing.assert_almost_equal(
                world_pos3.x, new_world_origin.x, 7,
                'withOrigin(new_origin, new_world_origin) returned wrong position')
        np.testing.assert_almost_equal(
                world_pos3.y, new_world_origin.y, 7,
                'withOrigin(new_origin, new_world_origin) returned wrong position')

    # Now some CelestialWCS types
    cubic_u = Cubic(2.9e-5, 2000., 'u')
    cubic_v = Cubic(-3.7e-5, 2000., 'v')
    center = galsim.CelestialCoord(23 * galsim.degrees, -13 * galsim.degrees)
    radec = lambda x,y: center.deproject_rad(cubic_u(x,y)*0.2, cubic_v(x,y)*0.2,
                                             projection='lambert')
    wcs_list = [ galsim.RaDecFunction(radec),
                 galsim.AstropyWCS('1904-66_TAN.fits', dir='fits_files'),
                 galsim.GSFitsWCS('tpv.fits', dir='fits_files'),
                 galsim.FitsWCS('sipsample.fits', dir='fits_files'),
               ]

    for wcs in wcs_list:
        # Original version of the shiftOrigin tests in do_celestial_wcs using deprecated name.
        new_origin = galsim.PositionI(123,321)
        wcs3 = wcs.shiftOrigin(new_origin)
        assert wcs != wcs3, name+' is not != wcs.shiftOrigin(pos)'
        wcs4 = wcs.local(wcs.origin)
        assert wcs != wcs4, name+' is not != wcs.local()'
        assert wcs4 != wcs, name+' is not != wcs.local() (reverse)'
        world_pos1 = wcs.toWorld(galsim.PositionD(0,0))
        wcs3 = wcs.shiftOrigin(new_origin)
        world_pos2 = wcs3.toWorld(new_origin)
        np.testing.assert_almost_equal(
                world_pos2.distanceTo(world_pos1) / galsim.arcsec, 0, 7,
                'shiftOrigin(new_origin) returned wrong world position')

@timer
def test_wfirst():
    """Test that the deprecated wfirst module works like the new roman module.
    """
    import galsim.roman
    check_dep(__import__, 'galsim.wfirst')

    assert galsim.wfirst.gain == galsim.roman.gain
    assert galsim.wfirst.pixel_scale == galsim.roman.pixel_scale
    assert galsim.wfirst.diameter == galsim.roman.diameter
    assert galsim.wfirst.obscuration == galsim.roman.obscuration
    assert galsim.wfirst.collecting_area == galsim.roman.collecting_area
    assert galsim.wfirst.exptime == galsim.roman.exptime
    assert galsim.wfirst.dark_current == galsim.roman.dark_current
    assert galsim.wfirst.nonlinearity_beta == galsim.roman.nonlinearity_beta
    assert galsim.wfirst.reciprocity_alpha == galsim.roman.reciprocity_alpha
    assert galsim.wfirst.read_noise == galsim.roman.read_noise
    assert galsim.wfirst.n_dithers == galsim.roman.n_dithers
    assert galsim.wfirst.thermal_backgrounds == galsim.roman.thermal_backgrounds
    assert galsim.wfirst.longwave_bands == galsim.roman.longwave_bands
    assert galsim.wfirst.shortwave_bands == galsim.roman.shortwave_bands
    assert galsim.wfirst.pupil_plane_file == galsim.roman.pupil_plane_file
    assert galsim.wfirst.pupil_plane_scale == galsim.roman.pupil_plane_scale
    assert galsim.wfirst.stray_light_fraction == galsim.roman.stray_light_fraction
    np.testing.assert_array_equal(galsim.wfirst.ipc_kernel, galsim.roman.ipc_kernel)
    np.testing.assert_array_equal(galsim.wfirst.persistence_coefficients,
                                  galsim.roman.persistence_coefficients)
    np.testing.assert_array_equal(galsim.wfirst.persistence_fermi_parameters,
                                  galsim.roman.persistence_fermi_parameters)
    assert galsim.wfirst.n_sca == galsim.roman.n_sca
    assert galsim.wfirst.n_pix_tot == galsim.roman.n_pix_tot
    assert galsim.wfirst.n_pix == galsim.roman.n_pix
    assert galsim.wfirst.jitter_rms == galsim.roman.jitter_rms
    assert galsim.wfirst.charge_diffusion == galsim.roman.charge_diffusion

    assert galsim.wfirst.getBandpasses is galsim.roman.getBandpasses
    assert galsim.wfirst.getSkyLevel is galsim.roman.getSkyLevel
    assert galsim.wfirst.getPSF is galsim.roman.getPSF
    assert galsim.wfirst.getWCS is galsim.roman.getWCS
    assert galsim.wfirst.findSCA is galsim.roman.findSCA
    assert galsim.wfirst.allowedPos is galsim.roman.allowedPos
    assert galsim.wfirst.bestPA is galsim.roman.bestPA
    assert galsim.wfirst.convertCenter is galsim.roman.convertCenter
    assert galsim.wfirst.applyNonlinearity is galsim.roman.applyNonlinearity
    assert galsim.wfirst.addReciprocityFailure is galsim.roman.addReciprocityFailure
    assert galsim.wfirst.applyIPC is galsim.roman.applyIPC
    assert galsim.wfirst.applyPersistence is galsim.roman.applyPersistence
    assert galsim.wfirst.allDetectorEffects is galsim.roman.allDetectorEffects
    assert galsim.wfirst.NLfunc is galsim.roman.NLfunc

@timer
def test_roman_psfs(run_slow):
    """Test the deprecated high_accuracy and approximate_struts options.
    """
    import galsim.roman

    test_kwargs = [
        ({ 'approximate_struts':True, 'high_accuracy':False },
         { 'pupil_bin':8 }),
        ({ 'approximate_struts':True, 'high_accuracy':True },
         { 'pupil_bin':4, 'gsparams':galsim.GSParams(folding_threshold=2.e-3) }),
        ({ 'approximate_struts':False, 'high_accuracy':False },
         { 'pupil_bin':4 }),
    ]
    if run_slow:
        test_kwargs.append(
            ({ 'approximate_struts':False, 'high_accuracy':True },
            { 'pupil_bin':1, 'gsparams':galsim.GSParams(folding_threshold=2.e-3) }),
        )

    use_sca = 5
    for kwargs1, kwargs2 in test_kwargs:
        psf1 = check_dep(galsim.roman.getPSF, use_sca, 'Y106', **kwargs1)
        psf2 = galsim.roman.getPSF(use_sca, 'Y106', **kwargs2)
        assert psf1 == psf2

    # Cheat to get coverage of False,True option without spending a long time doing the
    # pupil plane FFT for that one.
    with assert_raises(TypeError):
        check_dep(galsim.roman.getPSF, SCA=use_sca, bandpass='Z087',
                            approximate_struts=False, high_accuracy=True,
                            wavelength='Z087')

@timer
def test_surface_ops():

    # Based on test_sensor.py:test_wavelengths_and_angles, but massively simplified.

    rng = galsim.BaseDeviate(1234)

    fratio = 1.2
    obscuration = 0.2
    assigner = check_dep(galsim.FRatioAngles, fratio, obscuration, rng=rng)

    sed = galsim.SED('CWW_E_ext.sed', 'nm', 'flambda').thin()
    bandpass = galsim.Bandpass('LSST_i.dat', 'nm').thin()
    sampler = check_dep(galsim.WavelengthSampler, sed, bandpass, rng=rng)

    obj = galsim.Gaussian(flux=353, sigma=0.3)
    im = galsim.Image(63,63, scale=1)
    check_dep(obj.drawImage, im, method='phot', surface_ops=[sampler, assigner], rng=rng,
              save_photons=True)

    rng.reset(1234)
    assigner.rng.reset(rng)
    sampler.rng.reset(rng)
    photons = check_dep(obj.makePhot, surface_ops=[sampler, assigner], rng=rng)
    assert photons == im.photons

    rng.reset(1234)
    assigner.rng.reset(rng)
    sampler.rng.reset(rng)
    _, photons2 = check_dep(obj.drawPhot, image=im.copy(), surface_ops=[sampler, assigner], rng=rng)
    assert photons2 == im.photons


@timer
def test_midpoint_basic():
    """Test the basic functionality of the midpt() method.
    """
    # This shouldn't be super accurate, but just make sure it's not really broken.
    x = 0.01*np.arange(1000)
    f = x**2
    result = check_dep(galsim.integ.midpt, f, x)
    expected_val = 10**3./3.
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=2, verbose=True,
        err_msg='Simple test of midpt() method failed for f(x)=x^2 from 0 to 10')


@timer
def test_trapz_basic():
    """Test the basic functionality of the trapz() method.
    """
    # This shouldn't be super accurate, but just make sure it's not really broken.
    func = lambda x: x**2
    result = check_dep(galsim.integ.trapz, func, 0, 1)
    expected_val = 1.**3./3.
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=6, verbose=True,
        err_msg='Simple test of trapz() method failed for f(x)=x^2 from 0 to 1')

    result = check_dep(galsim.integ.trapz, func, 0, 1, np.linspace(0, 1, 100000))
    expected_val = 1.**3./3.
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=6, verbose=True,
        err_msg='Test of trapz() with points failed for f(x)=x^2 from 0 to 1')

    with assert_raises(ValueError):
        check_dep(galsim.integ.trapz, func, 0, 1, points=np.linspace(0, 1.1, 100))
    with assert_raises(ValueError):
        check_dep(galsim.integ.trapz, func, 0.1, 1, points=np.linspace(0, 1, 100))
    with assert_raises(TypeError):
        check_dep(galsim.integ.trapz, func, 0.1, 1, points=2.3)

@timer
def test_hsm_depr():
    hsmp = check_dep(galsim.hsm.HSMParams, max_moment_nsig2=25.0)
    assert hsmp.max_moment_nsig2 == 0.

@timer
def test_photon_array_depr():
    nphotons = 1000

    # First create from scratch
    photon_array = galsim.PhotonArray(nphotons)
    assert len(photon_array.x) == nphotons
    assert len(photon_array.y) == nphotons
    assert len(photon_array.flux) == nphotons
    assert not photon_array.hasAllocatedWavelengths()
    assert not photon_array.hasAllocatedAngles()
    assert not photon_array.hasAllocatedTimes()
    assert not photon_array.hasAllocatedPupil()

    # No deprecation warning using setter
    photon_array.dxdz = 0.17
    assert photon_array.hasAllocatedAngles()
    assert len(photon_array.dxdz) == nphotons
    assert len(photon_array.dydz) == nphotons
    np.testing.assert_array_equal(photon_array.dxdz, 0.17)
    np.testing.assert_array_equal(photon_array.dydz, 0.)

    photon_array.dydz = 0.59
    assert photon_array.hasAllocatedAngles()
    assert len(photon_array.dxdz) == nphotons
    assert len(photon_array.dydz) == nphotons
    np.testing.assert_array_equal(photon_array.dxdz, 0.17)
    np.testing.assert_array_equal(photon_array.dydz, 0.59)

    photon_array.wavelength = 500.
    assert photon_array.hasAllocatedWavelengths()
    assert len(photon_array.wavelength) == nphotons
    np.testing.assert_array_equal(photon_array.wavelength, 500)

    photon_array.pupil_u = 6.0
    assert photon_array.hasAllocatedPupil()
    assert len(photon_array.pupil_u) == nphotons
    assert len(photon_array.pupil_v) == nphotons
    np.testing.assert_array_equal(photon_array.pupil_u, 6.0)
    np.testing.assert_array_equal(photon_array.pupil_v, 0.0)

    photon_array.time = 0.0
    assert photon_array.hasAllocatedTimes()
    assert len(photon_array.time) == nphotons
    np.testing.assert_array_equal(photon_array.time, 0.0)

    # Using the getter is allowed, but deprecated.
    photon_array = galsim.PhotonArray(nphotons)
    dxdz = check_dep(getattr, photon_array, 'dxdz')
    assert photon_array.hasAllocatedAngles()
    assert photon_array.hasAllocatedAngles()
    assert len(photon_array.dxdz) == nphotons
    assert len(photon_array.dydz) == nphotons
    dxdz[:] = 0.17
    np.testing.assert_array_equal(photon_array.dxdz, 0.17)
    np.testing.assert_array_equal(photon_array.dydz, 0.)

    dydz = photon_array.dydz  # Allowed now.
    dydz[:] = 0.59
    np.testing.assert_array_equal(photon_array.dydz, 0.59)

    wave = check_dep(getattr, photon_array, 'wavelength')
    assert photon_array.hasAllocatedWavelengths()
    assert len(photon_array.wavelength) == nphotons
    wave[:] = 500.
    np.testing.assert_array_equal(photon_array.wavelength, 500)

    u = check_dep(getattr, photon_array, 'pupil_u')
    assert photon_array.hasAllocatedPupil()
    assert len(photon_array.pupil_u) == nphotons
    assert len(photon_array.pupil_v) == nphotons
    u[:] = 6.0
    np.testing.assert_array_equal(photon_array.pupil_u, 6.0)
    np.testing.assert_array_equal(photon_array.pupil_v, 0.0)
    v = photon_array.pupil_v
    v[:] = 10.0
    np.testing.assert_array_equal(photon_array.pupil_v, 10.0)

    t = check_dep(getattr, photon_array, 'time')
    assert photon_array.hasAllocatedTimes()
    assert len(photon_array.time) == nphotons
    np.testing.assert_array_equal(photon_array.time, 0.0)
    t[:] = 10
    np.testing.assert_array_equal(photon_array.time, 10.0)

    # For coverage, also need to test the two pair ones in other order.
    photon_array = galsim.PhotonArray(nphotons)
    dydz = check_dep(getattr, photon_array, 'dydz')
    assert photon_array.hasAllocatedAngles()
    assert photon_array.hasAllocatedAngles()
    assert len(photon_array.dxdz) == nphotons
    assert len(photon_array.dydz) == nphotons
    dydz[:] = 0.59
    np.testing.assert_array_equal(photon_array.dxdz, 0.)
    np.testing.assert_array_equal(photon_array.dydz, 0.59)

    dxdz = photon_array.dxdz  # Allowed now.
    dxdz[:] = 0.17
    np.testing.assert_array_equal(photon_array.dxdz, 0.17)

    v = check_dep(getattr, photon_array, 'pupil_v')
    assert photon_array.hasAllocatedPupil()
    assert len(photon_array.pupil_u) == nphotons
    assert len(photon_array.pupil_v) == nphotons
    v[:] = 10.0
    np.testing.assert_array_equal(photon_array.pupil_u, 0.0)
    np.testing.assert_array_equal(photon_array.pupil_v, 10.0)
    u = photon_array.pupil_u
    u[:] = 6.0
    np.testing.assert_array_equal(photon_array.pupil_u, 6.0)

    # Check assignAt
    pa1 = galsim.PhotonArray(50)
    pa1.x = photon_array.x[:50]
    for i in range(50):
        pa1.y[i] = photon_array.y[i]
    pa1.flux[0:50] = photon_array.flux[:50]
    pa1.dxdz = photon_array.dxdz[:50]
    pa1.dydz = photon_array.dydz[:50]
    pa1.pupil_u = photon_array.pupil_u[:50]
    pa1.pupil_v = photon_array.pupil_v[:50]
    pa2 = galsim.PhotonArray(100)
    check_dep(pa2.assignAt, 0, pa1)
    check_dep(pa2.assignAt, 50, pa1)
    np.testing.assert_almost_equal(pa2.x[:50], pa1.x)
    np.testing.assert_almost_equal(pa2.y[:50], pa1.y)
    np.testing.assert_almost_equal(pa2.flux[:50], pa1.flux)
    np.testing.assert_almost_equal(pa2.dxdz[:50], pa1.dxdz)
    np.testing.assert_almost_equal(pa2.dydz[:50], pa1.dydz)
    np.testing.assert_almost_equal(pa2.pupil_u[:50], pa1.pupil_u)
    np.testing.assert_almost_equal(pa2.pupil_v[:50], pa1.pupil_v)
    np.testing.assert_almost_equal(pa2.x[50:], pa1.x)
    np.testing.assert_almost_equal(pa2.y[50:], pa1.y)
    np.testing.assert_almost_equal(pa2.flux[50:], pa1.flux)
    np.testing.assert_almost_equal(pa2.dxdz[50:], pa1.dxdz)
    np.testing.assert_almost_equal(pa2.dydz[50:], pa1.dydz)
    np.testing.assert_almost_equal(pa2.pupil_u[50:], pa1.pupil_u)
    np.testing.assert_almost_equal(pa2.pupil_v[50:], pa1.pupil_v)

    # Error if it doesn't fit.
    with assert_raises(ValueError):
        check_dep(pa2.assignAt, 90, pa1)

@timer
def test_chromatic_flux():
    # This is based on a snippet of test_chromatic_flux in test_chromatic.py.

    bulge_SED = galsim.SED('CWW_E_ext.sed', wave_type='ang', flux_type='flambda')
    star = galsim.Gaussian(fwhm=1e-8) * bulge_SED
    mono_PSF = galsim.Gaussian(half_light_radius=0.8)
    zenith_angle = 20 * galsim.degrees
    bandpass = galsim.Bandpass('LSST_i.dat', 'nm').thin()
    PSF = galsim.ChromaticAtmosphere(mono_PSF, base_wavelength=500,
                                     zenith_angle=zenith_angle)
    PSF = PSF * 1.0
    PSF1 = PSF.interpolate(waves=np.linspace(bandpass.blue_limit, bandpass.red_limit, 30),
                          use_exact_sed=False)

    # Check deprecated use_exact_SED kwarg
    PSF2 = check_dep(PSF.interpolate,
                     waves=np.linspace(bandpass.blue_limit, bandpass.red_limit, 30),
                     use_exact_SED=False)
    assert PSF2 == PSF1

    # Also do this manually with the InterpolatedChromaticObject class
    PSF3 = check_dep(galsim.InterpolatedChromaticObject, PSF,
                     waves=np.linspace(bandpass.blue_limit, bandpass.red_limit, 30),
                     use_exact_SED=False)
    assert PSF3 == PSF1

    # And check deprecated SED attribute.
    sed = check_dep(getattr, PSF, 'SED')
    assert sed == PSF.sed
    sed1 = check_dep(getattr, PSF1, 'SED')
    assert sed1 == PSF1.sed
    sed2 = check_dep(getattr, mono_PSF, 'SED')
    assert sed1 == mono_PSF.sed == galsim.SED(1, 'nm', '1')

@timer
def test_W149():
    # Based on test_config_psf, using old W149 name.

    config = {
        'modules' : ['galsim.roman'],
        'psf' : { 'type' : 'RomanPSF', 'SCA': 4, 'bandpass': 'W149' }
    }

    galsim.config.ImportModules(config)
    psf1 = check_dep(galsim.config.BuildGSObject, config, 'psf')[0]
    psf2 = check_dep(galsim.roman.getPSF, SCA=4, bandpass='W149')
    print('psf1 = ',str(psf1))
    print('psf2 = ',str(psf2))
    assert psf1 == psf2

    config = galsim.config.CleanConfig(config)
    config['image'] = {
        'bandpass' : { 'type' : 'RomanBandpass', 'name' : 'W149' }
    }
    config['psf']['wavelength'] = 985
    config['psf']['pupil_bin'] = 8
    bp = check_dep(galsim.config.BuildBandpass, config['image'], 'bandpass', config)[0]
    config['bandpass'] = bp
    psf1 = check_dep(galsim.config.BuildGSObject, config, 'psf')[0]
    psf2 = check_dep(galsim.roman.getPSF, SCA=4, bandpass='W149', pupil_bin=8, wavelength=985.)
    print('psf1 = ',str(psf1))
    print('psf2 = ',str(psf2))
    assert psf1 == psf2

@timer
def test_photon_array_correlated():

    # This is part of test_convolve in test_photon_array.py
    nphotons = 1000
    obj = galsim.Gaussian(flux=1.7, sigma=2.3)
    rng = galsim.UniformDeviate(1234)
    pa1 = obj.shoot(nphotons, rng)
    rng2 = rng.duplicate()  # Save this state.
    pa2 = obj.shoot(nphotons, rng)

    # If not correlated then convolve is deterministic
    conv_x = pa1.x + pa2.x
    conv_y = pa1.y + pa2.y
    conv_flux = pa1.flux * pa2.flux * nphotons

    np.testing.assert_allclose(np.sum(pa1.flux), 1.7)
    np.testing.assert_allclose(np.sum(pa2.flux), 1.7)
    np.testing.assert_allclose(np.sum(conv_flux), 1.7*1.7)

    np.testing.assert_allclose(np.sum(pa1.x**2)/nphotons, 2.3**2, rtol=0.1)
    np.testing.assert_allclose(np.sum(pa2.x**2)/nphotons, 2.3**2, rtol=0.1)
    np.testing.assert_allclose(np.sum(conv_x**2)/nphotons, 2.*2.3**2, rtol=0.1)

    np.testing.assert_allclose(np.sum(pa1.y**2)/nphotons, 2.3**2, rtol=0.1)
    np.testing.assert_allclose(np.sum(pa2.y**2)/nphotons, 2.3**2, rtol=0.1)
    np.testing.assert_allclose(np.sum(conv_y**2)/nphotons, 2.*2.3**2, rtol=0.1)

    pa3 = galsim.PhotonArray(nphotons)
    pa3.copyFrom(pa1)  # copy from pa1
    pa3.convolve(pa2)
    np.testing.assert_allclose(pa3.x, conv_x)
    np.testing.assert_allclose(pa3.y, conv_y)
    np.testing.assert_allclose(pa3.flux, conv_flux)

    # If one of them is correlated, it is still deterministic.
    pa3.copyFrom(pa1)
    check_dep(pa3.setCorrelated)
    pa3.convolve(pa2)
    np.testing.assert_allclose(pa3.x, conv_x)
    np.testing.assert_allclose(pa3.y, conv_y)
    np.testing.assert_allclose(pa3.flux, conv_flux)

    pa3.copyFrom(pa1)
    check_dep(pa3.setCorrelated, False)
    check_dep(pa2.setCorrelated)
    pa3.convolve(pa2)
    np.testing.assert_allclose(pa3.x, conv_x)
    np.testing.assert_allclose(pa3.y, conv_y)
    np.testing.assert_allclose(pa3.flux, conv_flux)

    # But if both are correlated, then it's not this simple.
    pa3.copyFrom(pa1)
    check_dep(pa3.setCorrelated)
    assert check_dep(pa3.isCorrelated)
    assert check_dep(pa2.isCorrelated)
    pa3.convolve(pa2, rng=rng)
    with assert_raises(AssertionError):
        np.testing.assert_allclose(pa3.x, conv_x)
    with assert_raises(AssertionError):
        np.testing.assert_allclose(pa3.y, conv_y)
    np.testing.assert_allclose(np.sum(pa3.flux), 1.7*1.7)
    np.testing.assert_allclose(np.sum(pa3.x**2)/nphotons, 2*2.3**2, rtol=0.1)
    np.testing.assert_allclose(np.sum(pa3.y**2)/nphotons, 2*2.3**2, rtol=0.1)

    # Can also effect the convolution by treating the psf as a PhotonOp
    pa3.copyFrom(pa1)
    check_dep(pa3.setCorrelated)
    obj.applyTo(pa3, rng=rng2)
    np.testing.assert_allclose(pa3.x, conv_x)
    np.testing.assert_allclose(pa3.y, conv_y)
    np.testing.assert_allclose(pa3.flux, conv_flux)

    # Check the is_corr flag gets set
    assert not check_dep(pa1.isCorrelated)
    pa3 = check_dep(galsim.PhotonArray.fromArrays, pa1.x, pa1.y, pa1.flux, is_corr=True)
    assert check_dep(pa3.isCorrelated)

    # Check toggling is_corr
    assert not check_dep(pa1.isCorrelated)
    check_dep(pa1.setCorrelated)
    assert check_dep(pa1.isCorrelated)
    check_dep(pa1.setCorrelated, False)
    assert not check_dep(pa1.isCorrelated)
    check_dep(pa1.setCorrelated, True)
    assert check_dep(pa1.isCorrelated)


@timer
def test_atredshift():
    """Test the equivalence of obj.atRedshift and the equivalent with SED.atRedshift
    """
    from test_chromatic import disk_SED, bulge_SED, bandpass

    # First simple separable galaxy
    gal = galsim.Sersic(n=3, half_light_radius=1.5)
    gal = gal.shear(g1=0.3, g2=0.2)
    gal1 = gal * bulge_SED
    gal2 = gal * bulge_SED.atRedshift(1.7)

    psf = galsim.Moffat(beta=2.5, half_light_radius=0.3)
    psf = galsim.ChromaticAtmosphere(psf, base_wavelength=500.0, zenith_angle=17*galsim.degrees)

    final1 = galsim.Convolve(check_dep(gal1.atRedshift, 1.7), psf)
    final2 = galsim.Convolve(gal2, psf)
    final3 = galsim.Convolve(check_dep(gal1.expand(lambda w:1.0).atRedshift, 1.7), psf)

    image1 = final1.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    image2 = final2.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    image3 = final3.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    np.testing.assert_allclose(image1.array, image2.array)
    np.testing.assert_allclose(image1.array, image3.array, atol=1.e-5)
    check_pickle(final1)

    # ChromaticSum
    gal1 = gal * bulge_SED + gal.dilate(1.3) * disk_SED
    gal2 = gal * bulge_SED.atRedshift(1.7) + gal.dilate(1.3) * disk_SED.atRedshift(1.7)
    gal3 = check_dep((gal * bulge_SED).atRedshift, 1.7) + \
           check_dep((gal.dilate(1.3) * disk_SED).atRedshift, 1.7)
    final1 = galsim.Convolve(check_dep(gal1.atRedshift, 1.7), psf)
    final2 = galsim.Convolve(gal2, psf)
    final3 = galsim.Convolve(gal3, psf)
    image1 = final1.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    image2 = final2.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    image3 = final3.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    np.testing.assert_allclose(image1.array, image2.array)
    np.testing.assert_allclose(image1.array, image3.array)

    # Probably none of the other Chromatic classes make sense to call atRedshift, so let
    # them use the base class implementation if they do so.
    # Just check ChromaticConvolution as one that doesn't have its own implementation.
    # (This is probably the least implausible use of atRedshift for one of these other classes.)
    smear = galsim.ChromaticObject(galsim.Gaussian(sigma=0.4))
    smear1 = smear.expand(lambda wave: (wave/700)**0.3)
    gal1 = galsim.Convolve(gal * bulge_SED, smear1)
    # Smear is at redshift 1.7, so scale wave by factor of (1+1.7)
    smear2 = smear.expand(lambda wave: (wave/700/2.7)**0.3)
    gal2 = galsim.Convolve(gal * bulge_SED.atRedshift(1.7), smear2)
    final1 = galsim.Convolve(check_dep(gal1.atRedshift, 1.7), psf)
    final2 = galsim.Convolve(gal2, psf)
    image1 = final1.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    image2 = final2.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    np.testing.assert_allclose(image1.array, image2.array, atol=1.e-6)
    assert gal1.redshift == 0.
    assert check_dep(gal1.atRedshift, 1.7).redshift == 1.7
    assert gal2.redshift == 1.7

    # Finally, if we call atRedshift on a regular GSObject, it doesn't do much.
    gal3 = check_dep(gal.atRedshift, 1.7)
    assert gal == gal3
    # But it does add an attribute, which doesn't obviate equality.
    assert gal3.redshift == 1.7
    assert gal.redshift == 0.

    # One deprecated test from test_chromatic in test_config_image.py.
    # (Simplified to just test the deprecation.)
    config = {
        'gal': {
            'type': 'Exponential',
            'half_light_radius': 0.5,
            'sed': {
                'file_name': 'CWW_E_ext.sed',
                'wave_type': 'Ang',
                'flux_type': 'flambda',
                'norm_flux_density': 1.0,
                'norm_wavelength': 500,
            },
            'redshift': 0.8,
        },
    }
    gal1, _ = check_dep(galsim.config.BuildGSObject, config, 'gal')
    sed = galsim.SED('CWW_E_ext.sed', 'Ang', 'flambda').withFluxDensity(1.0, 500).atRedshift(0.8)
    gal2 = galsim.Exponential(half_light_radius=0.5) * sed
    assert gal1 == gal2

    config = {
        'gal': {
            'type': 'Sum',
            'items': [
                {
                    'type': 'DeVaucouleurs',
                    'half_light_radius': 0.5,
                    'sed': {
                        'file_name': 'CWW_E_ext.sed',
                        'wave_type': 'Ang',
                        'flux_type': 'flambda',
                        'norm_flux_density': 1.0,
                        'norm_wavelength': 500,
                    },
                },
                {
                    'type': 'Exponential',
                    'half_light_radius': 2.0,
                    'sed': {
                        'file_name': 'CWW_Im_ext.sed',
                        'wave_type': 'Ang',
                        'flux_type': 'flambda',
                        'norm_flux_density': 1.0,
                        'norm_wavelength': 500,
                    },
                },
            ],
            'redshift': 0.8,
        },
    }
    gal1, _ = check_dep(galsim.config.BuildGSObject, config, 'gal')
    sed1 = galsim.SED('CWW_E_ext.sed', 'Ang', 'flambda').withFluxDensity(1.0, 500).atRedshift(0.8)
    sed2 = galsim.SED('CWW_Im_ext.sed', 'Ang', 'flambda').withFluxDensity(1.0, 500).atRedshift(0.8)
    gal2 = galsim.DeVaucouleurs(half_light_radius=0.5) * sed1 + \
           galsim.Exponential(half_light_radius=2.0) * sed2
    print(gal1)
    print(gal2)
    print(gal1.obj_list == gal2.obj_list)
    print(gal1.obj_list[0] == gal2.obj_list[0])
    print(gal1.obj_list[0].original == gal2.obj_list[0].original)
    print(gal1.obj_list[0].sed == gal2.obj_list[0].sed)
    print(gal1.obj_list[1] == gal2.obj_list[1])
    print(gal1._gsparams == gal2._gsparams)
    print(gal1._propagate_gsparams == gal2._propagate_gsparams)
    assert gal1 == gal2


@timer
def test_save_photons():
    # These are the deprecated cases in test_save_photons.
    from test_chromatic import disk_SED, bulge_SED

    rng = galsim.BaseDeviate(1234)
    star_sed = galsim.SED('vega.txt', wave_type="nm", flux_type="fphotons")
    bandpass = galsim.Bandpass("LSST_r.dat", wave_type="nm")

    airy = galsim.ChromaticAiry(lam=500, diam=8)
    optical = galsim.ChromaticOpticalPSF(lam=500, diam=8, defocus=0.2, obscuration=0.3)
    disk = galsim.Exponential(half_light_radius=0.5).shear(g1=0.4, g2=0.2)
    bulge = galsim.Sersic(n=3, half_light_radius=0.3)

    objs = [
        check_dep((disk * disk_SED).atRedshift, 1.1),
        check_dep((optical * star_sed).atRedshift, 1.1),
        check_dep((bulge * bulge_SED + disk * disk_SED).atRedshift, 0.5),
        check_dep((airy * star_sed).expand(lambda w: (w/500)**0.0).atRedshift, 0.2),
        check_dep(galsim.ChromaticTransformation, disk, flux_ratio=disk_SED, redshift=1.1),
    ]

    flux = 1000
    for obj in objs:
        print('obj = ',obj)
        obj = obj.withFlux(flux, bandpass)
        image = obj.drawImage(bandpass=bandpass, method="phot",
                              n_photons=flux, save_photons=True,
                              scale=0.05, nx=32, ny=32, rng=rng)
        assert hasattr(image, 'photons')
        assert len(image.photons) == flux
        print(np.sum(image.photons.flux))
        # Note: tolerance is quite loose, since profiles that use InterpolatedImage can have
        # negative flux photons, which then don't necessarily sum to the right value.
        # Only the expectation value is right, and we're not shooting many photons here.
        assert np.allclose(np.sum(image.photons.flux), flux, rtol=0.1)

        # Sometimes there is a different path when n_photons is not given, so check that too.
        image = obj.drawImage(bandpass=bandpass, method="phot",
                              save_photons=True,
                              scale=0.05, nx=32, ny=32, rng=rng)
        assert hasattr(image, 'photons')
        print(np.sum(image.photons.flux))
        assert np.allclose(np.sum(image.photons.flux), flux, rtol=0.1)
        repr(obj)


if __name__ == "__main__":
    runtests(__file__)
