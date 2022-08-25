# Copyright (c) 2012-2022 by the GalSim developers team on GitHub
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
import sys
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
    do_pickle(rw)
    do_pickle(conv1)
    do_pickle(conv1, lambda x: x.drawImage(scale=1))


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
def test_roman_psfs():
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
    if __name__ == "__main__":
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


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
