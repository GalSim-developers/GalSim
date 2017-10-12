# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
import os
import numpy as np
from galsim_test_helpers import timer, do_shoot, do_pickle, all_obj_diff

try:
    import galsim

except ImportError:
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


imgdir = os.path.join(".", "Optics_comparison_images") # Directory containing the reference images.
pp_file = 'sample_pupil_rolled.fits'

theta0 = (0*galsim.arcmin, 0*galsim.arcmin)

@timer
def test_aperture():
    """Test various ways to construct Apertures."""
    # Simple tests for constructing and pickling Apertures.
    aper1 = galsim.Aperture(diam=1.0)
    im = galsim.fits.read(os.path.join(imgdir, pp_file))
    aper2 = galsim.Aperture(diam=1.0, pupil_plane_im=im)
    do_pickle(aper1)
    do_pickle(aper2)
    # Automatically created Aperture should match one created via OpticalScreen
    aper1 = galsim.Aperture(diam=1.0)
    aper2 = galsim.Aperture(diam=1.0, lam=500, screen_list=[galsim.OpticalScreen(diam=1.0)])
    err_str = ("Aperture created implicitly using Airy does not match Aperture created using "
               "OpticalScreen.")
    assert aper1 == aper2, err_str


@timer
def test_atm_screen_size():
    """Test for consistent AtmosphericScreen size and scale."""
    screen_size = 10.0
    screen_scale = 0.1
    atm = galsim.AtmosphericScreen(screen_size=screen_size, screen_scale=screen_scale)
    # AtmosphericScreen will preserve screen_scale, but will adjust screen_size as necessary to get
    # a good FFT size.
    assert atm.screen_scale == screen_scale
    assert screen_size < atm.screen_size < 1.5*screen_size
    np.testing.assert_equal(atm.screen_size, atm.npix * atm.screen_scale,
                            "Inconsistent atmospheric screen size and scale.")


@timer
def test_structure_function():
    """Test that AtmosphericScreen generates approximately the right structure function for infinite
    outer scale.
    """
    rng = galsim.BaseDeviate(4815162342)
    r0_500 = 0.2
    L0 = None
    screen_scale = 0.05
    screen_size = 100.0

    # Theoretical pure Kolmogorov structure function (at 500 nm!):
    D_kolm = lambda r: 6.8839 * (r/r0_500)**(5./3)

    atm = galsim.AtmosphericScreen(screen_size=screen_size, screen_scale=screen_scale,
                                   r0_500=r0_500, L0=L0, rng=rng)
    phase = atm._tab2d.table.getVals()[:-1, :-1].copy()
    phase *= 2 * np.pi / 500.0  # nm -> radians
    im = galsim.Image(phase, scale=screen_scale)
    D_sim = galsim.utilities.structure_function(im)

    print("r   D_kolm   D_sim")
    for r in [0.5, 2.0, 5.0]:  # Only check values far from the screen size and scale.
        # We're only attempting to verify that we haven't missed a factor of 2 or pi or
        # something like that here, so set the rtol below to be *very* forgiving.  Since the
        # structure function varies quite quickly as r**(5./3), this is still a useful test.
        # For the parameters above (including the random seed), D_kolm(r) and D_sim(r) are actually
        # consistent at about the 15% level in the test below.  It's difficult to predict how
        # consistent they *should* be though, since the simulated structure function estimate is
        # sensitive to resolution and edge effects, as well as the particular realization of the
        # field.
        print(r, D_kolm(r), D_sim(r))
        np.testing.assert_allclose(D_kolm(r), D_sim(r), rtol=0.5,
                                   err_msg="Simulated structure function not close to prediction.")


@timer
def test_phase_screen_list():
    """Test list-like behaviors of PhaseScreenList."""
    rng = galsim.BaseDeviate(1234)
    rng2 = galsim.BaseDeviate(123)

    aper = galsim.Aperture(diam=1.0)

    ar1 = galsim.AtmosphericScreen(10, 1, alpha=0.997, L0=None, time_step=0.01, rng=rng)
    assert ar1._time == 0.0, "AtmosphericScreen initialized with non-zero time."
    do_pickle(ar1)
    do_pickle(ar1, func=lambda x: x._tab2d(12.3, 45.6))
    do_pickle(ar1, func=lambda x: x._wavefront(aper.u, aper.v, None, theta0).sum())
    do_pickle(ar1, func=lambda x: x.wavefront(aper.u, aper.v, 0.0).sum())
    do_pickle(ar1, func=lambda x: np.sum(x.wavefront_gradient(aper.u, aper.v, 0.0)))
    t = np.empty_like(aper.u)
    ud = galsim.UniformDeviate(rng.duplicate())
    ud.generate(t.ravel())
    t *= 0.1  # Only do a few boiling steps
    do_pickle(ar1, func=lambda x: x.wavefront(aper.u, aper.v, t).sum())
    do_pickle(ar1, func=lambda x: np.sum(x.wavefront_gradient(aper.u, aper.v, t)))

    # Try seeking backwards
    assert ar1._time > 0.0
    ar1._seek(0.0)
    # But not before t=0.0
    try:
        np.testing.assert_raises(ValueError, ar1._seek, -1.0)
    except ImportError:
        pass

    # Check that L0=np.inf and L0=None yield the same thing here too.
    ar2 = galsim.AtmosphericScreen(10, 1, alpha=0.997, L0=np.inf, time_step=0.01, rng=rng)
    assert ar1 == ar2
    # Create a couple new screens with different types/parameters
    ar2 = galsim.AtmosphericScreen(10, 1, alpha=0.995, time_step=0.015, rng=rng2)
    assert ar1 != ar2
    ar3 = galsim.OpticalScreen(diam=1.0, aberrations=[0, 0, 0, 0, 0, 0, 0, 0, 0.1])
    do_pickle(ar3)
    do_pickle(ar3, func=lambda x:x._wavefront(aper.u, aper.v, None, theta0).sum())
    do_pickle(ar3, func=lambda x:np.sum(x._wavefront_gradient(aper.u, aper.v, None, theta0)))
    do_pickle(ar3, func=lambda x:x.wavefront(aper.u, aper.v).sum())
    do_pickle(ar3, func=lambda x:np.sum(x.wavefront_gradient(aper.u, aper.v)))
    atm = galsim.Atmosphere(screen_size=30.0,
                            altitude=[0.0, 1.0],
                            speed=[1.0, 2.0],
                            direction=[0.0*galsim.degrees, 120*galsim.degrees],
                            r0_500=0.15,
                            rng=rng)
    atm.append(ar3)
    do_pickle(atm)
    do_pickle(atm, func=lambda x:x._wavefront(aper.u, aper.v, None, theta0).sum())
    do_pickle(atm, func=lambda x:x.wavefront(aper.u, aper.v, 0.0, theta0).sum())
    do_pickle(atm, func=lambda x:np.sum(x.wavefront_gradient(aper.u, aper.v, 0.0)))
    do_pickle(atm, func=lambda x:np.sum(x._wavefront_gradient(aper.u, aper.v, 0.0, theta0)))

    # testing append, extend, __getitem__, __setitem__, __delitem__, __eq__, __ne__
    atm2 = galsim.PhaseScreenList(atm[:-1])  # Refers to first n-1 screens
    assert atm != atm2
    # Append a different screen to the end of atm2
    atm2.append(ar2)
    assert atm != atm2
    # Swap the last screen in atm2 for the one that should match atm.
    del atm2[-1]
    atm2.append(atm[-1])
    assert atm == atm2

    # Test building from empty PhaseScreenList
    atm3 = galsim.PhaseScreenList()
    atm3.extend(atm2)
    assert atm == atm3

    # Test constructing from existing PhaseScreenList
    atm4 = galsim.PhaseScreenList(atm3)
    del atm4[-1]
    assert atm != atm4
    atm4.append(atm[-1])
    assert atm == atm4

    # Test swap
    atm4[0], atm4[1] = atm4[1], atm4[0]
    assert atm != atm4
    atm4[0], atm4[1] = atm4[1], atm4[0]
    assert atm == atm4

    wf = atm._wavefront(aper.u, aper.v, None, theta0)
    wf2 = atm2._wavefront(aper.u, aper.v, None, theta0)
    wf3 = atm3._wavefront(aper.u, aper.v, None, theta0)
    wf4 = atm4._wavefront(aper.u, aper.v, None, theta0)

    np.testing.assert_array_equal(wf, wf2, "PhaseScreenLists are inconsistent")
    np.testing.assert_array_equal(wf, wf3, "PhaseScreenLists are inconsistent")
    np.testing.assert_array_equal(wf, wf4, "PhaseScreenLists are inconsistent")

    # Check copy
    import copy
    # Shallow copy copies by reference.
    atm5 = copy.copy(atm)
    assert atm[0] == atm5[0]
    assert atm[0] is atm5[0]
    atm._seek(1.0)
    assert atm[0]._time == 1.0, "Wrong time for AtmosphericScreen"
    assert atm[0] == atm5[0]
    assert atm[0] is atm5[0]
    # Deepcopy actually makes an indepedent object in memory.
    atm5 = copy.deepcopy(atm)
    assert atm[0] == atm5[0]
    assert atm[0] is not atm5[0]
    atm._seek(2.0)
    assert atm[0]._time == 2.0, "Wrong time for AtmosphericScreen"
    # But we still get equality, since this doesn't depend on mutable internal state:
    assert atm[0] == atm5[0]

    # Constructor should accept both list and indiv layers as arguments.
    atm6 = galsim.PhaseScreenList(atm[0])
    atm7 = galsim.PhaseScreenList([atm[0]])
    assert atm6 == atm7
    do_pickle(atm6, func=lambda x:x._wavefront(aper.u, aper.v, None, theta0).sum())
    do_pickle(atm6, func=lambda x:np.sum(x.wavefront_gradient(aper.u, aper.v, 0.0)))

    atm6 = galsim.PhaseScreenList(atm[0], atm[1])
    atm7 = galsim.PhaseScreenList([atm[0], atm[1]])
    atm8 = galsim.PhaseScreenList(atm[0:2])  # Slice returns PhaseScreenList, so this works too.
    assert atm6 == atm7
    assert atm6 == atm8

    # Check some actual derived PSFs too, not just phase screens.  Use a small pupil_plane_size and
    # relatively large pupil_plane_scale to speed up the unit test.
    atm._reset()
    assert atm[0]._time == 0.0, "Wrong time for AtmosphericScreen"
    kwargs = dict(exptime=0.05, time_step=0.01, diam=1.1, lam=1000.0)
    psf = atm.makePSF(**kwargs)
    do_pickle(psf)
    do_pickle(psf, func=lambda x:x.drawImage(nx=20, ny=20, scale=0.1))

    psf2 = atm2.makePSF(**kwargs)
    psf3 = atm3.makePSF(**kwargs)
    psf4 = atm4.makePSF(**kwargs)

    np.testing.assert_array_equal(psf, psf2, "PhaseScreenPSFs are inconsistent")
    np.testing.assert_array_equal(psf, psf3, "PhaseScreenPSFs are inconsistent")
    np.testing.assert_array_equal(psf, psf4, "PhaseScreenPSFs are inconsistent")


@timer
def test_frozen_flow():
    """Test that frozen flow screen really is frozen, i.e., phase(x=0, t=0) == phase(x=v*t, t=t)."""
    rng = galsim.BaseDeviate(1234)
    vx = 1.0  # m/s
    t = 0.05  # s
    x = vx*t  # 0.05 m
    dx = x
    alt = x/1000   # -> 0.00005 km; silly example, but yields exact results...

    screen = galsim.AtmosphericScreen(1.0, dx, alt, vx=vx, rng=rng)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aper = galsim.Aperture(diam=1, pupil_plane_size=20., pupil_plane_scale=20./dx)
    wf0 = screen._wavefront(aper.u, aper.v, None, theta0)
    dwdu0, dwdv0 = screen.wavefront_gradient(aper.u, aper.v, t=screen._time)
    screen._seek(t)
    assert screen._time == t, "Wrong time for AtmosphericScreen"
    wf1 = screen._wavefront(aper.u, aper.v, None, theta=(45*galsim.degrees, 0*galsim.degrees))
    dwdu1, dwdv1 = screen.wavefront_gradient(aper.u, aper.v, t=screen._time,
                                             theta=(45*galsim.degrees, 0*galsim.degrees))

    np.testing.assert_array_almost_equal(wf0, wf1, 5, "Flow is not frozen")
    np.testing.assert_array_almost_equal(dwdu0, dwdu1, 5, "Flow is not frozen")
    np.testing.assert_array_almost_equal(dwdu0, dwdu1, 5, "Flow is not frozen")

    # We should be able to rewind too.
    screen._seek(0.01)
    np.testing.assert_allclose(screen._time, 0.01, err_msg="Wrong time for AtmosphericScreen")
    wf2 = screen.wavefront(aper.u, aper.v, 0.0)
    np.testing.assert_array_almost_equal(wf0, wf2, 5, "Flow is not frozen")


@timer
def test_phase_psf_reset():
    """Test that phase screen reset() method correctly resets the screen to t=0."""
    rng = galsim.BaseDeviate(1234)
    # Test frozen AtmosphericScreen first
    atm = galsim.Atmosphere(screen_size=30.0, altitude=10.0, speed=0.1, alpha=1.0, rng=rng)
    aper = galsim.Aperture(diam=1.0, lam=500.0)
    wf1 = atm._wavefront(aper.u, aper.v, None, theta0)
    wf2 = atm.wavefront(aper.u, aper.v, 0.0, theta0)
    assert np.all(wf1 == wf2)

    atm._seek(1.0)
    wf3 = atm._wavefront(aper.u, aper.v, None, theta0)
    wf4 = atm.wavefront(aper.u, aper.v, 1.0, theta0)
    assert np.all(wf3 == wf4)

    # Verify that atmosphere did advance
    assert not np.all(wf1 == wf3)

    # Now verify that reset brings back original atmosphere
    atm._reset()
    wf3 = atm._wavefront(aper.u, aper.v, None, theta0)
    np.testing.assert_array_equal(wf1, wf3, "Phase screen didn't reset")

    # Now check with boiling, but no wind.
    atm = galsim.Atmosphere(screen_size=30.0, altitude=10.0, alpha=0.997, time_step=0.01, rng=rng)
    wf1 = atm._wavefront(aper.u, aper.v, None, theta0)
    atm._seek(0.1)
    wf2 = atm._wavefront(aper.u, aper.v, None, theta0)
    # Verify that atmosphere did advance
    assert not np.all(wf1 == wf2)

    # Now verify that reset brings back original atmosphere
    atm._reset()
    wf3 = atm._wavefront(aper.u, aper.v, None, theta0)
    np.testing.assert_array_equal(wf1, wf3, "Phase screen didn't reset")


@timer
def test_phase_psf_batch():
    """Test that PSFs generated and drawn serially match those generated and drawn in batch."""
    import time
    NPSFs = 10
    exptime = 0.3
    rng = galsim.BaseDeviate(1234)
    atm = galsim.Atmosphere(screen_size=10.0, altitude=10.0, alpha=0.997, time_step=0.01, rng=rng)
    theta = [(i*galsim.arcsec, i*galsim.arcsec) for i in range(NPSFs)]

    kwargs = dict(lam=1000.0, exptime=exptime, diam=1.0)

    t1 = time.time()
    psfs = [atm.makePSF(theta=th, **kwargs) for th in theta]
    imgs = [psf.drawImage() for psf in psfs]
    print('time for {0} PSFs in batch: {1:.2f} s'.format(NPSFs, time.time() - t1))

    t2 = time.time()
    more_imgs = []
    for th in theta:
        psf = atm.makePSF(theta=th, **kwargs)
        more_imgs.append(psf.drawImage())
    print('time for {0} PSFs in serial: {1:.2f} s'.format(NPSFs, time.time() - t2))

    for img1, img2 in zip(imgs, more_imgs):
        np.testing.assert_array_equal(
            img1, img2,
            "Individually generated AtmosphericPSF differs from AtmosphericPSF generated in batch")


@timer
def test_opt_indiv_aberrations():
    """Test that aberrations specified by name match those specified in `aberrations` list."""
    screen1 = galsim.OpticalScreen(diam=4.0, tip=0.2, tilt=0.3, defocus=0.4, astig1=0.5, astig2=0.6,
                                   coma1=0.7, coma2=0.8, trefoil1=0.9, trefoil2=1.0, spher=1.1)
    screen2 = galsim.OpticalScreen(diam=4.0, aberrations=[0.0, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                                          0.8, 0.9, 1.0, 1.1])

    psf1 = galsim.PhaseScreenList(screen1).makePSF(diam=4.0, lam=500.0)
    psf2 = galsim.PhaseScreenList(screen2).makePSF(diam=4.0, lam=500.0)

    np.testing.assert_array_equal(
            psf1.img, psf2.img,
            "Individually specified aberrations differs from aberrations specified as list.")


@timer
def test_scale_unit():
    """Test that `scale_unit` keyword correctly sets the units for PhaseScreenPSF."""
    aper = galsim.Aperture(diam=1.0)
    rng = galsim.BaseDeviate(1234)
    # Test frozen AtmosphericScreen first
    atm = galsim.Atmosphere(screen_size=30.0, altitude=10.0, speed=0.1, alpha=1.0, rng=rng)
    psf = galsim.PhaseScreenPSF(atm, 500.0, aper=aper, scale_unit=galsim.arcsec)
    im1 = psf.drawImage(nx=32, ny=32, scale=0.1, method='no_pixel')
    psf2 = galsim.PhaseScreenPSF(atm, 500.0, aper=aper, scale_unit=galsim.arcmin)
    im2 = psf2.drawImage(nx=32, ny=32, scale=0.1/60.0, method='no_pixel')
    np.testing.assert_almost_equal(
            im1.array, im2.array, 8,
            'PhaseScreenPSF inconsistent use of scale_unit')

    opt_psf1 = galsim.OpticalPSF(lam=500.0, diam=1.0, scale_unit=galsim.arcsec)
    opt_psf2 = galsim.OpticalPSF(lam=500.0, diam=1.0, scale_unit='arcsec')
    assert opt_psf1 == opt_psf2, "scale unit did not parse as string"


@timer
def test_stepk_maxk():
    """Test options to specify (or not) stepk and maxk.
    """
    aper = galsim.Aperture(diam=1.0)
    rng = galsim.BaseDeviate(123456)
    # Test frozen AtmosphericScreen first
    atm = galsim.Atmosphere(screen_size=30.0, altitude=10.0, speed=0.1, alpha=1.0, rng=rng)
    psf = galsim.PhaseScreenPSF(atm, 500.0, aper=aper, scale_unit=galsim.arcsec)
    stepk = psf.stepk
    maxk = psf.maxk

    psf2 = galsim.PhaseScreenPSF(atm, 500.0, aper=aper, scale_unit=galsim.arcsec,
                                 _force_stepk=stepk/1.5, _force_maxk=maxk*2.0)
    np.testing.assert_almost_equal(
            psf2.stepk, stepk/1.5, decimal=7,
            err_msg="PhaseScreenPSF did not adopt forced value for stepk")
    np.testing.assert_almost_equal(
            psf2.maxk, maxk*2.0, decimal=7,
            err_msg="PhaseScreenPSF did not adopt forced value for maxk")
    do_pickle(psf)
    do_pickle(psf2)

    # Try out non-geometric-shooting
    psf3 = atm.makePSF(lam=500.0, aper=aper, geometric_shooting=False)
    img = galsim.Image(32, 32, scale=0.2)
    do_shoot(psf3, img, "PhaseScreenPSF")
    # Also make sure a few other methods at least run
    psf3.centroid
    psf3.max_sb


@timer
def test_ne():
    """Test Apertures, PhaseScreens, PhaseScreenLists, and PhaseScreenPSFs for not-equals."""
    pupil_plane_im = galsim.fits.read(os.path.join(imgdir, pp_file))

    # Test galsim.Aperture __ne__
    objs = [galsim.Aperture(diam=1.0),
            galsim.Aperture(diam=1.1),
            galsim.Aperture(diam=1.0, oversampling=1.5),
            galsim.Aperture(diam=1.0, pad_factor=1.5),
            galsim.Aperture(diam=1.0, circular_pupil=False),
            galsim.Aperture(diam=1.0, obscuration=0.3),
            galsim.Aperture(diam=1.0, nstruts=3),
            galsim.Aperture(diam=1.0, nstruts=3, strut_thick=0.2),
            galsim.Aperture(diam=1.0, nstruts=3, strut_angle=15*galsim.degrees),
            galsim.Aperture(diam=1.0, pupil_plane_im=pupil_plane_im),
            galsim.Aperture(diam=1.0, pupil_plane_im=pupil_plane_im,
                            pupil_angle=10.0*galsim.degrees)]
    all_obj_diff(objs)

    # Test AtmosphericScreen __ne__
    rng = galsim.BaseDeviate(1)
    objs = [galsim.AtmosphericScreen(10.0, rng=rng),
            galsim.AtmosphericScreen(10.0, rng=rng, vx=1.0),
            galsim.AtmosphericScreen(10.0, rng=rng, vy=1.0),
            galsim.AtmosphericScreen(10.0, rng=rng, alpha=0.999, time_step=0.01),
            galsim.AtmosphericScreen(10.0, rng=rng, altitude=1.0),
            galsim.AtmosphericScreen(10.0, rng=rng, alpha=0.999, time_step=0.02),
            galsim.AtmosphericScreen(10.0, rng=rng, alpha=0.998, time_step=0.02),
            galsim.AtmosphericScreen(10.0, rng=rng, r0_500=0.1),
            galsim.AtmosphericScreen(10.0, rng=rng, L0=10.0),
            galsim.AtmosphericScreen(10.0, rng=rng, vx=10.0),
            ]
    all_obj_diff(objs)

    # Test OpticalScreen __ne__
    objs = [galsim.OpticalScreen(diam=1.0, ),
            galsim.OpticalScreen(diam=1.0, tip=1.0),
            galsim.OpticalScreen(diam=1.0, tilt=1.0),
            galsim.OpticalScreen(diam=1.0, defocus=1.0),
            galsim.OpticalScreen(diam=1.0, astig1=1.0),
            galsim.OpticalScreen(diam=1.0, astig2=1.0),
            galsim.OpticalScreen(diam=1.0, coma1=1.0),
            galsim.OpticalScreen(diam=1.0, coma2=1.0),
            galsim.OpticalScreen(diam=1.0, trefoil1=1.0),
            galsim.OpticalScreen(diam=1.0, trefoil2=1.0),
            galsim.OpticalScreen(diam=1.0, spher=1.0),
            galsim.OpticalScreen(diam=1.0, spher=1.0, lam_0=100.0),
            galsim.OpticalScreen(diam=1.0, aberrations=[0,0,1.1]), # tip=1.1
            ]
    all_obj_diff(objs)

    # Test PhaseScreenList __ne__
    atm = galsim.Atmosphere(10.0, vx=1.0)
    objs = [galsim.PhaseScreenList(atm),
            galsim.PhaseScreenList(objs),  # Reuse list of OpticalScreens above
            galsim.PhaseScreenList(objs[0:2])]
    all_obj_diff(objs)

    # Test PhaseScreenPSF __ne__
    psl = galsim.PhaseScreenList(atm)
    objs = [galsim.PhaseScreenPSF(psl, 500.0, exptime=0.03, diam=1.0)]
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.0)]
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.1)]
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.0, flux=1.1)]
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.0, interpolant='linear')]
    stepk = objs[0].stepk
    maxk = objs[0].maxk
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.0, _force_stepk=stepk/1.5)]
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.0, _force_maxk=maxk*2.0)]
    all_obj_diff(objs)


@timer
def test_phase_gradient_shoot():
    # Make the atmosphere
    seed = 12345
    r0_500 = 0.2  # m
    nlayers = 6
    screen_size = 102.4  # m
    screen_scale = 0.1  # m
    max_speed = 20  # m/s

    rng = galsim.BaseDeviate(seed)
    u = galsim.UniformDeviate(rng)

    # Use atmospheric weights from 1998 Gemini site selection process as something reasonably
    # realistic.  (Ellerbroek 2002, JOSA Vol 19 No 9).
    Ellerbroek_alts = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
    Ellerbroek_weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    Ellerbroek_interp = galsim.LookupTable(
            Ellerbroek_alts,
            Ellerbroek_weights,
            interpolant='linear')
    alts = np.max(Ellerbroek_alts)*np.arange(nlayers)/(nlayers-1)
    weights = Ellerbroek_interp(alts)
    weights /= sum(weights)

    spd = []  # Wind speed in m/s
    dirn = [] # Wind direction in radians
    r0_500s = [] # Fried parameter in m at a wavelength of 500 nm.
    for i in range(nlayers):
        spd.append(u()*max_speed)
        dirn.append(u()*360*galsim.degrees)
        r0_500s.append(r0_500*weights[i]**(-3./5))
    atm = galsim.Atmosphere(r0_500=r0_500, speed=spd, direction=dirn, altitude=alts, rng=rng,
                            screen_size=screen_size, screen_scale=screen_scale)

    lam = 500.0
    diam = 4.0
    pad_factor = 1.0
    oversampling = 1.0

    aper = galsim.Aperture(diam=diam, lam=lam,
                           screen_list=atm, pad_factor=pad_factor,
                           oversampling=oversampling)

    xs = np.empty((10,), dtype=float)
    ys = np.empty((10,), dtype=float)
    u.generate(xs)
    u.generate(ys)
    thetas = [(x*galsim.degrees, y*galsim.degrees) for x, y in zip(xs, ys)]

    if __name__ == '__main__':
        exptime = 15.0
        centroid_tolerance = 0.05
        second_moment_tolerance = 0.5
    else:
        exptime = 0.2
        centroid_tolerance = 0.2
        second_moment_tolerance = 1.5

    psfs = [atm.makePSF(lam, diam=diam, theta=th, exptime=exptime, aper=aper) for th in thetas]
    shoot_moments = []
    fft_moments = []

    # At the moment, Ixx and Iyy (but not Ixy) are systematically smaller in phase gradient shooting
    # mode than in FFT mode.  For now, I'm willing to accept this, but we should revisit it once we
    # get the "second kick" approximation implemented.
    offset = 0.5

    for psf in psfs:
        im_shoot = psf.drawImage(nx=48, ny=48, scale=0.05, method='phot', n_photons=100000, rng=rng)
        im_fft = psf.drawImage(nx=48, ny=48, scale=0.05)

        shoot_moment = galsim.utilities.unweighted_moments(im_shoot)
        fft_moment = galsim.utilities.unweighted_moments(im_fft)

        for key in ['Mx', 'My']:
            np.testing.assert_allclose(
                    shoot_moment[key], fft_moment[key], rtol=0, atol=centroid_tolerance,
                    err_msg='Phase gradient centroid {0} not close to fft centroid'.format(key))

        for key in ['Mxx', 'Myy']:
            np.testing.assert_allclose(
                    shoot_moment[key]+offset, fft_moment[key], rtol=0, atol=second_moment_tolerance,
                    err_msg='Phase gradient second moment {} not close to fft moment'.format(key))

        np.testing.assert_allclose(
            shoot_moment['Mxy'], fft_moment['Mxy'], rtol=0, atol=second_moment_tolerance,
            err_msg='Phase gradient second moment Mxy not close to fft moment')

        shoot_moments.append(shoot_moment)
        fft_moments.append(fft_moment)

    # Verify that shoot with rng=None runs
    psf.shoot(100, rng=None)

    # Constraints on the ensemble should be tighter than for individual PSFs.
    mean_shoot_moment = {}
    mean_fft_moment = {}
    for k in shoot_moments[0]:
        mean_shoot_moment[k] = np.mean([sm[k] for sm in shoot_moments])
        mean_fft_moment[k] = np.mean([fm[k] for fm in fft_moments])

    for key in ['Mx', 'My']:
        np.testing.assert_allclose(
                mean_shoot_moment[key], mean_fft_moment[key], rtol=0, atol=centroid_tolerance,
                err_msg='Mean phase gradient centroid {0} not close to mean fft centroid'
                        .format(key))

    for key in ['Mxx', 'Myy']:
        np.testing.assert_allclose(
                mean_shoot_moment[key]+offset, mean_fft_moment[key], rtol=0,
                atol=second_moment_tolerance,
                err_msg='Mean phase gradient second moment {} not close to mean fft moment'
                .format(key))

    np.testing.assert_allclose(
        mean_shoot_moment['Mxy'], mean_fft_moment['Mxy'], rtol=0, atol=second_moment_tolerance,
        err_msg='Mean phase gradient second moment Mxy not close to mean fft moment')


@timer
def test_input():
    """Check that exceptions are raised for invalid input"""

    # Specifying only one of alpha and time_step is an error.
    try:
        np.testing.assert_raises(ValueError, galsim.AtmosphericScreen,
                                 screen_size=10.0, time_step=0.01)
        np.testing.assert_raises(ValueError, galsim.AtmosphericScreen,
                                 screen_size=10.0, alpha=0.997)
    except ImportError:
        print('The assert_raises tests require nose')
    # But specifying both is alright.
    galsim.AtmosphericScreen(screen_size=10.0, alpha=0.997, time_step=0.01)

    # Try some variations for Atmosphere
    try:
        np.testing.assert_raises(ValueError, galsim.Atmosphere,
                                 screen_size=10.0, altitude=[0., 1.],
                                 r0_500=[0.2, 0.3, 0.2])
        np.testing.assert_raises(ValueError, galsim.Atmosphere,
                                 screen_size=10.0, r0_500=[0.4, 0.4, 0.4],
                                 r0_weights=[0.1, 0.3, 0.6])
    except ImportError:
        print('The assert_raises tests require nose')


@timer
def test_r0_weights():
    """Check that r0_weights functions as expected."""
    r0_500 = 0.2

    # Check that reassembled net r0_500 matches input
    atm = galsim.Atmosphere(screen_size=10.0, altitude=[0,1,2,3], r0_500=r0_500)
    r0s = [screen.r0_500 for screen in atm]
    np.testing.assert_almost_equal(np.sum([r0**(-5./3) for r0 in r0s])**(-3./5), r0_500)

    # Check that old manual calculation matches automatic calculation inside Atmosphere()
    weights = [1, 2, 3, 4]
    normalized_weights = np.array(weights, dtype=float)/np.sum(weights)
    r0s_ref = [r0_500 * w**(-3./5) for w in normalized_weights]
    atm = galsim.Atmosphere(screen_size=10.0, altitude=[0,1,2,3], r0_500=r0_500, r0_weights=weights)
    r0s_test = [screen.r0_500 for screen in atm]
    np.testing.assert_almost_equal(r0s_test, r0s_ref)
    np.testing.assert_almost_equal(np.sum([r0**(-5./3) for r0 in r0s_test])**(-3./5), r0_500)


@timer
def test_speedup():
    """Make sure that photon-shooting a PhaseScreenPSF with geometric approximation yields
    significant speedup.
    """
    import time
    atm = galsim.Atmosphere(screen_size=10.0, altitude=[0,1,2,3], r0_500=0.2)
    # Should be ~seconds if _prepareDraw() gets executed, ~0.01s otherwise.
    psf = atm.makePSF(lam=500.0, diam=1.0, exptime=15.0, time_step=0.025)
    t0 = time.time()
    psf.drawImage(method='phot', n_photons=1e3)
    t1 = time.time()
    assert (t1-t0) < 0.1, "Photon-shooting took too long ({0} s).".format(t1-t0)


if __name__ == "__main__":
    test_aperture()
    test_atm_screen_size()
    test_structure_function()
    test_phase_screen_list()
    test_frozen_flow()
    test_phase_psf_reset()
    test_phase_psf_batch()
    test_opt_indiv_aberrations()
    test_scale_unit()
    test_stepk_maxk()
    test_ne()
    test_phase_gradient_shoot()
    test_input()
    test_r0_weights()
    test_speedup()
