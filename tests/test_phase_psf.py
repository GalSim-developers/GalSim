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
import os
import numpy as np
from galsim_test_helpers import *

try:
    import galsim

except ImportError:
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


imgdir = os.path.join(".", "Optics_comparison_images") # Directory containing the reference images.
pp_file = 'sample_pupil_rolled.fits'


@timer
def test_aperture():
    # Simple tests for constructing and pickling Apertures.
    aper1 = galsim.Aperture(diam=1.0)
    im = galsim.fits.read(os.path.join(imgdir, pp_file))
    aper2 = galsim.Aperture(diam=1.0, pupil_plane_im=im)
    do_pickle(aper1)
    do_pickle(aper2)
    # Automatically created Aperture should match one created via OpticalScreen
    aper1 = galsim.Aperture(diam=1.0)
    aper2 = galsim.Aperture(diam=1.0, lam=500, screen_list=[galsim.OpticalScreen()])
    err_str = ("Aperture created implicitly using Airy does not match Aperture created using "
               "OpticalScreen.")
    assert aper1 == aper2, err_str


@timer
def test_phase_screen_list():
    # Check list-like behaviors of PhaseScreenList
    rng = galsim.BaseDeviate(1234)
    rng2 = galsim.BaseDeviate(123)

    aper = galsim.Aperture(diam=1.0)

    ar1 = galsim.AtmosphericScreen(10, 1, alpha=0.997, L0=None, rng=rng)
    do_pickle(ar1)
    do_pickle(ar1, func=lambda x: x.tab2d(12.3, 45.6))
    do_pickle(ar1, func=lambda x: x.wavefront(aper).sum())

    # Check that L0=np.inf and L0=None yield the same thing here too.
    ar2 = galsim.AtmosphericScreen(10, 1, alpha=0.997, L0=np.inf, rng=rng)
    assert ar1 == ar2
    # Create a couple new screens with different types/parameters
    ar2 = galsim.AtmosphericScreen(10, 1, alpha=0.995, rng=rng2)
    assert ar1 != ar2
    ar3 = galsim.OpticalScreen(aberrations=[0, 0, 0, 0, 0, 0, 0, 0, 0.1])
    do_pickle(ar3)
    do_pickle(ar3, func=lambda x:x.wavefront(aper).sum())
    atm = galsim.Atmosphere(screen_size=30.0,
                            altitude=[0.0, 1.0],
                            speed=[1.0, 2.0],
                            direction=[0.0*galsim.degrees, 120*galsim.degrees],
                            r0_500=0.15,
                            rng=rng)
    atm.append(ar3)
    do_pickle(atm)
    do_pickle(atm, func=lambda x:x.wavefront(aper).sum())

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
    assert atm == atm2

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

    wf = atm.wavefront(aper)
    wf2 = atm2.wavefront(aper)
    wf3 = atm3.wavefront(aper)
    wf4 = atm4.wavefront(aper)

    np.testing.assert_array_equal(wf, wf2, "PhaseScreenLists are inconsistent")
    np.testing.assert_array_equal(wf, wf3, "PhaseScreenLists are inconsistent")
    np.testing.assert_array_equal(wf, wf4, "PhaseScreenLists are inconsistent")

    # Check copy
    import copy
    # Shallow copy copies by reference.
    atm5 = copy.copy(atm)
    assert atm[0] == atm5[0]
    assert atm[0] is atm5[0]
    atm.advance()
    assert atm[0] == atm5[0]
    assert atm[0] is atm5[0]
    # Deepcopy actually makes an indepedent object in memory.
    atm5 = copy.deepcopy(atm)
    assert atm[0] == atm5[0]
    assert atm[0] is not atm5[0]
    atm.advance()
    assert atm[0] != atm5[0]

    # Constructor should accept both list and indiv layers as arguments.
    atm6 = galsim.PhaseScreenList(atm[0])
    atm7 = galsim.PhaseScreenList([atm[0]])
    assert atm6 == atm7
    atm6 = galsim.PhaseScreenList(atm[0], atm[1])
    atm7 = galsim.PhaseScreenList([atm[0], atm[1]])
    atm8 = galsim.PhaseScreenList(atm[0:2])  # Slice returns PhaseScreenList, so this works too.
    assert atm6 == atm7
    assert atm6 == atm8

    # Check some actual derived PSFs too, not just phase screens.  Use a small pupil_plane_size and
    # relatively large pupil_plane_scale to speed up the unit test.
    atm.advance_by(1.0)
    do_pickle(atm)
    atm.reset()
    kwargs = dict(exptime=0.06, diam=1.0, lam=1000.0)
    psf = atm.makePSF(**kwargs)
    do_pickle(psf)
    do_pickle(psf, func=lambda x:x.drawImage(nx=20, ny=20, scale=0.1))

    # Need to reset atm2 since both atm and atm2 reference the same layer objects (not copies).
    # Not sure if this is a feature or a bug, but it's also how regular python lists work.
    atm2.reset()
    psf2 = atm2.makePSF(**kwargs)

    atm3.reset()
    psf3 = atm3.makePSF(**kwargs)

    atm4.reset()
    psf4 = atm4.makePSF(**kwargs)

    np.testing.assert_array_equal(psf, psf2, "PhaseScreenPSFs are inconsistent")
    np.testing.assert_array_equal(psf, psf3, "PhaseScreenPSFs are inconsistent")
    np.testing.assert_array_equal(psf, psf4, "PhaseScreenPSFs are inconsistent")


@timer
def test_frozen_flow():
    # Check frozen flow: phase(x=0, t=0) == phase(x=v*t, t=t)
    rng = galsim.BaseDeviate(1234)
    vx = 1.0  # m/s
    dt = 0.01  # s
    t = 0.05  # s
    x = vx*t  # 0.05 m
    dx = x
    alt = x/1000   # -> 0.00005 km; silly example, but yields exact results...

    screen = galsim.AtmosphericScreen(1.0, dx, alt, vx=vx, time_step=dt, rng=rng)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aper = galsim.Aperture(diam=1, pupil_plane_size=20., pupil_plane_scale=20./dx)
    wf0 = screen.wavefront(aper)
    screen.advance_by(t)
    wf1 = screen.wavefront(aper, theta=(45*galsim.degrees, 0*galsim.degrees))

    np.testing.assert_array_almost_equal(wf0, wf1, 5, "Flow is not frozen")


@timer
def test_phase_psf_reset():
    rng = galsim.BaseDeviate(1234)
    # Test frozen AtmosphericScreen first
    atm = galsim.Atmosphere(screen_size=30.0, altitude=10.0, speed=0.1, alpha=1.0, rng=rng)
    aper = galsim.Aperture(diam=1.0, lam=500.0)
    wf1 = atm.wavefront(aper)
    atm.advance()
    wf2 = atm.wavefront(aper)
    # Verify that atmosphere did advance
    assert not np.all(wf1 == wf2)

    # Now verify that reset brings back original atmosphere
    atm.reset()
    wf3 = atm.wavefront(aper)
    np.testing.assert_array_equal(wf1, wf3, "Phase screen didn't reset")

    # Now check with boilin, but no wind.
    atm = galsim.Atmosphere(screen_size=30.0, altitude=10.0, alpha=0.997, rng=rng)
    wf1 = atm.wavefront(aper)
    atm.advance()
    wf2 = atm.wavefront(aper)
    # Verify that atmosphere did advance
    assert not np.all(wf1 == wf2)

    # Now verify that reset brings back original atmosphere
    atm.reset()
    wf3 = atm.wavefront(aper)
    np.testing.assert_array_equal(wf1, wf3, "Phase screen didn't reset")


@timer
def test_phase_psf_batch():
    # Check that PSFs generated serially match those generated in batch.
    import time
    NPSFs = 10
    exptime = 0.06
    rng = galsim.BaseDeviate(1234)
    atm = galsim.Atmosphere(screen_size=10.0, altitude=10.0, alpha=0.997, rng=rng)
    theta = [(i*galsim.arcsec, i*galsim.arcsec) for i in xrange(NPSFs)]

    kwargs = dict(lam=1000.0, exptime=exptime, diam=1.0)

    t1 = time.time()
    psfs = atm.makePSF(theta=theta, **kwargs)
    print 'time for {0} PSFs in batch: {1:.2f} s'.format(NPSFs, time.time() - t1)

    t2 = time.time()
    more_psfs = []
    for th in theta:
        atm.reset()
        more_psfs.append(atm.makePSF(theta=th, **kwargs))
    print 'time for {0} PSFs in serial: {1:.2f} s'.format(NPSFs, time.time() - t2)

    for psf1, psf2 in zip(psfs, more_psfs):
        np.testing.assert_array_equal(
            psf1.img, psf2.img,
            "Individually generated AtmosphericPSF differs from AtmosphericPSF generated in batch")


@timer
def test_opt_indiv_aberrations():
    screen1 = galsim.OpticalScreen(tip=0.2, tilt=0.3, defocus=0.4, astig1=0.5, astig2=0.6,
                                   coma1=0.7, coma2=0.8, trefoil1=0.9, trefoil2=1.0, spher=1.1)
    screen2 = galsim.OpticalScreen(aberrations=[0.0, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                1.0, 1.1])

    psf1 = galsim.PhaseScreenList(screen1).makePSF(diam=4.0, lam=500.0)
    psf2 = galsim.PhaseScreenList(screen2).makePSF(diam=4.0, lam=500.0)

    np.testing.assert_array_equal(
            psf1.img, psf2.img,
            "Individually specified aberrations differs from aberrations specified as list.")


@timer
def test_scale_unit():
    aper = galsim.Aperture(diam=1.0)
    rng = galsim.BaseDeviate(1234)
    # Test frozen AtmosphericScreen first
    atm = galsim.Atmosphere(screen_size=30.0, altitude=10.0, speed=0.1, alpha=1.0, rng=rng)
    psf = galsim.PhaseScreenPSF(atm, 500.0, aper=aper, scale_unit=galsim.arcsec)
    im1 = psf.drawImage(nx=32, ny=32, scale=0.1, method='no_pixel')
    atm.reset()
    psf2 = galsim.PhaseScreenPSF(atm, 500.0, aper=aper, scale_unit=galsim.arcmin)
    im2 = psf2.drawImage(nx=32, ny=32, scale=0.1/60.0, method='no_pixel')
    np.testing.assert_array_equal(
            im1.array, im2.array,
            'PhaseScreenPSF inconsistent use of scale_unit')


@timer
def test_ne():
    import copy
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
            galsim.AtmosphericScreen(10.0, rng=rng, vx=1.0),  # advance this one below
            galsim.AtmosphericScreen(10.0, rng=rng, vy=1.0),
            galsim.AtmosphericScreen(10.0, rng=rng, alpha=0.999),
            galsim.AtmosphericScreen(10.0, rng=rng, altitude=1.0),
            galsim.AtmosphericScreen(10.0, rng=rng, time_step=0.1),
            galsim.AtmosphericScreen(10.0, rng=rng, r0_500=0.1),
            galsim.AtmosphericScreen(10.0, rng=rng, L0=10.0),
            galsim.AtmosphericScreen(10.0, rng=rng, vx=10.0),
            ]
    objs[2].advance()
    all_obj_diff(objs)

    # Test OpticalScreen __ne__
    objs = [galsim.OpticalScreen(),
            galsim.OpticalScreen(tip=1.0),
            galsim.OpticalScreen(tilt=1.0),
            galsim.OpticalScreen(defocus=1.0),
            galsim.OpticalScreen(astig1=1.0),
            galsim.OpticalScreen(astig2=1.0),
            galsim.OpticalScreen(coma1=1.0),
            galsim.OpticalScreen(coma2=1.0),
            galsim.OpticalScreen(trefoil1=1.0),
            galsim.OpticalScreen(trefoil2=1.0),
            galsim.OpticalScreen(spher=1.0),
            galsim.OpticalScreen(spher=1.0, lam_0=100.0),
            galsim.OpticalScreen(aberrations=[0,0,1.1]), # tip=1.1
            ]
    all_obj_diff(objs)

    # Test PhaseScreenList __ne__
    atm = galsim.Atmosphere(10.0, vx=1.0)
    objs = [galsim.PhaseScreenList(atm),
            galsim.PhaseScreenList(copy.deepcopy(atm)),  # advance down below
            galsim.PhaseScreenList(objs),  # Reuse list of OpticalScreens above
            galsim.PhaseScreenList(objs[0:2])]
    objs[1].advance()
    all_obj_diff(objs)

    # Test PhaseScreenPSF __ne__
    objs[0].reset()
    psl = galsim.PhaseScreenList(atm)
    objs = [galsim.PhaseScreenPSF(psl, 500.0, exptime=0.03, diam=1.0),
            galsim.PhaseScreenPSF(psl, 500.0, exptime=0.03, diam=1.0)] # advanced so differs
    psl.reset()
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.0)]
    psl.reset()
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.1)]
    psl.reset()
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.0, flux=1.1)]
    psl.reset()
    objs += [galsim.PhaseScreenPSF(psl, 700.0, exptime=0.03, diam=1.0, interpolant='linear')]
    all_obj_diff(objs)


if __name__ == "__main__":
    test_aperture()
    test_phase_screen_list()
    test_frozen_flow()
    test_phase_psf_reset()
    test_phase_psf_batch()
    test_opt_indiv_aberrations()
    test_scale_unit()
    test_ne()
