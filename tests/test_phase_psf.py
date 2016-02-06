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
from galsim_test_helpers import funcname

try:
    import galsim

except ImportError:
    import os
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


def test_phase_screen_list():
    # Check list-like behaviors of PhaseScreenList

    import time
    t1 = time.time()
    rng = galsim.BaseDeviate(1234)
    rng2 = galsim.BaseDeviate(123)

    # Check that L0=np.inf and L0=None yield the same thing here too.
    ar1 = galsim.AtmosphericScreen(10, 1, alpha=0.997, L0=None, rng=rng)
    ar2 = galsim.AtmosphericScreen(10, 1, alpha=0.997, L0=np.inf, rng=rng)
    assert ar1 == ar2
    ar2 = galsim.AtmosphericScreen(10, 1, alpha=0.995, rng=rng2)
    assert ar1 != ar2

    atm = galsim.Atmosphere(altitude=[0.0, 1.0],
                            velocity=[1.0, 2.0],
                            direction=[0.0*galsim.degrees, 120*galsim.degrees],
                            r0_500=0.15,
                            rng=rng)
    atm.append(ar1)

    # testing append, extend, __getitem__, __setitem__, __delitem__, __eq__, __ne__
    atm2 = galsim.PhaseScreenList(atm[:-1])
    assert atm != atm2
    atm2.append(ar2)
    assert atm != atm2
    del atm2[-1]
    atm2.append(atm[-1])
    assert atm == atm2

    atm3 = galsim.PhaseScreenList([])
    atm3.extend(atm2)
    atm3[1] = atm2[1]
    assert atm == atm2

    atm4 = galsim.PhaseScreenList(atm3)
    del atm4[-1]
    atm4.append(atm[-1])
    assert atm == atm4

    aper = galsim.Aperture(20, 2000)
    pd = atm.path_difference(aper)
    pd2 = atm2.path_difference(aper)
    pd3 = atm3.path_difference(aper)
    pd4 = atm4.path_difference(aper)

    np.testing.assert_array_equal(pd, pd2, "PhaseScreenLists are inconsistent")
    np.testing.assert_array_equal(pd, pd3, "PhaseScreenLists are inconsistent")
    np.testing.assert_array_equal(pd, pd4, "PhaseScreenLists are inconsistent")

    # Check some actual derived PSFs too, not just phase screens.
    atm.reset()
    psf = atm.makePSF(exptime=0.06, diam=4.0)

    # Need to reset atm2 since both atm and atm2 reference the same layer objects (not copies).
    # Not sure if this is a feature or a bug, but it's also how regular python lists work.
    atm2.reset()
    psf2 = atm2.makePSF(exptime=0.06, diam=4.0)

    atm3.reset()
    psf3 = atm3.makePSF(exptime=0.06, diam=4.0)

    atm4.reset()
    psf4 = atm4.makePSF(exptime=0.06, diam=4.0)

    np.testing.assert_array_equal(psf.img, psf2.img, "PhaseScreenPSFs are inconsistent")
    np.testing.assert_array_equal(psf.img, psf3.img, "PhaseScreenPSFs are inconsistent")
    np.testing.assert_array_equal(psf.img, psf4.img, "PhaseScreenPSFs are inconsistent")

    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2-t1)


def test_frozen_flow():
    # Check frozen flow: phase(x=0, t=0) == phase(x=v*t, t=t)

    import time
    t1 = time.time()
    rng = galsim.BaseDeviate(1234)
    vx = 1.0  # m/s
    dt = 0.01  # s
    t = 0.05  # s
    x = vx*t  # 0.05 m
    dx = x
    alt = x/1000   # -> 0.00005 km; silly example, but yields exact results...

    screen = galsim.phase_psf.AtmosphericScreen(1.0, dx, alt, vx=vx, time_step=dt, rng=rng)
    aper = galsim.Aperture(20, 20/dx)
    pd0 = screen.path_difference(aper)
    screen.advance_by(t)
    pd1 = screen.path_difference(aper, theta_x=45*galsim.degrees)

    np.testing.assert_array_almost_equal(pd0, pd1, 5, "Flow is not frozen")

    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2-t1)


def test_phase_psf_reset():
    import time
    t1 = time.time()

    rng = galsim.BaseDeviate(1234)
    # Test frozen AtmosphericScreen first
    atm = galsim.Atmosphere(altitude=10.0, velocity=0.1, alpha=1.0, rng=rng)
    aper = galsim.Aperture(16, 160)
    pd1 = atm.path_difference(aper)
    atm.advance()
    pd2 = atm.path_difference(aper)
    # Verify that atmosphere did advance
    assert not np.all(pd1 == pd2)

    # Now verify that reset brings back original atmosphere
    atm.reset()
    pd3 = atm.path_difference(aper)
    np.testing.assert_array_equal(pd1, pd3, "Phase screen didn't reset")

    # Now check with boilin, but no wind.
    atm = galsim.Atmosphere(altitude=10.0, alpha=0.997, rng=rng)
    pd1 = atm.path_difference(aper)
    atm.advance()
    pd2 = atm.path_difference(aper)
    # Verify that atmosphere did advance
    assert not np.all(pd1 == pd2)

    # Now verify that reset brings back original atmosphere
    atm.reset()
    pd3 = atm.path_difference(aper)
    np.testing.assert_array_equal(pd1, pd3, "Phase screen didn't reset")

    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2-t1)


def test_phase_psf_batch():
    # Check that PSFs generated serially match those generated in batch.
    import time
    t1 = time.time()
    NPSFs = 20
    exptime = 0.09
    rng = galsim.BaseDeviate(1234)
    atm = galsim.Atmosphere(screen_size=10.0, altitude=10.0, alpha=0.997, rng=rng)
    theta_x = [i * galsim.arcsec for i in xrange(NPSFs)]
    theta_y = [i * galsim.arcsec for i in xrange(NPSFs)]

    t3 = time.time()
    psfs = atm.makePSF(theta_x=theta_x, theta_y=theta_y, exptime=exptime, diam=4.0)
    print 'time for {0} PSFs in batch: {1:.2f} s'.format(NPSFs, time.time() - t3)

    t4 = time.time()
    more_psfs = []
    for tx, ty in zip(theta_x, theta_y):
        atm.reset()
        more_psfs.append(atm.makePSF(theta_x=tx, theta_y=ty, exptime=exptime, diam=4.0))
    print 'time for {0} PSFs in serial: {1:.2f} s'.format(NPSFs, time.time() - t4)

    for psf1, psf2 in zip(psfs, more_psfs):
        np.testing.assert_array_equal(
            psf1.img, psf2.img,
            "Individually generated AtmosphericPSF differs from AtmosphericPSF generated in batch")

    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2-t1)


if __name__ == "__main__":
    test_phase_screen_list()
    test_frozen_flow()
    test_phase_psf_reset()
    test_phase_psf_batch()
