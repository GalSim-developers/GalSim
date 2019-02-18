# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

import galsim
from galsim_test_helpers import *


@timer
def test_vk(slow=False):
    """Test the generation of VonKarman profiles
    """
    if slow:
        lams = [300.0, 500.0, 1100.0]
        r0_500s = [0.05, 0.15, 0.3]
        L0s = [1e10, 25.0, 10.0]
        do_deltas = [False, True]
    else:
        lams = [500.0]
        r0_500s = [0.2]
        L0s = [25.0]
        do_deltas = [False]
    for lam in lams:
        for r0_500 in r0_500s:
            r0 = r0_500*(lam/500)**(6./5)
            for L0 in L0s:
                for do_delta in do_deltas:
                    kwargs = {'lam':lam, 'r0':r0, 'L0':L0, 'do_delta':do_delta}
                    print(kwargs)
                    delta_amp = np.exp(-0.5*0.172629*(r0/L0)**(-5./3.))
                    if delta_amp > 1.e-3:
                        print("Skip this combination, since delta > maxk_threshold")
                        continue

                    vk = galsim.VonKarman(flux=2.2, **kwargs)
                    np.testing.assert_almost_equal(vk.flux, 2.2)

                    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
                    vk2 = galsim.VonKarman(flux=2.2, gsparams=gsp, **kwargs)
                    assert vk2 != vk
                    assert vk2 == vk.withGSParams(gsp)

                    check_basic(vk, "VonKarman")
                    do_pickle(vk)

                    img = galsim.Image(16, 16, scale=0.2)
                    if not do_delta:
                        do_shoot(vk, img, "VonKarman")
                        do_kvalue(vk, img, "VonKarman")

    with np.testing.assert_raises(galsim.GalSimIncompatibleValuesError):
        vk = galsim.VonKarman(lam=500, r0=0.1, r0_500=0.2)
    with np.testing.assert_raises(galsim.GalSimIncompatibleValuesError):
        vk = galsim.VonKarman(lam=500)


@timer
def test_vk_delta():
    """Test a VonKarman with a significant delta-function amplitude"""
    kwargs = {'lam':1100.0, 'r0':0.8, 'L0':5.0, 'flux':2.2}
    # Try to see if we can catch the warning first
    with assert_warns(galsim.GalSimWarning):
        vk = galsim.VonKarman(**kwargs)

    kwargs['suppress_warning'] = True
    vk = galsim.VonKarman(**kwargs)
    do_pickle(vk)

    # This profile has more than 15% of its flux in the delta-function component.
    assert vk.delta_amplitude > 0.15 * vk.flux
    # If do_delta is False (the default), then the asymptotic kValue should still be zero.
    np.testing.assert_almost_equal(vk.kValue(1e10, 0).real, 0.0)
    # But if we use do_delta=True, then the asymptotic kValue should be that of the delta function.
    vkd = galsim.VonKarman(do_delta=True, **kwargs)
    do_pickle(vkd)
    np.testing.assert_almost_equal(vkd.kValue(1e10, 0).real, vkd.delta_amplitude)

    # Either way, the fluxes should be the same.
    np.testing.assert_almost_equal(vk.flux, vkd.flux)
    assert vk != vkd
    # The half-light-radius of the profile with do_delta=True should be smaller though, as we're
    # accounting for the 15% flux at r=0 in this case
    assert vkd.half_light_radius < vk.half_light_radius


@timer
def test_vk_scale():
    """Test vk scale argument"""
    kwargs = {'lam':500, 'r0':0.2, 'L0':25.0, 'flux':2.2}
    vk_arcsec = galsim.VonKarman(scale_unit=galsim.arcsec, **kwargs)
    vk_arcmin = galsim.VonKarman(scale_unit='arcmin', **kwargs)
    do_pickle(vk_arcmin)

    np.testing.assert_almost_equal(vk_arcsec.flux, vk_arcmin.flux)
    np.testing.assert_almost_equal(vk_arcsec.kValue(0.0, 0.0), vk_arcmin.kValue(0.0, 0.0))
    np.testing.assert_almost_equal(vk_arcsec.kValue(0.0, 10.0), vk_arcmin.kValue(0.0, 600.0))
    np.testing.assert_almost_equal(vk_arcsec.xValue(0.0, 6.0), vk_arcmin.xValue(0.0, 0.1))

    img1 = vk_arcsec.drawImage(nx=32, ny=32, scale=0.2)
    img2 = vk_arcmin.drawImage(nx=32, ny=32, scale=0.2/60.0)
    np.testing.assert_almost_equal(img1.array, img2.array)


@timer
def test_vk_ne():
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    objs = [galsim.VonKarman(lam=500.0, r0=0.2),
            galsim.VonKarman(lam=500.0, r0=0.2, L0=20.0),
            galsim.VonKarman(lam=500.0, r0=0.2, L0=20.0, flux=2.2),
            galsim.VonKarman(lam=500.0, r0=0.2, L0=1e11),
            galsim.VonKarman(lam=550.0, r0=0.1, L0=20.0),
            galsim.VonKarman(lam=550.0, r0=0.1, L0=20.0, do_delta=True),
            galsim.VonKarman(lam=550.0, r0=0.1, L0=20.0, scale_unit=galsim.arcmin),
            galsim.VonKarman(lam=550.0, r0=0.1, L0=20.0, gsparams=gsp)]
    all_obj_diff(objs)


@timer
def test_vk_eq_kolm():
    lam = 500.0
    r0 = 0.2
    L0 = 1e10  # Need to make this surprisingly large to make vk -> kolm.
    flux = 3.3
    kolm = galsim.Kolmogorov(lam=lam, r0=r0, flux=flux)
    vk = galsim.VonKarman(lam=lam, r0=r0, L0=L0, flux=flux)

    np.testing.assert_allclose(kolm.xValue(0,0), vk.xValue(0,0), rtol=1e-3, atol=0)

    kolm_img = kolm.drawImage(nx=24, ny=24, scale=0.2)
    vk_img = vk.drawImage(nx=24, ny=24, scale=0.2)
    np.testing.assert_allclose(kolm_img.array, vk_img.array, atol=flux*4e-5, rtol=0)


@timer
def test_vk_fitting_formulae():
    #         lam, r0_500, L0
    params = [(650, 0.15, 10.0),
              (450, 0.12, 25.0),
              (900, 0.18, 100.0)]

    def predicted_FWHM_ratio(r0, L0):
        """Fitting formula for VonKarman FWHM / Kolmogorov FWHM
        from Martinez++2014
        """
        return np.sqrt(1 - 2.183*(r0/L0)**0.356)

    def predicted_HLR_ratio(r0, L0):
        """Fitting formula for VonKarman HLR / Kolmogorov HLR
        from Martinez++2014
        """
        return np.sqrt(1 - 1.534*(r0/L0)**0.347)

    for lam, r0_500, L0 in params:
        print(lam, r0_500, L0)
        r0 = r0_500*(lam/500.0)**(6./5)
        kolm = galsim.Kolmogorov(lam=lam, r0=r0)
        vk = galsim.VonKarman(lam=lam, r0=r0, L0=L0)
        vk2 = galsim.VonKarman(lam=lam, r0_500=r0_500, L0=L0)
        np.testing.assert_allclose(vk.r0, vk2.r0)
        np.testing.assert_allclose(vk.r0_500, vk2.r0_500)
        for prof in [vk, vk2]:
            HLR_ratio = prof.calculateHLR() / kolm.calculateHLR()
            FWHM_ratio = prof.calculateFWHM() / kolm.calculateFWHM()
            print(HLR_ratio)
            print(FWHM_ratio)
            np.testing.assert_allclose(HLR_ratio, predicted_HLR_ratio(r0, L0), rtol=0.015)
            np.testing.assert_allclose(FWHM_ratio, predicted_FWHM_ratio(r0, L0), rtol=0.015)


@timer
def test_vk_gsp():
    """Test that we can construct a vK with non-standard folding_threshold.
    """
    # default folding_threshold is 5e-3.
    # We can't go too much smaller than this for such a flat asymptotic profile, but check a little
    # bit further works.
    gsp1 = galsim.GSParams(folding_threshold=1e-2)
    gsp2 = galsim.GSParams(folding_threshold=2e-3)

    # Just testing that these construct successfully
    galsim.VonKarman(lam=700, r0=0.1, L0=24.3, gsparams=gsp1)
    galsim.VonKarman(lam=700, r0=0.1, L0=24.3, gsparams=gsp2)


def vk_benchmark():
    import time
    t0 = time.time()
    vk = galsim.VonKarman(lam=700, r0=0.1, L0=24.3)
    vk.drawImage(nx=16, ny=16, scale=0.2)
    t1 = time.time()
    print("Time to create/draw first time: {:6.3f}s".format(t1-t0))  # ~0.7s
    for i in range(10):
        vk.drawImage(nx=16, ny=16, scale=0.2)
    t2 = time.time()
    print("Time to draw 10 more: {:6.3f}s".format(t2-t1))  # ~0.07s
    for i in range(100):
        vk.drawImage(nx=16, ny=16, scale=0.2, method='phot', n_photons=50000)
    t3 = time.time()
    print("Time to photon-shoot 100 more with 50000 photons each: {:6.3f}s".format(t3-t2))  # ~0.9s


@timer
def test_vk_r0():
    """Test a special r0 value that resulted in an error, reported in issue #957.
    """
    r0 = 0.146068884
    vk = galsim.VonKarman(L0=25.,lam=700.,r0=r0)
    # Note: the resolution of the bug was to add explicit split points for the first several
    # j0 zeros.  Without that, the integral in rawXValue can spuriously fail badly, leading to
    # an invalid estimate of the total integrated flux within R=pi/stepk.
    check_basic(vk, "VonKarman, r0=%s"%r0)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--slow", action='store', default=1, help="Run slow tests")
    parser.add_argument("--benchmark", action='store_true', help="Run timing benchmark")
    parser.add_argument("--profile", action='store_true', help="Profile the tests")
    args = parser.parse_args()

    if args.profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()

    test_vk(args.slow)
    test_vk_delta()
    test_vk_scale()
    test_vk_ne()
    test_vk_eq_kolm()
    test_vk_fitting_formulae()
    test_vk_gsp()
    test_vk_r0()
    if args.benchmark:
        vk_benchmark()

    if args.profile:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)
