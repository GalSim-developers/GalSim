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
import galsim
import time

from galsim_test_helpers import *


@timer
def test_init():
    """Test generation of SecondKick profiles
    """
    obscuration = 0.5

    if __name__ == '__main__':
        lams = [300.0, 500.0, 1100.0]
        r0_500s = [0.1, 0.15, 0.3]
        kcrits = [0.1, 0.2, 0.4]
    else:
        lams = [500.0]
        r0_500s = [0.15]
        kcrits = [0.2]
    for lam in lams:
        for r0_500 in r0_500s:
            r0 = r0_500*(lam/500)**(6./5)
            for kcrit in kcrits:
                t0 = time.time()
                kwargs = {'lam':lam, 'r0':r0, 'kcrit':kcrit, 'diam':4.0}
                print(kwargs)

                sk = galsim.SecondKick(flux=2.2, **kwargs)
                t1 = time.time()
                print('   stepk, maxk = ',sk.stepk, sk.maxk)
                np.testing.assert_almost_equal(sk.flux, 2.2)
                do_pickle(sk)
                t2 = time.time()

                gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
                sk2 = galsim.SecondKick(flux=2.2, gsparams=gsp, **kwargs)
                assert sk2 != sk
                assert sk2 == sk.withGSParams(gsp)

                # Raw sk objects are hard to draw due to a large maxk/stepk ratio.
                # Decrease maxk by convolving in a smallish Gaussian.
                obj = galsim.Convolve(sk, galsim.Gaussian(fwhm=0.2))
                print('   obj stepk, maxk = ',obj.stepk, obj.maxk)
                check_basic(obj, "SecondKick")
                t3 = time.time()
                img = galsim.Image(16, 16, scale=0.2)
                do_shoot(obj, img, "SecondKick")
                t4 = time.time()
                do_kvalue(obj, img, "SecondKick")
                t5 = time.time()
                print(' times = ',t1-t0,t2-t1,t3-t2,t4-t3,t5-t4)


@timer
def test_structure_function():
    """Test that SecondKick structure function is equivalent to vonKarman structure function when
    kcrit=0.  This is nontrivial since the SecondKick structure function is numerically integrated,
    while the vK structure function is evaluated analytically.
    """
    lam = 500.0
    r0 = 0.2
    diam = 8.36
    obscuration = 0.61

    sk = galsim.SecondKick(lam, r0, diam, obscuration, kcrit=0.0)
    vk = galsim.VonKarman(lam, r0, L0=1.e10)

    for rho in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        sksf = sk._structure_function(rho/r0)
        vksf = vk._structure_function(rho)
        print(sksf,vksf)
        np.testing.assert_allclose(sksf, vksf, rtol=2e-3, atol=1.e-3)


@timer
def test_limiting_cases():
    """SecondKick has some two interesting limiting cases.
    A) When kcrit = 0, SecondKick = Convolve(Airy, VonKarman).
    B) When kcrit = inf, SecondKick = Airy
    Test these.
    """
    lam = 500.0
    r0 = 0.2
    diam = 8.36
    obscuration = 0.61

    # First kcrit=0
    sk = galsim.SecondKick(lam, r0, diam, obscuration, kcrit=0.0)
    limiting_case = galsim.Convolve(
        galsim.VonKarman(lam, r0, L0=1.e8),
        galsim.Airy(lam=lam, diam=diam, obscuration=obscuration)
    )
    print(sk.stepk, sk.maxk)
    print(limiting_case.stepk, limiting_case.maxk)

    for k in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 20.0]:
        print(sk.kValue(0, k).real, limiting_case.kValue(0, k).real)
        np.testing.assert_allclose(
            sk.kValue(0, k).real,
            limiting_case.kValue(0, k).real,
            rtol=1e-3,
            atol=1e-4)

    # Normally, one wouldn't use SecondKick.xValue, since it does a real-space convolution,
    # so it's slow.  But we do allow it, so test it here.
    import time
    t0 = time.time()
    xv_2k = sk.xValue(0,0)
    print("xValue(0,0) = ",xv_2k)
    t1 = time.time()
    # The VonKarman * Airy xValue is much slower still, so don't do that.
    # Instead compare it to the 'sb' image.
    xv_image = limiting_case.drawImage(nx=1,ny=1,method='sb',scale=0.1)(1,1)
    print('from image ',xv_image)
    t2 = time.time()
    print('t = ',t1-t0, t2-t1)
    np.testing.assert_almost_equal(xv_2k, xv_image, decimal=3)

    # kcrit=inf
    sk = galsim.SecondKick(lam, r0, diam, obscuration, kcrit=np.inf)
    limiting_case = galsim.Airy(lam=lam, diam=diam, obscuration=obscuration)

    for k in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 20.0]:
        print(sk.kValue(0, k).real, limiting_case.kValue(0, k).real)
        np.testing.assert_allclose(
            sk.kValue(0, k).real,
            limiting_case.kValue(0, k).real,
            rtol=1e-3,
            atol=1e-4)

@timer
def test_sk_phase_psf():
    """Test that analytic second kick profile matches what can be obtained from PhaseScreenPSF with
    an appropriate truncated power spectrum.
    """
    # import matplotlib.pyplot as plt

    lam = 500.0
    r0 = 0.2
    diam = 4.0
    obscuration = 0.5

    rng = galsim.UniformDeviate(1234567890)
    weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    speed = [rng()*20 for _ in range(6)]
    direction = [rng()*360*galsim.degrees for _ in range(6)]
    aper = galsim.Aperture(4.0, obscuration=obscuration, pad_factor=0.5)
    kcrits = [1, 3, 10] if __name__ == '__main__' else [1]
    for kcrit in kcrits:
        # Technically, we should probably use a smaller screen_scale here, but that runs really
        # slowly.  The below seems to work well enough for the tested kcrits.
        atm = galsim.Atmosphere(r0_500=r0, r0_weights=weights, rng=rng,
                                speed=speed, direction=direction,
                                screen_size=102.4, screen_scale=0.05,
                                suppress_warning=True)
        atm.instantiate(kmin=kcrit)
        print('instantiated atm')
        psf = galsim.PhaseScreenPSF(atm, lam=lam, exptime=10, aper=aper, time_step=0.1)
        print('made psf')
        phaseImg = psf.drawImage(nx=64, ny=64, scale=0.02)
        sk = galsim.SecondKick(lam=lam, r0=r0, diam=diam, obscuration=obscuration,
                               kcrit=kcrit)
        print('made sk')
        skImg = sk.drawImage(nx=64, ny=64, scale=0.02)
        print('made skimg')
        phaseMom = galsim.hsm.FindAdaptiveMom(phaseImg)
        skMom = galsim.hsm.FindAdaptiveMom(skImg)

        print('moments: ',phaseMom.moments_sigma, skMom.moments_sigma)
        np.testing.assert_allclose(phaseMom.moments_sigma, skMom.moments_sigma, rtol=2e-2)

        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # vmin = -6
        # vmax = -1
        # phim = axes[0].imshow(np.log10(phaseImg.array), vmin=vmin, vmax=vmax)
        # axes[0].set_title("PhasePSF")
        # skim = axes[1].imshow(np.log10(skImg.array), vmin=vmin, vmax=vmax)
        # axes[1].set_title("SecondKick")
        # fig.tight_layout()
        # plt.show()


@timer
def test_sk_scale():
    """Test sk scale argument"""
    kwargs = {'lam':500, 'r0':0.2, 'diam':4.0, 'flux':2.2, 'obscuration':0.3}
    sk_arcsec = galsim.SecondKick(scale_unit=galsim.arcsec, **kwargs)
    sk_arcmin = galsim.SecondKick(scale_unit='arcmin', **kwargs)
    do_pickle(sk_arcsec)
    do_pickle(sk_arcmin)

    np.testing.assert_almost_equal(sk_arcsec.flux, sk_arcmin.flux)
    np.testing.assert_almost_equal(sk_arcsec.kValue(0.0, 0.0), sk_arcmin.kValue(0.0, 0.0))
    np.testing.assert_almost_equal(sk_arcsec.kValue(0.0, 1.0), sk_arcmin.kValue(0.0, 60.0))
    np.testing.assert_almost_equal(sk_arcsec.kValue(0.0, 10.0), sk_arcmin.kValue(0.0, 600.0))
    np.testing.assert_almost_equal(sk_arcsec._sbs.xValue(galsim.PositionD(0.0, 6.0)._p),
                                   sk_arcmin._sbs.xValue(galsim.PositionD(0.0, 0.1)._p)/60**2,
                                   decimal=5)
    np.testing.assert_almost_equal(sk_arcsec._sbs.xValue(galsim.PositionD(0.0, 0.6)._p),
                                   sk_arcmin._sbs.xValue(galsim.PositionD(0.0, 0.01)._p)/60**2,
                                   decimal=5)
    np.testing.assert_almost_equal(sk_arcsec._sba.xValue(galsim.PositionD(0.0, 6.0)._p),
                                   sk_arcmin._sba.xValue(galsim.PositionD(0.0, 0.1)._p)/60**2,
                                   decimal=5)
    np.testing.assert_almost_equal(sk_arcsec._sba.xValue(galsim.PositionD(0.0, 0.6)._p),
                                   sk_arcmin._sba.xValue(galsim.PositionD(0.0, 0.01)._p)/60**2,
                                   decimal=5)

    if __name__ == '__main__':
        # These are a little slow, since the xValue is doing a real-space convolution integral.
        np.testing.assert_almost_equal(sk_arcsec.xValue(0.0, 6.0),
                                       sk_arcmin.xValue(0.0, 0.1)/60**2,
                                       decimal=5)
        np.testing.assert_almost_equal(sk_arcsec.xValue(0.0, 1.2),
                                       sk_arcmin.xValue(0.0, 0.02)/60**2,
                                       decimal=5)

    img1 = sk_arcsec.drawImage(nx=32, ny=32, scale=0.2)
    img2 = sk_arcmin.drawImage(nx=32, ny=32, scale=0.2/60.0)
    np.testing.assert_almost_equal(img1.array, img2.array)

    # Also check that default flux works
    del kwargs['flux']
    sk_arcsec = galsim.SecondKick(scale_unit=galsim.arcsec, **kwargs)
    sk_arcmin = galsim.SecondKick(scale_unit='arcmin', **kwargs)
    do_pickle(sk_arcsec)
    do_pickle(sk_arcmin)
    np.testing.assert_almost_equal(sk_arcmin.flux, 1.0)
    np.testing.assert_almost_equal(sk_arcsec.flux, 1.0)


@timer
def test_sk_ne():
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    objs = [galsim.SecondKick(lam=500.0, r0=0.2, diam=4.0),
            galsim.SecondKick(lam=550.0, r0=0.2, diam=4.0),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.0),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2, obscuration=0.4),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2, kcrit=1.234),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2, flux=2.2),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2, scale_unit='arcmin'),
            galsim.SecondKick(lam=500.0, r0=0.2, diam=4.0, gsparams=gsp)]
    all_obj_diff(objs)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--profile", action='store_true', help="Profile tests")
    parser.add_argument("--prof_out", default=None, help="Profiler output file")
    args = parser.parse_args()

    if args.profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()

    test_init()
    test_structure_function()
    test_limiting_cases()
    test_sk_phase_psf()
    test_sk_scale()
    test_sk_ne()

    if args.profile:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)
        if args.prof_out:
            pr.dump_stats(args.prof_out)
