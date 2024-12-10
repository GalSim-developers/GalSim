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
import galsim
import time

from galsim_test_helpers import *


@timer
def test_init(run_slow):
    """Test generation of SecondKick profiles
    """
    obscuration = 0.5

    if run_slow:
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
                check_pickle(sk)
                t2 = time.time()

                gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
                sk2 = galsim.SecondKick(flux=2.2, gsparams=gsp, **kwargs)
                assert sk2 != sk
                assert sk2 == sk.withGSParams(gsp)
                assert sk2 == sk.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

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

                # Check negative flux
                sk2 = galsim.SecondKick(flux=-2.2, **kwargs)
                sk3 = sk.withFlux(-2.2)
                assert sk2 == sk3
                obj2 = galsim.Convolve(sk2, galsim.Gaussian(fwhm=0.2))
                check_basic(obj2, "SecondKick with negative flux")


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
def test_sk_phase_psf(run_slow):
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
    kcrits = [1, 3, 10] if run_slow else [1]
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
def test_sk_scale(run_slow):
    """Test sk scale argument"""
    kwargs = {'lam':500, 'r0':0.2, 'diam':4.0, 'flux':2.2, 'obscuration':0.3}
    sk_arcsec = galsim.SecondKick(scale_unit=galsim.arcsec, **kwargs)
    sk_arcmin = galsim.SecondKick(scale_unit='arcmin', **kwargs)
    check_pickle(sk_arcsec)
    check_pickle(sk_arcmin)

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

    if run_slow:
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
    check_pickle(sk_arcsec)
    check_pickle(sk_arcmin)
    np.testing.assert_almost_equal(sk_arcmin.flux, 1.0)
    np.testing.assert_almost_equal(sk_arcsec.flux, 1.0)


@timer
def test_sk_shoot():
    """Test SecondKick with photon shooting.  Particularly the flux of the final image.
    """
    rng = galsim.BaseDeviate(1234)
    obj = galsim.SecondKick(lam=500, r0=0.2, diam=4, flux=1.e4)
    im = galsim.Image(500,500, scale=1)
    im.setCenter(0,0)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng.duplicate())
    assert photons2 == photons, "SecondKick makePhot not equivalent to drawPhot"

    # Can treat the profile as a convolution of a delta function and put it in a photon_ops list.
    delta = galsim.DeltaFunction(flux=1.e4)
    psf = galsim.SecondKick(lam=500, r0=0.2, diam=4)
    photons3 = delta.makePhot(poisson_flux=False, rng=rng.duplicate(), photon_ops=[psf])
    np.testing.assert_allclose(photons3.x, photons.x)
    np.testing.assert_allclose(photons3.y, photons.y)
    np.testing.assert_allclose(photons3.flux, photons.flux)


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
    check_all_diff(objs)


if __name__ == '__main__':
    runtests(__file__)
