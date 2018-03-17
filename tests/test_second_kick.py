import numpy as np
import galsim

from galsim_test_helpers import *

@timer
def test_init(slow=False):
    """Test generation of SecondKick profiles
    """
    obscuration = 0.5
    bigGSP = galsim.GSParams(maximum_fft_size=8192)

    if __name__ == '__main__' and slow:
        lams = [300.0, 500.0, 1100.0]
        r0_500s = [0.1, 0.15, 0.3]
        L0s = [1e10, 25.0, 10.0]
        kcrits = [0.1, 0.2, 0.4]
    else:
        lams = [500.0]
        r0_500s = [0.15]
        L0s = [25.0]
        kcrits = [0.2]
    for lam in lams:
        for r0_500 in r0_500s:
            r0 = r0_500*(lam/500)**(6./5)
            for L0 in L0s:
                for kcrit in kcrits:
                    kwargs = {'lam':lam, 'r0':r0, 'L0':L0, 'kcrit':kcrit, 'diam':4.0}
                    print(kwargs)
                    kwargs['gsparams'] = bigGSP

                    sk = galsim.SecondKick(flux=2.2, **kwargs)
                    np.testing.assert_almost_equal(sk.flux, 2.2)
                    do_pickle(sk)
                    do_pickle(sk._sbp)
                    do_pickle(sk._sbp, lambda x: (x.getFlux(), x.getGSParams()))

                    # Raw sk objects are hard to draw due to a large maxk/stepk ratio.
                    # Decrease maxk by convolving in a smallish Gaussian.
                    obj = galsim.Convolve(sk, galsim.Gaussian(fwhm=0.2))
                    check_basic(obj, "SecondKick")
                    img = galsim.Image(16, 16, scale=0.2)
                    do_shoot(obj, img, "SecondKick")
                    do_kvalue(obj, img, "SecondKick")


@timer
def test_structure_function():
    """Test that SecondKick structure function is equivalent to vonKarman structure function when
    kcrit=0.  This is nontrivial since the SecondKick structure function is numerically integrated,
    while the vK structure function is evaluated analytically.
    """
    lam = 500.0
    r0 = 0.2
    L0 = 30.0
    diam = 8.36
    obscuration = 0.61

    sk = galsim.SecondKick(lam, r0, diam, obscuration, L0, kcrit=0.0)
    vk = galsim.VonKarman(lam, r0, L0=L0)

    for rho in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        sksf = sk._structure_function(rho)
        vksf = vk._structure_function(rho)
        np.testing.assert_allclose(sksf, vksf, rtol=1e-6)


@timer
def test_limiting_cases():
    """SecondKick has some two interesting limiting cases.
    A) When kcrit = 0, SecondKick = Convolve(Airy, VonKarman).
    B) When kcrit = inf, SecondKick = Airy
    Test these.
    """
    lam = 500.0
    r0 = 0.2
    L0 = 30.0
    diam = 8.36
    obscuration = 0.61

    # First kcrit=0
    sk = galsim.SecondKick(lam, r0, diam, obscuration, L0, kcrit=0.0)
    limiting_case = galsim.Convolve(
        galsim.VonKarman(lam, r0, L0=L0),
        galsim.Airy(lam=lam, diam=diam, obscuration=obscuration)
    )
    print(sk.stepk, sk.maxk)
    print(limiting_case.stepk, limiting_case.maxk)

    for k in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        np.testing.assert_allclose(
            sk.kValue(0, k).real,
            limiting_case.kValue(0, k).real,
            rtol=1e-7,
            atol=1e-8)

    # kcrit=inf
    sk = galsim.SecondKick(lam, r0, diam, obscuration, L0, kcrit=np.inf)
    limiting_case = galsim.Airy(lam=lam, diam=diam, obscuration=obscuration)

    for k in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        np.testing.assert_allclose(
            sk.kValue(0, k).real,
            limiting_case.kValue(0, k).real,
            rtol=1e-3,
            atol=1e-4)
    # Check half_light_radius agrees.  Only implemented for obscuration=0, and somewhat
    # approximate for second kick.
    airy = galsim.Airy(lam=lam, diam=diam)
    sk = galsim.SecondKick(lam=lam, r0=r0, diam=diam, kcrit=np.inf)
    assert np.abs(sk.half_light_radius/airy.half_light_radius - 1.0) < 1e-2


# @timer
# def test_sf_lut(slow=False):
#     """Test the suitability of the lookup table used to store the structure function by comparing
#     results of xValue and kValue both with and without using the LUT.
#     """
#     lam = 500.0
#     r0 = 0.2
#     L0 = 30.0
#     diam = 8.36
#     obscuration = 0.61
#     kcrit = 2*np.pi/r0
#
#     sk = galsim.SecondKick(lam, r0, diam, obscuration, L0, 0.5*kcrit)
#     print("stepk = {}".format(sk.stepk))
#     print("maxk = {}".format(sk.maxk))
#
#     print("Testing kValue")
#     for k in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 600.0]:
#         print()
#         print("k = {}".format(k))
#         print(sk._sbp.kValueDouble(k))
#         print(sk._sbp.kValueRaw(k))
#         np.testing.assert_allclose(
#             sk._sbp.kValueDouble(k),     # Uses LUT
#             sk._sbp.kValueRaw(k),  # No LUT
#             rtol=1e-3,
#             atol=1e-3
#         )
#
#     print()
#     print()
#     print("Testing xValue")
#     if __name__ == '__main__':
#         xs = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2]
#         if slow:
#             xs.extend([0.3, 0.6, 1.0, 2.0, 3.0, 6.0])
#     else:
#         xs = [0.0, 0.001, 0.003, 0.01, 0.03]
#     for x in xs:
#         print()
#         print("x = {}".format(x))
#         print(sk._sbp.xValueDouble(x))
#         print(sk._sbp.xValueRaw(x))
#         print(sk._sbp.xValueExact(x))
#         np.testing.assert_allclose(
#             sk._sbp.xValueDouble(x),     # Uses LUT
#             sk._sbp.xValueRaw(x),  # No LUT
#             rtol=1e-3,
#             atol=1e-3
#         )
#
#         np.testing.assert_allclose(
#             sk._sbp.xValueDouble(x), # Uses LUT
#             sk._sbp.xValueExact(x),  # No LUT
#             rtol=1e-2,
#             atol=1e-2
#         )


@timer
def test_sk_phase_psf():
    """Test that analytic second kick profile matches what can be obtained from PhaseScreenPSF with
    an appropriate truncated power spectrum.
    """
    # import matplotlib.pyplot as plt

    lam = 500.0
    r0 = 0.2
    L0 = 30.0
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
        atm = galsim.Atmosphere(r0_500=r0, r0_weights=weights, L0=L0, rng=rng,
                                speed=speed, direction=direction,
                                screen_size=102.4, screen_scale=0.05)
        atm.instantiate(kmin=kcrit)
        psf = galsim.PhaseScreenPSF(atm, lam=500, t0=0, exptime=10, aper=aper, time_step=0.1)
        phaseImg = psf.drawImage(nx=64, ny=64, scale=0.02)
        sk = galsim.SecondKick(lam=500, r0=r0, diam=diam, obscuration=obscuration, L0=L0,
                               kcrit=kcrit)
        skImg = sk.drawImage(nx=64, ny=64, scale=0.02)
        phaseMom = galsim.hsm.FindAdaptiveMom(phaseImg)
        skMom = galsim.hsm.FindAdaptiveMom(skImg)

        print(phaseMom.moments_sigma, skMom.moments_sigma)
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
    kwargs = {'lam':500, 'r0':0.2, 'L0':25.0, 'diam':4.0, 'flux':2.2, 'obscuration':0.3}
    sk_arcsec = galsim.SecondKick(scale_unit=galsim.arcsec, **kwargs)
    sk_arcmin = galsim.SecondKick(scale_unit='arcmin', **kwargs)
    do_pickle(sk_arcmin)

    np.testing.assert_almost_equal(sk_arcsec.flux, sk_arcmin.flux)
    np.testing.assert_almost_equal(sk_arcsec.kValue(0.0, 0.0), sk_arcmin.kValue(0.0, 0.0))
    np.testing.assert_almost_equal(sk_arcsec.kValue(0.0, 10.0), sk_arcmin.kValue(0.0, 600.0))
    np.testing.assert_almost_equal(sk_arcsec.xValue(0.0, 6.0), sk_arcmin.xValue(0.0, 0.1))

    img1 = sk_arcsec.drawImage(nx=32, ny=32, scale=0.2)
    img2 = sk_arcmin.drawImage(nx=32, ny=32, scale=0.2/60.0)
    np.testing.assert_almost_equal(img1.array, img2.array)


@timer
def test_sk_ne():
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    objs = [galsim.SecondKick(lam=500.0, r0=0.2, diam=4.0),
            galsim.SecondKick(lam=550.0, r0=0.2, diam=4.0),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.0),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2, obscuration=0.4),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2, L0=1e11),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2, kcrit=1.234),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2, flux=2.2),
            galsim.SecondKick(lam=500.0, r0=0.25, diam=4.2, scale_unit='arcmin'),
            galsim.SecondKick(lam=500.0, r0=0.2, diam=4.0, gsparams=gsp)]
    all_obj_diff(objs)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--slow", action='store_true', help="Run slow tests")
    parser.add_argument("--profile", action='store_true', help="Profile tests")
    parser.add_argument("--prof_out", default=None, help="Profiler output file")
    args = parser.parse_args()

    if args.profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()

    test_init(args.slow)
    test_structure_function()
    test_limiting_cases()
    # test_sf_lut(args.slow)
    test_sk_phase_psf()
    test_sk_scale()
    test_sk_ne()

    if args.profile:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)
        if args.prof_out:
            pr.dump_stats(args.prof_out)
