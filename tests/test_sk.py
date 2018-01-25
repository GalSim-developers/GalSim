import numpy as np
import galsim

from galsim_test_helpers import *

lam = 500.0
r0 = 0.2
L0 = 30.0
diam = 8.36
obscuration = 0.0
kcrit = 2*np.pi/r0
flux = 1.0

bigGSP = galsim.GSParams(maximum_fft_size=8192)

def test_sk(slow=False):
    """Test generation of SK profiles
    """
    if __name__ == '__main__' and slow:
        lams = [300.0, 500.0, 1100.0]
        r0_500s = [0.1, 0.15, 0.3]
        L0s = [1e10, 25.0, 10.0]
        kcrits = [20.0, 60.0]
    else:
        lams = [500.0]
        r0_500s = [0.15]
        L0s = [25.0]
        kcrits = [20.0]
    for lam in lams:
        for r0_500 in r0_500s:
            r0 = r0_500*(lam/500)**(6./5)
            for L0 in L0s:
                for kcrit in kcrits:
                    kwargs = {'lam':lam, 'r0':r0, 'L0':L0, 'kcrit':kcrit, 'diam':4.0}
                    kwargs['gsparams'] = bigGSP
                    print(kwargs)

                    sk = galsim.SK(flux=2.2, **kwargs)
                    print(sk.stepk, sk.maxk)
                    np.testing.assert_almost_equal(sk.flux, 2.2)
                    do_pickle(sk)
                    do_pickle(sk._sbp)
                    do_pickle(sk._sbp, lambda x: (x.getFlux(), x.getGSParams()))

                    # Raw sk objects are hard to draw due to a large maxk/stepk ratio.
                    # Sharply decrease maxk by convolving in a smallish Gaussian.
                    obj = galsim.Convolve(sk, galsim.Gaussian(fwhm=0.2))
                    check_basic(obj, "SK")
                    img = galsim.Image(16, 16, scale=0.2)
                    do_shoot(obj, img, "SK")
                    do_kvalue(obj, img, "SK")


def test_airy():
    """Check access to the airy subcomponent of SK.
    """
    sk = galsim.SK(lam, r0, diam, obscuration, L0, kcrit)
    airy = galsim.Airy(lam=lam, diam=diam)
    assert sk._sbairy == airy._sbp


def test_structure_function():
    """Test that SK structure function is equivalent to vonKarman structure function when kcrit=0.
    This is nontrivial since the SK structure function is numerically integrated, while the vK
    structure function is evaluated analytically.
    """
    sk = galsim.SK(lam, r0, diam, obscuration, L0, kcrit=0.0)
    vk = galsim.VonKarman(lam, r0, L0=L0)

    for rho in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        sksf = sk._structure_function(rho)
        vksf = vk._structure_function(rho)
        np.testing.assert_allclose(sksf, vksf, rtol=1e-6)


def test_limiting_cases():
    """SK has some two interesting limiting cases.
    A) When kcrit = 0, SK = Convolve(Airy, VonKarman).
    B) When kcrit = inf, SK = Airy
    Test these.
    """

    # First kcrit=0
    sk = galsim.SK(lam, r0, diam, obscuration, L0, kcrit=0.0)
    limiting_case = galsim.Convolve(
        galsim.VonKarman(lam, r0, L0=L0),
        galsim.Airy(lam=lam, diam=diam)
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
    sk = galsim.SK(lam, r0, diam, obscuration, L0, kcrit=np.inf)
    limiting_case = galsim.Airy(lam=lam, diam=diam)

    for k in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        np.testing.assert_allclose(
            sk.kValue(0, k).real,
            limiting_case.kValue(0, k).real,
            rtol=1e-3,
            atol=1e-4)


def test_sf_lut(slow=False):
    """Test the suitability of the lookup table used to store the structure function by comparing
    results of xValue and kValue both with and without using the LUT.
    """
    sk = galsim.SK(lam, r0, diam, obscuration, L0, 0.5*kcrit)

    print("Testing kValue")
    for k in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 600.0]:
        print()
        print(k)
        print(sk.kValue(0, k).real)
        print(sk._sbp.kValueSlow(k))
        np.testing.assert_allclose(
            sk.kValue(0, k).real,   # Uses LUT
            sk._sbp.kValueSlow(k),  # No LUT
            rtol=1e-5,
            atol=1e-5
        )

    print()
    print()
    print("Testing xValue")
    if __name__ == "__main__":
        xs = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2]
        if slow:
            xs.extend([0.3, 0.6, 1.0, 2.0, 3.0, 6.0])
    else:
        xs = [0.0, 0.001, 0.003, 0.01, 0.03]
    for x in xs:
        print()
        print(x)
        print(sk.xValue(0, x))
        print(sk._sbp.xValueSlow(x))
        np.testing.assert_allclose(
            sk.xValue(0, x),        # Uses LUT
            sk._sbp.xValueSlow(x),  # No LUT
            rtol=1e-5,
            atol=1e-2
        )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--slow", action='store_true', help="Run slow tests")
    args = parser.parse_args()

    test_sk(args.slow)
    test_airy()
    test_structure_function()
    test_limiting_cases()
    test_sf_lut(args.slow)
