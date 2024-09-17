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
from galsim_test_helpers import *


test_sigma = 7.                   # test value of Gaussian sigma for integral tests
test_rel_err = 1.e-7              # the relative accuracy at which to test
test_abs_err = 1.e-13             # the absolute accuracy at which to test
test_mock_inf = 2.e10             # number large enough to get interpreted as infinity by
                                  # integration routines
test_decimal = 7


@timer
def test_gaussian_finite_limits():
    """Test the integration of a 1D zero-mean Gaussian across intervals of [-1, 1], [0, 20]
    and [-50, -40].
    """
    # Define our test function
    def test_func(x): return np.exp(-.5 * x**2 / test_sigma**2)

    test_integral = galsim.integ.int1d(test_func, -1., 1., test_rel_err, test_abs_err)
    # test results easily calculated using Wolfram alpha
    true_result = 1.99321805307377285009
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Gaussian integral failed across interval [-1, 1].")

    test_integral = galsim.integ.int1d(test_func, 0., 20., test_rel_err, test_abs_err)
    true_result = 8.73569586966967345835
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Gaussian integral failed across interval [0, 20].")

    test_integral = galsim.integ.int1d(test_func, -50., -40., test_rel_err, test_abs_err)
    true_result = 9.66426031085587421984e-8
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Gaussian integral failed across interval [-50, -40].")


@timer
def test_gaussian_infinite_limits():
    """Test the integration of a 1D zero-mean Gaussian across intervals of [0, inf], [-inf, 5.4]
    and [-inf, inf].
    """
    # Define our test function
    def test_func(x): return np.exp(-.5 * x**2 / test_sigma**2)

    test_integral = galsim.integ.int1d(test_func, 0., test_mock_inf, test_rel_err, test_abs_err)
    # test results easily calculated using Wolfram alpha
    true_result = 8.77319896120850210849
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Gaussian integral failed across interval [0, inf].")

    test_integral = galsim.integ.int1d(test_func, -test_mock_inf, 5.4, test_rel_err, test_abs_err)
    true_result = 13.68221660030048620971
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Gaussian integral failed across interval [-inf, 5.4].")

    test_integral = galsim.integ.int1d(
        test_func, -test_mock_inf, test_mock_inf, test_rel_err, test_abs_err)
    true_result = 17.54639792241700421699
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Gaussian integral failed across interval [-inf, inf].")


@timer
def test_sinxsqexpabsx_finite_limits():
    """Test the integration of a slightly tricky oscillating sin(x^2) * exp(-|x|) function across
    finite intervals [-1, 1], [0, 20], [-15, 14].
    """
    # Define our test function
    def test_func(x): return np.sin(x**2) * np.exp(-np.abs(x))

    test_integral = galsim.integ.int1d(test_func, -1., 1., test_rel_err, test_abs_err)
    # test results easily calculated using Wolfram alpha
    true_result = 0.30182513444548879567
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Sin(x^2) * exp(-|x|) integral failed across interval [-1, 1].")

    test_integral = galsim.integ.int1d(test_func, 0., 20., test_rel_err, test_abs_err)
    true_result = 0.27051358019041255485
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Sin(x^2) * exp(-|x|) integral failed across interval [0, 20].")

    test_integral = galsim.integ.int1d(test_func, -15., -14., test_rel_err, test_abs_err)
    true_result = 7.81648378350593176887e-9
    np.testing.assert_almost_equal(
        (test_integral - true_result) / true_result, 0., decimal=test_decimal, verbose=True,
        err_msg="Sin(x^2) * exp(-|x|) integral failed across interval [-15, -14].")


@timer
def test_sinxsqexpabsx_infinite_limits():
    """Test the integration of a slightly tricky oscillating sin(x^2) * exp(-|x|) function across
    infinite intervals [0, inf], [-inf, 5.4], [-inf, inf].
    """
    # Define our test function
    def test_func(x): return np.sin(x**2) * np.exp(-np.abs(x))

    test_integral = galsim.integ.int1d(test_func, 0., test_mock_inf, test_rel_err, test_abs_err)
    # test results easily calculated using Wolfram alpha
    true_result = 0.27051358016221414426
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Sin(x^2) * exp(-|x|) integral failed across interval [0, inf].")

    test_integral = galsim.integ.int1d(test_func, -test_mock_inf, 5.4, test_rel_err, test_abs_err)
    true_result = 0.5413229824941895221
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Sin(x^2) * exp(-|x|) integral failed across interval [-inf, 5.4].")

    test_integral = galsim.integ.int1d(
        test_func, -test_mock_inf, test_mock_inf, test_rel_err, test_abs_err)
    true_result = 0.54102716032442828852
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="Sin(x^2) * exp(-|x|) integral failed across interval [-inf, inf].")


@timer
def test_invroot_finite_limits():
    """Test the integration of |x|^(-1/2) across intervals [0,1], [0,300] (integrable pole at x=0).
    """
    # Define our test function
    def test_func(x): return 1. / np.sqrt(np.abs(x))
    test_integral = galsim.integ.int1d(test_func, 0, 1., test_rel_err, test_abs_err)
    # test results easily calculated using Wolfram alpha
    true_result = 2.
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="|x|^(-1/2) integral failed across interval [0, 1].")

    test_integral = galsim.integ.int1d(test_func, 0., 300., test_rel_err, test_abs_err)
    true_result = 34.64101615137754587055
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="|x|^(-1/2) integral failed across interval [0, 300].")


@timer
def test_invroot_infinite_limits():
    """Test the integration of |x|^(-2) across intervals [1,2], [1,inf].
    Also check that [0,1] raises an exception.
    """
    # Define our test function
    def test_func(x): return x**-2
    test_integral = galsim.integ.int1d(test_func, 1., 2., test_rel_err, test_abs_err)
    true_result = 0.5
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="x^(-2) integral failed across interval [1, 2].")

    test_integral = galsim.integ.int1d(test_func, 1., test_mock_inf, test_rel_err, test_abs_err)
    true_result = 1.0
    np.testing.assert_almost_equal(
        test_integral, true_result, decimal=test_decimal, verbose=True,
        err_msg="x^(-2) integral failed across interval [1, inf].")

    with assert_raises(galsim.GalSimError):
        galsim.integ.int1d(test_func, 0., 1., test_rel_err, test_abs_err)


@timer
def test_midpoint_basic():
    """Test the basic functionality of the midptRule() function.
    """
    # This shouldn't be super accurate, but just make sure it's not really broken.
    x = 0.01*np.arange(1000)
    func = lambda x:x**2
    result = galsim.integ.midptRule(func, x)
    expected_val = 1.e3/3.
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=2, verbose=True,
        err_msg='Simple test of midptRule() method failed for f(x)=x^2 from 0 to 10')

    # Error with 0 or 1 ascissae
    with assert_raises(galsim.GalSimValueError):
        galsim.integ.midptRule(func, [0])
    with assert_raises(galsim.GalSimValueError):
        galsim.integ.midptRule(func, [])

@timer
def test_trapz_basic():
    """Test the basic functionality of the trapzRule() function.
    """
    # This shouldn't be super accurate, but just make sure it's not really broken.
    x = np.linspace(0, 1, 100000)
    func = lambda x: x**2
    result = galsim.integ.trapzRule(func, x)
    expected_val = 1./3.
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=6, verbose=True,
        err_msg='Test of trapzRule() with points failed for f(x)=x^2 from 0 to 1')

    # quadRule with no weight is equivalent.
    result2 = galsim.integ.quadRule(func, x)
    np.testing.assert_almost_equal(result2, result)

    # Error with 0 or 1 ascissae
    with assert_raises(galsim.GalSimValueError):
        galsim.integ.trapzRule(func, [0])
    with assert_raises(galsim.GalSimValueError):
        galsim.integ.trapzRule(func, [])

@timer
def test_quad_basic():
    """Test the basic functionality of the quadRule() function.
    """
    x = np.linspace(0, 1, 1000)
    # With func and weight both linear, quad should be exact.
    func = lambda x: x
    weight = lambda x: 1+x
    result1 = galsim.integ.midptRule(func, x, weight)
    result2 = galsim.integ.trapzRule(func, x, weight)
    result3 = galsim.integ.quadRule(func, x, weight)
    expected_val = 5./6.
    print('int(x^2 (1+x), 0..1) = ')
    print('midpt: ',result1)
    print('trapz: ',result2)
    print('quad:  ',result3)
    print('true:  ',expected_val)
    np.testing.assert_allclose(result1, expected_val, rtol=2.e-3)
    np.testing.assert_allclose(result2, expected_val, rtol=3.e-7)
    np.testing.assert_allclose(result3, expected_val, rtol=1.e-14)

    # Check only 2 abscissae
    x = np.linspace(0, 1, 2)
    func = lambda x: x
    result1 = galsim.integ.midptRule(func, x, weight)
    result2 = galsim.integ.trapzRule(func, x, weight)
    result3 = galsim.integ.quadRule(func, x, weight)
    np.testing.assert_allclose(result1, expected_val, rtol=2)
    np.testing.assert_allclose(result2, expected_val, rtol=0.3)
    np.testing.assert_allclose(result3, expected_val, rtol=1.e-10)

    # Error with 0 or 1 abscissae
    with assert_raises(galsim.GalSimError):
        galsim.integ.quadRule(func, [0], weight)
    with assert_raises(galsim.GalSimError):
        galsim.integ.quadRule(func, [], weight)
    with assert_raises(galsim.GalSimError):
        galsim.integ.quadRule(func, [0])
    with assert_raises(galsim.GalSimError):
        galsim.integ.quadRule(func, [])

@timer
def test_hankel():
    """Test the galsim.integ.hankel function
    """
    # Most of the use of this function is in C++, but we provide hooks to use it in python
    # too in case that's useful for people.
    # (We don't currently use this from python in GalSim proper.)

    f1 = lambda r: np.exp(-r)
    # The Hankel transform of exp(-r) is (1+k^2)**(-3/2)
    for k in [1, 1.e-2, 0.234, 23.9, 1.e-8, 0]:
        result = galsim.integ.hankel(f1, k)
        expected_val = (1+k**2)**-1.5
        np.testing.assert_allclose(result, expected_val)

    r0 = 1.7
    f2 = lambda r: 1.-(r/r0)**2
    # The truncated Hankel transform of (1-(r/r0)^2) up to r0 is 2 J_2(r0 k)/k^2
    for k in [1, 1.e-2, 0.234, 23.9, 1.e-8, 0]:
        result = galsim.integ.hankel(f2, k, rmax=r0, rel_err=1.e-8)
        if k == 0:
            expected_val = r0**2/4  # The limit as k->0.
        else:
            expected_val = 2*galsim.bessel.jn(2,r0*k)/k**2
        np.testing.assert_allclose(result, expected_val, rtol=1.e-6)

    # The generalized Hankel transform of exp(-r) is:
    # (1 + nu (1+k^2)^0.5) k^nu / ( (1+k^2)^1.5 (1 + (1+k^2)^0.5)^nu
    for nu in [0, 1, 7, 0.5, 0.003, 12.23]:
        for k in [1, 1.e-2, 0.234, 23.9, 1.e-5, 1.e-8, 0]:
            result = galsim.integ.hankel(f1, k, nu=nu)
            expected_val = (1+k**2)**-1.5 * (1+nu*(1+k**2)**0.5) * k**nu / (1+(1+k**2)**0.5)**nu
            print(nu, k, result, expected_val)
            np.testing.assert_allclose(result, expected_val, rtol=1.e-6, atol=1.e-12)

    # Test doing multiple k values at once.
    k = np.array([1, 1.e-2, 0.234, 23.9, 1.e-5, 1.e-8, 0])
    for nu in [0, 1, 7, 0.5, 0.003, 12.23]:
        result = galsim.integ.hankel(f1, k, nu=nu)
        expected_val = (1+k**2)**-1.5 * (1+nu*(1+k**2)**0.5) * k**nu / (1+(1+k**2)**0.5)**nu
        np.testing.assert_allclose(result, expected_val, rtol=1.e-6, atol=1.e-12)

    with assert_raises(galsim.GalSimValueError):
        galsim.integ.hankel(f1, k=-0.3)
    with assert_raises(galsim.GalSimValueError):
        galsim.integ.hankel(f1, k=0.3, nu=-1)
    with assert_raises(galsim.GalSimValueError):
        galsim.integ.hankel(f1, k=0.3, nu=-0.5)


@timer
def test_gq_annulus():
    """Test the galsim.integ.gq_annulus function
    """

    # We can use the normalization of annular Zernikes to test.
    # From the Zernike docs:
    # \int_\mathrm{annulus} Z_i Z_j dA =
    # \pi \left(R_\mathrm{outer}^2 - R_\mathrm{inner}^2\right) \delta_{i, j}
    # I.e., the integral is the annulus area times Kronecker delta.

    rng = galsim.BaseDeviate(1111).as_numpy_generator()
    for j1 in range(1, 28):   # up to 6th order xy polynomials
        r1 = rng.uniform(0.3, 0.6)
        r2 = rng.uniform(0.9, 1.1)
        area = np.pi*(r2**2 - r1**2)
        Z1 = galsim.zernike.Zernike([0]*j1+[1], R_inner=r1, R_outer=r2)
        for j2 in range(1, 28):
            Z2 = galsim.zernike.Zernike([0]*j2+[1], R_inner=r1, R_outer=r2)
            # Product of 2 6th order polynomials is 12th order, so we need 6 rings and 13 spokes.
            np.testing.assert_allclose(
                galsim.integ.gq_annulus(Z1*Z2, r2, r1, n_rings=6, n_spokes=13),
                area if j1 == j2 else 0,
                rtol=1.e-11, atol=1.e-11
            )


if __name__ == "__main__":
    runtests(__file__)
