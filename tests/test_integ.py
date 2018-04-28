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
"""Unit tests for integration routines at the Python layer.
"""

from __future__ import print_function
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
    """Test the basic functionality of the midpt() method.
    """
    # This shouldn't be super accurate, but just make sure it's not really broken.
    x = 0.01*np.arange(1000)
    f = x**2
    result = galsim.integ.midpt(f, x)
    expected_val = 10**3./3.
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=2, verbose=True,
        err_msg='Simple test of midpt() method failed for f(x)=x^2 from 0 to 10')

    # Also test midptRule
    result = galsim.integ.midptRule(lambda x:x**2, x)
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=2, verbose=True,
        err_msg='Simple test of midptRule() method failed for f(x)=x^2 from 0 to 10')


@timer
def test_trapz_basic():
    """Test the basic functionality of the trapz() method.
    """
    # This shouldn't be super accurate, but just make sure it's not really broken.
    func = lambda x: x**2
    result = galsim.integ.trapz(func, 0, 1)
    expected_val = 1.**3./3.
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=6, verbose=True,
        err_msg='Simple test of trapz() method failed for f(x)=x^2 from 0 to 1')

    result = galsim.integ.trapz(func, 0, 1, np.linspace(0, 1, 100000))
    expected_val = 1.**3./3.
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=6, verbose=True,
        err_msg='Test of trapz() with points failed for f(x)=x^2 from 0 to 1')

    #Also test trapzRule
    result = galsim.integ.trapzRule(func, np.linspace(0, 1, 100000))
    np.testing.assert_almost_equal(
        result/expected_val, 1.0, decimal=6, verbose=True,
        err_msg='Test of trapzRule() with points failed for f(x)=x^2 from 0 to 1')

    assert_raises(ValueError, galsim.integ.trapz, func, 0, 1, points=np.linspace(0, 1.1, 100))
    assert_raises(ValueError, galsim.integ.trapz, func, 0.1, 1, points=np.linspace(0, 1, 100))
    assert_raises(TypeError, galsim.integ.trapz, func, 0.1, 1, points=2.3)


if __name__ == "__main__":
    test_gaussian_finite_limits()
    test_gaussian_infinite_limits()
    test_sinxsqexpabsx_finite_limits()
    test_sinxsqexpabsx_infinite_limits()
    test_invroot_finite_limits()
    test_invroot_infinite_limits()
    test_midpoint_basic()
    test_trapz_basic()
