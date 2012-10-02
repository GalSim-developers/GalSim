"""Unit tests for integration routines at the Python layer.
"""

import numpy as np
try:
    import galsim
except ImportError:
    import os
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

test_sigma = 7.                   # test value of Gaussian sigma for integral tests
test_rel_err = 1.e-7              # the relative accuracy at which to test
test_abs_err = 1.e-13             # the absolute accuracy at which to test
test_mock_inf = 2.e10             # number that gets interpreted as infinity by integration routines


def test_gaussian_finite_limits():
    """Test the integration of a 1D zero-mean Gaussian across intervals of [-1, 1], [0, 20]
    and [-50, -40].
    """
    # Define our test function
    def test_func(x): return np.exp(-.5 * x**2 / test_sigma**2)

    test_integral = galsim.integ.int1d(test_func, -1., 1., test_rel_err, test_abs_err)
    # test results easily calculated using Wolfram alpha
    true_result = 1.99321805307377285009
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Gaussian integral failed across interval [-1, 1].")

    test_integral = galsim.integ.int1d(test_func, 0., 20., test_rel_err, test_abs_err)
    true_result = 8.73569586966967345835 
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Gaussian integral failed across interval [0, 20].")

    test_integral = galsim.integ.int1d(test_func, -50., -40., test_rel_err, test_abs_err)
    true_result = 9.66426031085587421984e-8
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Gaussian integral failed across interval [-50, -40].")

def test_gaussian_infinite_limits():
    """Test the integration of a 1D zero-mean Gaussian across intervals of [0, inf], [-inf, 5.4]
    and [-inf, inf].
    """
    # Define our test function
    def test_func(x): return np.exp(-.5 * x**2 / test_sigma**2)

    test_integral = galsim.integ.int1d(test_func, 0., test_mock_inf, test_rel_err, test_abs_err)
    # test results easily calculated using Wolfram alpha
    true_result = 8.77319896120850210849
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Gaussian integral failed across interval [0, inf].")

    test_integral = galsim.integ.int1d(test_func, -test_mock_inf, 5.4, test_rel_err, test_abs_err)
    true_result = 13.68221660030048620971
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Gaussian integral failed across interval [-inf, 5.4].")

    test_integral = galsim.integ.int1d(
        test_func, -test_mock_inf, test_mock_inf, test_rel_err, test_abs_err)
    true_result = 17.54639792241700421699
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Gaussian integral failed across interval [-inf, inf].")

def test_sinxsq_finite_limits():
    """Test the integration of a slightly tricky oscillating sin(x^2) * exp(-|x|) function across 
    finite intervals [-1, 1], [0, 20], [-50, 40].
    """
    # Define our test function
    def test_func(x): return np.sin(x**2) * np.exp(-np.abs(x))

    test_integral = galsim.integ.int1d(test_func, -1., 1., test_rel_err, test_abs_err)
    # test results easily calculated using Wolfram alpha
    true_result = 0.30182513444548879567
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Sin(x^2) integral failed across interval [-1, 1].")

    test_integral = galsim.integ.int1d(test_func, 0., 20., test_rel_err, test_abs_err)
    true_result = 0.27051358019041255485 
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Sin(x^2) integral failed across interval [0, 20].")

    test_integral = galsim.integ.int1d(test_func, -50., -40., test_rel_err, test_abs_err)
    true_result = 3.23169139033148542316e-20
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Sin(x^2) integral failed across interval [-50, -40].")

def test_sinxsq_infinite_limits():
    """Test the integration of a slightly tricky oscillating sin(x^2) * exp(-|x|) function across 
    infinite intervals [0, inf], [-inf, 5.4], [-inf, inf].
    """
    # Define our test function
    def test_func(x): return np.sin(x**2) * np.exp(-np.abs(x))

    test_integral = galsim.integ.int1d(test_func, 0., test_mock_inf, test_rel_err, test_abs_err)
    # test results easily calculated using Wolfram alpha
    true_result = 0.27051358016221414426
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Sin(x^2) * exp(-|x|) integral failed across interval [0, inf].")

    test_integral = galsim.integ.int1d(test_func, -test_mock_inf, 5.4, test_rel_err, test_abs_err)
    true_result = 0.5413229824941895221
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Sin(x^2) * exp(-|x|) integral failed across interval [-inf, 5.4].")

    test_integral = galsim.integ.int1d(
        test_func, -test_mock_inf, test_mock_inf, test_rel_err, test_abs_err)
    true_result = 0.54102716032442828852 
    np.testing.assert_allclose(
        test_integral, true_result, rtol=test_rel_err, atol=test_abs_err, verbose=True,
        err_msg="Sin(x^2) * exp(-|x|) integral failed across interval [-inf, inf].")

