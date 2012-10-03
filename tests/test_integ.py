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
test_mock_inf = 2.e10             # number large enough to get interpreted as infinity by 
                                  # integration routines
test_decimal = 7

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_gaussian_finite_limits():
    """Test the integration of a 1D zero-mean Gaussian across intervals of [-1, 1], [0, 20]
    and [-50, -40].
    """
    import time
    t1 = time.time()

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_gaussian_infinite_limits():
    """Test the integration of a 1D zero-mean Gaussian across intervals of [0, inf], [-inf, 5.4]
    and [-inf, inf].
    """
    import time
    t1 = time.time()

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sinxsqexpabsx_finite_limits():
    """Test the integration of a slightly tricky oscillating sin(x^2) * exp(-|x|) function across 
    finite intervals [-1, 1], [0, 20], [-15, 14].
    """
    import time
    t1 = time.time()

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sinxsqexpabsx_infinite_limits():
    """Test the integration of a slightly tricky oscillating sin(x^2) * exp(-|x|) function across 
    infinite intervals [0, inf], [-inf, 5.4], [-inf, inf].
    """
    import time
    t1 = time.time()

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_invroot_finite_limits():
    """Test the integration of |x|^(-1/2) across intervals [0,1], [0,300] (integrable pole at x=0).
    """
    import time
    t1 = time.time()

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_invroot_infinite_limits():
    """Test the integration of |x|^(-2) across intervals [1,2], [1,inf].
    Also check that [0,1] raises an exception.
    """
    import time
    t1 = time.time()

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

    np.testing.assert_raises(
        RuntimeError,
        galsim.integ.int1d, test_func, 0., 1., test_rel_err, test_abs_err)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_gaussian_finite_limits()
    test_gaussian_infinite_limits()
    test_sinxsqexpabsx_finite_limits()
    test_sinxsqexpabsx_infinite_limits()
    test_invroot_finite_limits()
    test_invroot_infinite_limits()

