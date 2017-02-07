# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
import warnings

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    import os
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


@timer
def test_j0():
    """Test the bessel.j0 function"""
    x_list = [ 0, 1.01, 0.2, 3.3, 5.9, 77. ]
    vals1 = [ galsim.bessel.j0(x) for x in x_list ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.j0(x) for x in x_list ]
        print('vals2 = ',vals2)
        np.testing.assert_almost_equal(
            vals1, vals2, 8, "bessel.j0 disagrees with scipy.special.j0")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of j0.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   1.0,
                0.76078097763218844,
                0.99002497223957631,
                -0.34429626039888467,
                0.12203335459282282,
                0.062379777089647245
            ]
    np.testing.assert_almost_equal(
        vals1, vals2, 8, "bessel.j0 disagrees with reference values")


@timer
def test_j1():
    """Test the bessel.j1 function"""
    x_list = [ 0, 1.01, 0.2, 3.3, 5.9, 77. ]
    vals1 = [ galsim.bessel.j1(x) for x in x_list ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.j1(x) for x in x_list ]
        print('vals2 = ',vals2)
        np.testing.assert_almost_equal(
            vals1, vals2, 8, "bessel.j1 disagrees with scipy.special.j1")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of j1.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   0.0,
                0.4432857612090717,
                0.099500832639236036,
                0.22066345298524112,
                -0.29514244472901613,
                0.066560642470571682
            ]
    np.testing.assert_almost_equal(
        vals1, vals2, 8, "bessel.j1 disagrees with reference values")


@timer
def test_jn():
    """Test the bessel.jn function"""
    n_list = [ 3, 4, 1, 0, 9, 7 ]
    x_list = [ 0, 1.01, 0.2, 3.3, 5.9, 77. ]
    vals1 = [ galsim.bessel.jn(n,x) for n,x in zip(n_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.jn(n,x) for n,x in zip(n_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_almost_equal(
            vals1, vals2, 8, "bessel.jn disagrees with scipy.special.jn")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of jn.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   0.0,
                0.0025745895535573995,
                0.099500832639236036,
                -0.34429626039888467,
                0.018796532416195257,
                -0.082526868218916541
            ]
    np.testing.assert_almost_equal(
        vals1, vals2, 8, "bessel.jn disagrees with reference values")


@timer
def test_jv():
    """Test the bessel.jv function"""
    v_list = [ 3.3, 4, 1.9, 0, 9.2, -7.1 ]
    x_list = [ 0, 1.01, 0.2, 3.3, 5.9, 77. ]
    vals1 = [ galsim.bessel.jv(v,x) for v,x in zip(v_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.jv(v,x) for v,x in zip(v_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_almost_equal(
            vals1, vals2, 8, "bessel.jv disagrees with scipy.special.jv")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of jv.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   0.0,
                0.0025745895535573995,
                0.0068656051839294848,
                -0.34429626039888467,
                0.015134049434950021,
                0.087784805831697565
            ]
    np.testing.assert_almost_equal(
        vals1, vals2, 8, "bessel.jv disagrees with reference values")


@timer
def test_kn():
    """Test the bessel.kn function"""
    n_list = [ 3, 4, 1, 0, 9, 7 ]
    x_list = [ 1, 2.01, 0.2, 3.3, 5.9, 7.7 ]
    vals1 = [ galsim.bessel.kn(n,x) for n,x in zip(n_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.kn(n,x) for n,x in zip(n_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_almost_equal(
            vals1, vals2, 8, "bessel.kn disagrees with scipy.special.kn")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of kn.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   7.1012628247379448,
                2.1461917781688697,
                4.7759725432204725,
                0.024610632145839577,
                0.43036201855245626,
                0.0035326394473326195
            ]
    np.testing.assert_almost_equal(
        vals1, vals2, 8, "bessel.kn disagrees with reference values")


@timer
def test_kv():
    """Test the bessel.kv function"""
    v_list = [ 3.3, 4, 1.9, 0, 9.2, -7.1 ]
    x_list = [ 1, 2.01, 0.2, 3.3, 5.9, 7.7 ]
    vals1 = [ galsim.bessel.kv(v,x) for v,x in zip(v_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.kv(v,x) for v,x in zip(v_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_almost_equal(
            vals1, vals2, 8, "bessel.kv disagrees with scipy.special.kv")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of kv.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   11.898213399340918,
                2.1461917781688693,
                37.787322153212926,
                0.024610632145839324,
                0.54492959074961467,
                0.0038228967734928758
            ]
    np.testing.assert_almost_equal(
        vals1, vals2, 8, "bessel.kv disagrees with reference values")


@timer
def test_j0_root():
    """Test the bessel.j0_root function"""
    # Our version uses tabulated values up to 40, so a useful test of the extrapolation
    # requires this to have more than 40 items.
    vals1 = [ galsim.bessel.j0_root(s) for s in range(1,51) ]
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = scipy.special.jn_zeros(0,50)
        print('vals2 = ',vals2)
        np.testing.assert_almost_equal(
            vals1, vals2, 8, "bessel.j0_root disagrees with scipy.special.jn_zeros")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of j0_root.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   2.40482556,    5.52007811,    8.65372791,   11.79153444,
                14.93091771,   18.07106397,   21.21163663,   24.35247153,
                27.49347913,   30.63460647,   33.77582021,   36.91709835,
                40.05842576,   43.19979171,   46.34118837,   49.4826099 ,
                52.62405184,   55.76551076,   58.90698393,   62.04846919,
                65.1899648 ,   68.33146933,   71.4729816 ,   74.61450064,
                77.75602563,   80.89755587,   84.03909078,   87.18062984,
                90.32217264,   93.46371878,   96.60526795,   99.74681986,
                102.88837425,  106.02993092,  109.17148965,  112.31305028,
                115.45461265,  118.59617663,  121.73774209,  124.87930891,
                128.02087701,  131.16244628,  134.30401664,  137.44558802,
                140.58716035,  143.72873357,  146.87030763,  150.01188246,
                153.15345802,  156.29503427
            ]
    np.testing.assert_almost_equal(
        vals1, vals2, 8, "bessel.j0_root disagrees with reference values")


if __name__ == "__main__":
    test_j0()
    test_j1()
    test_jn()
    test_jv()
    test_kn()
    test_kv()
    test_j0_root()
