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
import warnings

import galsim
from galsim_test_helpers import *

@timer
def test_j0():
    """Test the bessel.j0 function"""
    x_list = [ 0, 1.01, 0.2, 3.3, 5.9, 77., 1.e-12, 500. ]
    vals1 = [ galsim.bessel.j0(x) for x in x_list ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.j0(x) for x in x_list ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.j0 disagrees with scipy.special.j0")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of j0.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   1.0,
                0.76078097763218844,
                0.99002497223957631,
                -0.34429626039888467,
                0.12203335459282282,
                0.062379777089647245,
                1.0,
                -0.034100556880731728,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.j0 disagrees with reference values")


@timer
def test_j1():
    """Test the bessel.j1 function"""
    x_list = [ 0, 1.01, 0.2, 3.3, 5.9, 77., 1.e-12, 500. ]
    vals1 = [ galsim.bessel.j1(x) for x in x_list ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.j1(x) for x in x_list ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.j1 disagrees with scipy.special.j1")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of j1.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   0.0,
                0.4432857612090717,
                0.099500832639236036,
                0.22066345298524112,
                -0.29514244472901613,
                0.066560642470571682,
                5.0000000000000009e-13,
                0.010472613470372989,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.j1 disagrees with reference values")


@timer
def test_jn():
    """Test the bessel.jn function"""
    n_list = [ 3, 4, 1, 0, 9, -7, 4, 300, 39, 2 ]
    x_list = [ 0, 1.01, 0.2, 3.3, 15.9, 77., 1.e-12, 500., 3.4, 19.1 ]
    vals1 = [ galsim.bessel.jn(n,x) for n,x in zip(n_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.jn(n,x) for n,x in zip(n_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.jn disagrees with scipy.special.jn")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of jn.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   0.0,
                0.0025745895535573995,
                0.099500832639236036,
                -0.34429626039888467,
                -0.19886654556845093,
                0.082526868218916541,
                2.6041666666666486e-51,
                -0.0029540008506870503,
                4.4311593707049111e-38,
                -0.16584939097852849,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.jn disagrees with reference values")


@timer
def test_jv():
    """Test the bessel.jv function"""
    v_list = [ 3.3, 4, 1.9, 0, 0.2, -7.1, 4.7, 300.9, 39.8, 2.3 ]
    x_list = [ 0, 1.01, 0.2, 3.3, 15.9, 77., 1.e-12, 500., 3.4, 19.1 ]
    vals1 = [ galsim.bessel.jv(v,x) for v,x in zip(v_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.jv(v,x) for v,x in zip(v_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.jv disagrees with scipy.special.jv")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of jv.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   0.0,
                0.0025745895535573995,
                0.0068656051839294848,
                -0.34429626039888467,
                -0.12213323547609471,
                0.087784805831697565,
                2.1118132341112502e-60,
                0.027491061612400482,
                3.5539536207995954e-39,
                -0.11753152687355309,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.jv disagrees with reference values")


@timer
def test_yn():
    """Test the bessel.yn function"""
    n_list = [ 3, 4, 1, 0, 9, -7, 4, 300, 39, 2 ]
    x_list = [ 0.1, 1.01, 0.2, 3.3, 15.9, 77., 1.e-12, 500., 3.4, 19.1 ]
    vals1 = [ galsim.bessel.yn(n,x) for n,x in zip(n_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.yn(n,x) for n,x in zip(n_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.yn disagrees with scipy.special.yn")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of yn.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   -5099.3323786129049,
                -32.036214020138011,
                -3.3238249881118471,
                0.26909199505453402,
                -0.094117150977372527,
                -0.038617322619578592,
                -3.0557749073643908e+49,
                0.039784618934117735,
                -1.8489533038779474e+35,
                0.077438881119458292,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.yn disagrees with reference values")


@timer
def test_yv():
    """Test the bessel.yv function"""
    v_list = [ 3.3, 4, 1.9, 0, 0.2, -7.1, 4.7, 300.9, 39.8, 2.3 ]
    x_list = [ 0.1, 1.01, 0.2, 3.3, 15.9, 77., 1.e-12, 500., 3.4, 19.1 ]
    vals1 = [ galsim.bessel.yv(v,x) for v,x in zip(v_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.yv(v,x) for v,x in zip(v_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.yv disagrees with scipy.special.yv")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of yv.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   -16804.006307563286,
                -32.036214020138011,
                -24.595386714889109,
                0.26909199505453379,
                0.15844850776917807,
                -0.024429586880444932,
                -3.2069837713267924e+58,
                0.028956519948967349,
                -2.2586390441339445e+36,
                0.14053075843601004,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.yv disagrees with reference values")

@timer
def test_in():
    """Test the bessel.iv function with integer values.  (There is no bessel.in, nor
    scipy.special.in, since in is of course a reserved word.)
    """
    n_list = [ 3, 4, 1, 0, 9, 7, 4, 300, 39, 2 ]
    x_list = [ 0, 2.01, 0.2, 3.3, 15.9, 7.7, 1.e-12, 500., 3.4, 19.1 ]
    vals1 = [ galsim.bessel.iv(n,x) for n,x in zip(n_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.iv(n,x) for n,x in zip(n_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.iv disagrees with scipy.special.iv")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of iv.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   0.0,
                0.051851345436838572,
                0.10050083402812511,
                6.2426304651830256,
                62790.244278205071,
                13.590333620044147,
                2.6041666666666664e-51,
                2.1971573261481075e+177,
                5.1200364123222137e-38,
                16279753.373047709,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.iv disagrees with reference values")



@timer
def test_iv():
    """Test the bessel.iv function"""
    v_list = [ 3.3, 4, 1.9, 0, 0.2, -7.1, 4.7, 300.9, 39.8, 2.3 ]
    x_list = [ 0, 2.01, 0.2, 3.3, 15.9, 7.7, 1.e-12, 500., 3.4, 19.1 ]
    vals1 = [ galsim.bessel.iv(v,x) for v,x in zip(v_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.iv(v,x) for v,x in zip(v_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.iv disagrees with scipy.special.iv")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of iv.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   0.0,
                0.051851345436838572,
                0.0069131178533799404,
                6.2426304651830256,
                809950.48268613976,
                12.476741646263305,
                2.1118132341112364e-60,
                1.315385468943472e+177,
                4.0948398641503255e-39,
                15725726.116463179,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.iv disagrees with reference values")


@timer
def test_kn():
    """Test the bessel.kn function"""
    n_list = [ 3, 4, 1, 0, 9, 7, 4, 300, 39, 2 ]
    x_list = [ 1, 2.01, 0.2, 3.3, 15.9, 7.7, 1.e-12, 500., 3.4, 19.1 ]
    vals1 = [ galsim.bessel.kn(n,x) for n,x in zip(n_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.kn(n,x) for n,x in zip(n_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.kn disagrees with scipy.special.kn")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of kn.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   7.1012628247379448,
                2.1461917781688697,
                4.7759725432204725,
                0.024610632145839577,
                4.3581406072741754e-07,
                0.0035326394473326195,
                4.8e+49,
                3.902737598763313e-181,
                2.494520930845136e+35,
                1.5997755223851229e-09,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.kn disagrees with reference values")


@timer
def test_kv():
    """Test the bessel.kv function"""
    v_list = [ 3.3, 4, 1.9, 0, 0.2, -7.1, 4.7, 300.9, 39.8, 2.3 ]
    x_list = [ 1, 2.01, 0.2, 3.3, 15.9, 7.7, 1.e-12, 500., 3.4, 19.1 ]
    vals1 = [ galsim.bessel.kv(v,x) for v,x in zip(v_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import scipy.special
        vals2 = [ scipy.special.kv(v,x) for v,x in zip(v_list,x_list) ]
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.kv disagrees with scipy.special.kv")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of kv.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   11.898213399340918,
                2.1461917781688693,
                37.787322153212926,
                0.024610632145839324,
                3.8841496808266363e-08,
                0.0038228967734928758,
                5.0375183280909704e+58,
                6.513768858596484e-181,
                3.0568215777387072e+36,
                1.6532345134742898e-09,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.kv disagrees with reference values")


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
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10, err_msg="bessel.j0_root disagrees with scipy.special.jn_zeros")
    except ImportError:
        print('Unable to import scipy.  Skipping scipy tests of j0_root.')

    # These values are what scipy returns.  Check against these, so not require scipy.
    vals2 = [   2.404825557695773, 5.520078110286311, 8.653727912911013,
                11.791534439014281, 14.930917708487787, 18.071063967910924,
                21.21163662987926, 24.352471530749302, 27.493479132040257,
                30.634606468431976, 33.77582021357357, 36.917098353664045,
                40.05842576462824, 43.19979171317673, 46.341188371661815,
                49.482609897397815, 52.624051841115, 55.76551075501998,
                58.90698392608094, 62.04846919022717, 65.18996480020687,
                68.3314693298568, 71.47298160359374, 74.61450064370185,
                77.75602563038805, 80.89755587113763, 84.0390907769382,
                87.18062984364116, 90.32217263721049, 93.46371878194478,
                96.60526795099626, 99.7468198586806, 102.8883742541948,
                106.02993091645162, 109.17148964980538, 112.3130502804949,
                115.45461265366694, 118.59617663087253, 121.73774208795096,
                124.87930891323295, 128.02087700600833, 131.1624462752139,
                134.30401663830546, 137.44558802028428, 140.58716035285428,
                143.72873357368974, 146.87030762579664, 150.01188245695477,
                153.15345801922788, 156.29503426853353,
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.j0_root disagrees with reference values")


if __name__ == "__main__":
    test_j0()
    test_j1()
    test_jn()
    test_jv()
    test_yn()
    test_yv()
    test_in()
    test_iv()
    test_kn()
    test_kv()
    test_j0_root()
