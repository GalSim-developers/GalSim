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
import warnings

import galsim
from galsim_test_helpers import *

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    import scipy.special

@timer
def test_j0():
    """Test the bessel.j0 function"""
    x_list = [ 0, 1.01, 0.2, 3.3, 5.9, 77., 1.e-12, 500. ]
    vals1 = [ galsim.bessel.j0(x) for x in x_list ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    vals2 = [ scipy.special.j0(x) for x in x_list ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.j0 disagrees with scipy.special.j0")

    # Scipy is now considered required for tests, so these explicit values are not necessary
    # anymore as a backup, but I'm leaving them in as a regression test.  They should be the same
    # values that scipy returns above.
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

    vals2 = [ scipy.special.j1(x) for x in x_list ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.j1 disagrees with scipy.special.j1")

    # These values are what scipy returns.
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

    vals2 = [ scipy.special.jn(n,x) for n,x in zip(n_list,x_list) ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.jn disagrees with scipy.special.jn")

    # These values are what scipy returns.
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

    vals2 = [ scipy.special.jv(v,x) for v,x in zip(v_list,x_list) ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.jv disagrees with scipy.special.jv")

    # These values are what scipy returns.
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

    vals2 = [ scipy.special.yn(n,x) for n,x in zip(n_list,x_list) ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.yn disagrees with scipy.special.yn")

    # These values are what scipy returns.
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

    vals2 = [ scipy.special.yv(v,x) for v,x in zip(v_list,x_list) ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.yv disagrees with scipy.special.yv")

    # These values are what scipy returns.
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

    vals2 = [ scipy.special.iv(n,x) for n,x in zip(n_list,x_list) ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.iv disagrees with scipy.special.iv")

    # These values are what scipy returns.
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

    vals2 = [ scipy.special.iv(v,x) for v,x in zip(v_list,x_list) ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.iv disagrees with scipy.special.iv")

    # These values are what scipy returns.
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

    vals2 = [ scipy.special.kn(n,x) for n,x in zip(n_list,x_list) ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.kn disagrees with scipy.special.kn")

    # These values are what scipy returns.
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

    vals2 = [ scipy.special.kv(v,x) for v,x in zip(v_list,x_list) ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.kv disagrees with scipy.special.kv")

    # These values are what scipy returns.
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
def test_jv_root():
    """Test the bessel.j0_root and jv_root functions"""
    # Our version uses tabulated values up to 40, so a useful test of the extrapolation
    # requires this to have more than 40 items.
    vals1 = [ galsim.bessel.j0_root(s) for s in range(1,51) ]
    print('vals1 = ',vals1)

    vals2 = scipy.special.jn_zeros(0,50)
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.j0_root disagrees with scipy.special.jn_zeros")

    vals3 = [ galsim.bessel.jv_root(0,s) for s in range(1,51) ]
    np.testing.assert_allclose(
        vals3, vals2, rtol=1.e-10,
        err_msg="bessel.jv_root disagrees with scipy.special.jn_zeros with n=0")

    for v in vals1:
        np.testing.assert_allclose(galsim.bessel.j0(v), 0., atol=1.e-12)
        np.testing.assert_allclose(scipy.special.j0(v), 0., atol=1.e-12)

    for nu in [0, 1, 7, 13]:
        print('nu = ',nu)
        vals1 = [ galsim.bessel.jv_root(nu,s) for s in range(1,51) ]
        print('vals1 = ',vals1)
        vals2 = scipy.special.jn_zeros(nu,50)
        print('vals2 = ',vals2)
        np.testing.assert_allclose(
            vals1, vals2, rtol=1.e-10,
            err_msg="bessel.jv_root disagrees with scipy.special.jn_zeros with n=%f"%nu)
        for v in vals1:
            np.testing.assert_allclose(galsim.bessel.jv(nu,v), 0., atol=1.e-12)
            np.testing.assert_allclose(scipy.special.jv(nu,v), 0., atol=1.e-12)

    # We can also compute zeros of non-integral order, although scipy cannot.
    # So we can't compare with scipy's answers here.  Just check that they really are zeros.
    for nu in [0.5, 0.001, 1.39, 7.3, 13.9, 0.183]:
        print('nu = ',nu)
        vals1 = [ galsim.bessel.jv_root(nu,s) for s in range(1,51) ]
        print('vals1 = ',vals1)
        for v in vals1:
            np.testing.assert_allclose(galsim.bessel.jv(nu,v), 0., atol=1.e-12)
            np.testing.assert_allclose(scipy.special.jv(nu,v), 0., atol=1.e-12)

    with assert_raises(RuntimeError):
        galsim.bessel.jv_root(-0.3,1)
    with assert_raises(RuntimeError):
        galsim.bessel.jv_root(0.3,-1)

@timer
def test_gammainc():
    """Test the bessel.gammainc function"""
    a_list = [ 3.3, 4, 1.9, 0, 0.2, 4.7, 39.8, 2.3 ]
    x_list = [ 1, 2.01, 0.2, 3.3, 15.9, 1.e-12, 3.4, 19.1 ]
    vals1 = [ galsim.bessel.gammainc(a,x) for a,x in zip(a_list,x_list) ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    vals2 = [ scipy.special.gammainc(a,x) for a,x in zip(a_list,x_list) ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.gammainc disagrees with scipy.special.gammainc")

    # These values are what scipy returns.
    vals2 = [   0.05336162361853338,
                0.14468550606481548,
                0.022580514695946612,
                1.0,
                0.9999999971717825,
                5.489041152199198e-59,
                1.3292187146528721e-28,
                0.9999997850330204
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.gammainc disagrees with reference values")

@timer
def test_sinc():
    """Test the bessel.sinc function"""
    x_list = [ 0, 1, -1, 2.01, 0.2, 3.3, 15.9, -7.7, 1.e-12, 500., 3.4, 19.1 ]
    vals1 = [ galsim.bessel.sinc(x) for x in x_list ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    vals2 = [ scipy.special.sinc(x) for x in x_list ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.sinc disagrees with scipy.special.sinc")

    # These values are what scipy returns.
    vals2 = [   1.0,
                3.8981718325193755e-17,
                3.8981718325193755e-17,
                0.004974306043335856,
                0.935489283788639,
                -0.07803579012128542,
                -0.006186362535116209,
                -0.03344391005197945,
                1.0,
                -1.0231009598277845e-16,
                -0.0890384386636067,
                -0.005149903890489398
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.sinc disagrees with reference values")

@timer
def test_si():
    """Test the bessel.si function"""
    x_list = [ 0, 1, -1, 2, 3.99, 4.0, 4.01, 0.2, 3.3, 15.9, 18, 19, 20, -7.7, 1.e-12, 500. ]
    vals1 = [ galsim.bessel.si(x) for x in x_list ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    vals2 = [ scipy.special.sici(x)[0] for x in x_list ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.si disagrees with scipy.special.sici")

    # These values are what scipy returns.
    vals2 = [   0.0,
                0.9460830703671831,
                -0.9460830703671831,
                1.605412976802695,
                1.7600892984314866,
                1.758203138949053,
                1.7563053683733343,
                0.19955608852623385,
                1.8480807827952113,
                1.6328040281824159,
                1.5366080968611855,
                1.518630031769364,
                1.5482417010434397,
                -1.5361092381286596,
                1e-12,
                1.5725658822431687
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.si disagrees with reference values")


@timer
def test_ci():
    """Test the bessel.ci function"""
    x_list = [ 0, 1, -1, 2, 3.99, 4.0, 4.01, 0.2, 3.3, 15.9, 18, 19, 20, -7.7, 1.e-12, 500. ]
    vals1 = [ galsim.bessel.ci(x) for x in x_list ]
    print('x = ',x_list)
    print('vals1 = ',vals1)

    vals2 = [ scipy.special.sici(x)[1] for x in x_list ]
    print('vals2 = ',vals2)
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.ci disagrees with scipy.special.sici")

    # These values are what scipy returns.
    vals2 = [   -np.inf,
                0.33740392290096816,
                0.33740392290096816,
                0.422980828774865,
                -0.13933609432530814,
                -0.1409816978869305,
                -0.14260429630144628,
                -1.0422055956727818,
                0.02467828460795851,
                -0.008115782282273786,
                -0.04347510299950101,
                0.0051503710084261295,
                0.044419820845353314,
                0.1222458319118463,
                -27.053805451027014,
                -0.0009320008144042904
            ]
    np.testing.assert_allclose(
        vals1, vals2, rtol=1.e-10, err_msg="bessel.ci disagrees with reference values")



if __name__ == "__main__":
    runtests(__file__)
