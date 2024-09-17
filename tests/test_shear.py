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


# Below are a set of tests to make sure that we have achieved consistency in defining shears and
# ellipses using different conventions.  The underlying idea is that in other files we already
# have plenty of tests to ensure that a given Shear can be properly applied and gives the
# expected result.  So here, we just work at the level of Shears that we've defined,
# and make sure that they have the properties we expect given the values that were used to
# initialize them.  For that, we have some sets of fiducial shears/dilations/shifts for which
# calculations were done independently (e.g., to get what is eta given the value of g).  We go over
# the various way to initialize the shears, and make sure that their different values are properly
# set.  We also test the methods of the python Shear classes to make sure that they give
# the expected results.

##### set up necessary info for tests
# a few shear values over which we will loop so we can check them all
# note: Rachel started with these q and beta, and then calculated all the other numbers in IDL using
# the standard formulae
q = [1.0, 0.5, 0.3, 0.1, 0.7, 0.9, 0.99, 1.-8.75e-5]
n_shear = len(q)
beta = [0.0, 0.5*np.pi, 0.25*np.pi, 0.0*np.pi, np.pi/3.0, np.pi, -0.25*np.pi, -0.5*np.pi]
g = [0.0, 0.333333, 0.538462, 0.818182, 0.176471, 0.05263157897, 0.005025125626, 4.375191415e-5 ]
g1 = [0.0, -0.33333334, 0.0, 0.81818175, -0.088235296, 0.05263157897, 0.0, -4.375191415e-5 ]
g2 = [0.0, 0.0, 0.53846157, 0.0, 0.15282802, 0.0, -0.005025125626, 0.0 ]
e = [0.0, 0.600000, 0.834862, 0.980198, 0.342282, 0.1049723757, 0.01004999747, 8.750382812e-5 ]
e1 = [0.0, -0.6000000, 0.0, 0.98019803, -0.17114094, 0.1049723757, 0.0, -8.750382812e-5 ]
e2 = [0.0, 0.0, 0.83486235, 0.0, 0.29642480, 0.0, -0.01004999747, 0.0 ]
eta = [0.0, 0.693147, 1.20397, 2.30259, 0.356675, 0.1053605157, 0.01005033585, 8.750382835e-5 ]
eta1 = [0.0, -0.69314718, 0.0, 2.3025851, -0.17833748, 0.1053605157, 0.0, -8.750382835e-5 ]
eta2 = [0.0, 0.0, 1.2039728, 0.0, 0.30888958, 0.0, -0.01005033585, 0.0 ]
decimal = 5

#### some helper functions
def all_shear_vals(test_shear, index, mult_val = 1.0):
    print('test_shear = ',repr(test_shear))
    # this function tests that all values of some Shear object are consistent with the tabulated
    # values, given the appropriate index against which to test, and properly accounting for the
    # fact that sometimes the angle is in the range [pi, 2*pi)
    ### note: can only use mult_val = 1, 0, -1
    if mult_val != -1.0 and mult_val != 0.0 and mult_val != 1.0:
        raise ValueError("Cannot put multiplier that is not -1, 0, or 1!")
    beta_rad = test_shear.beta.rad
    while beta_rad < 0.0:
        beta_rad += np.pi

    test_beta = beta[index]
    if mult_val < 0.0:
        test_beta -= 0.5*np.pi
    while test_beta < 0.0:
        test_beta += np.pi
    # Special, if g == 0 exactly, beta is undefined, so just set it to zero.
    if test_shear.g == 0.0:
        test_beta = beta_rad = 0.

    vec = [test_shear.g, test_shear.g1, test_shear.g2, test_shear.e, test_shear.e1, test_shear.e2,
           test_shear.eta, test_shear.eta1, test_shear.eta2, test_shear.esq, test_shear.q,
           beta_rad % np.pi]
    test_vec = [np.abs(mult_val)*g[index], mult_val*g1[index], mult_val*g2[index],
                np.abs(mult_val)*e[index], mult_val*e1[index], mult_val*e2[index],
                np.abs(mult_val)*eta[index], mult_val*eta1[index], mult_val*eta2[index],
                mult_val*mult_val*e[index]*e[index], q[index], test_beta % np.pi]
    np.testing.assert_array_almost_equal(vec, test_vec, decimal=decimal,
                                         err_msg = "Incorrectly initialized Shear")
    if index == n_shear-1:
        # On the last one with values < 1.e-4, multiply everything by 1.e4 and check again.
        vec = [1.e4 * v for v in vec[:-2]]  # don't include q or beta now.
        test_vec = [1.e4 * v for v in test_vec[:-2]]
        np.testing.assert_array_almost_equal(vec, test_vec, decimal=decimal,
                                             err_msg = "Incorrectly initialized Shear")
    # Test that the utiltiy function g1g2_to_e1e2 is equivalent to the Shear calculation.
    test_e1, test_e2 = galsim.utilities.g1g2_to_e1e2(test_shear.g1, test_shear.g2)
    np.testing.assert_almost_equal(test_e1, test_shear.e1, err_msg="Incorrect e1 calculation")
    np.testing.assert_almost_equal(test_e2, test_shear.e2, err_msg="Incorrect e2 calculation")


def add_distortions(d1, d2, d1app, d2app):
    # add the distortions
    denom = 1.0 + d1*d1app + d2*d2app
    dapp_sq = d1app**2 + d2app**2
    if dapp_sq == 0:
        return d1, d2
    else:
        factor = (1.0 - np.sqrt(1.0-dapp_sq)) * (d2*d1app - d1*d2app) / dapp_sq
        d1tot = (d1 + d1app + d2app * factor)/denom
        d2tot = (d2 + d2app - d1app * factor)/denom
        return d1tot, d2tot


@timer
def test_shear_initialization():
    """Test that Shears can be initialized in a variety of ways and get the expected results."""
    # first make an empty Shear and make sure that it has zeros in the right places
    s = galsim.Shear()
    vec = [s.g, s.g1, s.g2, s.e, s.e1, s.e2, s.eta, s.eta1, s.eta2, s.esq]
    vec_ideal = np.zeros(len(vec))
    np.testing.assert_array_almost_equal(vec, vec_ideal, decimal = decimal,
                                         err_msg = "Incorrectly initialized empty shear")
    np.testing.assert_equal(s.q, 1.)
    # now loop over shear values and ways of initializing
    for ind in range(n_shear):
        # initialize with reduced shear components
        s = galsim.Shear(g1 = g1[ind], g2 = g2[ind])
        all_shear_vals(s, ind)
        if g1[ind] == 0.0:
            s = galsim.Shear(g2 = g2[ind])
            all_shear_vals(s, ind)
        if g2[ind] == 0.0:
            s = galsim.Shear(g1 = g1[ind])
            all_shear_vals(s, ind)
        # initialize with distortion components
        s = galsim.Shear(e1 = e1[ind], e2 = e2[ind])
        all_shear_vals(s, ind)
        if e1[ind] == 0.0:
            s = galsim.Shear(e2 = e2[ind])
            all_shear_vals(s, ind)
        if e2[ind] == 0.0:
            s = galsim.Shear(e1 = e1[ind])
            all_shear_vals(s, ind)
        # initialize with conformal shear components
        s = galsim.Shear(eta1 = eta1[ind], eta2 = eta2[ind])
        all_shear_vals(s, ind)
        if eta1[ind] == 0.0:
            s = galsim.Shear(eta2 = eta2[ind])
            all_shear_vals(s, ind)
        if eta2[ind] == 0.0:
            s = galsim.Shear(eta1 = eta1[ind])
            all_shear_vals(s, ind)
        # initialize with axis ratio and position angle
        s = galsim.Shear(q = q[ind], beta = beta[ind]*galsim.radians)
        all_shear_vals(s, ind)
        # initialize with reduced shear and position angle
        s = galsim.Shear(g = g[ind], beta = beta[ind]*galsim.radians)
        all_shear_vals(s, ind)
        # initialize with distortion and position angle
        s = galsim.Shear(e = e[ind], beta = beta[ind]*galsim.radians)
        all_shear_vals(s, ind)
        # initialize with conformal shear and position angle
        s = galsim.Shear(eta = eta[ind], beta = beta[ind]*galsim.radians)
        all_shear_vals(s, ind)
        # initialize with a complex number g1 + 1j * g2
        s = galsim.Shear(g1[ind] + 1j * g2[ind])
        all_shear_vals(s, ind)
        s = galsim._Shear(g1[ind] + 1j * g2[ind])
        all_shear_vals(s, ind)
        # which should also be the value of s.shear
        s2 = galsim.Shear(s.shear)
        all_shear_vals(s2, ind)

        # Check picklability
        check_pickle(s)

    # finally check some examples of invalid initializations for Shear
    assert_raises(TypeError,galsim.Shear,0.3)
    assert_raises(TypeError,galsim.Shear,0.3,0.3)
    assert_raises(TypeError,galsim.Shear,g1=0.3,e2=0.2)
    assert_raises(TypeError,galsim.Shear,eta1=0.3,beta=0.*galsim.degrees)
    assert_raises(TypeError,galsim.Shear,q=0.3)
    assert_raises(galsim.GalSimRangeError,galsim.Shear,q=1.3,beta=0.*galsim.degrees)
    assert_raises(galsim.GalSimRangeError,galsim.Shear,g1=0.9,g2=0.6)
    assert_raises(galsim.GalSimRangeError,galsim.Shear,e=-1.3,beta=0.*galsim.radians)
    assert_raises(galsim.GalSimRangeError,galsim.Shear,e=1.3,beta=0.*galsim.radians)
    assert_raises(galsim.GalSimRangeError,galsim.Shear,e1=0.7,e2=0.9)
    assert_raises(TypeError,galsim.Shear,g=0.5)
    assert_raises(TypeError,galsim.Shear,e=0.5)
    assert_raises(TypeError,galsim.Shear,eta=0.5)
    assert_raises(galsim.GalSimRangeError,galsim.Shear,eta=-0.5,beta=0.*galsim.radians)
    assert_raises(galsim.GalSimRangeError,galsim.Shear,g=1.3,beta=0.*galsim.radians)
    assert_raises(galsim.GalSimRangeError,galsim.Shear,g=-0.3,beta=0.*galsim.radians)
    assert_raises(TypeError,galsim.Shear,e=0.3,beta=0.)
    assert_raises(TypeError,galsim.Shear,eta=0.3,beta=0.)
    assert_raises(TypeError,galsim.Shear,randomkwarg=0.1)
    assert_raises(TypeError,galsim.Shear,g1=0.1,randomkwarg=0.1)
    assert_raises(TypeError,galsim.Shear,g1=0.1,e1=0.1)
    assert_raises(TypeError,galsim.Shear,g1=0.1,e=0.1)
    assert_raises(TypeError,galsim.Shear,g1=0.1,g=0.1)
    assert_raises(TypeError,galsim.Shear,beta=45.0*galsim.degrees)
    assert_raises(TypeError,galsim.Shear,beta=45.0*galsim.degrees,g=0.3,eta=0.1)
    assert_raises(TypeError,galsim.Shear,beta=45.0,g=0.3)
    assert_raises(TypeError,galsim.Shear,q=0.1,beta=0.)


@timer
def test_shear_methods():
    """Test that the most commonly-used methods of the Shear class give the expected results."""
    for ind in range(n_shear):
        s = galsim.Shear(e1=e1[ind], e2=e2[ind])
        # check negation
        s2 = -s
        all_shear_vals(s2, ind, mult_val = -1.0)
        # check addition
        s2 = s + s
        exp_e1, exp_e2 = add_distortions(s.e1, s.e2, s.e1, s.e2)
        np.testing.assert_array_almost_equal([s2.e1, s2.e2], [exp_e1, exp_e2], decimal=decimal,
                                             err_msg = "Failed to properly add distortions")
        # check subtraction
        s3 = s - s2
        exp_e1, exp_e2 = add_distortions(s.e1, s.e2, -1.0*s2.e1, -1.0*s2.e2)
        np.testing.assert_array_almost_equal([s3.e1, s3.e2], [exp_e1, exp_e2], decimal=decimal,
                                             err_msg = "Failed to properly subtract distortions")
        # check +=
        savee1 = s.e1
        savee2 = s.e2
        s += s2
        exp_e1, exp_e2 = add_distortions(savee1, savee2, s2.e1, s2.e2)
        np.testing.assert_array_almost_equal([s.e1, s.e2], [exp_e1, exp_e2], decimal=decimal,
                                             err_msg = "Failed to properly += distortions")
        # check -=
        savee1 = s.e1
        savee2 = s.e2
        s -= s
        exp_e1, exp_e2 = add_distortions(savee1, savee2, -1.0*savee1, -1.0*savee2)
        np.testing.assert_array_almost_equal([s.e1, s.e2], [exp_e1, exp_e2], decimal=decimal,
                                             err_msg = "Failed to properly -= distortions")

        # check ==
        s = galsim.Shear(g1 = g1[ind], g2 = g2[ind])
        s2 = galsim.Shear(g1 = g1[ind], g2 = g2[ind])
        np.testing.assert_equal(s == s2, True, err_msg = "Failed to check for equality")
        # check !=
        np.testing.assert_equal(s != s2, False, err_msg = "Failed to check for equality")


@timer
def test_shear_matrix():
    """Test that the shear matrix is calculated correctly.
    """
    for ind in range(n_shear):
        s1 = galsim.Shear(g1=g1[ind], g2=g2[ind])

        true_m1 = np.array([[ 1.+g1[ind],  g2[ind]  ],
                            [   g2[ind], 1.-g1[ind] ]]) / np.sqrt(1.-g1[ind]**2-g2[ind]**2)
        m1 = s1.getMatrix()

        np.testing.assert_array_almost_equal(m1, true_m1, decimal=12,
                                             err_msg="getMatrix returned wrong matrix")

        for ind2 in range(n_shear):
            s2 = galsim.Shear(g1=g1[ind2], g2=g2[ind2])
            m2 = s2.getMatrix()

            s3 = s1 + s2
            m3 = s3.getMatrix()

            theta = s1.rotationWith(s2)
            r = np.array([[  np.cos(theta), -np.sin(theta) ],
                          [  np.sin(theta),  np.cos(theta) ]])
            np.testing.assert_array_almost_equal(m3.dot(r), m1.dot(m2), decimal=12,
                                                 err_msg="rotationWith returned wrong angle")


if __name__ == "__main__":
    runtests(__file__)
