# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
import numpy as np
import os
import sys
import pyfits

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
import galsim.utilities

# Below are a set of tests to make sure that we have achieved consistency in defining shears and
# ellipses using different conventions.  The underlying idea is that in test_SBProfile.py we already
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
q = [0.5, 0.3, 0.1, 0.7]
n_shear = len(q)
beta = [0.5*np.pi, 0.25*np.pi, 0.0*np.pi, np.pi/3.0]
g = [0.333333, 0.538462, 0.818182, 0.176471]
g1 = [-0.33333334, 0.0, 0.81818175, -0.088235296]
g2 = [0.0, 0.53846157, 0.0, 0.15282802]
e = [0.600000, 0.834862, 0.980198, 0.342282]
e1 = [-0.6000000, 0.0, 0.98019803, -0.17114094]
e2 = [0.0, 0.83486235, 0.0, 0.29642480]
eta = [0.693147, 1.20397, 2.30259, 0.356675]
eta1 = [-0.69314718, 0.0, 2.3025851, -0.17833748]
eta2 = [0.0, 1.2039728, 0.0, 0.30888958]
decimal = 5

# some ellipse properties over which we will loop - use the shear values above, and:
mu = [0.0, 0.5, -0.1]
n_mu = len(mu)
x_shift = [0.0, 1.7, -3.0]
y_shift = [-1.3, 0.0, 9.1]
n_shift = len(x_shift)

def funcname():
    import inspect
    return inspect.stack()[1][3]

#### some helper functions
def all_shear_vals(test_shear, index, mult_val = 1.0):
    # this function tests that all values of some Shear object are consistent with the tabulated
    # values, given the appropriate index against which to test, and properly accounting for the
    # fact that SBProfile sometimes has the angle in the range [pi, 2*pi)
    ### note: can only use mult_val = 1, 0, -1
    if mult_val != -1.0 and mult_val != 0.0 and mult_val != 1.0:
        raise ValueError("Cannot put multiplier that is not -1, 0, or 1!")
    rad = test_shear.beta.rad()
    while rad < 0.0:
        rad += np.pi
    vec = [test_shear.g, test_shear.g1, test_shear.g2, test_shear.e, test_shear.e1, test_shear.e2,
           test_shear.eta, test_shear.esq, rad % np.pi]

    test_beta = beta[index]
    if mult_val < 0.0:
        test_beta -= 0.5*np.pi
    while test_beta < 0.0:
        test_beta += np.pi
    test_vec = [np.abs(mult_val)*g[index], mult_val*g1[index], mult_val*g2[index],
                np.abs(mult_val)*e[index], mult_val*e1[index], mult_val*e2[index],
                np.abs(mult_val)*eta[index], mult_val*mult_val*e[index]*e[index], test_beta % np.pi]
    np.testing.assert_array_almost_equal(vec, test_vec, decimal=decimal,
                                         err_msg = "Incorrectly initialized Shear")

def add_distortions(d1, d2, d1app, d2app):
    # add the distortions
    denom = 1.0 + d1*d1app + d2*d2app
    dapp_sq = d1app**2 + d2app**2
    d1tot = (d1 + d1app + d2app/dapp_sq*(1.0 - np.sqrt(1.0-dapp_sq))*(d2*d1app - d1*d2app))/denom
    d2tot = (d2 + d2app + d1app/dapp_sq*(1.0 - np.sqrt(1.0-dapp_sq))*(d1*d2app - d2*d1app))/denom
    return d1tot, d2tot

def test_shear_initialization():
    """Test that Shears can be initialized in a variety of ways and get the expected results."""
    import time
    t1 = time.time()
    # first make an empty Shear and make sure that it has zeros in the right places
    s = galsim.Shear()
    vec = [s.g, s.g1, s.g2, s.e, s.e1, s.e2, s.eta, s.esq]
    vec_ideal = np.zeros(len(vec))
    np.testing.assert_array_almost_equal(vec, vec_ideal, decimal = decimal,
                                         err_msg = "Incorrectly initialized empty shear")
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
        # initialize with a wrapped C++ Shear object
        s2 = galsim.Shear(s._shear)
        all_shear_vals(s2, ind)
    # finally check some examples of invalid initializations for Shear
    try:
        np.testing.assert_raises(TypeError,galsim.Shear,0.3)
        np.testing.assert_raises(TypeError,galsim.Shear,g1=0.3,e2=0.2)
        np.testing.assert_raises(TypeError,galsim.Shear,eta1=0.3,beta=0.*galsim.degrees)
        np.testing.assert_raises(TypeError,galsim.Shear,q=0.3)
        np.testing.assert_raises(ValueError,galsim.Shear,q=1.3,beta=0.*galsim.degrees)
        np.testing.assert_raises(ValueError,galsim.Shear,g1=0.9,g2=0.6)
        np.testing.assert_raises(ValueError,galsim.Shear,e=-1.3,beta=0.*galsim.radians)
        np.testing.assert_raises(ValueError,galsim.Shear,e=1.3,beta=0.*galsim.radians)
        np.testing.assert_raises(TypeError,galsim.Shear,randomkwarg=0.1)
        np.testing.assert_raises(TypeError,galsim.Shear,g1=0.1,randomkwarg=0.1)
        np.testing.assert_raises(TypeError,galsim.Shear,g1=0.1,e1=0.1)
        np.testing.assert_raises(TypeError,galsim.Shear,g1=0.1,e=0.1)
        np.testing.assert_raises(TypeError,galsim.Shear,g1=0.1,g=0.1)
        np.testing.assert_raises(TypeError,galsim.Shear,beta=45.0*galsim.degrees)
        np.testing.assert_raises(TypeError,galsim.Shear,beta=45.0*galsim.degrees,g=0.3,eta=0.1)
        np.testing.assert_raises(TypeError,galsim.Shear,beta=45.0,g=0.3)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shear_methods():
    """Test that the most commonly-used methods of the Shear class give the expected results."""
    import time
    t1 = time.time()
    for ind in range(n_shear):
        # check setE1E2
        s = galsim.Shear()
        s.setE1E2(e1[ind], e2[ind])
        all_shear_vals(s, ind)
        # check setEBeta
        s = galsim.Shear()
        s.setEBeta(e[ind], beta[ind]*galsim.radians)
        all_shear_vals(s, ind)
        # check setEta1Eta2
        s = galsim.Shear()
        s.setEta1Eta2(eta1[ind], eta2[ind])
        all_shear_vals(s, ind)
        # check setEtaBeta
        s = galsim.Shear()
        s.setEtaBeta(eta[ind], beta[ind]*galsim.radians)
        all_shear_vals(s, ind)
        # check setG1G2
        s = galsim.Shear()
        s.setG1G2(g1[ind], g2[ind])
        all_shear_vals(s, ind)
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

    # note: we don't have to check all the getWhatever methods because they were implicitly checked
    # in test_shear_initialization, where we checked the values directly

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_shear_initialization()
    test_ellipse_initialization()
    test_shear_methods()
