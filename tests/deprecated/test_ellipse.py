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

def all_ellipse_vals(test_ellipse, ind_shear, ind_mu, ind_shift, check_shear=1.0, check_mu=1.0,
                     check_shift = 1.0):
    # this function tests that the various numbers stored in some Ellipse object are consistent with
    # the tabulated values that we expect, given indices against which to test
    vec = [test_ellipse.getS().g1, test_ellipse.getS().g2, test_ellipse.getMu(),
           test_ellipse.getX0().x, test_ellipse.getX0().y]
    test_vec = [check_shear*g1[ind_shear], check_shear*g2[ind_shear], check_mu*mu[ind_mu],
                check_shift*x_shift[ind_shift], check_shift*y_shift[ind_shift]]
    np.testing.assert_array_almost_equal(vec, test_vec, decimal=decimal,
                                         err_msg = "Incorrectly initialized Ellipse")

def test_ellipse_initialization():
    """Test that Ellipses can be initialized in a variety of ways and get the expected results."""
    import time
    t1 = time.time()
    # make an empty Ellipse and make sure everything is zero
    e = galsim.deprecated.Ellipse()
    vec = [e.getS().g1, e.getS().g2, e.getMu(), e.getX0().x, e.getX0().y]
    vec_ideal = [0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_array_almost_equal(vec, vec_ideal, decimal = decimal,
                                         err_msg = "Incorrectly initialized empty ellipse")

    # then loop over the ways we can initialize, with all things initialized and with only those
    # that are non-zero initialized, using args, kwargs in various ways
    for ind_shear in range(n_shear):
        for ind_mu in range(n_mu):
            for ind_shift in range(n_shift):
                # initialize with all of shear, mu, shift
                ## using a Shear, either as arg or kwarg
                ## using a mu, either as arg or kwarg
                ## using a shift, either as Position arg or kwargs
                ## using the various ways of making a Shear passed through as kwargs
                s = galsim.Shear(g1 = g1[ind_shear], g2 = g2[ind_shear])
                p = galsim.PositionD(x_shift[ind_shift], y_shift[ind_shift])
                e = galsim.deprecated.Ellipse(s, mu[ind_mu], p)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift)
                e = galsim.deprecated.Ellipse(p, shear=s, mu=mu[ind_mu])
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift)
                e = galsim.deprecated.Ellipse(s, mu[ind_mu], x_shift=p.x, y_shift=p.y)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift)
                e = galsim.deprecated.Ellipse(shear=s, mu=mu[ind_mu], x_shift=p.x, y_shift=p.y)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift)
                e = galsim.deprecated.Ellipse(q = q[ind_shear], 
                                              beta = beta[ind_shear]*galsim.radians,
                                              mu=mu[ind_mu], x_shift = p.x, y_shift = p.y)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift)

                # now initialize with only 2 of the 3 and make sure the other is zero
                e = galsim.deprecated.Ellipse(mu[ind_mu], p)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shear=0.0)
                e = galsim.deprecated.Ellipse(p, mu=mu[ind_mu])
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shear=0.0)
                e = galsim.deprecated.Ellipse(mu[ind_mu], x_shift = p.x, y_shift = p.y)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shear=0.0)
                e = galsim.deprecated.Ellipse(mu = mu[ind_mu], x_shift = p.x, y_shift = p.y)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shear=0.0)
                e = galsim.deprecated.Ellipse(s, p)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_mu=0.0)
                e = galsim.deprecated.Ellipse(p, shear=s)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_mu=0.0)
                e = galsim.deprecated.Ellipse(s, x_shift = p.x, y_shift = p.y)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_mu=0.0)
                e = galsim.deprecated.Ellipse(shear=s, x_shift = p.x, y_shift = p.y)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_mu=0.0)
                e = galsim.deprecated.Ellipse(s, mu[ind_mu])
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shift=0.0)
                e = galsim.deprecated.Ellipse(s, mu=mu[ind_mu])
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shift=0.0)
                e = galsim.deprecated.Ellipse(mu[ind_mu], shear=s)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shift=0.0)
                e = galsim.deprecated.Ellipse(shear=s, mu=mu[ind_mu])
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shift=0.0)

                # now initialize with only 1 of the 3 and make sure the other is zero
                e = galsim.deprecated.Ellipse(s)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_mu=0.0, check_shift=0.0)
                e = galsim.deprecated.Ellipse(shear=s)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_mu=0.0, check_shift=0.0)
                e = galsim.deprecated.Ellipse(eta1=eta1[ind_shear], eta2=eta2[ind_shear])
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_mu=0.0, check_shift=0.0)
                e = galsim.deprecated.Ellipse(mu[ind_mu])
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shear=0.0, check_shift=0.0)
                e = galsim.deprecated.Ellipse(mu=mu[ind_mu])
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_shear=0.0, check_shift=0.0)
                e = galsim.deprecated.Ellipse(p)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_mu=0.0, check_shear=0.0)
                e = galsim.deprecated.Ellipse(x_shift = p.x, y_shift = p.y)
                all_ellipse_vals(e, ind_shear, ind_mu, ind_shift, check_mu=0.0, check_shear=0.0)
    # check for some cases that should fail
    s = galsim.Shear()
    try:
        np.testing.assert_raises(TypeError, galsim.deprecated.Ellipse, s, g2=0.3)
        np.testing.assert_raises(TypeError, galsim.deprecated.Ellipse, shear=s, x_shift=1, g1=0.2)
        np.testing.assert_raises(TypeError, galsim.deprecated.Ellipse, s,
                                 shift=galsim.PositionD(), x_shift=0.1)
        np.testing.assert_raises(TypeError, galsim.deprecated.Ellipse, s, s)
        np.testing.assert_raises(TypeError, galsim.deprecated.Ellipse, g1=0.1, randomkwarg=0.7)
        np.testing.assert_raises(TypeError, galsim.deprecated.Ellipse, shear=0.1)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_ellipse_initialization()
