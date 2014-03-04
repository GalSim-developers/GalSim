# Copyright 2012-2014 The GalSim developers:
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

import os

import numpy as np

import galsim

path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))

def test_SED_add():
    """Check that SEDs add like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = galsim.SED(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]),
                   flux_type='fphotons')
    c = a+b
    np.testing.assert_almost_equal(c.blue_limit, 1.1, 10,
                                   err_msg="Found wrong blue limit in SED.__add__")
    np.testing.assert_almost_equal(c.red_limit, 5.0, 10,
                                   err_msg="Found wrong red limit in SED.__add__")
    np.testing.assert_almost_equal(c(3.0), 3.3 + 3.33, 10,
                                   err_msg="Wrong sum in SED.__add__")
    np.testing.assert_almost_equal(c(1.1), a(1.1)+1.11, 10,
                                   err_msg="Wrong sum in SED.__add__")
    np.testing.assert_almost_equal(c(5.0), 5.5+b(5.0), 10,
                                   err_msg="Wrong sum in SED.__add__")

def test_SED_sub():
    """Check that SEDs subtract like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = galsim.SED(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]),
                   flux_type='fphotons')
    c = a-b
    np.testing.assert_almost_equal(c.blue_limit, 1.1, 10,
                                   err_msg="Found wrong blue limit in SED.__add__")
    np.testing.assert_almost_equal(c.red_limit, 5.0, 10,
                                   err_msg="Found wrong red limit in SED.__add__")
    np.testing.assert_almost_equal(c(3.0), 3.3 - 3.33, 10,
                                   err_msg="Wrong sum in SED.__sub__")
    np.testing.assert_almost_equal(c(1.1), a(1.1)-1.11, 10,
                                   err_msg="Wrong sum in SED.__add__")
    np.testing.assert_almost_equal(c(5.0), 5.5-b(5.0), 10,
                                   err_msg="Wrong sum in SED.__add__")

def test_SED_mul():
    """Check that SEDs multiply like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = lambda w: w**2
    # SED multiplied by function
    c = a*b
    np.testing.assert_almost_equal(c(3.0), 3.3 * 3**2, 10,
                                   err_msg="Found wrong value in SED.__mul__")
    # function multiplied by SED
    c = b*a
    np.testing.assert_almost_equal(c(3.0), 3.3 * 3**2, 10,
                                   err_msg="Found wrong value in SED.__rmul__")
    # SED multiplied by scalar
    d = c*4.2
    np.testing.assert_almost_equal(d(3.0), 3.3 * 3**2 * 4.2, 10,
                                   err_msg="Found wrong value in SED.__mul__")
    # assignment multiplication
    d *= 2
    np.testing.assert_almost_equal(d(3.0), 3.3 * 3**2 * 4.2 * 2, 10,
                                   err_msg="Found wrong value in SED.__mul__")

def test_SED_div():
    """Check that SEDs divide like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = lambda w: w**2
    # SED divided by function
    c = a/b
    np.testing.assert_almost_equal(c(3.0), 3.3 / 3**2, 10,
                                   err_msg="Found wrong value in SED.__div__")
    # function divided by SED
    c = b/a
    np.testing.assert_almost_equal(c(3.0), 3**2 / 3.3, 10,
                                   err_msg="Found wrong value in SED.__rdiv__")
    # SED divided by scalar
    d = c/4.2
    np.testing.assert_almost_equal(d(3.0), 3**2 / 3.3 / 4.2, 10,
                                   err_msg="Found wrong value in SED.__div__")
    # assignment division
    d /= 2
    np.testing.assert_almost_equal(d(3.0), 3**2 / 3.3 / 4.2 / 2, 10,
                                   err_msg="Found wrong value in SED.__div__")

def test_SED_atRedshift():
    """Check that SEDs redshift correctly.
    """
    a = galsim.SED(os.path.join(datapath, 'CWW_E_ext.sed'))
    for z1, z2 in zip([0.5, 1.0, 1.4], [1.0, 1.0, 1.0]):
        b = a.atRedshift(z1)
        c = b.atRedshift(z1) # same redshift, so should be no change
        d = c.atRedshift(z2) # do a relative redshifting from z1 to z2
        for w in [350, 500, 650]:
            np.testing.assert_almost_equal(a(w), b(w*(1.0+z1)), 10,
                                           err_msg="error redshifting SED")
            np.testing.assert_almost_equal(a(w), c(w*(1.0+z1)), 10,
                                           err_msg="error redshifting SED")
            np.testing.assert_almost_equal(a(w), d(w*(1.0+z2)), 10,
                                           err_msg="error redshifting SED")

if __name__ == "__main__":
    test_SED_add()
    test_SED_sub()
    test_SED_mul()
    test_SED_div()
    test_SED_atRedshift()
