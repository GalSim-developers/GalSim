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

import galsim

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

def test_SED_mul():
    """Check that SEDs multiply like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = lambda w: w**2
    c = a*b
    np.testing.assert_almost_equal(c(3.0), 3.3 * 3**2, 10,
                                   err_msg="Found value in SED.__mul__")
    c = b*a
    np.testing.assert_almost_equal(c(3.0), 3.3 * 3**2, 10,
                                   err_msg="Found value in SED.__rmul__")
    d = c*4.2
    np.testing.assert_almost_equal(d(3.0), 3.3 * 3**2 * 4.2, 10,
                                   err_msg="Found value in SED.__mul__")
    d *= 2
    np.testing.assert_almost_equal(d(3.0), 3.3 * 3**2 * 4.2 * 2, 10,
                                   err_msg="Found value in SED.__mul__")

def test_SED_div():
    """Check that SEDs divide like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = lambda w: w**2
    # SED divided by function
    c = a/b
    np.testing.assert_almost_equal(c(3.0), 3.3 / 3**2, 10,
                                   err_msg="Found value in SED.__div__")
    # function divided by SED
    c = b/a
    np.testing.assert_almost_equal(c(3.0), 3**2 / 3.3, 10,
                                   err_msg="Found value in SED.__rdiv__")
    # SED divided by scalar
    d = c/4.2
    np.testing.assert_almost_equal(d(3.0), 3**2 / 3.3 / 4.2, 10,
                                   err_msg="Found value in SED.__div__")
    # assignment division
    d /= 2
    np.testing.assert_almost_equal(d(3.0), 3**2 / 3.3 / 4.2 / 2, 10,
                                   err_msg="Found value in SED.__div__")

if __name__ == "__main__":
    test_SED_add()
    test_SED_sub()
    test_SED_mul()
    test_SED_div()
