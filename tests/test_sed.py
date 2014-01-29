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
    a = galsim.SED(wave=[1,2,3,4,5], fphotons=[1.1,2.2,3.3,4.4,5.5])
    b = galsim.SED(wave=[1.1,2.2,3.0,4.4,5.5], fphotons=[1.11,2.22,3.33,4.44,5.55])
    c = a+b
    np.testing.assert_almost_equal(c.wave[0], 1.1, 5,
                                   err_msg="Found wrong wavelength intersection interval in" +
                                   "SED.__add__")
    np.testing.assert_almost_equal(c.wave[-1], 5.0, 5,
                                   err_msg="Found wrong wavelength intersection interval in" +
                                   "SED.__add__")
    np.testing.assert_almost_equal(c(3.0), 3.3 + 3.33, 5,
                                   err_msg="Wrong sum in SED.__add__")

if __name__ == "__main__":
    test_SED_add()
