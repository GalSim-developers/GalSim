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
"""@file integ.py
A Python layer version of the C++ int1d function in galim::integ
"""

from . import _galsim

def int1d(func, min, max, rel_err=1.e-6, abs_err=1.e-12):
    """Integrate a 1-dimensional function from min to max.

    Example usage:

        >>> def func(x): return x**2
        >>> galsim.integ.int1d(func, 0, 1)
        0.33333333333333337
        >>> galsim.integ.int1d(func, 0, 2)
        2.666666666666667
        >>> galsim.integ.int1d(func, -1, 1)
        0.66666666666666674

    @param func     The function to be integrated.  y = func(x) should be valid.
    @param min      The lower end of the integration bounds (anything < -1.e10 is treated as
                    negative infinity).
    @param max      The upper end of the integration bounds (anything > 1.e10 is treated as positive
                    infinity).
    @param rel_err  The desired relative error (default `rel_err = 1.e-6`)
    @param abs_err  The desired absolute error (default `abs_err = 1.e-12`)
    """
    min = float(min)
    max = float(max)
    rel_err = float(rel_err)
    abs_err = float(abs_err)
    success, result = _galsim.PyInt1d(func,min,max,rel_err,abs_err)
    if success:
        return result
    else:
        raise RuntimeError(result)
