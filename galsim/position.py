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
"""@file position.py
A few adjustments to the Position classes at the Python layer.
"""

from . import _galsim

def Position_repr(self):
    return self.__class__.__name__+"(x="+str(self.x)+", y="+str(self.y)+")"

def Position_str(self):
    return "("+str(self.x)+", "+str(self.y)+")"

def Position_getinitargs(self):
    return self.x, self.y

for Class in (_galsim.PositionD, _galsim.PositionI):
    Class.__repr__ = Position_repr
    Class.__str__ = Position_str
    Class.__getinitargs__ = Position_getinitargs
    Class.__doc__ = """A class for representing 2D positions on the plane.

    PositionD describes positions with floating point values in `x` and `y`.
    PositionI described positions with integer values in `x` and `y`.

    Initialization
    --------------

    For the float-valued position class, example inits include:

        >>> pos = galsim.PositionD(x=0.5, y=-0.5)
        >>> pos = galsim.PositionD(0.5, -0.5)

    And for the integer-valued position class, example inits include:

        >>> pos = galsim.PositionI(x=45, y=13)
        >>> pos = galsim.PositionI(45, 13)

    Attributes
    ----------
    For an instance `pos` as instantiated above, `pos.x` and `pos.y` store the x and y values of the
    position.

    Arithmetic
    ----------
    Most arithmetic that makes sense for a position is allowed:

        >>> pos1 + pos2
        >>> pos1 - pos2
        >>> pos * x
        >>> pos / x
        >>> -pos
        >>> pos1 += pos2
        >>> pos1 -= pos2
        >>> pos *= x
        >>> pos -= x

    Note though that the types generally need to match.  For example, you cannot multiply
    a PositionI by a float or add a PositionI to a PositionD.
    """

del Class    # cleanup public namespace
