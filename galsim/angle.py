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
"""@file angle.py
A few adjustments to the Angle class at the Python layer.
"""

import galsim

galsim.AngleUnit.__doc__ = """A class for defining angular units in galsim.Angle objects.

Initialization
--------------

An AngleUnit takes a single argument for initialization, a float that specifies the size of the
desired angular unit in radians.  For example:

    >>> import math
    >>> gradian = galsim.AngleUnit(2. * math.pi / 400.)

There are five built-in AngleUnits which are always available for use:

    galsim.radians   # = galsim.AngleUnit(1.)
    galsim.degrees   # = galsim.AngleUnit(pi / 180.)
    galsim.hours     # = galsim.AngleUnit(pi / 12.)
    galsim.arcmin    # = galsim.AngleUnit(pi / 180. / 60.)
    galsim.arcsec    # = galsim.AngleUnit(pi / 180. / 3600.)
"""


galsim.Angle.__doc__ = """A class representing an Angle.

Initialization
--------------

Angles are a value with an AngleUnit.

You typically create an Angle by multiplying a number by a galsim.AngleUnit, for example

    >>> pixel = 0.27 * galsim.arcsec
    >>> ra = 13.4 * galsim.hours
    >>> dec = -32 * galsim.degrees
    >>> import math
    >>> theta = math.pi / 2. * galsim.radians

You can initialize explicitly, taking a value and a unit:

    >>> phi = galsim.Angle(90, galsim.degrees)

There are five built-in AngleUnits which are always available for use:

    galsim.radians   # = galsim.AngleUnit(1.)
    galsim.degrees   # = galsim.AngleUnit(pi / 180.)
    galsim.hours     # = galsim.AngleUnit(pi / 12.)
    galsim.arcmin    # = galsim.AngleUnit(pi / 180. / 60.)
    galsim.arcsec    # = galsim.AngleUnit(pi / 180. / 3600.)

Radian access method
--------------------

Since extracting the value in radians is extremely common, we have an accessor method to do this 
quickly:

    >>> x = theta.rad()
    >>> print x
    1.57079632679

It is equivalent to the more verbose:

    >>> x = theta / galsim.radians

but without actually requiring the FLOP of dividing by 1.

Operations
----------

Allowed arithmetic with Angles include the following:
(In the list below, x is a double, unit is a galsim.AngleUnit, and theta is a galsim.Angle)

    >>> theta = x * unit
    >>> x = theta / unit
    >>> theta3 = theta1 + theta2
    >>> theta3 = theta1 - theta2
    >>> theta2 = theta1 * x
    >>> theta2 = x * theta1
    >>> theta2 = theta1 / x
    >>> theta2 += theta1
    >>> theta2 -= theta1
    >>> theta *= x
    >>> theta /= x

Operations on Numpy arrays containing Angles are permitted, provided that they are within the bounds
of the allowed operations on Angles listed above (e.g., addition/subtraction of Angles,
multiplication of an Angle by a float, but not multiplication of Angles together).

Wrapping
--------

Depending on the context, theta = 2pi radians and theta = 0 radians are the same thing.
If you want your angles to be wrapped to [-pi,pi) radians, you can do this by calling

    >>> theta = theta.wrap()

This could be appropriate before testing for the equality of two angles for example, or 
calculating the difference between them.

"""

def __str__(self):
    angle_rad = self.rad()
    return str(angle_rad)+" radians"

def __repr__(self):
    angle_rad = self.rad()
    return str(angle_rad)+" * galsim.radians"

def __neg__(self):
    return -1. * self

galsim.Angle.__str__ = __str__
galsim.Angle.__repr__ = __repr__
galsim.Angle.__neg__ = __neg__

def get_angle_unit(unit):
    """Convert a string into the corresponding AngleUnit
    """
    unit = unit.strip().lower()
    if unit.startswith('rad') :
        return galsim.radians
    elif unit.startswith('deg') :
        return galsim.degrees
    elif unit.startswith('hour') :
        return galsim.hours
    elif unit.startswith('hr') :
        return galsim.hours
    elif unit.startswith('arcmin') :
        return galsim.arcmin
    elif unit.startswith('arcsec') :
        return galsim.arcsec
    else :
        raise AttributeError("Unknown Angle unit: %s"%unit)

def parse_dms(s):
    """Convert a string of the form ddmmss.decimal into decimal degrees."""

    sign = 1
    if s[0] == '-': 
        sign = -1
        s = s[1:]
    elif s[0] == '+':
        s = s[1:]

    d = int(s[0:2])
    m = int(s[2:4])
    s = float(s[4:])

    return sign * (d + m/60. + s/3600.)

def HMS_Angle(str):
    """Convert a string of the form hhmmss.decimal into an Angle.

    There may be an initial + or - (or neither), then 2 digits for the hours, two for the 
    minutes, and two for the seconds.  Then there may be a decimal point followed by more
    digits.

    @returns the corresponding Angle instance
    """
    return parse_dms(str) * galsim.hours

def DMS_Angle(str):
    """Convert a string of the form ddmmss.decimal into an Angle.

    There may be an initial + or - (or neither), then 2 digits for the degrees, two for the 
    minutes, and two for the seconds.  Then there may be a decimal point followed by more
    digits.

    @returns the corresponding Angle instance
    """
    return parse_dms(str) * galsim.degrees


class CelestialCoord(object):
    """This is a lightweight wrapper of two Angles, ra and dec.  

    Mostly this class exists to enforce the units when a position is really a location on
    the celestial sphere rather than using PositionD as we normally do for positions.

    Also, most of the arithmetic defined for PositionD is invalid on the sphere, so this 
    prevents you from doing things like `coord1 - coord2` for (ra,dec) positions.

    There is a `distanceTo` method which returns the great-circle angle between two
    positions, but there isn't anything else in terms of arithmetic.

    Initialization
    --------------
    A CelestialCoord object is initialized with the following command:

        coord = galsim.CelestialCoord(ra, dec)

    @param ra       The right ascension.  Must be a galsim.Angle object.
    @param dec      The declination.  Must be a galsim.Angle object.
    """
    def __init__(self, ra, dec):
        if not isinstance(ra, galsim.Angle):
            raise TypeError("ra must be a galsim.Angle")
        if not isinstance(dec, galsim.Angle):
            raise TypeError("dec must be a galsim.Angle")
        self._ra = ra
        self._dec = dec

    @property
    def ra(self): return self._ra
    @property
    def dec(self): return self._dec

    def distanceTo(self, other):
        """Returns the great circle distance between this coord and another one.

        The return value is a galsim.Angle object
        """
        # The easiest way to do this in a way that is stable for small separations
        # is to calculate the (x,y,z) position on the unit sphere corresponding to each
        # coordinate position.
        #
        # x = cos(dec) cos(ra)
        # y = cos(dec) sin(ra)
        # z = sin(dec)

        # Note: if this is a common operation, it is faster to compute x,y,z for each point
        # once in the constructor.  However, I don't think that is currently the case for 
        # any use case in GalSim.  Really, this function is kind of just for grins...  :)
        cosdec1 = math.cos(self._dec.rad())
        sindec1 = math.sin(self._dec.rad())
        cosra1 = math.cos(self._ra.rad())
        sinra1 = math.sin(self._ra.rad())
        cosdec2 = math.cos(other._dec.rad())
        sindec2 = math.sin(other._dec.rad())
        cosra2 = math.cos(other._ra.rad())
        sinra2 = math.sin(other._ra.rad())

        x1 = cosdec1 * cosra1
        y1 = cosdec1 * sinra1
        z1 = sindec1
        x2 = cosdec2 * cosra2
        y2 = cosdec2 * sinra2
        z2 = sindec2

        # The the direct distance between the two points is
        #
        # d^2 = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2

        dsq = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2

        # This direct distance can then be converted to a great circle distance via
        #
        # sin(theta/2) = d/2

        theta = 2. * math.asin(0.5 * math.sqrt(dsq))
        return theta * galsim.radians

    def copy(self): return CelestialCoord(self._ra, self._dec)


