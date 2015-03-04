# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
"""@file angle.py
A few adjustments to the Angle class at the Python layer.
"""

import galsim
from ._galsim import Angle, AngleUnit, radians, degrees, hours, arcmin, arcsec

AngleUnit.__doc__ = """A class for defining angular units in galsim.Angle objects.

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

def AngleUnit_repr(self):
    if self is galsim.radians:
        return 'galsim.radians'
    elif self is galsim.degrees:
        return 'galsim.degrees'
    elif self is galsim.hours:
        return 'galsim.house'
    elif self is galsim.arcmin:
        return 'galsim.arcmin'
    elif self is galsim.arcsec:
        return 'galsim.arcsec'
    else:
        return 'galsim.AngleUnit(' + str(self.getValue()) + ')'
AngleUnit.__repr__ = AngleUnit_repr

# Enable pickling
def AngleUnit_getinitargs(self):
    return self.getValue(),
AngleUnit.__getinitargs__ = AngleUnit_getinitargs


def get_angle_unit(unit):
    """Convert a string into the corresponding AngleUnit.
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



Angle.__doc__ = """A class representing an Angle.

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
(In the list below, `x` is a double, `unit` is a galsim.AngleUnit, and `theta` is a galsim.Angle)

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
    >>> x = unit1 / unit2   # equivalent to x = (1 * unit1) / unit2

Operations on NumPy arrays containing Angles are permitted, provided that they are within the bounds
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

def _make_dms_string(decimal, sep):
    if decimal >= 0:
        sign = '+'
    else:
        sign = '-'
        decimal = -decimal
    d = int(decimal)
    decimal -= d
    decimal *= 60.
    m = int(decimal)
    decimal -= m
    decimal *= 60.
    s = int(decimal)
    decimal -= s
    decimal *= 1.e8
    return '%s%02d%s%02d%s%02d.%08d'%(sign,d,sep,m,sep,s,decimal)

def hms(self, sep=":"):
    """Return an HMS representation of the angle as a string: hh:mm:ss.decimal.

    The returned representation will have 0 <= hh < 24.

    An optional `sep` parameter can change the : to something else (e.g. a space or 
    nothing at all).

    Note: the reverse process is effected by HMS_Angle:

        >>> angle = -5.357 * galsim.hours
        >>> hms = angle.hms()
        >>> print hms
        +18:38:34.80000000
        >>> angle2 = galsim.HMS_Angle(hms)
        >>> print angle2 / galsim.hours
        18.643
        >>> print angle2 / galsim.hours - 24
        -5.357
        >>> print angle2 - angle - 24 * galsim.hours
        0.0 radians

    @param sep      The token to put between the hh and mm, and beteen mm and ss.  [default: ':']

    @returns a string of the HMS representation of the angle.
    """
    # HMS convention is usually to have the hours between 0 and 24, not -12 and 12
    h = self.wrap() / galsim.hours
    if h < 0: h += 24.
    return _make_dms_string(h,sep)

def dms(self, sep=":"):
    """Return a DMS representation of the angle as a string: (+/-)ddmmss.decimal
    An optional `sep` parameter can change the : to something else (e.g. a space or 
    nothing at all).

    Note: the reverse process is effected by DMS_Angle:

        >>> angle = -(5 * galsim.degrees + 13 * galsim.arcmin + 23 * galsim.arcsec)
        >>> dms = angle.dms()
        >>> print dms
        -05:13:23.00000000
        >>> angle2 = galsim.DMS_Angle(dms)
        >>> print angle2 / galsim.degrees
        -5.22305555556
        >>> print angle2 - angle
        0.0 radians

    @param sep      The token to put between the dd and mm, and beteen mm and ss.  [default: ':']

    @returns a string of the DMS representation of the angle.
    """
    d = self.wrap() / galsim.degrees
    return _make_dms_string(d,sep)

Angle.__str__ = __str__
Angle.__repr__ = __repr__
Angle.__neg__ = __neg__
Angle.hms = hms
Angle.dms = dms

# Enable pickling
def Angle_getstate(self):
    return self.rad()
def Angle_setstate(self, theta):
    self.__init__(theta, galsim.radians)
Angle.__getstate__ = Angle_getstate
Angle.__setstate__ = Angle_setstate

def parse_dms(s):
    """Convert a string of the form dd:mm:ss.decimal into decimal degrees."""
    sign = 1
    k = 0
    if s[0] == '-':
        sign = -1
        k = 1
    elif s[0] == '+':
        k = 1

    d = int(s[k:k+2])
    k = k+2
    while not '0' <= s[k] < '9': k = k+1
    m = int(s[k:k+2])
    k = k+2
    while not '0' <= s[k] < '9': k = k+1
    s = float(s[k:])

    return sign * (d + m/60. + s/3600.)

def HMS_Angle(str):
    """Convert a string of the form hh:mm:ss.decimal into an Angle.

    There may be an initial + or - (or neither), then two digits for the hours, two for the
    minutes, and two for the seconds.  Then there may be a decimal point followed by more
    digits.  There may be a colon separating hh, mm, and ss, or whitespace, or nothing at all.
    In fact, the code will ignore any non-digits between the hours, minutes, and seconds.

    Note: the reverse process is effected by Angle.hms():

        >>> angle = -5.357 * galsim.hours
        >>> hms = angle.hms()
        >>> print hms
        +18:38:34.80000000
        >>> angle2 = galsim.HMS_Angle(hms)
        >>> print angle2 / galsim.hours
        18.643
        >>> print angle2 / galsim.hours - 24
        -5.357
        >>> print angle2 - angle - 24 * galsim.hours
        0.0 radians

    @returns the corresponding Angle instance
    """
    return parse_dms(str) * galsim.hours

def DMS_Angle(str):
    """Convert a string of the form dd:mm:ss.decimal into an Angle.

    There may be an initial + or - (or neither), then two digits for the degrees, two for the
    minutes, and two for the seconds.  Then there may be a decimal point followed by more
    digits.  There may be a colon separating dd, mm, and ss, or whitespace, or nothing at all.
    In fact, the code will ignore any non-digits between the degrees, minutes, and seconds.

    @returns the corresponding Angle instance
    """
    return parse_dms(str) * galsim.degrees

Angle.wrap.__func__.__doc__ = """Wrap Angle to lie in the range [-pi, pi) radians.

Depending on the context, theta = 2pi radians and theta = 0 radians are the same thing.
If you want your angles to be wrapped to [-pi, pi) radians, you can do this by calling

    >>> theta = theta.wrap()

This could be appropriate before testing for the equality of two angles for example, or
calculating the difference between them.
"""
