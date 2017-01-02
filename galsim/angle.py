# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
import math
from . import _galsim
from .utilities import set_func_doc

class AngleUnit(object):
    """
    A class for defining angular units in galsim.Angle objects.

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
    def __init__(self, value):
        self._value = float(value)

    @property
    def value(self):
        return self._value

    def getValue(self):
        """Equivalent to self.value, mostly for backwards compatibility."""
        return self._value

    def __rmul__(self, theta):
        return Angle(theta, self)

    def __div__(self, unit):
        if not isinstance(unit, AngleUnit):
            raise TypeError("Cannot divide AngleUnit by %s"%unit)
        return self.value / unit.value

    __truediv__ = __div__

    @staticmethod
    def from_name(unit):
        """Convert a string into the corresponding AngleUnit.

        Only the start of the string is checked, so for instance 'radian' or 'radians' is
        equivalent to 'rad'.

        Valid options are:

            unit            returns
            ----            -------
            rad             AngleUnit(1.)
            deg             AngleUnit(pi / 180.)
            hour or hr      AngleUnit(pi / 12.)
            arcmin          AngleUnit(pi / 180. / 60.)
            arcsec          AngleUnit(pi / 180. / 3600.)

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
            raise ValueError("Unknown Angle unit: %s"%unit)

    def __repr__(self):
        if self == galsim.radians:
            return 'galsim.radians'
        elif self == galsim.degrees:
            return 'galsim.degrees'
        elif self == galsim.hours:
            return 'galsim.hours'
        elif self == galsim.arcmin:
            return 'galsim.arcmin'
        elif self == galsim.arcsec:
            return 'galsim.arcsec'
        else:
            return 'galsim.AngleUnit(%r)'%self.value

    def __eq__(self, other):
        return isinstance(other,AngleUnit) and self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(('galsim.AngleUnit', self.value))

def get_angle_unit(unit):
    """Convert a string into the corresponding AngleUnit.
    """
    return AngleUnit.from_name(unit)

radians = AngleUnit(1.)
hours = AngleUnit(math.pi / 12.)
degrees = AngleUnit(math.pi / 180.)
arcmin = AngleUnit(math.pi / 10800.)
arcsec = AngleUnit(math.pi / 648000.)


class Angle(object):
    """A class representing an Angle.

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
    (In the list below, `x` is a double, `unit` is a galsim.AngleUnit, `theta` is a galsim.Angle)

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

    Operations on NumPy arrays containing Angles are permitted, provided that they are within the
    bounds of the allowed operations on Angles listed above (e.g., addition/subtraction of Angles,
    multiplication of an Angle by a float, but not multiplication of Angles together).

    There are convenience function for getting the sin, cos, and tan of an angle, along with
    one for getting sin and cos together, which should be more efficient than doing sin and
    cos separately:

        >>> sint = theta.sin()  # equivalent to sint = math.sin(theta.rad())
        >>> cost = theta.cos()  # equivalent to cost = math.cos(theta.rad())
        >>> tant = theta.tan()  # equivalent to tant = math.tan(theta.rad())
        >>> sint, cost = theta.sincos()

    Wrapping
    --------

    Depending on the context, theta = 2pi radians and theta = 0 radians are the same thing.
    If you want your angles to be wrapped to [-pi,pi) radians, you can do this by calling

        >>> theta = theta.wrap()

    This could be appropriate before testing for the equality of two angles for example, or
    calculating the difference between them.
    """
    def __init__(self, theta, unit):
        if not isinstance(unit, AngleUnit):
            raise TypeError("Invalid unit %s of type %s"%(unit,type(unit)))
        self._rad = float(theta) * unit.value

    @property
    def angle(self):
        """Return a C++-layer Angle, appropriate for passing to C++ functions that take an Angle.
        """
        return _galsim.Angle(self._rad, _galsim.AngleUnit(1.))

    def rad(self):
        """Return the Angle in radians.

        Equivalent to angle / galsim.radians
        """
        return self._rad

    def __neg__(self):
        return _Angle(-self._rad)

    def __add__(self, other):
        if not isinstance(other, Angle):
            raise TypeError("Cannot add %s of type %s to an Angle"%(other,type(other)))
        return _Angle(self._rad + other._rad)

    def __sub__(self, other):
        if not isinstance(other, Angle):
            raise TypeError("Cannot subtract %s of type %s from an Angle"%(other,type(other)))
        return _Angle(self._rad - other._rad)

    def __mul__(self, other):
        if other != float(other):
            raise TypeError("Cannot multiply Angle by %s of type %s"%(other,type(other)))
        return _Angle(self._rad * other)

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, AngleUnit):
            return self._rad / other.value
        elif other == float(other):
            return _Angle(self._rad / other)
        else:
            raise TypeError("Cannot divide Angle by %s of type %s"%(other,type(other)))

    __truediv__ = __div__

    def wrap(self, center=0.):
        """Wrap Angle to lie in the range [-pi, pi) radians.

        Depending on the context, theta = 2pi radians and theta = 0 radians are the same thing.
        If you want your angles to be wrapped to [-pi, pi) radians, you can do this by calling

            >>> theta = theta.wrap()

        This could be appropriate before testing for the equality of two angles for example, or
        calculating the difference between them.

        If you want to wrap to a different range than [-pi, pi), you can set the `center` argument
        to be the desired center of the the range.  e.g. for return values to fall in [0, 2pi),
        you would call

            >>> theta = theta.wrap(center=math.pi)

        @param center   The center point of the wrapped range. [default: 0]

        @returns the equivalent angle within the range [center-pi, center+pi)
        """
        start = center - math.pi
        offset = (self._rad - start) // (2.*math.pi)  # How many full cycles to subtract
        return _Angle(self._rad - offset * 2.*math.pi)

    def sin(self):
        """Return the sin of an Angle."""
        return math.sin(self._rad)

    def cos(self):
        """Return the cos of an Angle."""
        return math.cos(self._rad)

    def tan(self):
        """Return the tan of an Angle."""
        return math.tan(self._rad)

    def sincos(self):
        """Return both the sin and cos of an Angle as a tuple (sint, cost).

        (On some systems, this may be slightly faster than doing each separately.)
        """
        return _galsim.sincos(self._rad)

    def __str__(self):
        return str(self._rad) + ' radians'

    def __repr__(self):
        return 'galsim.Angle(%r, galsim.radians)'%self.rad()

    def __eq__(self, other):
        return isinstance(other,Angle) and self.rad() == other.rad()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(('galsim.Angle', self._rad))

    @staticmethod
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

        @param sep      The token to put between the hh and mm, and beteen mm and ss. [default: ':']

        @returns a string of the HMS representation of the angle.
        """
        # HMS convention is usually to have the hours between 0 and 24, not -12 and 12
        h = self.wrap() / galsim.hours
        if h < 0: h += 24.
        return self._make_dms_string(h,sep)

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

        @param sep      The token to put between the dd and mm, and beteen mm and ss. [default: ':']

        @returns a string of the DMS representation of the angle.
        """
        d = self.wrap() / galsim.degrees
        return self._make_dms_string(d,sep)

    @staticmethod
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


def _Angle(theta):
    """Equivalent to either `theta * galsim.radians` or `Angle(theta, galsim.radians)`, but without
    the normal overhead (which isn't much to be honest, but this is nonetheless slightly quicker).
    """
    ret = Angle.__new__(Angle)
    ret._rad = theta
    return ret


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
    return Angle.parse_dms(str) * galsim.hours

def DMS_Angle(str):
    """Convert a string of the form dd:mm:ss.decimal into an Angle.

    There may be an initial + or - (or neither), then two digits for the degrees, two for the
    minutes, and two for the seconds.  Then there may be a decimal point followed by more
    digits.  There may be a colon separating dd, mm, and ss, or whitespace, or nothing at all.
    In fact, the code will ignore any non-digits between the degrees, minutes, and seconds.

    @returns the corresponding Angle instance
    """
    return Angle.parse_dms(str) * galsim.degrees

_galsim.AngleUnit.__getinitargs__ = lambda self: (self.getValue(),)
_galsim.AngleUnit.__eq__ = lambda self, other: self.getValue() == other.getValue()
_galsim.AngleUnit.__hash__ = lambda self: hash(('galsim.AngleUnit',self.getValue()))
_galsim.AngleUnit.__repr__ = lambda self: 'galsim._galsim.AngleUnit(%r)'%self.getValue()

_galsim.Angle.__getinitargs__ = lambda self: (self.rad(), _galsim.AngleUnit(1.))
_galsim.Angle.__eq__ = lambda self, other: self.rad() == other.rad()
_galsim.Angle.__hash__ = lambda self: hash(('galsim.Angle',self.rad()))
_galsim.Angle.__repr__ = lambda self: 'galsim._galsim.Angle(%r, %r)'%(
        self.rad(), galsim._galsim.AngleUnit(1.))

