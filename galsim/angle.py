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

    >>> theta.wrap()

This could be appropriate before testing for the equality of two angles for example, or 
calculating the difference between them.

"""

def __str__(self):
    angle_rad = self.rad()
    return str(angle_rad)+" radians"

def __repr__(self):
    angle_rad = self.rad()
    return str(angle_rad)+" * galsim.radians"

galsim.Angle.__str__ = __str__
galsim.Angle.__repr__ = __repr__

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
 
