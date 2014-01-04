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

def _make_dms_string(decimal):
    if decimal >= 0:
        sign = '+'
    else:
        sign = '-'
        decimal = -decimal
    h = int(decimal)
    decimal -= h
    decimal *= 60.
    m = int(decimal)
    decimal -= m
    decimal *= 60.
    s = int(decimal)
    decimal -= s
    decimal *= 1.e8
    return '%s%02d%02d%02d.%08d'%(sign,h,m,s,decimal)

def hms(self):
    """Return an HMS representation of the angle as a string: (+/-)hhmmss.decimal"""
    return _make_dms_string(self / galsim.hours)

def dms(self):
    """Return a DMS representation of the angle as a string: (+/-)ddmmss.decimal"""
    return _make_dms_string(self / galsim.degrees)

galsim.Angle.__str__ = __str__
galsim.Angle.__repr__ = __repr__
galsim.Angle.__neg__ = __neg__
galsim.Angle.hms = hms
galsim.Angle.dms = dms

# Enable pickling
def Angle_getstate(self):
    return self.rad()
def Angle_setstate(self, theta):
    self.__init__(theta, galsim.radians)
galsim.Angle.__getstate__ = Angle_getstate
galsim.Angle.__setstate__ = Angle_setstate


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
    """This is class defines a position on the celestial sphere, normally given by
    two angles, ra and dec.

    Mostly this class exists to enforce the units when a position is really a location on
    the celestial sphere rather than using PositionD as we normally do for positions.
    In that role, it can be considered a lightweight wrapper around two angles, ra and dec.
    They are accessible as coord.ra and coord.dec.

    However, there are a few useful functions that we provide.

    The distance between two coordinate positions can be calculated with

            d = coord.distanceTo(other_coord)

    There are two tangent plane projections you can use:
        - a Lambert projection, which preserves area
        - a stereographic projection, which preserves angles
        - a Postel projection, which preserves distances from the tangent point
    See the project and deproject functions for details.

    You can also precess a coordinate from one epoch to another and get the galaxy
    coordinates with

            coord_1950 = coord2000.precess(2000, 1950)
            el, b = coord.getGalaxyPos()

    We don't use either of these for anything within GalSim, but I had the code to do it
    lying around, so I included it here.

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
        self._x = None  # Indicate that x,y,z are not set yet.

    @property
    def ra(self): return self._ra
    @property
    def dec(self): return self._dec

    def _set_aux(self):
        if self._x is None:
            import math
            self._cosdec = math.cos(self._dec.rad())
            self._sindec = math.sin(self._dec.rad())
            self._cosra = math.cos(self._ra.rad())
            self._sinra = math.sin(self._ra.rad())
            self._x = self._cosdec * self._cosra
            self._y = -self._cosdec * self._sinra
            self._z = self._sindec

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

        self._set_aux()
        other._set_aux()

        # The the direct distance between the two points is
        #
        # d^2 = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2

        dsq = (self._x-other._x)**2 + (self._y-other._y)**2 + (self._z-other._z)**2

        # This direct distance can then be converted to a great circle distance via
        #
        # sin(theta/2) = d/2

        import math
        theta = 2. * math.asin(0.5 * math.sqrt(dsq))
        return theta * galsim.radians

    def angleBetween(self, coord1, coord2):
        """Find the open angle at the location of the current coord between coord1 and coord2."""
        # Call A = coord1, B = coord2, C = self
        # Then we are looking for the angle ACB.
        # If we treat each coord as a (x,y,z) vector, then we can use the following spherical
        # trig identities:
        #
        # (A x C) . B = sina sinb sinC
        # (A x C) . (B x C) = sina sinb cosC
        #
        # Then we can just use atan2 to find C, and atan2 automatically gets the sign right.
        # And we only need 1 trig call, assuming that x,y,z are already set up, which is often
        # the case.

        self._set_aux()
        coord1._set_aux()
        coord2._set_aux()

        AxC = ( coord1._y * self._z - coord1._z * self._y ,
                coord1._z * self._x - coord1._x * self._z ,
                coord1._x * self._y - coord1._y * self._x )
        BxC = ( coord2._y * self._z - coord2._z * self._y ,
                coord2._z * self._x - coord2._x * self._z ,
                coord2._x * self._y - coord2._y * self._x )
        sinC = AxC[0] * coord2._x + AxC[1] * coord2._y + AxC[2] * coord2._z
        cosC = AxC[0] * BxC[0] + AxC[1] * BxC[1] + AxC[2] * BxC[2]
        import math
        C = math.atan2(sinC, cosC)
        return C * galsim.radians

    def project(self, other, projection='lambert'):
        """Use the currect coord as the center point of a tangent plane projection to project
        the other coordinate onto that plane.

        This function return the position (u,v) in the Euclidean coordinate system defined by
        a tangent plane projection around the current coordinate, with +v pointing north and
        +u pointing west.

        There are currently three options for the projection, which you can specify with the
        optional `projection` keyword argument:

            'lambert' (the default) uses a Lambert azimuthal projection, which preserves
                    area, but not angles.  For more information, see
                    http://mathworld.wolfram.com/LambertAzimuthalEqual-AreaProjection.html
            'stereographic' uses a stereographic proejection, which preserves angles, but
                    not area.  For more information, see
                    http://mathworld.wolfram.com/StereographicProjection.html
            'postel' uses a Postel equidistant proejection, which preserves distances from
                    the projection point, but not area or angles.  For more information, see
                    http://mathworld.wolfram.com/AzimuthalEquidistantProjection.html

        The distance or angle errors increase with distance from the projection point of course.

        Returns (u,v) in arcsec as a PositionD object.
        """
        # The equations are given at the above mathworld websites.  They are the same except
        # for the definition of k:
        #
        # x = k cos(dec) sin(ra-ra0)
        # y = k ( cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(ra-ra0) )
        #
        # Lambert:
        #   k = sqrt( 2 ( 1 + sin(dec0) sin(dec) + cos(dec0) cos(phi) cos(ra-ra0) )^-1 )
        # Stereographic:
        #   k = 2 ( 1 + sin(dec0) sin(dec) + cos(dec0) cos(phi) cos(ra-ra0) )^-1
        # Postel:
        #   k = c / sin(c)
        #   where cos(c) = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(ra-ra0)

        if projection not in [ 'lambert', 'stereographic', 'postel' ]:
            raise ValueError('Unknown projection ' + projection)

        self._set_aux()
        other._set_aux()

        # cosdra = cos(ra - ra0) = cosra cosra0 + sinra sinra0
        cosdra = self._cosra * other._cosra + self._sinra * other._sinra

        # sindra = -sin(ra - ra0);
        # Note: - sign here is to make +x correspond to -ra,
        #       so x increases for decreasing ra.
        #       East is to the left on the sky!
        # sindra = -sinra cosra0 + cosra sinra0
        sindra = -self._cosra * other._sinra + self._sinra * other._cosra

        # Calculate k according to which projection we are using
        cosc = self._sindec * other._sindec + self._cosdec * other._cosdec * cosdra
        if projection == 'postel':
            import math
            c = math.acos(cosc)
            k = c / math.sin(c)
        else:
            k = 2. / (1. + cosc)
            if projection == 'lambert':
                import math
                k = math.sqrt(k)

        x = k * other._cosdec * sindra
        y = k * ( self._cosdec * other._sindec - self._sindec * other._cosdec * cosdra )

        # Convert to arcsec
        x = x * galsim.radians / galsim.arcsec
        y = y * galsim.radians / galsim.arcsec

        return galsim.PositionD(x,y)

    def deproject(self, pos, projection='lambert'):
        """Do the reverse process from the project function.

        i.e. This takes in a position (u,v) as a PositionD object and returns the
        corresponding celestial coordinate, using the current coordinate as the center
        point of the tangent plane projection.
        """
        # The inverse equations are also given at the same web sites:
        #
        # sin(dec) = cos(c) sin(dec0) + v sin(c) cos(dec0)/r
        # tan(ra-ra0) = u sin(c) / (r cos(dec0) cos(c) - v sin(dec0) sin(c))
        #
        # where
        #
        # r = sqrt(u^2+v^2)
        # c = 2 sin^(-1)(r/2) for lambert
        # c = 2 tan^(-1)(r/2) for stereographic
        # c = sqrt(x^2+y^2)   for postel

        if projection not in [ 'lambert', 'stereographic', 'postel' ]:
            raise ValueError('Unknown projection ' + projection)

        # Convert from arcsec to radians
        u = pos.x * galsim.arcsec / galsim.radians
        v = pos.y * galsim.arcsec / galsim.radians

        import math
        # Compute r, c
        # r = sqrt(u*u + v*v)
        # c = 2 * arctan(r/2) for num == 1
        # c = 2 * arcsin(r/2) for num == 2
        # Note that we can rewrite the formulae as:
        #
        # sin(dec) = cos(c) sin(dec0) + v (sin(c)/r) cos(dec0)
        # tan(ra-ra0) = u (sin(c)/r) / (cos(dec0) cos(c) - v sin(dec0) (sin(c)/r))
        #
        # which means we only need cos(c) and sin(c)/r
        rsq = u*u + v*v
        if projection == 'lambert':
            # c = 2 * arcsin(r/2)
            # Some trig manipulations reveal:
            # cos(c) = 1 - r^2/2
            # sin(c) = r sqrt(4-r^2) / 2
            cosc = 1. - rsq/2.
            sinc_over_r = math.sqrt(4.-rsq) / 2.
        elif projection == 'stereographic':
            # c = 2 * arctan(r/2)
            # Some trig manipulations reveal:
            # cos(c) = (4-r^2) / (4+r^2)
            # sin(c) = 4r / (4+r^2)
            cosc = (4.-rsq) / (4.+rsq)
            sinc_over_r = 4. / (4.+rsq)
        else:
            r = math.sqrt(rsq)
            if r == 0.:
                cosc = 1
                sinc_over_r = 1
            else:
                cosc = math.cos(r)
                sinc_over_r = math.sin(r)/r

        # Compute sindec, tandra
        self._set_aux()
        sindec = cosc * self._sindec + v * sinc_over_r * self._cosdec
        # Remember the - sign so +dra is -u.  East is left.
        tandra_num = -u * sinc_over_r
        tandra_denom = self._cosdec * cosc - v * self._sindec * sinc_over_r

        dec = math.asin(sindec) * galsim.radians
        ra = self.ra + math.atan2(tandra_num, tandra_denom) * galsim.radians

        return CelestialCoord(ra,dec)

    def precess(self, from_epoch, to_epoch):
        """This function precesses equatorial ra and dec from one epoch to another.
           It is adapted from a set if fortran subroutines found in precess.f,
           which  were based on (a) pages 30-34 fo the Eplanatory Supplement
           to the AE, (b) Lieske, et al. (1977) A&A 58, 1-16, and
           (c) Lieske (1979) A&A 73, 282-284.
        """
        if from_epoch == to_epoch: return self

        # t0, t below correspond to Lieske's big T and little T
        t0 = (_epoch-2000.)/100.;
        t = (newepoch-_epoch)/100.;
        t02 = t0*t0;
        t2 = t*t;
        t3 = t2*t;

        # a,b,c below correspond to Lieske's zeta_A, z_A and theta_A
        a = ( (2306.2181 + 1.39656*t0 - 0.000139*t02) * t +
              (0.30188 - 0.000344*t0) * t2 + 0.017998 * t3 ) * galsim.arcsec
        b = ( (2306.2181 + 1.39656*t0 - 0.000139*t02) * t +
              (1.09468 + 0.000066*t0) * t2 + 0.018203 * t3 ) * galsim.arcsec
        c = ( (2004.3109 - 0.85330*t0 - 0.000217*t02) * t +
              (-0.42665 - 0.000217*t0) * t2 - 0.041833 * t3 ) * galsim.arcsec
        import math
        cosa = math.cos(a.rad())
        sina = math.sin(a.rad())
        cosb = math.cos(b.rad())
        sinb = math.sin(b.rad())
        cosc = math.cos(c.rad())
        sinc = math.sin(c.rad())

        # This is the precession rotation matrix:
        xx = cosa*cosc*cosb - sina*sinb;
        yx = -sina*cosc*cosb - cosa*sinb;
        zx = -sinc*cosb;
        xy = cosa*cosc*sinb + sina*cosb;
        yy = -sina*cosc*sinb + cosa*cosb;
        zy = -sinc*sinb;
        xz = cosa*sinc;
        yz = -sina*sinc;
        zz = cosc;

        # Perform the rotation:
        self._set_aux()
        x2 = xx*self._x + yx*self._y + zx*self._z,
        y2 = xy*self._x + yy*self._y + zy*self._z,
        z2 = xz*self._x + yz*self._y + zz*self._z,

        new_dec = math.atan2(z2,math.sqrt(x2**2+y2**2)) * galsim.radians
        new_ra = math.atan2(y2,x2) * galsim.radians
        new_coord = CelestialCoord(new_ra,new_dec)
        # Since we already knwo these, might as well set them.
        new_coord._x = x2
        new_coord._y = y2
        new_coord._z = z2
        return new_coord

    def getGalaxyPos(self, epoch=2000.):
        """Get the galaxy longitude and latitude corresponding to this position.

        It returns the longitude and latitude as a tuple (el, b).  They are each given
        as galsim.Angle instances.

        The formulae are implemented in terms of the 1950 coordinates, so it needs to
        precess from the current epoch to 1950.  The current epoch is assumed to be 2000
        by default, but you may also specify a different value with the epoch parameter.
        """
        # cos(b) cos(el-33) = cos(dec) cos(ra-282.25)
        # cos(b) sin(el-33) = sin(dec) sin(62.6) + cos(dec) sin(ra-282.25) cos(62.6)
        #            sin(b) = sin(dec) sin(62.6) - cos(dec) sin(ra-282.25) sin(62.6)
        import math
        el0 = 33. * galsim.degrees
        r0 = 282.25 * galsim.degrees
        d0 = 62.6 * galsim.degrees
        cosd0 = math.cos(d0.rad());
        sind0 = math.sin(d0.rad());

        temp = self.precess(epoch, 1950.);
        d = temp.dec
        r = temp.ra
        cosd = math.cos(d.rad())
        sind = math.sin(d.rad())
        cosr = math.cos(r.rad() - r0.rad())
        sinr = math.sin(r.rad() - r0.rad())

        cbcl = cosd*cosr;
        cbsl = sind*sind0 + cosd*sinr*cosd0;
        sb = sind*cosd0 - cosd*sinr*sind0;

        b = math.asin(sb) * galsim.radians;
        cl = cbcl/cosb;
        sl = cbsl/cosb;
        el = math.atan2(sl,cl) * galsim.radians + el0;

        return (el, b)

    def copy(self): return CelestialCoord(self._ra, self._dec)

    def __repr__(self): return 'CelestialCoord('+repr(self._ra)+','+repr(self._dec)+')'

