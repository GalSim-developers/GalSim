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
"""@file celestial.py
The CelestialCoord class describing coordinates on the celestial sphere.
"""

import galsim

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

    There are several tangent plane projections you can use:
        - a Lambert projection, which preserves area
        - a stereographic projection, which preserves angles
        - a gnomonic projection, which makes all great circles straight lines
        - a Postel projection, which preserves distances from the tangent point
    See the project and deproject functions for details.

    You can also precess a coordinate from one epoch to another and get the galaxy
    coordinates with

            coord1950 = coord2000.precess(2000, 1950)
            el, b = coord.getGalaxyPos()

    We don't use either of these for anything within GalSim, but I had the code to do it
    lying around, so I included it here in case someone might find it useful.

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
        self._ra = ra.wrap()
        if dec/galsim.degrees > 90. or dec/galsim.degrees < -90.:
            raise ValueError("dec must be between -90 deg and +90 deg.")
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
            'gnomonic' uses a gnomonic projection (i.e. a projection from the center of the
                    sphere, which has the property that all great circles become straight 
                    lines.  For more information, see
                    http://mathworld.wolfram.com/GnomonicProjection.html
            'postel' uses a Postel equidistant proejection, which preserves distances from
                    the projection point, but not area or angles.  For more information, see
                    http://mathworld.wolfram.com/AzimuthalEquidistantProjection.html

        The distance or angle errors increase with distance from the projection point of course.

        Returns (u,v) in arcsec as a PositionD object.
        """
        if projection not in [ 'lambert', 'stereographic', 'gnomonic', 'postel' ]:
            raise ValueError('Unknown projection ' + projection)

        self._set_aux()
        other._set_aux()

        # The core calculation is done in a helper function:
        u, v = self._project_core(other._cosra, other._sinra, other._cosdec, other._sindec,
                                 projection)

        return galsim.PositionD(u,v)

    def _project_core(self, cosra, sinra, cosdec, sindec, projection):
        # The equations are given at the above mathworld websites.  They are the same except
        # for the definition of k:
        #
        # x = k cos(dec) sin(ra-ra0)
        # y = k ( cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(ra-ra0) )
        #
        # Lambert:
        #   k = sqrt( 2  / ( 1 + cos(c) ) )
        # Stereographic:
        #   k = 2 / ( 1 + cos(c) )
        # Gnomonic:
        #   k = 1 / cos(c)
        # Postel:
        #   k = c / sin(c)
        # where cos(c) = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(ra-ra0)

        # cos(dra) = cos(ra-ra0) = cos(ra0) cos(ra) + sin(ra0) sin(ra)
        cosdra = self._cosra * cosra + self._sinra * sinra

        # sin(dra) = -sin(ra - ra0);
        # Note: - sign here is to make +x correspond to -ra,
        #       so x increases for decreasing ra.
        #       East is to the left on the sky!
        # sin(dra) = -cos(ra0) sin(ra) + sin(ra0) cos(ra)
        sindra = -self._cosra * sinra + self._sinra * cosra

        # Calculate k according to which projection we are using
        cosc = self._sindec * sindec + self._cosdec * cosdec * cosdra
        if projection[0] == 'l':
            import numpy
            k = numpy.sqrt( 2. / (1.+cosc) )
        elif projection[0] == 's':
            k = 2. / (1. + cosc)
        elif projection[0] == 'g':
            k = 1. / cosc
        else:
            import numpy
            c = numpy.arccos(cosc)
            k = c / numpy.sin(c)

        u = k * cosdec * sindra
        v = k * ( self._cosdec * sindec - self._sindec * cosdec * cosdra )

        # Convert to arcsec
        factor = 1. * galsim.radians / galsim.arcsec
        u *= factor
        v *= factor

        return u, v

    def project_rad(self, ra, dec, projection):
        """This is basically identical to the project function except that the input ra, dec are 
        given in radians rather than packaged as a CelestialCoord object.

        Also, the output is returned as a tuple (x,y), rather than packaged as a PositionD object.

        The main advantage to this is that it will work if ra and dec are numpy arrays, in which 
        case the output x, y will also be numpy arrays.
        """
        if projection not in [ 'lambert', 'stereographic', 'gnomonic', 'postel' ]:
            raise ValueError('Unknown projection ' + projection)

        self._set_aux()

        import numpy
        cosra = numpy.cos(ra)
        sinra = numpy.sin(ra)
        cosdec = numpy.cos(dec)
        sindec = numpy.sin(dec)

        return self._project_core(cosra, sinra, cosdec, sindec, projection)

    def deproject(self, pos, projection='lambert'):
        """Do the reverse process from the project function.

        i.e. This takes in a position (u,v) as a PositionD object and returns the
        corresponding celestial coordinate, using the current coordinate as the center
        point of the tangent plane projection.
        """
        if projection not in [ 'lambert', 'stereographic', 'gnomonic', 'postel' ]:
            raise ValueError('Unknown projection ' + projection)

        # Again, do the core calculations in a helper function
        ra, dec = self._deproject_core(pos.x, pos.y, projection)

        return CelestialCoord(ra*galsim.radians,dec*galsim.radians)

    def _deproject_core(self, u, v, projection):
        # The inverse equations are also given at the same web sites:
        #
        # sin(dec) = cos(c) sin(dec0) + v sin(c) cos(dec0) / r
        # tan(ra-ra0) = u sin(c) / (r cos(dec0) cos(c) - v sin(dec0) sin(c))
        #
        # where
        #
        # r = sqrt(u^2+v^2)
        # c = 2 sin^(-1)(r/2) for lambert
        # c = 2 tan^(-1)(r/2) for stereographic
        # c = tan^(-1)(r)     for gnomonic
        # c = r               for postel

        # Convert from arcsec to radians
        factor = 1. * galsim.arcsec / galsim.radians
        u = u * factor
        v = v * factor

        # Note that we can rewrite the formulae as:
        #
        # sin(dec) = cos(c) sin(dec0) + v (sin(c)/r) cos(dec0)
        # tan(ra-ra0) = u (sin(c)/r) / (cos(dec0) cos(c) - v sin(dec0) (sin(c)/r))
        #
        # which means we only need cos(c) and sin(c)/r.  For most of the projections, 
        # this saves us from having to take sqrt(rsq).

        import numpy
        rsq = u*u + v*v
        if projection[0] == 'l':
            # c = 2 * arcsin(r/2)
            # Some trig manipulations reveal:
            # cos(c) = 1 - r^2/2
            # sin(c) = r sqrt(4-r^2) / 2
            cosc = 1. - rsq/2.
            sinc_over_r = numpy.sqrt(4.-rsq) / 2.
        elif projection[0] == 's':
            # c = 2 * arctan(r/2)
            # Some trig manipulations reveal:
            # cos(c) = (4-r^2) / (4+r^2)
            # sin(c) = 4r / (4+r^2)
            cosc = (4.-rsq) / (4.+rsq)
            sinc_over_r = 4. / (4.+rsq)
        elif projection[0] == 'g':
            # c = arctan(r)
            # cos(c) = 1 / sqrt(1+r^2)
            # sin(c) = r / sqrt(1+r^2)
            cosc = sinc_over_r = 1./numpy.sqrt(1.+rsq)
        else:
            r = numpy.sqrt(rsq)
            cosc = numpy.cos(r)
            sinc_over_r = numpy.sinc(r)

        # Compute sindec, tandra
        self._set_aux()
        sindec = cosc * self._sindec + v * sinc_over_r * self._cosdec
        # Remember the - sign so +dra is -u.  East is left.
        tandra_num = -u * sinc_over_r
        tandra_denom = cosc * self._cosdec - v * sinc_over_r * self._sindec

        dec = numpy.arcsin(sindec)
        ra = self.ra.rad() + numpy.arctan2(tandra_num, tandra_denom)

        return ra, dec

    def deproject_rad(self, u, v, projection='lambert'):
        """This is basically identical to the deproject function except that the output ra, dec 
        are returned as a tuple (ra, dec) in radians rather than packaged as a CelestialCoord 
        object.

        Also, the input is taken as a tuple (u,v), rather than packaged as a PositionD object.

        The main advantage to this is that it will work if u and v are numpy arrays, in which 
        case the output ra, dec will also be numpy arrays.
        """
        if projection not in [ 'lambert', 'stereographic', 'gnomonic', 'postel' ]:
            raise ValueError('Unknown projection ' + projection)

        return self._deproject_core(u, v, projection)

    def deproject_jac(self, u, v, projection='lambert'):
        """Return the jacobian of the deprojection.

        i.e. if the input position is (u,v) (in arcsec) then return the matrix is

        J = ( dra/du cos(dec)  dra/dv cos(dec) )
            (    ddec/du          ddec/dv      )

        The matrix is returned as a tuple (J00, J01, J10, J11)
        """
        if projection not in [ 'lambert', 'stereographic', 'gnomonic', 'postel' ]:
            raise ValueError('Unknown projection ' + projection)

        factor = 1. * galsim.arcsec / galsim.radians
        u = u * factor
        v = v * factor

        # sin(dec) = cos(c) sin(dec0) + v sin(c)/r cos(dec0)
        # tan(ra-ra0) = u sin(c)/r / (cos(dec0) cos(c) - v sin(dec0) sin(c)/r)
        #
        # d(sin(dec)) = cos(dec) ddec = s0 dc + (v ds + s dv) c0
        # dtan(ra-ra0) = sec^2(ra-ra0) dra 
        #              = ( (u ds + s du) A - u s (dc c0 - (v ds + s dv) s0 ) )/A^2 
        # where s = sin(c) / r
        #       c = cos(c)
        #       s0 = sin(dec0)
        #       c0 = cos(dec0) 
        #       A = c c0 - v s s0

        import numpy
        rsq = u*u + v*v
        rsq1 = (u+1.e-4)**2 + v**2
        rsq2 = u**2 + (v+1.e-4)**2
        if projection[0] == 'l':
            c = 1. - rsq/2.
            s = numpy.sqrt(4.-rsq) / 2.
            dcdu = -u
            dcdv = -v
            dsdu = -u/(4.*s)
            dsdv = -v/(4.*s)
        elif projection[0] == 's':
            s = 4. / (4.+rsq)
            c = 2.*s-1.
            ssq = s*s
            dcdu = -u * ssq
            dcdv = -v * ssq
            dsdu = 0.5*dcdu
            dsdv = 0.5*dcdv
        elif projection[0] == 'g':
            c = s = 1./numpy.sqrt(1.+rsq)
            s3 = s*s*s
            dcdu = dsdu = -u*s3
            dcdv = dsdv = -v*s3
        else:
            r = numpy.sqrt(rsq)
            if r == 0.:
                c = s = 1
            else:
                c = numpy.cos(r)
                s = numpy.sin(r)/r
            dcdu = -s*u
            dcdv = -s*v
            dsdu = (c-s)*u/rsq
            dsdv = (c-s)*v/rsq

        self._set_aux()
        s0 = self._sindec
        c0 = self._cosdec
        sindec = c * s0 + v * s * c0
        cosdec = numpy.sqrt(1.-sindec*sindec)
        dddu = ( s0 * dcdu + v * dsdu * c0 ) / cosdec
        dddv = ( s0 * dcdv + (v * dsdv + s) * c0 ) / cosdec

        tandra_num = u * s
        tandra_denom = c * c0 - v * s * s0
        # Note: A^2 sec^2(dra) = denom^2 (1 + tan^2(dra) = denom^2 + num^2
        A2sec2dra = tandra_denom**2 + tandra_num**2
        drdu = ((u * dsdu + s) * tandra_denom - u * s * ( dcdu * c0 - v * dsdu * s0 ))/A2sec2dra
        drdv = (u * dsdv * tandra_denom - u * s * ( dcdv * c0 - (v * dsdv + s) * s0 ))/A2sec2dra

        drdu *= cosdec
        drdv *= cosdec
        return drdu, drdv, dddu, dddv

    def precess(self, from_epoch, to_epoch):
        """This function precesses equatorial ra and dec from one epoch to another.
           It is adapted from a set of fortran subroutines found in precess.f,
           which  were based on (a) pages 30-34 fo the Explanatory Supplement
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
        return new_coord

    def getGalaxyPos(self, epoch=2000.):
        """Get the galaxy longitude and latitude corresponding to this position.

        It returns the longitude and latitude as a tuple (el, b).  They are each given
        as galsim.Angle instances.

        The formulae are implemented in terms of the 1950 coordinates, so we need to
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

