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
"""@file wcs.py 
All the classes to implement different WCS transformations for GalSim Images.
"""

import galsim

class BaseWCS(object):
    """The base class for all other kinds of WCS transformations.  It doesn't really
    do much except provide a type for isinstance(wcs,BaseWCS) queries.
    """    
    def __init__(self):
        raise TypeError("BaseWCS is an abstract base class.  It cannot be instantiated.")

        # All derived classes must define the following:
        #
        #     _is_local        boolean variable declaring whether the WCS is local, linear
        #     _posToWorld      function converting image_pos to world_pos
        #     _posToImage      function converting world_pos to image_pos
        #     copy
        #     __eq__
        #     __ne__
        #
        # Non-variable WCS classes must define the following:
        #
        #     _profileToWorld  function converting image_profile to world_profile
        #     _profileToImage  function converting world_profile to image_profile
        #     _pixelArea       function returning the pixel area
        #     _minScale        function returning the minimum linear pixel scale
        #     _maxScale        function returning the maximum linear pixel scale
        #     _toAffine        function returning an equivalent AffineTransform
        #
        # Variable WCS classes and those where (x,y) = (0,0) is not (necessarily) at
        # (u,v) = (0,0) must define the following:
        #
        #     _local           function returning a non-variable WCS at a given location
        #                      where (x,y) = (0,0) is at (u,v) = (0,)


    def toWorld(self, arg, **kwargs):
        """Convert from image coordinates to world coordinates

        There are essentially two overloaded versions of this function here.

        1. The first converts a position from image coordinates to world coordinates.
           The argument may be either a PositionD or PositionI argument.  It returns
           the corresponding position in world coordinates as a PositionD if the WCS 
           is linear, or a CelestialCoord if it is in terms of RA/Dec.

               world_pos = wcs.toWorld(image_pos)

        2. The second converts a surface brightness profile (a GSObject) from image 
           coordinates to world coordinates, returning the profile in world coordinates
           as a new GSObject.  For variable WCS transforms, you must provide either
           image_pos or world_pos to say where the profile is located so the right
           transformation can be performed.

               world_profile = wcs.toWorld(image_profile, image_pos=None, world_pos=None)
        """
        if isinstance(arg, galsim.GSObject):
            return self.local(**kwargs)._profileToWorld(arg)
        else:
            return self._posToWorld(arg)

    def toImage(self, arg, **kwargs):
        """Convert from world coordinates to image coordinates

        There are essentially two overloaded versions of this function here.

        1. The first converts a position from world coordinates to image coordinates.
           If the WCS is linear, the argument may be either a PositionD or PositionI 
           argument.  If the WCS is defined on the sphere in terms of RA/Dec, then 
           the argument must be a CelestialCoord.  It returns the corresponding 
           position in image coordinates as a PositionD.

               image_pos = wcs.toImage(world_pos)

        2. The second converts a surface brightness profile (a GSObject) from world 
           coordinates to image coordinates, returning the profile in image coordinates
           as a new GSObject.  For variable WCS transforms, you must provide either
           image_pos or world_pos to say where the profile is located so the right
           transformation can be performed.

               image_profile = wcs.toImage(world_profile, image_pos=None, world_pos=None)
        """
        if isinstance(arg, galsim.GSObject):
            return self.local(**kwargs)._profileToImage(arg)
        else:
            return self._posToImage(arg)

    def pixelArea(self, image_pos=None, world_pos=None):
        """Return the area of a pixel in arcsec**2 (or in whatever units you are using for 
        world coordinates).

        For variable WCS transforms, you must provide either image_pos or world_pos
        to say where the pixel is located.

        @param image_pos    The image coordinate position (for variable WCS objects)
        @param world_pos    The world coordinate position (for variable WCS objects)
        @returns            The pixel area in arcsec**2
        """
        return self.local(image_pos, world_pos)._pixelArea()

    def minLinearScale(self, image_pos=None, world_pos=None):
        """Return the minimum linear scale of the transformation in any direction.

        This is basically the semi-minor axis of the Jacobian.  Sometimes you need a
        linear scale size for some calculation.  This function returns the smallest
        scale in any direction.  The function maxLinearScale returns the largest.

        For variable WCS transforms, you must provide either image_pos or world_pos
        to say where the pixel is located.

        @param image_pos    The image coordinate position (for variable WCS objects)
        @param world_pos    The world coordinate position (for variable WCS objects)
        @returns            The minimum pixel area in any direction in arcsec
        """
        return self.local(image_pos, world_pos)._minScale()

    def maxLinearScale(self, image_pos=None, world_pos=None):
        """Return the maximum linear scale of the transformation in any direction.

        This is basically the semi-major axis of the Jacobian.  Sometimes you need a
        linear scale size for some calculation.  This function returns the largest
        scale in any direction.  The function minLinearScale returns the smallest.

        For variable WCS transforms, you must provide either image_pos or world_pos
        to say where the pixel is located.

        @param image_pos    The image coordinate position (for variable WCS objects)
        @param world_pos    The world coordinate position (for variable WCS objects)
        @returns            The maximum pixel area in any direction in arcsec
        """
        return self.local(image_pos, world_pos)._maxScale()

    def isLocal(self):
        """Return whether the WCS solution is a local, linear approximation.
        
        There are two requirements for this to be true:
            1. The image position (x,y) = (0,0) is at the world position (u,v) = (0,0).
            2. The pixel area and shape do not vary with position.
        """
        return self._is_local

    def local(self, image_pos=None, world_pos=None):
        """Return the local linear approximation of the WCS at a given point.

        @param image_pos    The image coordinate position (for variable WCS objects)
        @param world_pos    The world coordinate position (for variable WCS objects)
        @returns local_wcs  A WCS object with wcs.isLocal() == True
        """
        if image_pos and world_pos:
            raise TypeError("Only one of image_pos or world_pos may be provided")
        if self._is_local:
            return self
        else:
            return self._local(image_pos, world_pos)

    def localAffine(self, image_pos=None, world_pos=None):
        """Return the local AffineTransform of the WCS at a given point.

        This is basically the same as wcs.local(...), but the return value is 
        guaranteed to be an AffineTransform, which has a few extra methods
        that are useful in some situations.  e.g. you can directly access
        the jacobian matrix to do calculations based on that. 
        
        If you do not need the extra functionality, then you should use local
        instead, since it may be more efficient.

        @param image_pos    The image coordinate position (for variable WCS objects)
        @param world_pos    The world coordinate position (for variable WCS objects)
        @returns local_wcs  An AffineTransform object
        """
        return self.local(image_pos, world_pos)._toAffine()

class PixelScale(BaseWCS):
    """This is the simplest possible WCS transformation.  It only involves a unit conversion
    from pixels to arcsec (or whatever units you want to take for your world coordinate system).

    The conversion functions are:

        u = x * scale
        v = y * scale

    Initialization
    --------------
    A PixelScale object is initialized with the command:

        wcs = galsim.PixelScale(scale)

    @param scale        The pixel scale, typically in units of arcsec/pixel.
    """
    _req_params = { "scale" : float }
    _opt_params = {}
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, scale):
        self._is_local = True
        self._scale = scale

    # Help make sure PixelScale is read-only.
    @property
    def scale(self): return self._scale

    def _posToWorld(self, image_pos):
        if not(isinstance(image_pos, galsim.PositionD) or isinstance(image_pos, galsim.PositionI)):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        return galsim.PositionD(image_pos.x * self._scale, image_pos.y * self._scale)

    def _posToImage(self, world_pos):
        if not isinstance(world_pos, galsim.PositionD):
            raise TypeError("toImage requires a PositionD argument")
        return galsim.PositionD(world_pos.x / self._scale, world_pos.y / self._scale)

    def _profileToWorld(self, image_profile):
        return image_profile.createDilated(self._scale)

    def _profileToImage(self, world_profile):
        return world_profile.createDilated(1./self._scale)

    def _pixelArea(self):
        return self._scale**2

    def _minScale(self):
        return self._scale

    def _maxScale(self):
        return self._scale

    def _toAffine(self):
        return AffineTransform(self._scale, 0., 0., self._scale)

    def copy(self):
        return PixelScale(self._scale)

    def __eq__(self, other):
        if not isinstance(other, PixelScale): 
            return False
        else: 
            return self._scale == other._scale

    def __ne__(self, other):
        return not self.__eq__(other)
 

class ShearWCS(BaseWCS):
    """This WCS is a uniformly sheared coordinate system.

    The conversion functions are:

        u = (x + g1 x + g2 y) * scale / sqrt(1-g1**2-g2**2)
        v = (y - g1 y + g2 x) * scale / sqrt(1-g1**2-g2**2)

    Initialization
    --------------
    A ShearWCS object is initialized with the command:

        wcs = galsim.ShearWCS(scale, shear)

    @param scale        The pixel scale, typically in units of arcsec/pixel.
    @param shear        The shear, which should be a galsim.Shear instance.
    """
    _req_params = { "scale" : float, "shear" : galsim.Shear }
    _opt_params = {}
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, scale, shear):
        import numpy
        self._is_local = True
        self._scale = scale
        self._shear = shear
        self._g1 = shear.g1
        self._g2 = shear.g2
        self._gsq = self._g1**2 + self._g2**2
        self._gfactor = 1. / numpy.sqrt(1. - self._gsq)

    # Help make sure ShearWCS is read-only.
    @property
    def scale(self): return self._scale
    @property
    def shear(self): return self._shear

    def _posToWorld(self, image_pos):
        if not(isinstance(image_pos, galsim.PositionD) or isinstance(image_pos, galsim.PositionI)):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        u = image_pos.x * (1.+self._g1) + image_pos.y * self._g2
        v = image_pos.y * (1.-self._g1) + image_pos.x * self._g2
        u *= self._scale * self._gfactor
        v *= self._scale * self._gfactor
        return galsim.PositionD(u,v)

    def _posToImage(self, world_pos):
        if not isinstance(world_pos, galsim.PositionD):
            raise TypeError("toImage requires a PositionD argument")
        # The inverse transformation is
        # x = (u - g1 u - g2 v) / scale / sqrt(1-|g|^2)
        # y = (v + g1 v - g2 u) / scale / sqrt(1-|g|^2)
        x = world_pos.x * (1.-self._g1) - world_pos.y * self._g2
        y = world_pos.y * (1.+self._g1) - world_pos.x * self._g2
        x *= self._gfactor / self._scale
        y *= self._gfactor / self._scale
        return galsim.PositionD(x,y)

    def _profileToWorld(self, image_profile):
        world_profile = image_profile.createDilated(self._scale)
        world_profile.applyShear(self.shear)
        return world_profile

    def _profileToImage(self, world_profile):
        image_profile = world_profile.createDilated(1./self._scale)
        image_profile.applyShear(-self.shear)
        return image_profile

    def _pixelArea(self):
        return self._scale**2

    def _minScale(self):
        # min stretch is (1-|g|) / sqrt(1-|g|^2)
        import numpy
        return self._scale * (1. - numpy.sqrt(self._gsq)) * self._gfactor

    def _maxScale(self):
        # max stretch is (1+|g|) / sqrt(1-|g|^2)
        import numpy
        return self._scale * (1. + numpy.sqrt(self._gsq)) * self._gfactor

    def _toAffine(self):
        return AffineTransform(
            (1.+self._g1) * self._scale * self._gfactor, 
            self._g2 * self._scale * self._gfactor,
            self._g2 * self._scale * self._gfactor,
            (1.-self._g1) * self._scale * self._gfactor)

    def copy(self):
        return ShearWCS(self.scale, self.shear)

    def __eq__(self, other):
        if not isinstance(other, ShearWCS): 
            return False
        else: 
            return self.scale == other.scale and self.shear == other.shear

    def __ne__(self, other):
        return not self.__eq__(other)


class AffineTransform(BaseWCS):
    """This WCS is the most general linear transformation.  It involves a 2x2 Jacobian
    matrix and an offset.  You can provide the offset in terms of either the image_pos
    (x0,y0) where (u,v) = (0,0), or the world_pos (u0,v0) where (x,y) = (0,0). 
    Or, in fact, you may provide both, in which case the image_pos (x0,y0) corresponds
    to the world_pos (u0,v0).

    The conversion functions are:

        u = dudx (x-x0) + dudy (y-y0) + u0
        v = dvdx (x-x0) + dvdy (y-y0) + v0

    Initialization
    --------------
    An AffineTransform object is initialized with the command:

        wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, image_origin=None, world_origin=None)

    @param dudx           du/dx
    @param dudy           du/dy
    @param dvdx           dv/dx
    @param dvdy           dv/dy
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI object.
                          [ Default: `image_origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD object.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "dudx" : float, "dudy" : float, "dvdx" : float, "dvdy" : float }
    _opt_params = { "image_origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, dudx, dudy, dvdx, dvdy, image_origin=None, world_origin=None):
        self._dudx = dudx
        self._dudy = dudy
        self._dvdx = dvdx
        self._dvdy = dvdy
        self._det = dudx * dvdy - dudy * dvdx
        if image_origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = image_origin.x
            self._y0 = image_origin.y
        if world_origin == None:
            self._u0 = 0
            self._v0 = 0
        else:
            self._u0 = world_origin.x
            self._v0 = world_origin.y
        self._is_local = self._x0 == 0. and self._y0 == 0. and self._u0 == 0. and self._v0 == 0.

    @property
    def dudx(self): return self._dudx
    @property
    def dudy(self): return self._dudy
    @property
    def dvdx(self): return self._dvdx
    @property
    def dvdy(self): return self._dvdy

    @property
    def image_origin(self): return galsim.PositionD(self._x0, self._y0)
    @property
    def world_origin(self): return galsim.PositionD(self._u0, self._v0)

    def _posToWorld(self, image_pos):
        if not(isinstance(image_pos, galsim.PositionD) or isinstance(image_pos, galsim.PositionI)):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        x = image_pos.x
        y = image_pos.y
        u = self._dudx * (x-self._x0) + self._dudy * (y-self._y0) + self._u0
        v = self._dvdx * (x-self._x0) + self._dvdy * (y-self._y0) + self._v0
        return galsim.PositionD(u,v)

    def _posToImage(self, world_pos):
        if not isinstance(world_pos, galsim.PositionD):
            raise TypeError("toImage requires a PositionD argument")
        #  J = ( dudx  dudy )
        #      ( dvdx  dvdy )
        #  J^-1 = (1/det) (  dvdy  -dudy )
        #                 ( -dvdx   dudx )
        u = world_pos.x
        v = world_pos.y
        x = (self._dvdy * (u-self._u0) - self._dudy * (v-self._v0))/self._det + self._x0
        y = (-self._dvdx * (u-self._u0) + self._dudx * (v-self._v0))/self._det + self._y0
        return galsim.PositionD(x,y)

    def _profileToWorld(self, image_profile):
        ret = image_profile.createTransformed(self._dudx, self._dudy, self._dvdx, self._dvdy)
        ret.scaleFlux(1./self._det)
        return ret

    def _profileToImage(self, world_profile):
        ret = world_profile.createTransformed(self._dvdy/self._det, -self._dudy/self._det,
                                               -self._dvdx/self._det, self._dudx/self._det)
        ret.scaleFlux(self._det)
        return ret

    def _pixelArea(self):
        return self._det

    def _minScale(self):
        # The corners of the parallelogram defined by:
        # J = ( a  b )
        #     ( c  d )
        # are:
        #     (0,0)  (a,c)
        #     (b,d)  (a+b,c+d)
        # That is, the corners of the unit square with positions (0,0), (0,1), (1,0), (1,1) 
        # are mapped onto these points.
        #
        # So the two diagonals are from (0,0) to (a+b, c+d) and from (a,c) to (b,d)
        # We divide these values by sqrt(2) to account for the fact that the original
        # square had a distance of sqrt(2) for each of these distances.
        import numpy
        d1 = numpy.sqrt( 0.5* ((self._dudx+self._dudy)**2 + (self._dvdx+self._dvdy)**2) )
        d2 = numpy.sqrt( 0.5* ((self._dudx-self._dudy)**2 + (self._dvdx-self._dvdy)**2) )
        return min(d1,d2)

    def _maxScale(self):
        import numpy
        d1 = numpy.sqrt( 0.5* ((self._dudx+self._dudy)**2 + (self._dvdx+self._dvdy)**2) )
        d2 = numpy.sqrt( 0.5* ((self._dudx-self._dudy)**2 + (self._dvdx-self._dvdy)**2) )
        return max(d1,d2)

    def _toAffine(self):
        return self

    def _local(self, image_pos, world_pos):
        return self.atOrigin(image_origin=galsim.PositionD(0.,0.),
                             world_origin=galsim.PositionD(0.,0.))

    def copy(self):
        return AffineTransform(self._dudx, self._dudy, self._dvdx, self._dvdy, 
                               self.image_origin, self.world_origin)

    def atOrigin(self, image_origin=None, world_origin=None):
        """Make a copy of this AffineTransform, but set new values for the image_origin
        and/or the world_origin.
        """
        if image_origin is None: image_origin = self.image_origin
        if world_origin is None: world_origin = self.world_origin
        return AffineTransform(self._dudx, self._dudy, self._dvdx, self._dvdy, 
                               image_origin, world_origin)

    def __eq__(self, other):
        if not isinstance(other, AffineTransform): 
            return False
        else: 
            return (
                self._dudx == other._dudx and 
                self._dudy == other._dudy and 
                self._dvdx == other._dvdx and 
                self._dvdy == other._dvdy and 
                self._x0 == other._x0 and 
                self._y0 == other._y0 and 
                self._u0 == other._u0 and 
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)

class UVFunction(BaseWCS):
    """This WCS takes two arbitrary functions for u(x,y) and v(x,y).

    The ufunc and vfunc parameters may be:
        - python functions that take (x,y) arguments
        - python objects with a __call__ method that takes (x,y) arguments
        - strings which can be parsed with eval('lambda x,y: '+str)

    Initialization
    --------------
    A UVFunction object is initialized with the command:

        wcs = galsim.UVFunction(ufunc, vfunc, image_origin=None, world_origin=None)

    @param ufunc          The function u(x,y)
    @param vfucn          The function v(x,y)
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI object.
                          [ Default: `image_origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD object.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "ufunc" : str, "vfunc" : str }
    _opt_params = { "image_origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, ufunc, vfunc, image_origin=None, world_origin=None):
        if isinstance(ufunc, basestring):
            self._ufunc = eval('lambda x,y : ' + ufunc)
        else:
            self._ufunc = ufunc
        if isinstance(vfunc, basestring):
            self._vfunc = eval('lambda x,y : ' + vfunc)
        else:
            self._vfunc = vfunc

        if image_origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = image_origin.x
            self._y0 = image_origin.y
        if world_origin == None:
            self._u0 = 0
            self._v0 = 0
        else:
            self._u0 = world_origin.x
            self._v0 = world_origin.y

        self._is_local = False

    @property
    def ufunc(self): return self._ufunc
    @property
    def vfunc(self): return self._vfunc
    @property
    def image_origin(self): return galsim.PositionD(self._x0, self._y0)
    @property
    def world_origin(self): return galsim.PositionD(self._u0, self._v0)

    def _u(self, x, y):
        return self._ufunc(x-self._x0, y-self._y0) + self._u0

    def _v(self, x, y):
        return self._vfunc(x-self._x0, y-self._y0) + self._v0
        
    def _posToWorld(self, image_pos):
        if not(isinstance(image_pos, galsim.PositionD) or isinstance(image_pos, galsim.PositionI)):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        u = self._u(image_pos.x, image_pos.y)
        v = self._v(image_pos.x, image_pos.y)
        return galsim.PositionD(u,v)

    def _posToImage(self, world_pos):
        raise NotImplementedError("World -> Image direction not implemented for UVFunction")

    def _local(self, image_pos, world_pos):
        if world_pos is not None:
            raise NotImplementedError('UVFunction.local() cannot take world_pos.')
        if image_pos is None:
            raise TypeError('UVFunction.local() requires an image_pos argument')
        x0 = image_pos.x
        y0 = image_pos.y
        u0 = self._u(x0,y0)
        v0 = self._v(x0,y0)
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1
        dudx = 0.5*(self._u(x0+dx,y0) - self._u(x0-dx,y0))/dx
        dudy = 0.5*(self._u(x0,y0+dy) - self._u(x0,y0-dy))/dy
        dvdx = 0.5*(self._v(x0+dx,y0) - self._v(x0-dx,y0))/dx
        dvdy = 0.5*(self._v(x0,y0+dy) - self._v(x0,y0-dy))/dy

        return AffineTransform(dudx, dudy, dvdx, dvdy)

    def copy(self):
        return UVFunction(self._ufunc, self._vfunc, self.image_origin, self.world_origin)

    def atOrigin(self, image_origin=None, world_origin=None):
        """Make a copy of this AffineTransform, but set new values for the image_origin
        and/or the world_origin.
        """
        if image_origin is None: image_origin = self.image_origin
        if world_origin is None: world_origin = self.world_origin
        return UVFunction(self._ufunc, self._vfunc, image_origin, world_origin)

    def __eq__(self, other):
        if not isinstance(other, UVFunction):
            return False
        else: 
            return (
                self._ufunc == other._ufunc and 
                self._vfunc == other._vfunc and 
                self._x0 == other._x0 and 
                self._y0 == other._y0 and 
                self._u0 == other._u0 and 
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)
 

def makeAffineFromNumericalRaDec(ra, dec, dx, dy):
    """Convert a list of list of ra, dec values for (0,0), (dx,0), (-dx,0), (0,dy), and (0,-dy)
       into an AffineTransform object.  The input ra, dec values should be in degrees.
    """
    ra0, ra1, ra2, ra3, ra4 = ra
    dec0, dec1, dec2, dec3, dec4 = dec
    # Note: This currently doesn't use position (ra0, dec0), but the option is here in case it 
    # would be useful in the future to have some check that the central value is consistent with 
    # the derivatives found from the +-dx,dy positions.
    # However, dec0 is used for the cos factor.
    
    import numpy
    # Note: our convention is that ra increases to the left! 
    # i.e. The u,v plane is the tangent plane as seen from Earth with +v pointing
    # north, and +u pointing west.
    # That means the du values are the negative of dra.
    dudx = -0.5*(ra1 - ra2)/dx * numpy.cos(dec0 * galsim.degrees / galsim.radians)
    dudy = -0.5*(ra3 - ra4)/dy * numpy.cos(dec0 * galsim.degrees / galsim.radians)
    dvdx = 0.5*(dec1 - dec2)/dx
    dvdy = 0.5*(dec3 - dec4)/dy

    # These values are all in degrees.  Convert to arcsec as per our usual standard.
    return AffineTransform(dudx*3600., dudy*3600., dvdx*3600., dvdy*3600.)


class AstropyWCS(BaseWCS):
    """This WCS uses astropy.wcs to read WCS information from a FITS file.

    Initialization
    --------------
    An AstropyWCS object is initialized with one of the following commands:

        wcs = galsim.AstropyWCS(file_name=file_name)  # Open a file on disk
        wcs = galsim.AstropyWCS(hdu=hdu)              # Use an existing pyfits hdu
        wcs = galsim.AstropyWCS(wcs=wcs)              # Use an axisting astropy.wcs.WCS object

    Exactly one of the parameters file_name, hdu or wcs is required.  Also, since the 
    most common usage will probably be the first, you can also give a file_name without it 
    being named:

        wcs = galsim.AstropyWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Either an open pyfits (or astropy.io) hdu, or the number of the HDU to 
                          use if reading from a file.  The default is to use either the primary
                          or first extension as appropriate for the given compression.
                          (e.g. for rice, the first extension is the one you normally want.)
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    @param wcs            An existing astropy.wcs.WCS object [ Default: `wcs = None` ]
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI object.
                          [ Default: `image_origin = None` ]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "image_origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, compression='auto',
                 wcs=None, image_origin=None):
        if file_name is not None:
            from galsim import pyfits
            if ( isinstance(hdu, pyfits.CompImageHDU) or 
                 isinstance(hdu, pyfits.ImageHDU) or
                 isinstance(hdu, pyfits.PrimaryHDU) ):
                raise TypeError("Cannot provide both file_name and pyfits hdu")
            if wcs is not None:    
                raise TypeError("Cannot provide both file_name and wcs")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)

        if hdu is not None:
            import astropy.wcs
            if wcs is not None:    
                raise TypeError("Cannot provide both pyfits hdu and wcs")
            wcs = astropy.wcs.WCS(hdu.header)
        if wcs is None:
            raise TypeError("Must provide one of file_name, hdu (as a pyfits HDU), or wcs")
            
        self._wcs = wcs
        self._is_local = False

        if image_origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = image_origin.x
            self._y0 = image_origin.y

        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

    @property
    def wcs(self): return self._wcs
    @property
    def image_origin(self): return galsim.PositionD(self._x0, self._y0)

    def _posToWorld(self, image_pos):
        if not(isinstance(image_pos, galsim.PositionD) or isinstance(image_pos, galsim.PositionI)):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        x = image_pos.x - self._x0
        y = image_pos.y - self._y0
        # Apparently, the returned values aren't _necessarily_ (ra, dec).  They could be 
        # (dec, ra) instead!  But if you add ra_dec_order=True, then it will be (ra, dec).
        # I can't imagnie why that isn't the default, but there you go.
        try:
            # This currently fails with an AttributeError about astropy.wcs.Wcsprm.lattype
            # c.f. https://github.com/astropy/astropy/pull/1463
            # Once they fix it, this is what we want.
            ra, dec = self._wcs.all_pix2world( [ [x, y] ], 1, ra_dec_order=True)[0]
        except AttributeError:
            # Until then, just assume that the returned values really are ra, dec.
            ra, dec = self._wcs.all_pix2world( [ [x, y] ], 1)[0]

        # astropy.wcs returns (ra, dec) in degrees.  Convert to our CelestialCoord class.
        return galsim.CelestialCoord(ra * galsim.degrees, dec * galsim.degrees)

    def _posToImage(self, world_pos):
        ra = world_pos.ra / galsim.degrees
        dec = world_pos.dec / galsim.degrees
        # Here we have to work around another astropy.wcs bug.  The way they use scipy's
        # Broyden's method doesn't work.  So I implement a fix here.
        try:
            # Try their version first (with and without ra_dec_order) in case they fix this.
            import warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, y = self._wcs.all_world2pix( [ [ra, dec] ], 1, ra_dec_order=True)[0]
            except AttributeError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, y = self._wcs.all_world2pix( [ [ra, dec] ], 1)[0]
        except:
            # This section is basically a copy of astropy.wcs's _all_world2pix function, but
            # simplified a bit to remove some features we don't need, and with corrections 
            # to make it work correctly.
            import astropy.wcs
            import scipy.optimize
            import numpy
            import warnings

            world = [ra,dec]
            origin = 1
            tolerance = 1.e-6

            # This call emits a RuntimeWarning about:
            #     /sw/lib/python2.7/site-packages/scipy/optimize/nonlin.py:943: RuntimeWarning: invalid value encountered in divide
            #       d = v / vdot(df, v)
            # It seems to be harmless, so we explicitly ignore it here:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x0 = self._wcs.wcs_world2pix(numpy.atleast_2d(world), origin).flatten()

            func = lambda pix: (self._wcs.all_pix2world(numpy.atleast_2d(pix),
                                origin) - world).flatten()

            # This is the main bit that the astropy function is missing.
            # The scipy.optimize.broyden1 function can't handle starting at exactly the right 
            # solution.  It iterates to its limit and then ends with:
            #     Traceback (most recent call last):
            #       File "test_wcs.py", line 654, in <module>
            #         test_astropywcs()
            #       File "test_wcs.py", line 645, in test_astropywcs
            #         pos = wcs.toImage(galsim.CelestialCoord(ra,dec))
            #       File "/sw/lib/python2.7/site-packages/galsim/wcs.py", line 106, in toImage
            #         return self._posToImage(arg)
            #       File "/sw/lib/python2.7/site-packages/galsim/wcs.py", line 793, in _posToImage
            #         soln = scipy.optimize.broyden1(func, x0, x_tol=tolerance, verbose=True, alpha=alpha)
            #       File "<string>", line 8, in broyden1
            #       File "/sw/lib/python2.7/site-packages/scipy/optimize/nonlin.py", line 331, in nonlin_solve
            #         raise NoConvergence(_array_like(x, x0))
            #     scipy.optimize.nonlin.NoConvergence: [ 113.74961526  179.99982209]
            #
            # Providing a good estimate of the scale size gets rid of this.  And even if we aren't
            # starting at exactly the right value, it is hugely more efficient to give it an 
            # estimate of alpha, since it is not typically near unity in this case, so it is much
            # faster to start with something closer to the right value.
            alpha = numpy.mean(numpy.abs(self._wcs.wcs.get_cdelt()))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                soln = scipy.optimize.broyden1(func, x0, x_tol=tolerance, alpha=alpha)
            x,y = soln

        return galsim.PositionD(x + self._x0, y + self._y0)

    def _local(self, image_pos, world_pos):
        if world_pos is not None:
            image_pos = self._posToImage(world_pos)
        if image_pos is None:
            raise TypeError('AstropyWCS.local() requires an image_pos or world_pos argument')
        x0 = image_pos.x - self._x0
        y0 = image_pos.y - self._y0
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        # all_pix2world can take an array to do everything at once.
        try:
            world = self._wcs.all_pix2world(
                    [ [x0,y0], [x0+dx,y0], [x0-dx,y0], [x0,y0+dy], [x0,y0-dy] ], 1,
                    ra_dec_order=True)
        except AttributeError:
            world = self._wcs.all_pix2world(
                    [ [x0,y0], [x0+dx,y0], [x0-dx,y0], [x0,y0+dy], [x0,y0-dy] ], 1)

        # Convert to a list of ra and dec separately
        ra = [ w[0] for w in world ]
        dec = [ w[1] for w in world ]

        return makeAffineFromNumericalRaDec(ra, dec, dx, dy)

    def copy(self):
        return AstropyWCS(wcs=self._wcs, image_origin=self.image_origin)

    def atOrigin(self, image_origin=None):
        """Make a copy of this AstropyWCS, but set a new value for the image_origin.
        """
        if image_origin is None: image_origin = self.image_origin
        return AstropyWCS(wcs=self._wcs, image_origin=image_origin)

    def __eq__(self, other):
        if not isinstance(other, AstropyWCS):
            return False
        else: 
            return (
                self._wcs == other._wcs and 
                self._x0 == other._x0 and 
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)

class PyAstWCS(BaseWCS):
    """This WCS uses PyAst (the python front end for the Starlink AST code) to read WCS 
    information from a FITS file.

    Initialization
    --------------
    A PyAstWCS object is initialized with one of the following commands:

        wcs = galsim.PyAstWCS(file_name=file_name)  # Open a file on disk
        wcs = galsim.PyAstWCS(hdu=hdu)              # Use an existing pyfits hdu
        wcs = galsim.PyAstWCS(wcsinfo=wcsinfo)      # Use an axisting starlink.Ast.FrameSet object

    Exactly one of the parameters file_name, hdu or wcsinfo is required.  Also, since the 
    most common usage will probably be the first, you can also give a file_name without it 
    being named:

        wcs = galsim.PyAstWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Either an open pyfits (or astropy.io) hdu, or the number of the HDU to 
                          use if reading from a file.  The default is to use either the primary
                          or first extension as appropriate for the given compression.
                          (e.g. for rice, the first extension is the one you normally want.)
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    @param wcsinfo        An existing starlink.Ast.WcsMap object [ Default: `wcsinfo = None` ]
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI object.
                          [ Default: `image_origin = None` ]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "image_origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, compression='auto',
                 wcsinfo=None, image_origin=None):
        # Note: More much of this class implementation, I've followed the example provided here:
        #    http://dsberry.github.io/starlink/node4.html
        if file_name is not None:
            from galsim import pyfits
            if ( isinstance(hdu, pyfits.CompImageHDU) or 
                 isinstance(hdu, pyfits.ImageHDU) or
                 isinstance(hdu, pyfits.PrimaryHDU) ):
                raise TypeError("Cannot provide both file_name and pyfits hdu")
            if wcsinfo is not None:    
                raise TypeError("Cannot provide both file_name and wcsinfo")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)

        if hdu is not None:
            import starlink.Ast
            import starlink.Atl
            if wcsinfo is not None:    
                raise TypeError("Cannot provide both pyfits hdu and wcsinfo")
            fitschan = starlink.Ast.FitsChan( starlink.Atl.PyFITSAdapter(hdu) )
            wcsinfo = fitschan.read()
        if wcsinfo is None:
            raise TypeError("Must provide one of file_name, hdu (as a pyfits HDU), or wcsinfo")

        #  Check that the FITS header contained WCS in a form that can be
        #  understood by AST.
        if wcsinfo == None:
            raise RuntimeError("Failed to read WCS information from fits file")
        #  Check that the object read from the FitsChan is of the expected class
        #  (Ast.FrameSet).
        elif not isinstance( wcsinfo, starlink.Ast.FrameSet ):
            raise RuntimeError("A "+wcsinfo.__class__.__name__+" was read from test.fit - "+
                                "was expecting a starlink.Ast.FrameSet")
        #  We can only handle WCS with 2 pixel axes (given by Nin) and 2 WCS axes 
        # (given by Nout).
        elif wcsinfo.Nin != 2 or wcsinfo.Nout != 2:
            raise RuntimeError("The world coordinate system is not 2-dimensional")

        self._wcsinfo = wcsinfo
        self._is_local = False

        if image_origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = image_origin.x
            self._y0 = image_origin.y

        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

    @property
    def wcsinfo(self): return self._wcsinfo
    @property
    def image_origin(self): return galsim.PositionD(self._x0, self._y0)

    def _posToWorld(self, image_pos):
        if not(isinstance(image_pos, galsim.PositionD) or isinstance(image_pos, galsim.PositionI)):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        x = image_pos.x - self._x0
        y = image_pos.y - self._y0
        ra, dec = self._wcsinfo.tran( [ [x], [y] ] )
        # PyAst returns ra, dec in radians
        return galsim.CelestialCoord(ra[0] * galsim.radians, dec[0] * galsim.radians)

    def _posToImage(self, world_pos):
        ra = world_pos.ra / galsim.radians
        dec = world_pos.dec / galsim.radians
        x,y = self._wcsinfo.tran( [ [ra], [dec] ], False)
        return galsim.PositionD(x[0] + self._x0, y[0] + self._y0)

    def _local(self, image_pos, world_pos):
        if world_pos is not None:
            image_pos = self._posToImage(world_pos)
        if image_pos is None:
            raise TypeError('AstropyWCS.local() requires an image_pos or world_pos argument')

        x0 = image_pos.x - self._x0
        y0 = image_pos.y - self._y0
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        # wcsinfo.tran can take arrays to do everything at once.
        ra, dec = self._wcsinfo.tran( [ [x0, x0+dx, x0-dx, x0,    x0],
                                        [y0, y0,    y0,    y0+dy, y0-dy ] ])

        # Convert to degrees as needed by makeAffineFromNumericalRaDec:
        ra = [ r * galsim.radians / galsim.degrees for r in ra ]
        dec = [ d * galsim.radians / galsim.degrees for d in dec ]
        return makeAffineFromNumericalRaDec(ra, dec, dx, dy)

    def copy(self):
        return PyAstWCS(wcsinfo=self._wcsinfo, image_origin=self.image_origin)

    def atOrigin(self, image_origin=None):
        """Make a copy of this AstropyWCS, but set a new value for the image_origin.
        """
        if image_origin is None: image_origin = self.image_origin
        return AstropyWCS(wcs=self._wcs, image_origin=image_origin)

    def __eq__(self, other):
        if not isinstance(other, AstropyWCS):
            return False
        else: 
            return (
                self._wcs == other._wcs and 
                self._x0 == other._x0 and 
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)

