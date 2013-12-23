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
        #     _is_variable     boolean variable declaring whether the pixel is variable
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
        # Variable WCS classes must define the following:
        #
        #     _local           function returning a non-variable WCS at a given location


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
            if self.isVariable():
                return self.local(**kwargs)._profileToWorld(arg)
            else:
                return self._profileToWorld(arg)
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
            if self.isVariable():
                return self.local(**kwargs)._profileToImage(arg)
            else:
                return self._profileToImage(arg)
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

    def isVariable(self):
        """Return whether the WCS solution has a pixel that varies in either area
        or shape across the field.
        """
        return self._is_variable

    def local(self, image_pos=None, world_pos=None):
        """Return the local linear approximation of the WCS at a given point.

        @param image_pos    The image coordinate position (for variable WCS objects)
        @param world_pos    The world coordinate position (for variable WCS objects)
        @returns local_wcs  A WCS object with wcs.isVariable() == False
        """
        if image_pos and world_pos:
            raise TypeError("Only one of image_pos or world_pos may be provided")
        if self.isVariable():
            return self._local(image_pos, world_pos)
        else:
            return self

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
        self._is_variable = False
        self._scale = scale

    # Help make sure PixelScale is read-only.
    @property
    def scale(self): return self._scale

    def _posToWorld(self, image_pos):
        if not(isinstance(image_pos, galsim.PositionD) or isinstance(image_pos, galsim.PositionI)):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        return image_pos * self._scale

    def _posToImage(self, world_pos):
        if not isinstance(world_pos, galsim.PositionD):
            raise TypeError("toImage requires a PositionD argument")
        return world_pos / self._scale

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
        self._is_variable = False
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
        self._is_variable = False
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
        raise NotImplementedError("TODO")

    def _maxScale(self):
        raise NotImplementedError("TODO")

    def _toAffine(self):
        return self

    def copy(self):
        return AffineTransform(self._dudx, self._dudy, self._dvdx, self._dvdy, 
                               galsim.PositionD(self._x0, self._y0),
                               galsim.PositionD(self._u0, self._v0))

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

