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
        #     _atOrigin        function returning a WCS with a given new origin point


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

    def atNewOrigin(self, new_origin):
        """Returns a new WCS object that has the origin point moved to a new location.

        If you change the definition of the origin of an image with setOrigin or 
        setCenter, we need to adjust the WCS to use the same new defintion for what
        (x,y) means.  The way we do this is to make a new wcs using the new value 
        for what the image means by (x,y) = (0,0).

        The new_origin argument is the image position in the current WCS that you
        want to be defined as (0,0) in the returned WCS.

        The following code should help make clear what we mean by this:

            world_cen = wcs.toWorld(image_cen)
            world_pos = wcs.toWorld(image_pos)

            new_wcs = wcs.atNewOrigin(image_cen)
            world_cen2 = new_wcs.toWorld(PositionD(0.,0.))
            world_pos2 = new_wcs.toWorld(image_pos - image_cen)

            assert world_cen == world_cen2
            assert world_pos == world_pos2

        Note: In actual operations, the above asserts might fail due to numerical
        rounding differences.  But the point is that they should be essentially the
        same positions in world coordinates.

        @param new_origin   The image coordinate that you want to become (0,0)
        @returns new_wcs    A new WCS with that position as the origin.
        """
        return self._atOrigin(new_origin)


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
        self.scale = scale

    def _posToWorld(self, image_pos):
        if not(isinstance(image_pos, galsim.PositionD) or isinstance(image_pos, galsim.PositionI)):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        return image_pos * self.scale

    def _posToImage(self, world_pos):
        if not isinstance(world_pos, galsim.PositionD):
            raise TypeError("toImage requires a PositionD argument")
        return world_pos / self.scale

    def _profileToWorld(self, image_profile):
        return image_profile.createDilated(self.scale)

    def _profileToImage(self, world_profile):
        return world_profile.createDilated(1./self.scale)

    def _pixelArea(self):
        return self.scale*self.scale

    def _minScale(self):
        return self.scale

    def _maxScale(self):
        return self.scale

    def _toAffine(self):
        raise NotImplementedError("Waiting until AffineTransform is defined")

    def copy(self):
        return PixelScale(self.scale)

    def __eq__(self, other):
        if not isinstance(other, PixelScale): 
            return False
        else: 
            return self.scale == other.scale

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
        self._is_variable = False
        self.scale = scale
        self.shear = shear

    def _posToWorld(self, image_pos):
        if not(isinstance(image_pos, galsim.PositionD) or isinstance(image_pos, galsim.PositionI)):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        import numpy
        g1 = self.shear.g1
        g2 = self.shear.g2
        gsq = g1*g1+g2*g2
        u = image_pos.x * (1.+g1) + image_pos.y * g2
        v = image_pos.y * (1.-g1) + image_pos.x * g2
        factor = self.scale / numpy.sqrt(1.-gsq)
        u *= factor
        v *= factor
        return galsim.PositionD(u,v)

    def _posToImage(self, world_pos):
        if not isinstance(world_pos, galsim.PositionD):
            raise TypeError("toImage requires a PositionD argument")
        # The inverse transformation is
        # x = (u - g1 u - g2 v) / scale / sqrt(1-g1**2-g2**2)
        # y = (v + g1 v - g2 u) / scale / sqrt(1-g1**2-g2**2)
        import numpy
        g1 = self.shear.g1
        g2 = self.shear.g2
        gsq = g1*g1+g2*g2
        x = world_pos.x * (1.-g1) - world_pos.y * g2
        y = world_pos.y * (1.+g1) - world_pos.x * g2
        factor = 1. / (self.scale * numpy.sqrt(1.-gsq))
        x *= factor
        y *= factor
        return galsim.PositionD(x,y)

    def _profileToWorld(self, image_profile):
        world_profile = image_profile.createDilated(self.scale)
        world_profile.applyShear(self.shear)
        return world_profile

    def _profileToImage(self, world_profile):
        image_profile = world_profile.createDilated(1./self.scale)
        image_profile.applyShear(-self.shear)
        return image_profile

    def _pixelArea(self):
        return self.scale*self.scale

    def _minScale(self):
        # min stretch is (1-|g|) / sqrt(1-|g|^2)
        import numpy
        g1 = self.shear.g1
        g2 = self.shear.g2
        g = numpy.sqrt(g1*g1+g2*g2)
        return self.scale / numpy.sqrt(1.+g) 

    def _maxScale(self):
        # max stretch is (1+|g|) / sqrt(1-|g|^2)
        import numpy
        g1 = self.shear.g1
        g2 = self.shear.g2
        g = numpy.sqrt(g1*g1+g2*g2)
        return self.scale / numpy.sqrt(1.-g)

    def _toAffine(self):
        raise NotImplementedError("Waiting until AffineTransform is defined")

    def copy(self):
        return ShearWCS(self.scale, self.shear)

    def __eq__(self, other):
        if not isinstance(other, ShearWCS): 
            return False
        else: 
            return self.scale == other.scale and self.shear == other.shear

    def __ne__(self, other):
        return not self.__eq__(other)


