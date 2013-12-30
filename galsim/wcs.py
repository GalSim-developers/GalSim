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
        #return AffineTransform(self._scale, 0., 0., self._scale)
        raise NotImplementedError("Waiting until AffineTransform is defined")

    def copy(self):
        return PixelScale(self._scale)

    def __eq__(self, other):
        if not isinstance(other, PixelScale): 
            return False
        else: 
            return self._scale == other._scale

    def __ne__(self, other):
        return not self.__eq__(other)
 

