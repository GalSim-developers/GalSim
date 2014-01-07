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
    """The base class for all other kinds of WCS transformations.

    All the user-functions are defined here, which defines the common interface
    for all subclasses.

    There are two types of WCS classes that we implement.

    1. Local WCS classes are those which really just define a pixel size and shape.
       They implicitly have the origin in image coordinates correspond to the origin
       in world coordinates.  They primarily designed to handle local transformations
       at the location of a single galaxy, where it should usually be a good approximation
       to consider the pixel shape to be constant over the size of the galaxy.

       Currently we define the following local WCS classes:

            PixelScale
            ShearWCS
            JacobianWCS

    2. Non-local WCS classes may have a constant pixel size and shape, but they don't have to.
       They may also have an arbitrary origin in both image coordinates and world coordinates.
       Furthermore, the world coordinates may be either a regular Euclidean coordinate system 
       (using galsim.PositionD for the world positions) or coordinates on the celestial sphere 
       (using galsim.CelestialCoord for the world positions).

       Currently we define the following non-local WCS classes:

            OffsetWCS
            OffsetShearWCS
            AffineTransform
            UVFunction
            RaDecFunction
            AstropyWCS          -- requires astropy.wcs python module to be installed
            PyAstWCS            -- requires starlink.Ast python module to be installed
            WcsToolsWCS         -- requires wcstools command line functions to be installed
            GSFitsWCS           -- native code, but has less functionality than the above

    There is also a factory function called FitsWCS, which is intended to act like a 
    class initializer.  It tries to read a fits file using one of the above classes
    and returns an instance of whichever one it found was successful.  It should always
    be successful, since it's final attempt uses AffineTransform, which has reasonable 
    defaults when the WCS key words are not in the file, but of course this will only be 
    a very rough approximation of the true WCS.


    Some things you can do with a WCS instance:

    - Convert positions between image coordinates and world coordinates (sometimes referred
      to as sky coordinates):

                world_pos = wcs.toWorld(image_pos)
                image_pos = wcs.toImage(world_pos)

      Note: the transformation from world to image coordinates is not guaranteed to be
      implemented.  If it is not implemented for a particular WCS class, a NotImplementedError
      will be raised.

      The image_pos parameter should be a galsim.PositionD.  However, world_pos will
      be a galsim.CelestialCoord if the transformation is in terms of celestial coordinates
      (c.f. wcs.isCelestial()).  Otherwise, it will be a PositionD as well.

    - Convert a GSObject, which is naturally defined in world coordinates, to the equivalent
      profile using image coordinates (or vice versa):

                image_profile = wcs.toImage(world_profile)
                world_profile = wcs.toWorld(image_profile)

      For non-uniform WCS types (c.f. wcs.isUniform()), these need either an image_pos or
      world_pos parameter to say where this conversion should happen:

                image_profile = wcs.toImage(world_profile, image_pos=image_pos)

    - Construct a local linear approximation of a WCS at a given location:

                local_wcs = wcs.local(image_pos = image_pos)
                local_wcs = wcs.local(world_pos = world_pos)

      If wcs.toWorld(image_pos) is not implemented for a particular WCS class, then a
      NotImplementedError will be raised if you pass in a world_pos argument.

    - Construct a full affine approximation of a WCS at a given location:

                affine_wcs = wcs.affine(image_pos = image_pos)
                affine_wcs = wcs.affine(world_pos = world_pos)

      This preserves the transformation near the location of image_pos, but it is linear, so
      the transformed values may not agree as you get farther from the given point.

    - Shift a transformation to use a new location for what is currently considered
      image_pos = (0,0).  For local WCS types, this also converts to a non-local WCS.

                world_pos1 = wcs.toWorld(PositionD(0,0))
                shifted = wcs.setOrigin(image_origin)
                world_pos2 = shifted.toWorld(image_origin)
                # world_pos1 should be equal to world_pos2

    - Get some properties of the pixel size and shape:

                area = local_wcs.pixelArea()
                min_linear_scale = local_wcs.minLinearScale()
                max_linear_scale = local_wcs.maxLinearScale()
                jac = local_wcs.jacobian()
                # Use jac.dudx, jac.dudy, jac.dvdx, jac.dvdy

      Global WCS types also have these functions, but for them, you must supply either
      image_pos or world_pos.  So the following are equivalent:

                area = wcs.pixelArea(image_pos)
                area = wcs.local(image_pos).pixelArea()

    - Query some overall attributes of the WCS transformation:

                wcs.isLocal()       # is this a local WCS?
                wcs.isUniform()     # does this WCS have a uniform pixel size/shape?
                wcs.isCelestial()   # are the world coordinates on the celestial sphere?
                wcs.isPixelScale()  # is this a PixelScale or OffsetWCS?
    """
    def __init__(self):
        raise TypeError("BaseWCS is an abstract base class.  It cannot be instantiated.")

        # All derived classes must define the following:
        #
        #     _is_local         boolean variable declaring whether the WCS is local, linear
        #     _is_uniform       boolean variable declaring whether the pixels are uniform
        #     _is_celestial     boolean variable declaring whether the world coords are celestial
        #     _posToWorld       function converting image_pos to world_pos
        #     _posToImage       function converting world_pos to image_pos
        #     _setOrigin        function returning a version with a new origin (or origins).
        #     _writeHeader      function that writes the WCS to a fits header.
        #     _readHeader       static function that reads the WCS from a fits header.
        #     copy              return a copy
        #     __eq__            check if this equals another WCS
        #     __ne__            check if this is not equal to another WCS
        #     __repr__          convert to string
        #
        # Local WCS classes (i.e. those where (x,y) = (0,0) corresponds to (u,v) = (0,0)
        # and the pixel shape is invariant) must define the following:
        #
        #     _profileToWorld   function converting image_profile to world_profile
        #     _profileToImage   function converting world_profile to image_profile
        #     _pixelArea        function returning the pixel area
        #     _minScale         function returning the minimum linear pixel scale
        #     _maxScale         function returning the maximum linear pixel scale
        #     _toJacobian       function returning an equivalent JacobianWCS
        #
        # Non-local WCS classes must define the following:
        #
        #     _local            function returning a local WCS at a given location


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
            return self.profileToWorld(arg, **kwargs)
        else:
            return self.posToWorld(arg, **kwargs)

    def posToWorld(self, image_pos):
        """Convert a position from image coordinates to world coordinates

        This is equivalent to wcs.toWorld(image_pos).
        """
        if isinstance(image_pos, galsim.PositionI):
            image_pos = galsim.PositionD(image_pos.x, image_pos.y)
        elif not isinstance(image_pos, galsim.PositionD):
            raise TypeError("toWorld requires a PositionD or PositionI argument")
        return self._posToWorld(image_pos)

    def profileToWorld(self, image_profile, image_pos=None, world_pos=None):
        """Convert a profile from image coordinates to world coordinates

        This is equivalent to wcs.toWorld(image_profile, ...).
        """
        return self.local(image_pos, world_pos)._profileToWorld(image_profile)

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
            return self.profileToImage(arg, **kwargs)
        else:
            return self.posToImage(arg, **kwargs)

    def posToImage(self, world_pos):
        """Convert a position from world coordinates to image coordinates

        This is equivalent to wcs.toImage(world_pos).
        """
        if self._is_celestial and not isinstance(world_pos, galsim.CelestialCoord):
            raise TypeError("toImage requires a CelestialCoord argument")
        elif not self._is_celestial and isinstance(world_pos, galsim.PositionI):
            world_pos = galsim.PositionD(world_pos.x, world_pos.y)
        elif not self._is_celestial and not isinstance(world_pos, galsim.PositionD):
            raise TypeError("toImage requires a PositionD or PositionI argument")
        return self._posToImage(world_pos)

    def profileToImage(self, world_profile, image_pos=None, world_pos=None):
        """Convert a profile from world coordinates to image coordinates

        This is equivalent to wcs.toImage(world_profile, ...).
        """
        return self.local(image_pos, world_pos)._profileToImage(world_profile)

    def pixelArea(self, image_pos=None, world_pos=None):
        """Return the area of a pixel in arcsec**2 (or in whatever units you are using for
        world coordinates).

        For variable WCS transforms, you must provide either image_pos or world_pos
        to say where the pixel is located.

        @param image_pos    The image coordinate position (for variable WCS types)
        @param world_pos    The world coordinate position (for variable WCS types)
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

        @param image_pos    The image coordinate position (for variable WCS types)
        @param world_pos    The world coordinate position (for variable WCS types)
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

        @param image_pos    The image coordinate position (for variable WCS types)
        @param world_pos    The world coordinate position (for variable WCS types)
        @returns            The maximum pixel area in any direction in arcsec
        """
        return self.local(image_pos, world_pos)._maxScale()

    def isPixelScale(self):
        """Return whether the WCS transformation is a simple PixelScale or OffsetWCS.

        These are the simplest two WCS transformations.  PixelScale is local and OffsetWCS
        is non-local.  If an Image has one of these WCS transformations as its WCS, then 
        im.scale works to read and write the pixel scale.  If not, im.scale will raise a 
        TypeError exception.
        """
        return isinstance(self,PixelScale) or isinstance(self,OffsetWCS)

    def isLocal(self):
        """Return whether the WCS transformation is a local, linear approximation.

        There are two requirements for this to be true:
            1. The image position (x,y) = (0,0) is at the world position (u,v) = (0,0).
            2. The pixel area and shape do not vary with position.
        """
        return self._is_local

    def isUniform(self):
        """Return whether the pixels in this WCS have uniform size and shape"""
        return self._is_uniform

    def isCelestial(self):
        """Return whether the world coordinates are CelestialCoord (i.e. ra,dec).  """
        return self._is_celestial

    def local(self, image_pos=None, world_pos=None):
        """Return the local linear approximation of the WCS at a given point.

        @param image_pos    The image coordinate position (for variable WCS types)
        @param world_pos    The world coordinate position (for variable WCS types)
        @returns local_wcs  A WCS with wcs.isLocal() == True
        """
        if image_pos and world_pos:
            raise TypeError("Only one of image_pos or world_pos may be provided")
        if self._is_local:
            return self
        else:
            if not self._is_uniform and image_pos==None and world_pos==None:
                raise TypeError("Either image_pos or world_pos must be provided")
            return self._local(image_pos, world_pos)

    def jacobian(self, image_pos=None, world_pos=None):
        """Return the local JacobianWCS of the WCS at a given point.

        This is basically the same as wcs.local(...), but the return value is
        guaranteed to be a JacobianWCS, which can be useful in some situations,
        since you can access the values of the 2x2 Jacobian matrix directly:

                jac = wcs.jacobian(image_pos)
                x,y = np.meshgrid(np.arange(0,32,1), np.arange(0,32,1))
                u = jac.dudx * x + jac.dudy * y
                v = jac.dvdx * x + jac.dvdy * y
                ... use u,v values to work directly in sky coordinates.

        If you do not need the extra functionality, then you should use wcs.local()
        instead, since it may be more efficient.

        @param image_pos    The image coordinate position (for variable WCS types)
        @param world_pos    The world coordinate position (for variable WCS types)
        @returns local_wcs  A JacobianWCS
        """
        return self.local(image_pos, world_pos)._toJacobian()

    def affine(self, image_pos=None, world_pos=None):
        """Return the local AffineTransform of the WCS at a given point.

        This returns a linearized version of the current WCS at a given point.  It 
        returns an AffineTransform that is locally approximately the same as the WCS in 
        the vicinity of the given point.

        It is similar to jacobian(), except that this preserves the offset information
        between the image coordinates and world coordinates rather than setting both
        origins to (0,0).  Instead, the image origin is taken to be `image_pos`.
        
        For non-celestial coordinate systems, the world origin is taken to be 
        `wcs.toWorld(image_pos)`.  In fact, `wcs.affine(image_pos)` is really just
        shorthand for:
        
                wcs.jacobian(image_pos).setOrigin(image_pos, wcs.toWorld(image_pos))

        For celestial coordinate systems, there is not well-defined choice for the 
        origin of the Euclidean world coordinate system.  So we just take (u,v) = (0,0)
        at the given position.  So, `wcs.affine(image_pos)` is equivalent to:

                wcs.jacobian(image_pos).setOrigin(image_pos, galsim.PositionD(0,0))

        As usual, you may provide either `image_pos` or `world_pos` as you prefer.

        You can use the returned AffineTransform to access the relevant values of the 2x2 
        Jacobian matrix and the origins directly:

                affine = wcs.affine(image_pos)
                x,y = np.meshgrid(np.arange(0,32,1), np.arange(0,32,1))
                u = affine.dudx * (x-affine.x0) + jac.dudy * (y-affine.y0) + affine.u0
                v = affine.dvdx * (y-affine.y0) + jac.dvdy * (y-affine.y0) + affine.v0
                ... use u,v values to work directly in sky coordinates.

        @param image_pos        The image coordinate position (for variable WCS types)
        @param world_pos        The world coordinate position (for variable WCS types)
        @returns affine_wcs     An AffineTransform
        """
        jac = self.jacobian(image_pos, world_pos)
        # That call checked that only one of image_pos or world_pos is provided.
        if world_pos is not None:
            image_pos = self.toImage(world_pos)
        elif image_pos is None:
            # Both are None.  Must be a local WCS
            image_pos = galsim.PositionD(0,0)

        if self._is_celestial:
            return jac.setOrigin(image_pos, galsim.PositionD(0,0))
        else:
            if world_pos is None:
                world_pos = self.toWorld(image_pos)
            return jac.setOrigin(image_pos, world_pos)

    def setOrigin(self, image_origin, world_origin=None):
        """Recenter the current WCS function at a new origin location, returning the new WCS.

        This function creates a new WCS instance (always a non-local WCS) that treats
        the image_origin position the same way the current WCS treats (x,y) = (0,0).

        If the current WCS is a local WCS, this essentially declares where on the image
        you want the origin of the world coordinate system to be.  i.e. where is (u,v) = (0,0).
        So, for example, to set a WCS that has a constant pixel size with the world coordinates
        centered at the center of an image, you could write:

                wcs = galsim.PixelScale(scale).setOrigin(im.center())

        This is equivalent to the following:

                wcs = galsim.OffsetWCS(scale, image_origin=im.center())

        For more non-local WCS types, the image_origin defines what image_pos should mean the same
        thing as (0,0) does in the current WCS.  The following example should work regardless
        of what kind of WCS this is:

                world_pos1 = wcs.toWorld(PositionD(0,0))
                wcs2 = wcs.setOrigin(new_image_origin)
                world_pos2 = wcs2.toWorld(new_image_origin)
                # world_pos1 should be equal to world_pos2

        Furthermore, if the current WCS uses Euclidean world coordinates (isCelestial() == False)
        you may also provide a world_origin argument which defines what (u,v) position you want
        to correspond to the new image_origin.  Continuing the previous example:

                wcs3 = wcs.setOrigin(new_image_origin, new_world_origin)
                world_pos3 = wcs3.toWorld(new_image_origin)
                # world_pos3 should be equal to new_world_origin

        @param image_origin  The image coordinate position to use as the origin.
        @param world_origin  The world coordinate position to use as the origin.  Only valid if
                             wcs.isUniform() == True.  [ Default `world_origin=None` ]
        @returns wcs         The new recentered WCS
        """
        if isinstance(image_origin, galsim.PositionI):
            image_origin = galsim.PositionD(image_origin.x, image_origin.y)
        elif not isinstance(image_origin, galsim.PositionD):
            raise TypeError("image_origin must be a PositionD or PositionI argument")

        # Current u,v are:
        #     u = ufunc(x-x0, y-y0) + u0
        #     v = vfunc(x-x0, y-y0) + v0
        # where ufunc, vfunc represent the underlying wcs transformations.
        #
        # The _setOrigin call is expecting new values for the (x0,y0) and (u0,v0), so
        # we need to figure out how to modify the parameters give the current values.
        #
        #     Use (x1,y1) and (u1,v1) for the new values that we will pass to _setOrigin.
        #     Use (x2,y2) and (u2,v2) for the values passed as arguments.
        #
        # If world_origin is None, we want the new functions to be:
        #
        #     u' = u(x-x2, y-y2)
        #     v' = v(x-x2, y-y2)
        #     u' = ufunc(x-x0-x2, y-y0-y2) + u0
        #     v' = vfunc(x-x0-x2, y-y0-y2) + v0
        #
        # So, x1 = x0 + x2
        #     y1 = y0 + y2,
        #     u1 = u0
        #     v1 = v0
        #
        # However, if world_origin is given, u2,v2 aren't quite as simple.
        #
        #     u' = ufunc(x-x0-x2, y-y0-y2) + u1
        #     v' = vfunc(x-x0-x2, y-y0-y2) + v1
        #
        # We want to have:
        #
        #     u2 = u'(x2, y2)
        #     u2 = v'(x2, y2)
        #
        #     u2 = ufunc(x2-x0-x2, y2-y0-y2) + u1
        #     v2 = vfunc(x2-x0-x2, y2-y0-y2) + v1
        #     u1 = u2 - ufunc(-x0, -y0)
        #     v1 = v2 - vfunc(-x0, -y0)
        #     u1 = u2 - u(0, 0) + u0
        #     v1 = v2 - v(0, 0) + u0
        #
        # So, x1 = x0 + x2
        #     y1 = y0 + y2,
        #     u1 = u0 + u2 - u(0,0)
        #     v1 = v0 + v2 - v(0,0)
        #
        # So we only update image_origin if world_origin is None.
        if world_origin is None and not self._is_local:
            image_origin += self.image_origin

        if not self._is_celestial:
            if world_origin is None:
                if not self._is_local:
                    world_origin = self.world_origin
            else:
                if isinstance(world_origin, galsim.PositionI):
                    world_origin = galsim.PositionD(world_origin.x, world_origin.y)
                elif not isinstance(image_origin, galsim.PositionD):
                    raise TypeError("world_origin must be a PositionD or PositionI argument")
                if not self._is_local:
                    world_origin += self.toWorld(galsim.PositionD(0,0))
                world_origin -= self.toWorld(galsim.PositionD(0,0))
            return self._setOrigin(image_origin, world_origin)
        else:
            if world_origin is not None:
                raise TypeError("world_origin is invalid for non-uniform WCS classes")
            return self._setOrigin(image_origin)

    def writeHeader(self, header, bounds):
        """Write this WCS function to a fits header.

        This is normally called automatically from within the galsim.fits.write() function.

        The code will attempt to write standard FITS WCS keys so that the WCS will be readable 
        by other software (e.g. ds9).  It may not be able to do so accurately, in which case a 
        linearized version will be used instead.  (Specifically, it will use the local Jacobian 
        at the image center.)  

        However, this is not necessary for the WCS to survive a round trip through the FITS
        header, as it will also write GalSim-specific key words that (normally) allow it to 
        reconstruct the WCS correctly.

        @param header       The fits header to write the data to.
        @param bounds       The bounds of the image.
        """
        # Always need these, so just do them here.
        header["GS_XMIN"] = (bounds.xmin, "GalSim image minimum x coordinate")
        header["GS_YMIN"] = (bounds.ymin, "GalSim image minimum y coordinate")
        return self._writeHeader(header, bounds)



#########################################################################################
#
# Local WCS classes are those where (x,y) = (0,0) corresponds to (u,v) = (0,0).
#
# We have the following local WCS classes:
#
#     PixelScale
#     ShearWCS
#     JacobianWCS
#
# They must define the following:
#
#     _is_local         boolean variable declaring whether the WCS is local, linear
#     _is_uniform       boolean variable declaring whether the pixels are uniform
#     _is_celestial     boolean variable declaring whether the world coords are celestial
#     _posToWorld       function converting image_pos to world_pos
#     _posToImage       function converting world_pos to image_pos
#     copy              return a copy
#     __eq__            check if this equals another WCS
#     __ne__            check if this is not equal to another WCS
#     __repr__          convert to string
#     _profileToWorld   function converting image_profile to world_profile
#     _profileToImage   function converting world_profile to image_profile
#     _pixelArea        function returning the pixel area
#     _minScale         function returning the minimum linear pixel scale
#     _maxScale         function returning the maximum linear pixel scale
#     _toJacobian       function returning an equivalent JacobianWCS
#     _setOrigin        function returning a non-local WCS corresponding to this WCS
#
#########################################################################################

class PixelScale(BaseWCS):
    """This is the simplest possible WCS transformation.  It only involves a unit conversion
    from pixels to arcsec (or whatever units you want to take for your world coordinate system).

    The conversion functions are:

        u = x * scale
        v = y * scale

    Initialization
    --------------
    A PixelScale is initialized with the command:

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
        self._is_uniform = True
        self._is_celestial = False
        self._scale = scale

    # Help make sure PixelScale is read-only.
    @property
    def scale(self): return self._scale

    @property
    def image_origin(self): return galsim.PositionD(0,0)
    @property
    def world_origin(self): return galsim.PositionD(0,0)

    def _posToWorld(self, image_pos):
        return galsim.PositionD(image_pos.x * self._scale, image_pos.y * self._scale)

    def _posToImage(self, world_pos):
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

    def _toJacobian(self):
        return JacobianWCS(self._scale, 0., 0., self._scale)

    def _setOrigin(self, image_origin, world_origin):
        return OffsetWCS(self._scale, image_origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("PixelScale", "GalSim WCS name")
        header["GS_SCALE"] = (self.scale, "GalSim image scale")
        return self.affine()._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        scale = header["GS_SCALE"]
        return PixelScale(scale)

    def copy(self):
        return PixelScale(self._scale)

    def __eq__(self, other):
        if not isinstance(other, PixelScale):
            return False
        else:
            return self._scale == other._scale

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self): return "PixelScale(%r)"%self.scale


class ShearWCS(BaseWCS):
    """This WCS is a uniformly sheared coordinate system.

    The conversion functions are:

        u = (x + g1 x + g2 y) * scale / sqrt(1-g1**2-g2**2)
        v = (y - g1 y + g2 x) * scale / sqrt(1-g1**2-g2**2)

    Initialization
    --------------
    A ShearWCS is initialized with the command:

        wcs = galsim.ShearWCS(scale, shear)

    @param scale          The pixel scale, typically in units of arcsec/pixel.
    @param shear          The shear, which should be a galsim.Shear instance.
    """
    _req_params = { "scale" : float, "shear" : galsim.Shear }
    _opt_params = {}
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, scale, shear):
        self._is_local = True
        self._is_uniform = True
        self._is_celestial = False
        self._scale = scale
        self._shear = shear
        self._g1 = shear.g1
        self._g2 = shear.g2
        self._gsq = self._g1**2 + self._g2**2
        import numpy
        self._gfactor = 1. / numpy.sqrt(1. - self._gsq)

    # Help make sure ShearWCS is read-only.
    @property
    def scale(self): return self._scale
    @property
    def shear(self): return self._shear

    @property
    def image_origin(self): return galsim.PositionD(0,0)
    @property
    def world_origin(self): return galsim.PositionD(0,0)

    def _posToWorld(self, image_pos):
        u = image_pos.x * (1.+self._g1) + image_pos.y * self._g2
        v = image_pos.y * (1.-self._g1) + image_pos.x * self._g2
        u *= self._scale * self._gfactor
        v *= self._scale * self._gfactor
        return galsim.PositionD(u,v)

    def _posToImage(self, world_pos):
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

    def _toJacobian(self):
        return JacobianWCS(
            (1.+self._g1) * self._scale * self._gfactor,
            self._g2 * self._scale * self._gfactor,
            self._g2 * self._scale * self._gfactor,
            (1.-self._g1) * self._scale * self._gfactor)

    def _setOrigin(self, image_origin, world_origin):
        return OffsetShearWCS(self._scale, self._shear, image_origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("ShearWCS", "GalSim WCS name")
        header["GS_SCALE"] = (self.scale, "GalSim image scale")
        header["GS_G1"] = (self.shear.g1, "GalSim image shear g1")
        header["GS_G2"] = (self.shear.g2, "GalSim image shear g2")
        return self.affine()._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        scale = header["GS_SCALE"]
        g1 = header["GS_G1"]
        g2 = header["GS_G2"]
        return ShearWCS(scale, galsim.Shear(g1,g2))

    def copy(self):
        return ShearWCS(self._scale, self._shear)

    def __eq__(self, other):
        if not isinstance(other, ShearWCS):
            return False
        else:
            return self.scale == other.scale and self.shear == other.shear

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self): return "ShearWCS(%r,%r)"%(self.scale,self.shear)


class JacobianWCS(BaseWCS):
    """This WCS is the most general local linear WCS implementing a 2x2 jacobian matrix.

    The conversion functions are:

        u = dudx x + dudy y
        v = dvdx x + dvdy y

    A JacobianWCS has attributes dudx, dudy, dvdx, dvdy that you can access directly if that 
    is convenient.

    Initialization
    --------------
    A JacobianWCS is initialized with the command:

        wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)

    @param dudx           du/dx
    @param dudy           du/dy
    @param dvdx           dv/dx
    @param dvdy           dv/dy
    """
    _req_params = { "dudx" : float, "dudy" : float, "dvdx" : float, "dvdy" : float }
    _opt_params = {}
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, dudx, dudy, dvdx, dvdy):
        self._is_local = True
        self._is_uniform = True
        self._is_celestial = False
        self._dudx = dudx
        self._dudy = dudy
        self._dvdx = dvdx
        self._dvdy = dvdy
        self._det = dudx * dvdy - dudy * dvdx

    # Help make sure JacobianWCS is read-only.
    @property
    def dudx(self): return self._dudx
    @property
    def dudy(self): return self._dudy
    @property
    def dvdx(self): return self._dvdx
    @property
    def dvdy(self): return self._dvdy

    @property
    def image_origin(self): return galsim.PositionD(0,0)
    @property
    def world_origin(self): return galsim.PositionD(0,0)

    def _posToWorld(self, image_pos):
        x = image_pos.x
        y = image_pos.y
        u = self._dudx * x + self._dudy * y
        v = self._dvdx * x + self._dvdy * y
        return galsim.PositionD(u,v)

    def _posToImage(self, world_pos):
        #  J = ( dudx  dudy )
        #      ( dvdx  dvdy )
        #  J^-1 = (1/det) (  dvdy  -dudy )
        #                 ( -dvdx   dudx )
        u = world_pos.x
        v = world_pos.y
        x = (self._dvdy * u - self._dudy * v)/self._det
        y = (-self._dvdx * u + self._dudx * v)/self._det
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

    def _toJacobian(self):
        return self

    def _setOrigin(self, image_origin, world_origin):
        return AffineTransform(self._dudx, self._dudy, self._dvdx, self._dvdy, image_origin,
                               world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("JacobianWCS", "GalSim WCS name")
        return self.affine()._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        dudx = header.get("CD1_1",1.)
        dudy = header.get("CD1_2",0.)
        dvdx = header.get("CD2_1",0.)
        dvdy = header.get("CD2_2",1.)
        return JacobianWCS(dudx, dudy, dvdx, dvdy)

    def copy(self):
        return JacobianWCS(self._dudx, self._dudy, self._dvdx, self._dvdy)

    def __eq__(self, other):
        if not isinstance(other, JacobianWCS):
            return False
        else:
            return ( self._dudx == other._dudx and
                     self._dudy == other._dudy and
                     self._dvdx == other._dvdx and
                     self._dvdy == other._dvdy )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self): return "JacobianWCS(%r,%r,%r,%r)"%(self.dudx,self.dudy,self.dvdx,self.dvdy)


#########################################################################################
#
# Global WCS classes are those where (x,y) = (0,0) does not (necessarily) correspond
# to (u,v) = (0,0).  Furthermore, the world coordinates may be given either by (u,v)
# or by (ra, dec).  In the former case the world_pos is a PositionD, but in the latter,
# it is a CelestialCoord.
#
# We have the following non-local WCS classes:
#
#     OffsetWCS
#     OffsetShearWCS
#     AffineTransform
#     UVFunction
#     RaDecFunction
#     AstropyWCS
#     PyAstWCS
#     WcsToolsWCS
#
# They must define the following:
#
#     _is_local         boolean variable declaring whether the WCS is local, linear
#     _is_uniform       boolean variable declaring whether the pixels are uniform
#     _is_celestial     boolean variable declaring whether the world coords are celestial
#     _posToWorld       function converting image_pos to world_pos
#     _posToImage       function converting world_pos to image_pos
#     copy              return a copy
#     __eq__            check if this equals another WCS
#     __ne__            check if this is not equal to another WCS
#     _local            function returning a local WCS at a given location
#     _writeHeader      function that writes the WCS to a fits header.
#     _readHeader       static function that reads the WCS from a fits header.
#
#########################################################################################


class OffsetWCS(BaseWCS):
    """This WCS is similar to PixelScale, except the origin is not necessarily (0,0) in both
    the image and world coordinates.

    The conversion functions are:

        u = (x-x0) * scale + u0
        v = (y-y0) * scale + v0

    Initialization
    --------------
    An OffsetWCS is initialized with the command:

        wcs = galsim.OffsetWCS(scale, image_origin=None, world_origin=None)

    @param scale          The pixel scale, typically in units of arcsec/pixel.
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `image_origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "scale" : float }
    _opt_params = { "image_origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, scale, image_origin=None, world_origin=None):
        self._is_local = False
        self._is_uniform = True
        self._is_celestial = False
        self._scale = scale
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
    def scale(self): return self._scale

    @property
    def image_origin(self): return galsim.PositionD(self._x0, self._y0)
    @property
    def world_origin(self): return galsim.PositionD(self._u0, self._v0)

    def _posToWorld(self, image_pos):
        x = image_pos.x
        y = image_pos.y
        u = self._scale * (x-self._x0) + self._u0
        v = self._scale * (y-self._y0) + self._v0
        return galsim.PositionD(u,v)

    def _posToImage(self, world_pos):
        u = world_pos.x
        v = world_pos.y
        x = (u-self._u0) / self._scale + self._x0
        y = (v-self._v0) / self._scale + self._y0
        return galsim.PositionD(x,y)

    def _local(self, image_pos, world_pos):
        return PixelScale(self._scale)

    def _setOrigin(self, image_origin, world_origin):
        return OffsetWCS(self._scale, image_origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("OffsetWCS", "GalSim WCS name")
        header["GS_SCALE"] = (self.scale, "GalSim image scale")
        header["GS_X0"] = (self.image_origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.image_origin.y, "GalSim image origin y")
        header["GS_U0"] = (self.world_origin.x, "GalSim world origin u")
        header["GS_V0"] = (self.world_origin.y, "GalSim world origin v")
        return self.affine()._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        scale = header["GS_SCALE"]
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        u0 = header["GS_U0"]
        v0 = header["GS_V0"]
        return OffsetWCS(scale, galsim.PositionD(x0,y0), galsim.PositionD(u0,v0))

    def copy(self):
        return OffsetWCS(self._scale, self.image_origin, self.world_origin)

    def __eq__(self, other):
        if not isinstance(other, OffsetWCS):
            return False
        else:
            return ( self._scale == other._scale and
                     self._x0 == other._x0 and
                     self._y0 == other._y0 and
                     self._u0 == other._u0 and
                     self._v0 == other._v0 )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self): return "OffsetWCS(%r,%r,%r)"%(self.scale, self.image_origin,
                                                      self.world_origin)


class OffsetShearWCS(BaseWCS):
    """This WCS is a uniformly sheared coordinate system with image and world origins
    that are not necessarily coincident.

    The conversion functions are:

        u = ( (1+g1) (x-x0) + g2 (y-y0) ) * scale / sqrt(1-g1**2-g2**2) + u0
        v = ( (1-g1) (y-y0) + g2 (x-x0) ) * scale / sqrt(1-g1**2-g2**2) + v0

    Initialization
    --------------
    An OffsetShearWCS is initialized with the command:

        wcs = galsim.OffsetShearWCS(scale, shear, image_origin=None, world_origin=None)

    @param scale          The pixel scale, typically in units of arcsec/pixel.
    @param shear          The shear, which should be a galsim.Shear instance.
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `image_origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "scale" : float, "shear" : galsim.Shear }
    _opt_params = { "image_origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, scale, shear, image_origin=None, world_origin=None):
        self._is_local = False
        self._is_uniform = True
        self._is_celestial = False
        # The shear stuff is not too complicated, but enough so that it is worth
        # encapsulating in the ShearWCS class.  So here, we just create one of those
        # and we'll pass along any shear calculations to that.
        self._shearwcs = ShearWCS(scale, shear)
        if image_origin == None:
            self._image_origin = galsim.PositionD(0,0)
        else:
            self._image_origin = image_origin
        if world_origin == None:
            self._world_origin = galsim.PositionD(0,0)
        else:
            self._world_origin = world_origin

    @property
    def scale(self): return self._shearwcs.scale
    @property
    def shear(self): return self._shearwcs.shear

    @property
    def image_origin(self): return self._image_origin
    @property
    def world_origin(self): return self._world_origin

    def _posToWorld(self, image_pos):
        return self._shearwcs._posToWorld(image_pos - self._image_origin) + self._world_origin

    def _posToImage(self, world_pos):
        return self._shearwcs._posToImage(world_pos - self._world_origin) + self._image_origin

    def _local(self, image_pos, world_pos):
        return self._shearwcs

    def _setOrigin(self, image_origin, world_origin):
        return OffsetShearWCS(self.scale, self.shear, image_origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("OffsetShearWCS", "GalSim WCS name")
        header["GS_SCALE"] = (self.scale, "GalSim image scale")
        header["GS_G1"] = (self.shear.g1, "GalSim image shear g1")
        header["GS_G2"] = (self.shear.g2, "GalSim image shear g2")
        header["GS_X0"] = (self.image_origin.x, "GalSim image origin x coordinate")
        header["GS_Y0"] = (self.image_origin.y, "GalSim image origin y coordinate")
        header["GS_U0"] = (self.world_origin.x, "GalSim world origin u coordinate")
        header["GS_V0"] = (self.world_origin.y, "GalSim world origin v coordinate")
        return self.affine()._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        scale = header["GS_SCALE"]
        g1 = header["GS_G1"]
        g2 = header["GS_G2"]
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        u0 = header["GS_U0"]
        v0 = header["GS_V0"]
        return OffsetShearWCS(scale, galsim.Shear(g1=g1, g2=g2), galsim.PositionD(x0,y0),
                              galsim.PositionD(u0,v0))

    def copy(self):
        return OffsetShearWCS(self.scale, self.shear, self.image_origin, self.world_origin)

    def __eq__(self, other):
        if not isinstance(other, OffsetShearWCS):
            return False
        else:
            return ( self._shearwcs == other._shearwcs and
                     self._image_origin == other._image_origin and
                     self._world_origin == other._world_origin )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "OffsetShearWCS(%r,%r, %r,%r)"%(self.scale, self.shear,
                                               self.image_origin, self.world_origin)


class AffineTransform(BaseWCS):
    """This WCS is the most general linear transformation.  It involves a 2x2 Jacobian
    matrix and an offset.  You can provide the offset in terms of either the image_pos
    (x0,y0) where (u,v) = (0,0), or the world_pos (u0,v0) where (x,y) = (0,0).
    Or, in fact, you may provide both, in which case the image_pos (x0,y0) corresponds
    to the world_pos (u0,v0).

    The conversion functions are:

        u = dudx (x-x0) + dudy (y-y0) + u0
        v = dvdx (x-x0) + dvdy (y-y0) + v0

    An AffineTransform has attributes dudx, dudy, dvdx, dvdy, x0, y0, u0, v0 that you can 
    access directly if that is convenient.

    Initialization
    --------------
    An AffineTransform is initialized with the command:

        wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, image_origin=None, world_origin=None)

    @param dudx           du/dx
    @param dudy           du/dy
    @param dvdx           dv/dx
    @param dvdy           dv/dy
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `image_origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "dudx" : float, "dudy" : float, "dvdx" : float, "dvdy" : float }
    _opt_params = { "image_origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, dudx, dudy, dvdx, dvdy, image_origin=None, world_origin=None):
        self._is_local = False
        self._is_uniform = True
        self._is_celestial = False
        # As with OffsetShearWCS, we store a JacobianWCS, rather than reimplement everything.
        self._jacwcs = JacobianWCS(dudx, dudy, dvdx, dvdy)
        if image_origin == None:
            self._image_origin = galsim.PositionD(0,0)
        else:
            self._image_origin = image_origin
        if world_origin == None:
            self._world_origin = galsim.PositionD(0,0)
        else:
            self._world_origin = world_origin

    @property
    def dudx(self): return self._jacwcs.dudx
    @property
    def dudy(self): return self._jacwcs.dudy
    @property
    def dvdx(self): return self._jacwcs.dvdx
    @property
    def dvdy(self): return self._jacwcs.dvdy

    @property
    def x0(self): return self._image_origin.x
    @property
    def y0(self): return self._image_origin.y
    @property
    def u0(self): return self._world_origin.x
    @property
    def v0(self): return self._world_origin.y

    @property
    def image_origin(self): return self._image_origin
    @property
    def world_origin(self): return self._world_origin

    def _posToWorld(self, image_pos):
        return self._jacwcs._posToWorld(image_pos - self._image_origin) + self._world_origin

    def _posToImage(self, world_pos):
        return self._jacwcs._posToImage(world_pos - self._world_origin) + self._image_origin

    def _local(self, image_pos, world_pos):
        return self._jacwcs

    def _setOrigin(self, image_origin, world_origin):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy,
                               image_origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("AffineTransform", "GalSim WCS name")
        return self._writeLinearWCS(header, bounds)

    def _writeLinearWCS(self, header, bounds):
        header["CTYPE1"] = ("LINEAR", "name of the world coordinate axis")
        header["CTYPE2"] = ("LINEAR", "name of the world coordinate axis")
        header["CRVAL1"] = (self.u0, "world coordinate at reference pixel = u0")
        header["CRVAL2"] = (self.v0, "world coordinate at reference pixel = v0")
        header["CRPIX1"] = (self.x0, "image coordinate of reference pixel = x0")
        header["CRPIX2"] = (self.y0, "image coordinate of reference pixel = y0")
        header["CD1_1"] = (self.dudx, "CD1_1 = dudx")
        header["CD1_2"] = (self.dudy, "CD1_2 = dudy")
        header["CD2_1"] = (self.dvdx, "CD2_1 = dvdx")
        header["CD2_2"] = (self.dvdy, "CD2_2 = dvdy")
        return header

    @staticmethod
    def _readHeader(header):
        # We try to make this work to produce a linear WCS, no matter what kinds of key words
        # are in the header.
        if 'CD1_1' in header:
            # The others should be too, but use get with a default to be safe
            dudx = header.get("CD1_1",1.)
            dudy = header.get("CD1_2",0.)
            dvdx = header.get("CD2_1",0.)
            dvdy = header.get("CD2_2",1.)
        elif 'CDELT1' in header or 'CDELT2' in header:
            dudx = header.get("CDELT1",1.)
            dudy = 0.
            dvdx = 0.
            dvdy = header.get("CDELT2",1.)
        x0 = header.get("CRPIX1",0.)
        y0 = header.get("CRPIX2",0.)
        u0 = header.get("CRVAL1",0.)
        v0 = header.get("CRVAL2",0.)

        return AffineTransform(dudx, dudy, dvdx, dvdy, galsim.PositionD(x0,y0),
                               galsim.PositionD(u0,v0))

    def copy(self):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy,
                               self.image_origin, self.world_origin)

    def __eq__(self, other):
        if not isinstance(other, AffineTransform):
            return False
        else:
            return ( self._jacwcs == other._jacwcs and
                     self._image_origin == other._image_origin and
                     self._world_origin == other._world_origin )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "AffineTransform(%r,%r,%r,%r,%r,%r)"%(self.dudx, self.dudy, self.dvdx, self.dvdy,
                                                     self.image_origin, self.world_origin)

 
# Some helper functions for serializing arbitrary functions.  Used by both UVFunction and 
# RaDecFunction.
def _writeFuncToHeader(func, letter, header):
    import types, cPickle, marshal, base64
    if type(func) == types.FunctionType:
        # Note: I got the starting point for this code from:
        # http://stackoverflow.com/questions/1253528/
        # In particular, marshal can serialize arbitrary code. (!)
        code = marshal.dumps(func.func_code)
        name = func.func_name
        defaults = func.func_defaults

        # Functions may also have something called closure cells.  If there are any, we need
        # to include them as well.  Help for this part came from:
        # http://stackoverflow.com/questions/573569/
        if func.func_closure:
            closure = tuple(c.cell_contents for c in func.func_closure)
        else:
            closure = None
        all = (0,code,name,defaults,closure)
    else:
        # For things other than regular functions, we can try to pickle it directly, but
        # it might not work.  Let pickle raise the appropriate error if it fails.

        # The first item in the tuple is what I'm calling a type_code to indicate what to
        # do with the results of unpickling.  So far I just have 0 = function, 1 = other,
        # but this could be extended if we find a good reason to.
        all = (1,func)

    # Now we can use pickle to serialize the full thing.
    s = cPickle.dumps(all)

    # Fits can't handle arbitrary strings.  Shrink to a base-64 alphabet that are printable.
    # (This is like UUencoding for those of you who remember that...)
    s = base64.b64encode(s)

    # Fits header strings cannot be more than 68 characters long, so split it up.
    fits_len = 68
    n = (len(s)-1)/fits_len
    s_array = [ s[i*fits_len:(i+1)*fits_len] for i in range(n) ] + [ s[n*fits_len:] ]

    # The total number of string splits is stored in fits key GS_U_N.
    header["GS_" + letter + "_N"] = n+1
    for i in range(n+1):
        # Use key names: GS_U0000, GS_U00001, etc.
        key = 'GS_%s%04d'%(letter,i)
        header[key] = s_array[i]

def _makecell(value):
    # This is a little trick to make a closure cell.
    # We make a function that has the given value in closure, then then get the 
    # first (only) closure item, which will be the closure cell we need.
    return (lambda : value).func_closure[0]

def _readFuncFromHeader(letter, header):
    # This undoes the process of _writeFuncToHeader.  See the comments in that code for details.
    import types, cPickle, marshal, base64, types
    n = header["GS_" + letter + "_N"]
    s = ''
    for i in range(n):
        key = 'GS_%s%04d'%(letter,i)
        s += header[key]
    s = base64.b64decode(s)
    all = cPickle.loads(s)
    type_code = all[0]
    if type_code == 0:
        code_str, name, defaults, closure_tuple = all[1:]
        code = marshal.loads(code_str)
        if closure_tuple is None:
            closure = None
        else:
            closure = tuple(_makecell(c) for c in closure_tuple)
        func = types.FunctionType(code, globals(), name, defaults, closure)
        return func
    else:
        return all[1]

class UVFunction(BaseWCS):
    """This WCS takes two arbitrary functions for u(x,y) and v(x,y).

    The ufunc and vfunc parameters may be:
        - python functions that take (x,y) arguments
        - python objects with a __call__ method that takes (x,y) arguments
        - strings which can be parsed with eval('lambda x,y: '+str)

    Initialization
    --------------
    A UVFunction is initialized with the command:

        wcs = galsim.UVFunction(ufunc, vfunc, image_origin=None, world_origin=None)

    @param ufunc          The function u(x,y)
    @param vfunc          The function v(x,y)
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `image_origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "ufunc" : str, "vfunc" : str }
    _opt_params = { "image_origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, ufunc, vfunc, image_origin=None, world_origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = False
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
        u = self._u(image_pos.x, image_pos.y)
        v = self._v(image_pos.x, image_pos.y)
        return galsim.PositionD(u,v)

    def _posToImage(self, world_pos):
        raise NotImplementedError("World -> Image direction not implemented for UVFunction")

    def _local(self, image_pos, world_pos):
        if world_pos is not None:
            raise NotImplementedError('UVFunction.local() cannot take world_pos.')
        x0 = image_pos.x - self._x0
        y0 = image_pos.y - self._y0
        u0 = self._u(x0,y0)
        v0 = self._v(x0,y0)
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1
        dudx = 0.5*(self._u(x0+dx,y0) - self._u(x0-dx,y0))/dx
        dudy = 0.5*(self._u(x0,y0+dy) - self._u(x0,y0-dy))/dy
        dvdx = 0.5*(self._v(x0+dx,y0) - self._v(x0-dx,y0))/dx
        dvdy = 0.5*(self._v(x0,y0+dy) - self._v(x0,y0-dy))/dy

        return JacobianWCS(dudx, dudy, dvdx, dvdy)

    def _setOrigin(self, image_origin, world_origin):
        return UVFunction(self._ufunc, self._vfunc, image_origin, world_origin)
 
    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("UVFunction", "GalSim WCS name")
        header["GS_X0"] = (self.image_origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.image_origin.y, "GalSim image origin y")
        header["GS_U0"] = (self.world_origin.x, "GalSim world origin u")
        header["GS_V0"] = (self.world_origin.y, "GalSim world origin v")

        _writeFuncToHeader(self._ufunc, 'U', header)
        _writeFuncToHeader(self._vfunc, 'V', header)

        return self.affine(bounds.trueCenter())._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        u0 = header["GS_U0"]
        v0 = header["GS_V0"]
        ufunc = _readFuncFromHeader('U', header)
        vfunc = _readFuncFromHeader('V', header)
        return UVFunction(ufunc, vfunc, galsim.PositionD(x0,y0), galsim.PositionD(u0,v0))

    def copy(self):
        return UVFunction(self._ufunc, self._vfunc, self.image_origin, self.world_origin)

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

    def __repr__(self):
        return "UVFunction(%r,%r,%r,%r)"%(self.ufunc, self.vfunc,
                                          self.image_origin, self.world_origin)


def makeJacFromNumericalRaDec(ra, dec, dx, dy):
    """Convert a list of list of ra, dec values for (0,0), (dx,0), (-dx,0), (0,dy), and (0,-dy)
       into a JacobianWCS.  The input ra, dec values should be in degrees.
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
    cosdec = numpy.cos(dec0 * galsim.degrees / galsim.radians)
    dudx = -0.5*(ra1 - ra2)/dx * cosdec
    dudy = -0.5*(ra3 - ra4)/dy * cosdec
    dvdx = 0.5*(dec1 - dec2)/dx
    dvdy = 0.5*(dec3 - dec4)/dy

    # These values are all in degrees.  Convert to arcsec as per our usual standard.
    return JacobianWCS(dudx*3600., dudy*3600., dvdx*3600., dvdy*3600.)

class RaDecFunction(BaseWCS):
    """This WCS takes two arbitrary functions for ra(x,y) and dec(x,y).

    The rafunc and decfunc parameters may be:
        - python functions that take (x,y) arguments
        - python objects with a __call__ method that takes (x,y) arguments
        - strings which can be parsed with eval('lambda x,y: '+str)
    The functions should return galsim.Angles.

    Initialization
    --------------
    An RaDecFunction is initialized with the command:

        wcs = galsim.RaDecFunction(rafunc, decfunc, image_origin=None)

    @param rafunc         The function ra(x,y)
    @param decfunc        The function dec(x,y)
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `image_origin = None` ]
    """
    _req_params = { "rafunc" : str, "decfunc" : str }
    _opt_params = { "image_origin" : galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, rafunc, decfunc, image_origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        if isinstance(rafunc, basestring):
            self._rafunc = eval('lambda x,y : ' + rafunc)
        else:
            self._rafunc = rafunc
        if isinstance(decfunc, basestring):
            self._decfunc = eval('lambda x,y : ' + decfunc)
        else:
            self._decfunc = decfunc
        if image_origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = image_origin.x
            self._y0 = image_origin.y

    @property
    def rafunc(self): return self._rafunc
    @property
    def decfunc(self): return self._decfunc

    @property
    def image_origin(self): return galsim.PositionD(self._x0, self._y0)

    def _ra(self, x, y):
        return self._rafunc(x-self._x0, y-self._y0)

    def _dec(self, x, y):
        return self._decfunc(x-self._x0, y-self._y0)

    def _posToWorld(self, image_pos):
        ra = self._ra(image_pos.x, image_pos.y)
        dec = self._dec(image_pos.x, image_pos.y)
        return galsim.CelestialCoord(ra,dec)

    def _posToImage(self, world_pos):
        raise NotImplementedError("World -> Image direction not implemented for RaDecFunction")

    def _local(self, image_pos, world_pos):
        if world_pos is not None:
            raise NotImplementedError('RaDecFunction.local() cannot take world_pos.')
        x0 = image_pos.x - self._x0
        y0 = image_pos.y - self._y0
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        pos_list = [ (x0,y0), (x0+dx,y0), (x0-dx,y0), (x0,y0+dy), (x0,y0-dy) ]
        ra = [ self._ra(x,y) / galsim.degrees for (x,y) in pos_list ]
        dec = [ self._dec(x,y) / galsim.degrees for (x,y) in pos_list ]

        return makeJacFromNumericalRaDec(ra, dec, dx, dy)

    def _setOrigin(self, image_origin):
        return RaDecFunction(self._rafunc, self._decfunc, image_origin)
 
    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("RaDecFunction", "GalSim WCS name")
        header["GS_X0"] = (self.image_origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.image_origin.y, "GalSim image origin y")

        _writeFuncToHeader(self._rafunc, 'R', header)
        _writeFuncToHeader(self._decfunc, 'D', header)

        return self.affine(bounds.trueCenter())._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        rafunc = _readFuncFromHeader('R', header)
        decfunc = _readFuncFromHeader('D', header)
        return RaDecFunction(rafunc, decfunc, galsim.PositionD(x0,y0))

    def copy(self):
        return RaDecFunction(self._rafunc, self._decfunc, self.image_origin)

    def __eq__(self, other):
        if not isinstance(other, RaDecFunction):
            return False
        else:
            return (
                self._rafunc == other._rafunc and
                self._decfunc == other._decfunc and
                self._x0 == other._x0 and
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "RaDecFunction(%r,%r,%r)"%(self.rafunc, self.decfunc, self.image_origin)


class AstropyWCS(BaseWCS):
    """This WCS uses astropy.wcs to read WCS information from a FITS file.
    It requires the astropy.wcs python module to be installed.

    Astropy may be installed using pip, fink, or port:

            pip install astropy
            fink install astropy-py27
            port install py27-astropy

    It also comes by default with Enthought and Anaconda. For more information, see their website:

            http://www.astropy.org/

    Initialization
    --------------
    An AstropyWCS is initialized with one of the following commands:

        wcs = galsim.AstropyWCS(file_name=file_name)  # Open a file on disk
        wcs = galsim.AstropyWCS(header=header)        # Use an existing pyfits header
        wcs = galsim.AstropyWCS(wcs=wcs)              # Use an existing astropy.wcs.WCS instance

    Exactly one of the parameters file_name, header or wcs is required.  Also, since the
    most common usage will probably be the first, you can also give a file_name without it
    being named:

        wcs = galsim.AstropyWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [ Default `hdu = None` ]
    @param header         The header of an open pyfits (or astropy.io) hdu. 
                          [ Default `header = None` ]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    @param wcs            An existing astropy.wcs.WCS instance [ Default: `wcs = None` ]
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `image_origin = None` ]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "image_origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 wcs=None, image_origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        import astropy.wcs
        self._tag = None # Write something useful here.
        if file_name is not None:
            self._tag = file_name
            if header is not None:
                raise TypeError("Cannot provide both file_name and pyfits header")
            if wcs is not None:
                raise TypeError("Cannot provide both file_name and wcs")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)
            header = hdu.header

        if header is not None:
            if self._tag is None: self._tag = 'header'
            if wcs is not None:
                raise TypeError("Cannot provide both pyfits header and wcs")
            self._fix_header(header)
            import warnings
            with warnings.catch_warnings():
                # The constructor might emit warnings if it wants to fix the header
                # information (e.g. RADECSYS -> RADESYSa).  We'd rather ignore these
                # warnings, since we don't much care if the input file is non-standard
                # so long as we can make it work.
                warnings.simplefilter("ignore")
                wcs = astropy.wcs.WCS(header)
        if wcs is None:
            raise TypeError("Must provide one of file_name, header, or wcs")
        if self._tag is None: self._tag = 'wcs'
        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

        # If astropy.wcs cannot parse the header, it won't notice from just doing the 
        # WCS(header) command.  It will silently move on, thinking things are fine until
        # later when if will fail (with `RuntimeError: NULL error object in wcslib`).
        # We're rather get that to happen now rather than later.
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ra, dec = wcs.all_pix2world( [ [0, 0] ], 1)[0]
        except Exception as err:
            raise RuntimeError("AstropyWCS was unable to read the WCS specification in the header.")

        self._wcs = wcs
        if image_origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = image_origin.x
            self._y0 = image_origin.y


    @property
    def wcs(self): return self._wcs

    @property
    def image_origin(self): return galsim.PositionD(self._x0, self._y0)

    def _fix_header(self, header):
        # We allow for the option to fix up the header information when a modification can
        # make it readable by astropy.wcs.

        # So far, we don't have any, but something could be added in the future.
        pass

    def _posToWorld(self, image_pos):
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

        return makeJacFromNumericalRaDec(ra, dec, dx, dy)

    def _setOrigin(self, image_origin):
        return AstropyWCS(wcs=self._wcs, image_origin=image_origin)

    def _writeHeader(self, inital_header, bounds):
        # Make a new header with the contents of this WCS.
        # Note: relax = True means to write out non-standard FITS types.
        # Weirdly, this is the default when reading the header, but not when writing.
        header = self._wcs.to_header(relax=True)

        # Add in whatever was already written to the header dict.
        galsim.fits._writeDictToFitsHeader(inital_header, header)

        # And write the name as a special GalSim key
        header["GS_WCS"] = ("AstropyWCS", "GalSim WCS name")
        # Finally, update the CRPIX items if necessary.
        header["CRPIX1"] = header["CRPIX1"] + self.image_origin.x
        header["CRPIX2"] = header["CRPIX2"] + self.image_origin.y
        return header

    @staticmethod
    def _readHeader(header):
        return AstropyWCS(header=header)

    def copy(self):
        return AstropyWCS(wcs=self._wcs, image_origin=self.image_origin)

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

    def __repr__(self):
        return "AstropyWCS(%r,%r)"%(self._tag, self.image_origin)


class PyAstWCS(BaseWCS):
    """This WCS uses PyAst (the python front end for the Starlink AST code) to read WCS
    information from a FITS file.  It requires the starlink.Ast python module to be installed.

    Starlink may be installed using pip:

            pip install starlink-pyast

    For more information, see their website:

            https://pypi.python.org/pypi/starlink-pyast/

    Note: There were bugs in starlink.Ast prior to version 2.6, so if you have an earlier version,
    you should upgrate to at least 2.6.

    Initialization
    --------------
    A PyAstWCS is initialized with one of the following commands:

        wcs = galsim.PyAstWCS(file_name=file_name)  # Open a file on disk
        wcs = galsim.PyAstWCS(header=header)        # Use an existing pyfits header
        wcs = galsim.PyAstWCS(wcsinfo=wcsinfo)      # Use an existing starlink.Ast.FrameSet

    Exactly one of the parameters file_name, header or wcsinfo is required.  Also, since the
    most common usage will probably be the first, you can also give a file_name without it
    being named:

        wcs = galsim.PyAstWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [ Default `hdu = None` ]
    @param header         The header of an open pyfits (or astropy.io) hdu. 
                          [ Default `header = None` ]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    @param wcsinfo        An existing starlink.Ast.WcsMap [ Default: `wcsinfo = None` ]
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `image_origin = None` ]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "image_origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 wcsinfo=None, image_origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        import starlink.Ast, starlink.Atl
        # Note: For much of this class implementation, I've followed the example provided here:
        #       http://dsberry.github.io/starlink/node4.html
        self._tag = None # Write something useful here.
        hdu = None
        if file_name is not None:
            self._tag = file_name
            if header is not None:
                raise TypeError("Cannot provide both file_name and pyfits header")
            if wcsinfo is not None:
                raise TypeError("Cannot provide both file_name and wcsinfo")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)
            header = hdu.header

        if header is not None:
            if self._tag is None: self._tag = 'header'
            if wcsinfo is not None:
                raise TypeError("Cannot provide both pyfits header and wcsinfo")
            self._fix_header(header)
            # PyFITSAdapter requires an hdu, not a header, so if we were given a header directly,
            # then we need to mock it up.
            if hdu is None:
                from galsim import pyfits
                hdu = pyfits.PrimaryHDU()
                hdu.header = header
            fc = starlink.Ast.FitsChan( starlink.Atl.PyFITSAdapter(hdu) )
            wcsinfo = fc.read()
            if wcsinfo == None:
                raise RuntimeError("Failed to read WCS information from fits file")

        if wcsinfo is None:
            raise TypeError("Must provide one of file_name, header, or wcsinfo")
        if self._tag is None: self._tag = 'wcsinfo'

        #  We can only handle WCS with 2 pixel axes (given by Nin) and 2 WCS axes
        # (given by Nout).
        if wcsinfo.Nin != 2 or wcsinfo.Nout != 2:
            raise RuntimeError("The world coordinate system is not 2-dimensional")

        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

        self._wcsinfo = wcsinfo
        if image_origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = image_origin.x
            self._y0 = image_origin.y

    @property
    def wcsinfo(self): return self._wcsinfo

    @property
    def image_origin(self): return galsim.PositionD(self._x0, self._y0)

    def _fix_header(self, header):
        # We allow for the option to fix up the header information when a modification can
        # make it readable by PyAst.

        # There was an older proposed standard that used TAN with PV values, which is used by
        # SCamp, so we want to support it if possible.  The standard is now called TPV, which
        # PyAst understands.  All we need to do is change the names of the CTYPE values.
        if ( 'CTYPE1' in header and header['CTYPE1'].endswith('TAN') and
             'CTYPE2' in header and header['CTYPE2'].endswith('TAN') and
             'PV1_10' in header ):
            header['CTYPE1'] = header['CTYPE1'].replace('TAN','TPV')
            header['CTYPE2'] = header['CTYPE2'].replace('TAN','TPV')

    def _posToWorld(self, image_pos):
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
        x0 = image_pos.x - self._x0
        y0 = image_pos.y - self._y0
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        # wcsinfo.tran can take arrays to do everything at once.
        ra, dec = self._wcsinfo.tran( [ [ x0, x0+dx, x0-dx, x0,    x0    ],
                                        [ y0, y0,    y0,    y0+dy, y0-dy ] ])

        # Convert to degrees as needed by makeJacFromNumericalRaDec:
        ra = [ r * galsim.radians / galsim.degrees for r in ra ]
        dec = [ d * galsim.radians / galsim.degrees for d in dec ]
        return makeJacFromNumericalRaDec(ra, dec, dx, dy)

    def _setOrigin(self, image_origin):
        return PyAstWCS(wcsinfo=self._wcsinfo, image_origin=image_origin)

    def _writeHeader(self, inital_header, bounds):
        # See https://github.com/Starlink/starlink/issues/24 for helpful information from 
        # David Berry, who assisted me in getting this working.

        # Note: As David described on that page, starlink knows how to write using a 
        # FITS-WCS encoding that things like ds9 can read.  However, it doesn't do so at 
        # very high precision.  So the WCS after a round trip through the FITS-WCS encoding
        # is only accurate to about 1.e-2 arcsec.  The NATIVE encoding (which is the default
        # used here) usually writes things with enough digits to remain accurate.  But even 
        # then, there are a couple of WCS types where the round trip is only accurate to 
        # about 1.e-2 arcsec.
        
        from galsim import pyfits
        import starlink.Atl

        hdu = pyfits.PrimaryHDU()
        fc2 = starlink.Ast.FitsChan( None, starlink.Atl.PyFITSAdapter(hdu) )
        fc2.write(self._wcsinfo)
        fc2.writefits()
        header = hdu.header

        # Add in whatever was already written to the header dict.
        galsim.fits._writeDictToFitsHeader(inital_header, header)

        # And write the name as a special GalSim key
        header["GS_WCS"] = ("PyAstWCS", "GalSim WCS name")
        # And the image origin.
        header["GS_X0"] = (self.image_origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.image_origin.y, "GalSim image origin y")
        return header

    @staticmethod
    def _readHeader(header):
        x0 = header.get("GS_X0",0.)
        y0 = header.get("GS_Y0",0.)
        return PyAstWCS(header=header, image_origin=galsim.PositionD(x0,y0))
 
    def copy(self):
        return PyAstWCS(wcsinfo=self._wcsinfo, image_origin=self.image_origin)

    def __eq__(self, other):
        if not isinstance(other, PyAstWCS):
            return False
        else:
            return (
                self._wcsinfo == other._wcsinfo and
                self._x0 == other._x0 and
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "PyAstWCS(%r,%r)"%(self._tag, self.image_origin)


class WcsToolsWCS(BaseWCS):
    """This WCS uses wcstools executables to perform the appropriate WCS transformations
    for a given FITS file.  It requires wcstools command line functions to be installed.

    Note: It uses the wcstools executalbes xy2sky and sky2xy, so it can be quite a bit less
          efficient than other options that keep the WCS in memory.

    See their website for information on downloading and installing wcstools:

            http://tdc-www.harvard.edu/software/wcstools/

    Initialization
    --------------
    A WcsToolsWCS is initialized with the following command:

        wcs = galsim.WcsToolsWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `image_origin = None` ]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "image_origin" : galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name, dir=None, image_origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        import os
        if dir:
            file_name = os.path.join(dir, file_name)
        if not os.path.isfile(file_name):
            raise IOError('Cannot find file '+file_name)

        # Check wcstools is installed and that it can read the file.
        import subprocess
        # If xy2sky is not installed, this will raise an OSError
        p = subprocess.Popen(['xy2sky', '-d', '-n', '10', file_name, '0', '0'],
                             stdout=subprocess.PIPE)
        results = p.communicate()[0]
        p.stdout.close()
        if len(results) == 0:
            raise IOError('wcstools (specifically xy2sky) was unable to read '+file_name)

        self._file_name = file_name
        if image_origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = image_origin.x
            self._y0 = image_origin.y

    @property
    def file_name(self): return self._file_name

    @property
    def image_origin(self): return galsim.PositionD(self._x0, self._y0)

    def _posToWorld(self, image_pos):
        x = image_pos.x - self._x0
        y = image_pos.y - self._y0

        import subprocess
        # We'd like to get the output to 10 digits of accuracy.  This corresponds to
        # an accuracy of about 1.e-6 arcsec.  But sometimes xy2sky cannot handle it,
        # in which case the output will start with *************.  If this happens, just
        # decrease digits and try again.
        for digits in range(10,5,-1):
            # If xy2sky is not installed, this will raise an OSError
            p = subprocess.Popen(['xy2sky', '-d', '-n', str(digits), self._file_name,
                                  str(x), str(y)], stdout=subprocess.PIPE)
            results = p.communicate()[0]
            p.stdout.close()
            if len(results) == 0:
                raise IOError('wcstools (specifically xy2sky) was unable to read '+self._file_name)
            if results[0] != '*': break
        if results[0] == '*':
            raise IOError('wcstools (specifically xy2sky) was unable to read '+self._file_name)
        # Each line of output should looke like:
        #    x y J2000 ra dec
        # However, if there was an error, the J200 might be missing or the output might look like
        #    Off map x y
        vals = results.split()
        if len(vals) != 5:
            raise RuntimeError('wcstools xy2sky returned invalid result for %f,%f'%(x0,y0))
        ra = float(vals[0])
        dec = float(vals[1])

        return galsim.CelestialCoord(ra * galsim.degrees, dec * galsim.degrees)

    def _posToImage(self, world_pos):
        ra = world_pos.ra / galsim.degrees
        dec = world_pos.dec / galsim.degrees

        import subprocess
        for digits in range(10,5,-1):
            p = subprocess.Popen(['sky2xy', '-n', str(digits), self._file_name,
                                  str(ra), str(dec)], stdout=subprocess.PIPE)
            results = p.communicate()[0]
            p.stdout.close()
            if len(results) == 0:
                raise IOError('wcstools (specifically sky2xy) was unable to read '+self._file_name)
            if results[0] != '*': break
        if results[0] == '*':
            raise IOError('wcstools (specifically sky2xy) was unable to read '+self._file_name)

        # The output should looke like:
        #    ra dec J2000 -> x y
        # However, if there was an error, the J200 might be missing.
        vals = results.split()
        if len(vals) < 6:
            raise RuntimeError('wcstools sky2xy returned invalid result for %f,%f'%(ra,dec))
        if len(vals) > 6:
            import warnings
            warnings.warn('wcstools sky2xy indicates that %f,%f is off the image\n'%(ra,dec) +
                          'output is %r'%results)
        x = float(vals[4])
        y = float(vals[5])
        return galsim.PositionD(x + self._x0, y + self._y0)

    def _local(self, image_pos, world_pos):
        if world_pos is not None:
            image_pos = self._posToImage(world_pos)

        x0 = image_pos.x - self._x0
        y0 = image_pos.y - self._y0
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        import subprocess
        for digits in range(10,5,-1):
            xy = [ str(z) for z in [ x0,y0, x0+dx,y0, x0-dx,y0, x0,y0+dy, x0,y0-dy ] ]
            p = subprocess.Popen(['xy2sky', '-d', '-n', str(digits), self._file_name] + xy,
                                 stdout=subprocess.PIPE)
            results = p.communicate()[0]
            p.stdout.close()
            if len(results) == 0:
                raise IOError('wcstools (specifically xy2sky) was unable to read '+self._file_name)
            if results[0] != '*': break
        if results[0] == '*':
            raise IOError('wcstools (specifically xy2sky) was unable to read '+self._file_name)
        lines = results.splitlines()

        # Each line of output should looke like:
        #    x y J2000 ra dec
        # However, if there was an error, the J200 might be missing or the output might look like
        #    Off map x y
        ra = []
        dec = []
        for line in lines:
            vals = line.split()
            if len(vals) != 5:
                raise RuntimeError('wcstools xy2sky returned invalid result near %f,%f'%(x0,y0))
            ra.append(float(vals[0]))
            dec.append(float(vals[1]))

        return makeJacFromNumericalRaDec(ra, dec, dx, dy)

    def _setOrigin(self, image_origin):
        return WcsToolsWCS(self._file_name, image_origin=image_origin)

    def _writeHeader(self, header, bounds):
        # These are all we need to load it back.  Just use the original file.
        header["GS_WCS"]  = ("WcsToolsWCS", "GalSim WCS name")
        header["GS_FILE"] = (self._file_name, "GalSim original file with WCS data")
        header["GS_X0"] = (self.image_origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.image_origin.y, "GalSim image origin y")

        # We also copy over some of the fields we need.  wcstools doesn't seem to have something
        # that lists _all_ the keys that define the WCS.  This just gets the approximate WCS.
        import subprocess
        p = subprocess.Popen(['wcshead', self._file_name], stdout=subprocess.PIPE)
        results = p.communicate()[0]
        p.stdout.close()
        v = results.split()
        header["CTYPE1"] = v[3]
        header["CTYPE2"] = v[4]
        header["CRVAL1"] = v[5]
        header["CRVAL2"] = v[6]
        header["CRPIX1"] = v[8]
        header["CRPIX2"] = v[9]
        header["CDELT1"] = v[10]
        header["CDELT2"] = v[11]
        header["CROTA2"] = v[12]
        return header

    @staticmethod
    def _readHeader(header):
        file = header["GS_FILE"]
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        return WcsToolsWCS(file, image_origin=galsim.PositionD(x0,y0))

    def copy(self):
        return WcsToolsWCS(self._file_name, image_origin=self.image_origin)

    def __eq__(self, other):
        if not isinstance(other, WcsToolsWCS):
            return False
        else:
            return (
                self._file_name == other._file_name and
                self._x0 == other._x0 and
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "WcsToolsWCS(%r,%r)"%(self._file_name, self.image_origin)


class GSFitsWCS(BaseWCS):
    """This WCS uses a GalSim implementation to read a WCS from a FITS file.

    It doesn't do nearly as many WCS types as the other options, and it does not try to be
    as rigorous about supporting all possible valid variations in the FITS parameters.
    However, it does a few popular WCS types properly, and it doesn't require any additional 
    python modules to be installed, which can be helpful.

    Currrently, it is able to parse the following WCS types: TAN, TPV

    Initialization
    --------------
    A GSFitsWCS is initialized with one of the following commands:

        wcs = galsim.GSFitsWCS(file_name=file_name)  # Open a file on disk
        wcs = galsim.GSFitsWCS(header=header)        # Use an existing pyfits header

    Also, since the most common usage will probably be the first, you can also give a file_name 
    without it being named:

        wcs = galsim.GSFitsWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [ Default `hdu = None` ]
    @param header         The header of an open pyfits (or astropy.io) hdu. 
                          [ Default `header = None` ]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `image_origin = None` ]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "image_origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 image_origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        if file_name is not None:
            if header is not None:
                raise TypeError("Cannot provide both file_name and pyfits header")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)
            header = hdu.header

        if header is None:
            raise TypeError("Must provide either file_name or header")

        self._read_header(header)

        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

        if image_origin is not None:
            self.crpix += [ image_origin.x, image_origin.y ]

    @property
    def image_origin(self): return galsim.PositionD(0.,0.)

    def _read_header(self, header):
        # Start by reading the basic WCS stuff that most types have.
        ctype1 = header['CTYPE1']
        ctype2 = header['CTYPE2']
        if ctype1 in [ 'RA---TAN', 'RA---TPV' ]:
            self.wcs_type = ctype1[-3:]
            if ctype2 != 'DEC--' + self.wcs_type:
                raise RuntimeError("ctype1, ctype2 are not as expected")
        else:
            raise RuntimeError("GSFitsWCS cannot read this type of FITS WCS")
        crval1 = float(header['CRVAL1'])
        crval2 = float(header['CRVAL2'])
        crpix1 = float(header['CRPIX1'])
        crpix2 = float(header['CRPIX2'])
        if 'CD1_1' in header:
            cd11 = float(header['CD1_1'])
            cd12 = float(header['CD1_2'])
            cd21 = float(header['CD2_1'])
            cd22 = float(header['CD2_2'])
        elif 'CDELT1' in header:
            cd11 = float(header['CDELT1'])
            cd12 = 0.
            cd21 = 0.
            cd22 = float(header['CDELT2'])
        else:
            cd11 = 1.
            cd12 = 0.
            cd21 = 0.
            cd22 = 1.

        import numpy
        self.crpix = numpy.array( [ crpix1, crpix2 ] )
        self.cd = numpy.array( [ [ cd11, cd12 ], 
                                 [ cd21, cd22 ] ] )

        # Usually the units are degrees, but make sure
        if 'CUNIT1' in header:
            cunit1 = header['CUNIT1']
            cunit2 = header['CUNIT2']
            self.ra_units = galsim.angle.get_angle_unit(cunit1)
            self.dec_units = galsim.angle.get_angle_unit(cunit2)
        else:
            self.ra_units = galsim.degrees
            self.dec_units = galsim.degrees

        self.center = galsim.CelestialCoord(crval1 * self.ra_units, crval2 * self.dec_units)

        # There was an older proposed standard that used TAN with PV values, which is used by
        # SCamp, so we want to support it if possible.  The standard is now called TPV, so
        # use that for our wcs_type if we see the PV values with TAN.
        if self.wcs_type == 'TAN' and 'PV1_10' in header:
            self.wcs_type = 'TPV'

        self.pv = None
        if self.wcs_type == 'TPV':
            self._read_tpv(header)

    def _read_tpv(self, header):

        # Strangely, the PV values skip k==3.
        # Well, the reason is that it is for coefficients of r = sqrt(u^2+v^2).
        # But no one seems to use it, so it is almost always skipped.
        pv1 = [ float(header['PV1_'+str(k)]) for k in range(11) if k != 3 ]
        pv2 = [ float(header['PV2_'+str(k)]) for k in range(11) if k != 3 ]

        # In fact, the standard allows up to PVi_39, which is for r^7.  And all
        # unlisted values have defaults of 0 (except PVi_1, which defaults to 1).
        # A better implementation would check how high up the numbers go and build
        # the appropriate matrix, but since it is usually up to PVi_10, so far
        # we just implement that.
        # See http://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html for details.
        if 'PV1_3' in header or 'PV1_11' in header:
            raise NotImplementedError("We don't implement odd powers of r for TPV")
        if 'PV1_12' in header:
            raise NotImplementedError("We don't implement past 3rd order terms for TPV")

        import numpy
        # Another strange thing is that the two matrices are define in the opposite order
        # with respect to their element ordering.  And remember that we skipped k=3 in the
        # original reading, so indices 3..9 here were originally called PVi_4..10
        self.pv = numpy.array( [ [ [ pv1[0], pv1[2], pv1[5], pv1[9] ],
                                   [ pv1[1], pv1[4], pv1[8],   0.   ],
                                   [ pv1[3], pv1[7],   0.  ,   0.   ],
                                   [ pv1[6],   0.  ,   0.  ,   0.   ] ],
                                 [ [ pv2[0], pv2[1], pv2[3], pv2[6] ],
                                   [ pv2[2], pv2[4], pv2[7],   0.   ],
                                   [ pv2[5], pv2[8],   0.  ,   0.   ],
                                   [ pv2[9],   0.  ,   0.  ,   0.   ] ] ] )

    def _posToWorld(self, image_pos):
        import numpy

        # Start with (x,y) = the image position
        p1 = numpy.array( [ image_pos.x, image_pos.y ] )

        # This converts to (u,v) in the tangent plane
        p2 = numpy.dot(self.cd, p1 - self.crpix) 

        if self.wcs_type == 'TPV':
            # Now we apply the distortion terms
            u = p2[0]
            v = p2[1]
            usq = u*u
            vsq = v*v
            upow = numpy.array([ 1., u, usq, usq*u ])
            vpow = numpy.array([ 1., v, vsq, vsq*v ])
            p2 = numpy.dot(numpy.dot(self.pv, vpow), upow)

        # Convert (u,v) from degrees (typically) to arcsec
        p2 *= [ -1. * self.ra_units / galsim.arcsec , 1. * self.dec_units / galsim.arcsec ]

        # Finally convert from (u,v) to (ra, dec)
        # The TAN projection is also known as a gnomonic projection, which is what
        # we call it in the CelestialCoord class.
        world_pos = self.center.deproject( galsim.PositionD(p2[0],p2[1]) , projection='gnomonic' )
        return world_pos

    def _posToImage(self, world_pos):
        import numpy, numpy.linalg

        uv = self.center.project( world_pos, projection='gnomonic' )
        u = uv.x * (-1. * galsim.arcsec / self.ra_units)
        v = uv.y * (1. * galsim.arcsec / self.dec_units)
        p2 = numpy.array( [ u, v ] )

        if self.wcs_type == 'TPV':
            # Let (s,t) be the current value of (u,v).  Then we want to find a new (u,v) such that
            #
            #       [ s t ] = [ 1 u u^2 u^3 ] pv [ 1 v v^2 v^3 ]^T
            #
            # Start with (u,v) = (s,t)
            #
            # Then use Newton-Raphson iteration to improve (u,v).


            MAX_ITER = 10
            TOL = 1.e-8 * galsim.arcsec / galsim.degrees   # pv always uses degrees units
            prev_err = None
            for iter in range(MAX_ITER):
                usq = u*u
                vsq = v*v
                upow = numpy.array([ 1., u, usq, usq*u ])
                vpow = numpy.array([ 1., v, vsq, vsq*v ])

                diff = numpy.dot(numpy.dot(self.pv, vpow), upow) - p2

                # Check that things are improving...
                err = numpy.max(diff)
                if prev_err:
                    if err > prev_err:
                        raise RuntimeError("Unable to solve for image_pos (not improving)")
                prev_err = err

                # If we are below tolerance, break out of the loop
                if err < TOL: 
                    # Update p2 to the new value.
                    p2 = numpy.array( [ u, v ] )
                    break
                else:
                    dupow = numpy.array([ 0., 1., 2.*u, 3.*usq ])
                    dvpow = numpy.array([ 0., 1., 2.*v, 3.*vsq ])
                    j1 = numpy.transpose([ numpy.dot(numpy.dot(self.pv, vpow), dupow) ,
                                           numpy.dot(numpy.dot(self.pv, dvpow), upow) ])
                    dp = numpy.linalg.solve(j1, diff)
                    u -= dp[0]
                    v -= dp[1]
            if not err < TOL:
                raise RuntimeError("Unable to solve for image_pos (max iter reached)")

        p1 = numpy.dot(numpy.linalg.inv(self.cd), p2) + self.crpix

        return galsim.PositionD(p1[0], p1[1])

    def _local(self, image_pos, world_pos):
        if image_pos is None:
            image_pos = self._posToImage(world_pos)
        # The key lemma here is that chain rule for jacobians is just matrix multiplication.
        # i.e. if s = s(u,v), t = t(u,v) and u = u(x,y), v = v(x,y), then
        # ( dsdx  dsdy ) = ( dsdu dudx + dsdv dvdx   dsdu dudy + dsdv dvdy )
        # ( dtdx  dtdy ) = ( dtdu dudx + dtdv dvdx   dtdu dudy + dtdv dvdy )
        #                = ( dsdu  dsdv )  ( dudx  dudy )
        #                  ( dtdu  dtdv )  ( dvdx  dvdy )
        #
        # So if we can find the jacobian for each step of the process, we just multiply the 
        # jacobians.
        #
        # We also need to keep track of the position along the way, so we have to repeat the 
        # steps in _posToWorld.

        import numpy
        p1 = numpy.array( [ image_pos.x, image_pos.y ] )

        p2 = numpy.dot(self.cd, p1 - self.crpix) 
        # The jacobian here is just the cd matrix.
        jac = self.cd

        if self.wcs_type == 'TPV':
            # Now we apply the distortion terms
            u = p2[0]
            v = p2[1]
            usq = u*u
            vsq = v*v

            upow = numpy.array([ 1., u, usq, usq*u ])
            vpow = numpy.array([ 1., v, vsq, vsq*v ])

            p2 = numpy.dot(numpy.dot(self.pv, vpow), upow)

            # The columns of the jacobian for this step are the same function with dupow 
            # or dvpow.
            dupow = numpy.array([ 0., 1., 2.*u, 3.*usq ])
            dvpow = numpy.array([ 0., 1., 2.*v, 3.*vsq ])
            j1 = numpy.transpose([ numpy.dot(numpy.dot(self.pv, vpow), dupow) ,
                                   numpy.dot(numpy.dot(self.pv, dvpow), upow) ])
            jac = numpy.dot(j1,jac)

        unit_convert = [ -1. * self.ra_units / galsim.arcsec , 1. * self.dec_units / galsim.arcsec ]
        p2 *= unit_convert
        # Subtle point: Don't use jac *= ..., because jac might currently be self.cd, and 
        #               that would change self.cd!
        jac = jac * numpy.transpose( [ unit_convert ] )

        # Finally convert from (u,v) to (ra, dec).  We have a special function that computes
        # the jacobian of this set in the CelestialCoord class.
        drdu, drdv, dddu, dddv = self.center.deproject_jac( galsim.PositionD(p2[0],p2[1]) ,
                                                            projection='gnomonic' )
        j2 = numpy.array([ [ drdu, drdv ],
                           [ dddu, dddv ] ])
        jac = numpy.dot(j2,jac)

        return JacobianWCS(jac[0,0], jac[0,1], jac[1,0], jac[1,1])


    def _setOrigin(self, image_origin):
        ret = self.copy()
        if image_origin is not None:
            ret.crpix = ret.crpix + [ image_origin.x, image_origin.y ]
        return ret

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("GSFitsWCS", "GalSim WCS name")
        header["CTYPE1"] = 'RA---' + self.wcs_type
        header["CTYPE2"] = 'DEC--' + self.wcs_type
        header["CRPIX1"] = self.crpix[0]
        header["CRPIX2"] = self.crpix[1]
        header["CD1_1"] = self.cd[0][0]
        header["CD1_2"] = self.cd[0][1]
        header["CD2_1"] = self.cd[1][0]
        header["CD2_2"] = self.cd[1][1]
        header["CUNIT1"] = 'deg'
        header["CUNIT2"] = 'deg'
        header["CRVAL1"] = self.center.ra / galsim.degrees
        header["CRVAL2"] = self.center.dec / galsim.degrees
        if self.wcs_type == 'TPV':
            k = 0
            for n in range(4):
                for j in range(n+1):
                    i = n-j
                    header["PV1_" + str(k)] = self.pv[0, i, j]
                    header["PV2_" + str(k)] = self.pv[1, j, i]
                    k = k + 1
                    if k == 3: k = k + 1
        return header

    @staticmethod
    def _readHeader(header):
        return GSFitsWCS(header=header)

    def copy(self):
        # The copy module version of copying the dict works fine here.
        import copy
        return copy.copy(self)

    def __eq__(self, other):
        if not isinstance(other, GSFitsWCS):
            return False
        else:
            return (
                self.wcs_type == other.wcs_type and
                (self.crpix == other.crpix).all() and
                (self.cd == other.cd).all() and
                self.center == other.center and
                self.ra_units == other.ra_units and
                self.dec_units == other.dec_units and
                self.pv == other.pv )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "GSFitsWCS(%r,%r,%r,%r,%r,%r,%r)"%(self.wcs_type, self.crpix, self.cd, self.center,
                                                  self.ra_units, self.dec_units, self.pv)



# This is a list of all the WCS types that can potentially read a WCS from a FITS file.
# The function FitsWCS will try each of these in order and return the first one that
# succeeds.  AffineTransform should be last, since it will always succeed.
# The list is defined here at global scope so that external modules can add extra
# WCS types to the list if desired.

fits_wcs_types = [ 

    GSFitsWCS,      # This doesn't work for very many WCS types, but it works for the very common
                    # TAN projection, and also TPV, which is used by SCamp.  If it does work, it 
                    # is a good choice, since it is easily the fastest of any of these.

    AstropyWCS,     # This requires `import astropy.wcs` to succeed.  So far, they only handle
                    # the standard official WCS types.  So not TPV, for instance.

    PyAstWCS,       # This requires `import starlink.Ast` to succeed.  This handles the largest
                    # number of WCS types of any of these.  In fact, it worked for every one
                    # we tried in our unit tests (which was not exhaustive).

    WcsToolsWCS,    # This requires the wcstool command line functions to be installed.
                    # It is very slow, so it should only be used as a last resort.

    AffineTransform # Finally, this one is really the last resort, since it only reads in
                    # the linear part of the WCS.  It defaults to the equivalent of a 
                    # pixel scale of 1.0 if even these are not present.
]

def FitsWCS(file_name=None, dir=None, hdu=None, header=None, compression='auto'):
    """This factory function will try to read the WCS from a FITS file and return a WCS that will 
    work.  It tries a number of different WCS classes until it finds one that succeeds in reading 
    the file.
    
    If none of them work, then the last class it tries, AffineTransform, is guaranteed to succeed, 
    but it will only model the linear portion of the WCS (the CD matrix, CRPIX, and CRVAL), using 
    reasonable defaults if even these are missing.

    Note: The list of classes this function will try may be edited, e.g. by an external module 
    that wants to add an additional WCS type.  The list is `galsim.wcs.fits_wcs_types`.

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [ Default `hdu = None` ]
    @param header         The header of an open pyfits (or astropy.io) hdu. 
                          [ Default `header = None` ]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    """
    if file_name is not None:
        if header is not None:
            raise TypeError("Cannot provide both file_name and pyfits header")
        hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)
        header = hdu.header
    else:
        file_name = 'header' # For sensible error messages below.
    if header is None:
        raise TypeError("Must provide either file_name or header")

    for type in fits_wcs_types:
        try:
            wcs = type._readHeader(header)
            return wcs
        except Exception as err:
            #print 'caught ',err
            pass
    raise RuntimeError("All possible fits WCS types failed to read "+file_name)

# Let this function work like a class in config.
FitsWCS._req_params = { "file_name" : str }
FitsWCS._opt_params = { "dir" : str, "hdu" : int, "compression" : str }
FitsWCS._single_params = []
FitsWCS._takes_rng = False
FitsWCS._takes_logger = False


