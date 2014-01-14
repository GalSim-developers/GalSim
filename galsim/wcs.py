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

WCS stands for World Coordinate System.  This is the traditional term for the coordinate system
on the sky.  (I know, the world's down here, and the sky's up there, so you'd think it would
be reversed, but that's the way it goes.  Astronomy is full of terms that don't quite make sense
when you look at them too closely.)  

There are two kinds of world coordinates that we use here:

- Celestial coordinates are defined in terms of right ascension (ra) and declination (dec).
  They are a spherical coordinate system on the sky, are akin to longitude and latitude on Earth.
  c.f. http://en.wikipedia.org/wiki/Celestial_coordinate_system

- Euclidean coordinates are defined relative to a tangent plane projection of the sky. 
  If you imagine the sky coordinates on an actual sphere with a particular radius, then the 
  tangent plane is tangent to that sphere.  We use the labels (u,v) for the coordinates in 
  this system, where +v points north and +u points west.  (Yes, west, not east.  As you look
  up into the sky, if north is up, then west is to the right.)

The CelestialCoord class (in celestial.py) can convert between these two kinds of coordinates
given a projection point.

The classes in this file convert between one of these kinds of world coordinates and positions
on an image, which we call image coordinates.  We use the labels (x,y) for the image coordinates.

See the doc string for BaseWCS for explanations about the basic functionality that all WCS
classes share.  The doc strings for the individual classes explain the features specific to
each one.
"""

import galsim

class BaseWCS(object):
    """The base class for all other kinds of WCS transformations.

    All the user-functions are defined here, which defines the common interface
    for all subclasses.

    There are three types of WCS classes that we implement.

    1. Local WCS classes are those which really just define a pixel size and shape.
       They implicitly have the origin in image coordinates correspond to the origin
       in world coordinates.  They are primarily designed to handle local transformations
       at the location of a single galaxy, where it should usually be a good approximation
       to consider the pixel shape to be constant over the size of the galaxy.  We sometimes
       use the notation (u,v) for the world coordinates and (x,y) for the image coordinates.

       Currently we define the following local WCS classes:

            PixelScale
            ShearWCS
            JacobianWCS

    2. Non-local, Euclidean WCS classes may have a constant pixel size and shape, but they don't 
       have to.  They may also have an arbitrary origin in both image coordinates and world 
       coordinates.  The world coordinates are a regular Euclidean coordinate system, using
       galsim.PositionD for the world positions.  We sometimes use the notation (u,v) for the 
       world coordinates and (x,y) for the image coordinates.

       Currently we define the following non-local, Euclidean WCS classes:

            OffsetWCS
            OffsetShearWCS
            AffineTransform
            UVFunction

    3. Celestial WCS classes are defined with their world coordinates on the celestial sphere
       in terms of right ascension (RA) and declination (Dec).  The pixel size and shape are
       always variable.  We use galsim.CelestialCoord for the world coordinates to facilitate
       some of the spherical trigonometry that is sometimes required.

       Currently we define the following celestial WCS classes: (The ones marked with a *
       are defined in the file fitswcs.py.)

            RaDecFunction
           *AstropyWCS          -- requires astropy.wcs python module to be installed
           *PyAstWCS            -- requires starlink.Ast python module to be installed
           *WcsToolsWCS         -- requires wcstools command line functions to be installed
           *GSFitsWCS           -- native code, but has less functionality than the above

    There is also a factory function called FitsWCS (also defined in fitswcs.py, which is 
    intended to act like a class initializer.  It tries to read a fits file using one of the
    above classes and returns an instance of whichever one it found was successful.  It should 
    always be successful, since it's final attempt uses AffineTransform, which has reasonable 
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
                shifted = wcs.setOrigin(origin)
                world_pos2 = shifted.toWorld(origin)
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
                wcs.isPixelScale()  # is this either a PixelScale or an OffsetWCS?
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
        if self.isCelestial() and not isinstance(world_pos, galsim.CelestialCoord):
            raise TypeError("toImage requires a CelestialCoord argument")
        elif not self.isCelestial() and isinstance(world_pos, galsim.PositionI):
            world_pos = galsim.PositionD(world_pos.x, world_pos.y)
        elif not self.isCelestial() and not isinstance(world_pos, galsim.PositionD):
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
        if self.isLocal():
            return self
        else:
            if not self.isUniform() and image_pos==None and world_pos==None:
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

        if self.isCelestial():
            return jac.setOrigin(image_pos, galsim.PositionD(0,0))
        else:
            if world_pos is None:
                world_pos = self.toWorld(image_pos)
            return jac.setOrigin(image_pos, world_pos)

    def setOrigin(self, origin, world_origin=None):
        """Recenter the current WCS function at a new origin location, returning the new WCS.

        This function creates a new WCS instance (always a non-local WCS) that treats
        the origin position the same way the current WCS treats (x,y) = (0,0).

        If the current WCS is a local WCS, this essentially declares where on the image
        you want the origin of the world coordinate system to be.  i.e. where is (u,v) = (0,0).
        So, for example, to set a WCS that has a constant pixel size with the world coordinates
        centered at the center of an image, you could write:

                wcs = galsim.PixelScale(scale).setOrigin(im.center())

        This is equivalent to the following:

                wcs = galsim.OffsetWCS(scale, origin=im.center())

        For more non-local WCS types, the origin defines what image_pos should mean the same
        thing as (0,0) does in the current WCS.  The following example should work regardless
        of what kind of WCS this is:

                world_pos1 = wcs.toWorld(PositionD(0,0))
                wcs2 = wcs.setOrigin(new_origin)
                world_pos2 = wcs2.toWorld(new_origin)
                # world_pos1 should be equal to world_pos2

        Furthermore, if the current WCS uses Euclidean world coordinates (isCelestial() == False)
        you may also provide a world_origin argument which defines what (u,v) position you want
        to correspond to the new origin.  Continuing the previous example:

                wcs3 = wcs.setOrigin(new_origin, new_world_origin)
                world_pos3 = wcs3.toWorld(new_origin)
                # world_pos3 should be equal to new_world_origin

        @param origin        The image coordinate position to use as the origin.
        @param world_origin  The world coordinate position to use as the origin.  Only valid if
                             wcs.isUniform() == True.  [ Default `world_origin=None` ]
        @returns wcs         The new recentered WCS
        """
        if isinstance(origin, galsim.PositionI):
            origin = galsim.PositionD(origin.x, origin.y)
        elif not isinstance(origin, galsim.PositionD):
            raise TypeError("origin must be a PositionD or PositionI argument")

        # Current u,v are:
        #     u = ufunc(x-x0, y-y0) + u0
        #     v = vfunc(x-x0, y-y0) + v0
        # where ufunc, vfunc represent the underlying wcs transformations.
        #
        # The _setOrigin call is expecting new values for the (x0,y0) and (u0,v0), so
        # we need to figure out how to modify the parameters given the current values.
        #
        #     Use (x1,y1) and (u1,v1) for the new values that we will pass to _setOrigin.
        #     Use (x2,y2) and (u2,v2) for the values passed as arguments.
        #
        # If the wcs is a celestial WCS, then we want the new wcs to have wcs.toWorld(x2,y)
        # match the current wcs.toWorld(0,0).  So,
        #
        #     u' = ufunc(x-x1, y-y1)        # In this case, there are no u0,v0
        #     v' = vfunc(x-x1, y-y1)
        #
        #     u'(x2,y2) = u(0,0)    v'(x2,y2) = v(0,0)
        #
        #     x2 - x1 = 0 - x0      y2 - y1 = 0 - y0
        # =>  x1 = x0 + x2          y1 = y0 + y2

        if self.isCelestial():
            if world_origin is not None:
                raise TypeError("world_origin is invalid for non-uniform WCS classes")
            origin += self.origin
            return self._setOrigin(origin)

        # If world_origin is None, then we want to do basically the same thing, 
        # except we also need to pass the function the current value of wcs.world_pos
        # to keep it from resetting the world_pos back to None.

        elif world_origin is None:
            if not self.isLocal():
                origin += self.origin
                world_origin = self.world_origin
            return self._setOrigin(origin, world_origin)

        # Finally, if world_origin is given, it isn't quite as simple.
        #
        #     u' = ufunc(x-x1, y-y1) + u1
        #     v' = vfunc(x-x1, y-y1) + v1
        #
        # We want to have:
        #     u'(x2,y2) = u2
        #     ufunc(x2-x1, y2-y1) + u1 = u2
        #
        # We don't have access to ufunc directly, just u, so 
        #     (u(x2-x1+x0, y2-y1+y0) - u0) + u1 = u2
        #
        # If we take
        #     x1 = x2
        #     y1 = y2
        #
        # Then 
        #     u(x0,y0) - u0 + u1 = u2
        # =>  u1 = u0 + u2 - u(x0,y0)
        #
        # And similarly,
        #     v1 = v0 + v2 - v(x0,y0)

        else:
            if isinstance(world_origin, galsim.PositionI):
                world_origin = galsim.PositionD(world_origin.x, world_origin.y)
            elif not isinstance(origin, galsim.PositionD):
                raise TypeError("world_origin must be a PositionD or PositionI argument")
            if not self.isLocal():
                world_origin += self.world_origin - self._posToWorld(self.origin)
            return self._setOrigin(origin, world_origin)

    def writeToFitsHeader(self, header, bounds):
        """Write this WCS function to a FITS header.

        This is normally called automatically from within the galsim.fits.write() function.

        The code will attempt to write standard FITS WCS keys so that the WCS will be readable 
        by other software (e.g. ds9).  It may not be able to do so accurately, in which case a 
        linearized version will be used instead.  (Specifically, it will use the local Jacobian 
        at the image center.)  

        However, this is not necessary for the WCS to survive a round trip through the FITS
        header, as it will also write GalSim-specific key words that should allow it to 
        reconstruct the WCS correctly.

        @param header       The fits header to write the data to.
        @param bounds       The bounds of the image.
        """
        # First write the XMIN, YMIN values
        header["GS_XMIN"] = (bounds.xmin, "GalSim image minimum x coordinate")
        header["GS_YMIN"] = (bounds.ymin, "GalSim image minimum y coordinate")

        if bounds.xmin != 1 or bounds.ymin != 1:
            # ds9 always assumes the image has an origin at (1,1), so we always write the 
            # WCS to the file with this convention.  We'll convert back when we read it 
            # in if necessary.
            delta = galsim.PositionI(1-bounds.xmin, 1-bounds.ymin)
            bounds = bounds.shift(delta)
            wcs = self.setOrigin(delta)
        else:
            wcs = self

        # PyFits has changed its syntax for writing to fits headers, so rather than have our
        # various things that write to the fits header do so directly, we have them write to
        # a dict, which we then write to the actual fits header, making sure to do things 
        # correctly given the PyFits version.
        h = wcs._writeHeader({}, bounds)

        if isinstance(h, dict):
            # For dicts, we want the keys in sorted order, so the normal python dict order doesn't
            # randomly scramble things up.
            items = sorted(h.items())
        else:
            # Otherwise, h is probably a PyFits header, so the keys come out in natural order.
            items = h.items()

        from galsim import pyfits_version
        if pyfits_version < '3.1':
            for key, value in items:
                try:
                    header.update(key, value)
                except:
                    header.update(key, value[0], value[1])
        else:
            for key, value in items:
                try:
                    header.set(key, value)
                except:
                    header.set(key, value[0], value[1])

    @staticmethod
    def readFromFitsHeader(header):
        """Read a WCS function from a FITS header.

        This is normally called automatically from within the galsim.fits.read() function.

        If the file was originally written by GalSim using one of the galsim.fits.write functions,
        then this should always succeed in reading back in the original WCS.  It may not end up 
        as exactly the same class as the original, but the underlying world coordinate system
        transformation should be preserved.

        If the file was not written by GalSim, then this code will do its best to read the 
        WCS information in the FITS header.  Depending on what kind of WCS is encoded in the 
        header, this may or may not be successful.

        If there is no WCS information in the header, then this will default to a pixel scale
        of 1.

        In addition to the wcs, this function will also return the image origin that the WCS
        is assuming for the image.  If the file was originally written by GalSim, this should
        correspond to the original image origin.  If not, it will default to (1,1).

        Note that this function is a static method of BaseWCS.  So to use it, you would write

                wcs, origin = BaseWCS.readFromFitsHeader(header)


        @param header           The fits header to write the data to.

        @returns wcs, origin    The wcs and the image origin.
        """
        xmin = header.get("GS_XMIN", 1)
        ymin = header.get("GS_YMIN", 1)
        origin = galsim.PositionI(xmin, ymin)
        wcs_name = header.get("GS_WCS", None)
        if wcs_name:
            wcs_type = eval('galsim.' + wcs_name)
            wcs = wcs_type._readHeader(header)
        elif 'CTYPE1' in header:
            wcs = galsim.FitsWCS(header=header)
        else:
            wcs = galsim.PixelScale(1.)

        if xmin != 1 or ymin != 1:
            # ds9 always assumes the image has an origin at (1,1), so convert back to actual
            # xmin, ymin if necessary.
            delta = galsim.PositionI(xmin-1, ymin-1)
            wcs = wcs.setOrigin(delta)

        return wcs, origin

    def makeSkyImage(self, image, sky_level):
        """Make an image of the sky, correctly accounting for the pixel area, which might be
        variable over the image.
        
        @param image        The image onto which the sky values will be put.
        @param sky_level    The sky level in ADU/arcsec^2 (or whatever your world coordinate
                            system units are, if not arcsec).
        """
        if self.isUniform():
            image.fill(sky_level * self.pixelArea())
        elif self.isCelestial():
            self._makeCelestialSkyImage(image, sky_level)
        else:
            self._makeVariableSkyImage(image, sky_level)

    def _makeVariableSkyImage(self, image, sky_level):
        # A specialization for variable pixels when there are _u(x,y) and _v(x,y) functions
        import numpy
        b = image.bounds
        nx = b.xmax-b.xmin+1 + 2  # +2 more than in image to get row/col off each edge.
        ny = b.ymax-b.ymin+1 + 2
        x,y = numpy.meshgrid( numpy.linspace(b.xmin-1,b.xmax+1,nx),
                              numpy.linspace(b.ymin-1,b.ymax+1,ny) )
        try:
            # First try to use the _u, _v function with the numpy arrays.
            u = self._u(x,y)
            v = self._v(x,y)
        except:
            # If that didn't work, we have to do it manually for each position. :(
            u = numpy.zeros((ny,nx))
            v = numpy.zeros((ny,nx))
            for i in range(ny):
                for j in range(nx):
                    x1 = x[i,j]
                    y1 = y[i,j]
                    u[i,j] = self._u(x1,y1)
                    v[i,j] = self._v(x1,y1)
        # Use the finite differences to estimate the derivatives.
        dudx = 0.5 * (u[1:ny-1,2:nx] - u[1:ny-1,0:nx-2])
        dudy = 0.5 * (u[2:ny,1:nx-1] - u[0:ny-2,1:nx-1])
        dvdx = 0.5 * (v[1:ny-1,2:nx] - v[1:ny-1,0:nx-2])
        dvdy = 0.5 * (v[2:ny,1:nx-1] - v[0:ny-2,1:nx-1])

        area = dudx * dvdy - dvdx * dudy
        image.image.array[:,:] = area

    def _makeCelestialSkyImage(self, image, sky_level):
        # TODO: This method is really slow.  It would not be too hard to provide something like I 
        # did for the above for Variable WCS classes that would work when the classes have 
        # an _ra and _dec function.  But I haven't done that yet.
        b = image.bounds
        for x in range(b.xmin,b.xmax+1):
            for y in range(b.ymin,b.ymax+1):
                image_pos = galsim.PositionD(x,y)
                image.setValue(x,y, self.pixelArea(image_pos))



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
    def origin(self): return galsim.PositionD(0,0)
    @property
    def world_origin(self): return galsim.PositionD(0,0)

    def _u(self, x, y):
        return x * self._scale

    def _v(self, x, y):
        return y * self._scale

    def _posToWorld(self, image_pos):
        return image_pos * self._scale

    def _posToImage(self, world_pos):
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

    def _toJacobian(self):
        return JacobianWCS(self._scale, 0., 0., self._scale)

    def _setOrigin(self, origin, world_origin):
        return OffsetWCS(self._scale, origin, world_origin)

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

    The shear is given as the shape that a round object has when observed in image coordinates.

    The conversion functions in terms of (g1,g2) are therefore:

        x = (u + g1 u + g2 v) / scale / sqrt(1-g1**2-g2**2)
        y = (v - g1 v + g2 u) / scale / sqrt(1-g1**2-g2**2)

    or, writing this in the usual way of (u,v) as a function of (x,y):

        u = (x - g1 x - g2 y) * scale / sqrt(1-g1**2-g2**2)
        v = (y + g1 y - g2 x) * scale / sqrt(1-g1**2-g2**2)

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
    def origin(self): return galsim.PositionD(0,0)
    @property
    def world_origin(self): return galsim.PositionD(0,0)

    def _u(self, x, y):
        u = x * (1.-self._g1) - y * self._g2
        u *= self._scale * self._gfactor
        return u;

    def _v(self, x, y):
        v = y * (1.+self._g1) - x * self._g2
        v *= self._scale * self._gfactor
        return v;

    def _posToWorld(self, image_pos):
        x = image_pos.x
        y = image_pos.y
        return galsim.PositionD(self._u(x,y),self._v(x,y))

    def _posToImage(self, world_pos):
        x = world_pos.x * (1.+self._g1) + world_pos.y * self._g2
        y = world_pos.y * (1.-self._g1) + world_pos.x * self._g2
        x *= self._gfactor / self._scale
        y *= self._gfactor / self._scale
        return galsim.PositionD(x,y)

    def _profileToWorld(self, image_profile):
        world_profile = image_profile.createDilated(self._scale)
        world_profile.applyShear(-self.shear)
        return world_profile

    def _profileToImage(self, world_profile):
        image_profile = world_profile.createDilated(1./self._scale)
        image_profile.applyShear(self.shear)
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
            (1.-self._g1) * self._scale * self._gfactor,
            -self._g2 * self._scale * self._gfactor,
            -self._g2 * self._scale * self._gfactor,
            (1.+self._g1) * self._scale * self._gfactor)

    def _setOrigin(self, origin, world_origin):
        return OffsetShearWCS(self._scale, self._shear, origin, world_origin)

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
    is convenient.  You can also access these as a numpy matrix directly with 

        J = jac_wcs.getMatrix()

    Also, JacobianWCS has an additional method that other WCS classes do not have. The call

        scale, shear, theta, flip = jac_wcs.getDecomposition()

    will return the equivalent expansion, shear, rotation and possible flip corresponding to 
    this transformation.  See the docstring for that method for more information.

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
    def origin(self): return galsim.PositionD(0,0)
    @property
    def world_origin(self): return galsim.PositionD(0,0)

    def _u(self, x, y):
        return self._dudx * x + self._dudy * y
    def _v(self, x, y):
        return self._dvdx * x + self._dvdy * y

    def _posToWorld(self, image_pos):
        x = image_pos.x
        y = image_pos.y
        return galsim.PositionD(self._u(x,y),self._v(x,y))

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
        ret.scaleFlux(1./self._pixelArea())
        return ret

    def _profileToImage(self, world_profile):
        ret = world_profile.createTransformed(self._dvdy/self._det, -self._dudy/self._det,
                                              -self._dvdx/self._det, self._dudx/self._det)
        ret.scaleFlux(self._pixelArea())
        return ret

    def _pixelArea(self):
        return abs(self._det)

    def getMatrix(self):
        """Get the jacobian as a numpy matrix:

                numpy.matrix( [[ dudx, dudy ],
                               [ dvdx, dvdy ]] )
        """
        import numpy
        return numpy.matrix( [[ self._dudx, self._dudy ],
                              [ self._dvdx, self._dvdy ]] )

    def getDecomposition(self):
        """Get the equivalent expansion, shear, rotation and possible flip corresponding to 
        this jacobian transformation.

        A non-singular real matrix can always be decomposed into a symmetric positive definite 
        matrix times an orthogonal matrix:
        
            M = P Q

        In our case, P includes an overall scale and a shear, and Q is a rotation and possibly
        a flip of (x,y) -> (y,x).

            ( dudx  dudy ) = scale/sqrt(1-g1^2-g2^2) ( 1+g1  g2  ) ( cos(theta)  -sin(theta) ) F
            ( dvdx  dvdy )                           (  g2  1-g2 ) ( sin(theta)   cos(theta) )

        where F is either the identity matrix, ( 1 0 ), or a flip matrix, ( 0 1 ).
                                               ( 0 1 )                    ( 1 0 )

        If there is no flip, then this means that the effect of 
        
                prof.applyTransformation(dudx, dudy, dvdx, dvdy)

        is equivalent to 

                prof.applyRotation(theta)
                prof.applyShear(shear)
                prof.applyExpansion(scale)

        in that order.  (Rotation and shear do not commute.)

        The decomposition is returned as a tuple: (scale, shear, theta, flip), where scale is a 
        float, shear is a galsim.Shear, theta is a galsim.Angle, and flip is a bool.
        """
        import math
        # First we need to see whether or not the transformation includes a flip.  The evidence
        # for a flip is that the determinant is negative.
        if self._det == 0.:
            raise RuntimeError("Transformation is singular")
        elif self._det < 0.:
            flip = True
            scale = math.sqrt(-self._det)
            dudx = self._dudy
            dudy = self._dudx
            dvdx = self._dvdy
            dvdy = self._dvdx
        else:
            flip = False
            scale = math.sqrt(self._det)
            dudx = self._dudx
            dudy = self._dudy
            dvdx = self._dvdx
            dvdy = self._dvdy

        # A small bit of algebraic manipulations yield the following two equations that # let us 
        # determine theta:
        #
        # (dudx + dvdy) = 2 scale/sqrt(1-g^2) cos(t)
        # (dvdx - dudy) = 2 scale/sqrt(1-g^2) sin(t)

        C = dudx + dvdy
        S = dvdx - dudy
        theta = math.atan2(S,C) * galsim.radians

        # The next step uses the following equations that you can get from a bit more algebra:
        #
        # cost (dudx - dvdy) - sint (dudy + dvdx) = 2 scale/sqrt(1-g^2) g1
        # sint (dudx - dvdy) + cost (dudy + dvdx) = 2 scale/sqrt(1-g^2) g2
        factor = C*C+S*S    # factor = (2 scale/sqrt(1-g^2))^2
        C /= factor         # C is now cost / (2 scale/sqrt(1-g^2))
        S /= factor         # S is now sint / (2 scale/sqrt(1-g^2))

        g1 = C*(dudx-dvdy) - S*(dudy+dvdx)
        g2 = S*(dudx-dvdy) + C*(dudy+dvdx)

        return scale, galsim.Shear(g1=g1, g2=g2), theta, flip

    def _minScale(self):
        import math
        # min scale is scale * (1-|g|) / sqrt(1-|g|^2)
        # We can get this from the decomposition:
        scale, shear, theta, flip = self.getDecomposition()
        g1 = shear.g1
        g2 = shear.g2
        gsq = g1*g1 + g2*g2
        # I'm sure there is a more efficient calculation of this.  Certainly, we don't need the 
        # atan2 call from getDecomposition.  But I think we at least need at least 2 of the 3
        # sqrts, which are also rather slow.  So I'm not sure how much of a speedup is actually
        # possible.  Plus I doubt this is ever a time critical operation.  :)
        return scale * (1.-math.sqrt(gsq)) / math.sqrt(1.-gsq)

    def _maxScale(self):
        import math
        # max scale is scale * (1+|g|) / sqrt(1-|g|^2)
        scale, shear, theta, flip = self.getDecomposition()
        g1 = shear.g1
        g2 = shear.g2
        gsq = g1*g1 + g2*g2
        return scale * (1.+math.sqrt(gsq)) / math.sqrt(1.-gsq)

    def _toJacobian(self):
        return self

    def _setOrigin(self, origin, world_origin):
        return AffineTransform(self._dudx, self._dudy, self._dvdx, self._dvdy, origin,
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
# We have the following non-local WCS classes: (There are more in fitswcs.py)
#
#     OffsetWCS
#     OffsetShearWCS
#     AffineTransform
#     UVFunction
#     RaDecFunction
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
# Furthermore, if the function is celestial, it must provide the functions:
#
#     _ra               function returning ra(x,y)
#     _dec              function returning dec(x,y)
#
# If not, it must provide the functions:
#
#     _u                function returning u(x,y)
#     _v                function returning v(x,y)
#
# Ideally, the above functions would work with numpy arrays as inputs.
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

        wcs = galsim.OffsetWCS(scale, origin=None, world_origin=None)

    @param scale          The pixel scale, typically in units of arcsec/pixel.
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "scale" : float }
    _opt_params = { "origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, scale, origin=None, world_origin=None):
        self._is_local = False
        self._is_uniform = True
        self._is_celestial = False
        self._scale = scale
        if origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = origin.x
            self._y0 = origin.y
        if world_origin == None:
            self._u0 = 0
            self._v0 = 0
        else:
            self._u0 = world_origin.x
            self._v0 = world_origin.y

    @property
    def scale(self): return self._scale

    @property
    def origin(self): return galsim.PositionD(self._x0, self._y0)
    @property
    def world_origin(self): return galsim.PositionD(self._u0, self._v0)

    def _u(self, x, y):
        return self._scale * (x-self._x0) + self._u0
    def _v(self, x, y):
        return self._scale * (y-self._y0) + self._v0

    def _posToWorld(self, image_pos):
        x = image_pos.x
        y = image_pos.y
        return galsim.PositionD(self._u(x,y),self._v(x,y))

    def _posToImage(self, world_pos):
        u = world_pos.x
        v = world_pos.y
        x = (u-self._u0) / self._scale + self._x0
        y = (v-self._v0) / self._scale + self._y0
        return galsim.PositionD(x,y)

    def _local(self, image_pos, world_pos):
        return PixelScale(self._scale)

    def _setOrigin(self, origin, world_origin):
        return OffsetWCS(self._scale, origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("OffsetWCS", "GalSim WCS name")
        header["GS_SCALE"] = (self.scale, "GalSim image scale")
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")
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
        return OffsetWCS(self._scale, self.origin, self.world_origin)

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

    def __repr__(self): return "OffsetWCS(%r,%r,%r)"%(self.scale, self.origin,
                                                      self.world_origin)


class OffsetShearWCS(BaseWCS):
    """This WCS is a uniformly sheared coordinate system with image and world origins
    that are not necessarily coincident.

    The conversion functions are:

        x = ( (1+g1) (u-u0) + g2 (v-v0) ) / scale / sqrt(1-g1**2-g2**2) + x0
        y = ( (1-g1) (v-v0) + g2 (u-u0) ) / scale / sqrt(1-g1**2-g2**2) + y0

        u = ( (1-g1) (x-x0) - g2 (y-y0) ) * scale / sqrt(1-g1**2-g2**2) + u0
        v = ( (1+g1) (y-y0) - g2 (x-x0) ) * scale / sqrt(1-g1**2-g2**2) + v0

    Initialization
    --------------
    An OffsetShearWCS is initialized with the command:

        wcs = galsim.OffsetShearWCS(scale, shear, origin=None, world_origin=None)

    @param scale          The pixel scale, typically in units of arcsec/pixel.
    @param shear          The shear, which should be a galsim.Shear instance.
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "scale" : float, "shear" : galsim.Shear }
    _opt_params = { "origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, scale, shear, origin=None, world_origin=None):
        self._is_local = False
        self._is_uniform = True
        self._is_celestial = False
        # The shear stuff is not too complicated, but enough so that it is worth
        # encapsulating in the ShearWCS class.  So here, we just create one of those
        # and we'll pass along any shear calculations to that.
        self._shearwcs = ShearWCS(scale, shear)
        if origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = origin.x
            self._y0 = origin.y
        if world_origin == None:
            self._u0 = 0
            self._v0 = 0
        else:
            self._u0 = world_origin.x
            self._v0 = world_origin.y


    @property
    def scale(self): return self._shearwcs.scale
    @property
    def shear(self): return self._shearwcs.shear

    @property
    def origin(self): return galsim.PositionD(self._x0, self._y0)
    @property
    def world_origin(self): return galsim.PositionD(self._u0, self._v0)

    def _u(self, x, y):
        return self._shearwcs._u(x-self._x0,y-self._y0) + self._u0
    def _v(self, x, y):
        return self._shearwcs._v(x-self._x0,y-self._y0) + self._v0

    def _posToWorld(self, image_pos):
        x = image_pos.x
        y = image_pos.y
        return galsim.PositionD(self._u(x,y),self._v(x,y))

    def _posToImage(self, world_pos):
        return self._shearwcs._posToImage(world_pos - self.world_origin) + self.origin

    def _local(self, image_pos, world_pos):
        return self._shearwcs

    def _setOrigin(self, origin, world_origin):
        return OffsetShearWCS(self.scale, self.shear, origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("OffsetShearWCS", "GalSim WCS name")
        header["GS_SCALE"] = (self.scale, "GalSim image scale")
        header["GS_G1"] = (self.shear.g1, "GalSim image shear g1")
        header["GS_G2"] = (self.shear.g2, "GalSim image shear g2")
        header["GS_X0"] = (self.origin.x, "GalSim image origin x coordinate")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y coordinate")
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
        return OffsetShearWCS(self.scale, self.shear, self.origin, self.world_origin)

    def __eq__(self, other):
        if not isinstance(other, OffsetShearWCS):
            return False
        else:
            return ( self._shearwcs == other._shearwcs and
                     self.origin == other.origin and
                     self.world_origin == other.world_origin )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "OffsetShearWCS(%r,%r, %r,%r)"%(self.scale, self.shear,
                                               self.origin, self.world_origin)


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

        wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=None, world_origin=None)

    @param dudx           du/dx
    @param dudy           du/dy
    @param dvdx           dv/dx
    @param dvdy           dv/dy
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "dudx" : float, "dudy" : float, "dvdx" : float, "dvdy" : float }
    _opt_params = { "origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, dudx, dudy, dvdx, dvdy, origin=None, world_origin=None):
        self._is_local = False
        self._is_uniform = True
        self._is_celestial = False
        # As with OffsetShearWCS, we store a JacobianWCS, rather than reimplement everything.
        self._jacwcs = JacobianWCS(dudx, dudy, dvdx, dvdy)
        if origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = origin.x
            self._y0 = origin.y
        if world_origin == None:
            self._u0 = 0
            self._v0 = 0
        else:
            self._u0 = world_origin.x
            self._v0 = world_origin.y

    @property
    def dudx(self): return self._jacwcs.dudx
    @property
    def dudy(self): return self._jacwcs.dudy
    @property
    def dvdx(self): return self._jacwcs.dvdx
    @property
    def dvdy(self): return self._jacwcs.dvdy

    @property
    def origin(self): return galsim.PositionD(self._x0, self._y0)
    @property
    def world_origin(self): return galsim.PositionD(self._u0, self._v0)
 
    def _u(self, x, y):
        return self._jacwcs._u(x-self._x0,y-self._y0) + self._u0
    def _v(self, x, y):
        return self._jacwcs._v(x-self._x0,y-self._y0) + self._v0

    def _posToWorld(self, image_pos):
        x = image_pos.x
        y = image_pos.y
        return galsim.PositionD(self._u(x,y),self._v(x,y))

    def _posToImage(self, world_pos):
        return self._jacwcs._posToImage(world_pos - self.world_origin) + self.origin

    def _local(self, image_pos, world_pos):
        return self._jacwcs

    def _setOrigin(self, origin, world_origin):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy,
                               origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("AffineTransform", "GalSim WCS name")
        return self._writeLinearWCS(header, bounds)

    def _writeLinearWCS(self, header, bounds):
        header["CTYPE1"] = ("LINEAR", "name of the world coordinate axis")
        header["CTYPE2"] = ("LINEAR", "name of the world coordinate axis")
        header["CRVAL1"] = (self._u0, "world coordinate at reference pixel = u0")
        header["CRVAL2"] = (self._v0, "world coordinate at reference pixel = v0")
        header["CRPIX1"] = (self._x0, "image coordinate of reference pixel = x0")
        header["CRPIX2"] = (self._y0, "image coordinate of reference pixel = y0")
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
                               self.origin, self.world_origin)

    def __eq__(self, other):
        if not isinstance(other, AffineTransform):
            return False
        else:
            return ( self._jacwcs == other._jacwcs and
                     self.origin == other.origin and
                     self.world_origin == other.world_origin )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "AffineTransform(%r,%r,%r,%r,%r,%r)"%(self.dudx, self.dudy, self.dvdx, self.dvdy,
                                                     self.origin, self.world_origin)

 
# Some helper functions for serializing arbitrary functions.  Used by both UVFunction and 
# RaDecFunction.
def _writeFuncToHeader(func, func_str, letter, header):
    if func_str is not None:
        # If we have the string version, then just write that
        s = func_str
        first_key = 'GS_'+letter+'_STR'

    elif func is not None:
        # Otherwise things get more interesting.  We have to serialize a python function.
        # I got the starting point for this code from:
        #     http://stackoverflow.com/questions/1253528/
        # In particular, marshal can serialize arbitrary code. (!)
        import types, cPickle, marshal, base64
        if type(func) == types.FunctionType:
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
        first_key = 'GS_'+letter+'_FN'
    else:
        # Nothing to write.
        return

    # Fits header strings cannot be more than 68 characters long, so split it up.
    fits_len = 68
    n = (len(s)-1)/fits_len
    s_array = [ s[i*fits_len:(i+1)*fits_len] for i in range(n) ] + [ s[n*fits_len:] ]

    # The total number of string splits is stored in fits key GS_U_N.
    header["GS_" + letter + "_N"] = n+1
    for i in range(n+1):
        # Use key names like GS_U0000, GS_U00001, etc. for the function versions
        # and like GS_SU000, GS_SU001, etc. for the string versions.
        if i == 0: key = first_key
        else: key = 'GS_%s%04d'%(letter,i)
        header[key] = s_array[i]

def _makecell(value):
    # This is a little trick to make a closure cell.
    # We make a function that has the given value in closure, then then get the 
    # first (only) closure item, which will be the closure cell we need.
    return (lambda : value).func_closure[0]

def _readFuncFromHeader(letter, header):
    # This undoes the process of _writeFuncToHeader.  See the comments in that code for details.
    import types, cPickle, marshal, base64, types
    if 'GS_'+letter+'_STR' in header:
        # Read in a regular string
        n = header["GS_" + letter + "_N"]
        s = ''
        for i in range(n):
            if i == 0: key = 'GS_'+letter+'_STR'
            else: key = 'GS_%s%04d'%(letter,i)
            s += header[key]
        return s
    elif 'GS_'+letter+'_FN' in header:
        # Read in an encoded function
        n = header["GS_" + letter + "_N"]
        s = ''
        for i in range(n):
            if i == 0: key = 'GS_'+letter+'_FN'
            else: key = 'GS_%s%04d'%(letter,i)
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
    else:
        return None

class UVFunction(BaseWCS):
    """This WCS takes two arbitrary functions for u(x,y) and v(x,y).

    The ufunc and vfunc parameters may be:
        - python functions that take (x,y) arguments
        - python objects with a __call__ method that takes (x,y) arguments
        - strings which can be parsed with eval('lambda x,y: '+str)

    You may also provide the inverse functions x(u,v) and y(u,v) as xfunc and yfunc. 
    These are not required, but if you do not provide them, then any operation that requires 
    going from world to image coordinates will raise a NotImplementedError.

    Initialization
    --------------
    A UVFunction is initialized with the command:

        wcs = galsim.UVFunction(ufunc, vfunc, origin=None, world_origin=None)

    @param ufunc          The function u(x,y)
    @param vfunc          The function v(x,y)
    @param xfunc          The function x(u,v) (optional)
    @param yfunc          The function y(u,v) (optional)
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD.
                          [ Default: `world_origin = None` ]
    """
    _req_params = { "ufunc" : str, "vfunc" : str }
    _opt_params = { "xfunc" : str, "yfunc" : str,
                    "origin" : galsim.PositionD, "world_origin": galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, ufunc, vfunc, xfunc=None, yfunc=None, origin=None, world_origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = False
        import math  # In case needed by function evals
        if isinstance(ufunc, basestring):
            self._ufunc = eval('lambda x,y : ' + ufunc)
            self._ufunc_str = ufunc
        else:
            self._ufunc = ufunc
            self._ufunc_str = None

        if isinstance(vfunc, basestring):
            self._vfunc = eval('lambda x,y : ' + vfunc)
            self._vfunc_str = vfunc
        else:
            self._vfunc = vfunc
            self._vfunc_str = None

        if isinstance(xfunc, basestring):
            self._xfunc = eval('lambda u,v : ' + xfunc)
            self._xfunc_str = xfunc
        else:
            self._xfunc = xfunc
            self._xfunc_str = None

        if isinstance(yfunc, basestring):
            self._yfunc = eval('lambda u,v : ' + yfunc)
            self._yfunc_str = yfunc
        else:
            self._yfunc = yfunc
            self._yfunc_str = None

        if origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = origin.x
            self._y0 = origin.y
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
    def xfunc(self): return self._xfunc
    @property
    def yfunc(self): return self._yfunc

    @property
    def origin(self): return galsim.PositionD(self._x0, self._y0)
    @property
    def world_origin(self): return galsim.PositionD(self._u0, self._v0)

    def _u(self, x, y):
        import math
        return self._ufunc(x-self._x0, y-self._y0) + self._u0

    def _v(self, x, y):
        import math
        return self._vfunc(x-self._x0, y-self._y0) + self._v0

    def _x(self, u, v):
        import math
        return self._xfunc(u-self._u0, v-self._v0) + self._x0

    def _y(self, u, v):
        import math
        return self._yfunc(u-self._u0, v-self._v0) + self._y0

    def _posToWorld(self, image_pos):
        u = self._u(image_pos.x, image_pos.y)
        v = self._v(image_pos.x, image_pos.y)
        return galsim.PositionD(u,v)

    def _posToImage(self, world_pos):
        if self._xfunc is None or self._yfunc is None:
            raise NotImplementedError(
                "World -> Image direction not implemented for this UVFunction")
        x = self._x(world_pos.x, world_pos.y)
        y = self._y(world_pos.x, world_pos.y)
        return galsim.PositionD(x,y)

    def _local(self, image_pos, world_pos):
        if image_pos is None:
            image_pos = self._posToImage(world_pos)
        x0 = image_pos.x - self._x0
        y0 = image_pos.y - self._y0
        # For this, we ignore the possible _u0,_v0 values, since they don't affect the derivatives.
        u0 = self._ufunc(x0,y0)
        v0 = self._vfunc(x0,y0)
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        dudx = 0.5*(self._ufunc(x0+dx,y0) - self._ufunc(x0-dx,y0))/dx
        dudy = 0.5*(self._ufunc(x0,y0+dy) - self._ufunc(x0,y0-dy))/dy
        dvdx = 0.5*(self._vfunc(x0+dx,y0) - self._vfunc(x0-dx,y0))/dx
        dvdy = 0.5*(self._vfunc(x0,y0+dy) - self._vfunc(x0,y0-dy))/dy

        return JacobianWCS(dudx, dudy, dvdx, dvdy)

    def _setOrigin(self, origin, world_origin):
        return UVFunction(self._ufunc, self._vfunc, self._xfunc, self._yfunc, origin, world_origin)
 
    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("UVFunction", "GalSim WCS name")
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")
        header["GS_U0"] = (self.world_origin.x, "GalSim world origin u")
        header["GS_V0"] = (self.world_origin.y, "GalSim world origin v")

        _writeFuncToHeader(self._ufunc, self._ufunc_str, 'U', header)
        _writeFuncToHeader(self._vfunc, self._vfunc_str, 'V', header)
        _writeFuncToHeader(self._xfunc, self._xfunc_str, 'X', header)
        _writeFuncToHeader(self._yfunc, self._yfunc_str, 'Y', header)

        return self.affine(bounds.trueCenter())._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        u0 = header["GS_U0"]
        v0 = header["GS_V0"]
        ufunc = _readFuncFromHeader('U', header)
        vfunc = _readFuncFromHeader('V', header)
        xfunc = _readFuncFromHeader('X', header)
        yfunc = _readFuncFromHeader('Y', header)
        return UVFunction(ufunc, vfunc, xfunc, yfunc, galsim.PositionD(x0,y0),
                          galsim.PositionD(u0,v0))

    def copy(self):
        return UVFunction(self._ufunc, self._vfunc, self._xfunc, self._yfunc, self.origin,
                          self.world_origin)

    def __eq__(self, other):
        if not isinstance(other, UVFunction):
            return False
        else:
            return (
                self._ufunc == other._ufunc and
                self._vfunc == other._vfunc and
                self._xfunc == other._xfunc and
                self._yfunc == other._yfunc and
                self._x0 == other._x0 and
                self._y0 == other._y0 and
                self._u0 == other._u0 and
                self._v0 == other._v0)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "UVFunction(%r,%r,%r,%r,%r,%r)"%(self._ufunc, self._vfunc, self._xfunc, self._yfunc,
                                                self.origin, self.world_origin)


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

        wcs = galsim.RaDecFunction(rafunc, decfunc, origin=None)

    @param rafunc         The function ra(x,y)
    @param decfunc        The function dec(x,y)
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
    """
    _req_params = { "rafunc" : str, "decfunc" : str }
    _opt_params = { "origin" : galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, rafunc, decfunc, origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        if isinstance(rafunc, basestring):
            self._rafunc = eval('lambda x,y : ' + rafunc)
            self._rafunc_str = rafunc
        else:
            self._rafunc = rafunc
            self._rafunc_str = None

        if isinstance(decfunc, basestring):
            self._decfunc = eval('lambda x,y : ' + decfunc)
            self._decfunc_str = decfunc
        else:
            self._decfunc = decfunc
            self._decfunc_str = None

        if origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = origin.x
            self._y0 = origin.y

    @property
    def rafunc(self): return self._rafunc
    @property
    def decfunc(self): return self._decfunc

    @property
    def origin(self): return galsim.PositionD(self._x0, self._y0)

    def _ra(self, x, y):
        import math
        return self._rafunc(x-self._x0, y-self._y0)

    def _dec(self, x, y):
        import math
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

    def _setOrigin(self, origin):
        return RaDecFunction(self._rafunc, self._decfunc, origin)
 
    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("RaDecFunction", "GalSim WCS name")
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")

        _writeFuncToHeader(self._rafunc, self._rafunc_str, 'R', header)
        _writeFuncToHeader(self._decfunc, self._decfunc_str, 'D', header)

        return self.affine(bounds.trueCenter())._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        rafunc = _readFuncFromHeader('R', header)
        decfunc = _readFuncFromHeader('D', header)
        return RaDecFunction(rafunc, decfunc, galsim.PositionD(x0,y0))

    def copy(self):
        return RaDecFunction(self._rafunc, self._decfunc, self.origin)

    def __eq__(self, other):
        if not isinstance(other, RaDecFunction):
            return False
        else:
            return (
                self._rafunc == other._rafunc and
                self._decfunc == other._decfunc and
                self._x0 == other._x0 and
                self._y0 == other._y0 and
                self._u0 == other._u0 and
                self._v0 == other._v0)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "RaDecFunction(%r,%r,%r)"%(self.rafunc, self.decfunc, self.origin)

