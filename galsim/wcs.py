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

    There are several types of WCS classes that we implement. The class hierarchy is:

    BaseWCS
        --- EuclideanWCS
                --- UniformWCS
                        --- LocalWCS
        --- CelestialWCS

    These base classes are not constructible.  They do not have __init__ defined.

    1. LocalWCS classes are those which really just define a pixel size and shape.
       They implicitly have the origin in image coordinates correspond to the origin
       in world coordinates.  They are primarily designed to handle local transformations
       at the location of a single galaxy, where it should usually be a good approximation
       to consider the pixel shape to be constant over the size of the galaxy.  We sometimes
       use the notation (u,v) for the world coordinates and (x,y) for the image coordinates.

       Currently we define the following LocalWCS classes:

            PixelScale
            ShearWCS
            JacobianWCS

    2. UniformWCS classes have a constant pixel size and shape, but they have an arbitrary origin 
       in both image coordinates and world coordinates.  A LocalWCS class can be turned into a 
       non-local UniformWCS class when an image has its bounds changed (e.g. by the commands 
       `setCenter`, `setOrigin` or `shift`.
       
       Currently we define the following non-local, UniformWCS classes:

            OffsetWCS
            OffsetShearWCS
            AffineTransform

    3. EuclideanWCS classes use a regular Euclidean coordinate system for the world coordinates,
       using galsim.PositionD for the world positions.  We sometimes use the notation (u,v) for 
       the world coordinates and (x,y) for the image coordinates.

       Currently we define the following non-uniform, Euclidean WCS class:

            UVFunction

    4. CelestialWCS classes are defined with their world coordinates on the celestial sphere
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
        return False   # Overridden by PixelScale and OffsetWCS

    def isLocal(self):
        """Return whether the WCS transformation is a local, linear approximation.

        There are two requirements for this to be true:
            1. The image position (x,y) = (0,0) is at the world position (u,v) = (0,0).
            2. The pixel area and shape do not vary with position.
        """
        return False   # Overridden by LocalWCS

    def isUniform(self):
        """Return whether the pixels in this WCS have uniform size and shape"""
        return False   # Overridden by UniformWCS

    def isCelestial(self):
        """Return whether the world coordinates are CelestialCoord (i.e. ra,dec).  """
        return False   # Overridden by CelestialWCS

    def local(self, image_pos=None, world_pos=None):
        """Return the local linear approximation of the WCS at a given point.

        @param image_pos    The image coordinate position (for variable WCS types)
        @param world_pos    The world coordinate position (for variable WCS types)
        @returns local_wcs  A WCS with wcs.isLocal() == True
        """
        if image_pos and world_pos:
            raise TypeError("Only one of image_pos or world_pos may be provided")
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

        Furthermore, if the current WCS is a EuclideanWCS (wcs.isCelestial() == False) you may 
        also provide a world_origin argument which defines what (u,v) position you want to 
        correspond to the new origin.  Continuing the previous example:

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
        from galsim import pyfits_version
        if pyfits_version < '3.1':
            header.update("GS_XMIN", bounds.xmin, "GalSim image minimum x coordinate")
            header.update("GS_YMIN", bounds.ymin, "GalSim image minimum y coordinate")
        else:
            header.set("GS_XMIN", bounds.xmin, "GalSim image minimum x coordinate")
            header.set("GS_YMIN", bounds.ymin, "GalSim image minimum y coordinate")

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
        self._makeSkyImage(image, sky_level)

#########################################################################################
#
# Our class hierarchy is:
#
#    BaseWCS
#        --- EuclideanWCS
#                --- UniformWCS
#                        --- LocalWCS
#        --- CelestialWCS
#
# Here we define the rest of these classes (besides BaseWCS that is), and implement some 
# functionality that is common among the subclasses of these when possible.
#
#########################################################################################


class EuclideanWCS(BaseWCS):
    """A EuclideanWCS is a BaseWCS whose world coordinates are on a Euclidean plane.
    We usually use the notation (u,v) to refer to positions in world coordinates, and 
    they use the class PositionD.
    """

    # All EuclideanWCS classes must define origin and world_origin.
    # Sometimes it is convenient to access x0,y0,u0,v0 directly.
    @property
    def x0(self): return self.origin.x
    @property
    def y0(self): return self.origin.y
    @property
    def u0(self): return self.world_origin.x
    @property
    def v0(self): return self.world_origin.y

    # Simple.  Just call _u, _v.  The inverse is not so easy in general, so each class needs
    # to define that itself.
    def _posToWorld(self, image_pos):
        x = image_pos.x - self.x0
        y = image_pos.y - self.y0
        return galsim.PositionD(self._u(x,y), self._v(x,y)) + self.world_origin

    # Also simple if _x,_y are implemented.  However, they are allowed to raise a 
    # NotImplementedError.
    def _posToImage(self, world_pos):
        u = world_pos.x - self.u0
        v = world_pos.y - self.v0
        return galsim.PositionD(self._x(u,v),self._y(u,v)) + self.origin

    # Each subclass has a function _newOrigin, which just calls the constructor with new
    # values for origin and world_origin.  This function figures out what those values 
    # should be to match the desired behavior of setOrigin.
    def _setOrigin(self, origin, world_origin):
        # Current u,v are:
        #     u = ufunc(x-x0, y-y0) + u0
        #     v = vfunc(x-x0, y-y0) + v0
        # where ufunc, vfunc represent the underlying wcs transformations.
        #
        # The _newOrigin call is expecting new values for the (x0,y0) and (u0,v0), so
        # we need to figure out how to modify the parameters given the current values.
        #
        #     Use (x1,y1) and (u1,v1) for the new values that we will pass to _newOrigin.
        #     Use (x2,y2) and (u2,v2) for the values passed as arguments.
        #
        # If world_origin is None, then we want to do basically the same thing as in the 
        # non-uniform case, except that we also need to pass the function the current value of 
        # wcs.world_pos to keep it from resetting the world_pos back to None.

        if world_origin is None:
            if not self.isLocal():
                origin += self.origin
                world_origin = self.world_origin
            return self._newOrigin(origin, world_origin)

        # But if world_origin is given, it isn't quite as simple.
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
            return self._newOrigin(origin, world_origin)

    # If the class doesn't define something else, then we can approximate the local jacobian
    # from finite differences for the derivatives.  This will be overridden by UniformWCS.
    def _local(self, image_pos, world_pos):
        if image_pos is None:
            if world_pos is None:
                raise TypeError("Either image_pos or world_pos must be provided")
            image_pos = self._posToImage(world_pos)

        # Calculate the Jacobian using finite differences for the derivatives.
        x0 = image_pos.x - self.x0
        y0 = image_pos.y - self.y0
        u0 = self._u(x0,y0)
        v0 = self._v(x0,y0)

        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        import numpy
        xlist = numpy.array([ x0+dx, x0-dx, x0,    x0    ]).astype(float)
        ylist = numpy.array([ y0,    y0,    y0+dy, y0-dy ]).astype(float)
        try :
            # Try using numpy arrays first, since it should be faster if it works.
            u = self._u(xlist,ylist)
            v = self._v(xlist,ylist)
        except:
            # Otherwise do them one at a time.
            u = [ self._u(x,y) for (x,y) in zip(xlist,ylist) ]
            v = [ self._v(x,y) for (x,y) in zip(xlist,ylist) ]

        dudx = 0.5 * (u[0] - u[1]) / dx
        dudy = 0.5 * (u[2] - u[3]) / dx
        dvdx = 0.5 * (v[0] - v[1]) / dx
        dvdy = 0.5 * (v[2] - v[3]) / dx

        return JacobianWCS(dudx, dudy, dvdx, dvdy)

    # The naive way to make the sky image is to loop over pixels and call pixelArea(pos)
    # for that position.  This is extremely slow.  Here, we use the fact that the _u and _v
    # functions might work with numpy arrays.  If they do, this function is quite fast.
    # If not, we still get some gain from calculating u,v for each pixel and sharing some 
    # of those calculations for multiple finite difference derivatives.  But the latter 
    # option is still pretty slow, so it's much better to have the _u and _v work with 
    # numpy arrays!
    def _makeSkyImage(self, image, sky_level):
        import numpy
        b = image.bounds
        nx = b.xmax-b.xmin+1 + 2  # +2 more than in image to get row/col off each edge.
        ny = b.ymax-b.ymin+1 + 2
        x,y = numpy.meshgrid( numpy.linspace(b.xmin-1,b.xmax+1,nx),
                              numpy.linspace(b.ymin-1,b.ymax+1,ny) )
        x -= self.x0
        y -= self.y0
        try:
            # First try to use the _u, _v function with the numpy arrays.
            u = self._u(x.flatten(),y.flatten())
            v = self._v(x.flatten(),y.flatten())
        except:
            # If that didn't work, we have to do it manually for each position. :(  (SLOW!)
            u = numpy.array([ self._u(x1,y1) for x1,y1 in zip(x.flatten(),y.flatten()) ])
            v = numpy.array([ self._v(x1,y1) for x1,y1 in zip(x.flatten(),y.flatten()) ])
        u = numpy.reshape(u, x.shape)
        v = numpy.reshape(v, x.shape)
        # Use the finite differences to estimate the derivatives.
        dudx = 0.5 * (u[1:ny-1,2:nx] - u[1:ny-1,0:nx-2])
        dudy = 0.5 * (u[2:ny,1:nx-1] - u[0:ny-2,1:nx-1])
        dvdx = 0.5 * (v[1:ny-1,2:nx] - v[1:ny-1,0:nx-2])
        dvdy = 0.5 * (v[2:ny,1:nx-1] - v[0:ny-2,1:nx-1])

        area = numpy.abs(dudx * dvdy - dvdx * dudy)
        image.image.array[:,:] = area * sky_level

    # Each class should define the __eq__ function.  Then __ne__ is obvious.
    def __ne__(self, other):
        return not self.__eq__(other)


class UniformWCS(EuclideanWCS):
    """A UniformWCS is a EuclideanWCS which has a uniform pixel size and shape.
    """
    def isUniform(self): return True

    # These can also just pass through to the _localwcs attribute.
    def _u(self, x, y):
        return self._local_wcs._u(x,y)
    def _v(self, x, y):
        return self._local_wcs._v(x,y)
    def _x(self, u, v):
        return self._local_wcs._x(u,v)
    def _y(self, u, v):
        return self._local_wcs._y(u,v)

    # For UniformWCS, the local WCS is an attribute.  Just return it.
    def _local(self, image_pos=None, world_pos=None): 
        return self._local_wcs

    # This is very simple if the pixels are uniform.
    def _makeSkyImage(self, image, sky_level):
        image.fill(sky_level * self.pixelArea())

    # Just check if the locals match and if the origins match.
    def __eq__(self, other):
        return ( isinstance(other, self.__class__) and
                 self._local_wcs == other._local_wcs and
                 self.origin == other.origin and
                 self.world_origin == other.world_origin )



class LocalWCS(UniformWCS):
    """A LocalWCS is a UniformWCS in which (0,0) in image coordinates is at the same place
    as (0,0) in world coordinates
    """
    def isLocal(self): return True

    # The origins are definitionally (0,0) for these.  So just define them here.
    @property
    def origin(self): return galsim.PositionD(0,0)
    @property
    def world_origin(self): return galsim.PositionD(0,0)

    # For LocalWCS, there is no origin to worry about.
    def _posToWorld(self, image_pos):
        x = image_pos.x
        y = image_pos.y
        return galsim.PositionD(self._u(x,y),self._v(x,y))

    # For LocalWCS, there is no origin to worry about.
    def _posToImage(self, world_pos):
        u = world_pos.x
        v = world_pos.y
        return galsim.PositionD(self._x(u,v),self._y(u,v))

    # For LocalWCS, this is of course trivial.
    def _local(self, image_pos, world_pos): 
        return self


class CelestialWCS(BaseWCS):
    """A CelestialWCS is a BaseWCS whose world coordinates are on the celestial sphere.
    We use the CelestialCoord class for the world coordinates.
    """
    def isCelestial(self): return True

    # CelestialWCS classes still have origin, but not world_origin.
    @property
    def x0(self): return self.origin.x
    @property
    def y0(self): return self.origin.y

    # This is a bit simpler than the EuclideanWCS version, since there is no world_origin.
    def _setOrigin(self, origin, world_origin):
        # We want the new wcs to have wcs.toWorld(x2,y) match the current wcs.toWorld(0,0).
        # So,
        #
        #     u' = ufunc(x-x1, y-y1)        # In this case, there are no u0,v0
        #     v' = vfunc(x-x1, y-y1)
        #
        #     u'(x2,y2) = u(0,0)    v'(x2,y2) = v(0,0)
        #
        #     x2 - x1 = 0 - x0      y2 - y1 = 0 - y0
        # =>  x1 = x0 + x2          y1 = y0 + y2
        if world_origin is not None:
            raise TypeError("world_origin is invalid for CelestialWCS classes")
        origin += self.origin
        return self._newOrigin(origin)

    # If the class doesn't define something else, then we can approximate the local jacobian
    # from finite differences for the derivatives of ra and dec.  Very similar to the 
    # version for EuclideanWCS, but convert from dra, ddec to du, dv locallat at the given
    # position.
    def _local(self, image_pos, world_pos):
        if image_pos is None:
            if world_pos is None:
                raise TypeError("Either image_pos or world_pos must be provided")
            image_pos = self._posToImage(world_pos)

        x0 = image_pos.x - self.x0
        y0 = image_pos.y - self.y0
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        import numpy
        xlist = numpy.array([ x0, x0+dx, x0-dx, x0,    x0    ])
        ylist = numpy.array([ y0, y0,    y0,    y0+dy, y0-dy ])
        try :
            # Try using numpy arrays first, since it should be faster if it works.
            ra, dec = self._radec(xlist,ylist)
        except:
            # Otherwise do them one at a time.
            world = [ self._radec(x,y) for (x,y) in zip(xlist,ylist) ]
            ra = [ w[0] for w in world ]
            dec = [ w[1] for w in world ]

        import numpy
        # Note: our convention is that ra increases to the left!
        # i.e. The u,v plane is the tangent plane as seen from Earth with +v pointing
        # north, and +u pointing west.
        # That means the du values are the negative of dra.
        cosdec = numpy.cos(dec[0])
        dudx = -0.5 * (ra[1] - ra[2]) / dx * cosdec
        dudy = -0.5 * (ra[3] - ra[4]) / dy * cosdec
        dvdx = 0.5 * (dec[1] - dec[2]) / dx
        dvdy = 0.5 * (dec[3] - dec[4]) / dy

        # These values are all in radians.  Convert to arcsec as per our usual standard.
        factor = 1. * galsim.radians / galsim.arcsec
        return JacobianWCS(dudx*factor, dudy*factor, dvdx*factor, dvdy*factor)

    # This is similar to the version for EuclideanWCS, but uses dra, ddec.
    # Again, it is much faster if the _radec function works with numpy arrays.
    def _makeSkyImage(self, image, sky_level):
        import numpy
        b = image.bounds
        nx = b.xmax-b.xmin+1 + 2  # +2 more than in image to get row/col off each edge.
        ny = b.ymax-b.ymin+1 + 2
        x,y = numpy.meshgrid( numpy.linspace(b.xmin-1,b.xmax+1,nx),
                              numpy.linspace(b.ymin-1,b.ymax+1,ny) )
        x -= self.x0
        y -= self.y0
        try:
            # First try to use the _radec function with the numpy arrays.
            ra, dec = self._radec(x.flatten(),y.flatten())
        except:
            # If that didn't work, we have to do it manually for each position. :(  (SLOW!)
            rd = [ self._radec(x1,y1) for x1,y1 in zip(x.flatten(),y.flatten()) ]
            ra = numpy.array([ radec[0] for radec in rd ])
            dec = numpy.array([ radec[1] for radec in rd ])
        ra = numpy.reshape(ra, x.shape)
        dec = numpy.reshape(dec, x.shape)

        # Use the finite differences to estimate the derivatives.
        cosdec = numpy.cos(dec[1:ny-1,1:nx-1])
        dudx = -0.5 * (ra[1:ny-1,2:nx] - ra[1:ny-1,0:nx-2]) * cosdec
        dudy = -0.5 * (ra[2:ny,1:nx-1] - ra[0:ny-2,1:nx-1]) * cosdec
        dvdx = 0.5 * (dec[1:ny-1,2:nx] - dec[1:ny-1,0:nx-2])
        dvdy = 0.5 * (dec[2:ny,1:nx-1] - dec[0:ny-2,1:nx-1])

        area = numpy.abs(dudx * dvdy - dvdx * dudy)
        factor = 1. * galsim.radians / galsim.arcsec
        image.image.array[:,:] = area * sky_level * factor**2


    # Simple.  Just call _radec.  The inverse is not so easy in general, so each class needs
    # to define that itself.
    def _posToWorld(self, image_pos):
        x = image_pos.x - self.x0
        y = image_pos.y - self.y0
        ra, dec = self._radec(x,y)
        return galsim.CelestialCoord(ra*galsim.radians, dec*galsim.radians)

    # Also simple if _xy is implemented.  However, it is allowed to raise a NotImplementedError.
    def _posToImage(self, world_pos):
        ra = world_pos.ra.rad()
        dec = world_pos.dec.rad()
        x, y = self._xy(ra,dec)
        return galsim.PositionD(x,y) + self.origin

    # Each class should define the __eq__ function.  Then __ne__ is obvious.
    def __ne__(self, other):
        return not self.__eq__(other)



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
#     origin            attribute or property returning the origin
#     world_origin      attribute or property returning the world origin
#     _u                function returning u(x,y)
#     _v                function returning v(x,y)
#     _x                function returning x(u,v)
#     _y                function returning y(u,v)
#     _profileToWorld   function converting image_profile to world_profile
#     _profileToImage   function converting world_profile to image_profile
#     _pixelArea        function returning the pixel area
#     _minScale         function returning the minimum linear pixel scale
#     _maxScale         function returning the maximum linear pixel scale
#     _toJacobian       function returning an equivalent JacobianWCS
#     _writeHeader      function that writes the WCS to a fits header.
#     _readHeader       static function that reads the WCS from a fits header.
#     _newOrigin        function returning a non-local WCS corresponding to this WCS
#     copy              return a copy
#     __eq__            check if this equals another WCS
#     __repr__          convert to string
#
#########################################################################################

class PixelScale(LocalWCS):
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
        self._scale = scale

    # Help make sure PixelScale is read-only.
    @property
    def scale(self): return self._scale

    def isPixelScale(self):
        return True

    def _u(self, x, y):
        return x * self._scale

    def _v(self, x, y):
        return y * self._scale

    def _x(self, u, v):
        return u / self._scale

    def _y(self, u, v):
        return v / self._scale

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

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("PixelScale", "GalSim WCS name")
        header["GS_SCALE"] = (self.scale, "GalSim image scale")
        return self.affine()._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        scale = header["GS_SCALE"]
        return PixelScale(scale)

    def _newOrigin(self, origin, world_origin):
        return OffsetWCS(self._scale, origin, world_origin)

    def copy(self):
        return PixelScale(self._scale)

    def __eq__(self, other):
        return ( isinstance(other, PixelScale) and
                 self.scale == other.scale )

    def __repr__(self): return "PixelScale(%r)"%self.scale


class ShearWCS(LocalWCS):
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
        u *= self._gfactor * self._scale
        return u;

    def _v(self, x, y):
        v = y * (1.+self._g1) - x * self._g2
        v *= self._gfactor * self._scale
        return v;

    def _x(self, u, v):
        x = u * (1.+self._g1) + v * self._g2
        x *= self._gfactor / self._scale
        return x;

    def _y(self, u, v):
        y = v * (1.-self._g1) + u * self._g2
        y *= self._gfactor / self._scale
        return y;

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

    def _newOrigin(self, origin, world_origin):
        return OffsetShearWCS(self._scale, self._shear, origin, world_origin)

    def copy(self):
        return ShearWCS(self._scale, self._shear)

    def __eq__(self, other):
        return ( isinstance(other, ShearWCS) and
                 self.scale == other.scale and
                 self.shear == other.shear )

    def __repr__(self): return "ShearWCS(%r,%r)"%(self.scale,self.shear)


class JacobianWCS(LocalWCS):
    """This WCS is the most general local linear WCS implementing a 2x2 jacobian matrix.

    The conversion functions are:

        u = dudx x + dudy y
        v = dvdx x + dvdy y

    A JacobianWCS has attributes dudx, dudy, dvdx, dvdy that you can access directly if that 
    is convenient.  You can also access these as a numpy matrix directly with 

        J = jac_wcs.getMatrix()

    Also, JacobianWCS has another method that other WCS classes do not have. The call

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

    def _x(self, u, v):
        #  J = ( dudx  dudy )
        #      ( dvdx  dvdy )
        #  J^-1 = (1/det) (  dvdy  -dudy )
        #                 ( -dvdx   dudx )
        return (self._dvdy * u - self._dudy * v)/self._det

    def _y(self, u, v):
        return (-self._dvdx * u + self._dudx * v)/self._det

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

    def _newOrigin(self, origin, world_origin):
        return AffineTransform(self._dudx, self._dudy, self._dvdx, self._dvdy, origin,
                               world_origin)

    def copy(self):
        return JacobianWCS(self._dudx, self._dudy, self._dvdx, self._dvdy)

    def __eq__(self, other):
        return ( isinstance(other, JacobianWCS) and
                 self.dudx == other.dudx and
                 self.dudy == other.dudy and
                 self.dvdx == other.dvdx and
                 self.dvdy == other.dvdy )

    def __repr__(self): return "JacobianWCS(%r,%r,%r,%r)"%(self.dudx,self.dudy,self.dvdx,self.dvdy)


#########################################################################################
#
# Non-local UniformWCS classes are those where (x,y) = (0,0) does not (necessarily) 
# correspond to (u,v) = (0,0).
#
# We have the following non-local UniformWCS classes: 
#
#     OffsetWCS
#     OffsetShearWCS
#     AffineTransform
#
# They must define the following:
#
#     origin            attribute or property returning the origin
#     world_origin      attribute or property returning the world origin
#     _local_wcs        property returning a local WCS with the same pixel shape
#     _writeHeader      function that writes the WCS to a fits header.
#     _readHeader       static function that reads the WCS from a fits header.
#     _newOrigin        function returning the saem WCS, but with new origin, world_origin
#     copy              return a copy
#     __repr__          convert to string
#
#########################################################################################


class OffsetWCS(UniformWCS):
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
        self._scale = scale
        self._local_wcs = PixelScale(scale)
        if origin == None:
            self._origin = galsim.PositionD(0,0)
        else:
            self._origin = origin
        if world_origin == None:
            self._world_origin = galsim.PositionD(0,0)
        else:
            self._world_origin = world_origin

    @property
    def scale(self): return self._scale

    @property
    def origin(self): return self._origin
    @property
    def world_origin(self): return self._world_origin

    def isPixelScale(self):
        return True

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

    def _newOrigin(self, origin, world_origin):
        return OffsetWCS(self._scale, origin, world_origin)

    def copy(self):
        return OffsetWCS(self._scale, self.origin, self.world_origin)

    def __repr__(self): return "OffsetWCS(%r,%r,%r)"%(self.scale, self.origin,
                                                      self.world_origin)


class OffsetShearWCS(UniformWCS):
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
        # The shear stuff is not too complicated, but enough so that it is worth
        # encapsulating in the ShearWCS class.  So here, we just create one of those
        # and we'll pass along any shear calculations to that.
        self._local_wcs = ShearWCS(scale, shear)
        if origin == None:
            self._origin = galsim.PositionD(0,0)
        else:
            self._origin = origin
        if world_origin == None:
            self._world_origin = galsim.PositionD(0,0)
        else:
            self._world_origin = world_origin


    @property
    def scale(self): return self._local_wcs.scale
    @property
    def shear(self): return self._local_wcs.shear

    @property
    def origin(self): return self._origin
    @property
    def world_origin(self): return self._world_origin
    
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

    def _newOrigin(self, origin, world_origin):
        return OffsetShearWCS(self.scale, self.shear, origin, world_origin)

    def copy(self):
        return OffsetShearWCS(self.scale, self.shear, self.origin, self.world_origin)

    def __repr__(self):
        return "OffsetShearWCS(%r,%r, %r,%r)"%(self.scale, self.shear,
                                               self.origin, self.world_origin)


class AffineTransform(UniformWCS):
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
        # As with OffsetShearWCS, we store a JacobianWCS, rather than reimplement everything.
        self._local_wcs = JacobianWCS(dudx, dudy, dvdx, dvdy)
        if origin == None:
            self._origin = galsim.PositionD(0,0)
        else:
            self._origin = origin
        if world_origin == None:
            self._world_origin = galsim.PositionD(0,0)
        else:
            self._world_origin = world_origin

    @property
    def dudx(self): return self._local_wcs.dudx
    @property
    def dudy(self): return self._local_wcs.dudy
    @property
    def dvdx(self): return self._local_wcs.dvdx
    @property
    def dvdy(self): return self._local_wcs.dvdy

    @property
    def origin(self): return self._origin
    @property
    def world_origin(self): return self._world_origin
 
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

    def _newOrigin(self, origin, world_origin):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy,
                               origin, world_origin)

    def copy(self):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy,
                               self.origin, self.world_origin)

    def __repr__(self):
        return "AffineTransform(%r,%r,%r,%r,%r,%r)"%(self.dudx, self.dudy, self.dvdx, self.dvdy,
                                                     self.origin, self.world_origin)


#########################################################################################
#
# Non-uniform WCS classes are those where the pixel size and shape are not necessarily
# constant across the image.  There are two varieties of these, EuclideanWCS and CelestialWCS.
#
# Here, we have the following non-uniform WCS classes: (There are more in fitswcs.py)
#
#     UVFunction is a EuclideanWCS
#     RaDecFunction is a CelestialWCS
#
# They must define the following:
#
#     origin            attribute or property returning the origin
#     _writeHeader      function that writes the WCS to a fits header.
#     _readHeader       static function that reads the WCS from a fits header.
#     _newOrigin        function returning the saem WCS, but with new origin
#     copy              return a copy
#     __eq__            check if this equals another WCS
#     __repr__          convert to string
#
# Non-uniform, EuclideanWCS classes must define the following:
#
#     world_origin      attribute or property returning the world origin
#     _u                function returning u(x,y)
#     _v                function returning v(x,y)
#     _x                function returning x(u,v)  (May raise a NotImplementedError)
#     _y                function returning y(u,v)  (May raise a NotImplementedError)
#
# CelestialWCS classes must define the following:
#
#     _radec            function returning (ra, dec) in _radians_ at position (x,y)
#
# Ideally, the above functions would work with numpy arrays as inputs.
#
#########################################################################################


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

class UVFunction(EuclideanWCS):
    """This WCS takes two arbitrary functions for u(x,y) and v(x,y).

    The ufunc and vfunc parameters may be:
        - python functions that take (x,y) arguments
        - python objects with a __call__ method that takes (x,y) arguments
        - strings which can be parsed with eval('lambda x,y: '+str)

    You may also provide the inverse functions x(u,v) and y(u,v) as xfunc and yfunc. 
    These are not required, but if you do not provide them, then any operation that requires 
    going from world to image coordinates will raise a NotImplementedError.

    Note: some internal calculations will be faster if the functions can take numpy arrays
    for x,y and output arrays for u,v.  Usually this does not require any change to your 
    function, but it is worth keeping in mind.  For example, if you want to do a sqrt, you 
    may be better off using `numpy.sqrt` rather than `math.sqrt`.

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
        import math  # In case needed by function evals
        import numpy
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
            self._origin = galsim.PositionD(0,0)
        else:
            self._origin = origin
        if world_origin == None:
            self._world_origin = galsim.PositionD(0,0)
        else:
            self._world_origin = world_origin

    @property
    def ufunc(self): return self._ufunc
    @property
    def vfunc(self): return self._vfunc
    @property
    def xfunc(self): return self._xfunc
    @property
    def yfunc(self): return self._yfunc

    @property
    def origin(self): return self._origin
    @property
    def world_origin(self): return self._world_origin

    def _u(self, x, y):
        import math
        import numpy
        return self._ufunc(x,y)

    def _v(self, x, y):
        import math
        import numpy
        return self._vfunc(x,y)

    def _x(self, u, v):
        if self._xfunc is None:
            raise NotImplementedError(
                "World -> Image direction not implemented for this UVFunction")
        else:
            import math
            import numpy
            return self._xfunc(u,v)

    def _y(self, u, v):
        if self._yfunc is None:
            raise NotImplementedError(
                "World -> Image direction not implemented for this UVFunction")
        else:
            import math
            import numpy
            return self._yfunc(u,v)

    def _newOrigin(self, origin, world_origin):
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
        return ( isinstance(other, UVFunction) and
                 self._ufunc == other._ufunc and
                 self._vfunc == other._vfunc and
                 self._xfunc == other._xfunc and
                 self._yfunc == other._yfunc and
                 self.origin == other.origin and
                 self.world_origin == other.world_origin )

    def __repr__(self):
        return "UVFunction(%r,%r,%r,%r,%r,%r)"%(self._ufunc, self._vfunc, self._xfunc, self._yfunc,
                                                self.origin, self.world_origin)


class RaDecFunction(CelestialWCS):
    """This WCS takes an arbitrary function for the Right Ascension and Declination.

    The radec_func(x,y) may be:
        - a python functions that take (x,y) arguments
        - a python object with a __call__ method that takes (x,y) arguments
        - a string which can be parsed with eval('lambda x,y: '+str)

    The functions should return a tuple of ( ra , dec ) in _radians_.
    
    We don't want a function that return galsim.Angles, because we want to allow for the 
    possibility of using numpy arrays as inputs and outputs to speed up some calculations.  The 
    function isn't _required_ to work with numpy arrays, but it is possible that some things 
    will be faster if it does.  If it were expected to return galsim.Angles, then it definitely
    couldn't work with arrays.

    Initialization
    --------------
    An RaDecFunction is initialized with the command:

        wcs = galsim.RaDecFunction(radec_func, origin=None)

    @param radec_func     A function radec(x,y) returning (ra, dec) in radians.
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
    """
    _req_params = { "radec_func" : str }
    _opt_params = { "origin" : galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, radec_func, origin=None):
        # Allow the input function to use either math or numpy functions
        if isinstance(radec_func, basestring):
            import math
            import numpy
            self._radec_func = eval('lambda x,y : ' + radec_func)
            self._radec_func_str = radec_func
        else:
            self._radec_func = radec_func
            self._radec_func_str = None

        if origin == None:
            self._origin = galsim.PositionD(0,0)
        else:
            self._origin = origin

    @property
    def radec_func(self): return self._radec_func

    @property
    def origin(self): return self._origin

    def _radec(self, x, y):
        import math
        import numpy
        return self._radec_func(x,y)

    def _xy(self, ra, dec):
        raise NotImplementedError("World -> Image direction not implemented for RaDecFunction")

    def _newOrigin(self, origin):
        return RaDecFunction(self._radec_func, origin)
 
    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("RaDecFunction", "GalSim WCS name")
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")

        _writeFuncToHeader(self._radec_func, self._radec_func_str, 'F', header)

        return self.affine(bounds.trueCenter())._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        radec_func = _readFuncFromHeader('F', header)
        return RaDecFunction(radec_func, galsim.PositionD(x0,y0))

    def copy(self):
        return RaDecFunction(self._radec_func, self.origin)

    def __eq__(self, other):
        return ( isinstance(other, RaDecFunction) and
                 self._radec_func == other._radec_func and
                 self.origin == other.origin )

    def __repr__(self):
        return "RaDecFunction(%r,%r,%r)"%(self.radec_func, self.origin)

