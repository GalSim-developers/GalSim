# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import numpy as np

from .gsobject import GSObject
from .position import Position, PositionD, _PositionI, _PositionD
from .celestial import CelestialCoord
from .shear import Shear
from .errors import GalSimError, GalSimIncompatibleValuesError, GalSimNotImplementedError
from .errors import GalSimValueError
from .utilities import doc_inherit, lazy_property

class BaseWCS(object):
    """The base class for all other kinds of WCS transformations.

    All the functions the user will typically need are defined here.  Most subclasses just
    define helper functions to implement each particular WCS definition.  So this base
    class defines the common interface for all WCS classes.

    There are several types of WCS classes that we implement. The basic class hierarchy is::

        `BaseWCS`
            --- `EuclideanWCS`
                    --- `UniformWCS`
                            --- `LocalWCS`
            --- `CelestialWCS`

    These base classes are not constructible.  They do not have __init__ defined.

    1. `LocalWCS` classes are those which really just define a pixel size and shape.
       They implicitly have the origin in image coordinates correspond to the origin
       in world coordinates.  They are primarily designed to handle local transformations
       at the location of a single galaxy, where it should usually be a good approximation
       to consider the pixel shape to be constant over the size of the galaxy.

       Currently we define the following `LocalWCS` classes::

       - `PixelScale`
       - `ShearWCS`
       - `JacobianWCS`

    2. `UniformWCS` classes have a constant pixel size and shape, but they have an arbitrary origin
       in both image coordinates and world coordinates.  A `LocalWCS` class can be turned into a
       non-local `UniformWCS` class when an image has its bounds changed, e.g. by the commands
       `Image.setCenter`, `Image.setOrigin` or `Image.shift`.

       Currently we define the following non-local, `UniformWCS` classes::

       - `OffsetWCS`
       - `OffsetShearWCS`
       - `AffineTransform`

    3. `EuclideanWCS` classes use a regular Euclidean coordinate system for the world coordinates,
       using `PositionD` for the world positions.  We use the notation (u,v) for the world
       coordinates and (x,y) for the image coordinates.

       Currently we define the following non-uniform, `EuclideanWCS` class::

       - `UVFunction`

    4. `CelestialWCS` classes are defined with their world coordinates on the celestial sphere
       in terms of right ascension (RA) and declination (Dec).  The pixel size and shape are
       always variable.  We use `CelestialCoord` for the world coordinates, which helps
       facilitate the spherical trigonometry that is sometimes required.

       Currently we define the following `CelestialWCS` classes: (All but the first are defined
       in the file fitswcs.py.)

       - `RaDecFunction`
       - `AstropyWCS`          -- requires astropy.wcs python module to be installed
       - `PyAstWCS`            -- requires starlink.Ast python module to be installed
       - `WcsToolsWCS`         -- requires wcstools command line functions to be installed
       - `GSFitsWCS`           -- native code, but has less functionality than the above

    There are also a few factory functions in fitswcs.py intended to act like class initializers:

    - `FitsWCS` tries to read a fits file using one of the above classes and returns an instance of
      whichever one it found was successful.  It should always be successful, since its final
      attempt uses `AffineTransform`, which has reasonable defaults when the WCS key words are not
      in the file, but of course this will only be a very rough approximation of the true WCS.

    - `TanWCS` constructs a simple tangent plane projection WCS directly from the projection
      parameters instead of from a fits header.

    - `FittedSIPWCS` constructs a TAN-SIP WCS by fitting to a list of reference celestial and image
      coordinates.

    Some things you can do with a WCS instance:

    - Convert positions between image coordinates and world coordinates (sometimes referred
      to as sky coordinates)::

            >>> world_pos = wcs.toWorld(image_pos)
            >>> image_pos = wcs.toImage(world_pos)

      Note: the transformation from world to image coordinates is not guaranteed to be
      implemented.  If it is not implemented for a particular WCS class, a NotImplementedError
      will be raised.

      The ``image_pos`` parameter should be a `PositionD`.  However, ``world_pos`` will
      be a `CelestialCoord` if the transformation is in terms of celestial coordinates
      (if ``wcs.isCelestial() == True``).  Otherwise, it will be a `PositionD` as well.

    - Convert a `GSObject` that is defined in world coordinates to the equivalent profile defined
      in terms of image coordinates (or vice versa)::

            >>> image_profile = wcs.toImage(world_profile)
            >>> world_profile = wcs.toWorld(image_profile)

      For non-uniform WCS types (for which ``wcs.isUniform() == False``), these need either an
      ``image_pos`` or ``world_pos`` parameter to say where this conversion should happen::

            >>> image_profile = wcs.toImage(world_profile, image_pos=image_pos)

    - Construct a local linear approximation of a WCS at a given location::

            >>> local_wcs = wcs.local(image_pos = image_pos)
            >>> local_wcs = wcs.local(world_pos = world_pos)

      If ``wcs.toWorld(image_pos)`` is not implemented for a particular WCS class, then a
      NotImplementedError will be raised if you pass in a ``world_pos`` argument.

      The returned ``local_wcs`` is usually a `JacobianWCS` instance, but see the doc string for
      `local` for more details.

    - Construct a full affine approximation of a WCS at a given location::

            >>> affine_wcs = wcs.affine(image_pos = image_pos)
            >>> affine_wcs = wcs.affine(world_pos = world_pos)

      This preserves the transformation near the location of ``image_pos``, but it is linear, so
      the transformed values may not agree as you get farther from the given point.

      The returned ``affine_wcs`` is always an `AffineTransform` instance.

    - Get some properties of the pixel size and shape::

            >>> area = local_wcs.pixelArea()
            >>> min_linear_scale = local_wcs.minLinearScale()
            >>> max_linear_scale = local_wcs.maxLinearScale()
            >>> jac = local_wcs.jacobian()
            >>> # Use jac.dudx, jac.dudy, jac.dvdx, jac.dvdy

      Non-uniform WCS types also have these functions, but for them, you must supply either
      ``image_pos`` or ``world_pos``.  So the following are equivalent::

            >>> area = wcs.pixelArea(image_pos)
            >>> area = wcs.local(image_pos).pixelArea()

    - Query some overall attributes of the WCS transformation::

            >>> wcs.isLocal()       # is this a local WCS?
            >>> wcs.isUniform()     # does this WCS have a uniform pixel size/shape?
            >>> wcs.isCelestial()   # are the world coordinates on the celestial sphere?
            >>> wcs.isPixelScale()  # is this either a PixelScale or an OffsetWCS?
    """
    def toWorld(self, *args, **kwargs):
        """Convert from image coordinates to world coordinates.

        There are essentially three overloaded versions of this function here.

        1. The first converts a `Position` from image coordinates to world coordinates.
           It returns the corresponding position in world coordinates as a `PositionD` if the WCS
           is a `EuclideanWCS`, or a `CelestialCoord` if it is a `CelestialWCS`::

               >>> world_pos = wcs.toWorld(image_pos)

           Equivalent to ``wcs.posToWorld(image_pos)``.

        2. The second is nearly the same, but takes x and y values directly and returns
           either u, v or ra, dec, depending on the kind of wcs being used.  For this version,
           x and y may be numpy arrays, in which case the returned values are also numpy
           arrays::

               >>> u, v = wcs.toWorld(x, y)                 # For EuclideanWCS types
               >>> ra, dec = wcs.toWorld(x, y, units=units) # For CelestialWCS types

           Equivalent to ``wcs.xyTouv(x, y)`` or ``wcs.xyToradec(x, y, units=units)``.

        3. The third converts a surface brightness profile (a `GSObject`) from image
           coordinates to world coordinates, returning the profile in world coordinates
           as a new `GSObject`.  For non-uniform WCS transforms, you must provide either
           ``image_pos`` or ``world_pos`` to say where the profile is located, so the right
           transformation can be performed.  And optionally, you may provide a flux scaling
           to be performed at the same time::

               >>> world_profile = wcs.toWorld(image_profile, image_pos=None, world_pos=None,
                                               flux_ratio=1, offset=(0,0))

           Equivalent to ``wcs.profileToWorld(image_profile, ...)``.
        """
        from .chromatic import ChromaticObject
        if len(args) == 1:
            if isinstance(args[0], (GSObject, ChromaticObject)):
                return self.profileToWorld(*args, **kwargs)
            else:
                return self.posToWorld(*args, **kwargs)
        elif len(args) == 2:
            if self._isCelestial:
                return self.xyToradec(*args, **kwargs)
            else:
                return self.xyTouv(*args, **kwargs)
        else:
            raise TypeError("toWorld() takes either 1 or 2 positional arguments")

    def posToWorld(self, image_pos, color=None, **kwargs):
        """Convert a position from image coordinates to world coordinates.

        This is equivalent to ``wcs.toWorld(image_pos)``.

        Parameters:
            image_pos:      The position in image coordinates
            color:          For color-dependent WCS's, the color term to use. [default: None]
            project_center: (Only valid for `CelestialWCS`) A `CelestialCoord` to use for
                            projecting the result onto a tangent plane world system rather
                            than returning a `CelestialCoord`. [default: None]
            projection:     If project_center != None, the kind of projection to use.  See
                            `CelestialCoord.project` for the valid options. [default: 'gnomonic']

        Returns:
            world_pos
        """
        if color is None: color = self._color
        if not isinstance(image_pos, Position):
            raise TypeError("image_pos must be a PositionD or PositionI argument")
        return self._posToWorld(image_pos, color=color, **kwargs)

    def profileToWorld(self, image_profile, image_pos=None, world_pos=None, color=None,
                       flux_ratio=1., offset=(0,0)):
        """Convert a profile from image coordinates to world coordinates.

        This is equivalent to ``wcs.toWorld(image_profile, ...)``.

        Parameters:
            image_profile:  The profile in image coordinates to transform.
            image_pos:      The image coordinate position (for non-uniform WCS types)
            world_pos:      The world coordinate position (for non-uniform WCS types)
            color:          For color-dependent WCS's, the color term to use. [default: None]
            flux_ratio:     An optional flux scaling to be applied at the same time.
                            [default: 1]
            offset:         An optional offset to be applied at the same time. [default: 0,0]
        """
        if color is None: color = self._color
        return self.local(image_pos, world_pos, color=color)._profileToWorld(
                    image_profile, flux_ratio, PositionD(offset))

    def toImage(self, *args, **kwargs):
        """Convert from world coordinates to image coordinates

        There are essentially three overloaded versions of this function here.

        1. The first converts a position from world coordinates to image coordinates.
           If the WCS is a `EuclideanWCS`, the argument may be either a `PositionD` or `PositionI`
           argument.  If it is a `CelestialWCS`, then the argument must be a `CelestialCoord`.
           It returns the corresponding position in image coordinates as a `PositionD`::

               >>> image_pos = wcs.toImage(world_pos)

           Equivalent to `posToImage`.

        2. The second is nearly the same, but takes either u and v values or ra and dec values
           (depending on the kind of wcs being used) directly and returns x and y values.
           For this version, the inputs may be numpy arrays, in which case the returned values
           are also numpy arrays::

               >>> x, y = wcs.toImage(u, v)                 # For EuclideanWCS types
               >>> x, y = wcs.toImage(ra, dec, units=units) # For CelestialWCS types

           Equivalent to `uvToxy` or `radecToxy`.

        3. The third converts a surface brightness profile (a `GSObject`) from world
           coordinates to image coordinates, returning the profile in image coordinates
           as a new `GSObject`.  For non-uniform WCS transforms, you must provide either
           ``image_pos`` or ``world_pos`` to say where the profile is located so the right
           transformation can be performed.  And optionally, you may provide a flux scaling
           to be performed at the same time::

               >>> image_profile = wcs.toImage(world_profile, image_pos=None, world_pos=None,
                                               flux_ratio=1, offset=(0,0))

           Equivalent to `profileToImage`.
        """
        from .chromatic import ChromaticObject
        if len(args) == 1:
            if isinstance(args[0], (GSObject, ChromaticObject)):
                return self.profileToImage(*args, **kwargs)
            else:
                return self.posToImage(*args, **kwargs)
        elif len(args) == 2:
            if self._isCelestial:
                return self.radecToxy(*args, **kwargs)
            else:
                return self.uvToxy(*args, **kwargs)
        else:
            raise TypeError("toImage() takes either 1 or 2 positional arguments")

    def posToImage(self, world_pos, color=None):
        """Convert a position from world coordinates to image coordinates.

        This is equivalent to ``wcs.toImage(world_pos)``.

        Parameters:
            world_pos:  The world coordinate position
            color:      For color-dependent WCS's, the color term to use. [default: None]
        """
        if color is None: color = self._color
        if self._isCelestial and not isinstance(world_pos, CelestialCoord):
            raise TypeError("world_pos must be a CelestialCoord argument")
        elif not self._isCelestial and not isinstance(world_pos, Position):
            raise TypeError("world_pos must be a PositionD or PositionI argument")
        return self._posToImage(world_pos, color=color)

    def profileToImage(self, world_profile, image_pos=None, world_pos=None, color=None,
                       flux_ratio=1., offset=(0,0)):
        """Convert a profile from world coordinates to image coordinates.

        This is equivalent to ``wcs.toImage(world_profile, ...)``.

        Parameters:
            world_profile:  The profile in world coordinates to transform.
            image_pos:      The image coordinate position (for non-uniform WCS types)
            world_pos:      The world coordinate position (for non-uniform WCS types)
            color:          For color-dependent WCS's, the color term to use. [default: None]
            flux_ratio:     An optional flux scaling to be applied at the same time.
                                [default: 1]
            offset:         An optional offset to be applied at the same time. [default: 0,0]
        """
        if color is None: color = self._color
        return self.local(image_pos, world_pos, color=color)._profileToImage(
                    world_profile, flux_ratio, PositionD(offset))

    def pixelArea(self, image_pos=None, world_pos=None, color=None):
        """Return the area of a pixel in arcsec**2 (or in whatever units you are using for
        world coordinates if it is a `EuclideanWCS`).

        For non-uniform WCS transforms, you must provide either ``image_pos`` or ``world_pos``
        to say where the pixel is located.

        Parameters:
            image_pos:  The image coordinate position (for non-uniform WCS types)
            world_pos:  The world coordinate position (for non-uniform WCS types)
            color:      For color-dependent WCS's, the color term for which to evaluate the
                        pixel area. [default: None]

        Returns:
            the pixel area in arcsec**2.
        """
        if color is None: color = self._color
        return self.local(image_pos, world_pos, color=color)._pixelArea()

    def minLinearScale(self, image_pos=None, world_pos=None, color=None):
        """Return the minimum linear scale of the transformation in any direction.

        This is basically the semi-minor axis of the Jacobian.  Sometimes you need a
        linear scale size for some calculation.  This function returns the smallest
        scale in any direction.  The function maxLinearScale() returns the largest.

        For non-uniform WCS transforms, you must provide either ``image_pos`` or ``world_pos``
        to say where the pixel is located.

        Parameters:
            image_pos:  The image coordinate position (for non-uniform WCS types)
            world_pos:  The world coordinate position (for non-uniform WCS types)
            color:      For color-dependent WCS's, the color term for which to evaluate the
                        scale. [default: None]

        Returns:
            the minimum pixel area in any direction in arcsec.
        """
        if color is None: color = self._color
        return self.local(image_pos, world_pos, color=color)._minScale()

    def maxLinearScale(self, image_pos=None, world_pos=None, color=None):
        """Return the maximum linear scale of the transformation in any direction.

        This is basically the semi-major axis of the Jacobian.  Sometimes you need a
        linear scale size for some calculation.  This function returns the largest
        scale in any direction.  The function minLinearScale() returns the smallest.

        For non-uniform WCS transforms, you must provide either ``image_pos`` or ``world_pos``
        to say where the pixel is located.

        Parameters:
            image_pos:  The image coordinate position (for non-uniform WCS types)
            world_pos:  The world coordinate position (for non-uniform WCS types)
            color:      For color-dependent WCS's, the color term for which to evaluate the
                        scale. [default: None]

        Returns:
            the maximum pixel area in any direction in arcsec.
        """
        if color is None: color = self._color
        return self.local(image_pos, world_pos, color=color)._maxScale()

    def isPixelScale(self):
        """Return whether the WCS transformation is a simple `PixelScale` or `OffsetWCS`.

        These are the simplest two WCS transformations.  `PixelScale` is local and `OffsetWCS`
        is non-local.  If an `Image` has one of these WCS transformations as its WCS, then
        ``im.scale`` works to read and write the pixel scale.  If not, ``im.scale`` will raise a
        TypeError exception.

        ``wcs.isPixelScale()`` is shorthand for ``isinstance(wcs, (galsim.PixelScale,
        galsim.OffsetWCS))``.
        """
        return self._isPixelScale

    @property
    def _isPixelScale(self):
        return False   # Overridden by PixelScale and OffsetWCS

    def isLocal(self):
        """Return whether the WCS transformation is a local, linear approximation.

        ``wcs.isLocal()`` is shorthand for ``isinstance(wcs, galsim.LocalWCS)``.
        """
        return self._isLocal

    @property
    def _isLocal(self):
        return False   # Overridden by LocalWCS

    def isUniform(self):
        """Return whether the pixels in this WCS have uniform size and shape.

        ``wcs.isUniform()`` is shorthand for ``isinstance(wcs, galsim.UniformWCS)``.
        """
        return self._isUniform

    @property
    def _isUniform(self):
        return False   # Overridden by UniformWCS

    def isCelestial(self):
        """Return whether the world coordinates are `CelestialCoord` (i.e. ra,dec).

        ``wcs.isCelestial()`` is shorthand for ``isinstance(wcs, galsim.CelestialWCS)``.
        """
        return self._isCelestial

    @property
    def _isCelestial(self):
        return False   # Overridden by CelestialWCS

    def local(self, image_pos=None, world_pos=None, color=None):
        """Return the local linear approximation of the WCS at a given point.

        Parameters:
            image_pos:  The image coordinate position (for non-uniform WCS types)
            world_pos:  The world coordinate position (for non-uniform WCS types)
            color:      For color-dependent WCS's, the color term for which to evaluate the
                        local WCS. [default: None]

        Returns:
            a `LocalWCS` instance.
        """
        if color is None: color = self._color
        if world_pos is not None:
            if image_pos is not None:
                raise GalSimIncompatibleValuesError(
                    "Only one of image_pos or world_pos may be provided",
                    image_pos=image_pos, world_pos=world_pos)
            image_pos = self.posToImage(world_pos, color)
        if image_pos is not None and not isinstance(image_pos, Position):
            raise TypeError("image_pos must be a PositionD or PositionI argument")
        return self._local(image_pos, color)

    def jacobian(self, image_pos=None, world_pos=None, color=None):
        """Return the local `JacobianWCS` of the WCS at a given point.

        This is basically the same as local(), but the return value is guaranteed to be a
        `JacobianWCS`, which can be useful in some situations, since you can access the values
        of the 2x2 Jacobian matrix directly::

            >>> jac = wcs.jacobian(image_pos)
            >>> x,y = np.meshgrid(np.arange(0,32,1), np.arange(0,32,1))
            >>> u = jac.dudx * x + jac.dudy * y
            >>> v = jac.dvdx * x + jac.dvdy * y
            >>> # ... use u,v values to work directly in world coordinates.

        If you do not need the extra functionality, then you should use local()
        instead, since it may be more efficient.

        Parameters:
            image_pos:  The image coordinate position (for non-uniform WCS types)
            world_pos:  The world coordinate position (for non-uniform WCS types)
            color:      For color-dependent WCS's, the color term for which to evaluate the
                        local jacobian. [default: None]

        Returns:
            a `JacobianWCS` instance.
        """
        if color is None: color = self._color
        return self.local(image_pos, world_pos, color=color)._toJacobian()

    def affine(self, image_pos=None, world_pos=None, color=None):
        """Return the local `AffineTransform` of the WCS at a given point.

        This returns a linearized version of the current WCS at a given point.  It
        returns an `AffineTransform` that is locally approximately the same as the WCS in
        the vicinity of the given point.

        It is similar to jacobian(), except that this preserves the offset information
        between the image coordinates and world coordinates rather than setting both
        origins to (0,0).  Instead, the image origin is taken to be ``image_pos``.

        For non-celestial coordinate systems, the world origin is taken to be
        ``wcs.toWorld(image_pos)``.  In fact, ``wcs.affine(image_pos)`` is really just
        shorthand for::

            >>> wcs.jacobian(image_pos).withOrigin(image_pos, wcs.toWorld(image_pos))

        For celestial coordinate systems, there is no well-defined choice for the
        origin of the Euclidean world coordinate system.  So we just take (u,v) = (0,0)
        at the given position.  So, ``wcs.affine(image_pos)`` is equivalent to::

            >>> wcs.jacobian(image_pos).withOrigin(image_pos)

        You can use the returned `AffineTransform` to access the relevant values of the 2x2
        Jacobian matrix and the origins directly::

            >>> affine = wcs.affine(image_pos)
            >>> x,y = np.meshgrid(np.arange(0,32,1), np.arange(0,32,1))
            >>> u = affine.dudx * (x-affine.x0) + jac.dudy * (y-affine.y0) + affine.u0
            >>> v = affine.dvdx * (x-affine.x0) + jac.dvdy * (y-affine.y0) + affine.v0
            >>> # ... use u,v values to work directly in sky coordinates.

        As usual, you may provide either ``image_pos`` or ``world_pos`` as you prefer to
        specify the location at which to approximate the WCS.

        Parameters:
            image_pos:  The image coordinate position (for non-uniform WCS types)
            world_pos:  The world coordinate position (for non-uniform WCS types)
            color:      For color-dependent WCS's, the color term for which to evaluate the
                        local affine transform. [default: None]

        Returns:
            an `AffineTransform` instance
        """
        if color is None: color = self._color
        jac = self.jacobian(image_pos, world_pos, color=color)
        # That call checked that only one of image_pos or world_pos is provided.
        if world_pos is not None:
            image_pos = self.toImage(world_pos, color=color)
        elif image_pos is None:
            # Both are None.  Must be a local WCS
            image_pos = _PositionD(0,0)

        if self._isCelestial:
            return jac.withOrigin(image_pos)
        else:
            if world_pos is None:
                world_pos = self.toWorld(image_pos, color=color)
            return jac.withOrigin(image_pos, world_pos, color=color)

    def shiftOrigin(self, origin, world_origin=None, color=None):
        """Shift the origin of the current WCS function, returning the new WCS.

        This function creates a new WCS instance (always a non-local WCS) that shifts the
        origin by the given amount.  In other words, it treats the image position ``origin``
        the same way the current WCS treats (x,y) = (0,0).

        If the current WCS is a local WCS, this essentially declares where on the image
        you want the origin of the world coordinate system to be.  i.e. where is (u,v) = (0,0).
        So, for example, to set a WCS that has a constant pixel size with the world coordinates
        centered at the center of an image, you could write::

            >>> wcs = galsim.PixelScale(scale).shiftOrigin(im.center)

        This is equivalent to the following::

            >>> wcs = galsim.OffsetWCS(scale, origin=im.center)

        For non-local WCS types, the origin defines the location in the image coordinate system
        should mean the same thing as (x,y) = (0,0) does for the current WCS.  The following
        example should work regardless of what kind of WCS this is::

            >>> world_pos1 = wcs.toWorld(PositionD(0,0))
            >>> wcs2 = wcs.shiftOrigin(new_origin)
            >>> world_pos2 = wcs2.toWorld(new_origin)
            >>> # world_pos1 should be equal to world_pos2

        Furthermore, if the current WCS is a `EuclideanWCS` (wcs.isCelestial() == False) you may
        also provide a ``world_origin`` argument which defines what (u,v) position you want to
        correspond to the new origin.  Continuing the previous example::

            >>> wcs3 = wcs.shiftOrigin(new_origin, new_world_origin)
            >>> world_pos3 = wcs3.toWorld(new_origin)
            >>> # world_pos3 should be equal to new_world_origin

        Parameters:
            origin:         The image coordinate position to use for what is currently treated
                            as (0,0).
            world_origin:   The world coordinate position to use at the origin.  Only valid if
                            wcs.isCelestial() == False. [default: None]
            color:          For color-dependent WCS's, the color term to use in the connection
                            between the current origin and world_origin. [default: None]

        Returns:
            the new shifted WCS
        """
        if color is None: color = self._color
        if not isinstance(origin, Position):
            raise TypeError("origin must be a PositionD or PositionI argument")
        return self._shiftOrigin(origin, world_origin, color)

    def withOrigin(self, origin, world_origin=None, color=None):
        from .deprecated import depr
        depr('withOrigin', 2.3, 'shiftOrigin')
        return self.shiftOrigin(origin, world_origin, color)

    def fixColor(self, color):
        """Fix the color to a particular value.

        This changes a color-dependent WCS into the corresponding color-independent WCS
        for the given color.

        Parameters:
            color:      The value of the color term to use.

        Returns:
            the new color-independent WCS
        """
        ret = self.copy()
        ret._color = color
        return ret

    def writeToFitsHeader(self, header, bounds):
        """Write this WCS function to a FITS header.

        This is normally called automatically from within the galsim.fits.write() function.

        The code will attempt to write standard FITS WCS keys so that the WCS will be readable
        by other software (e.g. ds9).  It may not be able to do so accurately, in which case a
        linearized version will be used instead.  (Specifically, it will use the local affine
        transform with respect to the image center.)

        However, this is not necessary for the WCS to survive a round trip through the FITS
        header, as it will also write GalSim-specific key words that should allow it to
        reconstruct the WCS correctly.

        .. note:
            For `UVFunction` and `RaDecFunction`, if the functions are real python functions
            (rather than a string that is converted to a function), then the mechanism we use to
            convert the function to a string that can be written to the header has a few
            limitations.

            1. It apparently only works for cpython implementations.
            2. It probably won't work to write from one version of python and read from another.
               (At least for major version differences.)
            3. If the function uses globals, you'll need to make sure the globals are present
               when you read it back in as well, or it probably won't work.
            4. It looks really ugly in the header.
            5. We haven't thought much about the security implications of this, so beware using
               GalSim to open FITS files from untrusted sources.

        Parameters:
            header:     A FitsHeader (or dict-like) object to write the data to.
            bounds:     The bounds of the image.
        """
        from . import fits
        # First write the XMIN, YMIN values
        header["GS_XMIN"] = (bounds.xmin, "GalSim image minimum x coordinate")
        header["GS_YMIN"] = (bounds.ymin, "GalSim image minimum y coordinate")

        if bounds.xmin != 1 or bounds.ymin != 1:
            # ds9 always assumes the image has an origin at (1,1), so we always write the
            # WCS to the file with this convention.  We'll convert back when we read it
            # in if necessary.
            delta = _PositionI(1-bounds.xmin, 1-bounds.ymin)
            bounds = bounds.shift(delta)
            wcs = self.shiftOrigin(delta)
        else:
            wcs = self

        wcs._writeHeader(header, bounds)

        if hasattr(self, 'header'):
            # Store the items that are in self.header in the header if they weren't already put
            # there by the call to wcs._writeHeader() call.  (We don't want to overwrite the WCS.)
            for key in self.header:
                if (key not in header and key.strip() != '' and
                    key.strip() != 'COMMENT' and key.strip() != 'HISTORY'):
                    header[key] = self.header[key]

    def makeSkyImage(self, image, sky_level, color=None):
        """Make an image of the sky, correctly accounting for the pixel area, which might be
        variable over the image.

        Note: This uses finite differences of the wcs mapping to calculate the area of each
              pixel in world coordinates.  It is usually pretty accurate everywhere except
              within a few arcsec of the north or south poles.

        Parameters:
            image:      The image onto which the sky values will be put.
            sky_level:  The sky level in ADU/arcsec^2 (or whatever your world coordinate
                        system units are, if not arcsec).
            color:      For color-dependent WCS's, the color term to use for making the
                        sky image. [default: None]
        """
        if color is None: color = self._color
        self._makeSkyImage(image, sky_level, color)


    # A lot of classes will need these checks, so consolidate them here
    def _set_origin(self, origin, world_origin=None):
        if origin is None:
            self._origin = _PositionD(0,0)
        else:
            if not isinstance(origin, Position):
                raise TypeError("origin must be a PositionD or PositionI argument")
            self._origin = origin
        if world_origin is None:
            self._world_origin = _PositionD(0,0)
        else:
            if not isinstance(world_origin, Position):
                raise TypeError("world_origin must be a PositionD argument")
            self._world_origin = world_origin


def readFromFitsHeader(header, suppress_warning=True):
    """Read a WCS function from a FITS header.

    This is normally called automatically from within the `galsim.fits.read` function, but
    you can also call it directly as::

        wcs, origin = galsim.wcs.readFromFitsHeader(header)

    If the file was originally written by GalSim using one of the galsim.fits.write() functions,
    then this should always succeed in reading back in the original WCS.  It may not end up
    as exactly the same class as the original, but the underlying world coordinate system
    transformation should be preserved.

    .. note::
        For `UVFunction` and `RaDecFunction`, if the functions that were written to the FITS
        header were real python functions (rather than a string that is converted to a function),
        then the mechanism we use to write to the header and read it back in has some limitations:

        1. It apparently only works for cpython implementations.
        2. It probably won't work to write from one version of python and read from another.
           (At least for major version differences.)
        3. If the function uses globals, you'll need to make sure the globals are present
           when you read it back in as well, or it probably won't work.
        4. It looks really ugly in the header.
        5. We haven't thought much about the security implications of this, so beware using
           GalSim to open FITS files from untrusted sources.

    If the file was not written by GalSim, then this code will do its best to read the
    WCS information in the FITS header.  Depending on what kind of WCS is encoded in the
    header, this may or may not be successful.

    If there is no WCS information in the header, then this will default to a pixel scale
    of 1.

    In addition to the wcs, this function will also return the image origin that the WCS
    is assuming for the image.  If the file was originally written by GalSim, this should
    correspond to the original image origin.  If not, it will default to (1,1).

    Parameters:
        header:             The fits header with the WCS information.
        suppress_warning:   Whether to suppress a warning that the WCS could not be read from the
                            FITS header, so the WCS defaulted to either a `PixelScale` or
                            `AffineTransform`. [default: True]

    Returns:
        a tuple (wcs, origin) of the wcs from the header and the image origin.
    """
    from . import fits
    from .fitswcs import FitsWCS
    if not isinstance(header, fits.FitsHeader):
        header = fits.FitsHeader(header)
    xmin = header.get("GS_XMIN", 1)
    ymin = header.get("GS_YMIN", 1)
    origin = _PositionI(xmin, ymin)
    wcs_name = header.get("GS_WCS", None)
    if wcs_name is not None:
        gdict = globals().copy()
        exec('import galsim', gdict)
        wcs_type = eval('galsim.' + wcs_name, gdict)
        wcs = wcs_type._readHeader(header)
    else:
        # If we aren't told which type to use, this should find something appropriate
        wcs = FitsWCS(header=header, suppress_warning=suppress_warning)

    if xmin != 1 or ymin != 1:
        # ds9 always assumes the image has an origin at (1,1), so convert back to actual
        # xmin, ymin if necessary.
        delta = _PositionI(xmin-1, ymin-1)
        wcs = wcs.shiftOrigin(delta)

    return wcs, origin


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
    """A EuclideanWCS is a `BaseWCS` whose world coordinates are on a Euclidean plane.
    We usually use the notation (u,v) to refer to positions in world coordinates, and
    they use the class `PositionD`.
    """

    # All EuclideanWCS classes must define origin and world_origin.
    # Sometimes it is convenient to access x0,y0,u0,v0 directly.
    @property
    def x0(self):
        """The x component of self.origin.
        """
        return self.origin.x

    @property
    def y0(self):
        """The y component of self.origin.
        """
        return self.origin.y

    @property
    def u0(self):
        """The x component of self.world_origin (aka u).
        """
        return self.world_origin.x

    @property
    def v0(self):
        """The y component of self.world_origin (aka v).
        """
        return self.world_origin.y

    def xyTouv(self, x, y, color=None):
        """Convert x,y from image coordinates to world coordinates.

        This is equivalent to ``wcs.toWorld(x,y)``.

        It is also equivalent to ``wcs.posToWorld(galsim.PositionD(x,y))`` when x and y are scalars;
        however, this routine allows x and y to be numpy arrays, in which case, the calculation
        will be vectorized, which is often much faster than using the pos interface.

        Parameters:
            x:          The x value(s) in image coordinates
            y:          The y value(s) in image coordinates
            color:      For color-dependent WCS's, the color term to use. [default: None]

        Returns:
            ra, dec
        """
        from .angle import AngleUnit
        if color is None: color = self._color
        return self._xyTouv(x, y, color=color)

    def uvToxy(self, u, v, color=None):
        """Convert u,v from world coordinates to image coordinates.

        This is equivalent to ``wcs.toWorld(u,v)``.

        It is also equivalent to ``wcs.posToImage(galsim.PositionD(u,v))`` when u and v are scalars;
        however, this routine allows u and v to be numpy arrays, in which case, the calculation
        will be vectorized, which is often much faster than using the pos interface.

        Parameters:
            u:          The u value(s) in world coordinates
            v:          The v value(s) in world coordinates
            color:      For color-dependent WCS's, the color term to use. [default: None]
        """
        if color is None: color = self._color
        return self._uvToxy(u, v, color)

    # Simple.  Just call _u, _v.
    def _posToWorld(self, image_pos, color):
        x = image_pos.x - self.x0
        y = image_pos.y - self.y0
        return _PositionD(self._u(x,y,color), self._v(x,y,color)) + self.world_origin

    def _xyTouv(self, x, y, color):
        x = x - self.x0  # Not -=, since don't want to modify the input arrays in place.
        y = y - self.y0
        u = self._u(x,y,color)
        v = self._v(x,y,color)
        u += self.u0
        v += self.v0
        return u,v

    # Also simple if _x,_y are implemented.  However, they are allowed to raise a
    # NotImplementedError.
    def _posToImage(self, world_pos, color):
        u = world_pos.x - self.u0
        v = world_pos.y - self.v0
        return _PositionD(self._x(u,v,color),self._y(u,v,color)) + self.origin

    def _uvToxy(self, u, v, color):
        u = u - self.u0
        v = v - self.v0
        x = self._x(u,v,color)
        y = self._y(u,v,color)
        x += self.x0
        y += self.y0
        return x, y

    # Each subclass has a function _newOrigin, which just calls the constructor with new
    # values for origin and world_origin.  This function figures out what those values
    # should be to match the desired behavior of shiftOrigin.
    def _shiftOrigin(self, origin, world_origin, color):
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
            if not self._isLocal:
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
            if not isinstance(world_origin, Position):
                raise TypeError("world_origin must be a PositionD or PositionI argument")
            if not self._isLocal:
                world_origin += self.world_origin - self._posToWorld(self.origin, color=color)
            return self._newOrigin(origin, world_origin)

    # If the class doesn't define something else, then we can approximate the local Jacobian
    # from finite differences for the derivatives.  This will be overridden by UniformWCS.
    def _local(self, image_pos, color):

        if image_pos is None:
            raise TypeError("origin must be a PositionD or PositionI argument")

        # Calculate the Jacobian using finite differences for the derivatives.
        x0 = image_pos.x - self.x0
        y0 = image_pos.y - self.y0

        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        xlist = np.array([ x0+dx, x0-dx, x0,    x0    ], dtype=float)
        ylist = np.array([ y0,    y0,    y0+dy, y0-dy ], dtype=float)
        u = self._u(xlist,ylist,color)
        v = self._v(xlist,ylist,color)

        dudx = 0.5 * (u[0] - u[1]) / dx
        dudy = 0.5 * (u[2] - u[3]) / dy
        dvdx = 0.5 * (v[0] - v[1]) / dx
        dvdy = 0.5 * (v[2] - v[3]) / dy

        return JacobianWCS(dudx, dudy, dvdx, dvdy)

    # The naive way to make the sky image is to loop over pixels and call pixelArea(pos)
    # for that position.  This is extremely slow.  Here, we use the fact that the _u and _v
    # functions might work with numpy arrays.  If they do, this function is quite fast.
    # If not, we still get some gain from calculating u,v for each pixel and sharing some
    # of those calculations for multiple finite difference derivatives.  But the latter
    # option is still pretty slow, so it's much better to have the _u and _v work with
    # numpy arrays!
    def _makeSkyImage(self, image, sky_level, color):
        b = image.bounds
        nx = b.xmax-b.xmin+1 + 2  # +2 more than in image to get row/col off each edge.
        ny = b.ymax-b.ymin+1 + 2
        x,y = np.meshgrid( np.linspace(b.xmin-1,b.xmax+1,nx),
                           np.linspace(b.ymin-1,b.ymax+1,ny) )
        x -= self.x0
        y -= self.y0
        u = self._u(x.ravel(),y.ravel(),color)
        v = self._v(x.ravel(),y.ravel(),color)
        u = np.reshape(u, x.shape)
        v = np.reshape(v, x.shape)
        # Use the finite differences to estimate the derivatives.
        dudx = 0.5 * (u[1:ny-1,2:nx] - u[1:ny-1,0:nx-2])
        dudy = 0.5 * (u[2:ny,1:nx-1] - u[0:ny-2,1:nx-1])
        dvdx = 0.5 * (v[1:ny-1,2:nx] - v[1:ny-1,0:nx-2])
        dvdy = 0.5 * (v[2:ny,1:nx-1] - v[0:ny-2,1:nx-1])

        area = np.abs(dudx * dvdy - dvdx * dudy)
        image.array[:,:] = area * sky_level

    # Each class should define the __eq__ function.  Then __ne__ is obvious.
    def __ne__(self, other): return not self.__eq__(other)


class UniformWCS(EuclideanWCS):
    """A UniformWCS is a `EuclideanWCS` which has a uniform pixel size and shape.
    """
    @property
    def _isUniform(self):
        return True

    # These can also just pass through to the _localwcs attribute.
    def _u(self, x, y, color=None):
        return self._local_wcs._u(x,y)
    def _v(self, x, y, color=None):
        return self._local_wcs._v(x,y)
    def _x(self, u, v, color=None):
        return self._local_wcs._x(u,v)
    def _y(self, u, v, color=None):
        return self._local_wcs._y(u,v)

    # For UniformWCS, the local WCS is an attribute.  Just return it.
    def _local(self, image_pos, color):
        return self._local_wcs

    # UniformWCS transformations can be inverted easily, so might as well provide that function.
    def inverse(self):
        """Return the inverse transformation, i.e. the transformation that swaps the roles of
        the "image" and "world" coordinates.
        """
        return self._inverse()

    # We'll override this for LocalWCS classes. Non-local UniformWCS classes can use that function
    # do the inversion.
    def _inverse(self):
        return self._local_wcs._inverse()._newOrigin(self.world_origin, self.origin)

    # This is very simple if the pixels are uniform.
    def _makeSkyImage(self, image, sky_level, color):
        image.fill(sky_level * self.pixelArea())

    # Just check if the locals match and if the origins match.
    def __eq__(self, other):
        return (self is other or
                (isinstance(other, self.__class__) and
                 self._local_wcs == other._local_wcs and
                 self.origin == other.origin and
                 self.world_origin == other.world_origin))


class LocalWCS(UniformWCS):
    """A LocalWCS is a `UniformWCS` in which (0,0) in image coordinates is at the same place
    as (0,0) in world coordinates
    """
    def withOrigin(self, origin, world_origin=None, color=None):
        """Recenter the current WCS function at a new origin location, returning the new WCS.

        This function creates a new WCS instance (a non-local WCS) with the same local
        behavior as the current WCS, but with the given origin.  In other words, you are
        declaring where on the image you want the new origin of the world coordinate
        system to be.  i.e. where is (u,v) = (0,0).

        So, for example, to set a WCS that has a constant pixel size with the world coordinates
        centered at the center of an image, you could write::

            >>> wcs = galsim.PixelScale(scale).withOrigin(im.center)

        This is equivalent to the following::

            >>> wcs = galsim.OffsetWCS(scale, origin=im.center)

        You may also provide a ``world_origin`` argument which defines what (u,v) position you
        want to correspond to the new origin.

            >>> wcs2 = galsim.PixelScale(scale).withOrigin(new_origin, new_world_origin)
            >>> world_pos2 = wcs2.toWorld(new_origin)
            >>> assert world_pos2 == new_world_origin

        .. note::

            This is equivalent to `shiftOrigin`, but for for local WCS's, the shift is also
            the new location of the origin, so `withOrigin` is a convenient alternate name
            for this action.  Indeed the `shiftOrigin` function used to be named `withOrigin`,
            but that name was confusing for non-local WCS's, as the action in that case is really
            shifting the origin, not setting the new value.

        Parameters:
            origin:         The image coordinate position to use as the origin.
            world_origin:   The world coordinate position to use as the origin. [default: None]
            color:          For color-dependent WCS's, the color term to use in the connection
                            between the current origin and world_origin. [default: None]

        Returns:
            the new recentered WCS
        """
        if not isinstance(origin, Position):
            raise TypeError("origin must be a PositionD or PositionI argument")
        if not isinstance(world_origin, (Position, type(None))):
                raise TypeError("world_origin must be a PositionD or PositionI argument")
        return self._newOrigin(origin, world_origin)

    @property
    def _isLocal(self):
        return True

    # The origins are definitionally (0,0) for these.  So just define them here.
    @property
    def origin(self):
        """The image coordinate position to use as the origin.
        """
        return _PositionD(0,0)

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin.
        """
        return _PositionD(0,0)

    # For LocalWCS, there is no origin to worry about.
    def _posToWorld(self, image_pos, color):
        x = image_pos.x
        y = image_pos.y
        return _PositionD(self._u(x,y),self._v(x,y))

    def _xyTouv(self, x, y, color):
        return self._u(x,y), self._v(x,y)

    # For LocalWCS, there is no origin to worry about.
    def _posToImage(self, world_pos, color):
        u = world_pos.x
        v = world_pos.y
        return _PositionD(self._x(u,v),self._y(u,v))

    def _uvToxy(self, u, v, color):
        return self._x(u,v), self._y(u,v)

    # For LocalWCS, this is of course trivial.
    def _local(self, image_pos, color):
        return self


class CelestialWCS(BaseWCS):
    """A CelestialWCS is a `BaseWCS` whose world coordinates are on the celestial sphere.
    We use the `CelestialCoord` class for the world coordinates.
    """

    @property
    def _isCelestial(self):
        return True

    # CelestialWCS classes still have origin, but not world_origin.
    @property
    def x0(self):
        """The x coordinate of self.origin.
        """
        return self.origin.x

    @property
    def y0(self):
        """The y coordinate of self.origin.
        """
        return self.origin.y

    def xyToradec(self, x, y, units=None, color=None):
        """Convert x,y from image coordinates to world coordinates.

        This is equivalent to ``wcs.toWorld(x,y, units=units)``.

        It is also equivalent to ``wcs.posToWorld(galsim.PositionD(x,y)).rad`` when x and y are
        scalars if units is 'radians'; however, this routine allows x and y to be numpy arrays,
        in which case, the calculation will be vectorized, which is often much faster than using
        the pos interface.

        Parameters:
            x:          The x value(s) in image coordinates
            y:          The y value(s) in image coordinates
            units:      (Only valid for `CelestialWCS`, in which case it is required)
                        The units to use for the returned ra, dec values.
            color:      For color-dependent WCS's, the color term to use. [default: None]

        Returns:
            ra, dec
        """
        from .angle import AngleUnit
        if color is None: color = self._color
        if units is None:
            raise TypeError("units is required for CelestialWCS types")
        elif isinstance(units, str):
            units = AngleUnit.from_name(units)
        elif not isinstance(units, AngleUnit):
            raise GalSimValueError("units must be either an AngleUnit or a string", units,
                                    AngleUnit.valid_names)
        return self._xyToradec(x, y, units, color)

    def radecToxy(self, ra, dec, units, color=None):
        """Convert ra,dec from world coordinates to image coordinates.

        This is equivalent to ``wcs.toWorld(ra,dec, units=units)``.

        It is also equivalent to ``wcs.posToImage(galsim.CelestialCoord(ra * units, dec * units))``
        when ra and dec are scalars; however, this routine allows ra and dec to be numpy arrays,
        in which case, the calculation will be vectorized, which is often much faster than using
        the pos interface.

        Parameters:
            ra:         The ra value(s) in world coordinates
            dec:        The dec value(s) in world coordinates
            units:      The units to use for the input ra, dec values.
            color:      For color-dependent WCS's, the color term to use. [default: None]
        """
        from .angle import AngleUnit
        if color is None: color = self._color
        if isinstance(units, str):
            units = AngleUnit.from_name(units)
        elif not isinstance(units, AngleUnit):
            raise GalSimValueError("units must be either an AngleUnit or a string", units,
                                    AngleUnit.valid_names)
        return self._radecToxy(ra, dec, units, color)

    # This is a bit simpler than the EuclideanWCS version, since there is no world_origin.
    def _shiftOrigin(self, origin, world_origin, color):
        # We want the new wcs to have wcs.toWorld(x2,y2) match the current wcs.toWorld(0,0).
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

    # If the class doesn't define something else, then we can approximate the local Jacobian
    # from finite differences for the derivatives of ra and dec.  Very similar to the
    # version for EuclideanWCS, but convert from dra, ddec to du, dv locallat at the given
    # position.
    def _local(self, image_pos, color):
        from .angle import radians, arcsec

        if image_pos is None:
            raise TypeError("origin must be a PositionD or PositionI argument")

        x0 = image_pos.x - self.x0
        y0 = image_pos.y - self.y0
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        xlist = np.array([ x0, x0+dx, x0-dx, x0,    x0    ], dtype=float)
        ylist = np.array([ y0, y0,    y0,    y0+dy, y0-dy ], dtype=float)
        ra, dec = self._radec(xlist,ylist,color)

        # Note: our convention is that ra increases to the left!
        # i.e. The u,v plane is the tangent plane as seen from Earth with +v pointing
        # north, and +u pointing west.
        # That means the du values are the negative of dra.
        cosdec = np.cos(dec[0])
        dudx = -0.5 * (ra[1] - ra[2]) / dx * cosdec
        dudy = -0.5 * (ra[3] - ra[4]) / dy * cosdec
        dvdx = 0.5 * (dec[1] - dec[2]) / dx
        dvdy = 0.5 * (dec[3] - dec[4]) / dy

        # These values are all in radians.  Convert to arcsec as per our usual standard.
        factor = radians / arcsec
        return JacobianWCS(dudx*factor, dudy*factor, dvdx*factor, dvdy*factor)

    # This is similar to the version for EuclideanWCS, but uses dra, ddec.
    # Again, it is much faster if the _radec function works with numpy arrays.
    def _makeSkyImage(self, image, sky_level, color):
        from .angle import radians, arcsec
        b = image.bounds
        nx = b.xmax-b.xmin+1 + 2  # +2 more than in image to get row/col off each edge.
        ny = b.ymax-b.ymin+1 + 2
        x,y = np.meshgrid( np.linspace(b.xmin-1,b.xmax+1,nx),
                           np.linspace(b.ymin-1,b.ymax+1,ny) )
        x -= self.x0
        y -= self.y0
        ra, dec = self._radec(x.ravel(),y.ravel(),color)
        ra = np.reshape(ra, x.shape)
        dec = np.reshape(dec, x.shape)

        # Use the finite differences to estimate the derivatives.
        cosdec = np.cos(dec[1:ny-1,1:nx-1])
        dudx = -0.5 * (ra[1:ny-1,2:nx] - ra[1:ny-1,0:nx-2])
        dudy = -0.5 * (ra[2:ny,1:nx-1] - ra[0:ny-2,1:nx-1])
        # Check for discontinuities in ra.  ra can jump by 2pi, so when it does
        # add (or subtract) pi to dudx, which is dra/2
        dudx[dudx > 1] -= np.pi
        dudx[dudx < -1] += np.pi
        dudy[dudy > 1] -= np.pi
        dudy[dudy < -1] += np.pi
        # Now account for the cosdec factor
        dudx *= cosdec
        dudy *= cosdec
        dvdx = 0.5 * (dec[1:ny-1,2:nx] - dec[1:ny-1,0:nx-2])
        dvdy = 0.5 * (dec[2:ny,1:nx-1] - dec[0:ny-2,1:nx-1])

        area = np.abs(dudx * dvdy - dvdx * dudy)
        factor = radians / arcsec
        image.array[:,:] = area * sky_level * factor**2


    # Simple.  Just call _radec.
    def _posToWorld(self, image_pos, color, project_center=None, projection='gnomonic'):
        from .angle import radians, arcsec
        x = image_pos.x - self.x0
        y = image_pos.y - self.y0
        ra, dec = self._radec(x,y,color)
        coord = CelestialCoord(ra*radians, dec*radians)
        if project_center is None:
            return coord
        else:
            u,v = project_center.project(coord, projection=projection)
            return _PositionD(u/arcsec, v/arcsec)

    def _xyToradec(self, x, y, units, color):
        from .angle import radians
        x = x - self.x0  # Not -=, since don't want to modify the input arrays in place.
        y = y - self.y0
        ra, dec = self._radec(x,y,color)
        ra *= radians / units
        dec *= radians / units
        return ra, dec

    # Also simple if _xy is implemented.  However, it is allowed to raise a NotImplementedError.
    def _posToImage(self, world_pos, color):
        ra = world_pos.ra.rad
        dec = world_pos.dec.rad
        x, y = self._xy(ra,dec,color)
        return _PositionD(x,y) + self.origin

    def _radecToxy(self, ra, dec, units, color):
        from .angle import radians
        ra = ra * (units / radians)
        dec = dec * (units / radians)
        x, y = self._xy(ra,dec,color)
        x += self.origin.x
        y += self.origin.y
        return x, y

    # Each class should define the __eq__ function.  Then __ne__ is obvious.
    def __ne__(self, other): return not self.__eq__(other)



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

    A PixelScale is initialized with the command::

        >>> wcs = galsim.PixelScale(scale)

    Parameters:
        scale:  The pixel scale, typically in units of arcsec/pixel.
    """
    _req_params = { "scale" : float }

    def __init__(self, scale):
        self._color = None
        self._scale = float(scale)

    # Help make sure PixelScale is read-only.
    @property
    def scale(self):
        """The pixel scale
        """
        return self._scale

    @lazy_property
    def _invscale(self):
        return 1./self._scale

    @property
    def _isPixelScale(self):
        return True

    def _u(self, x, y, color=None):
        return x * self._scale

    def _v(self, x, y, color=None):
        return y * self._scale

    def _x(self, u, v, color=None):
        return u * self._invscale

    def _y(self, u, v, color=None):
        return v * self._invscale

    def _profileToWorld(self, image_profile, flux_ratio, offset):
        from .transform import _Transform, Transform
        # In the usual case of GSObject, it's more efficient to use the _Transform version.
        # else, it's a ChromaticObject, and we need to use the regular Transform function.
        Transform = _Transform if isinstance(image_profile, GSObject) else Transform
        if self._scale == 1.:
            j = None
        else:
            j = np.array(((self._scale, 0.), (0., self._scale)))
        return Transform(image_profile, j, flux_ratio=self._invscale**2 * flux_ratio,
                         offset=(offset.x, offset.y))

    def _profileToImage(self, world_profile, flux_ratio, offset):
        from .transform import _Transform, Transform
        Transform = _Transform if isinstance(world_profile, GSObject) else Transform
        if self._scale == 1.:
            j = None
        else:
            j = np.array(((self._invscale, 0.), (0., self._invscale)))
        return Transform(world_profile, j, flux_ratio=self._scale**2 * flux_ratio,
                         offset=(offset.x, offset.y))

    def _pixelArea(self):
        return self._scale**2

    def _minScale(self):
        return self._scale

    def _maxScale(self):
        return self._scale

    def _inverse(self):
        return PixelScale(self._invscale)

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
        return (self is other or
                (isinstance(other, PixelScale) and
                 self.scale == other.scale))

    def __repr__(self): return "galsim.PixelScale(%r)"%self.scale
    def __hash__(self): return hash(repr(self))


class ShearWCS(LocalWCS):
    """This WCS is a uniformly sheared coordinate system.

    The shear is given as the shape that a round object has when observed in image coordinates.

    The conversion functions in terms of (g1,g2) are therefore:

        x = (u + g1 u + g2 v) / scale / sqrt(1-g1**2-g2**2)
        y = (v - g1 v + g2 u) / scale / sqrt(1-g1**2-g2**2)

    or, writing this in the usual way of (u,v) as a function of (x,y):

        u = (x - g1 x - g2 y) * scale / sqrt(1-g1**2-g2**2)
        v = (y + g1 y - g2 x) * scale / sqrt(1-g1**2-g2**2)

    A ShearWCS is initialized with the command::

        >>> wcs = galsim.ShearWCS(scale, shear)

    Parameters:
        scale:      The pixel scale, typically in units of arcsec/pixel.
        shear:      The shear, which should be a `Shear` instance.

    The Shear transformation conserves object area, so if the input ``scale == 1`` then the
    transformation represented by the ShearWCS will conserve object area also.
    """
    _req_params = { "scale" : float, "shear" : Shear }

    def __init__(self, scale, shear):
        self._color = None
        self._scale = float(scale)
        self._shear = shear
        self._g1 = shear.g1
        self._g2 = shear.g2
        self._gsq = self._g1**2 + self._g2**2
        import math
        self._gfactor = 1. / math.sqrt(1. - self._gsq)

    # Help make sure ShearWCS is read-only.
    @property
    def scale(self):
        """The pixel scale.
        """
        return self._scale

    @property
    def shear(self):
        """The applied `Shear`.
        """
        return self._shear

    def _u(self, x, y, color=None):
        u = x * (1.-self._g1) - y * self._g2
        u *= self._gfactor * self._scale
        return u

    def _v(self, x, y, color=None):
        v = y * (1.+self._g1) - x * self._g2
        v *= self._gfactor * self._scale
        return v

    def _x(self, u, v, color=None):
        x = u * (1.+self._g1) + v * self._g2
        x *= self._gfactor / self._scale
        return x

    def _y(self, u, v, color=None):
        y = v * (1.-self._g1) + u * self._g2
        y *= self._gfactor / self._scale
        return y

    def _profileToWorld(self, image_profile, flux_ratio, offset):
        return image_profile.dilate(self._scale).shear(-self.shear).shift(offset) * flux_ratio

    def _profileToImage(self, world_profile, flux_ratio, offset):
        return world_profile.dilate(1./self._scale).shear(self.shear).shift(offset) * flux_ratio

    def _pixelArea(self):
        return self._scale**2

    def _minScale(self):
        # min stretch is (1-|g|) / sqrt(1-|g|^2)
        import math
        return self._scale * (1. - math.sqrt(self._gsq)) * self._gfactor

    def _maxScale(self):
        # max stretch is (1+|g|) / sqrt(1-|g|^2)
        import math
        return self._scale * (1. + math.sqrt(self._gsq)) * self._gfactor

    def _inverse(self):
        return ShearWCS(1./self._scale, -self._shear)

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
        return ShearWCS(scale, Shear(g1=g1, g2=g2))

    def _newOrigin(self, origin, world_origin):
        return OffsetShearWCS(self._scale, self._shear, origin, world_origin)

    def copy(self):
        return ShearWCS(self._scale, self._shear)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ShearWCS) and
                 self.scale == other.scale and
                 self.shear == other.shear))

    def __repr__(self): return "galsim.ShearWCS(%r, %r)"%(self.scale,self.shear)
    def __hash__(self): return hash(repr(self))


class JacobianWCS(LocalWCS):
    """This WCS is the most general local linear WCS implementing a 2x2 Jacobian matrix.

    The conversion functions are:

        u = dudx x + dudy y
        v = dvdx x + dvdy y

    A JacobianWCS has attributes dudx, dudy, dvdx, dvdy that you can access directly if that
    is convenient.  You can also access these as a NumPy array directly with::

        >>> J = jac_wcs.getMatrix()

    Also, JacobianWCS has another method that other WCS classes do not have. The call::

        >>> scale, shear, theta, flip = jac_wcs.getDecomposition()

    will return the equivalent expansion, shear, rotation and possible flip corresponding to
    this transformation.  See the docstring for that method for more information.

    A JacobianWCS is initialized with the command::

        >>> wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)

    Parameters:
        dudx:       du/dx
        dudy:       du/dy
        dvdx:       dv/dx
        dvdy:       dv/dy
    """
    _req_params = { "dudx" : float, "dudy" : float, "dvdx" : float, "dvdy" : float }

    def __init__(self, dudx, dudy, dvdx, dvdy):
        self._color = None
        self._dudx = float(dudx)
        self._dudy = float(dudy)
        self._dvdx = float(dvdx)
        self._dvdy = float(dvdy)
        self._det = dudx * dvdy - dudy * dvdx

    # Help make sure JacobianWCS is read-only.
    @property
    def dudx(self):
        """du/dx
        """
        return self._dudx

    @property
    def dudy(self):
        """du/dy
        """
        return self._dudy

    @property
    def dvdx(self):
        """dv/dx
        """
        return self._dvdx

    @property
    def dvdy(self):
        """dv/dy
        """
        return self._dvdy

    def _u(self, x, y, color=None):
        return self._dudx * x + self._dudy * y

    def _v(self, x, y, color=None):
        return self._dvdx * x + self._dvdy * y

    def _x(self, u, v, color=None):
        #  J = ( dudx  dudy )
        #      ( dvdx  dvdy )
        #  J^-1 = (1/det) (  dvdy  -dudy )
        #                 ( -dvdx   dudx )
        return (self._dvdy * u - self._dudy * v)*self._invdet

    def _y(self, u, v, color=None):
        return (-self._dvdx * u + self._dudx * v)*self._invdet

    def _profileToWorld(self, image_profile, flux_ratio, offset):
        from .transform import _Transform, Transform
        # In the usual case of GSObject, it's more efficient to use the _Transform version.
        # else, it's a ChromaticObject, and we need to use the regular Transform function.
        Transform = _Transform if isinstance(image_profile, GSObject) else Transform
        j = np.array(((self._dudx, self._dudy), (self._dvdx, self._dvdy)))
        return Transform(image_profile, j, flux_ratio=flux_ratio*abs(self._invdet),
                         offset=(offset.x, offset.y))

    def _profileToImage(self, world_profile, flux_ratio, offset):
        from .transform import _Transform, Transform
        Transform = _Transform if isinstance(world_profile, GSObject) else Transform
        j = np.array(((self._dvdy, -self._dudy), (-self._dvdx, self._dudx))) * self._invdet
        return Transform(world_profile, j, flux_ratio=flux_ratio*abs(self._det),
                         offset=(offset.x, offset.y))

    @lazy_property
    def _invdet(self):
        try:
            return 1./self._det
        except ZeroDivisionError:
            raise GalSimError("Transformation is singular")

    def _pixelArea(self):
        return abs(self._det)

    def getMatrix(self):
        """Get the Jacobian as a NumPy matrix:

                numpy.array( [[ dudx, dudy ],
                              [ dvdx, dvdy ]] )
        """
        return np.array([[ self._dudx, self._dudy ],
                         [ self._dvdx, self._dvdy ]], dtype=float)

    def getDecomposition(self):
        """Get the equivalent expansion, shear, rotation and possible flip corresponding to
        this Jacobian transformation.

        A non-singular real matrix can always be decomposed into a symmetric positive definite
        matrix times an orthogonal matrix:

            M = P Q

        In our case, P includes an overall scale and a shear, and Q is a rotation and possibly
        a flip of (x,y) -> (y,x).

            ( dudx  dudy ) = scale/sqrt(1-g1^2-g2^2) ( 1+g1  g2  ) ( cos(theta)  -sin(theta) ) F
            ( dvdx  dvdy )                           (  g2  1-g1 ) ( sin(theta)   cos(theta) )

        where F is either the identity matrix, ( 1 0 ), or a flip matrix, ( 0 1 ).
                                               ( 0 1 )                    ( 1 0 )

        If there is no flip, then this means that the effect of::

            >>> prof.transform(dudx, dudy, dvdx, dvdy)

        is equivalent to::

            >>> prof.rotate(theta).shear(shear).expand(scale)

        in that order.  (Rotation and shear do not commute.)

        The decomposition is returned as a tuple: (scale, shear, theta, flip), where scale is a
        float, shear is a `Shear`, theta is an `Angle`, and flip is a bool.
        """
        import math
        from .angle import radians
        # First we need to see whether or not the transformation includes a flip.  The evidence
        # for a flip is that the determinant is negative.
        if self._det == 0.:
            raise GalSimError("Transformation is singular")
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

        # A small bit of algebraic manipulations yield the following two equations that let us
        # determine theta:
        #
        # (dudx + dvdy) = 2 scale/sqrt(1-g^2) cos(t)
        # (dvdx - dudy) = 2 scale/sqrt(1-g^2) sin(t)

        C = dudx + dvdy
        S = dvdx - dudy
        theta = math.atan2(S,C) * radians

        # The next step uses the following equations that you can get from a bit more algebra:
        #
        # cost (dudx - dvdy) - sint (dudy + dvdx) = 2 scale/sqrt(1-g^2) g1
        # sint (dudx - dvdy) + cost (dudy + dvdx) = 2 scale/sqrt(1-g^2) g2

        factor = C*C+S*S    # factor = (2 scale/sqrt(1-g^2))^2
        C /= factor         # C is now cost / (2 scale/sqrt(1-g^2))
        S /= factor         # S is now sint / (2 scale/sqrt(1-g^2))

        g1 = C*(dudx-dvdy) - S*(dudy+dvdx)
        g2 = S*(dudx-dvdy) + C*(dudy+dvdx)

        return scale, Shear(g1=g1, g2=g2), theta, flip

    def _minScale(self):
        import math
        # min scale is scale * (1-|g|) / sqrt(1-|g|^2)
        # We could get this from the decomposition, but some algebra finds that this
        # reduces to the following calculation:
        # NB: The unit tests test for the equivalence with the above formula.
        h1 = math.sqrt( (self._dudx + self._dvdy)**2 + (self._dudy - self._dvdx)**2 )
        h2 = math.sqrt( (self._dudx - self._dvdy)**2 + (self._dudy + self._dvdx)**2 )
        return 0.5 * abs(h1 - h2)

    def _maxScale(self):
        import math
        # min scale is scale * (1+|g|) / sqrt(1-|g|^2)
        # which is equivalent to the following:
        # NB: The unit tests test for the equivalence with the above formula.
        h1 = math.sqrt( (self._dudx + self._dvdy)**2 + (self._dudy - self._dvdx)**2 )
        h2 = math.sqrt( (self._dudx - self._dvdy)**2 + (self._dudy + self._dvdx)**2 )
        return 0.5 * (h1 + h2)

    def _inverse(self):
        return JacobianWCS(self._dvdy*self._invdet, -self._dudy*self._invdet,
                           -self._dvdx*self._invdet, self._dudx*self._invdet)

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
        return (self is other or
                (isinstance(other, JacobianWCS) and
                 self.dudx == other.dudx and
                 self.dudy == other.dudy and
                 self.dvdx == other.dvdx and
                 self.dvdy == other.dvdy))

    def __repr__(self): return "galsim.JacobianWCS(%r, %r, %r, %r)"%(
            self.dudx,self.dudy,self.dvdx,self.dvdy)
    def __hash__(self): return hash(repr(self))


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
    """This WCS is similar to `PixelScale`, except the origin is not necessarily (0,0) in both
    the image and world coordinates.

    The conversion functions are:

        u = (x-x0) * scale + u0
        v = (y-y0) * scale + v0

    An OffsetWCS is initialized with the command::

        >>> wcs = galsim.OffsetWCS(scale, origin=None, world_origin=None)

    Parameters:
        scale:          The pixel scale, typically in units of arcsec/pixel.
        origin:         Optional origin position for the image coordinate system.
                        If provided, it should be a `PositionD` or `PositionI`.
                        [default: PositionD(0., 0.)]
        world_origin:   Optional origin position for the world coordinate system.
                        If provided, it should be a `PositionD`.
                        [default: galsim.PositionD(0., 0.)]
    """
    _req_params = { "scale" : float }
    _opt_params = { "origin" : PositionD, "world_origin": PositionD }

    def __init__(self, scale, origin=None, world_origin=None):
        self._color = None
        self._set_origin(origin, world_origin)
        self._scale = scale
        self._local_wcs = PixelScale(scale)

    @property
    def scale(self):
        """The pixel scale.
        """
        return self._scale

    @property
    def origin(self):
        """The image coordinate position to use as the origin.
        """
        return self._origin

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin.
        """
        return self._world_origin

    @property
    def _isPixelScale(self):
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
        return OffsetWCS(scale, _PositionD(x0,y0), _PositionD(u0,v0))

    def _newOrigin(self, origin, world_origin):
        return OffsetWCS(self._scale, origin, world_origin)

    def copy(self):
        return OffsetWCS(self._scale, self.origin, self.world_origin)

    def __repr__(self): return "galsim.OffsetWCS(%r, %r, %r)"%(
            self.scale, self.origin, self.world_origin)
    def __hash__(self): return hash(repr(self))


class OffsetShearWCS(UniformWCS):
    """This WCS is a uniformly sheared coordinate system with image and world origins
    that are not necessarily coincident.

    The conversion functions are:

        x = ( (1+g1) (u-u0) + g2 (v-v0) ) / scale / sqrt(1-g1**2-g2**2) + x0
        y = ( (1-g1) (v-v0) + g2 (u-u0) ) / scale / sqrt(1-g1**2-g2**2) + y0

        u = ( (1-g1) (x-x0) - g2 (y-y0) ) * scale / sqrt(1-g1**2-g2**2) + u0
        v = ( (1+g1) (y-y0) - g2 (x-x0) ) * scale / sqrt(1-g1**2-g2**2) + v0

    An OffsetShearWCS is initialized with the command::

        >>> wcs = galsim.OffsetShearWCS(scale, shear, origin=None, world_origin=None)

    Parameters:
        scale:          The pixel scale, typically in units of arcsec/pixel.
        shear:          The shear, which should be a `Shear` instance.
        origin:         Optional origin position for the image coordinate system.
                        If provided, it should be a `PositionD` or `PositionI`.
                        [default: PositionD(0., 0.)]
        world_origin:   Optional origin position for the world coordinate system.
                        If provided, it should be a `PositionD`.
                        [default: PositionD(0., 0.)]
    """
    _req_params = { "scale" : float, "shear" : Shear }
    _opt_params = { "origin" : PositionD, "world_origin": PositionD }

    def __init__(self, scale, shear, origin=None, world_origin=None):
        self._color = None
        self._set_origin(origin, world_origin)
        # The shear stuff is not too complicated, but enough so that it is worth
        # encapsulating in the ShearWCS class.  So here, we just create one of those
        # and we'll pass along any shear calculations to that.
        self._local_wcs = ShearWCS(scale, shear)

    @property
    def scale(self):
        """The pixel scale.
        """
        return self._local_wcs.scale

    @property
    def shear(self):
        """The applied `Shear`.
        """
        return self._local_wcs.shear

    @property
    def origin(self):
        """The image coordinate position to use as the origin.
        """
        return self._origin

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin.
        """
        return self._world_origin

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
        return OffsetShearWCS(scale, Shear(g1=g1, g2=g2), _PositionD(x0,y0), _PositionD(u0,v0))

    def _newOrigin(self, origin, world_origin):
        return OffsetShearWCS(self.scale, self.shear, origin, world_origin)

    def copy(self):
        return OffsetShearWCS(self.scale, self.shear, self.origin, self.world_origin)

    def __repr__(self):
        return "galsim.OffsetShearWCS(%r, %r, %r, %r)"%(
                self.scale, self.shear, self.origin, self.world_origin)
    def __hash__(self): return hash(repr(self))


class AffineTransform(UniformWCS):
    """This WCS is the most general linear transformation.  It involves a 2x2 Jacobian
    matrix and an offset.  You can provide the offset in terms of either the ``image_pos``
    (x0,y0) where (u,v) = (0,0), or the ``world_pos`` (u0,v0) where (x,y) = (0,0).
    Or, in fact, you may provide both, in which case the ``image_pos`` (x0,y0) corresponds
    to the ``world_pos`` (u0,v0).

    The conversion functions are:

        u = dudx (x-x0) + dudy (y-y0) + u0
        v = dvdx (x-x0) + dvdy (y-y0) + v0

    An AffineTransform has attributes dudx, dudy, dvdx, dvdy, x0, y0, u0, v0 that you can
    access directly if that is convenient.

    An AffineTransform is initialized with the command::

        >>> wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=None, world_origin=None)

    Parameters:
        dudx:           du/dx
        dudy:           du/dy
        dvdx:           dv/dx
        dvdy:           dv/dy
        origin:         Optional origin position for the image coordinate system.
                        If provided, it should be a `PositionD` or `PositionI`.
                        [default: PositionD(0., 0.)]
        world_origin:   Optional origin position for the world coordinate system.
                        If provided, it should be a `PositionD`.
                        [default: PositionD(0., 0.)]
    """
    _req_params = { "dudx" : float, "dudy" : float, "dvdx" : float, "dvdy" : float }
    _opt_params = { "origin" : PositionD, "world_origin": PositionD }

    def __init__(self, dudx, dudy, dvdx, dvdy, origin=None, world_origin=None):
        self._color = None
        self._set_origin(origin, world_origin)
        # As with OffsetShearWCS, we store a JacobianWCS, rather than reimplement everything.
        self._local_wcs = JacobianWCS(dudx, dudy, dvdx, dvdy)

    @property
    def dudx(self):
        """du/dx
        """
        return self._local_wcs.dudx

    @property
    def dudy(self):
        """du/dy
        """
        return self._local_wcs.dudy

    @property
    def dvdx(self):
        """dv/dx
        """
        return self._local_wcs.dvdx

    @property
    def dvdy(self):
        """dv/dy
        """
        return self._local_wcs.dvdy

    @property
    def origin(self):
        """The image coordinate position to use as the origin.
        """
        return self._origin

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin.
        """
        return self._world_origin

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
        else:
            dudx = header.get("CDELT1",1.)
            dudy = 0.
            dvdx = 0.
            dvdy = header.get("CDELT2",1.)
        x0 = header.get("CRPIX1",0.)
        y0 = header.get("CRPIX2",0.)
        u0 = header.get("CRVAL1",0.)
        v0 = header.get("CRVAL2",0.)

        return AffineTransform(dudx, dudy, dvdx, dvdy, _PositionD(x0,y0), _PositionD(u0,v0))

    def _newOrigin(self, origin, world_origin):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy,
                               origin, world_origin)

    def copy(self):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy,
                               self.origin, self.world_origin)

    def __repr__(self):
        return ("galsim.AffineTransform(%r, %r, %r, %r, origin=%r, world_origin=%r)")%(
                self.dudx, self.dudy, self.dvdx, self.dvdy, self.origin, self.world_origin)
    def __hash__(self): return hash(repr(self))


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
# Non-uniform EuclideanWCS classes must define the following:
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
# Ideally, the above functions would work with NumPy arrays as inputs.
#
#########################################################################################


# Some helper functions for serializing arbitrary functions.  Used by both UVFunction and
# RaDecFunction.
def _writeFuncToHeader(func, letter, header):
    if isinstance(func, str):
        # If we have the string version, then just write that
        s = func
        first_key = 'GS_'+letter+'_STR'

    elif func is not None:
        # Otherwise things get more interesting.  We have to serialize a python function.
        # I got the starting point for this code from:
        #     http://stackoverflow.com/questions/1253528/
        # In particular, marshal can serialize arbitrary code. (!)
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        import types, marshal, base64
        if type(func) == types.FunctionType:
            code = marshal.dumps(func.__code__)
            name = func.__name__
            defaults = func.__defaults__
            closure = func.__closure__

            # Functions may also have something called closure cells.  If there are any, we need
            # to include them as well.  Help for this part came from:
            # http://stackoverflow.com/questions/573569/
            if closure:
                from types import ModuleType
                closure_list = []
                for c in closure:
                    if isinstance(c.cell_contents, ModuleType):
                        # Can't really pickle the modules.  e.g. math if they use math functions.
                        # The modules just need to be loaded on the other side.  But we still need
                        # to make a cell for the module closure item, so just use its name and
                        # mark it as a module so we can recover it correctly.
                        closure_list.append( 'module_'+c.cell_contents.__name__ )
                    else:
                        closure_list.append( c.cell_contents )
            else:
                closure_list = None
            all = (0,code,name,defaults,closure_list)
        else:
            # For things other than regular functions, we can try to pickle it directly, but
            # it might not work.  Let pickle raise the appropriate error if it fails.

            # The first item in the tuple is what I'm calling a type_code to indicate what to
            # do with the results of unpickling.  So far I just have 0 = function, 1 = other,
            # but this could be extended if we find a good reason to.
            all = (1,func)

        # Now we can use pickle to serialize the full thing.
        s = pickle.dumps(all)

        # Fits can't handle arbitrary strings.  Shrink to a base-64 alphabet that is printable.
        # (This is like UUencoding for those of you who remember that...)
        s = base64.b64encode(s).decode()
        first_key = 'GS_'+letter+'_FN'
    else:
        # Nothing to write.
        return

    # Fits header strings cannot be more than 68 characters long, so split it up.
    fits_len = 68
    n = (len(s)-1)//fits_len + 1
    s_array = [ s[i*fits_len:(i+1)*fits_len] for i in range(n) ]

    # The total number of string splits is stored in fits key GS_U_N.
    header["GS_" + letter + "_N"] = n
    for i in range(n):
        # Use key names like GS_U0000, GS_U00001, etc. for the function versions
        # and like GS_SU000, GS_SU001, etc. for the string versions.
        if i == 0: key = first_key
        else: key = 'GS_%s%04d'%(letter,i)
        header[key] = s_array[i]

def _makecell(value):  # pragma: no cover
                       # (codecov gets confused, because the lambda function is never called.)
    # This is a little trick to make a closure cell.
    # We make a function that has the given value in closure, then then get the
    # first (only) closure item, which will be the closure cell we need.
    return (lambda : value).__closure__[0]

def _readFuncFromHeader(letter, header):
    # This undoes the process of _writeFuncToHeader.  See the comments in that code for details.
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    import types, marshal, base64
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
        all = pickle.loads(s)
        type_code = all[0]
        if type_code == 0:
            code_str, name, defaults, closure_items = all[1:]
            code = marshal.loads(code_str)
            if closure_items is None:
                closure = None
            else:
                closure = []
                for value in closure_items:
                    if isinstance(value,str) and value.startswith('module_'):
                        module_name = value[7:]
                        closure.append(_makecell(__import__(module_name)))
                    else:
                        closure.append(_makecell(value))
                closure = tuple(closure)
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

    Note: some internal calculations will be faster if the functions can take NumPy arrays
    for x,y and output arrays for u,v.  Usually this does not require any change to your
    function, but it is worth keeping in mind.  For example, if you want to do a sqrt, you
    may be better off using ``numpy.sqrt`` rather than ``math.sqrt``.

    A UVFunction is initialized with the command::

        >>> wcs = galsim.UVFunction(ufunc, vfunc, origin=None, world_origin=None)

    Parameters:
        ufunc:          The function u(x,y)
        vfunc:          The function v(x,y)
        xfunc:          The function x(u,v) (optional)
        yfunc:          The function y(u,v) (optional)
        origin:         Optional origin position for the image coordinate system.
                        If provided, it should be a `PositionD` or `PositionI`.
                        [default: PositionD(0., 0.)]
        world_origin    Optional origin position for the world coordinate system.
                        If provided, it should be a `PositionD`.
                        [default: PositionD(0., 0.)]
        uses_color:     If True, then the functions take three parameters (x,y,c) or (u,v,c)
                        where the third term is some kind of color value.  (The exact meaning
                        of "color" here is user-defined. You just need to be consistent with
                        the color values you use when using the wcs.) [default: False]
    """
    _req_params = { "ufunc" : str, "vfunc" : str }
    _opt_params = { "xfunc" : str, "yfunc" : str,
                    "origin" : PositionD, "world_origin": PositionD }

    def __init__(self, ufunc, vfunc, xfunc=None, yfunc=None, origin=None, world_origin=None,
                 uses_color=False):
        self._color = None
        self._set_origin(origin, world_origin)

        # Keep these to use in copies, etc.
        self._orig_ufunc = ufunc
        self._orig_vfunc = vfunc
        self._orig_xfunc = xfunc
        self._orig_yfunc = yfunc
        self._uses_color = uses_color

        # Turn these into the real functions
        self._initialize_funcs()

    def _initialize_funcs(self):
        global galsim  # Because if a user's function used galsim, it's probably at global scope.
        import galsim
        from . import utilities
        if isinstance(self._orig_ufunc, str):
            if self._uses_color:
                self._ufunc = utilities.math_eval('lambda x,y,c : ' + self._orig_ufunc)
            else:
                self._ufunc = utilities.math_eval('lambda x,y : ' + self._orig_ufunc)
        else:
            self._ufunc = self._orig_ufunc

        if isinstance(self._orig_vfunc, str):
            if self._uses_color:
                self._vfunc = utilities.math_eval('lambda x,y,c : ' + self._orig_vfunc)
            else:
                self._vfunc = utilities.math_eval('lambda x,y : ' + self._orig_vfunc)
        else:
            self._vfunc = self._orig_vfunc

        if isinstance(self._orig_xfunc, str):
            if self._uses_color:
                self._xfunc = utilities.math_eval('lambda u,v,c : ' + self._orig_xfunc)
            else:
                self._xfunc = utilities.math_eval('lambda u,v : ' + self._orig_xfunc)
        else:
            self._xfunc = self._orig_xfunc

        if isinstance(self._orig_yfunc, str):
            if self._uses_color:
                self._yfunc = utilities.math_eval('lambda u,v,c : ' + self._orig_yfunc)
            else:
                self._yfunc = utilities.math_eval('lambda u,v : ' + self._orig_yfunc)
        else:
            self._yfunc = self._orig_yfunc

    @property
    def ufunc(self):
        """The input ufunc
        """
        return self._ufunc

    @property
    def vfunc(self):
        """The input vfunc
        """
        return self._vfunc

    @property
    def xfunc(self):
        """The input xfunc
        """
        return self._xfunc

    @property
    def yfunc(self):
        """The input yfunc
        """
        return self._yfunc

    @property
    def origin(self):
        """The image coordinate position to use as the origin.
        """
        return self._origin

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin.
        """
        return self._world_origin

    def _u(self, x, y, color=None):
        if self._uses_color:
            try:
                return self._ufunc(x,y,color)
            except Exception as e:
                # If that didn't work, we have to do it manually for each position. :(  (SLOW!)
                try:
                    return np.array([self._ufunc(x1,y1,color) for [x1,y1] in zip(x,y)])
                except Exception:
                    raise e  # Raise the original if this fails, since it's probably more relevant.
        else:
            try:
                return self._ufunc(x,y)
            except Exception as e:
                try:
                    return np.array([self._ufunc(x1,y1) for [x1,y1] in zip(x,y)])
                except Exception:
                    raise e

    def _v(self, x, y, color=None):
        if self._uses_color:
            try:
                return self._vfunc(x,y,color)
            except Exception as e:
                try:
                    return np.array([self._vfunc(x1,y1,color) for [x1,y1] in zip(x,y)])
                except Exception:
                    raise e
        else:
            try:
                return self._vfunc(x,y)
            except Exception as e:
                try:
                    return np.array([self._vfunc(x1,y1) for [x1,y1] in zip(x,y)])
                except Exception:
                    raise e

    def _x(self, u, v, color=None):
        if self._xfunc is None:
            raise GalSimNotImplementedError(
                "World -> Image direction not implemented for this UVFunction")
        else:
            if self._uses_color:
                try:
                    return self._xfunc(u,v,color)
                except Exception as e:
                    try:
                        return np.array([self._xfunc(u1,v1,color) for [u1,v1] in zip(u,v)])
                    except Exception:
                        raise e
            else:
                try:
                    return self._xfunc(u,v)
                except Exception as e:
                    try:
                        return np.array([self._xfunc(u1,v1) for [u1,v1] in zip(u,v)])
                    except Exception:
                        raise e

    def _y(self, u, v, color=None):
        if self._yfunc is None:
            raise GalSimNotImplementedError(
                "World -> Image direction not implemented for this UVFunction")
        else:
            if self._uses_color:
                try:
                    return self._yfunc(u,v,color)
                except Exception as e:
                    try:
                        return np.array([self._yfunc(u1,v1,color) for [u1,v1] in zip(u,v)])
                    except Exception:
                        raise e
            else:
                try:
                    return self._yfunc(u,v)
                except Exception as e:
                    try:
                        return np.array([self._yfunc(u1,v1) for [u1,v1] in zip(u,v)])
                    except Exception:
                        raise e

    def _newOrigin(self, origin, world_origin):
        return UVFunction(self._orig_ufunc, self._orig_vfunc, self._orig_xfunc, self._orig_yfunc,
                          origin, world_origin, self._uses_color)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("UVFunction", "GalSim WCS name")
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")
        header["GS_U0"] = (self.world_origin.x, "GalSim world origin u")
        header["GS_V0"] = (self.world_origin.y, "GalSim world origin v")
        header["GS_COLOR"] = (int(self._uses_color), "GalSim wcs uses color?")

        _writeFuncToHeader(self._orig_ufunc, 'U', header)
        _writeFuncToHeader(self._orig_vfunc, 'V', header)
        _writeFuncToHeader(self._orig_xfunc, 'X', header)
        _writeFuncToHeader(self._orig_yfunc, 'Y', header)

        return self.affine(bounds.true_center)._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        u0 = header["GS_U0"]
        v0 = header["GS_V0"]
        uses_color = bool(header["GS_COLOR"])
        ufunc = _readFuncFromHeader('U', header)
        vfunc = _readFuncFromHeader('V', header)
        xfunc = _readFuncFromHeader('X', header)
        yfunc = _readFuncFromHeader('Y', header)
        return UVFunction(ufunc, vfunc, xfunc, yfunc, _PositionD(x0,y0),
                          _PositionD(u0,v0), uses_color)

    def copy(self):
        return UVFunction(self._orig_ufunc, self._orig_vfunc, self._orig_xfunc, self._orig_yfunc,
                          self.origin, self.world_origin, self._uses_color)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, UVFunction) and
                 self._orig_ufunc == other._orig_ufunc and
                 self._orig_vfunc == other._orig_vfunc and
                 self._orig_xfunc == other._orig_xfunc and
                 self._orig_yfunc == other._orig_yfunc and
                 self.origin == other.origin and
                 self.world_origin == other.world_origin and
                 self._uses_color == other._uses_color))

    def __repr__(self):
        return ("galsim.UVFunction(%r, %r, %r, %r, %r, %r, %r)")%(
                self._orig_ufunc, self._orig_vfunc, self._orig_xfunc, self._orig_yfunc,
                self.origin, self.world_origin, self._uses_color)

    def __hash__(self): return hash(repr(self))

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_ufunc']
        del d['_vfunc']
        del d['_xfunc']
        del d['_yfunc']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._initialize_funcs()


class RaDecFunction(CelestialWCS):
    """This WCS takes an arbitrary function for the Right Ascension (ra) and Declination (dec).

    In many cases, it can be more convenient to calculate both ra and dec in a single function,
    since there will typically be intermediate values that are common to both, so it may be more
    efficient to just calculate those once and thence calculate both ra and dec.  Thus, we
    provide the option to provide either a single function or two separate functions.

    The function parameters used to initialize an RaDecFunction may be:
        - a python functions that take (x,y) arguments
        - a python object with a __call__ method that takes (x,y) arguments
        - a string which can be parsed with eval('lambda x,y: '+str)

    The return values, ra and dec, should be given in _radians_.

    The first argument is called ``ra_func``, but if ``dec_func`` is omitted, then it is assumed
    to calculate both ra and dec.  The two values should be returned as a tuple (ra,dec).

    We don't want a function that returns `Angle` instances, because we want to allow for the
    possibility of using NumPy arrays as inputs and outputs to speed up some calculations.  The
    function isn't _required_ to work with NumPy arrays, but it is possible that some things
    will be faster if it does.  If it were expected to return `Angle` instances, then it definitely
    couldn't work with arrays.

    An RaDecFunction is initialized with either of the following commands::

        >>> wcs = galsim.RaDecFunction(radec_func, origin=None)
        >>> wcs = galsim.RaDecFunction(ra_func, dec_func, origin=None)

    Parameters:
        ra_func:    If ``dec_func`` is also given: A function ra(x,y) returning ra in radians.
                    If ``dec_func=None``: A function returning a tuple (ra,dec), both in radians.
        dec_func:   Either a function dec(x,y) returning dec in radians, or None (in which
                    case ``ra_func`` is expected to return both ra and dec. [default: None]
        origin:     Optional origin position for the image coordinate system.
                    If provided, it should be a `PositionD` or `PositionI`.
                    [default: PositionD(0., 0.)]
    """
    _req_params = { "ra_func" : str, "dec_func" : str }
    _opt_params = { "origin" : PositionD }

    def __init__(self, ra_func, dec_func=None, origin=None):
        self._color = None
        self._set_origin(origin)

        # Keep these to use in copies, etc.
        self._orig_ra_func = ra_func
        self._orig_dec_func = dec_func

        # Turn these into the real functions
        self._initialize_funcs()

    def _initialize_funcs(self):
        global galsim  # Because if a user's function used galsim, it's probably at global scope.
        import galsim
        from . import utilities
        if self._orig_dec_func is None:
            if isinstance(self._orig_ra_func, str):
                self._radec_func = utilities.math_eval('lambda x,y : ' + self._orig_ra_func)
            else:
                self._radec_func = self._orig_ra_func
        else:
            if isinstance(self._orig_ra_func, str):
                ra_func = utilities.math_eval('lambda x,y : ' + self._orig_ra_func)
            else:
                ra_func = self._orig_ra_func
            if isinstance(self._orig_dec_func, str):
                dec_func = utilities.math_eval('lambda x,y : ' + self._orig_dec_func)
            else:
                dec_func = self._orig_dec_func
            self._radec_func = lambda x,y : (ra_func(x,y), dec_func(x,y))

    @property
    def radec_func(self):
        """The input radec_func
        """
        return self._radec_func

    @property
    def origin(self):
        """The image coordinate position to use as the origin.
        """
        return self._origin

    def _radec(self, x, y, color=None):
        try:
            return self._radec_func(x,y)
        except Exception as e:
            try:
                world = [ self._radec(x1,y1) for (x1,y1) in zip(x,y) ]
            except Exception:
                raise e  # Raise the original one if this fails, since it's probably more relevant.
            ra = np.array([ w[0] for w in world ])
            dec = np.array([ w[1] for w in world ])
            return ra, dec

    def _xy(self, ra, dec, color=None):
        raise GalSimNotImplementedError(
                "World -> Image direction not implemented for RaDecFunction")

    def _newOrigin(self, origin):
        return RaDecFunction(self._orig_ra_func, self._orig_dec_func, origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"]  = ("RaDecFunction", "GalSim WCS name")
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")

        _writeFuncToHeader(self._orig_ra_func, 'R', header)
        _writeFuncToHeader(self._orig_dec_func, 'D', header)

        return self.affine(bounds.true_center)._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        ra_func = _readFuncFromHeader('R', header)
        dec_func = _readFuncFromHeader('D', header)
        return RaDecFunction(ra_func, dec_func, _PositionD(x0,y0))

    def copy(self):
        return RaDecFunction(self._orig_ra_func, self._orig_dec_func, self.origin)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, RaDecFunction) and
                 self._orig_ra_func == other._orig_ra_func and
                 self._orig_dec_func == other._orig_dec_func and
                 self.origin == other.origin))

    def __repr__(self):
        return "galsim.RaDecFunction(%r, %r, %r)"%(
                self._orig_ra_func, self._orig_dec_func, self.origin)

    def __hash__(self): return hash(repr(self))

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_radec_func']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._initialize_funcs()

def compatible(wcs1, wcs2):
    """
    A utility to check the compatibility of two WCS.  In particular, if two WCS are consistent with
    each other modulo a shifted origin, we consider them to be compatible, even though they are not
    equal.
    """
    if wcs1._isUniform and wcs2._isUniform:
        return wcs1.jacobian() == wcs2.jacobian()
    else:
        return wcs1 == wcs2.shiftOrigin(wcs1.origin, wcs1.world_origin)
