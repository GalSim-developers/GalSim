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
       Furthermore, the world coordinates may be either a regular Euclidean coordinate
       system (using galsim.PositionD objects for the world positions) or coordinates on
       the celestial sphere (using galsim.CelestialCoord objects for the world positions).

       Currently we define the following non-local WCS classes:

            OffsetWCS
            OffsetShearWCS
            AffineTransform
            UVFunction
            RaDecFunction
            AstropyWCS          -- requires astropy.wcs python module to be installed
            PyAstWCS            -- requires starlink.Ast python module to be installed
            WcsToolsWCS         -- requires wcstools command line functions to be installed

    Some things you can do with a WCS object:

    - Convert positions between image coordinates and world coordinates (sometimes referred
      to as sky coordinates):

                world_pos = wcs.toWorld(image_pos)
                image_pos = wcs.toImage(world_pos)

      Note: the transformation from world to image coordinates is not guaranteed to be
      implemented.  If it is not implemented for a particular WCS class, a NotImplementedError
      will be raised.

    - Convert a GSObject, which is naturally defined in world coordinates, to the equivalent
      profile using image coordinates (or vice versa):

                image_profile = wcs.toImage(world_profile)
                world_profile = wcs.toWorld(image_profile)

    - Construct a local linear approximation of a WCS at a given location:

                local_wcs = wcs.local(image_pos = image_pos)
                local_wcs = wcs.local(world_pos = world_pos)

      If wcs.toWorld(image_pos) is not implemented for a particular WCS class, then a
      NotImplementedError will be raised if you pass in a world_pos argument.

    - Construct a non-local WCS using a given image position as the origin of the world
      coordinate system.

                world_pos1 = wcs.toWorld(PositionD(0,0))
                shifted = wcs.atOrigin(image_origin)
                world_pos2 = shifted.toWorld(image_origin)
                # world_pos1 should be equal to world_pos2

    - Get some properties of the pixel size and shape:

                area = local_wcs.pixelArea()
                min_linear_scale = local_wcs.minLinearScale()
                max_linear_scale = local_wcs.maxLinearScale()
                jac = local_wcs.jacobian()
                # Use jac.dudx, jac.dudy, jac.dvdx, jac.dvdy

      Global WCS objects also have these functions, but for them, you must supply either
      image_pos or world_pos.  So the following are equivalent:

                area = wcs.pixelArea(image_pos)
                area = wcs.local(image_pos).pixelArea()

    - Query some overall attributes of the WCS transformation:

                wcs.isLocal()       # is this a local WCS?
                wcs.isUniform()     # does this WCS have a uniform pixel size/shape?
                wcs.isCelestial()   # are the world coordinates on the celestial sphere?
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
        #     _atOrigin         function returning a version with a new origin (or origins).
        #     copy              return a copy
        #     __eq__            check if this equals another WCS object
        #     __ne__            check if this is not equal to another WCS object
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
            return self.local(**kwargs)._profileToWorld(arg)
        else:
            if isinstance(arg, galsim.PositionI):
                arg = galsim.PositionD(arg.x, arg.y)
            elif not isinstance(arg, galsim.PositionD):
                raise TypeError("toWorld requires a PositionD or PositionI argument")
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
            if self._is_celestial and not isinstance(arg, galsim.CelestialCoord):
                raise TypeError("toImage requires a CelestialCoord argument")
            elif not self._is_celestial and isinstance(arg, galsim.PositionI):
                arg = galsim.PositionD(arg.x, arg.y)
            elif not self._is_celestial and not isinstance(arg, galsim.PositionD):
                raise TypeError("toImage requires a PositionD or PositionI argument")
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

        @param image_pos    The image coordinate position (for variable WCS objects)
        @param world_pos    The world coordinate position (for variable WCS objects)
        @returns local_wcs  A WCS object with wcs.isLocal() == True
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

        @param image_pos    The image coordinate position (for variable WCS objects)
        @param world_pos    The world coordinate position (for variable WCS objects)
        @returns local_wcs  A JacobianWCS object
        """
        return self.local(image_pos, world_pos)._toJacobian()

    def atOrigin(self, image_origin, world_origin=None):
        """Recenter the current WCS function at a new origin location, returning the new WCS.

        This function creates a new WCS object (always to non-local WCS) that treats
        the image_origin position the same way the current WCS treats (x,y) = (0,0).

        If the current WCS is a local WCS, this essentially declares where on the image
        you want the origin of the world coordinate system to be.  i.e. where is (u,v) = (0,0).
        So, for example, to set a WCS that has a constant pixel size with the world coordinates
        centered at the center of an image, you could write:

                wcs = galsim.PixelScale(scale).atOrigin(im.center())

        This is equivalent to the following:

                wcs = galsim.OffsetWCS(scale, image_origin=im.center())

        For more non-local WCS types, the image_origin defines what image_pos should mean the same
        thing as (0,0) does in the current WCS.  The following example should work regardless
        of what kind of WCS this is:

                world_pos1 = wcs.toWorld(PositionD(0,0))
                wcs2 = wcs.atOrigin(new_image_origin)
                world_pos2 = wcs2.toWorld(new_image_origin)
                # world_pos1 should be equal to world_pos2

        Furthermore, if the current WCS uses Euclidean world coordinates (isCelestial() == False)
        you may also provide a world_origin argument which defines what (u,v) position you want
        to correspond to the new image_origin.  Continuing the previous example:

                wcs3 = wcs.atOrigin(new_image_origin, new_world_origin)
                world_pos3 = wcs3.toWorld(new_image_origin)
                # world_pos3 should be equal to new_world_origin

        @param image_origin  The image coordinate position to use as the origin.
                             [ Default `image_origin=None` ]
        @param world_origin  The world coordinate position to use as the origin.  Only valid if
                             wcs.isUniform() == True.  [ Default `world_origin=None` ]
        @returns wcs         The new recentered WCS object
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
        # The _atOrigin call is expecting new values for the (x0,y0) and (u0,v0), so
        # we need to figure out how to modify the parameters give the current values.
        #
        #     Use (x1,y1) and (u1,v1) for the new values that we will pass to _atOrigin.
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
            return self._atOrigin(image_origin, world_origin)
        else:
            if world_origin is not None:
                raise TypeError("world_origin is invalid for non-uniform WCS classes")
            return self._atOrigin(image_origin)


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
#     __eq__            check if this equals another WCS object
#     __ne__            check if this is not equal to another WCS object
#     __repr__          convert to string
#     _profileToWorld   function converting image_profile to world_profile
#     _profileToImage   function converting world_profile to image_profile
#     _pixelArea        function returning the pixel area
#     _minScale         function returning the minimum linear pixel scale
#     _maxScale         function returning the maximum linear pixel scale
#     _toJacobian       function returning an equivalent JacobianWCS
#     _atOrigin         function returning a non-local WCS corresponding to this WCS
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
        self._is_uniform = True
        self._is_celestial = False
        self._scale = scale

    # Help make sure PixelScale is read-only.
    @property
    def scale(self): return self._scale

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

    def _atOrigin(self, image_origin, world_origin):
        return OffsetWCS(self._scale, image_origin, world_origin)

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

    def _atOrigin(self, image_origin, world_origin):
        return OffsetShearWCS(self._scale, self._shear, image_origin, world_origin)

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

    Initialization
    --------------
    A JacobianWCS object is initialized with the command:

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

    def _atOrigin(self, image_origin, world_origin):
        return AffineTransform(self._dudx, self._dudy, self._dvdx, self._dvdy, image_origin,
                               world_origin)

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
#     __eq__            check if this equals another WCS object
#     __ne__            check if this is not equal to another WCS object
#     _local            function returning a local WCS at a given location
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
    An OffsetWCS object is initialized with the command:

        wcs = galsim.OffsetWCS(scale, image_origin=None, world_origin=None)

    @param scale          The pixel scale, typically in units of arcsec/pixel.
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI object.
                          [ Default: `image_origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD object.
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

    def _atOrigin(self, image_origin, world_origin):
        return OffsetWCS(self._scale, image_origin, world_origin)

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
    An OffsetShearWCS object is initialized with the command:

        wcs = galsim.OffsetShearWCS(scale, shear, image_origin=None, world_origin=None)

    @param scale          The pixel scale, typically in units of arcsec/pixel.
    @param shear          The shear, which should be a galsim.Shear instance.
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI object.
                          [ Default: `image_origin = None` ]
    @param world_origin   Optional origin position for the world coordinate system.
                          If provided, it should be a PostionD object.
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

    def _atOrigin(self, image_origin, world_origin):
        return OffsetShearWCS(self.scale, self.shear, image_origin, world_origin)

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
    def image_origin(self): return self._image_origin
    @property
    def world_origin(self): return self._world_origin

    def _posToWorld(self, image_pos):
        return self._jacwcs._posToWorld(image_pos - self._image_origin) + self._world_origin

    def _posToImage(self, world_pos):
        return self._jacwcs._posToImage(world_pos - self._world_origin) + self._image_origin

    def _local(self, image_pos, world_pos):
        return self._jacwcs

    def _atOrigin(self, image_origin, world_origin):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy,
                               image_origin, world_origin)

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
    @param vfunc          The function v(x,y)
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

    def _atOrigin(self, image_origin, world_origin):
        return UVFunction(self._ufunc, self._vfunc, image_origin, world_origin)

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
       into an JacobianWCS object.  The input ra, dec values should be in degrees.
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
    The functions should return galsim.Angle objects.

    Initialization
    --------------
    A RaDecFunction object is initialized with the command:

        wcs = galsim.RaDecFunction(rafunc, decfunc, image_origin=None)

    @param rafunc         The function ra(x,y)
    @param decfunc        The function dec(x,y)
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI object.
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

    def _atOrigin(self, image_origin):
        return RaDecFunction(self._rafunc, self._decfunc, image_origin)

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
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        import astropy.wcs
        self._tag = None # Write something useful here.
        if file_name is not None:
            self._tag = file_name
            from galsim import pyfits
            if ( isinstance(hdu, pyfits.CompImageHDU) or
                 isinstance(hdu, pyfits.ImageHDU) or
                 isinstance(hdu, pyfits.PrimaryHDU) ):
                raise TypeError("Cannot provide both file_name and pyfits hdu")
            if wcs is not None:
                raise TypeError("Cannot provide both file_name and wcs")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)

        if hdu is not None:
            if self._tag is None: self._tag = str(hdu)
            if wcs is not None:
                raise TypeError("Cannot provide both pyfits hdu and wcs")
            self._fix_header(hdu.header)
            import warnings
            with warnings.catch_warnings():
                # The constructor might emit warnings if it wants to fix the header
                # information (e.g. RADECSYS -> RADESYSa).  We'd rather ignore these
                # warnings, since we don't much care if the input file is non-standard
                # so long as we can make it work.
                warnings.simplefilter("ignore")
                wcs = astropy.wcs.WCS(hdu.header)
        if wcs is None:
            raise TypeError("Must provide one of file_name, hdu (as a pyfits HDU), or wcs")
        else:
            if self._tag is None: self._tag = str(wcs)
        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

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

    def _atOrigin(self, image_origin):
        return AstropyWCS(wcs=self._wcs, image_origin=image_origin)

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
    information from a FITS file.  It requires the starlinkAst python module to be installed.

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
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        import starlink.Ast
        import starlink.Atl
        # Note: More much of this class implementation, I've followed the example provided here:
        #    http://dsberry.github.io/starlink/node4.html
        self._tag = None # Write something useful here.
        if file_name is not None:
            self._tag = file_name
            from galsim import pyfits
            if ( isinstance(hdu, pyfits.CompImageHDU) or
                 isinstance(hdu, pyfits.ImageHDU) or
                 isinstance(hdu, pyfits.PrimaryHDU) ):
                raise TypeError("Cannot provide both file_name and pyfits hdu")
            if wcsinfo is not None:
                raise TypeError("Cannot provide both file_name and wcsinfo")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)

        if hdu is not None:
            if self._tag is None: self._tag = str(hdu)
            if wcsinfo is not None:
                raise TypeError("Cannot provide both pyfits hdu and wcsinfo")
            self._fix_header(hdu.header)
            fitschan = starlink.Ast.FitsChan( starlink.Atl.PyFITSAdapter(hdu) )
            wcsinfo = fitschan.read()
        if wcsinfo is None:
            if self._tag is None: self._tag = str(hdu)
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
        ra, dec = self._wcsinfo.tran( [ [x0, x0+dx, x0-dx, x0,    x0],
                                        [y0, y0,    y0,    y0+dy, y0-dy ] ])

        # Convert to degrees as needed by makeJacFromNumericalRaDec:
        ra = [ r * galsim.radians / galsim.degrees for r in ra ]
        dec = [ d * galsim.radians / galsim.degrees for d in dec ]
        return makeJacFromNumericalRaDec(ra, dec, dx, dy)

    def _atOrigin(self, image_origin):
        return PyAstWCS(wcsinfo=self._wcsinfo, image_origin=image_origin)

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

    Initialization
    --------------
    A WcsToolsWCS object is initialized with the following command:

        wcs = galsim.WcsToolsWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param image_origin   Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI object.
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
        if image_pos is None:
            raise TypeError('WcsToolsWCS.local() requires an image_pos or world_pos argument')

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

    def _atOrigin(self, image_origin):
        return WcsToolsWCS(self._file_name, image_origin=image_origin)

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

