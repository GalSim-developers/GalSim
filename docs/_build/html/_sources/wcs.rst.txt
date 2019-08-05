
World Coordinate Systems
########################

The World Coordinate System (or WCS) is the traditional term for the mapping from pixel coordinates
to the coordinate system on the sky.
(I know, the world's down here, and the sky's up there, so you'd think it would
be reversed, but that's the way it goes.  Astronomy is full of terms that don't quite make sense
when you look at them too closely.)

There are two kinds of world coordinates that we use here:

- Celestial coordinates are defined in terms of right ascension (RA) and declination (Dec).
  They are a spherical coordinate system on the sky, akin to longitude and latitude on Earth.
  cf. http://en.wikipedia.org/wiki/Celestial_coordinate_system

- Euclidean coordinates are defined relative to a tangent plane projection of the sky.
  If you imagine the sky coordinates on an actual sphere with a particular radius, then the
  tangent plane is tangent to that sphere.  We use the labels (u,v) for the coordinates in
  this system, where +v points north and +u points west.  (Yes, west, not east.  As you look
  up into the sky, if north is up, then west is to the right.)

The classes in this file provide a mapping from image coordinates (in pixels) to one of these
two kinds of world coordinates.  We use the labels ``(x,y)`` for the image coordinates.

WCS Base Classes
================

.. autoclass:: galsim.BaseWCS
    :members:
    :special-members:

.. autoclass:: galsim.wcs.CelestialWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.wcs.EuclideanWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.wcs.UniformWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.wcs.LocalWCS
    :members:
    :special-members:
    :show-inheritance:

Euclidean WCS's
===============

.. autoclass:: galsim.PixelScale
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.OffsetWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.ShearWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.OffsetShearWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.JacobianWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.AffineTransform
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.UVFunction
    :members:
    :special-members:
    :show-inheritance:


Celestial WCS's
===============

.. autoclass:: galsim.RaDecFunction
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.AstropyWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.PyAstWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.WcsToolsWCS
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.GSFitsWCS
    :members:
    :special-members:
    :show-inheritance:

.. autofunction:: galsim.FitsWCS

.. autofunction:: galsim.TanWCS


WCS Utilities
=============

.. autofunction:: galsim.wcs.readFromFitsHeader

.. autofunction:: galsim.wcs.compatible
