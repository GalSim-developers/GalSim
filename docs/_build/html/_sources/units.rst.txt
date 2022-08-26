
Units
=====

Size Units
----------

GalSim models surface brightnesses of objects in sky coordinates.  The physical size of a
galaxy in light years or kpc is not really relevant to its appearance in the sky.  Rather,
the size is an angle, being the angle subtended by the galaxy as seen from Earth.

The most common choice of unit for these objects is arcseconds.  This happens to be the
right order of magnitude for most objects of astronomical interest.  The smallest objects
we typically observe are somewhat less than an arcsecond in size.  The largest objects
(e.g. M31) are more than a degree, but by far the majority of the objects of interest are
only a few arcseconds across.

So this is usually the units one would use for the various `GSObject` size parameters
(e.g. ``fwhm``, ``half_light_radius`` and so forth) when you are building them.  However,
this choice is actually up to the user.  You may choose to define all your sizes in
degrees or radians if you want.  You just need to be consistent with the unit you use
for the sizes of all your objects and with the units of the pixel scale when you are
building your `Image` (via the ``scale`` parameter of the `Image` constructor) or when
drawing (the ``scale`` argument to `GSObject.drawImage`)

.. note::
    If you are using a more complicated WCS than a simple `PixelScale` (i.e. using ``wcs`` when
    building the `Image` rather than ``scale``), then you need to be even more careful about this.
    Some of the FITS-based WCS classes assume you are using arcseconds for the distance unit,
    and it is not always easy to coerce them into using a different unit.  In these cases,
    you are probably well-advised to just stick with arcseconds for your sizes.

Some classes specify their angular sizes somewhat indirectly.  For the `Airy` class, for instance,
you can specify ``lam`` as the wavelength of the light (:math:`\lambda`, say
at the middle of the bandpass) in nm, and ``diam``, the telescope diameter (:math:`D`) in meters.
The ratio of these :math:`\lambda / D` (after putting them into the same units) gives the
fundamental scale radius for an `Airy` profile.  But this angle is in radians, which is normally
not particularly convenient to use for the image pixel scale.  The `Airy` constructor thus takes
another parameter, ``scale_unit``, which defaults to arcsec to specify the unit you want to
convert this angle to.

Flux Units
----------

The units for the ``flux`` of a `GSObject` are nominally photons/cm^2/s, and the units for an
image are ADUs (analog-to-digital units).  There are thus four conversion factors to apply to
go from one to the other.

1. The exposure time (s)
2. The effective collecting area of the telescope (cm^2)
3. The quantum efficiency (QE) of the collector (e-/photon)
4. The gain of the read-out amplifier (e-/ADU)

In GalSim, we generally just lump the QE in with the gain, so our gain is taken to have units of
photons/ADU, and it really represents gain / QE.

Note, however, that the default exposure time, telescope collecting area, and gain are 1 s, 1 cm^2,
and 1 ADU/photon respectively, so users who wish to ignore the intricacies of managing exposure
times, collecting areas, and gains can simply think of the flux of a `GSObject` in either ADUs or
photons.

However, if you prefer to think of your flux as having physical units, then you can declare
the appropriate telescope collecting area (``area``), the exposure time (``exptime``), and the
total effective gain (``gain``) as arguments to `GSObject.drawImage`.

SED Units
---------

These details matter more when working with `ChromaticObject` instances, where the flux
normalization is handled with an `SED` object.  The units of an input `SED` can be any of
several possible options:

1. erg/nm/cm^2/s or erg/A/cm^2/s (use ``flux_type='flambda'``)
2. erg/Hz/cm^2/s (use ``flux_typ='fnu'``)
3. photons/nm/cm^2/s or photons/A/cm^2/s (use ``flux_typ='fphotons'``)
4. Any units that qualify as an ``astropy.units.spectral_density`` using the AstroPy ``units``
   module
5. dimensionless (use ``flux_typ='1'``)

.. note::
    The last one is a bit different from the others.  It is generally only appropriate for the
    "SED" of a PSF, not that of a galaxy or star.  The PSF may have a different effect as a
    function of wavelength, in which case that can be treated similarly to how we treat an SED.
    In any object that is a convolution of several components, only one of them should have a
    spectral SED.  The rest should be dimensionless (possibly flat).  The net SED of the
    composite object will then also be spectral.

Internally, all spectral units are converted to photons/nm/cm^2/s.  Then when drawing a
`ChromaticObject`, spectrum is integrated over the `Bandpass` to obtain the normal units of
photons/cm^2/s.  If you trust your SED, you can then just draw with the appropriate ``area``,
``exptime`` and ``gain`` when you call `ChromaticObject.drawImage`.

However, it is often more convenient to target a particular flux or magnitude of your object
as observed through a particular `Bandpass` (probably in ADU) and then ignore all of these
parameters when you are drawing.  This is possible using the methods `SED.withFlux` or
`SED.withMagnitude`.

Angles
------

For nearly all angular values, we require the argument to be an `Angle` instance.
We use the ``LSSTDESC.Coord`` package for this (and its `CelestialCoord` class):

https://github.com/LSSTDESC/Coord

An earlier version of this code was originally implemented in GalSim, so we
still import the relevant classes into the ``galsim`` namespace, so for example
``gasim.Angle`` is a valid alias for ``coord.Angle``.  You may therefor use either namespace
for your use of these classes.

.. autoclass:: galsim.Angle
    :members:

.. autofunction:: galsim._Angle

.. autoclass:: galsim.AngleUnit
    :members:
