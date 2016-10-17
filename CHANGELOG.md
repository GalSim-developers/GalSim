Changes from v1.4 to v1.5
=========================

API Changes
-----------

- `drawImage()` now accepts `exptime` and `area` keywords to indicate the image
  exposure time and telescope collecting area. (#789)
- dimensions of `SED` changed from [photons/wavelength-interval] to either
  [photons/wavelength-interval/area/time] or [1] (dimensionless).  
  `ChromaticObject`s representing stars or galaxies take SEDs with the former
  dimensions, those representing a chromatic PSF take SEDs with the latter
  dimensions. (#789)
- Added restrictions to `ChromaticObject`s and `SED`s consistent with
  dimensional analysis.  E.g., only `ChromaticObject`s with dimensionful SEDs
  can be drawn. (#789)
- Simplified the return value of galsim.config.ReadConfig. (#580)


Dependency Changes
------------------
- `astropy` is now required for chromatic functionality. (#789)


Bug Fixes
---------

- Added checks to `SED`s and `ChromaticObject`s for dimensional sanity. (#789)
- Fixed an error in the magnification calculated by NFWHalo.getLensing(). (#580)
- Fixed bug when whitening noise in images based on COSMOS training datasets
  using the config functionality. (#792)


Deprecated Features
-------------------

- `Chromatic`.  Class functionality subsumed by `ChromaticTransformation`.
  (#789)
- `.copy()` methods for immutable classes, including `GSObject`,
  `ChromaticObject`, `SED`, and `Bandpass` have been deprecated as unnecessary.
  (#789)


New Features
------------

- Added ability to use `numpy`, `np`, or `math` in all places where we evaluate
  user input, including DistDeviate (aka RandomDistribution in config files),
  PowerSpectrum, UVFunction, RaDecFunction, Bandpass, and SED.  Some of these
  had allowed `np.` for numpy commands, but inconsistently, so now they should
  all reliably work with any of these three module names. (#776)
- `SED`s can now be constructed with flexible units via the `astropy.units`
  module. (#789).
- Added new surface brightness profile, 'InclinedExponential'. This represents
  the 2D projection the 3D profile:
      I(R,z) = I_0 / (2h_s) * sech^2 (z/h_s) * exp(-R/R_s),
  inclined to the line of sight at a desired angle. If face-on (inclination =
  0 degrees), this will be identical to the Exponential profile.  (#782)
- Added ability to specify optical aberrations in terms of annular Zernike
  coefficients.  (#771)


New config features
-------------------

- Output slightly more information about the COSMOSCatalog() (if any) being used
  as the basis of simulations, at the default verbosity level. (#804)
