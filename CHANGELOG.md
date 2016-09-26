Changes from v1.4 to v1.5
=========================

API Changes
-----------

- `drawImage()` now accepts `exptime` and `area` keywords to indicate the image
  exposure time and telescope collecting area.  (#789)
- dimensions of `SED` changed from [photons/wavelength-interval] to
  [photons/wavelength-interval/area/time].  (#789)
- Simplified the return value of galsim.config.ReadConfig. (#580)


Dependency Changes
------------------



Bug Fixes
---------

- Added checks to `SED`s and `ChromaticObject`s for dimensional sanity.  (#789)
- Fixed an error in the magnification calculated by NFWHalo.getLensing(). (#580)
- Fixed bug when whitening noise in images based on COSMOS training datasets
  using the config functionality. (#792)

Deprecated Features
-------------------

- `ChromaticObject.__init__()`.  `ChromaticObject` is now purely abstract.
  (#789)
- `Chromatic`.  Class functionality subsumed by `ChromaticTransformation`.
  (#789)


New Features
------------

- Added ability to use `numpy`, `np`, or `math` in all places where we evaluate
  user input, including DistDeviate (aka RandomDistribution in config files),
  PowerSpectrum, UVFunction, RaDecFunction, Bandpass, and SED.  Some of these
  had allowed `np.` for numpy commands, but inconsistently, so now they should
  all reliably work with any of these three module names. (#776)


New config features
-------------------
