Changes from v1.4 to v1.5
=========================

API Changes
-----------
- `drawImage()` now accepts `exptime` and `area` keywords to indicate the image
  exposure time and telescope collecting area.  (#789)
- dimensions of `SED` changed from [photons/wavelength-interval] to
  [photons/wavelength-interval/area/time].  (#789)


Dependency Changes
------------------



Bug Fixes
---------
- Added checks to `SED`s and `ChromaticObject`s for dimensional sanity.  (#789)



Deprecated Features
-------------------
- `ChromaticObject.__init__()`.  `ChromaticObject` is now purely abstract.
  (#789)
- `Chromatic`.  Class functionality subsumed by `ChromaticTransformation`.
  (#789)


New Features
------------



New config features
-------------------
