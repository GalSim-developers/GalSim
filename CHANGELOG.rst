Changes from v2.5 to v2.6
=========================

We currently support Python 3.7 through 3.12.

Dependency Changes
------------------

- Removed an accidental implicit dependency we had on scipy in `FittedSIPWCS`. (#1253, #1305)

API Changes
-----------


Config Updates
--------------



New Features
------------

- Added `InterpolatedChromaticObject.from_images`. (#1294, #1296)
- Allow PosixPath instances in constructors for `Bandpass` and `SED`. (#1270, #1304)


Performance Improvements
------------------------



Bug Fixes
---------

- Fixed a bug in the config-layer parsing of Position items. (#1299, #1300)
- Fixed a bug in `DoubleZernike` to handle integer arguments. (#1283, #1303)
- Fixed a bug in `ChromaticConvolution` when one of the items is a simple `GSObject`
  and the other has an inseparable SED. (#1302, #1306)
