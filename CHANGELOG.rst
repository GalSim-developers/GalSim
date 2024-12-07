Changes from v2.5 to v2.6
=========================

We currently support Python 3.7 through 3.12.

A complete list of all new features and changes is given below.
`Relevant PRs and Issues,
<https://github.com/GalSim-developers/GalSim/milestone/23?closed=1>`_
whose issue numbers are listed below for the relevant items.

Dependency Changes
------------------

- Removed an accidental implicit dependency we had on scipy in `FittedSIPWCS`. (#1253, #1305)


API Changes
-----------

- Changed the behavior of random_seed. (See below) For most use cases, this is essentially a bug
  fix, but if users were relying on the old behavior, you may need to change your config file to
  work with the new behavior.  See `Image Field Attributes` for more details about the new
  behavior. (#1309)


Config Updates
--------------

- Changed the behavior of random_seed to be less confusing.  Now the first random_seed is always
  converted into a sequence based on obj_num, and later ones (if any) in a list are not.
  If you want a non-standard seed sequence, you should now put it in a list somewhere after
  the first item.  The first item will always evaluate as an integer value and create a sequence
  based on that indexed by obj_num. (#1309)
- Added Quantity and Unit types to allow more intuitive specification of values with units
  in the config file. (#1311)


New Features
------------

- Added `InterpolatedChromaticObject.from_images`. (#1294, #1296)
- Allow PosixPath instances in constructors for `Bandpass` and `SED`. (#1270, #1304)
- Added filter information for the Prism and Grism in the roman module. (#1307)
- Added options to give some unitful values as an astropy Quantity rather than rely on
  implicit units specified in the doc string. (#1311)


Bug Fixes
---------

- Fixed a bug in the config-layer parsing of Position items. (#1299, #1300)
- Fixed a bug in `DoubleZernike` to handle integer arguments. (#1283, #1303)
- Fixed a bug in `ChromaticConvolution` when one of the items is a simple `GSObject`
  and the other has an inseparable SED. (#1302, #1306)


Changes from 2.6.0 to 2.6.1
---------------------------

- Fixed a build problem for some compilers when GPU offloading is enabled. (#1313, #1314)


Changes from 2.6.1 to 2.6.2
---------------------------

- Fixed a bug that could cause errors when drawing some chromatic profiles with photon shooting
  if the realized flux is zero. (#1317)
- Fixed a bug that could occasionally cause singular matrix exceptions in the new
  `FittedSIPWCS` solver. (#1319)

Changes from 2.6.2 to 2.6.3
---------------------------

- Fixed a bug in the object centering when drawing with nx, ny and center. (#1322, #1323)
