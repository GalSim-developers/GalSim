Changes from v2.4 to v2.5
=========================

This release adds support for Pyton 3.11 and 3.12. We currently support 3.7 through 3.12.

API Changes
-----------

- Deprecated the use of filter W149 in roman module.  New name is W146. (#1017)
- Deprecated automatic allocation of `PhotonArray` arrays via "get" access rather than
  "set" access for the arrays that are not initially allocated.  E.g. writing
  ``dxdz = photon_array.dxdz`` will emit a warning (and eventually this will be an error)
  if the angle arrays have not been either set or explicitly allocated.  One should be sure
  to either set them (e.g. using ``photon_array.dxdz = [...]``) or explicitly allocate
  them (using ``photon_array.allocateAngles()``).  (#1191)
- Changed the ``.SED`` attribute name of `ChromaticObject` to lowercase ``.sed``. (#1245)


Config Updates
--------------

- Fixed a bug in Scattered type, where it didn't respect world_pos being specified in the
  stamp field, as it should.  (#1190)
- Added a new ``initial_image`` input type that lets you read in an existing image file
  and draw onto it. (#1237)
- Added skip_failures option in stamp fields.  (#1238)
- Let input items depend on other input items, even if they appear later in the input field.
  (#1239)
- Allow profiling output to reach the logger when running with -v0. (#1245)
- Added Eval type for GSObjects. (#1250)


New Features
------------

- Updated Roman telescope data to Phase C (aka Cycle 9) specifications (#1017)
- Added `ShapeData.applyWCS` method to convert HSM shapes to sky coordinates.  Also added
  the option ``use_sky_coords=True`` to `hsm.FindAdaptiveMom` to apply this automatically. (#1219)
- Added `DoubleZernike` class and related functionality. (#1221)
- Added some utility functions that we have found useful in our test suite, and which other
  people might want to use to the installed galsim.utilities. (#1240)
- Added `utilities.merge_sorted` which merges two or more sorted numpy arrays faster than
  the available numpy options. (#1243)
- Added `EmissionLine` class to represent emission line SEDs. (#1247, #1249)
- Updated data in `roman` module to Phase C (Cycle 9) information. (#1017, #1251)


Performance Improvements
------------------------

- Added support for GPU offloading.  So far this has only been applied to the Silicon sensor
  calculations of the brighter-fatter effect. (#1212, #1217, #1218, #1222, #1224, #1230)
- Drawing chromatic objects with photon shooting automatically adds a WavelengthSampler photon_op.
  It used to do this regardless of if one was already in a photon_ops list, which is inefficient.
  Now it only adds it if there is not already one given by the user. (#1229, #1236)
- Work around an OMP bug that disables multiprocessing on some systems when omp_get_max_threads
  is called. (#1241)
- Sped up the `combine_wave_list` function, using the new `merge_sorted` function.  (#1243)
- No longer keep a separate ``wave_list`` array in `ChromaticObject`.  These are always
  equal to the ``wave_list`` in the ``sed`` attribute, so there is no need to duplicate the
  work of computing the ``wave_list``. (#1245)
- Delayed the calculation of the ``sed`` attributes of `ChromaticObject` until they are actually
  needed.  Since sometimes they are not needed, this is a performance improvement in those cases.
  (#1245)
- Reduce long-term memory usage of Silicon class after drawing onto a very large stamp and
  then moving on to smaller stamps. (#1246)


Bug Fixes
---------

- Fixed a bug that could lead to overflow in extremely large images. (#1017)
- Fixed a slight error in the Moffat maxk calculation. (#1208, #1210)
- Fixed a bug that prevented Eval types from generating lists in config files in some contexts.
  (#1220, #1223)
- Fixed the absorption depth calculation in the Silicon class to allow wavelengths that are
  outside the given range of the absorption lookup table.  It now just uses the limiting values,
  rather than raising an exception. (#1227)
- Changed the SED class to correctly broadcast over waves when the SED is constant. (#1228, #1235)
- Fixed some errors when drawing ChromaticTransformation objects with photon shooting. (#1229)
- Fixed the flux drawn by ChromaticConvolution with photon shooting when poisson_flux=True. (#1229)
- Fixed a slight inaccuracy in the FFT phase shifts for single-precision images. (#1231, #1234)
- Fixed a bug that prevented a convolution of two PhaseScreenPSF objects from being drawn with
  photon shooting. (#1242)


Changes from v2.5.0 to v2.5.1
=============================

- Fixed an incompatibility with Python 3.12 that we had missed.
- Fixed a bug in the SED class normalization when using astropy.units for flux_type.  Thanks
  to Sid Mau for finding and fixing this bug. (#1254, #1256)
- Fixed a bug in the `EmissionLine.atRedshift` method. (#1257)
- Added interpolant option to `SED` and `Bandpass` classes to use when reading from a file.
  (#1257)
- Improved the behavior of SEDs when using spline interpolant. (#1187, #1257)
- No longer pickle the SED of chromatic objects when the SED is a derived value. (#1257)
- Added interpolant option to `galsim.trapz`. (#1257)
- Added clip_neg option to `DistDeviate` class. (#1257)
- Fixed a bug in `SiliconSensor` if the image is outside the range where tree rings are defined.
  (#1258)
- Implemented algorithm for `ChromaticSum` to be used as a photon_op. (#1259)
- Added `PhotonArray.copyFrom` method. (#1259)
- Deprecated `PhotonArray.setCorrelated` and `PhotonArray.isCorrelated`, since they are not
  necessary anymore. (#1259)
- Deprecated `PhotonArray.assignAt` in favor of the more flexible `PhotonArray.copyFrom`
  method. (#1259)

Changes from v2.5.1 to v2.5.2
=============================

- Added galsim.roman.max_sun_angle as a module-level named variable. (#1261)
- Fixed an error in the CelestialWCS.radecToxy doc string. (#1275)
- Fixed a pandas deprecation. (#1278)
- Fixed some broken links in the installation docs. (#1279)
- Added observed_e1, observed_e2 properties to the hsm.ShapeData structure. (#1279)
- Added check=False option to hsm routines FindAdaptiveMom and EstimateShdear. (#1279)
- Fixed a bug in the use of evaluated template strings in the config dict. (#1281)
- Fixed a bug in CelestialWCS computation of the local jacobian near ra=0. (#1282)
