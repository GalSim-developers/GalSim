Changes from v2.4 to v2.5
=========================

This version adds support for Pyton 3.11.  We currently support 3.7, 3.8, 3.9, 3.10, and 3.11.

API Changes
-----------

- Deprecated automatic allocation of `PhotonArray` arrays via "get" access rather than
  "set" access for the arrays that are not initially allocated.  E.g. writing
  ``dxdz = photon_array.dxdz`` will emit a warning (and eventually this will be an error)
  if the angle arrays have not been either set or explicitly allocated.  One should be sure
  to either set them (e.g. using ``photon_array.dxdz = [...]``) or explicitly allocate
  them (using ``photon_array.allocateAngles()``).  (#1191)


Config Updates
--------------

- Fixed a bug in Scattered type, where it didn't respect world_pos being specified in the
  stamp field, as it should.  (#1190)
- Added a new ``initial_image`` input type that lets you read in an existing image file
  and draw onto it. (#1237)
- Added skip_failures option in stamp fields.  (#1238)


New Features
------------

- Added `ShapeData.applyWCS` method to convert HSM shapes to sky coordinates.  Also added
  the option ``use_sky_coords=True`` to `FindAdaptiveMom` to apply this automatically. (#1219)
- Added some utility functions that we have found useful in our test suite, and which other
  people might want to use to the installed galsim.utilities. (#1240)


Performance Improvements
------------------------

- Added support for GPU offloading.  So far this has only been applied to the Silicon sensor
  calculations of the brighter-fatter effect. (#1212, #1217, #1218, #1222, #1224, #1230)
- Drawing chromatic objects with photon shooting automatically adds a WavelengthSampler photon_op.
  It used to do this regardless of if one was already in a photon_ops list, which is inefficient.
  Now it only adds it if there is not already one given by the user. (#1229)
- Work around an OMP bug that disables multiprocessing on some systems when omp_get_max_threads
  is called. (#1241)


Bug Fixes
---------

- Fixed a slight error in the Moffat maxk calculation. (#1210)
- Fixed a bug that prevented Eval types from generating lists in config files in some contexts.
  (#1220, #1223)
- Fixed the absorption depth calculation in the Silicon class to allow wavelengths that are
  outside the given range of the absorption lookup table.  It now just uses the limiting values,
  rather than raising an exception. (#1227)
- Changed the SED class to correctly broadcast over waves when the SED is constant. (#1228)
- Fixed some errors when drawing ChromaticTransformation objects with photon shooting. (#1229)
- Fixed the flux drawn by ChromaticConvolution with photon shooting when poisson_flux=True. (#1229)
