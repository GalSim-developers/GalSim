Changes from v2.4 to v2.5
=========================


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


New Features
------------

- Added `ShapeData.applyWCS` method to convert HSM shapes to sky coordinates.  Also added
  the option ``use_sky_coords=True`` to `FindAdaptiveMom` to apply this automatically.


Performance Improvements
------------------------



Bug Fixes
---------

