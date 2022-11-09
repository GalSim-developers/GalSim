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



New Features
------------



Performance Improvements
------------------------



Bug Fixes
---------

