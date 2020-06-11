Changes from v2.2 to v2.3
=========================


Deprecated Features
-------------------

- Changed the WCS method name withOrigin to shiftOrigin for non-local WCS
  types.  The functionality hasn't changed, but the name withOrigin is
  only really appropriate for LocalWCS types.  When the WCS already has a
  non-zero origin, then the action takes in really to shift the origin, not
  set a new value. (#1073)


API Changes
-----------

- The numerical details of the Kolmogorov class were updated to use a more
  precise version of a constant from Racine (1996).  Technically this changes
  the definition of the Kolmogorov profile at the 6th decimal place.  So if
  you carefully tuned an r0 value to 6 decimal places for some purpose, this
  might break that. (#1084)


Config Updates
--------------



New Features
------------


Bug Fixes
---------

- Fixed horner and horner2d when inputs are complex. (#1054)
