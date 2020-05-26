Changes from v2.2 to v2.3
=========================


Dependency Changes
------------------

- Removed future as a dependency. (#1082)

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

- Fixed a few issues with parsing truth catalog items and, in general, places
  where a Current item might be used before it was parsed. (#1083)
- Added value-type-specific type names like Random_float, Random_Angle, etc.
  to help in some cases where the config processing cannot know what value
  type to use for further processing.  (#1083)
- Fixed a subtle issue in Eval string processing if one Current item is a
  shorter version of another one.  e.g. @gal and @gal.index.  It had been
  the case that the longer one might not be parsed properly. (#1083)


New Features
------------


Bug Fixes
---------

- Fixed horner and horner2d when inputs are complex. (#1054)
