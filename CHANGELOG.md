Changes from v2.0 to v2.1
=========================

API Changes
-----------


Deprecated Features
-------------------


New Features
------------

- Added spline as LookupTable2D interpolant. (#982)
- Added ability to use a galsim.Interpolant for LookupTable and LookupTable2D
  interpolants. (#982)
- Added option for faster grid interpolation of LookupTable2D. (#982)


Bug Fixes
---------

- Updated the diffusion functional form for the Silicon class to match lab
  experiments of ITL sensors. (#981)
