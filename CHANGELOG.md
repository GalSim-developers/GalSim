Changes from v2.0 to v2.1
=========================

Most of the changes in version 2.1 involve efficiency improvements in running
the photon shooting rendering method.  Especially using PhaseScreenPSFs.
These changes were particularly important for the LSST DESC DC2 simulation
runs using ImSim.


Deprecated Features
-------------------

- Deprecated PhaseScreenPSF attributes `img` and `finalized`.  These are now
  implementation details, not part of the public API. (#990)
- Deprecated GSParams items `allowed_flux_variation`, `small_fraction_of_flux`,
  and `range_division_for_extreama`.  These are no longer used. (#993)

New Features
------------

- Added ability for RandomWalk profile to follow any arbitrary profile (any
  GSObject that can be photon shot) rather than just a Gaussian. (#821)
- Added spline as LookupTable2D interpolant. (#982)
- Added ability to use a galsim.Interpolant for LookupTable and LookupTable2D
  interpolants. (#982)
- Added option for faster grid interpolation of LookupTable2D. (#982)
- Added optional `offset` and `flux_ratio` parameters to `WCS.toWorld` and
  `WCS.toImage` functions. (#993)

Bug Fixes
---------

- Updated the diffusion functional form for the Silicon class to match lab
  experiments of ITL sensors. (#981)
- Fixed a bug in the PhaseScreenPSF `withGSParams` function when using
  geometric_shooting=True.  It had been erroneously preparing the screens
  for FFT rendering, which was both incorrect and slow. (#990)


Changes from v2.1.0 to v2.1.2
=============================

Bug Fix
-------

- Fixed a seg fault bug when PoissonDeviate is given `mean=0`. (#996)


Changes from v2.1.2 to v2.1.3
=============================

Bug Fix
-------

- Fixed the galsim executable to work correctly when installed by SCons.


Changes from v2.1.3 to v2.1.4
=============================

Bug Fixes
---------

- Fixed Convolve and Sum to recognize when objects all have the same gsparams,
  and thus avoid making gratuitous copies of the components.
- Added some caching for some non-trivial calculations for PhaseScreens.


Changes from v2.1.4 to v2.1.5
=============================

Bug Fixes
---------

- Fixed error when using non-int integer types as seed of BaseDeviate (#1009)
- Fixed error in use of non-integer grid_spacing in PowerSpectrum (#1020)
- Fixed FitsHeader to not unnecessarily read data of fits file. (#1024)
- Switched to yaml.safe_load to avoid PyYAML v5.0 warnings (#1025)
- Fixed some cases where numpy objected to subtracting floats from ints. (#1025)
