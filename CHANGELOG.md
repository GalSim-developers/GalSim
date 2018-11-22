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
