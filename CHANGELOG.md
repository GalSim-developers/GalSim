Changes from v2.1 to v2.2
=========================

Deprecated Features
-------------------

API Changes
-----------

- Removed functionality to store and re-load WFIRST PSFs, and to obtain multiple WFIRST
  PSFs simultaneously. (#919)

New Features
------------
- Added r0_500 argument to VonKarman.
- Updated WFIRST WCS and PSF routines to use Cycle 7 specifications for detector configurations,
  pupil planes, and aberrations. In particular, there is now a different
  pupil plane image for shorter- and longer-wavelength bands.  (#919)
- Enabled Zernikes up to 22 (previously 11) in WFIRST PSFs, and added dependence on position
  within the SCA. (#919)
- Added spline as LookupTable2D interpolant. (#982)
- Added ability to use a galsim.Interpolant for LookupTable and LookupTable2D
  interpolants. (#982)
- Added option for faster grid interpolation of LookupTable2D. (#982)
- Added wfirst fermi persistence model and utilities.LinkedList. (#992) 

Bug Fixes
---------
