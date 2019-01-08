Changes from v2.1 to v2.2
=========================

Deprecated Features
-------------------

- Removed functionality to store and re-load WFIRST PSFs, and to obtain multiple WFIRST
  PSFs simultaneously. (#919)

New Features
------------

- Updated WFIRST WCS and PSF routines to use Cycle 7 specifications for detector configurations,
  pupil planes, and aberrations. (#919)
- Enabled Zernikes up to 22 (previously 11) in WFIRST PSFs, and added dependence on position
  within the SCA. (#919)

Bug Fixes
---------

