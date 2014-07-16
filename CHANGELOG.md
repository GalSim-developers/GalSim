Changes from v1.1 to v1.2
=========================

New Features
------------

- Added magnitudes as a method to set the flux of SED objects. (#547)
- Added SED.calculateDCRMomentShifts and SED.calculateChromaticSeeingRatio convenience functions
  for estimating chromatic PSF moment shifts (#547)
- Renamed the GSParams parameter `alias_threshold` to `folding_threshold`, a clearer term for the
  profile image folding in real space that this GSParam controls (#562)

Bug Fixes and Improvements
--------------------------

- SED and Bandpass methods that return a new SED or Bandpass now attempt to preserve the type of
  the object calling the method. (#547)
- Changed the the `file_name` argument to `CorrelatedNoise.getCOSMOSNoise()` to no longer be
  required.  The normal file to use is now installed along with GalSim (in the directory
  PREFIX/share/galsim), so that file can be used by default. (#548)

Updates to config options
-------------------------
