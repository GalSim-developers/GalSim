Changes from v1.1 to v1.2
=========================

New Features
------------

- Added magnitudes as a method to set the flux of SED objects. (#547)
- Added SED.calculateDCRMomentShifts and SED.calculateChromaticSeeingRatio convenience functions
  for estimating chromatic PSF moment shifts (#547)

Bug Fixes and Improvements
--------------------------

- SED and Bandpass methods that return a new SED or Bandpass now attempt to preserve the type of
  the object calling the method. (#547)

Updates to config options
-------------------------
