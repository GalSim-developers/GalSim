Changes from v1.1 to v1.2
=========================

New Features
------------

- Added magnitudes as a method to set the flux of SED objects. (#547)
- Added SED.calculateDCRMomentShifts and SED.calculateChromaticSeeingRatio convenience functions
  for estimating chromatic PSF moment shifts. (#547)
- Renamed the GSParams parameter `alias_threshold` to `folding_threshold`, a clearer term for the
  profile image folding in real space that this GSParam controls. (#562)
- Extended to the `rotate`, `shear`, and `transform` methods of ChromaticObject the ability
  to take functions of wavelength for the arguments. (#581)

Bug Fixes and Improvements
--------------------------

- Changed SED and Bandpass methods that return a new SED or Bandpass to attempt to preserve the
  type of the calling object if it is a subclass of SED or Bandpass respectively. (#547)

Updates to config options
-------------------------
