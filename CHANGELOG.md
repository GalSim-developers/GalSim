Changes from v1.1 to v1.2
=========================

New Features
------------

- Added magnitudes as a method to set the flux of SED objects. (#547)
- Added SED.calculateDCRMomentShifts and SED.calculateChromaticSeeingRatio convenience functions
  for estimating chromatic PSF moment shifts (#547)
- Added an option to impose N-fold symmetry (for user-selected even values of
  N>=4) on correlated noise fields, parallel to the existing noise whitening
  options. Changed name for noise whitening routine from applyWhiteningTo() to
  whitenImage(), parallel to the new symmetrizeImage(); use of applyWhiteningTo()
  is deprecated.  (#529)

Bug Fixes and Improvements
--------------------------

- SED and Bandpass methods that return a new SED or Bandpass now attempt to preserve the type of
  the object calling the method. (#547)

Updates to config options
-------------------------

- Moved noise whitening option from being an attribute of the RealGalaxy class,
  to being a part of the description of the noise. (#529)
