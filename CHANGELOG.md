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

- Changed the default seed used for Deviate objects when no seed is given to use /dev/urandom
  if it is available.  If not, it reverts to the old behavior of using the current time to
  generate a seed value. (#537)
- Changed SED and Bandpass methods that return a new SED or Bandpass to attempt to preserve the
  type of the calling object if it is a subclass of SED or Bandpass respectively. (#547)
- Changed the the `file_name` argument to `CorrelatedNoise.getCOSMOSNoise()` to no longer be
  required.  The normal file to use is now installed along with GalSim (in the directory
  PREFIX/share/galsim), so that file can be used by default. (#548)

Updates to config options
-------------------------

- Added RandomPoisson, RandomBinomial, RandomWeibull, RandomGamma, and RandomChi2 random number
  generators, corresponding to the random deviate classes in the python layer. (#537)

