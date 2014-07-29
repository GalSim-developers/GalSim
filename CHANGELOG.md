Changes from v1.1 to v1.2
=========================

New Features
------------

- Changed name of noise whitening routine from noise.applyWhiteningTo(image)
  to image.whitenNoise(noise), parallel to image.addNoise(noise); use of 
  noise.applyWhiteningTo() is deprecated. (#529)
- Added an option to impose N-fold symmetry (for user-selected even values of
  N>=4) on the noise in images with correlated noise rather than fully whiten
  the noise called image.symmetrizeNoise(noise, N). (#529)
- Added magnitudes as a method to set the flux of SED objects. (#547)
- Added SED.calculateDCRMomentShifts and SED.calculateChromaticSeeingRatio convenience functions
  for estimating chromatic PSF moment shifts. (#547)
- Renamed the GSParams parameter `alias_threshold` to `folding_threshold`, a clearer term for the
  profile image folding in real space that this GSParam controls. (#562)
- Modified the internals of noise generation by correlated noise models to use Hermitian symmetry,
  for greater efficiency. (#563)
- Extended to the `rotate`, `shear`, and `transform` methods of ChromaticObject the ability
  to take functions of wavelength for the arguments. (#581)

Bug Fixes and Improvements
--------------------------

- Modified BoundsI and PositionI initialization to ensure that integer elements
  in NumPy arrays with `dtype==int` are handled without error. (#486)
- Changed the default seed used for Deviate objects when no seed is given to use /dev/urandom
  if it is available.  If not, it reverts to the old behavior of using the current time to
  generate a seed value. (#537)
- Changed SED and Bandpass methods that return a new SED or Bandpass to attempt to preserve the
  type of the calling object if it is a subclass of SED or Bandpass respectively. (#547)
- Changed the the `file_name` argument to `CorrelatedNoise.getCOSMOSNoise()` to no longer be
  required.  The normal file to use is now installed along with GalSim (in the directory
  PREFIX/share/galsim), so that file can be used by default. (#548)
- Fixed the `dtype=` kwarg used when initializing `Image` instances to interpret the aliases `int`
  and `float` as the `numpy.int32` and `numpy.float64` data types, respectively.  Previously the
  behavior was unpredictable and platform dependent. (#571)

Updates to config options
-------------------------

- Moved noise whitening option from being an attribute of the RealGalaxy class,
  to being a part of the description of the noise. (#529)
- Added RandomPoisson, RandomBinomial, RandomWeibull, RandomGamma, and RandomChi2 random number
  generators, corresponding to the random deviate classes in the python layer. (#537)

