Changes from v1.4 to v1.5
=========================

API Changes
-----------

- Simplified the return value of galsim.config.ReadConfig. (#580)
- Changed the dimensions of `SED` from [photons/wavelength-interval] to either
  [photons/wavelength-interval/area/time] or [1] (dimensionless).
  `ChromaticObject`s representing stars or galaxies take SEDs with the former
  dimensions; those representing a chromatic PSF take SEDs with the latter
  dimensions. (#789)
- Added keywords `exptime` and `area` to `drawImage()` to indicate the image
  exposure time and telescope collecting area. (#789)
- Added restrictions to `ChromaticObject`s and `SED`s consistent with
  dimensional analysis.  E.g., only `ChromaticObject`s with dimensionful SEDs
  can be drawn. (#789)
- Changed `drawKImage` to return a single ImageC instance rather than two
  ImageD instances (for real and imag parts).  The old syntax of
  `re, im = obj.drawKImage(...)` will still work, but it will raise a
  deprecation warning. (#799)
- Changed `InterpolatedKImage` to take an ImageC rather than two ImageD
  instances. The old syntax will work, but it will raise a deprecation
  warning. (#799)


Dependency Changes
------------------
- Added `astropy` as a required dependency for chromatic functionality. (#789)


Bug Fixes
---------

- Added checks to `SED`s and `ChromaticObject`s for dimensional sanity. (#789)
- Fixed an error in the magnification calculated by NFWHalo.getLensing(). (#580)
- Fixed bug when whitening noise in images based on COSMOS training datasets
  using the config functionality. (#792)
- Fixed some handling of images with undefined bounds. (#799)


Deprecated Features
-------------------

- Deprecated `Chromatic` class.  This functionality has been subsumed by
  `ChromaticTransformation`.  (#789)
- Deprecated `.copy()` methods for immutable classes, including `GSObject`,
  `ChromaticObject`, `SED`, and `Bandpass`, which are unnecessary. (#789)
- Deprecated `wmult` parameter of `drawImage`. (#799)
- Deprecated `Image.at` method. Normally im(x,y) or im[x,y] would be the
  preferred syntax, but for the case where you want a named method, the
  new name is `getValue` in parallel with `setValue`. (#799)
- Deprecated `gain` parameter of `drawKImage`.  It does not really make
  sense.  If you had been using it, you should instead just divide the
  returned image by gain, which will have the same effect and probably
  be clearer in your own code about what you meant. (#799)


New Features
------------

- Added support for reading in of unsigned int Images (#715)
- Added ability to specify optical aberrations in terms of annular Zernike
  coefficients.  (#771)
- Added ability to use `numpy`, `np`, or `math` in all places where we evaluate
  user input, including DistDeviate (aka RandomDistribution in config files),
  PowerSpectrum, UVFunction, RaDecFunction, Bandpass, and SED.  Some of these
  had allowed `np.` for numpy commands, but inconsistently, so now they should
  all reliably work with any of these three module names. (#776)
- `SED`s can now be constructed with flexible units via the `astropy.units`
  module. (#789).
- Added new surface brightness profile, 'InclinedExponential'. This represents
  the 2D projection the 3D profile:
      I(R,z) = I_0 / (2h_s) * sech^2 (z/h_s) * exp(-R/R_s),
  inclined to the line of sight at a desired angle. If face-on (inclination =
  0 degrees), this will be identical to the Exponential profile.  (#782)
- Added possibility of using `dtype=complex` for Images, the shorthand alias
  for which is called ImageC. (#799)
- Added `maxSB()` method to GSObjects to return an estimate of the maximum
  surface brightness.  For analytic profiles, it returns the correct value,
  but for compound objects (convolutions in particular), it cannot know the
  exact value without fully drawing the object (which would defeat the point
  for the use cases where we want this information).  So it does its best to
  estimate something close, generally erring on the high side.  So the true
  maximum SB <~ obj.maxSB(). (#799)
- Added `im[x,y] = value` as a valid replacement for im.setValue(x,y,value).
  Likewise `value = im[x,y]` is equivalent to `value = im(x,y)` or `value =
  im.getValue(x,y)`. (#799).
- Added ability to do FFTs directly on images.  The relevant methods for
  doing so are `im.calculate_fft()` and `im.calculate_inverse_fft()`.  There
  is also `im.wrap()` which can be used to wrap an image prior to doing the
  FFT to properly alias the data if necessary. (#799)
- Added new profile `galsim.RandomWalk`, a class for generating a set of 
  point sources distributed using a random walk.  Uses of this profile include
  representing an "irregular" galaxy, or adding this profile to an Exponential
  to represent knots of star formation. (#819)


New config features
-------------------

- Output slightly more information about the COSMOSCatalog() (if any) being used
  as the basis of simulations, at the default verbosity level. (#804)
