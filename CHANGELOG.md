Changes from v1.4 to v1.5
=========================

API Changes
-----------

- Simplified the return value of galsim.config.ReadConfig. (#580)
- Changed RealGalaxyCatalog methods `getGal` and `getPSF` to return
  `GSObject`s instead of `Image`s; added `getGalImage` and `getPSFImage` to
  enable former behavior (#640)
- Moved packaged `SED` and `Bandpass` files from `.../share/galsim/` to
  `.../share/galsim/SEDs` and `.../share/galsim/bandpasses` respectively.
  (#640)
- Removed option to pass subclass of GSObject to GSObject initializer. (#640)
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
- Changed `drawKImage` to return a single ImageCD instance rather than two
  ImageD instances (for real and imag parts).  The old syntax of
  `re, im = obj.drawKImage(...)` will still work, but it will raise a
  deprecation warning. (#799)
- Changed `InterpolatedKImage` to take an ImageCD rather than two ImageD
  instances. The old syntax will work, but it will raise a deprecation
  warning. (#799)
- Dynamic PhaseScreenPSFs now require an explicit start time and time step.
  Clock management of phase screens now handled implicitly. (#824)
- The time_step for updating atmospheric screens and the time_step used for
  integrating a PhaseScreenPSF over time are now independent (#824)
- OpticalScreen now requires `diam` argument. (#824)
- Some of the backend (but nonetheless public API) methods of PhaseScreen and
  PhaseScreenList have changed.  See the docstrings of these classes for
  the new API if you have been using these methods. (#824)
- Slightly changed the signatures of some back-end, but nonetheless public,
  config-layer functions.  If you have been using custom config modules,
  there may be slight changes to your code required.  See the doc strings of
  these functions for more information. (#865)
- Switched galsim.Image(image) to make a copy of the image rather than a view.
  If you want a view, you should use the more intuitive image.view().  (#873)
- Changed behaviour of the `preload` option in RealGalaxyCatalog and
  COSMOSCatalog to preload data in memory, not just the fits HDUs (#884)


Dependency Changes
------------------
- Added `astropy` as a required dependency for chromatic functionality. (#789)
- Switched `scons tests` test runner from `nosetests` to `pytest`. (#892)


Bug Fixes
---------

- Added checks to `SED`s and `ChromaticObject`s for dimensional sanity. (#789)
- Fixed an error in the magnification calculated by NFWHalo.getLensing(). (#580)
- Fixed bug when whitening noise in images based on COSMOS training datasets
  using the config functionality. (#792)
- Fixed some handling of images with undefined bounds. (#799)
- Fixed bug in image.subImage that could cause seg faults in some cases. (#848)
- Fixed minor bug in shear == implementation. (#865)
- Fixed bug in GSFitsWCS that made `toImage` sometimes fail to converge. (#880)


Deprecated Features
-------------------

- Deprecated `simReal` method, a little-used way of simulating images
  based on realistic galaxies. (#787)
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
- Deprecated ability to create multiple PhaseScreenPSFs with single call
  to makePSF, since it is now just as efficient to call makePSF multiple
  times. (#824)


New Features
------------

- Added new surface brightness profile, 'DeltaFunction'. This represents a
  point source with a flux value. (#533)
- Added `ChromaticRealGalaxy`, which can use multi-band HST-images to model
  realistic galaxies, including color gradients (#640)
- Added `CovarianceSpectrum` to propagate noise through
  `ChromaticRealGalaxy` (#640)
- Updated packaged bandpasses and SEDs and associated download scripts (#640)
- Added HST bandpasses covering AEGIS and CANDELS surveys (#640)
- Added `drawKImage` method for `ChromaticObject` and `CorrelatedNoise` (#640)
- Added support for reading in of unsigned int Images (#715)
- Added a new Sensor class hierarchy, including SiliconSensor, which models
  the repulsion of incoming electrons by the electrons already accumulated on
  the sensor.  This effect is known as the "brighter-fatter effect", since it
  means that brighter objects are a bit larger than dimmer but otherwise-
  identical objects. (#722)
- Added `save_photons` option to `drawImage` to output the photons that were
  shot when photon shooting (if applicable). (#722)
- Added image.bin and image.subsample methods. (#722)
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
  the 2D projection of the 3D profile:
      I(R,z) = I_0 / (2h_s) * sech^2 (z/h_s) * exp(-R/R_s),
  inclined to the line of sight at a desired angle. If face-on (inclination =
  0 degrees), this will be identical to the Exponential profile.  (#782)
- Allow selection of random galaxies from a RealGalaxyCatalog or COSMOSCatalog
  in a way that accounts for any selection effects in catalog creation, using
  the 'weight' entries in the catalog. (#787)
- Added possibility of using `dtype=complex` or `numpy.complex128` for Images,
  the shorthand alias for which is ImageCD. Also `dtype=numpy.complex64` is
  allowed, the alias for which is ImageCF. (#799, #873)
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
- Added new surface brightness profile, 'InclinedSersic'. This is a
  generalization of the InclinedExponential profile, allowing Sersic disks and
  a truncation radius for the disk. (#811)
- Added new profile `galsim.RandomWalk`, a class for generating a set of
  point sources distributed using a random walk.  Uses of this profile include
  representing an "irregular" galaxy, or adding this profile to an Exponential
  to represent knots of star formation. (#819)
- Allowed drawImage to take a non-uniform WCS when default constructing the
  image from either nx,ny or bounds. (#820)
- Added 'generate' function to BaseDeviate and 'sed.sampleWavelength' to draw
  random wavelengths from an SED. (#822)
- Added function assignPhotonAngles to add arrival directions (in the form of
  dx/dz and dy/dz slopes) to an existing photon array. (#823)
- Added geometric optics approximation for photon-shooting through
  PhaseScreenPSFs (includes atmospheric PSF and OpticalPSF).  This
  approximation is now the default for non-optical PhaseScreenPSFs. (#824)
- Added gradient method to LookupTable2D. (#824)
- Added `surface_ops` option to `drawImage` function, which applies a list of
  surface operations to the photon array before accumulating on the image.
  (#827)
- Added `ii_pad_factor` kwarg to PhaseScreenPSF and OpticalPSF to control the
  zero-padding of the underlying InterpolatedImage. (#835)
- Added galsim.fft module that includes functions that act as drop-in
  replacements for np.fft functions, but using the C-layer FFTW package.
  Our functions have more restrictions on the input arrays, but when valid
  are generally somewhat faster than the numpy functions. (#840)
- Added some variants of normal functions and methods with a leading underscore.
  These variants skip the normal sanity checks of the input parameters and
  often have more limited options for the input arguments.  Some examples:
  `_Image`, `_Shear`, `_BoundsI`, `_Transform`, `obj._shear`, `obj._shift`,
  `obj._drawKImage`, `image._view`, `image._shift`.  These are appropriate
  for advanced users who are optimizing a tight loop and find that the normal
  Python checks are taking a significant amount of time. (#840, #873)
- Added a hook to the WCS classes to allow them to vary with color, although
  most of our current WCS classes are not able to use this feature.  The only
  one that can is UVFunction, which may now optionally have a color term
  if you set `uses_color=True`.  (Note however that there is not yet a
  mechanism to assign colors to objects in the config parser.) (#865)
- Added optional `variance` parameter to PowerSpectrum.buildGrid to
  renormalize the variance of the returned shear values. (#865)
- Added ability to get position (x,y,z) on the unit sphere corresponding to
  a CelestialCoord with `coord.get_xyz()`.  Also make a CelestialCoord from
  (x,y,z) using `CeletialCoord.from_xyz(x,y,z)`. (#865)
- Added an optional `center` argument for `Angle.wrap()`. (#865)
- Added `recenter` option to drawKImage to optionally not recenter the input
  image at (0,0).  The default `recenter=True` is consistent with how this
  function has worked in previous versions. (#873)


New config features
-------------------

- Output slightly more information about the COSMOSCatalog() (if any) being used
  as the basis of simulations, at the default verbosity level. (#804)
- Changed the name of galsim.config.CalculateNoiseVar to the slightly more
  verbose, but also more readable CalculateNoiseVariance. (#820)
- Setting config['rng'] is no longer required when manually running commands
  like `galsim.config.BuildGSObject`.  If you do not care about deterministic
  pseudo-rngs, then it will just use /dev/urandom for you. (#820)
- Allow PoissonNoise without any sky level, in which case only the shot noise
  in the signal photons contribute to the noise.  Likewise for CCDNoise. (#820)
- Let 'None' in the config file mean `None`. (#820)
- Require 'max_extra_noise' to be explicitly set for photon shooting if you
  want it rather than have it default to 0.01.  (#820)
- Added --except_abort option to galsim executable to abort execution if a file
  has an error, rather than just reporting the exception and continuing on
  (which is still the default behavior). (#820)
- Added optional probability parameter 'p' for Random bool values. (#820)
- Added ability to specify world_pos in celestial coordinates (#865)
- Added the ability to have multiple rngs with different update sequences
  (e.g. to have some random galaxy properties repeat for the corresponding
  galaxies on multiple images).  (#865)
- Added ngrid, center, variance, index options to power_spectrum input field.
  (#865)
- Skip drawing objects whose postage stamps will end up completely off the
  main image currently being worked on. (#865)
- Added skip option in stamp field, which works the same as the skip parameter
  in gal or psf fields. (#865)
- Added ':field' syntax for templates, which use the current dict as the base
  rather than reading from another file (with 'file:field'). (#865)
- No longer tries to process extra output items for stamps that are skipped.
  This is normally better, since the extra output processing probably depends
  on the stamp processing having been completed.  But it is customizable via
  the `processSkippedStamp` method of ExtraOutputBuilders, so you can override
  this behavior in your custom modules if you prefer. (#865)
