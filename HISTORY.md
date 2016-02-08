Below is a summary of the major changes with each new tagged version of GalSim.
Each version also includes various other minor changes and bug fixes, which are 
not listed here for brevity.  See the CHANGLELOG.md files associated with each 
version for a more complete list.  Issue numbers related to each change are 
given in parentheses.

v1.3
====

Installation Changes
--------------------

- Require functionality in TMV 0.72. (#616)


API Changes
-----------

- Changed the name of the `bounds.addBorder()` method to `withBorder`. (#218)
- Removed (from the python layer) Interpolant2d and InterpolantXY. (#218)
- Removed the MultipleImage way of constructing an SBInterpolatedImage. (#218, #642)
- Made the default tolerance for all Interpolants equal to 1.e-4.. (#218)
- Deprecated the __rdiv__ operator from Bandpass and SED. (#218)
- Made all returned matrices consistently use numpy.array, rather than
  numpy.matrix. (#218)
- Made the classes PositionI, PositionD, GSParams, and HSMParams immutable.
  (#218, #643)
- Deprecated CorrelatedNoise.calculateCovarianceMatrix. (#630)
- Officially deprecated the methods and functions that had been described as
  having been removed or changed to a different name. (#643)
- Added function to interleave a set of dithered images into a single
  higher-resolution image. (#666)


New Features
------------

- Made all GalSim objects picklable unless they use fundamentally unpicklable
  things such as lambda expressions, improved str and repr representations,
  made __eq__, __ne__, and __hash__ work better. (#218)
- Added ability to set the zeropoint of a bandpass to a numeric value on
  construction. (#218)
- Added ability to set the redshift of an SED on construction. (#218)
- Updated CorrelatedNoise to work with images that have a non-trivial WCS.
  (#501)
- Added new methods of the image class to simulate detector effects (#555, #558).
- Enabled constructing a FitsHeader object from a dict, and allow
  FitsHeader to be default constructed with no keys. (#590)
- Added a module, galsim.wfirst, that includes information about the planned
  WFIRST mission, along with helper routines for constructing appropriate PSFs,
  bandpasses, WCS, etc.  (#590)
- Added native support for the TAN-SIP WCS type using GSFitsWCS. (#590)
- Added a helper program, galsim_download_cosmos, that downloads the COSMOS
  RealGalaxy catalog. (#590)
- Added new class with methods for making realistic galaxy samples using COSMOS:
  the COSMOSCatalog class and its method makeObj(). (#590 / #635).
- Added information about PSF to the data returned by EstimateShear(). (#612)
- Added Spergel(2010) profile GSObject (#616).
- Added an option to the ChromaticObject class that allows for faster image
  rendering via interpolation between stored images.  (#618)
- Added new `ChromaticAiry` and `ChromaticOpticalPSF` classes for representing
  chromatic optical PSFs. (#618)
- Enable initializing a DES_PSFEx object using a pyfits HDU directly instead
  of a filename. (#626)
- Added TopHat class implementing a circular tophat profile. (#639)
- Added ability of Noise objects to take a new random number generator (a
  BaseDeviate instance) when being copied. (#643)
- Added InterpolatedKImage GSObject for constructing a surface brightness
  profile out of samples of its Fourier transform. (#642)
- Enabled constructing a FitsHeader object from a list of (key, value) pairs,
  which preserves the order of the items in the header. (#672)

Bug Fixes and Improvements
--------------------------

- Fixed a bug in the normalization of SEDs that use `wave_type='A'`. (#218)
- Switched the sign of the angle returned by `CelestialCoord.angleBetween`.
  (#590)
- Fixed a bug in UncorrelatedNoise where the variance was set incorrectly.
  (#630)
- Changed the implementation of drawing Box and Pixel profiles in real space
  (i.e. without being convolved by anything) to actually draw the surface
  brightness at the center of each pixel. (#639)
- Fixed a bug where InterpolatedImage and Box profiles were not correctly
  rendered when transformed by something that includes a flip. (#645)
- Fixed a bug in rendering profiles that involve two separate shifts. (#645)
- Fixed a bug if drawImage was given odd nx, ny parameters, the drawn profile
  was not correctly centered in the image. (#645)
- Added intermediate results cache to `ChromaticObject.drawImage` and
  `ChromaticConvolution.drawImage` to speed up the rendering of groups
  of similar (same SED, same Bandpass, same PSF) chromatic profiles. (#670)

Updates to config options
-------------------------

- Added COSMOSGalaxy type, with corresponding cosmos_catalog input type. (#590)
- Added Spergel type. (#616)
- Added lam, diam, scale_units options to Airy and OpticalPSF types. (#618)
- Added TopHat type. (#639)


v1.2
====

New Features
------------

- Changed name of noise whitening routine from noise.applyWhiteningTo(image)
  to image.whitenNoise(noise). (#529)
- Added image.symmetrizeNoise. (#529)
- Added magnitudes as a method to set the flux of SED objects. (#547)
- Added SED.calculateDCRMomentShifts, SED.calculateChromaticSeeingRatio. (#547)
- Added image.applyNonlinearity and image.addReciprocityFaiure. (#552)
- Renamed `alias_threshold` to `folding_threshold`. (#562)
- Extended to the `rotate`, `shear`, and `transform` methods of ChromaticObject
  the ability to take functions of wavelength for the arguments. (#581)
- Added cdmodel module to describe charge deflection in CCD pixels. (#524)
- Added `pupil_plane_im` option to OpticalPSF. (#601)
- Added `nx`, `ny`, and `bounds` keywords to drawImage() and drawKImage()
  methods. (#603)

Bug Fixes and Improvements
--------------------------

- Improved efficiency of noise generation by correlated noise models. (#563)
- Modified BoundsI and PositionI initialization to ensure that integer elements
  in NumPy arrays with `dtype==int` are handled without error. (#486)
- Changed the default seed used for Deviate objects when no seed is given to
  use /dev/urandom if it is available. (#537)
- Changed SED and Bandpass methods to preserve type when returning a new object
  when possible. (#547)
- Made `file_name` argument to `CorrelatedNoise.getCOSMOSNoise()` be able
  to have a default value in the repo. (#548)
- Fixed the `dtype=` kwarg of `Image` constructor on some platforms. (#571)
- Added workaround for bug in pyfits 3.0 in `galsim.fits.read`. (#572)
- Fixed a bug in the Image constructor when passed a NumPy array with the
  opposite byteorder as the system native one. (#594)
- Fixed bug that prevented calling LookupTables on non-square 2d arrays. (#599)
- Updated the code to account for a planned change in NumPy 1.9. (#604)
- Fixed a bug where the dtype of an Image could change when resizing. (#604)
- Defined a hidden `__version__` attribute according to PEP 8 standards. (#610)

Updates to config options
-------------------------

- Moved noise whitening option from being an attribute of the RealGalaxy class,
  to being a part of the description of the noise. (#529)
- Added RandomPoisson, RandomBinomial, RandomWeibull, RandomGamma, and
  RandomChi2 random number generators. (#537)


v1.1
====

Non-backward-compatible API changes
-----------------------------------

* Changed `Pixel` to take a single `scale` parameter. (#364)
* Added new `Box` class. (#364)
* Changed `Angle.wrap()` to return the wrapped angle. (#364)
* Changed Bounds methods `addBorder`, `shift`, and `expand` to return new
  Bounds objects. (#364)
* Merged the GSParams parameters `shoot_relerr` and `shoot_abserr` into the
  parameters `integration_relerr` and `integration_abserr`. (#535)

Other changes to the API
------------------------

* Changed the name of the `dx` parameter in various places to `scale`. (#364)
* Combined the old `Image`, `ImageView` and `ConstImageView` arrays of class
  names into a single python layer `Image` class. (#364)
* Changed the methods createSheared, createRotated, etc. to more succinct
  names `shear`, `rotate`, etc. (#511)
* Changed the `setFlux` and `scaleFlux` methods to return new objects. (#511)
* Changed the Shapelet.fitImage method to `FitShapelet` (#511)
* Changed the `nyquistDx` method to `nyquistScale`. (#511)
* Moved as many classes as possible toward an immutable design. (#511)
* Combined the `draw` and `drawShoot` methods into a single `drawImage` method
  with more options about how the profile should be rendered. (#535)
* Changed the name of `drawK` to `drawKImage`. (#535)

New Features
------------

* Added new set of WCS classes. (#364)
* Added `CelestialCoord` class to represent (ra,dec) coordinates. (#364)
* Added `Bandpass`, `SED`, and `ChromaticObject` classes. (#467)
* Added `aberrations` parameter of OpticalPSF. (#409)
* Added `max_size` parameter to OpticalPSF. (#478)
* Added `text_file` parameter to FitsHeader and FitsWCS. (#508)
* Modified addNoiseSNR() method to return the added variance. (#526)
* Added `dtype` option to `drawImage` and `drawKImage`. (#526)

Bug fixes and improvements
--------------------------

* Sped up the gzip and bzip2 I/O. (#344)
* Fixed some bugs in the treatment of correlated noise. (#526, #528)

Updates to config options
-------------------------

* Added more options for `image.wcs` field. (#364)
* Changed the name of `sky_pos` to `world_pos`. (#364)
* Removed `pix` top layer in config structure.  Add `draw_method=no_pixel` to
  do what `pix : None` used to do. (#364)
* Added `draw_method=real_space` to try to use real-space convolution. (#364)
* Added ability to index `Sequence` types by any running index. (#364, #536)
* Added `Sum` type for value types for which it makes sense. (#457)
* Allowed modification of config parameters from the command line. (#479)
* Added `image.retry_failures`. (#482)
* Added `output.retry_io` item to retry failed write commands. (#482)
* Changed the default sequence indexing for most things to be 'obj_num_in_file'
  rather than 'obj_num'. (#487)
* Added `draw_method=sb`. (#535)
* Changed the `output.psf.real_space` option to `output.psf.draw_method`
  and allow all of the options that exist for `image.draw_method`. (#535)
* Added an `index` item for Ring objects. (#536)


v1.0.1
======

* Fixed some bugs in the config machinery when files have varying numbers
  of objects. (#487)
* Support astropy.io.fits in lieu of stand-alone pyfits module. (#488)
* Fixed a bug in config where 'safe' objects were not being correctly 
  invalidated when a new input item should have invalidated them.
* Fixed a bug in the drawing of a Pixel all by itself. (#497)


v1.0
====

Notable bug fixes and improvements
----------------------------------

* Fixed bug in the rendering of shifted images. (#424)
* Improved the fidelity of the Lanczos `conserve_dc=True` option. (#442)
* Switched default interpolant for RealGalaxy to Quintic, since it was
  found to be more accurate in general. (#442)
* Fixed a bug in InterpolatedImage calculateStepK function. (#454)
* Fixed a bug in Image class resize function. (#461)
* Sped up OpticalPSF and RealGalaxy significantly. (#466, #474)
* Fixed a bug in the fourier rendering of truncated Sersic profiles. (#470)

New features
------------

* Added `galsim` executable (instead of `galsim_yaml`, `galsim_json`). (#460)
* Updated the allowed range for Sersic n to 0.3 -- 6.2. (#325)
* Made RealGalaxy objects keep track of their (correlated) noise. (#430)
* Changed noise padding options for RealGalaxy and InterpolatedImage. (#430)
* Added VariableGaussianNoise class. (#430)
* Added offset parameter to both draw and drawShoot. (#439)
* Changed the name of InputCatalog to just Catalog. (#449)
* Added Dict class. (#449)
* Added MEDS file output to des module. (#376)
* Removed des module from default imports of GalSim.  Now need to import
  galsim.des explicitly or load with `galsim -m des ...` (#460)

Updates to config options
-------------------------
  
* Added RealGalaxyOriginal galaxy type. (#389)
* Added whiten option for RealGalaxy objects. (#430)
* Added `Current` type. (#430)
* Added offset option in image field. (#449)
* Added the ability to have multiple input catalogs, dicts, etc. (#449)
* Added `signal_to_noise` option for PSFs when there is no galaxy. (#459)
* Added `output.skip` and `output.noclobber` options. (#474)


v0.5
====

New features
------------

* Added Shapelet class. (#350)
* Added ability to truncate Sersic profiles. (#388)
* Added trefoil and struts to OpticalPSF. (#302, #390)
* Updates to lensing engine:
  * Added document describing how lensing engine code works. (#248)
  * Added ability to draw (gamma,kappa) from same power spectrum. (#304)
  * Added kmin_factor and kmax_factor parameters to buildGrid. (#377)
  * Added PowerSpectrumEstimator class in pse module. (#382)
* Added GSParams (#343, #426) and HSMParams (#365) classes.
* Added des module and example scripts. (#350)
* Added applyWhiteningTo method to CorrelatedNoise class. (#352)
* Changed the default centering convention for even-sized images to be in the
  actual center, rather than 1/2 pixel off-center. (#380)
* Enabled InputCatalog to read FITS catalogs. (#350)
* Added FitsHeader class and config option. (#350)
* Added the ability to read/write to a specific HDU. (#350)
* Add new function galsim.fits.writeFile. (#417)
* Added LINKFLAGS SCons option. (#380)

Updates to config
-----------------

* Added index_convention option. (#380)
* Changed the name of the center item for the Scattered image type to 
  image_pos, and added a new sky_pos item. (#380)

Bug fixes
---------

* Fix some errors related to writing to an HDUList. (#417)
* Fixed ringing when Sersic objectss were drawn with FFTs. (#426)
* Fixed bugs in obj.drawK() function. (#407)
* Fixed bugs with InterpolatedImage objects. (#389, #432)
* Fixed bug in draw routine for shifted objects. (#380)
* Fixed bug in the generation of correlated noise fields. (#352)



v0.4
====

* Added ability to pad images for InterpolatedImage or RealGalaxy with either
  correlated or uncorrelated noise. (#238)
* Added python-level LookupTable class which wraps the existing C++ Table 
  class. (#305)
* Lensing engine updates: (#305)
  - Added the option of drawing shears from a tabulated P(k)
  - Added ability to handle conversions between different angular units.
  - Fixed an important bug in the normalization of the generated shears.
* Added a DistDeviate class. (#306)
* Added galsim.correlatednoise.get_COSMOS_CorrFunc(...). (#345)
* Added im.addNoiseSNR(). (#349)
* Made a new Noise hierarchy for CCDNoise (no longer a BaseDeviate), 
  GaussianNoise, PoissonNoise, DeviateNoise. (#349)


v0.3
====

* Fixed several bugs in the Sersic class that had been causing ringing. 
  (#319, #330)
* Added support for reading and writing compressed fits files. (#299)
* Added InterpolatedImage class to wrap existing C++ level SBInterpolatedImage. 
  (#333)
* Added a new class structure for representing 2D correlation functions, used 
  to describe correlated noise in images. (#297).
* Add FormattedStr option for string values in config files. (#315)
* Added obj.drawK() to the python layer. (#319)
* Fixed several sources of memory leaks. (#327)
* Updated the moments and PSF correction code to use the Image class and TMV;
  to handle weight and bad pixel maps for the input Images; and to run ~2-3 
  times faster. (#331, #332)
* Fixed bug in config RandomCircle when using inner_radius option.


v0.2
====

* Significant revamping and commenting of demos, including both python and 
  config versions (#243, #285, #289).
* Added python-level int1d function to wrap C++-level integrator, which
  allowed us to remove our dependency on scipy. (#288)
* Significant expansion in config functionality, using YAML/JSON format 
  config files (#291, #295).
* Fixed some bugs in Image handling (including bugs related to duplicate 
  numpy.int32 types), and made Image handling generally more robust (#293, #294).
* Fixed bug in wrapping of angles (now not automatic -- use wrap() explicitly).


v0.1
====

Initial version of GalSim with most of the basic functionality.
