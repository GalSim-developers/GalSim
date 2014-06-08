Below is a summary of the major changes with each new tagged version of GalSim.
Each version also includes various other minor changes and bug fixes, which are 
not listed here for brevity.  See the CHANGLELOG.md files associated with each 
version for a more complete list.  Issue numbers related to each change are 
given in parentheses.

v1.1
====

Non-backward-compatible API changes
-----------------------------------

* Changed the `xw` and `yw` parameters of the `Pixel` constructor to a
  single `scale` parameter. (#364)
* Added new `Box` class to take up the functionality that had been `Pixel`
  with unequal values of `xw` and `yw`. (#364)
* Changed `Angle.wrap()` to return the wrapped angle rather than modifying the
  original. (#364)
* Changed Bounds methods `addBorder`, `shift`, and `expand` to return new
  Bounds objects rather than changing the original. (#364)
* Merged the GSParams parameters `shoot_relerr` and `shoot_abserr` into the
  parameters `integration_relerr` and `integration_abserr`.  The latter items
  now cover all integrations other than real-space rendering. (#535)

Other changes to the API
------------------------

* Changed the name of the `dx` parameter in various places to `scale`. (#364)
* Combined the old `Image`, `ImageView` and `ConstImageView` arrays of class
  names into a single python layer `Image` class. (#364)
* Changed the methods createSheared, createRotated, etc. to more succinct
  names.  The applyShear, applyRotation, etc. methods are also discouraged
  and will eventually be deprecated. (#511)
* Changed the `setFlux` and `scaleFlux` methods to versions that return new
  objects, rather than change the object in place. (#511)
* Changed the Shapelet.fitImage method to a factory function named
  `FitShapelet` (#511)
* Changed the `nyquistDx` method to `nyquistScale`. (#511)
* Moved as many classes as possible toward an immutable design, meaning that
  we discourage use of setters in various classes that had them. (#511)
* Combined the `draw` and `drawShoot` methods into a single `drawImage` method
  with more options about how the profile should be rendered.  Furthermore, in
  most cases, you no longer need to convolve by a Pixel by hand.  The default
  rendering method will include the pixel convolution for you. (#535)
* Changed the name of `drawK` to `drawKImage` to be more parallel with the
  new `drawImage` name. (#535)

New Features
------------

* Added new set of WCS classes.  See wcs.py and fitswcs.py for details. (#364)
* Every place in the code that can take a `scale` parameter  can now take a
  `wcs` parameter. (#364)
* Added `CelestialCoord` class to represent (ra,dec) coordinates. (#364)
* Added `Bandpass` class to represent throughput functions. (#467)
* Added `SED` class to represent stellar and galactic spectra. (#467)
* Added `ChromaticObject` class to represent wavelength-dependent surface
  brightness profiles. (#467)
* Added `ChromaticAtmosphere` function to easily handle chromatic effects
  due to the atmosphere. (#467)
* Permit users to initialize `OpticalPSF` with a list or array of aberrations,
  as an alternative to specifying each one individually. (#409)
* Added `max_size` optional parameter to OpticalPSF that lets you limit the
  size of the image that it constructs internally. (#478)
* Added option to FitsHeader and FitsWCS to read in SCamp-style text files with
  the header information using the parameter `text_file=True`. (#508)
* Modified addNoiseSNR() method to return the variance of the noise that was
  added. (#526)
* Added `dtype` option to `drawImage` and `drawKImage`, which sets the data
  type to use for automatically constructed images. (#526)

Bug fixes and improvements
--------------------------

* Sped up the gzip and bzip2 I/O. (#344)
* Fixed some bugs in the treatment of correlated noise. (#526, #528)

Updates to config options
-------------------------

* Changed the previous behavior of the `image.wcs` field to allow several WCS
  types: 'PixelScale', 'Shear', 'Jacobian', 'Affine', 'UVFunction',
  'RaDecFunction', 'Fits', and 'Tan'. (#364)
* Changed the name of `sky_pos` to `world_pos`. (#364)
* Removed `pix` top layer in config structure.  Add `draw_method=no_pixel` to
  do what `pix : None` used to do. (#364)
* Added `draw_method=real_space` to try to use real-space convolution. (#364)
* Added ability to index `Sequence` types by any running index, rather than
  just the default by specifying an `index_key` parameter.  The options are
  'obj_num', 'image_num', 'file_num', or 'obj_num_in_file'.  (#364, #536)
* Added `Sum` type for value types for which it makes sense: float, int, angle,
  shear, position. (#457)
* Allowed the user to modify or add config parameters from the command line.
  (#479)
* Added `image.retry_failures` to retry making a GSObject that fails for
  any reason. (#482)
* Added `output.retry_io` item to retry failed write commands. (#482)
* Changed the default sequence indexing for most things to be 'obj_num_in_file'
  rather than 'obj_num'. (#487)
* Added `draw_method=sb` to make this parameter completely consistent with
  the way the `method` parameter for drawImage() now works. (#535)
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
