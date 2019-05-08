Below is a summary of the major changes with each new tagged version of GalSim.
Each version may also include various other minor changes and bug fixes not
listed here for brevity.  See the CHANGLELOG.md files associated with each
version for a more complete list.  Issue numbers related to each change are
given in parentheses.



v2.1
====

Deprecated Features
-------------------

- Deprecated PhaseScreenPSF attributes img and finalized. (#990)
- Deprecated GSParams items allowed_flux_variation, small_fraction_of_flux,
  and range_division_for_extreama. (#993)

New Features
------------

- Added RandomWalk profile option. (#821)
- Added spline as LookupTable2D interpolant. (#982)
- Added ability to use an Interpolant in LookupTable and LookupTable2D. (#982)
- Added option for faster grid interpolation of LookupTable2D. (#982)
- Added offset and flux_ratio options to WCS.toWorld and toImage. (#993)

Bug Fixes
---------

- Corrected the diffusion functional form in SiliconSensor. (#981)
- Fixed a bug in the PhaseScreenPSF withGSParams function. (#990)
- Fixed a seg fault bug when PoissonDeviate is given mean=0. (#996)
- Fixed the galsim executable to work correctly when installed by SCons.
- Fixed Convolve and Sum sometimes making unnecessary copies.
- Fixed error when using non-int integer types as seed of BaseDeviate (#1009)
- Fixed error in use of non-integer grid_spacing in PowerSpectrum (#1020)
- Fixed FitsHeader to not unnecessarily read data of fits file. (#1024)
- Switched to yaml.safe_load to avoid PyYAML v5.0 warnings (#1025)
- Fixed cases where numpy objected to subtracting floats from ints. (#1025)


v2.0
====

Installation Changes
--------------------

- Now installable via pip or setup.py install. (#809)

Dependency Changes
------------------

- Officially no longer support Python 2.6 or 3.4. (#755)
- No longer support pre-astropy versions of pyfits or astropy <v1.0 (#755)
- No longer support pre-2016 version of the COSMOS catalog. (#755)
- Added dependency on LSSTDESC.Coord. (#809)
- Removed dependency on boost. (#809)
- Removed dependency on TMV. (#809)
- Added dependency on pybind11 for setup.py installations. (#809)
- Added dependency on Eigen for setup.py installations. (#809)

API Changes
-----------

- Changed the default maximum_fft_size to 8192 from 4096. (#755)
- Changed the order of arguments of galsim.wfirst.allDetectorEffects. (#755)
- Changed how CelestialCoord.project and deproject work. (#809)
- Changed name of InclinedExponential.disk_half_light_radius. (#809)
- Removed galsim_yaml and galsim_json scripts. (#809)
- Removed lsst module, which was broken. (#964)
- Changed how gsparams work for objects that wrap other objects. (#968)

Deprecated Features
-------------------

- Removed all features deprecated in 1.x versions.

New Features
------------

- Changed errors to raise a GalSimError or a subclass thereof. (#755)
- Changed the type of warnings raised by GalSim to GalSimWarning. (#755)
- Added the withGSParams() method for all GSObjects. (#968)


v1.6
====

API Changes
-----------

- Delayed AtmosphericScreen instantiation until its first use. (#864)
- Simplified return values of NFWHalo and PowerSpectrum methods. (#855)
- Simplified return value of LookupTable, SED and Bandpass access. (#955)

Bug Fixes
---------

- Fixed error in amplitude of phase screens created by AtmosphericScreen (#864)
- Fixed a bug in the DES MEDS writer setting the cutout row/col wrong. (#928)
- Fixed some small bugs in complicated uses of config processing. (#928)
- Fixed memory leak when drawing PhaseScreenPSFs using photon-shooting (#942)
- Fixed a few minor bugs in the Silicon code. (#963)
- Fixed a bug in the SED.thin() rel_err value. (#963)

Deprecated Features
-------------------

- Deprecated passing Image arguments to kappaKaiserSquires function. (#855)
- Deprecated the interpolant argument for PowerSpectrum methods getShear,
  getConvergence, getMagnification, and getLensing. (#855)
- Deprecated PowerSpectrum.subsampleGrid. (#855)

New Features
------------

- Added Zernike submodule. (#832, #951)
- Updated PhaseScreen to accept None as a valid time argument. (#864)
- Added SecondKick profile GSObject. (#864)
- Updated PhaseScreenPSFs to use SecondKick with geometric_shooting. (#864)
- Added VonKarman profile GSObject. (#940)
- Added PhotonDCR surface op. (#955)
- Added astropy units as allowed values of wave_type in Bandpass. (#955)
- Added SiliconSensor.calculate_pixel_area. (#963)
- Added transpose option in SiliconSensor. (#963)


v1.5
====

API Changes
-----------

- Simplified the return value of galsim.config.ReadConfig. (#580)
- Changed return type of RealGalaxyCatalog.getGal and getPSF. (#640)
- Reorganized files in share/galsim directory. (#640)
- Changed SED objects to have real dimensions. (#789)
- Changed drawKImage to return a single ImageCD. (#799)
- Changed InterpolatedKImage to take an ImageCD. (#799)
- Dynamic PhaseScreenPSFs require an explicit start time and time step. (#824)
- OpticalScreen now requires diam argument. (#824)
- Switched galsim.Image(image) to make a copy rather than a view. (#873)
- Changed the behavior of RealGalaxyCatalog.preload (#884)

Dependency Changes
------------------

- Added astropy as a required dependency for chromatic functionality. (#789)
- Switched scons tests test runner from nosetests to pytest. (#892)

Bug Fixes
---------

- Fixed parity mistake in configuration of WFIRST focal plane. (#675)
- Fixed an error in the magnification calculated by NFWHalo.getLensing(). (#580)
- Fixed bug when whitening noise in images based on COSMOS training datasets
  using the config functionality. (#792)
- Fixed bug in image.subImage that could cause seg faults in some cases. (#848)
- Fixed bug in GSFitsWCS that made toImage sometimes fail to converge. (#880)
- Fixed bug that could cause Kolmogorov to go into an endless loop. (#952)

Deprecated Features
-------------------

- Deprecated simReal function. (#787)
- Deprecated Chromatic class. (#789)
- Deprecated .copy() methods for immutable classes, including GSObject,
  ChromaticObject, SED, and Bandpass. (#789)
- Deprecated wmult parameter of drawImage. (#799)
- Deprecated Image.at method. (#799)
- Deprecated gain parameter of drawKImage.  (#799)
- Deprecated ability to create multiple PhaseScreenPSFs with single call
  to makePSF. (#824)
- Deprecated the use of np.trapz and galsim.integ.mipdt as valid
  integration rules for use by ImageIntegrators. (#887)
- Changed the Angle.rad method to a property. (#904)
- Deprecated the functions HMS_Angle and DMS_Angle. (#904)
- Deprecated the function ShapeletSize and FitShapelet. (#904)
- Deprecated using Interpolant base class as a factory function. (#904)
- Deprecated use of the SBProfile attribute of GSObject. (#904)
- Deprecated making a GSObject directly. (#904)
- Deprecated use of the image attribute of Image. (#904)
- PhotonArray.addTo(image) now takes a regular galsim.Image argument. (#904)
- Deprecated the various PhotonArray.get* functions. (#904)
- Deprecated calculateFlux(bandpass=None). (#905)
- Deprecated the various get* methods that are equivalent to a property.
  e.g. obj.getFlux() -> obj.flux, etc. (#904)
- Deprecated ChromaticObject.obj.  (#904)
- Changed the objlist attribute of ChromaticSum and ChromaticConvolution to
  obj_list. (#904)
- Deprecated OpticalScreen.coef_array. (#904)
- Changed a number of GSObject methods to properties. (#904)
    - obj.stepK() -> obj.stepk
    - obj.maxK() -> obj.maxk
    - obj.nyquistScale() -> obj.nyquist_scale
    - obj.centroid() -> obj.centroid
    - obj.getPositiveFlux() -> obj.positive_flux
    - obj.getNegativeFlux() -> obj.negative_flux
    - obj.maxSB() -> obj.max_sb
    - obj.isAxisymmetric() -> obj.is_axisymmetric
    - obj.isAnalyticX() -> obj.is_analytic_x
    - obj.isAnalyticK() -> obj.is_analytic_k
    - obj.hasHardEdges() -> obj.has_hard_edges
- Renamed ChromaticObject.centroid(bandpass) to calculateCentroid. (#904)
- Changed a few Image methods to properties. (#904)
    - image.center() -> image.center
    - image.trueCenter() -> image.true_center
    - image.origin() -> image.origin

New Features
------------

- Added DeltaFunction. (#533)
- Added ChromaticRealGalaxy. (#640)
- Added CovarianceSpectrum. (#640)
- Added HST bandpasses covering AEGIS and CANDELS surveys (#640)
- Added drawKImage method for ChromaticObject and CorrelatedNoise (#640)
- Updated WFIRST WCS some other basic numbers to Cycle 7 design. (#675)
- Added support for unsigned int Images. (#715)
- Added a new Sensor class hierarchy, including SiliconSensor. (#722)
- Added save_photons option to drawImage. (#722)
- Added image.bin and image.subsample methods. (#722)
- Added annular Zernike option for optical aberration coefficients. (#771)
- Added ability to use numpy, np, or math in all places where we evaluate
  user input. (#776)
- Added keywords exptime and area to drawImage(). (#789)
- Added ability to use astropy.units for units of SEDs. (#789).
- Added InclinedExponential and InclinedSersic. (#782, #811)
- Added ability to select from a RealGalaxyCatalog or COSMOSCatalog using
  the 'weight' entries to account for selection effects. (#787)
- Added complex Image dtypes (aka ImageCD and ImageCF). (#799, #873)
- Added maxSB() method to GSObjects. (#799)
- Added im[x,y] = value and value = im[x,y] syntax. (#799)
- Added ability to do FFTs directly on images. (#799)
- Added galsim.RandomWalk. (#819)
- Added generate function to BaseDeviate and sed.sampleWavelength. (#822)
- Added function assignPhotonAngles (#823)
- Added geometric optics approximation for photon-shooting PhaseScreenPSFs.
  (#824)
- Added gradient method to LookupTable2D. (#824)
- Added surface_ops option to drawImage function. (#827)
- Added ii_pad_factor kwarg to PhaseScreenPSF and OpticalPSF. (#835)
- Added galsim.fft module. (#840)
- Added a hook to the WCS classes to allow them to vary with color. (#865)
- Added optional variance parameter to PowerSpectrum.buildGrid. (#865)
- Added CelestialCoord.get_xyz() and CeletialCoord.from_xyz(). (#865)
- Added an optional center argument for Angle.wrap(). (#865)
- Added recenter option to drawKImage. (#873)
- Added option to use circular weight function in HSM moments. (#917)

New config features
-------------------

- Changed galsim.config.CalculateNoiseVar to CalculateNoiseVariance. (#820)
- Setting config['rng'] is no longer required when manually running commands
  like galsim.config.BuildGSObject.  (#820)
- Allow PoissonNoise and CCDNoise without any sky level. (#820)
- Let 'None' in the config file mean None. (#820)
- Remove default value for 'max_extra_noise' for photon shooting. (#820)
- Added --except_abort option to galsim executable. (#820)
- Added optional probability parameter 'p' for Random bool values. (#820)
- Added ability to specify world_pos in celestial coordinates. (#865)
- Added the ability to have multiple rngs. (#865)
- Added ngrid, center, variance, index options to power_spectrum input field.
  (#865)
- Added skip option in stamp field. (#865)
- Added ':field' syntax for templates. (#865)


v1.4
====

API Changes
-----------

- Changed the galsim.Bandpass and galsim.SED classes to require formerly
  optional keywords wave_type and flux_type. (#745)

Dependency Changes
------------------

- Added future module as a dependency. (#534)
- Changed PyYAML to a non-optional dependency. (#768)

Bug Fixes
---------

- Improved ability of galsim.fits.read to handle invalid FITS headers. (#602)
- Fixed bug in des module, building meds file with wcs from input images. (#654)
- Fixed a bug in the way Images are instantiated for certain combinations of
  ChromaticObjects and image-setup keyword arguments (#683)
- Added ability to manipulate the width of the moment-measuring weight function
  for the KSB shear estimation method of the galsim.hsm package. (#686)
- Fixed an error in the CCDNoise.getVariance() function. (#713)
- Fixed an assert failure in InterpolatedImage if image is all zeros. (#720)
- Updated ups table file so that setup command is setup galsim. (#724)
- Improved algorithm for thinning SEDs and Bandpasses. (#739)
- Fixed a bug in how DistDeviate handles nearly flat pdfs. (#741)
- Fixed a bug in chromatic parametric COSMOS galaxy models. (#745)
- Fixed a bug in the Sum and Convolution constructors when list has only a
  single element. (#763)
- Fixed a bug related to boost-v1.60 python shared_ptr registration. (#764)
- Changed an assert in the HSM module to an exception. (#784)

Deprecated Features
-------------------

- Deprecated the gal.type=Ring option in the config files. (#698)

New Features
------------

- Added OutputCatalog class. (#301, #691)
- Added methods calculateHLR, calculateMomentRadius, and calculateFWHM. (#308)
- Added LookupTable2D. (#465)
- Added support for Python 3. (#534)
- Added AtmosphericScreen, OpticalScreen, and PhaseScreenList. (#549)
- Added PhaseScreenPSF. (#549)
- Added Atmosphere function. (#549)
- Rewrote OpticalPSF using new PhaseScreen framework. (#549)
- Extended OpticalPSF to handle arbitrary Zernike order. (#549)
- Added a simple, linear model for persistence. (#554)
- Added BoundsI.numpyShape(). (#654)
- Enabled FITS files with unsigned integer to read as ImageI or ImageS. (#654)
- Made COSMOSCatalog write an index parameter. (#654, #694)
- Added ability to specify lambda and r0 separately for Kolmogorov. (#657)
- Enabled initializing an InterpolatedImage from a user-specified HDU. (#660)
- Changed galsim.fits.writeMulti to allow hdus in "image" list. (#691)
- Added wcs argument to Image.resize(). (#691)
- Added BaseDeviate.discard(n) and BaseDeviate.raw(). (#691)
- Added sersic_prec option to COSMOSCatalog.makeGalaxy(). (#691)
- Enabled image quality cuts in the COSMOSCatalog class. (#693)
- Added convergence_threshold in HSMParams. (#709)
- Improved the readability of Image and BaseDeviate reprs. (#723)
- Sped up Bandpass, SED, and LookupTable classes. (#735)
- Added the FourierSqrt operator. (#748)
- Made Bandpass.thin() and truncate() preserve the zeropoint. (#711)
- Added version information to the compiled C++ library. (#750)

Updates to galsim executable
----------------------------

- Dropped default verbosity from 2 to 1. (#691)
- Added galsim -n njobs -j jobnum to split run into multiple jobs. (#691)
- Added galsim -p to perform profiling on the run. (#691)

New config features
-------------------

- Added ability to write truth catalogs using output.truth field. (#301, #691)
- Improved the extensibility of the config parsing. (#691, #774)
- Added the 'template' option. (#691)
- Made '$' and '@' shorthand for Eval and Current. (#691)
- Allowed gsobjects to be referenced from Current types. (#691)
- Added x,f specification for a RandomDistribution. (#691)
- Added a new 'stamp' top level field. (#691)
- Added new stamp type=Ring to effect ring tests. (#698)


v1.3
====

Installation Changes
--------------------

- Require functionality in TMV 0.72. (#616)

API Changes
-----------

- Changed the name of the bounds.addBorder() method to withBorder. (#218)
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
- Added new ChromaticAiry and ChromaticOpticalPSF classes for representing
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

- Fixed a bug in the normalization of SEDs that use wave_type='A'. (#218)
- Switched the sign of the angle returned by CelestialCoord.angleBetween.
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
- Added intermediate results cache to ChromaticObject.drawImage and
  ChromaticConvolution.drawImage to speed up the rendering of groups
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
- Renamed alias_threshold to folding_threshold. (#562)
- Extended to the rotate, shear, and transform methods of ChromaticObject
  the ability to take functions of wavelength for the arguments. (#581)
- Added cdmodel module to describe charge deflection in CCD pixels. (#524)
- Added pupil_plane_im option to OpticalPSF. (#601)
- Added nx, ny, and bounds keywords to drawImage() and drawKImage()
  methods. (#603)

Bug Fixes and Improvements
--------------------------

- Improved efficiency of noise generation by correlated noise models. (#563)
- Modified BoundsI and PositionI initialization to ensure that integer elements
  in NumPy arrays with dtype==int are handled without error. (#486)
- Changed the default seed used for Deviate objects when no seed is given to
  use /dev/urandom if it is available. (#537)
- Changed SED and Bandpass methods to preserve type when returning a new object
  when possible. (#547)
- Made file_name argument to CorrelatedNoise.getCOSMOSNoise() be able
  to have a default value in the repo. (#548)
- Fixed the dtype= kwarg of Image constructor on some platforms. (#571)
- Added workaround for bug in pyfits 3.0 in galsim.fits.read. (#572)
- Fixed a bug in the Image constructor when passed a NumPy array with the
  opposite byteorder as the system native one. (#594)
- Fixed bug that prevented calling LookupTables on non-square 2d arrays. (#599)
- Updated the code to account for a planned change in NumPy 1.9. (#604)
- Fixed a bug where the dtype of an Image could change when resizing. (#604)
- Defined a hidden __version__ attribute according to PEP 8 standards. (#610)

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

- Changed Pixel to take a single scale parameter. (#364)
- Added new Box class. (#364)
- Changed Angle.wrap() to return the wrapped angle. (#364)
- Changed Bounds methods addBorder, shift, and expand to return new
  Bounds objects. (#364)
- Merged the GSParams parameters shoot_relerr and shoot_abserr into the
  parameters integration_relerr and integration_abserr. (#535)

Other changes to the API
------------------------

- Changed the name of the dx parameter in various places to scale. (#364)
- Combined the old Image, ImageView and ConstImageView arrays of class
  names into a single python layer Image class. (#364)
- Changed the methods createSheared, createRotated, etc. to more succinct
  names shear, rotate, etc. (#511)
- Changed the setFlux and scaleFlux methods to return new objects. (#511)
- Changed the Shapelet.fitImage method to FitShapelet (#511)
- Changed the nyquistDx method to nyquistScale. (#511)
- Moved as many classes as possible toward an immutable design. (#511)
- Combined the draw and drawShoot methods into a single drawImage method
  with more options about how the profile should be rendered. (#535)
- Changed the name of drawK to drawKImage. (#535)

New Features
------------

- Added new set of WCS classes. (#364)
- Added CelestialCoord class to represent (ra,dec) coordinates. (#364)
- Added Bandpass, SED, and ChromaticObject classes. (#467)
- Added aberrations parameter of OpticalPSF. (#409)
- Added max_size parameter to OpticalPSF. (#478)
- Added text_file parameter to FitsHeader and FitsWCS. (#508)
- Modified addNoiseSNR() method to return the added variance. (#526)
- Added dtype option to drawImage and drawKImage. (#526)

Bug fixes and improvements
--------------------------

- Sped up the gzip and bzip2 I/O. (#344)
- Fixed some bugs in the treatment of correlated noise. (#526, #528)

Updates to config options
-------------------------

- Added more options for image.wcs field. (#364)
- Changed the name of sky_pos to world_pos. (#364)
- Removed pix top layer in config structure.  Add draw_method=no_pixel to
  do what pix : None used to do. (#364)
- Added draw_method=real_space to try to use real-space convolution. (#364)
- Added ability to index Sequence types by any running index. (#364, #536)
- Added Sum type for value types for which it makes sense. (#457)
- Allowed modification of config parameters from the command line. (#479)
- Added image.retry_failures. (#482)
- Added output.retry_io item to retry failed write commands. (#482)
- Changed the default sequence indexing for most things to be 'obj_num_in_file'
  rather than 'obj_num'. (#487)
- Added draw_method=sb. (#535)
- Changed the output.psf.real_space option to output.psf.draw_method
  and allow all of the options that exist for image.draw_method. (#535)
- Added an index item for Ring objects. (#536)


v1.0
====

Notable bug fixes and improvements
----------------------------------

- Fixed bug in the rendering of shifted images. (#424)
- Improved the fidelity of the Lanczos conserve_dc=True option. (#442)
- Switched default interpolant for RealGalaxy to Quintic, since it was
  found to be more accurate in general. (#442)
- Fixed a bug in InterpolatedImage calculateStepK function. (#454)
- Fixed a bug in Image class resize function. (#461)
- Sped up OpticalPSF and RealGalaxy significantly. (#466, #474)
- Fixed a bug in the fourier rendering of truncated Sersic profiles. (#470)
- Fixed some bugs in the config machinery when files have varying numbers
  of objects. (#487)
- Support astropy.io.fits in lieu of stand-alone pyfits module. (#488)
- Fixed a bug in config where 'safe' objects were not being correctly
  invalidated when a new input item should have invalidated them.
- Fixed a bug in the drawing of a Pixel all by itself. (#497)

New features
------------

- Added galsim executable (instead of galsim_yaml, galsim_json). (#460)
- Updated the allowed range for Sersic n to 0.3 -- 6.2. (#325)
- Made RealGalaxy objects keep track of their (correlated) noise. (#430)
- Changed noise padding options for RealGalaxy and InterpolatedImage. (#430)
- Added VariableGaussianNoise class. (#430)
- Added offset parameter to both draw and drawShoot. (#439)
- Changed the name of InputCatalog to just Catalog. (#449)
- Added Dict class. (#449)
- Added MEDS file output to des module. (#376)
- Removed des module from default imports of GalSim.  Now need to import
  galsim.des explicitly or load with galsim -m des ... (#460)

Updates to config options
-------------------------

- Added RealGalaxyOriginal galaxy type. (#389)
- Added whiten option for RealGalaxy objects. (#430)
- Added Current type. (#430)
- Added offset option in image field. (#449)
- Added the ability to have multiple input catalogs, dicts, etc. (#449)
- Added signal_to_noise option for PSFs when there is no galaxy. (#459)
- Added output.skip and output.noclobber options. (#474)


v0.5
====

New features
------------

- Added Shapelet class. (#350)
- Added ability to truncate Sersic profiles. (#388)
- Added trefoil and struts to OpticalPSF. (#302, #390)
- Updates to lensing engine:
  - Added document describing how lensing engine code works. (#248)
  - Added ability to draw (gamma,kappa) from same power spectrum. (#304)
  - Added kmin_factor and kmax_factor parameters to buildGrid. (#377)
  - Added PowerSpectrumEstimator class in pse module. (#382)
- Added GSParams (#343, #426) and HSMParams (#365) classes.
- Added des module and example scripts. (#350)
- Added applyWhiteningTo method to CorrelatedNoise class. (#352)
- Changed the default centering convention for even-sized images to be in the
  actual center, rather than 1/2 pixel off-center. (#380)
- Enabled InputCatalog to read FITS catalogs. (#350)
- Added FitsHeader class and config option. (#350)
- Added the ability to read/write to a specific HDU. (#350)
- Add new function galsim.fits.writeFile. (#417)
- Added LINKFLAGS SCons option. (#380)

Updates to config
-----------------

- Added index_convention option. (#380)
- Changed the name of the center item for the Scattered image type to
  image_pos, and added a new sky_pos item. (#380)

Bug fixes
---------

- Fix some errors related to writing to an HDUList. (#417)
- Fixed ringing when Sersic objectss were drawn with FFTs. (#426)
- Fixed bugs in obj.drawK() function. (#407)
- Fixed bugs with InterpolatedImage objects. (#389, #432)
- Fixed bug in draw routine for shifted objects. (#380)
- Fixed bug in the generation of correlated noise fields. (#352)


v0.4
====

- Added ability to pad images for InterpolatedImage or RealGalaxy with either
  correlated or uncorrelated noise. (#238)
- Added python-level LookupTable class which wraps the existing C++ Table
  class. (#305)
- Lensing engine updates: (#305)
  - Added the option of drawing shears from a tabulated P(k)
  - Added ability to handle conversions between different angular units.
  - Fixed an important bug in the normalization of the generated shears.
- Added a DistDeviate class. (#306)
- Added galsim.correlatednoise.get_COSMOS_CorrFunc(...). (#345)
- Added im.addNoiseSNR(). (#349)
- Made a new Noise hierarchy for CCDNoise (no longer a BaseDeviate),
  GaussianNoise, PoissonNoise, DeviateNoise. (#349)


v0.3
====

- Fixed several bugs in the Sersic class that had been causing ringing.
  (#319, #330)
- Added support for reading and writing compressed fits files. (#299)
- Added InterpolatedImage class to wrap existing C++ level SBInterpolatedImage.
  (#333)
- Added a new class structure for representing 2D correlation functions, used
  to describe correlated noise in images. (#297).
- Add FormattedStr option for string values in config files. (#315)
- Added obj.drawK() to the python layer. (#319)
- Fixed several sources of memory leaks. (#327)
- Updated the moments and PSF correction code to use the Image class and TMV;
  to handle weight and bad pixel maps for the input Images; and to run ~2-3
  times faster. (#331, #332)
- Fixed bug in config RandomCircle when using inner_radius option.


v0.2
====

- Significant revamping and commenting of demos, including both python and
  config versions (#243, #285, #289).
- Added python-level int1d function to wrap C++-level integrator, which
  allowed us to remove our dependency on scipy. (#288)
- Significant expansion in config functionality, using YAML/JSON format
  config files (#291, #295).
- Fixed some bugs in Image handling (including bugs related to duplicate
  numpy.int32 types), and made Image handling generally more robust (#293, #294).
- Fixed bug in wrapping of angles (now not automatic -- use wrap() explicitly).


v0.1
====

Initial version of GalSim with most of the basic functionality.
