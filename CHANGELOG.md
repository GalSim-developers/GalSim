Changes from v1.2 to v1.3
=========================

Installation Changes
--------------------

- We have officially claimed to require TMV version 0.72 or later since
  GalSim version 1.1.  However, TMV 0.71 still worked for most users.
  With this release, we make use of features that are not in TMV 0.71, so
  you really do need to upgrade to version 0.72 now. (#616)


API Changes
-----------

- Changed the name of the `bounds.addBorder()` method to `withBorder` to make
  it clearer that the method returns a new Bounds object. (#218)
- Removed (from the python layer) Interpolant2d and InterpolantXY, since we
  we have only actually implemented 1-d interpolants. (#218)
- Removed the MultipleImage way of constructing an SBInterpolatedImage, since
  we do not use it anywhere, nor are there unit tests for it.  The C++ layer
  still has the implementation of it if we ever find a need for it. (#218)
- Made the default tolerance for all Interpolants equal to 1.e-4.  It already
  was for Cubic, Quintic, and Lanczos, which are the ones we normally use,
  so this just changes the default for the others. (#218)
- Deprecated the __rdiv__ operator from Bandpass and SED, since we realized
  that they did not have any clear use cases.  If you have one, please open an
  issue, and we can add them back. (#218)
- Made all returned matrices consistently use numpy.array, rather than
  numpy.matrix.  We had been inconsistent with which type different functions
  returned, so now all matrices are numpy.arrays.  If you rely on the 
  numpy.matrix behavior, you should recast with numpy.asmatrix(M). (#218)
- Deprecated CorrelatedNoise.calculateCovarianceMatrix, since it is not used 
  anywhere, and we think no one really has a need for it. (#630)
- Officially deprecated the methods and functions that had been described as
  having been removed or changed to a different name.  In fact, many of them 
  had been still valid, but no longer documented.  This was intentional to
  allow people time to change their code.  Now these methods are officially
  deprecated and will emit a warning message if used. (#643)
- Made the classes PositionI, PositionD, GSParams, and HSMParams immutable.
  It was an oversight that we failed to make them immutable in version 1.1 when
  we made most other GalSim classes immutable.  Now rather than write to their
  various attributes, you should make a new object. e.g. instead of `p.x = 4`
  and `p.y = 5`, you now need to do `p = galsim.PositionD(4,5)`. (#218, #643)


New Features
------------

- Made all GalSim objects picklable unless they use fundamentally unpicklable
  things such as lambda expressions.  Also gave objects better str and repr
  representations (str(obj) should be descriptive, but relatively succinct, 
  and repr(obj) should be unambiguous).  Also made __eq__, __ne__, and __hash__
  work better. (#218)
- Added ability to set the zeropoint of a bandpass on construction.  (Only
  a numeric value; more complicated calculations still need to use the method
  `bandpass.withZeropoint()`.) (#218)
- Added ability to set the redshift of an SED on construction. (#218)
- Updated CorrelatedNoise to work with images that have a non-trivial WCS. 
  (#501)
- Added new methods of the image class to simulate detector effects:
  inter-pixel capacitance (#555) and image quantization (#558).
- Enable constructing a FitsHeader object from a dict.  This had been a hidden
  feature that only half worked.  It should now work correctly.  Also, allow
  FitsHeader to be default constructed, which creates it with no keys. (#590)
- Added a module, galsim.wfirst, that includes information about the planned
  WFIRST mission, along with helper routines for constructing appropriate PSFs,
  bandpasses, WCS, etc.  (#590)
- Added native support for the TAN-SIP WCS type using GSFitsWCS. (#590)
- Added a helper program, galsim_download_cosmos, that downloads the COSMOS
  RealGalaxy catalog and places it in the $PREFIX/share/galsim installation
  directory, and allow this to be the default for RealGalaxyCatalog(). (#590)
- Added new class with methods for making realistic galaxy samples using COSMOS:
  the COSMOSCatalog class and its method makeObj(). (#590 / #635).
- Added information about PSF size and shape to the data structure that is
  returned by EstimateShear(). (#612)
- Added Spergel(2010) profile GSObject (#616).
- Added an option to the ChromaticObject class that allows for image rendering
  via interpolation between stored images.  This option can speed up the image
  rendering process compared to brute force evaluation for chromatic objects
  with basic properties that are wavelength-dependent. (#618)
- Added new `ChromaticAiry` and `ChromaticOpticalPSF` classes for representing
  optical PSFs.  These new classes allow the diffraction effects and (in the 
  latter case) aberrations to be wavelength-dependent. (#618)
- Enable initializing a DES_PSFEx object using a pyfits HDU directly instead
  of a filename. (#626)
- Added TopHat class implementing a circular tophat profile. (#639)
- Added ability of Noise objects to take a new random number generator (a
  BaseDeviate instance) when being copied. (#643)


Bug Fixes and Improvements
--------------------------

- Fixed a bug in the normalization of SEDs that use `wave_type='A'`. (#218)
- Switched the sign of the angle returned by `CelestialCoord.angleBetween`.
  The sign is now positive when the angle as seen from the ground sweeps in
  the counter-clockwise direction, which is a more sensible definition than
  what it had used. (#590)
- Fixed a bug in UncorrelatedNoise where the variance was set incorrectly.
  (#630)
- Changed the implementation of drawing Box and Pixel profiles in real space
  (i.e. without being convolved by anything) to actually draw the surface 
  brightness at the center of each pixel.  This is what all other profiles do,
  but had not been what a Box or Pixel did. (#639)
- Fixed a bug where InterpolatedImage and Box profiles were not correctly
  rendered when transformed by something that includes a flip. (#645)
- Fixed a bug in rendering profiles that involve two separate shifts. (#645)
- Fixed a bug if drawImage was given odd nx, ny parameters, the drawn profile
  was not correctly centered in the image. (#645)


Updates to config options
-------------------------

- Added Spergel type. (#616)
- Added lam, diam, scale_units options to Airy and OpticalPSF types. (#618)
- Added TopHat type. (#639)

