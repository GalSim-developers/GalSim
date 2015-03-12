Changes from v1.2 to v1.3
=========================


New Features
------------

- Updated CorrelatedNoise to work with images that have a non-trivial WCS. (#501)
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


Deprecated Features
-------------------

- Deprecated CorrelatedNoise.calculateCovarianceMatrix, since it is not used
  anywhere. (#630)


Bug Fixes and Improvements
--------------------------

- Switched the sign of the angle returned by `CelestialCoord.angleBetween`.
  The sign is now positive when the angle as seen from the ground sweeps in
  the counter-clockwise direction, which is a more sensible definition than
  what it had used. (#590)
- Fixed a bug in UncorrelatedNoise where the variance was set incorrectly. (#630)
- Changed the implementation of drawing Box and Pixel profiles in real space
  (i.e. without being convolved by anything) to actually draw the surface 
  brightness at the center of each pixel.  This is what all other profiles do,
  but had not been what a Box or Pixel did. (#639)


Updates to config options
-------------------------

- Added TopHat type. (#639)

