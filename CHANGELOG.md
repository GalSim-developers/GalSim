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

- Officially deprecated the methods and functions that had been described as
  having been removed or changed to a different name.  In fact, many of them 
  had been still valid, but no longer documented.  This was intentional to
  allow people time to change their code.  Now these methods are officially
  deprecated and will emit a warning message if used. (#643)
- Made the classes PositionI, PositionD, and GSParams immutable.  It was an
  oversight that we failed to make them immutable in version 1.1 when we made
  most other GalSim classes immutable.  Now rather than write to their various
  attributes, you should make a new object. e.g. instead of `p.x = 4` and
  `p.y = 5`, you now need to do `p = galsim.PositionD(4,5)`. (#643)


New Features
------------

- Updated CorrelatedNoise to work with images that have a non-trivial WCS. (#501)
- Added new methods of the image class to simulate detector effects:
  inter-pixel capacitance (#555) and image quantization (#558).
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


Deprecated Features
-------------------

- Deprecated CorrelatedNoise.calculateCovarianceMatrix, since it is not used anywhere. (#630)


Bug Fixes and Improvements
--------------------------

- Fixed a bug in UncorrelatedNoise where the variance was set incorrectly. (#630)
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

- Added TopHat type. (#639)

