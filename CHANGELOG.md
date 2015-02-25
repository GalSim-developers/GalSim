Changes from v1.2 to v1.3
=========================


API Changes
-----------

- Made the classes PositionI, PositionD, and GSParams immutable.  It was an
  oversight that we failed to make them immutable in version 1.1 when we made
  most other GalSim classes immutable.  Now rather than write to their various
  attributes, you should make a new object. e.g. instead of `p.x = 4` and
  `p.y = 5`, you now need to do `p = galsim.PositionD(4,5)`. (#643)

New Features
------------

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


Bug Fixes and Improvements
--------------------------

- Changed the implementation of drawing Box and Pixel profiles in real space
  (i.e. without being convolved by anything) to actually draw the surface 
  brightness at the center of each pixel.  This is what all other profiles do,
  but had not been what a Box or Pixel did. (#639)


Updates to config options
-------------------------

- Added TopHat type. (#639)

