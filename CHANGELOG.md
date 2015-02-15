Changes from v1.2 to v1.3
=========================


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
- Added new `ChromaticOpticalPSF` class, which allows the diffraction effects
  and aberrations to be wavelength-dependent. (#618)
- Enable initializing a DES_PSFEx object using a pyfits HDU directly instead
  of a filename. (#626)

Bug Fixes and Improvements
--------------------------


Updates to config options
-------------------------
