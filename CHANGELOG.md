Changes from v1.2 to v1.3
=========================

New Features
------------

- Added an option to the ChromaticObject class that allows for image rendering
  via interpolation between stored images.  This option can speed up the image
  rendering process compared to brute force evaluation for chromatic objects
  with basic properties that are wavelength-dependent.  New
  `ChromaticOpticalPSF` class, which allow the diffraction limit and aberrations
  to be wavelength-dependent, can particularly benefit from this
  optimization. (#618)

Bug Fixes and Improvements
--------------------------


Updates to config options
-------------------------

