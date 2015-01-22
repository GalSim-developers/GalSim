Changes from v1.2 to v1.3
=========================

New Features
------------

- Added new methods of the image class to simulate detector effects:
  inter-pixel capacitance (#555) and image quantization (#558).
- Enable constructing a FitsHeader object from a dict.  This had been a hidden
  feature that only half worked.  It should now work correctly.  Also, allow
  FitsHeader to be default constructed, which creates it with no keys. (#590)
- Added an option to the ChromaticObject class that allows for image rendering
  via interpolation between stored images.  This option can speed up the image
  rendering process compared to brute force evaluation for chromatic objects
  with basic properties that are wavelength-dependent.  New
  `ChromaticOpticalPSF` class, which allow the diffraction limit and aberrations
  to be wavelength-dependent, can particularly benefit from this
  optimization. (#618)
- Added information about PSF size and shape to the data structure that is
  returned by EstimateShear(). (#612)


Bug Fixes and Improvements
--------------------------

- Switched the sign of the angle returned by `CelestialCoord.angleBetween`.
  The sign is now positive when the angle as seen from the ground sweeps in
  the counter-clockwise direction, which is a more sensible definition than
  what it had used. (#590)


Updates to config options
-------------------------

