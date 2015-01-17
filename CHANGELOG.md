Changes from v1.2 to v1.3
=========================

New Features
------------

- Added new methods of the image class to simulate detector effects:
  nonlinearity and reciprocity failure (#552), inter-pixel capacitance (#555),
  and image quantization (#558).
- Added `InterpolatedChromaticObject` class that can facilitate faster drawing
  compared to brute force for chromatic objects with basic properties that are
  wavelength-dependent (e.g., optical PSFs).  However, it can also be used to
  carry out the brute force comparison for easy accuracy tests.  New
  `ChromaticOpticalPSF` method takes advantage of the
  `InterpolatedChromaticObject` class, allowing the diffraction limit and
  aberrations to be wavelength-dependent. (#618)


Bug Fixes and Improvements
--------------------------

- Switched the sign of the angle returned by `CelestialCoord.angleBetween`.
  The sign is now positive when the angle as seen from the ground sweeps in
  the counter-clockwise direction, which is a more sensible definition than
  what it had used. (#590)


Updates to config options
-------------------------

