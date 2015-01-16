Changes from v1.2 to v1.3
=========================

New Features
------------

- Added `InterpolatedChromaticObject` class that can facilitate faster drawing
  compared to brute force for chromatic objects with basic properties that are
  wavelength-dependent (e.g., optical PSFs).  However, it can also be used to
  carry out the brute force comparison for easy accuracy tests.  New
  `ChromaticOpticalPSF` method takes advantage of the
  `InterpolatedChromaticObject` class, allowing the diffraction limit and
  aberrations to be wavelength-dependent. (#618)

Bug Fixes and Improvements
--------------------------


Updates to config options
-------------------------

