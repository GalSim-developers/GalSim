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
- Added Spergel(2010) profile GSObject. (#616)
- Enable initializing a DES_PSFEx object using a pyfits HDU directly instead
  of a filename. (#626)

Bug Fixes and Improvements
--------------------------


Updates to config options
-------------------------
