Changes from v1.2 to v1.3
=========================


API Changes
-----------

- Made the Position classes immutable.  It was an oversight that we failed
  to make it immutable in version 1.1 when we made most other GalSim classes
  immutable.  Now rather than `p.x = 4` and `p.y = 5`, you would need to do 
  `p = galsim.PositionD(4,5)`. (#643)

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
