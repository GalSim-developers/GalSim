Changes from v1.2 to v1.3
=========================

New Features
------------

- Added new methods of the image class to simulate detector effects:
  inter-pixel capacitance (#555) and image quantization (#558).
- Added information about PSF size and shape to the data structure that is
  returned by EstimateShear(). (#612)
- Enable initializing a DES_PSFEx object using a pyfits HDU directly instead
  of a filename. (#626)

Bug Fixes and Improvements
--------------------------


Updates to config options
-------------------------
