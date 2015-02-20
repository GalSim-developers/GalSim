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

- Changed the implementation of drawing Box and Pixel profiles in real space
  (i.e. without being convolved by anything) to actually draw the surface 
  brightness at the center of each pixel.  This is what all other profiles do,
  but had not been what a Box or Pixel did. (#639)


Updates to config options
-------------------------

