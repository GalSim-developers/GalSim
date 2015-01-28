Changes from v1.2 to v1.3
=========================

New Features
------------

- Updated CorrelatedNoise to work with images that have a non-trivial WCS. (#501)
- Added new methods of the image class to simulate detector effects:
  inter-pixel capacitance.  (#555)
- Added information about PSF size and shape to the data structure that is
  returned by EstimateShear(). (#612)


Deprecated Features
-------------------

- Deprecated CorrelatedNoise.calculateCovarianceMatrix, since it is not used anywhere. (#630)


Bug Fixes and Improvements
--------------------------

- Fixed a bug in UncorrelatedNoise where the variance was set incorrectly. (#630)


Updates to config options
-------------------------

