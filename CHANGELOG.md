Changes from v1.4 to v1.5
=========================

API Changes
-----------



Dependency Changes
------------------



Bug Fixes
---------

- Fixed bug when whitening noise in images based on COSMOS training datasets using the config
  functionality. (#792)

Deprecated Features
-------------------



New Features
------------

- Added new light distribution 'InclinedExponential' (#782). This represents the 2D projection of
  the 3D profile I(R,z) = I_0 / (2h_s) * sech^2 (z/h_s) * exp(-R/R_s), inclined to the line of
  sight at a desired angle. If face-on (inclination = 0 degrees), this will be identical to the
  Exponential profile. 

New config features
-------------------


