Changes from v1.4 to v1.5
=========================

API Changes
-----------

- Simplified the return value of galsim.config.ReadConfig. (#580)


Dependency Changes
------------------



Bug Fixes
---------

- Fixed an error in the magnification calculated by NFWHalo.getLensing(). (#580)
- Fixed bug when whitening noise in images based on COSMOS training datasets
  using the config functionality. (#792)

Deprecated Features
-------------------



New Features
------------

- Added ability to use `numpy`, `np`, or `math` in all places where we evaluate
  user input, including DistDeviate (aka RandomDistribution in config files),
  PowerSpectrum, UVFunction, RaDecFunction, Bandpass, and SED.  Some of these
  had allowed `np.` for numpy commands, but inconsistently, so now they should
  all reliably work with any of these three module names. (#776)
- Added new surface brightness profile, 'InclinedExponential'. This represents
  the 2D projection the 3D profile:
      I(R,z) = I_0 / (2h_s) * sech^2 (z/h_s) * exp(-R/R_s),
  inclined to the line of sight at a desired angle. If face-on (inclination =
  0 degrees), this will be identical to the Exponential profile.  (#782)


New config features
-------------------


