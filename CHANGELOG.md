Changes from v1.4 to v1.5
=========================

API Changes
-----------

- Simplified the return value of galsim.config.ReadConfig. (#580)


Dependency Changes
------------------



Bug Fixes
---------

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


New config features
-------------------


