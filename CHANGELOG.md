Changes from v1.5 to v2.0
=========================

The principal change in GalSim 2.0 is that it is now pip installable.
See the updated INSTALL file for details on how to install GalSim using
either pip or setup.py.

Dependency Changes
------------------

- Added LSSTDESC.Coord, which contains the functionality that used to be in GalSim as the Angle
  and CelestialCoord classes.  We moved it to a separate repo so people could more easily use
  this functionality without requiring all of GalSim as a dependency. (#809b)
- Removed dependency on boost.
- Added dependency on (pybind11 or cffi...)


API Changes
-----------

- Most of the functionality associated with C++-layer objects has been
  redesigned or removed.  These were non-public-API features, so if you have
  been using the public API, you should be fine.  But if you have been relying
  on features of the exposed C++-layer, this might break your code. (#809)
- There were some minor API changes to the Angle and CelestialCoord classes we made when we
  moved it over into LSSTDESC.Coord.  Some were to sever (weak) ties to other GalSim classes and
  some were just deemed API improvements.  Probably the most relevant of these for most
  users will be that Angle.rad is now a property rather than a function.  So the value of an
  angle in radians is just `theta.rad` not `theta.rad()`. (#809b)
- Removed ShapeletSize and FitShapelet from the galsim namespace and made the functionality
  classmethods of the Shapelet class: `Shapelet.size(order)` and `Shapelet.fit(image)`.
  Also LVector is not longer in the galsim.shapelet namespece.  It was an implementation
  detail of Shapelet, which should not be needed for any use case.  (#809e)
- The Interpolant base class can no longer be used as a factory function.  Instead, use
  `Interpolant.from_name(name)`.
- The `SBProfile` attribute of GSObject has changed to `_sbp` and is now officially an
  implementation detail that users should not need to access.  If you think you have a use case
  that is not covered by the public API (either for functionality of efficiency), please open
  an issue.  (#809e)


Bug Fixes
---------



Deprecated Features
-------------------

- Removed all features deprecated in 1.x versions.


New Features
------------



New config features
-------------------

