Changes from v1.5 to v2.0
=========================

Dependency Changes
------------------

- Added LSSTDESC.Coord. (#809b)


API Changes
-----------

- There were some minor API changes to the Angle and CelestialCoord classes we made when we
  moved it over into LSSTDESC.Coord.  Some to sever (weak) ties to other GalSim classes and
  some that were just deamed API improvements.  Probably the most relevant of these for most
  users will be that Angle.rad is now a property rather than a function.  So the value of an
  angle in radians is just `theta.rad` not `theta.rad()`. (#809b)


Bug Fixes
---------



Deprecated Features
-------------------

- Removed all features deprecated in 1.x versions.


New Features
------------



New config features
-------------------

