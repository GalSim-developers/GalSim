Changes from v1.5 to v2.0
=========================

The principal change in GalSim 2.0 is that it is now pip installable.
See the updated INSTALL file for details on how to install GalSim using
either pip or setup.py.

Dependency Changes
------------------

- Added LSSTDESC.Coord, which contains the functionality that used to be in
  GalSim as the Angle and CelestialCoord classes.  We moved it to a separate
  repo so people could more easily use this functionality without requiring all
  of GalSim as a dependency. (#809b)
- Removed dependency on boost.
- Added dependency on (pybind11 or cffi...)


API Changes
-----------

- Removed all features deprecated in 1.x versions.
- Most of the functionality associated with C++-layer objects has been
  redesigned or removed.  These were non-public-API features, so if you have
  been using the public API, you should be fine.  But if you have been relying
  on features of the exposed C++-layer, this might break your code. (#809)
- There were some minor API changes to the Angle and CelestialCoord classes we
  made when we moved it over into LSSTDESC.Coord.  Some were to sever (weak)
  ties to other GalSim classes and some were just deemed API improvements.
  Most of these were already deprecated in v1.5.  The ones that we were not
  able to deprecate (and preserve the existing functionality) in advance of
  v2.0 are the `CelestialCoord.project` and `deproject` functions.  The new
  functionality has better units handing (taking and returning Angles rather
  then PositionD instances).  If you have been using these functions, you
  should check the new doc strings for the appropriate types and units for the
  parameters and return values. (#809b)
- The return type of a LookupTable when given a list or tuple input is now a
  numpy array rather than a list or tuple. (#809e)
- The return type of Bandpass and SED calls when given a list or tuple input
  is also now a numpy array. (#809e)
- Similarly, the output of getShear, getConvergence and similar methods of
  NFWHalo and PowerSpectrum are always either scalars or numpy arrays. (#809e)


Bug Fixes
---------



Deprecated Features
-------------------



New Features
------------



New config features
-------------------

