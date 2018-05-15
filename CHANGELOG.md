Changes from v1.6 to v2.0
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
- Removed dependency on boost. (#809)
- Removed dependency on TMV. (#809)
- Added dependency on pybind11. (#809)
- Added dependency on Eigen. (#809)
- FFTW is now the only dependency that pip cannot handle automatically. (#809)
- Officially no longer support Python 2.6. (Pretty sure no one cares.)


API Changes
-----------

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
- The attribute half_light_radius of both InclinedExponential and
  InclinedSersic has been changed to disk_half_light_radius, since it does
  not really correspond to the realized half-light radius of the inclined
  profile (unless the inclination angle is 0 degrees). (#809f)
- Removed galsim_yaml and galsim_json scripts, which were essentially just
  aliases for galsim -f yaml and galsim -f json respectively. (#809f)
- Removed lsst module, which depended on the LSST stack and had gotten quite
  out of sync and broken. (#964)


Bug Fixes
---------


Deprecated Features
-------------------

- Removed all features deprecated in 1.x versions.


New Features
------------

- Added Zernike submodule. (#832, #951)
- Updated PhaseScreen wavefront and wavefront_gradient methods to accept `None`
  as a valid time argument, which means to use the internally stored time in
  the screen(s). (#864)
- Added SecondKick profile GSObject. (#864)
- Updated PhaseScreenPSFs to automatically include SecondKick objects when
  being drawn with geometric photon shooting. (#864)
- Added option to use circular weight function in HSM adaptive moments code.
  (#917)
- Added VonKarman profile GSObject. (#940)
- Added PhotonDCR surface op to apply DCR for photon shooting. (#955)
- Added astropy units as allowed values of wave_type in Bandpass. (#955)
- Added ability to get net pixel areas from the Silicon code for a given flux
  image. (#963)
- Added ability to transpose the meaning of (x,y) in the Silicon class. (#963)
