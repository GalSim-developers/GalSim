Changes from v1.6 to v2.0
=========================

The principal change in GalSim 2.0 is that it is now pip installable.
See the updated INSTALL file for details on how to install GalSim using
either pip or setup.py.  The functionality is essentially equivalent to
v1.6, although there are a few (mostly minor) API changes in some classes.

Dependency Changes
------------------

- Officially no longer support Python 2.6. (#755)
- No longer support pre-astropy versions of pyfits (now bundled in astropy
  as astropy.io.fits).  Nor astropy versions <1.0. (#755)
- No longer support pre-2016 version of the COSMOS catalog.  You may be
  asked to run galsim_download_cosmos again if your version is found to
  be obsolete. (#755)
- Added LSSTDESC.Coord, which contains the functionality that used to be in
  GalSim as the Angle and CelestialCoord classes.  We moved it to a separate
  repo so people could more easily use this functionality without requiring all
  of GalSim as a dependency. (#809b)
- Removed dependency on boost. (#809)
- Removed dependency on TMV. (#809)
- Added dependency on pybind11.  (You can still use boost if you want using
  the SCons installation method.) (#809)
- Added dependency on Eigen. (You can still use TMV if you want using the
  SCons installation method.) (#809)
- FFTW is now the only dependency that pip cannot handle automatically. (#809)


API Changes
-----------

- Changed the default maximum_fft_size in GSParams to 8192 from 4096.  This
  increases the potential memory used by an FFT when drawing an object with
  an FFT from 256 MB to 1 GB. (#755)
- Changed the order of arguments of galsim.wfirst.allDetectorEffects. (#755)
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
- Changed how gsparams work for objects that wrap other objects (e.g. Sum,
  Convolution, etc.). Now if you specify a gsparams at that level, it is
  propagated to all of the component objects.  (This behavior can be turned
  off with `propagate_gsparams=False`.) If you do not specify one, then the
  most restrictive combination of parameters from the components are applied
  to all of them. This is how gsparams should have worked originally, but it
  was not really possible in 1.x. (#968)


Deprecated Features
-------------------

- Removed all features deprecated in 1.x versions.


New Features
------------

- Added a new class hierarchy for exceptions raised by GalSim with the base
  class `GalSimError`, a subclass of `RuntimeError`. For complete details
  about the various sub-classes within this hierarchy, see the file errors.py.
  In most cases, if you were catching a specific exception such as ValueError
  or RuntimeError, the new error will still be caught properly.  However, some
  cases have changed to an incompatible error type, so users who have written
  `except` statements with specific error types should be careful to make
  sure that the errors you wanted to catch are still being caught. (#755)
- Changed the type of warnings raised by GalSim to GalSimWarning, which is
  a subclass of UserWarning. (#755)
- Added the withGSParams() method for all GSObjects. (#968)
