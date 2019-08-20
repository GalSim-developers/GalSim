Changes from v2.1 to v2.2
=========================

Deprecated Features
-------------------

- Deprecated the nominally private class galsim.correlatednoise._BaseCorrelatedNoise.  If you
  were using it for any purpose, you should now use `galsim.BaseCorrelatedNoise`. (#160)
- Deprecated the ``tol`` parameter of the various Interpolant classes.  Users should use the
  ``kvalue_accuracy`` parameter of ``gsparams`` instead. (#1038)

API Changes
-----------

- Removed functionality to store and re-load WFIRST PSFs, and to obtain multiple WFIRST
  PSFs simultaneously. (#919)
- Changed the function signature of StampBuilder.addNoise.  If you have a custom module that
  uses that, you should remove the ``skip`` parameter (which had always been False). (#1048)

Changes to Shared Files
-----------------------

- Added option to set the `galsim.meta_data.share_dir` via environment variable GALSIM_SHARE_DIR.
  (#1014)
- Changed hosting of the RealGalaxy COSMOS catalog to `Zenodo <https://zenodo.org/record/3242143>`_,
  which seems to have solved some flakiness in the previous hosting at the University of
  Manchester. (#1033)

Config Updates
--------------

- Added some more cumstomization hooks in the StampBuilder class. (#1048)
- Added ``quick_skip`` option to skip an object before doing any work. (#1048)
- Added ``obj_rng=False`` option to use the same rng for all objects in image. (#1048)
- Added ``rng_index_key`` option to use a different rng just for particular values. (#1048)
- Fixed ``@`` strings to work with input objects as well as normal variables. (#1048)
- Fixed various minor bugs when doing complicated things with index_key and rng_num. (#1048)

Documentation Updates
---------------------

- The documentation is now rendered in Sphinx, rather than Doxygen, which looks much nicer.  The
  new docs are accessible at http://galsim-developers.github.io/GalSim/.  (#160)

New Features
------------

- Added `FitsHeader.extend` method.  Also, read_header option to `galsim.fits.read`. (#877)
- Updated WFIRST WCS and PSF routines to use Cycle 7 specifications for detector configurations,
  pupil planes, and aberrations. In particular, there is now a different
  pupil plane image for shorter- and longer-wavelength bands.  (#919)
- Enabled Zernikes up to 22 (previously 11) in WFIRST PSFs, and added dependence on position
  within the SCA. (#919)
- Added WFIRST fermi persistence model. (#992)
- Added ``r0_500`` argument to VonKarman. (#1005)
- Improved ability of `AtmosphericScreen` to use shared memory in multiprocessing context. (#1006)
- Use OpenMP when appropriate in `SiliconSensor.accumulate` (#1008)
- Added array versions of `BaseWCS.toWorld` and `BaseWCS.toImage`. (#1026)
- Exposed some methods of `Interpolant` classes that had only been in the C++ layer. (#1038)
- Added Zernike polynomial +, -, and * operators. (#1047)
- Added Zernike polynomial properties .laplacian and .hessian. (#1047)

Bug Fixes
---------

- Fixed a couple places where negative fluxes were not working correctly. (#472)
- Fixed FITS I/O to write out comments of header items properly. (#877)
- Fixed error in `PhaseScreenPSF` when aberrations has len=1. (#1006, #1029)
- Fixed error in `BaseWCS.makeSkyImage` when crossing ra=0 line for some WCS classes. (#1030)
- Fixed slight error in the realized flux of some profiles when using photon shooting.
  The bug was most apparent for Kolmogorov and VonKarman, where the realized flux
  could be too small by about 1.e-3. (#1036)
- Fixed error in `Sersic` class when n is very, very close to 0.5. (#1041)
