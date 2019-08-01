Changes from v2.1 to v2.2
=========================

Deprecated Features
-------------------

- Deprecated the nominally private class galsim.correlatednoise._BaseCorrelatedNoise.  If you
  were using it for any purpose, you should now use galsim.BaseCorrelatedNoise. (#160)
- Deprecated the ``tol`` parameter of the various Interpolant classes.  Users should use the
  ``kvalue_accuracy`` parameter of ``gsparams`` instead. (#1038)

API Changes
-----------

- Removed functionality to store and re-load WFIRST PSFs, and to obtain multiple WFIRST
  PSFs simultaneously. (#919)

New Features
------------
- Updated WFIRST WCS and PSF routines to use Cycle 7 specifications for detector configurations,
  pupil planes, and aberrations. In particular, there is now a different
  pupil plane image for shorter- and longer-wavelength bands.  (#919)
- Enabled Zernikes up to 22 (previously 11) in WFIRST PSFs, and added dependence on position
  within the SCA. (#919)
- Added WFIRST fermi persistence model. (#992)
- Added ``r0_500`` argument to VonKarman. (#1005)
- Added array versions of ``wcs.toWorld`` and ``wcs.toImage``. (#1026)
- Exposed some methods of Interpolants that had only been in the C++ layer. (#1038)

Bug Fixes
---------

- Fixed error in ``wcs.makeSkyImage`` when crossing ra=0 line for some WCS classes. (#1030)
- Fixed slight error in the realized flux of some profiles when using photon shooting.
  The bug was most apparent for Kolmogorov and VonKarman, where the realized flux
  could be too small by about 1.e-3. (#1036)
- Fixed error in Sersic class when n is very, very close to 0.5. (#1041)
