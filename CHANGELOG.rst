Changes from v2.2 to v2.3
=========================


Dependency Changes
------------------

- Changed the WCS method name withOrigin to shiftOrigin for non-local WCS
  types.  The functionality hasn't changed, but the name withOrigin is
  only really appropriate for LocalWCS types.  When the WCS already has a
  non-zero origin, then the action takes in really to shift the origin, not
  set a new value. (#1073)
- Removed future as a dependency. (#1082)


API Changes
-----------

- The numerical details of the Kolmogorov class were updated to use a more
  precise version of a constant from Racine (1996).  Technically this changes
  the definition of the Kolmogorov profile at the 6th decimal place.  So if
  you carefully tuned an r0 value to 6 decimal places for some purpose, this
  might break that. (#1084)
- Deprecated withOrigin method for non-local WCS types in favor of the new
  method shiftOrigin.  This has the same functionality, but the name is
  more in line with the actual action of the function.  For local WCS types,
  shiftOrigin is equivalent to withOrigin, which still exists as a valid
  name for this action. (#1085)
- Deprecated galsim.wfirst module.  Now called galsim.roman. (#1088)
- Changed the default ii_pad_factor for PhaseScreenPSF and OpticalPSF to 1.5.
  The old default of 4.0 (matching the InterpolatedImage default) is not
  necessary for objects that will not be significantly sheared, which
  PSF objects typically aren't.  For almost all use cases, this will be
  simply a performance improvement, but if you need the higher ii_pad_factor
  for some reason, you will now need to manually specify it. (#1089)
- Deprecated the high_accuracy and approximate_struts parameters for the
  roman.getPSF function.  You should now use pupil_bin and gsparams to
  effect similar adjustments to the default (which is now much better than
  before in terms of both speed and accuracy). (#1089)
- Deprecated the ``surface_ops`` parameter to `GSObject.drawImage`, switching
  to the name ``photon_ops``, since these don't have to be something that
  happens at the surface of the sensor. (#1093)


Config Updates
--------------

- Added ability to draw chromatic objects with config files. (#510)
- Fixed a few issues with parsing truth catalog items and, in general, places
  where a Current item might be used before it was parsed. (#1083)
- Added value-type-specific type names like Random_float, Random_Angle, etc.
  to help in some cases where the config processing cannot know what value
  type to use for further processing.  (#1083)
- Fixed a subtle issue in Eval string processing if one Current item is a
  shorter version of another one.  e.g. @gal and @gal.index.  It had been
  the case that the longer one might not be parsed properly. (#1083)
- Added ``photon_ops`` and ``sensor`` as options in the stamp processing.
  (#1093)


New Features
------------

- Added atRedshift method for ChromaticObject. (#510)
- Added galsim.utilities.pickle_shared() context, which lets the shared
  portion of an AtmosphericScreen be included in the pickle.  This allows
  the pickles to be recovered after writing to disk and reading back in,
  which otherwise would not work. (#1057)
- Added force_stepk option to VonKarman. (#1060)
- Added Refraction and FocusDepth photon ops. (#1068, #1069)
- Updated LSST sensor files to match new lab measurements and use improved
  Poisson code calculations. (#1077, #1081)
- Added makePhot method of GSObject. (#1078)
- Made it easier to set specific GSParams parameters using the syntax (e.g.)
  obj.withGSParams(folding_threshold=1.e-3) to just change that one value
  and keep any other non-default parameters the same. (#1089)
- Added a new pupil_bin option to the Roman getPSF function.  This controls
  the resolution of the pupil plane mask, which involves a trade-off between
  speed and accuracy of the PSF rendering. (#1089)
- Added FittedSIPWCS to fit a WCS from a list of image and celestial
  coordinates. (#1092)
- Extended GSFitsWCS to support -SIP distortions for non-TAN WCSs. (#1092)
- Added wcs option to Roman getPSF function to more easily get the right PSF
  in world coordinates for a particular observation. (#1094)


Performance Improvements
------------------------

- Improved the rendering of Roman PSFs to always show 12 diffraction spikes
  (rather than 6 in the now-deprecated approximate_struts mode), remove an
  FFT artifact in the exact pupil plane mode, and significantly speed up all
  PSF renderings. (#1089)
- Sped up GSFitsWCS.radecToxy for SIP and PV distorted WCSs by ~20x. (#1092)


Bug Fixes
---------

- Fixed horner and horner2d when inputs are complex. (#1054)
- Fixed VonKarman integration to be more reliable for various combinations
  of (r0, L0, lam). (#1058)
- Fixed minor bug in repr of OpticalPSF class. (#1061)
- Fixed bug in RandomKnots when multiplied by an SED. (#1064)
- Fixed bug that galsim.fits.writeMulti didn't properly write the header
  information in each hdu. (#1091)
