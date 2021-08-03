Changes from v2.2 to v2.3
=========================

This release mostly focuses on improvements needed for LSST Rubin simulations
for the Dark Energy Science Collaboration and ones needed for Roman Space
Telescope simulations for the Roman HLS SIT. These mostly involve adding
features to the config layer processing that are relevant to these two
telescopes as well as some new features to add additional realism to the
simulations.

With this release, we are no longer supporting Python 3.5.  And we have
upgraded Python 3.8 and 3.9 to primary supported platforms, which now include
2.7, 3.6, 3.7, 3.8, 3.9.

.. warning::

    This will be the last GalSim release to support the following legacy
    options:

    * Python 2.7
    * TMV for matrices in C++
    * Boost Python for C++ bindings
    * SCons builds

    If any of these will cause hardship for you, please open an issue, so we
    can help you find mitigation strategies.

A full list of changes in this release are below.  The numbers in parentheses
are GalSim issue or pull request numbers where the change was implemented.

cf. https://github.com/GalSim-developers/GalSim/milestone/20?closed=1


Dependency Changes
------------------

- Removed future as a dependency. (#1082)
- Removed eigency as an optional way to get eigen dependency.  Now it will
  download the eigen code directly if it cannot find eigen installed on
  your system. (#1086)


API Changes
-----------

- Deprecated the ``rng`` parameter to `WavelengthSampler` and `FRatioAngles`
  constructors.  Now they will use the rng that you provide to drawImage or
  whatever else you are using to apply these photon operators. (#540)
- Deprecated ``withOrigin`` method for non-local WCS types in favor of the new
  method `BaseWCS.shiftOrigin`.  This has the same functionality, but the name is
  more in line with the actual action of the function.  For local WCS types,
  `BaseWCS.shiftOrigin` is equivalent to `LocalWCS.withOrigin`, which still
  exists as a valid name for this action. (#1073)
- The numerical details of the `Kolmogorov` class were updated to use a more
  precise version of a constant from Racine (1996).  Technically this changes
  the definition of the Kolmogorov profile at the 6th decimal place.  So if
  you carefully tuned an r0 value to 6 decimal places for some purpose, this
  might break that. (#1084)
- Deprecated ``galsim.wfirst`` module.  Now called ``galsim.roman``. (#1088)
- Changed the default ``ii_pad_factor`` for `PhaseScreenPSF` and `OpticalPSF` to 1.5.
  The old default of 4.0 (matching the `InterpolatedImage` default) is not
  necessary for objects that will not be significantly sheared, which
  PSF objects typically aren't.  For almost all use cases, this will be
  simply a performance improvement, but if you need the higher ``ii_pad_factor``
  for some reason, you will now need to manually specify it. (#1089)
- Deprecated the ``high_accuracy`` and ``approximate_struts`` parameters for the
  `galsim.roman.getPSF` function.  You should now use ``pupil_bin`` and ``gsparams`` to
  effect similar adjustments to the default (which is now much better than
  before in terms of both speed and accuracy). (#1089)
- Deprecated the ``surface_ops`` parameter to `GSObject.drawImage`, switching
  to the name ``photon_ops``, since these don't have to be something that
  happens at the surface of the sensor. (#1093)
- Added ``logger`` option to some config functions and methods. If you are using
  custom Image or Output types, you may need to add a ``logger=None`` optional
  parameter to some method signatures. (#1095)
- Deprecated ``galsim.integ.trapz``.  You should use `galsim.integ.int1d`
  instead, which is almost always more efficient. (#1098)
- Deprecated ``galsim.integ.midpt``.  You should use ``np.trapz`` or
  `galsim.trapz` instead, which are almost equivalent, but slightly more
  accurate. (#1098)
- Changed the convention for the ``f`` array passed to the `LookupTable2D`
  constructor to be the transpose of what it was.  This is arguably a bug
  fix, since the old convention was the opposite of every other array used
  in GalSim that reprented (x,y) positions.  But if you have been using
  `LookupTable2D`, you will need to add ``.T`` (or remove it) from the
  ``f`` argument you are passing to the class. (#1103)
- Changed the behavior of `PhaseScreenPSF`, `OpticalPSF`, and
  `ChromaticOpticalPSF` by adding the kwarg ``fft_sign``, which controls the
  sign in the exponent of the Fourier kernel using in the Fourier optics
  PSF equation.  The new default value of '+' produces a profile that is
  180 degrees rotated compared to the former behavior.  To revive the
  former behavior, set ``fft_sign='-'``. (#1104)


Config Updates
--------------

- Added ability to draw chromatic objects with config files. (#510)
- Added demo12.yaml and demo13.yaml to the demo suite. (#510, #1121)
- Fixed a few issues with parsing truth catalog items and, in general, places
  where a ``Current`` item might be used before it was parsed. (#1083)
- Added value-type-specific type names like ``Random_float``, ``Random_Angle``, etc.
  to help in some cases where the config processing cannot know what value
  type to use for further processing.  (#1083)
- Fixed a subtle issue in ``Eval`` string processing if one ``Current`` item is a
  shorter version of another one.  e.g. ``@gal`` and ``@gal.index``.  It had been
  the case that the longer one might not be parsed properly. (#1083)
- Added ``photon_ops`` and ``sensor`` as options in the stamp processing.
  (#1093)
- Removed the ``_nobjects_only`` mechanism from input objects.  If this
  pattern was useful for a custom input type, you should switch to using lazy
  properties to delay the loading of data other than what is needed for
  determining the number of objects.  See `Catalog` for an example of how
  to do this. (#1095)
- Allowed ``Eval`` fields to use any modules that are listed in the top-level
  ``modules`` field. (#1121)
- Added Roman config types: ``RomanSCA``, ``RomanBandpass``, and ``RomanPSF``. (#1121)


New Features
------------

- Added `ChromaticObject.atRedshift` method. (#510)
- Added `galsim.utilities.pickle_shared` context, which lets the shared
  portion of an `AtmosphericScreen` be included in the pickle.  This allows
  the pickles to be recovered after writing to disk and reading back in,
  which otherwise would not work. (#1057)
- Added ``force_stepk`` option to `VonKarman`. (#1059)
- Added `Refraction` and `FocusDepth` photon ops. (#1065, #1069)
- Updated LSST sensor files to match new lab measurements and use improved
  Poisson code calculations. (#1077, #1081)
- Added `GSObject.makePhot` method. (#1078)
- Added individual kwargs syntax to `GSObject.withGSParams` to make it easier
  to set specific parameters. e.g. ``obj.withGSParams(folding_threshold=1.e-3)``
  to just change that one value and keep any other parameters the same. (#1089)
- Added a new pupil_bin option to the `galsim.roman.getPSF` function.  This
  controls the resolution of the pupil plane mask, which involves a trade-off
  between speed and accuracy of the PSF rendering. (#1089)
- Added `FittedSIPWCS` to fit a WCS from a list of image and celestial
  coordinates. (#1092)
- Extended `GSFitsWCS` to support -SIP distortions for non-TAN WCSs. (#1092)
- Added ``wcs`` option to `galsim.roman.getPSF` function to more easily get the
  right PSF in world coordinates for a particular observation. (#1094)
- Added `Position.shear` method. (#1090)
- Added `LookupTable.integrate` and `LookupTable.integrate_product`, along
  with `galsim.trapz` as a drop in replacement for ``numpy.trapz``, which
  is often somewhat faster. (#1098)
- Added `galsim.integ.hankel` function. (#1099)
- Added `galsim.bessel.jv_root` function. (#1099)
- Added support for TPV WCS files with order > 3. (#1101)
- Added `UserScreen` for arbitrary user-supplied phase screens (#1102)
- Added `galsim.zernike.describe_zernike` to construct an algebraic string describing
  circular Zernike terms in the Cartesian basis. (#1104)
- Added option to emit WCS warnings when reading a file via `galsim.fits.read`
  e.g. if the WCS defaulted to a `PixelScale`, or it reverted to an approximate
  `AffineTransform` rather than the correct WCS. (#1120)
- Added ``area`` and ``exptime`` parameters to `COSMOSCatalog` constructor to make it
  easier to rescale the fluxes to a different telescope than HST. (#1121)


Performance Improvements
------------------------

- Implemented `Transformation` ``_drawReal`` and ``_drawKImage`` in python to
  provide hooks for performance improvements in user code when these are used in
  tight loops. (#934)
- Sped up the draw routines for `InterpolatedImage`. (#935)
- Improved the rendering of Roman PSFs to always show 12 diffraction spikes
  (rather than 6 in the now-deprecated approximate_struts mode), remove an
  FFT artifact in the exact pupil plane mode, and significantly speed up all
  PSF renderings. (#1089)
- Sped up `GSFitsWCS` RA,Dec -> x,y calculation for SIP and PV distorted WCSs
  by ~20x. (#1092)
- Various speed improvements in config processing. (#1095, #1098)
- Sped up `SED.calculateFlux` and a few other SED and Bandpass calculations
  by switching to `LookupTable.integrate_product` for the implementation of
  the integrals. (#1098)
- Sped up the Hankel transforms several classes use for computing either the
  k-space values (e.g. `Sersic`) or real-space values (e.g. `Kolmogorov`). (#1099)
- Improved the accuracy of ``stepk`` for `Kolmogorov` profiles, especially when
  ``folding_threshold`` is very small. (#1110)
- Sped up Zernike arithmetic for the case where you just want to evaluate a
  resulting Zernike series without knowing its coefficients. (#1124)
- Removed some small bits of overhead in some "leading underscore" methods
  (e.g. ``_drawReal``, ``_Transform``, ``_shift``, etc.) to make them faster. (#1126)


Bug Fixes
---------

- Fixed `horner` and `horner2d` when inputs are complex. (#1054)
- Fixed `VonKarman` integration to be more reliable for various combinations
  of (r0, L0, lam). (#1058)
- Fixed minor bug in ``repr`` of `OpticalPSF` class. (#1061)
- Fixed bug in `RandomKnots` when multiplied by an SED. (#1064)
- Fixed bug that `galsim.fits.writeMulti` didn't properly write the header
  information in each hdu. (#1091)


Changes from v2.3.0 to v2.3.1
=============================

- Fixed some problems with the shared library build. (#1128)

Changes from v2.3.1 to v2.3.2
=============================

- Fixed a rare problem with SED.sampleWavelength sometimes generating photons
  slightly outside of bandpass range. (#1131)

Changes from v2.3.2 to v2.3.3
=============================

- Fixed a bug where InterpolatedImage.drawReal could possibly cause seg faults
  from writing past the end of the image arrays.
