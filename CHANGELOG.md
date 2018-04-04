Changes from v1.5 to v1.6
=========================

API Changes
-----------

- Reduced the number of types for the return value of various NFWHalo and
  PowerSpectrum methods.  Now they either return a single value if the input
  `pos` is a single Position or a numpy array if multiple positions were
  provided. (#855)
- When using LookupTable, SED or Bandpass as a function only return either a
  float or a numpy array. (#955)


Dependency Changes
------------------


Bug Fixes
---------

- Fixed a bug in the DES MEDS writer setting the cutout row/col wrong. (#928)
- Fixed a number of small bugs in the config processing uncovered by the
  galsim_extra FocalPlane output type. (#928)
- Fixed python3 unicode/str mismatches in tests/SConscript (#932)
- Fixed memory leak when drawing PhaseScreenPSFs using photon-shooting (#942)

Deprecated Features
-------------------

- Deprecated passing Image arguments to kappaKaiserSquires function. (#855)
- Deprecated the interpolant argument for PowerSpectrum methods getShear,
  getConvergence, getMagnification, and getLensing.  The interpolant should
  be set when calling buildGrid. (#855)
- Deprectated PowerSpectrum.subsampleGrid. (#855)


New Features
------------

- Add option to use circular weight function in HSM adaptive moments code. (#917)
- Add VonKarman profile GSObject.



New config features
-------------------
