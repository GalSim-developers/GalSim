Changes from v1.5 to v1.6
=========================

API Changes
-----------
- AtmosphericScreen instantiation is now delayed until first use, and the
  nature of the first use can change the value of the generated screens.  Use
  the .instantiate() method to manually override auto-instantiation. (#864)
- PhaseScreen wavefront and wavefront_gradient methods now accept `None` as a
  valid time argument, which means to use the internally stored time in the
  screen(s). (#864)
- Reduced the number of types for the return value of various NFWHalo and
  PowerSpectrum methods.  Now they either return a single value if the input
  `pos` is a single Position or a numpy array if multiple positions were
  provided. (#855)


Dependency Changes
------------------


Bug Fixes
---------
- Fixed error in amplitude of phase screens created by AtmosphericScreen (#864)
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
- Deprecated PowerSpectrum.subsampleGrid. (#855)


New Features
------------
- Add Zernike submodule. (#832, #951)
- Add SecondKick profile GSObject. (#864)
- Automatically include SecondKick objects when drawing PhaseScreenPSFs using
  geometric photon shooting. (#864)
- Add option to use circular weight function in HSM adaptive moments code. (#917)
- Add VonKarman profile GSObject. (#940)


New config features
-------------------
