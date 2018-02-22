Changes from v1.5 to v1.6
=========================

API Changes
-----------
- AtmosphericScreen instantiation is now delayed until first use, and the
  nature of the first use can change the value of the generated screens.  Use
  the .instantiate() method to manually override auto-instantiation. (#846)


Dependency Changes
------------------


Bug Fixes
---------

- Fixed a bug in the DES MEDS writer setting the cutout row/col wrong. (#928)
- Fixed a number of small bugs in the config processing uncovered by the
  galsim_extra FocalPlane output type. (#928)
- Fixed python3 unicode/str mismatches in tests/SConscript (#932)
- Fixed memory leak when drawing PhaseScreenPSFs using photon-shooting (#942)
- Fixed error in amplitude of phase screens created by AtmosphericScreen (#864)

Deprecated Features
-------------------



New Features
------------
- Add option to use circular weight function in HSM adaptive moments code. (#917)
- Add VonKarman profile GSObject. (#940)
- Add SecondKick profile GSObject. (#864)
- Automatically include SecondKick objects when drawing PhaseScreenPSFs using
  geometric photon shooting. (#864)



New config features
-------------------
