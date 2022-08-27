Changes from v2.3 to v2.4
=========================

With this release, we are no longer supporting Python 2.7 or 3.6.
GalSim is supported for the following Python versions: 3.7, 3.8, 3.9, 3.10.

.. note::

    As advertised in the 2.3 release notes, GalSim no longer supports the
    following legacy options:

    * Python 2
    * TMV for matrices in C++
    * Boost Python for C++ bindings
    * SCons builds

A full list of changes in this release are below.  The numbers in parentheses
are GalSim issue or pull request numbers where the change was implemented.

cf. https://github.com/GalSim-developers/GalSim/milestone/21?closed=1

API Changes
-----------

- Removed CppEllipse in C++ layer, which had been deprecated since the 1.x series, but we forgot
  to actually get rid of. (#1129)
- Removed AstronomicalConstants.h in C++ layer, which we never used. (#1129)
- Removed AttributeDict, which we had sitting in utilities, but which we have never used.
  (#1129)
- Some changes to the C++ Image constructors to include a maxptr value. (#1149)
- Documented an API change, which was actually introduced in v2.3, that `_InterpolatedImage` does
  not recenter the image to (0,0) as `InterpolatedImage` does. (#1151)
- Changed `SincInterpolant.ixrange` to be consistent with the value of xrange, rather than inf.
  (#1154)
- Changed ``galsim.scene`` namespace name to ``galsim.galaxy_sample``. (#1174)


Config Updates
--------------

- Added ``Correlated`` noise type as a generalization of the more specific ``COSMOS`` noise type.
  (#731, #1174)
- Added ``galaxy_sample`` input type with corresponding ``SampleGalaxy`` GSObject type.
  (#795, #1174)
- Added ``COSMOSValue`` and ``SampleValue`` value types. (#954, #1174)
- Allowed template file names to be evaluated using the "$" shorthand notation. (#1138)
- Added `RegisterTemplate` to allow the ability to register templates by name. (#1143)
- Fixed some errors in `PhotonDCR` usage in the config layer. (#1148)
- Added option to specify a dtype other than np.float32 for images built by config. (#1160)
- Fixed inconsistent behavior of image.world_pos in image type=Single. (#1160)
- Let a flux item for an object with an SED normalize the SED for the bandpass being
  simulated. (#1160)
- Fixed some edge cases where the created image could not have the requested wcs. (#1160)
- Added option to ``initialize`` input objects in an `InputLoader`. (#1162, #1163)
- Fixed error in returned variance for ``CCDNoise`` builder, which was in e- rather than ADU.
  (#1166, #1167)
- Changed the way the internal random number sequence works so that running multiple simulations
  with sequential random seed values doesn't end up with duplicated random values across the
  two (or more) simulations. (#1169)


New Features
------------

- Added `BaseCorrelatedNoise.from_file` class method. (#731, #1174)
- Added `GalaxySample` class as generalization of `COSMOSCatalog`. (#795, #1174)
- Added methods `Image.transpose`, `Image.flip_ud`, `Image.flip_lr`, `Image.rot_cw`,
  `Image.rot_ccw`, and `Image.rot_180`. (#1139)
- Exposed our Si, Ci, sinc, and gammainc functions from C++. (#1146)
- Added pupil_u and pupil_v to `PhotonArray` and persist the values thereof when they are
  computed as part of a phase PSF computation. (#1147)
- Added `Image.depixelize` and ``depixelize=True`` option for `InterpolatedImage`. (#1154)
- Let `Bounds.expand` take an optional second argument to scale differently in different
  directions. (#1153, #1155)
- Added `BaseWCS.shearToWorld` and `BaseWCS.shearToImage` along with overloading
  `BaseWCS.toWorld` and `BaseWCS.toImage` to mean the same thing when the argument is a
  `Shear` value. (#1158, #1172)
- Added `PupilImageSampler` and `PupilAnnulusSampler` photon operators. (#1176)
- Added `TimeSampler` photon operator. (#1178)
- Added `BaseDeviate.as_numpy_generator`. (#1067, $1179)
- Added ``timeout`` option to control multiprocessing timeout limit and increased the default. (#1180)


Performance Improvements
------------------------

- Change the implementation of the Silicon sensor code to use ~half as many points for the pixels
  by sharing boundaries between neighboring pixels. (#1118, #1137)
- Use single precision for Silicon pixel boundaries, which further reduces the memory required
  for the SiliconSensor implementation. (#1140)
- Moved some of the logic related to the Silicon sensor to the python layer.  This is not per se
  a performance improvement, but it enables some potential future improvements. (#1141)
- Let `BaseDeviate.generate` use multiple threads in C++ layer. (#1177)


Bug Fixes
---------

- Fixed some cases where HSM would fail to converge for apparently very well-behaved images.
  (#1132, #1149)
- Fixed error in `InterpolatedImage.withGSParams` not correctly updating stepk and maxk
  if the updated parameters merited it. (#1154)
- Fixed error in `ChromaticSum` photon shooting when ``n_photons`` is explicitly given.
  (#1156, #1157)
- Fixed some rounding errors that could happen when rendering integer-typed images
  (e.g. ImageI) that could cause values to be off by 1. (#1160)
