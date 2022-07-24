Changes from v2.3 to v2.4
=========================

With this release, we are no longer supporting Python 2.7 or 3.6.
GalSim is supported for the following Python versions: 3.7, 3.8, 3.9.
[MJ: Note: I fully expect to include 3.10 in the mix by the time we release this.]

.. note::

    As advertised in the 2.3 release notes, GalSim no longer supports the
    following legacy options:

    * Python 2
    * TMV for matrices in C++
    * Boost Python for C++ bindings
    * SCons builds


Dependency Changes
------------------



API Changes
-----------

- Changed `galsim.scene` namespace name to `galaxy.galaxy_sample`. (#731)
- Removed CppEllipse in C++ layer, which had been deprecated since the 1.x series, but we forgot
  to actually get rid of. (#1129)
- Removed AstronomicalConstants.h in C++ layer, which we never used. (#1129)
- Removed AttributeDict, which we had sitting in utilities, but which we have never used.
  (#1129)
- Changed SincInterpolant.ixrange to be consistent with the value of xrange, rather than inf.
  (#1154)


Config Updates
--------------

- Added Correlated noise type as a generalization of the more specific COSMOS noise type.
  (#731)
- Changed the way the internal random number sequence works so that running multiple simulations
  with sequential random seed values doesn't end up with duplicated random values across the
  two (or more) simulations. (#1169)


New Features
------------

- Added `BaseCorrelatedNoise.from_file` class method. (#731)
- Added `GalaxySample` class as generalization of `COSMOSCatalog`. (#731)
- Added methods `Image.transpose`, `Image.flip_ud`, `Image.flip_lr`, `Image.rot_cw`,
  `Image.rot_ccw`, and `Image.rot_180`. (#1139)
- Added `Image.depixelize` and ``depixelize=True`` option for `InterpolatedImage`. (#1154)
- Let `galsim.Bounds.expand` take an optional second argument to scale differently in different
  directions. (#1155)
- Added `BaseWCS.shearToWorld` and `BaseWCS.shearToImage` along with overloading
  `BaseWCS.toWorld` and `BaseWCS.toImage` to mean the same thing when the argument is a
  `Shear` value. (#1158)


Performance Improvements
------------------------



Bug Fixes
---------

- Fixed error in InterpolatedImage.withGSParams not correctly updating stepk and maxk
  if the updated parameters merited it. (#1154)
- Fix error in ChromaticSum photon shooting when n_photons is explicitly given. (#1156)
