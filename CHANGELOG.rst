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

- Let `expand` method of a `galsim.Bounds` instance take an optional second argument to scale
  differently in different directions. (#1155)



Config Updates
--------------



New Features
------------

- Added Image methods: tranpose, flip_ud, flip_lr, rot_cw, rot_ccw, rot_180. (#1139)


Performance Improvements
------------------------



Bug Fixes
---------

- Fixed error in InterpolatedImage.withGSParams not correctly updating stepk and maxk
  if the updated parameters merited it. (#1154)
- Fix error in ChromaticSum photon shooting when n_photons is explicitly given. (#1156)
