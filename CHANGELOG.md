Changes from v1.5 to v2.0
=========================

The principal change in GalSim 2.0 is that it is now pip installable.
See the updated INSTALL file for details on how to install GalSim using
either pip or setup.py.

API Changes
-----------

- Most of the functionality associated with C++-layer objects has been
  redesigned or removed.  These were non-public-API features, so if you have
  been using the public API, you should be fine.  But if you have been relying
  on features of the exposed C++-layer, this might break your code. (#809)


Dependency Changes
------------------

- GalSim no longer depends on boost.


Bug Fixes
---------


Deprecated Features
-------------------

- All features that were deprecated in 1.x are now removed.


New Features
------------


New config features
-------------------

