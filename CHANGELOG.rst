Changes from v2.6 to v2.7
=========================

We currently support Python 3.8 through 3.13.

A complete list of all new features and changes is given below.
`Relevant PRs and Issues,
<https://github.com/GalSim-developers/GalSim/milestone/24?closed=1>`_
whose issue numbers are listed below for the relevant items.


New Features
------------

- Added `DoubleZernike.xycoef` to directly access the pupil coefficient array rather than
  going through `Zernike`. (#1327)
- Added a setter for the `Image.array` property, so ``im.array = rhs`` is equivelent to
  ``im.array[:] = rhs``.  This cannot be used to replace the underlying object, only its
  contents.  But it avoids some confusing behavior that can happen when doing this operation
  in an interactive session (such as Jupyter). (#1272, #1329)
- Added an option ``recalc=True`` to `SiliconSensor.accumulate`.  In conjunction with
  ``nrecalc=0`` and ``resume=True``, this allows the user more precise control as to when the
  pixel boundaries are recalculated during the accumulation. (#1328)


Performance Improvements
------------------------

- Switched to inbuilt operators on lists rather than numpy operators in a few places where
  they are faster. (#1316)
- Added ``robust=True`` option to `Zernike.__call__`, which is both more accurate and faster
  for large Noll indices, but is usually slower for small indices. (#1326, #1327)


Bug Fixes
---------

- Fixed an error in the `Spergel` stepk calculation. (#1324, #1325)


Changes from v2.7.0 to v2.7.1
-----------------------------

- Fixed an error in PhotonDCR use of zenith_angle if sky_pos is also given. (#1330)

Changes from v2.7.1 to v2.7.2
-----------------------------

- Reduced memory use in the Silicon class. (#1331)
