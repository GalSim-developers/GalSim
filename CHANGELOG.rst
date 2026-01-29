Changes from v2.7 to v2.8
=========================

We currently support Python 3.9 through 3.14.

A complete list of all new features and changes is given below.
`Relevant PRs and Issues,
<https://github.com/GalSim-developers/GalSim/milestone/25?closed=1>`_
whose issue numbers are listed below for the relevant items.

Dependency Changes
------------------

- No longer pinned to setuptools<72, which had been required previously due to breaking changes
  that had been implemented then. We've now updated GalSim code to be compliant with the
  new behavior of setuptools. (#1335)


API Changes
-----------

- Changed GalSim's response to large FFTs to be a warning rather than an error. Given that
  people run GalSim on a wide variety of systems, some of which can handle the memory for
  quite large FFTs, it has become more of an annoyance than a help to have to predict the
  maximum size of all your FFTs using ``maximum_fft_size``. So now, rather than give an
  error when you exceed that, GalSim will just emit a warning, so if your system crashes
  from the memory use, you can still see the warning message and trouble shoot.
  If you relied on the old behavior, you may re-enable it by setting
  ``galsim.errors.raise_fft_size_error = True``  With this setting, you will get errors
  as in GalSim versions <= 2.7.  (#1332, #1341)


Bug Fixes
---------

- Fixed a bug in `Image.calculateFWHM` that started with numpy version 2.3. (#1336, #1337)


Changes from v2.8.0 to v2.8.1
-----------------------------

- Nothing substantive. Just a pypi problem with 2.8.0.
