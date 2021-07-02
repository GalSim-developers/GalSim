.. image:: https://github.com/GalSim-developers/GalSim/workflows/GalSim%20CI/badge.svg
        :target: https://github.com/GalSim-developers/GalSim
.. image:: https://codecov.io/gh/GalSim-developers/GalSim/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/GalSim-developers/GalSim
.. image:: https://img.shields.io/badge/astro--ph.IM-1407.7676-B31B1B.svg
        :target: https://arxiv.org/abs/1407.7676
.. image:: https://img.shields.io/badge/ADS-Rowe%20et%20al%2C%202015-blue.svg
        :target: http://adsabs.harvard.edu/abs/2015A%26C....10..121R

GalSim is open-source software for simulating images of astronomical objects
(stars, galaxies) in a variety of ways.  The bulk of the calculations are
carried out in C++, and the user interface is in Python.  In addition, the code
can operate directly on "config" files, for those users who prefer not to work
in Python.  The impetus for the software package was a weak lensing community
data challenge, called GREAT3:

    http://great3challenge.info/

However, the code has numerous additional capabilities beyond those needed for
the challenge, and has been useful for a number of projects that needed to
simulate high-fidelity galaxy images with accurate sizes and shears.  At the
end of this file, there is a list of the code capabilities and plans for future
development.  For details of algorithms and code validation, please see

    http://adsabs.harvard.edu/abs/2015A%26C....10..121R

The GalSim version numbering tries to follow `Semantic Versioning <https://semver.org/>`_
This means that releases are numbered as M.m.r, where M is a major version number,
m is the minor version, and r is the revision (or patch or bugfix) number.

The public API is preserved within a given major version number.  So code that works
with version 2.2.3 (say) should continue to work for all subsequent 2.x.x versions.
Minor versions indicate new features being added to the API.  Revision versions
don't add any new features, but fix bugs in the previous release.

Basic Installation
==================

Normally, to install GalSim, you should just need to run::

    pip install galsim

Depending on your setup, you may need to add either sudo to the start
or --user to the end of this command as you normally do when pip installing
packages.

See `Installation Instructions` for full details including one dependency (FFTW) that is not
pip installable, so you may need to install before running this command.

You can also use conda via conda-forge::

    conda install -c conda-forge galsim


Source Distribution
===================

To get the latest version of the code, you can grab the tarball (or zip file) from

https://github.com/GalSim-developers/GalSim/releases/

Also, feel free to fork the repository:

https://github.com/GalSim-developers/GalSim/fork

Or clone the repository with either of the following::

    git clone git@github.com:GalSim-developers/GalSim.git
    git clone https://github.com/GalSim-developers/GalSim.git

The code is also distributed via Fink, Macports, and Homebrew for Mac users.
See `Installation Instructions` (in INSTALL.rst) for more information.

The code is licensed under a BSD-style license.  See the file LICENSE for more
details.


Keeping up-to-date with GalSim
==============================

There is a GalSim mailing list, organized through the Google Group
galsim-announce.  Members of the group will receive news and updates about the
GalSim code, including notifications of major version releases, new features
and bugfixes.

You do not need a Google Account to subscribe to the group, simply send any
email to::

    galsim-announce+subscribe@googlegroups.com

If you receive a confirmation request (check junk mail filters!) simply reply
directly to that email, with anything, to confirm.  You may also click the link
in the confirmation request, but you may be asked for a Google Account login.

To unsubscribe, simply send any email to::

    galsim-announce+unsubscribe@googlegroups.com

You should receive notification that your unsubscription was successful.


How to communicate with the GalSim developers
=============================================

Currently, the lead developers for GalSim are:

  - Mike Jarvis (mikejarvis17 at gmail)
  - Rachel Mandelbaum (rmandelb at andrew dot cmu dot edu)
  - Josh Meyers (jmeyers314 at gmail)

However, many others have contributed to GalSim over the years as well, for
which we are very grateful.

If you have a question about how to use GalSim, a good place to ask it is at
`StackOverflow <http://stackoverflow.com/>`_.  Some of the GalSim developers
have alerts set up to be automatically notified about questions with the
'galsim' tag, so there is a good chance that your question will be answered.

If you have any trouble installing or using the code, or find a bug, or have a
suggestion for a new feature, please open up an Issue on our `GitHub
repository <https://github.com/GalSim-developers/GalSim/issues>`_.  We also accept
pull requests if you have something you'd like to contribute to the code base.

If none of these communication avenues seem appropriate, you can also contact
us directly at the above email addresses.


Demonstration scripts
=====================

There are a number of scripts in ``examples/`` that demonstrate how the code can
be used.  These are called ``demo1.py`` ... ``demo13.py``.  You can run them by
typing (e.g.) ``python demo1.py`` while sitting in ``examples/``, All demo scripts
are designed to be run in the ``examples/`` directory.  Some of them access
files in subdirectories of the ``examples/`` directory, so they would not work
correctly from other locations.

A completely parallel sequence of configuration files, called ``demo1.yaml`` ...
``demo13.yaml``, demonstrates how to make the same set of simulations using
config files that are parsed by the executable ``bin/galsim``.

Two other scripts in the ``examples/`` directory that may be of interest, but
are not part of the GalSim tutorial series, are ``make_coadd.py``, which
demonstrates the use of the FourierSqrt transformation to optimally coadd
images, and ``psf_wf_movie.py``, which demonstrates the realistic atmospheric
PSF code by making a movie of a time-variable PSF and wavefront.

As the project develops through further versions, and adds further
capabilities to the software, more demo scripts may be added to ``examples/``
to illustrate what GalSim can do.


Summary of current capabilities
===============================

Currently, GalSim has the following capabilities:

* Can generate PSFs from a variety of simple parametric models such as Moffat,
  Kolmogorov, and Airy, as well as an optical PSF model that includes Zernike
  aberrations to arbitrary order, and an optional central obscuration and
  struts.

* Can simulate galaxies from a variety of simple parametric models as well as
  from real HST data.  For information about downloading a suite of COSMOS
  images, see

  https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data

* Can simulate atmospheric PSFs from realistic turbulent phase screens.

* Can make the images either via i) Fourier transform, ii) real-space
  convolution (real-space being occasionally faster than Fourier), or
  iii) photon-shooting.  The exception is that objects that include a
  deconvolution (such as RealGalaxy objects) must be carried out using Fourier
  methods only.

* Can handle wavelength-dependent profiles and integrate over filter
  bandpasses appropriately, including handling wavlengths properly when
  photon shooting.

* Can apply shear, magnification, dilation, or rotation to a galaxy profile
  including lensing-based models from a power spectrum or NFW halo profile.

* Can draw galaxy images into arbitrary locations within a larger image.

* Can add noise using a variety of noise models, including correlated noise.

* Can whiten or apply N-fold symmetry to existing correlated noise that is
  already in an image.

* Can read in input values from a catalog, a dictionary file (such as a JSON
  or YAML file), or a fits header.

* Can write images in a variety of formats: regular FITS files, FITS data
  cubes, or multi-extension FITS files.  It can also compress the output files
  using various compressions including gzip, bzip2, and rice.

* Can carry out nearly any simulation that a user might want using two parallel
  methods: directly using Python code, or by specifying the simulation
  properties in an input configuration script.  See the demo scripts in
  the examples/ directory for examples of each.

* Supports a variety of possible WCS options from a simple pixel scale factor
  of arcsec/pixel to affine transforms to arbitrary functions of (x,y),
  including a variety of common FITS WCS specifications.

* Can include a range of simple detector effects such as nonlinearity,
  brighter-fatter effect, etc.

* Has a module that is particularly meant to simulate images for the Roman
  Space Telescope.


Planned future development
--------------------------

We plan to add the following additional capabilities in future versions of
GalSim:

* Simulating more sophisticated detector defects and image artifacts.  E.g.
  vignetting, fringing, cosmic rays, saturation, bleeding, ... (cf. Issues
  #553, #828)

* Proper modeling of extinction due to dust. (cf. Issues #541, #550)

* More kinds of realistic galaxies. (cf. Issues #669, #795, #808)

* Various speed improvements.  (cf. Issues #205, #566, #875)

There are many others as well.  Please see

https://github.com/GalSim-developers/GalSim/issues

for a list of the current open issues.  And feel free to add an issue if there
is something useful that you think should be possible, but is not currently
implemented.
