@mainpage

GalSim: The modular galaxy image simulation toolkit
===================================================

GalSim is open-source software for simulating images of astronomical objects
(stars, galaxies) in a variety of ways.  The bulk of the calculations are
carried out in C++, and the user interface is in python.  In addition, the code
can operate directly on "config" files, for those users who prefer not to work
in python.  The code is being developed as a collaborative project for the
upcoming weak lensing community data challenge, GREAT3
(http://great3challenge.info/), though it has additional capabilities beyond
those needed for the challenge.  At the end of this file, there is a list of 
the code capabilities and plans for future development.


Distribution
------------

Please feel free to fork this repository at any time.  However, please be aware
that the code is still actively being developed and tested (hence the current
version number is below 1.0).  The release of v1.0 in mid-2013 will be
accompanied with a publication that users should cite.


Keeping up-to-date with GalSim
------------------------------

There is a GalSim mailing list, organized through the Google Group 
galsim-announce.  Members of the group will receive news and updates about the
GalSim code, including notifications of major version releases, new features
and bugfixes.

You do not need a Google Account to subscribe to the group, simply send any
email to

    galsim-announce+subscribe@googlegroups.com

If you receive a confirmation request (check junk mail filters!) simply reply 
directly to that email, with anything, to confirm.  You may also click the link
in the confirmation request, but you may be asked for a Google Account login.

To unsubscribe, simply send any email to

    galsim-announce+unsubscribe@googlegroups.com

You should receive notification that your unsubscription was successful.


How to communicate with the GalSim developers
---------------------------------------------

If you have any comments, questions, or suggestions, please open up an Issue on
our GitHub repository:

https://github.com/GalSim-developers/GalSim/issues?state=open

Alternatively, if you prefer e-mail, then you can find contact information on
the GREAT3 webpage linked above.


Installation
------------

For installation instructions, please see the file `INSTALL.md` in the main
repository directory. 

There are tagged versions of the code corresponding to specific project 
releases and development milestones. (For more info, see the "Tagged versions" 
section below, and `devel/git.txt`)


Getting started
---------------

* Install the code as in `INSTALL.md`.

* Optional, but recommended whenever you try a new version of the code: run the
  unit tests to make sure that there are no errors.  You can do this by running
  `scons tests`.  If there are any issues, please open an Issue on our GitHub 
  page.

* Optional: run `doxygen` to generate documentation, using `Doxyfile` in the
  main repository directory to specify all doxygen settings.  Alternatively,
  you can view the documentation online at

  http://galsim-developers.github.com/GalSim/


Reference documentation
-----------------------

For an overview of GalSim workflow and python tools, please see
`doc/GalSim_Quick_Reference.pdf` in the GalSim repository.  A guide to using 
the configuration files to generate simulations, a FAQ for installation issues,
and other useful references can be found on the GalSim wiki,

https://github.com/GalSim-developers/GalSim/wiki

More thorough documentation for all parts of the code can be found in the
doxygen documentation mentioned in the previous section, or in the python
docstrings in `galsim/*.py`.


Repository directory structure
------------------------------

The repository has a number of subdirectories. Below is a guide to their
contents:

* bin/ :      executables (after the compilation procedure is done).
* devel/ :    an assortment of developer tools.
* doc/ :      documentation, including a `Quick Reference` guide and, if the
              user generates doxygen documentation using Doxyfile, the outputs
              will also go in this directory.
* examples/ : example scripts (see the following section).
* galsim/ :   the python code for GalSim (which is what most end-users interact
              with).
* include/ :  the .h header files for the C++ parts of GalSim.
* lib/ :      compiled libraries (after the compilation procedure is done).
* pysrc/ :    the code that makes the purely C++ parts of GalSim accessible to
              the python layer of GalSim.
* src/ :      the source code for the purely C++ parts of GalSim.
* tests/ :    unit tests.


Demonstration scripts
---------------------

There are a number of scripts in `examples/` that demonstrate how the code can
be used.  These are called `demo1.py`-`demo11.py`, and can be run either using
(e.g.) `python demo1.py` while sitting in `examples/`, or by doing `scons
examples` and then using the executable `bin/demo1`.  A completely parallel
sequence of configuration scripts, `demo1.yaml`-`demo11.yaml`, demonstrates how
to make the same set of simulations using config scripts that can be input to
`bin/galsim_yaml`.  

All demonstration scripts (including `bin/demo1`, etc) are meant to be run 
within the `examples/` directory.  The demos can be run in a different 
directory, but then the demo script will not be able to find the required files
or directories.

As the project develops through further versions, and adds
further capabilities to the software, more demo scripts will be added to
`examples/` to illustrate what GalSim can do.


Additional scripts
------------------

While the demo scripts can be run from the command-line while sitting in
`examples/` without any arguments, the remaining scripts are auxiliary 
utilities that take various command-line arguments, which are always explained 
in comments at the top of the file.

* `ShootInterpolated.py` is a script that takes as input a filename for a FITS
image, which it will simulate (optionally sheared and/or resampled) via
photon-shooting.

* `MeasMoments.py` can be used to measure the adaptive moments (best-fit
elliptical Gaussian) for a FITS image.

* `MeasShape.py` can be used to carry out PSF correction using one of four
methods, given FITS images of the galaxy and PSF.


Tagged versions
---------------

After every GalSim release and development milestone we tag a snapshot of the 
code at that moment, with the tag name `vX.X` or `milestoneN` where N is the 
milestone number.  The milestoneN versions are not recommended now that we
have official tagged versions, `vX.X`.

You can see the available tags using the command

    git tag -l

at a terminal from within the repository.

The version of the code at any given snapshot can be downloaded from our
GitHub webpage, or checked out from the repository using the tag name, e.g.:

    $ git checkout v0.2

This will then update your directory tree to the snapshot of the code at the 
milestone requested.  (You will also get a message about being in a "detached" 
HEAD state.  That is normal.)

For a version history and a description of how the current version of the code
differs from the last tagged version, see HISTORY.md and CHANGELOG.md
(respectively).  These files are found in the main GalSim directory, and are
also displayed on our wiki which is linked above.


Summary of current capabilities
-------------------------------

Currently, GalSim has the following capabilities:

* Can generate PSFs from a variety of simple parametric models and first-order
  optics.

* Can simulate galaxies from a variety of simple parametric models and based on
  HST training data.  Some additional testing will be done in future versions 
  to ensure that the treatment of the latter is sufficiently accurate to use 
  for precision tests of shear.

* Can make the images either via i) Fourier transform / real-space convolution 
  (real-space being occasionally faster than Fourier), and interpolation (for 
  shearing); or via ii) photon-shooting.  The exception is that simulations 
  based on real galaxies images must be carried out using Fourier methods only.

* Can add uncorrelated noise using a variety of noise models.

* Can draw galaxy images into arbitrary locations within a larger image.

* It is possible to carry out nearly any simulation that a user might want 
  using two parallel methods: directly using python code, or by specifying the
  simulation properties in an input configuration script.

* Constant shears and lensing magnifications can be applied to the galaxies.

* Non-constant shears and magnifications can be drawn from a shear field
  expected for an NFW profile dark matter halo (as for weak lensing by galaxy
  clusters).  For gridded galaxy positions, lensing shears can be drawn 
  randomly from a specified shear power spectrum, and they can then be 
  interpolated to non-gridded positions.

* Simulating correlated Gaussian noise fields as according to user-specified,
  correlation functions derived from images.


Summary of planned future development
-------------------------------------

In addition to carrying out further testing on some existing parts of the code,
we plan to add the following additional capabilities in future versions of
GalSim:

* PSFs from stochastic atmosphere models.

* Simulating simple detector defects or image artifacts.
