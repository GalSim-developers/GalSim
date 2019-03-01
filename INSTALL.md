Installation Instructions
=========================

System requirements: GalSim currently only supports Linux and Mac OSX.

Table of Contents:

1) [Overall summary](#overall-summary)

2) [Installing FFTW](#installing-fftw)
   * [Installing FFTW yourself](#i-installing-fftw-yourself)
   * [Using an existing installation](#ii-using-an-existing-installation-of-fftw)
   * [Using conda](#iii-installing-fftw-with-conda)
   * [Using apt-get](#iv-installing-fftw-with-apt-get)
   * [Using fink](#v-installing-fftw-with-fink)
   * [Using MacPorts](#vi-installing-fftw-with-macports)

3) [Installing Eigen](#installing-eigen)
   * [Installing Eigen yourself](#i-installing-eigen-yourself)
   * [Using an existing installation](#ii-using-an-existing-installation-of-eigen)
   * [Using conda](#iii-installing-eigen-with-conda)
   * [Using apt-get](#iv-installing-eigen-with-apt-get)
   * [Using fink](#v-installing-eigen-with-fink)
   * [Using MacPorts](#vi-installing-eigen-with-macports)
   * [Using eigency](#vii-using-eigency)

4) [Using Conda](#using-conda)

5) [Installing With SCons](#installing-with-scons)

6) [Running tests](#running-tests)

7) [Running example scripts](#running-example-scripts)


Overall summary
===============

GalSim is a python module that has much of its implementation in C++ for
improved computational efficiency.  GalSim supports both Python 2 and
Python 3.  It is regularly tested on Python versions (2.7, 3.5, 3.6).

The usual way to install GalSim is now (starting with version 2.0) simply

    pip install galsim

which will install the latest official release of GalSim.

Note that you may need to use sudo with the above command if you are installing
into system directories.  If you do not have write privileges for the directory
it is trying to install into, you can use the --user flag to install into a
local directory instead.  (Normally something like $HOME/Library/Python/2.7
or $HOME/.local, depending on your system.)

This might fail if certain libraries are installed in non-standard locations.
In this case, add the paths for these libraries to both the LIBRARY_PATH and
LD_LIBRARY_PATH environmental variables before running pip:

	export LIBRARY_PATH=$LIBARY_PATH:/path/to/lib:/other/path/to/lib
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/lib:/other/path/to/lib

If you would rather install from source (e.g. to work on a development branch),
you can do

    git clone git@github.com:GalSim-developers/GalSim.git
    cd GalSim
    pip install -r requirements.txt
    python setup.py install

(again possibly with either sudo or --user).

**Note**: If you use Anaconda Python, you can use that to install most of the
requirements with their conda installer.  See [Using Conda](using-conda)
below.

Either of these installation methods should handle most of the required
dependencies for you if you do not have them already installed on your machine.
In particular, all of the python dependencies should be automatically installed
for you.  See section 2 below if you have trouble with any of these.

FFTW is not directly pip installable, so if the above installation fails,
you may need to install it separately. See section 3 below for more details
about how to do this.

Fianlly, a version of Eigen can be installed by pip, but you might prefer to
install this manually.  See section 4 below for more details.

Installing Python Dependencies
==============================

Normally, all of the python package dependencies will be automatically installed
by pip.  The following versions are known to work with GalSim 2.1.  In most cases,
other recent (especially later) versions will also work:

- NumPy (1.16.1)
- Future (0.17.1)
- Astropy (3.0.5)
- PyBind11 (2.2.3)
- LSSTDESC.Coord (1.0.5)

There are a few others modules are not technically required, but we let pip
install them along with GalSim, because they either add useful functionality
or efficiency to GalSim.  These are listed in the requirements.txt file that
pip uses to determine what else to install.  But if you install with
`python setup.py install`, then these will not be installed.

- Starlink (3.10.0)  (Improved WCS functionality)
- PyYaml (3.12)      (Reads YAML config files)
- Pandas (0.20)      (Faster reading of ASCII input files)

If you want to install these yourself, the quickest way is to do

    pip install -r requirements.txt

If you want more control about which version you get or otherwise want to install
each package individually, you can do

    pip install numpy
    pip install future
    pip install astropy
    pip install pybind11
    pip install LSSTDESC.Coord

    pip install starlink-pyast
    pip install pyyaml
    pip install pandas

In all cases, you may need to precede the above commands with `sudo` or
add `--user` to the end as you normally do when pip installing on your system.


Installing FFTW
===============

GalSim uses FFTW (The Fastest Fourier Transform in the West) for performing
fast fourier transforms.

We require FFTW version >= 3.0.  Most tests have been done with FFTW 3.3.7,
so if you have trouble with an earlier version, try upgrading to 3.3.7 or later.

i) Installing FFTW yourself
-------------------------

FFTW is available at the URL

    http://www.fftw.org/download.html

As of this writing, version 3.3.7 is the current latest release, for which
the following commands should work to download and install it:

    wget http://www.fftw.org/fftw-3.3.7.tar.gz
    tar xfz fftw-3.3.7.tar.gz
    cd fftw-3.3.7
    ./configure --enable-shared
    make
    sudo make install

If you want to install into a different directory (e.g. because you do not
have sudo privileges on your machine), then specify the alternate directory
with the --prefix flag to configure.  E.g.

    ./configure --enable-shared --prefix=$HOME

which will install the library into $HOME/lib and the header file into
$HOME/include.  In this case, leave off the sudo from the last line.
Also, you should make sure these directories are in your LD_LIBRARY_PATH
and C_INCLUDE_PATH environment variables, respectively.

Alternatively, if you do not want to modify your LD_LIBRARY_PATH and/or
C_INCLUDE_PATH, you can instead set an environment variable to tell GalSim
where the files are

    export FFTW_DIR=/path/to/fftw/prefix

E.g. in the above case where prefix is $HOME, you would do

    export FFTW_DIR=$HOME

Probably, you should put this into your shell login file (e.g. .bash_profile)
so it always gets set when you log in.


ii) Using an existing installation of FFTW
------------------------------------------

If FFTW is already installed on your system, there may be nothing to do.
If it is in a standard location like /usr/local/lib or in some other
directory in your LD_LIBRARY_PATH, then GalSim should find it without
any extra work on your part.

If it is in a non-standard location, and you do not want to add this path
to your LD_LIBRARY_PATH (or you are on a modern Mac that hides such system
variables from setup.py), then you can instead set the FFTW_DIR environment
variable to tell GalSim where to look

    export FFTW_DIR=/some/path/to/fftw

For instance, if libfftw3.so is located in /opt/cray/pe/lib64, you could use
that with

    export FFTW_DIR=/opt/cray/pe/lib64

This command would normally be done in your .bash_profile file so it gets
executed every time you log in.

If you have multiple versions of FFTW installed on your system, this variable
can be used to specify which version you want GalSim to use as this will be
the first location it will check during the installation process.


iii) Installing FFTW with conda
-------------------------------

If you use conda, FFTW can be install with

    conda install fftw

This will put it into the anaconda/lib directory on your system (within your
active environment if appropriate).  GalSim knows to look here, so there is
nothing additional you need to do.


iv) Installing FFTW with apt-get
--------------------------------

On Linux machines that use apt-get, FFTW can be installed with

    apt-get install libfftw3-dev


v) Installing FFTW with fink
----------------------------

If you use fink on a Mac, FFTW can be installed with

    fink install fftw3

(Make sure to use fftw3, not fftw, since fftw is version 2.)

This will put it into the /sw/lib directory on your system. GalSim knows to
look here, so there is nothing additional you need to do.


vi) Installing FFTW with MacPorts
---------------------------------

If you use MacPorts, FFTW can be installed with

    port install fftw-3

This will put it into the /opt/local/lib directory on your system. GalSim knows
to look here, so there is nothing additional you need to do.


Installing Eigen
================

GalSim uses Eigen for the C++-layer linear algebra calculations.  It is a
header-only library, which means that nothing needs to be compiled to use it.
You can download the header files yourself, but if you do not, then we use
the pip-installable eigency module, which bundles the header files in their
installed python directory.  So usually, this dependency should require no
work on your part.

However, it might become useful to install Eigen separately from eigency
e.g. if you want to upgrade to a newer version of Eigen than the one that is
bundled with eigency.  (Eigen 3.2.8 is bundled with eigency 1.77.)  Therefore,
this section describes several options for how to obtain and install Eigen.

We require Eigen version >= 3.0.  Most tests have been done with Eigen 3.2.8
or 3.3.4, but we have also tested on 3.0.4, so probably any 3.x version will
work.  However, if you have trouble with another version, try upgrading to
3.2.8 or later.

Note: Prior to version 2.0, GalSim used TMV for the linear algebra back end.
This is still an option if you prefer (e.g. it may be faster for some use
cases, since it can use an optimized BLAS library on your system), but to
use TMV, you need to use the SCons installation option described below.


i) Installing Eigen yourself
----------------------------

Eigen is available at the URL

    http://eigen.tuxfamily.org/index.php

As of this writing, version 3.3.4 is the current latest release, for which
the following commands should work to download and install it:

    wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
    tar xfj 3.3.4.tar.bz2
    sudo cp eigen-eigen-5a0156e40feb/Eigen /usr/local/include

In the final cp line, the MD5 hash (5a0156e40feb) will presumably change for
other versions, so use whatever directory tar expands into if you are using
a different version than 3.3.4.

If you do not have sudo privileges, you can copy to a different directory such
as $HOME/include instead and leave off the sudo from the cp command.  In this
case, make sure this directory is in your C_INCLUDE_PATH environment variable.

Finally, you can also skip the last command above and instead set EIGEN_DIR
as an environment variable to tell GalSim where the files are

    export EIGEN_DIR=/some/path/to/eigen

This should be the directory in which the Eigen subdirectory is found.  E.g.

    export EIGEN_DIR=$HOME/eigen-eigen-5a0156e40feb

Probably, you should put this into your .bash_profile file so it always gets
set when you log in.


ii) Using an existing installation of Eigen
-------------------------------------------

If Eigen is already installed on your system, there may be nothing to do.
If it is in a standard location like /usr/local/include or in some other
directory in your C_INCLUDE_PATH, then GalSim should find it without
any extra work on your part.

If it is in a non-standard location, and you do not want to add this path
to your C_INCLUDE_PATH, then you can instead set the EIGEN_DIR environment
variable to tell GalSim where to look

    export EIGEN_DIR=/some/path/to/eigen

For instance, if Eigen was installed into /usr/include/eigen3, then you
could use that with

    export EIGEN_DIR=/usr/include/eigen3

This command would normally be done in your .bash_profile file so it gets
executed every time you log in.

If you have multiple versions of Eigen installed on your system, this variable
can be used to specify which version you want GalSim to use as this will be
the first location it will check during the installation process.


iii) Installing Eigen with conda
--------------------------------

If you use conda, Eigen can be install with

    conda install eigen

This will put it into the anaconda/include directory on your system (within
your active environment if appropriate).  GalSim knows to look here, so there
is nothing additional you need to do.


iv) Installing Eigen with apt-get
---------------------------------

On Linux machines that use apt-get, Eigen can be installed with

    apt-get install libeigen3-dev


v) Installing Eigen with fink
-----------------------------

If you use fink on a Mac, Eigen can be installed with

    fink install eigen

This will put it into the /sw/include directory on your system. GalSim knows
to look here, so there is nothing additional you need to do.


vi) Installing Eigen with MacPorts
----------------------------------

If you use MacPorts, Eigen can be installed with

    port install eigen

This will put it into the /opt/local/include directory on your system. GalSim
knows to look here, so there is nothing additional you need to do.


vii) Using eigency
------------------

Eigency is a pip-installable module that bundles the Eigen header files, so it
can also be used to install these files on your system.  Indeed, as mentioned
above, we will use eigency automatically if Eigen is not found in one of the
above locations.  So the above installations will take precendence, but
eigency should work as a fall-back.

Note: At the time of this writing, installation of eigency depends on having
cython already installed.  I thought I fixed this with PR #26, but it was
not quite complete.  There is now an open PR #27, which I believe will
finish making pip install eigency work, even if you do not have cython
installed.  But for now, you can do

    pip install cython
    pip install eigency

(in that order) to get it to work.  Alternatively, you can use my (MJ) version
which is the source of PR #27.  This is pip installable as

    pip install rmjarvis.eigency


Using Conda
===========

If you use conda (normally via the Anaconda Python distribution), then all of
the prerequisites and galsim itself are available from the conda-forge channel,
so you can use that as follows:

    conda install -c conda-forge galsim

Also, if you prefer to use the defaults channel, then (at least as of this
writing), it had all the items in conda_requirements.txt, except for pybind11.
So if you have conda-forge in your list of channels, but it comes after
defaults, then that should still work and pybind11 will be the only one that
will need the conda-forge channel.


Installing With SCons
=====================

Prior to version 2.0, GalSim installation used SCons.  This installation
mode is still supported, but is not recommended unless you have difficulties
with the setup.py installation.

Note: Two options that are available with the SCons installation method,
but not the setup.py method, are (1) using TMV instead of Eigen for the linear
algebra back end, and (2) using Boost.Python instead of PyBind11 for the
wrapping the C++ code to be called from Python.  If you need either of these
options, then you should use the SCons installation.

See the file INSTALL_SCONS.md for complete details about this method of
installation.


Running tests
=============

You can run our test suite by typing

    python setup.py test

This should run all the python-layer tests with pytest and also compile and
run the C++ test suite.

There are a number of packages that are used by the tests, but which are not
required for GalSim installation and running.  These should be installed
automatically by the above command, but you can install them manually via

    pip install -r test_requirements.txt

(As usually, you may need to add either `sudo` or `--user`.)

By default, the tests will run in parallel using the pytest plugins
`pytest-xdist` and `pytest-timeout` (to manage how much time each test is
allowed to run).  If you want to run the python tests in serial instead,
you can do this via

    python setup.py test -j1

You can also use this to modify how many jobs will be spawned for running the
tests.

**Note**: If your system does not have `pytest` installed, and you do not want
to install it, you can run all the Python tests with the script run_all_tests
in the `tests` directory. If this finishes without an error, then all the tests
have passed.  However, note that this script runs more tests than our normal
test run using pytest, so it may take quite a while to finish.  (The *all* in
the file name means run all the tests including the slow ones that we normally
skip.)


Running example scripts
=======================

The `examples` directory has a series of demo scripts:

    demo1.py, demo2.py, ...

These can be considered a tutorial on getting up to speed with GalSim. Reading
through these in order will introduce you to how to use most of the features of
GalSim in Python.  To run these scripts, type (e.g.):

    python demo1.py

There are also a corresponding set of config files:

    demo1.yaml, demo2.yaml, ...

These files can be run using the executable `galsim`, and will produce the
same output images as the Python scripts:

    galsim demo1.yaml

They are also well commented, and can be considered a parallel tutorial for
learning the config file usage of GalSim.

All demo scripts are designed to be run in the `GalSim/examples` directory.
Some of them access files in subdirectories of the `examples` directory, so they
would not work correctly from other locations.
