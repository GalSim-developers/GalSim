Installing With Pip
===================

Overall summary
---------------

The usual way to install GalSim is now (starting with version 2.0) simply::

    pip install galsim

which will install the latest official release of GalSim.

Note that you may need to use sudo with the above command if you are installing
into system directories.  If you do not have write privileges for the directory
it is trying to install into, you can use the --user flag to install into a
local directory instead.  (Normally something like $HOME/Library/Python/2.7
or $HOME/.local, depending on your system.)

This might fail if certain libraries are installed in non-standard locations.
In this case, add the paths for these libraries to both the LIBRARY_PATH and
LD_LIBRARY_PATH environmental variables before running pip::

    export LIBRARY_PATH=$LIBARY_PATH:/path/to/lib:/other/path/to/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/lib:/other/path/to/lib

If you would rather install from source (e.g. to work on a development branch),
you can do::

    git clone git@github.com:GalSim-developers/GalSim.git
    cd GalSim
    pip install -r requirements.txt
    python setup.py install

(again possibly with either sudo or --user).

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
------------------------------

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
``python setup.py install``, then these will not be installed.

- Starlink (3.10.0)  (Improved WCS functionality)
- PyYaml (3.12)      (Reads YAML config files)
- Pandas (0.20)      (Faster reading of ASCII input files)

If you want to install these yourself, the quickest way is to do::

    pip install -r requirements.txt

If you want more control about which version you get or otherwise want to install
each package individually, you can do::

    pip install numpy
    pip install future
    pip install astropy
    pip install pybind11
    pip install LSSTDESC.Coord

    pip install starlink-pyast
    pip install pyyaml
    pip install pandas

In all cases, you may need to precede the above commands with ``sudo`` or
add ``--user`` to the end as you normally do when pip installing on your system.


Installing FFTW
---------------

GalSim uses FFTW (The Fastest Fourier Transform in the West) for performing
fast fourier transforms.

We require FFTW version >= 3.0.  Most tests have been done with FFTW 3.3.7,
so if you have trouble with an earlier version, try upgrading to 3.3.7 or later.

Installing FFTW yourself
^^^^^^^^^^^^^^^^^^^^^^^^

FFTW is available at the URL:

http://www.fftw.org/download.html

As of this writing, version 3.3.7 is the current latest release, for which
the following commands should work to download and install it::

    wget http://www.fftw.org/fftw-3.3.7.tar.gz
    tar xfz fftw-3.3.7.tar.gz
    cd fftw-3.3.7
    ./configure --enable-shared
    make
    sudo make install

If you want to install into a different directory (e.g. because you do not
have sudo privileges on your machine), then specify the alternate directory
with the --prefix flag to configure.  E.g.::

    ./configure --enable-shared --prefix=$HOME

which will install the library into $HOME/lib and the header file into
$HOME/include.  In this case, leave off the sudo from the last line.
Also, you should make sure these directories are in your LD_LIBRARY_PATH
and C_INCLUDE_PATH environment variables, respectively.

Alternatively, if you do not want to modify your LD_LIBRARY_PATH and/or
C_INCLUDE_PATH, you can instead set an environment variable to tell GalSim
where the files are::

    export FFTW_DIR=/path/to/fftw/prefix

E.g. in the above case where prefix is $HOME, you would do::

    export FFTW_DIR=$HOME

Probably, you should put this into your shell login file (e.g. .bash_profile)
so it always gets set when you log in.


Using an existing installation of FFTW
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If FFTW is already installed on your system, there may be nothing to do.
If it is in a standard location like /usr/local/lib or in some other
directory in your LD_LIBRARY_PATH, then GalSim should find it without
any extra work on your part.

If it is in a non-standard location, and you do not want to add this path
to your LD_LIBRARY_PATH (or you are on a modern Mac that hides such system
variables from setup.py), then you can instead set the FFTW_DIR environment
variable to tell GalSim where to look::

    export FFTW_DIR=/some/path/to/fftw

For instance, if libfftw3.so is located in /opt/cray/pe/lib64, you could use
that with::

    export FFTW_DIR=/opt/cray/pe/lib64

This command would normally be done in your .bash_profile file so it gets
executed every time you log in.

If you have multiple versions of FFTW installed on your system, this variable
can be used to specify which version you want GalSim to use as this will be
the first location it will check during the installation process.


Installing FFTW with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use conda, FFTW can be install with::

    conda install fftw

This will put it into the anaconda/lib directory on your system (within your
active environment if appropriate).  GalSim knows to look here, so there is
nothing additional you need to do.


Installing FFTW with apt-get
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On Linux machines that use apt-get, FFTW can be installed with::

    apt-get install libfftw3-dev


Installing FFTW with fink
^^^^^^^^^^^^^^^^^^^^^^^^^

If you use fink on a Mac, FFTW can be installed with::

    fink install fftw3

(Make sure to use fftw3, not fftw, since fftw is version 2.)

This will put it into the /sw/lib directory on your system. GalSim knows to
look here, so there is nothing additional you need to do.


Installing FFTW with MacPorts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use MacPorts, FFTW can be installed with::

    port install fftw-3

This will put it into the /opt/local/lib directory on your system. GalSim knows
to look here, so there is nothing additional you need to do.


Installing Eigen
----------------

GalSim uses Eigen for the C++-layer linear algebra calculations.  It is a
header-only library, which means that nothing needs to be compiled to use it.
You can download the header files yourself, but if you do not, then the
installation script will download it for you automatically.  So usually,
this dependency should require no work on your part.

However, if you have a version of Eigen already installed on your system,
you may want to use that.  If the right directory is in your path for
include file (C_INCLUDE_PATH), it should find it.  If not, you may specify
the right directory to use by setting the EIGEN_DIR environment variable.

We require Eigen version >= 3.0.  The version we download automatically is
3.3.4, so that version is known to work.  We have also tested with versions
3.2.8 and 3.0.4, so probably any 3.x version will work.  However, if you have
trouble with another version, try upgrading to 3.3.4 or later.

Note: Prior to version 2.0, GalSim used TMV for the linear algebra back end.
This is still an option if you prefer (e.g. it may be faster for some use
cases, since it can use an optimized BLAS library on your system), but to
use TMV, you need to use the SCons installation option described below.
(cf. `Installing With SCons`)


Installing Eigen yourself
^^^^^^^^^^^^^^^^^^^^^^^^^

Eigen is available at the URL

http://eigen.tuxfamily.org/index.php

As of this writing, version 3.3.4 is the current latest release, for which
the following commands should work to download and install it::

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
as an environment variable to tell GalSim where the files are::

    export EIGEN_DIR=/some/path/to/eigen

This should be the directory in which the Eigen subdirectory is found.  E.g.::

    export EIGEN_DIR=$HOME/eigen-eigen-5a0156e40feb

Probably, you should put this into your .bash_profile file so it always gets
set when you log in.


Using an existing installation of Eigen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If Eigen is already installed on your system, there may be nothing to do.
If it is in a standard location like /usr/local/include or in some other
directory in your C_INCLUDE_PATH, then GalSim should find it without
any extra work on your part.

If it is in a non-standard location, and you do not want to add this path
to your C_INCLUDE_PATH, then you can instead set the EIGEN_DIR environment
variable to tell GalSim where to look::

    export EIGEN_DIR=/some/path/to/eigen

For instance, if Eigen was installed into /usr/include/eigen3, then you
could use that with::

    export EIGEN_DIR=/usr/include/eigen3

This command would normally be done in your .bash_profile file so it gets
executed every time you log in.

If you have multiple versions of Eigen installed on your system, this variable
can be used to specify which version you want GalSim to use as this will be
the first location it will check during the installation process.


Installing Eigen with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use conda, Eigen can be install with::

    conda install eigen

This will put it into the anaconda/include directory on your system (within
your active environment if appropriate).  GalSim knows to look here, so there
is nothing additional you need to do.


Installing Eigen with apt-get
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On Linux machines that use apt-get, Eigen can be installed with::

    apt-get install libeigen3-dev


Installing Eigen with fink
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use fink on a Mac, Eigen can be installed with::

    fink install eigen

This will put it into the /sw/include directory on your system. GalSim knows
to look here, so there is nothing additional you need to do.


Installing Eigen with MacPorts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use MacPorts, Eigen can be installed with::

    port install eigen

This will put it into the /opt/local/include directory on your system. GalSim
knows to look here, so there is nothing additional you need to do.
