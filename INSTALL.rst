Installation Instructions
=========================

GalSim is a python module that has much of its implementation in C++ for
improved computational efficiency.  GalSim supports both Python 2 and
Python 3.  It is regularly tested on Python versions (2.7, 3.6, 3.7, 3.8, 3.9).

System requirements: GalSim currently only supports Linux and Mac OSX.
Possibly other POSIX-compliant systems, but we specifically do not
currently support Windows.

WARNING: The GalSim 2.3.x release series will be the last to support
Python 2.7, as it is currently past its end-of-life.  Please migrate to
Python 3 in order to be able to use future versions of GalSim.

Pip Installation
----------------

The usual way to install GalSim is now (starting with version 2.0) simply::

    pip install galsim

which will install the latest official release of GalSim.

Note that you may need to use sudo with the above command if you are installing
into system directories.  If you do not have write privileges for the directory
it is trying to install into, you can use the --user flag to install into a
local directory instead.  (Normally something like $HOME/Library/Python/3.8
or $HOME/.local, depending on your system.)

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
for you.

FFTW is not directly pip installable, so if the above installation fails,
you may need to install it separately.  See the link below for details
about how to do this.

For complete details, see:

http://galsim-developers.github.io/GalSim/docs/_build/html/install_pip.html

Conda Installation
------------------

Another option If you use Anaconda Python is to use ``conda``::

    conda install -c conda-forge galsim

The conda installation method will install all of the dependencies for you.
But if you want more information, see:

http://galsim-developers.github.io/GalSim/docs/_build/html/install_conda.html

SCons Installation
------------------

Prior to version 2.0, GalSim installation used SCons.  This installation
mode is still supported, but is not recommended unless you have difficulties
with the pip or conda installation methods.  The GalSim 2.3.x release series
will be the last to support the SCons installation method.

For details about this methos, see:

http://galsim-developers.github.io/GalSim/docs/_build/html/install_scons.html
