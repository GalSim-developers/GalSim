Installation Instructions
=========================

System requirements: GalSim currently only supports Linux and Mac OSX.

Table of Contents:

1. Software required before building GalSim

2. Installing the GalSim Python package

3. Running tests and installing example executables

4. Platform-specific notes

5. More SCons options


1. Software required before building GalSim
===========================================

Please note: Mac users who want to use fink can skip down to Section 4.ii and use
that to satisfy all dependencies before installing.

i) Python (2.6 or 2.7 series), with some additional modules installed
---------------------------------------------------------------------

The interface to the GalSim code is via the Python package "galsim", and its
associated modules. Therefore you must have Python installed on your system.
Python is free, and available from a number of sources online (see below).
Currently GalSim only supports Python versions 2.6.X and 2.7.X, but earlier
versions of Python 2.X may be supported in future: please let us know via
Github if you need this support.

Many systems have a version of Python pre-installed. To check whether you
already have a compatible version, type

    python --version

at the terminal prompt. If you get a "Command not found" error, or the reported
version is not 2.6.X or 2.7.X, you will need to install 2.6 or 2.7 series 
Python. In this case you should read the "Getting Python and required modules" 
section below.

It may be that there is or soon will be more than one version of Python 
installed on your operating system, in which case please see the "Making sure
you are using the right Python" Section below.

### Getting Python and required modules ###

For a list of places to download Python, see http://www.python.org/getit/.

The GalSim package also requires

* the numerical Python module NumPy (http://www.numpy.org)

* the astronomical FITS file input/output module PyFITS
  (http://www.stsci.edu/institute/software_hardware/pyfits)

* Optional dependency: the Python YAML parser and emitter module PyYAML
  (http://pyyaml.org/wiki/PyYAML)
  Note: PyYAML is in fact only required for full config file parsing
  functionality, and can be omitted if users are happy to use JSON-style
  config parsing or prefer to write python scripts directly.

These should installed onto your Python system so that they can be imported by:

    >>> import numpy
    >>> import pyfits
    >>> import yaml [ if using the galsim_yaml executable or otherwise plan to
                      parse .yaml configuration files ]

within Python.  You can test this by loading up the Python interpreter for the
version of Python you will be using with the GalSim toolkit. This is usually
achieved by typing `python` or `/path/to/executable/bin/python` if your desired
Python is not the system default, and typing the `import` commands above.  If
you get no warning message, things are OK.

If you do not have these modules, follow the links above or alternatively try
`easy_install` (or equivalently `/path/to/executable/bin/easy_install` if your
desired Python is not the default).

As an example, if using the default system Python, connected to the internet
and with root/admin privileges simply type

    easy_install numpy
    easy_install pyfits
    easy_install pyyaml

at the prompt.  If not using an admin account, prefix the commands above with 
`sudo` and enter your admin password when prompted. The required modules should
then be installed.

See http://packages.python.org/distribute/easy_install.html#using-easy-install
for more details about the extremely useful `easy_install` feature.

### Third party-maintained Python packages ###

There are a number of third party-maintained packages which bundle Python with
many of the numerical and scientific libraries that are commonly used, and
many of these are free for non-commercial or academic use.

One good example of such a package, which includes all of the Python
dependencies required by GalSim (NumPy, PyFITS, PyYAML as well as SCons and
nosetests; see Section 2 below) is the Enthought Python Distribution (EPD; see
http://enthought.com/products/edudownload.php for the academic download
instructions). Other re-packaged Python downloads can be found at
http://www.python.org/getit/.

### Making sure you are using the right Python ###

Some users will find they have a few versions of Python around their operating
system (determined, for example, using "locate python" at the prompt). A common
way this will happen if there is already an older build (e.g. Python 2.4.X)
being used by the operating system and then you install a newer version from
one of the sources described above.

It will be important to make sure that the version of Python for which NumPy,
PyFITS and PyYAML etc. are installed is also the one being used for GalSim,
and that this is the one *you* want to use GalSim from! Knowing which installed
version of Python will be used is also important for the installation of the 
Boost libraries (see Section 1.v, below).

To check which Python is your default you can identify the location of the
executable by, for example, typing

    which python

at the prompt. This will tell you the location of the executable, something like
    /path/to/executable/bin/python

If this is not the Python you want, please edit your startup scripts (e.g.
.profile or .bashrc), and be sure to specify where your desired Python version
resides when installing the Boost C++ libraries (see Section 1.v).

See Section 4 of this document for some suggestions about getting Python, Boost
and all the other dependencies all working well together on your specific
system.


ii) SCons (http://www.scons.org)
--------------------------------

GalSim uses SCons to configure and run its installation procedures, and so SCons
needs to be installed in advance. Versions 2.0 and 2.1 of SCons get reasonable
testing with GalSim, but it should also work with 1.x versions. You can check
if it is installed, and if so which version, by typing

    scons --version

See Section 4 for some more suggestions about installing this on your platform.


iii) FFTW (http://www.fftw.org)
-------------------------------

These Fast Fourier Transform libraries must be installed as GalSim will link
to them during the build. We require version 3 (or greater presumably), which
is often distributed as fftw3.  See Section 4 for some suggestions about 
installing this on your platform.


iv) TMV (http://code.google.com/p/tmv-cpp/) (version >= 0.71 required)
-----------------------------------------------------------------------

GalSim uses the TMV library for its linear algebra routines. You should
download it from the site above and follow the instructions in its INSTALL
file for how to install it. Usually installing TMV just requires the command

    scons install PREFIX=<installdir>

but there are additional options you might consider, so you should read the TMV
INSTALL file for full details. Also note, you may not need to specify the
installation directory if you are comfortable installing it into /usr/local.
However, if you are trying to install it into a system directory then you need
to use sudo scons install [PREFIX=<installdir>].

Note: On Mac OS 10.7, the Apple BLAS library has problems when run using
multiple processes.  So if you have such a system, we recommend getting a 
different BLAS library, such as ATLAS (and making sure TMV finds it instead
of the system BLAS) or compiling TMV with no BLAS library at all (using 
the SCons option WITH_BLAS=false).  Otherwise, Galsim programs may hang
when run with multiple processes.  e.g. `scons tests` by default uses 
multiple processes, and multiple people reported problems using the Apple
system BLAS on OS 10.7.


v) Boost C++ (http://www.boost.org)
-----------------------------------

GalSim makes use of some of the Boost C++ libraries, and these parts of Boost
must be installed. It is particularly important that your installed Boost
library links to the same version of Python which which you will be using
GalSim and on which you have installed NumPy and PyFITS (see Section ii, above).
Boost can be downloaded from the above website, and must be installed per the
(rather limited) instructions there, which essentially amount to using a command

    ./bootstrap.sh

(Additional `bootstrap.sh` options may be necessary to ensure Boost is built
against the correct version of Python; see below).

followed by

    ./b2 link=shared
    ./b2 --prefix=<INSTALL_DIR> link=shared install

If you are installing to a system directory, the second needs to be run as
root, of course: `sudo ./b2`...  Also, you should be aware that if you are
running `b2` a second time, you should use `b2 -a` to tell boost to 
recompile everything rather than use the existing libraries.

The `link=shared` is necessary to ensure that they are built as shared 
libraries; this is automatic on some platforms, but not all.

Note: if you do not want to install everything related to Boost (which takes a
while), you can restrict to Boost Python and math by using `--with-python`
`--with-math` on the `./b2` commands.  Currently we are only using Boost Python,
but there may be parts of the math library we want to use so installing these
two will likely be sufficient for the foreseeable future.

Once you have installed Boost, you can check that it links to the version of
Python that will be used for GalSim and on which you have installed NumPy and
PyFITS by typing

    ldd <YOUR_BOOST_LIB_DIR>/libboost_python<POSSIBLE_SUFFIX>.so (Linux)
    otool -L <YOUR_BOOST_LIB_DIR>/libboost_python<POSSIBLE_SUFFIX>.dylib (OSX)

(If the ldd command on Linux does not show the Python version, the command
`ls -l <YOUR_BOOST_LIB_LOCATION>/libboost_python*` may show the version of
libboost_python.so linked to, for example, `libboost_python_py26.so.1.40.0`.
In such a case you can tell both the Python and Boost versions being used, 2.6
and `1.40.0`, respectively, in this example.)  On some Linux systems,
ldd will not indicate the Python library against which boost was
compiled; in this case, continue with the installation procedure and
any issues will be revealed at a later stage.

If the Python library listed is the one you will be using, all is well. If not,
Boost can be forced to use a different version by specifying the following
options to the ./bootstrap.sh installation script (defaults in `[]` brackets):

* `--with-python=PYTHON` specify the Python executable [python]

* `--with-python-root=DIR` specify the root of the Python installation
                           [automatically detected, but some users have found
                           they have to force it to use a specific one because
                           it detected the wrong one]


2. Installing the GalSim Python package
=======================================

Once you have installed all the dependencies described above, you are ready to
build GalSim. From the GalSim base directory (in which this INSTALL.md file is
found) type

    scons

If everything above was installed in fairly standard locations, this may work
the first time. Otherwise, you may have to tell SCons where to find some of
those libraries. There are quite a few options that you can use to tell SCons
where to look, as well as other things about the build process. To see a list
of options you can pass to SCons, type

    scons -h

(See also Section 5 below.)

As an example, to specify where your TMV library is located, you can type

    scons TMV_DIR=<tmv-dir>

where `<tmv-dir>` would be the same as the `PREFIX` you specified when 
installing TMV, i.e. The TMV library and include files are installed in 
`<tmv-dir>/lib` and `<tmv-dir>/include`. Some important options that you may 
need to set are:

* `TMV_DIR`: Explicitly give the TMV prefix

* `FFTW_DIR`: Explicitly give the FFTW prefix

* `BOOST_DIR`: Explicitly give the Boost prefix

* `EXTRA_LIBS`: Additional libraries to send to the linker

* `EXTRA_INCLUDE_PATH`: Extra paths for header files (separated by : if more
                        than 1)

* `EXTRA_FLAGS`: Extra flags to send to the compiler other than what is
                 automatically used. (e.g. -m64 to force 64 bit compilation)
    
Again, you can see the full list of options using `scons -h`.

Another common option is `CXX=<c++compiler>`. So, to compile with `icpc` rather
than the default `g++`, type

    scons CXX=icpc

One nice feature of SCons is that once you have specified a parameter, it will
save that value for future builds in the file `gs_scons.conf`, so once you have
the build process working, for later builds you only need to type `scons`. It
can also be useful to edit this file directly -- mostly if you want to unset a
parameter and return to the default value, it can be easier to just delete the
line from this file, rather than explicitly set it back to the default value.

SCons caches the results of the various checks it does for the required
external libraries (TMV, Boost, etc.). This is usually very helpful, since
they do not generally change, so it makes later builds much faster.  However,
sometimes (rarely) SCons can get confused and not realized that things on your
system have changed, which might cause problems for you. You can delete
everything scons knows about what it has tried to build previously with

    /bin/rm -rf .scon*

This will force SCons to recheck and recompile everything from scratch.

Once you have a successful build, you can install the GalSim library, Python
modules, and header files into standard locations (like `/usr/local` and your
Python site-packages directory) with

    scons install

or

    sudo scons install

If you want to install into a different location, the prefix for the library
and header files can be specified with `PREFIX=<installdir>`, and the location
for the Python modules can be specified with `PYPREFIX=<pythondir>`. So the
command would be

    scons install PREFIX=<installdir> PYPREFIX=<pythondir>

Note: if you specify a specific directory for the Python modules with PYPREFIX,
this directory should be in the sys.path search path for the version of 
Python you are using.  You can check with 

    python -c "import sys; print sys.path"

If your PYPREFIX directory is not there, then Python will not be able to find
the installed galsim module.  You should therefore add this directory to your
PYTHONPATH environment variable.  For example, if you use bash, then you 
should add the line

    export PYTHONPATH=$PYTHONPATH:<pythondir>

where `<pythondir>` is the same directory you used above for `PYPREFIX`.

The installed files can be removed with the command

    scons uninstall

Finally, to clean all compiled objects from the GalSim directory, you can use

    scons -c

This is rather like a "make clean" command.

If you are having trouble with installing, you may find some helpful hints at
the GalSim Installation FAQ page on the Wiki:
https://github.com/GalSim-developers/GalSim/wiki/Installation%20FAQ

If you are still having trouble, please consider opening a new Issue on the
GalSim Github page at https://github.com/GalSim-developers/GalSim to help find
a solution for the problem.


3. Running tests and installing example executables
===================================================

You can run our test suite by typing

    scons tests

This should compile the test suite and run it. The tests of our C++ library
will always be run, but we use `nosetests` for our Python test suite, so that 
will only be run if `nosetests` is present on your system.  We do not require 
this as a dependency, since you can still do everything with the GalSim library
without this.  But it is required for a complete run of the test suite.

To install `nosetests`, you can also use easy_install as described in Section 1
above (see also http://nose.readthedocs.org/en/latest/). Many third party-
maintained Python distributions, such as the Enthought Python Distribution,
include `nosetests`.

Note: if your system does not have `nosetests` installed, and you do not want to
install it, you can run all the Python tests with the script run_all_tests in
the tests directory. If this finishes without an error, then all the tests
have passed.

There are also some example executables that come with GalSim, and will be
useful for testing and development particularly outside the Python layer. These
can be built by typing

    scons examples

The executables will then be visible in `GalSim/bin/`.

The examples directory also has a series of demo scripts:
    demo1.py, demo2.py, ...

These can be considered a tutorial on getting up to speed with GalSim. Reading
through these in order will introduce you to how to use most of the features of
GalSim in Python.

There are also a corresponding set of config files:
    demo1.yaml, demo2.yaml, ...

These files can be run using the executable galsim_yaml, and will produce the
same output images as the Python scripts. They are also well commented, and
can be considered a parallel tutorial for learning the config file usage of
GalSim.

All demo scripts are designed to be run in the `GalSim/examples` directory. Or,
the executable versions of the demos in `GalSim/bin` (which are the outputs of
running `scons examples`) can be run from either `GalSim/bin` or
`GalSim/examples`. 


4. Platform-specific notes
==========================

i) Linux
--------
The vast majority of Linux distributions provide installable packages for most
of our dependencies. In may cases, however, it is also necessary to install
"-devel" or "-dev" packages (e.g. `python-dev` or `libboost-dev` on Debian-
derivatives). However, as above we stress that you should make sure that the
version of Python that Boost is built against must be the same as that you
intend to use for running GalSim.

The solution may be to install Boost C++ manually. This can be done by following
the instructions of Section 1.v), above.

ii) Mac OSX
-----------
a) Use of Fink -- the `fink` (http://www.finkproject.org) package management 
software is popular with Mac users.  Once it is installed, you can get either
most or all of the prerequisites using it, depending on whether you want
to use GalSim with the fink version of Python (e.g. that in `/sw/bin/python`) or
the system Python (`/usr/bin/python`) or something else still.

It is in general a good idea to update fink prior to installing any new modules:

    fink selfupdate
    fink update-all

If you are happy with running GalSim using the fink version of python 2.7, you 
can install everything with the following command:

    fink install galsim

and it should just work.  However, there are some caveats that are worth knowing
about (assuming your fink installation is in `/sw`):

1. This will install GalSim as a module of the python2.7 installed by fink.  
This is not the default Python (usually `/usr/bin/python` or some other package,
such as EPD, if installed).  Any Python scripts you write that use the galsim 
module should therefore have `#!/sw/bin/python2.7` as the  first line rather 
than the usual `#!/usr/bin/env python`.  Similarly, if you want to use galsim 
in an interactive Python session, you should run `/sw/bin/python2.7` (simply 
`python2.7` may also work) rather than just `python`.  (Of course, you can 
always change your `PATH` environment variable to make the fink Python the 
system default if you wish...)

2. The executables `galsim_yaml` and `galsim_json`, which parse YAML and JSON 
configuration files, will be installed in `/sw/bin`.  You should not need to do
anything special to use these, since `/sw/bin` should already be in your path if
using fink.

3. If you want to run through the example scripts (such as the demo tutorials 
`demo1.py`, `demo2.py` etc. and the `.yaml` and `.json` config versions of the 
same demos), you will still need to download the GalSim tarball.  But you can 
skip all the instructions above about installation and just use the fink 
version.  So `python2.7 demo1.py` (assuming `which python2.7` is the fink one) 
and `galsim_yaml demo1.yaml` should run those scripts for you.   

4. At the end of the fink installation, there will be a WARNING message.  Just 
ignore it.  It happens because fink installs things a bit differently from how 
we do it in the local build directory.  There are plans to suppress the warning
in an imminent release version.

If you want to work with GalSim as a developer, rather than just a user, then 
you cannot use the fink-installed GalSim.  However, the process above will have
installed all the prerequisites.  So `fink uninstall galsim` will leave you able
to install GalSim using the master branch with:

    scons TMV_DIR=/sw PYTHON=/sw/bin/python2.7

from within the repository base directory.

To run the unit tests, you will also need nosetests, which you can also get from
fink:

    fink install nose-py27
    scons tests NOSETESTS=/sw/bin/nosetests

If you want to use the system Python, or some other version, then the fink 
Python installations will not work.  You will need to manually install
NumPy, PyFITS, PyYAML and nose, for example using easy_install, with your
chosen Python. 

For the system Python, you can use fink for Boost, but you will want a 
different package than the boost1.35.python27 that gets installed using
`fink install galsim` above:

    fink install scons fftw3 tmv0 boost1.46.1.cmake
    scons TMV_DIR=/sw

For other Python versions, the fink-installed Boost usually will not work, so
you can only use fink for SCons, FFTW and TMV.  So you will probably need to 
install Boost manually.  This can be done by following the instructions of 
Section 1.v), above.

b) MacPorts -- this is another popular Mac package management project
(http://www.macports.org/) with similar functionality to fink.  Neither TMV nor
GalSim are currently on the official MacPorts distribution list, so users cannot
find them by searching the MacPorts site.  However, it is possible to install
both TMV and GalSim, plus the other dependencies of GalSim, using MacPorts
following the instructions below.

It is in general a good idea to upgrade all modules, prior to installing any new
modules:

    sudo port selfupdate
    sudo port upgrade outdated

Below is a list of steps to take to install GalSim using MacPorts:

    i) Take the `Portfiles` from the GalSim repository:
    https://github.com/GalSim-developers/GalSim/blob/master/devel/ports.tar.gz
    (If you do not clone the repository, there is a "copy" button on the website
    that you can use to download the file directly.)
    ii) Place the file in your home directory.
    iii) `tar xvzf ports.tar.gz`
    iv) `cd ports`
    v) `sudo portindex`
    vi) `sudo port install python27`
    vii) `sudo port select --set python python27`
    viii) `sudo sh -c "echo file:///Users/username/ports >>
          /opt/local/etc/macports/sources.conf"`
    ix) `sudo port install galsim`
    x) Add /opt/local/lib to DYLD_LIBRARY_PATH

Some users may find that the last step results in an inability to import the
GalSim module.  In that case, you can clear that addition to DYLD_LIBRARY_PATH
and instead add /opt/local/lib to DYLD_FALLBACK_LIBRARY_PATH.

Notes on MacPorts with Mac OS X 10.8:
The use of `sudo` in the above commands may elicit an error message that says
"dyld: DYLD_ environment variables being ignored because main executable
(/usr/bin/sudo) is setuid or setgid".  This is the result of a bug in Mac OS X
10.8, and will not prevent the installation of GalSim with the above steps from
being successful.

Notes on MacPorts version of gcc with Mac OS X 10.5.8:
If you have installed a MacPorts version of gcc (e.g., "mp-gcc47"), it may not 
link correctly with the other MacPorts installed modules, which are compiled in
the system gcc versions.  To check what gcc versions are available to you, try
the command

    port select --list gcc

then switch to the system gcc version (either 4.0 or 4.2) with

    sudo port select --set gcc gcc42

and compile GalSim with the system gcc.

c) Homebrew (http://mxcl.github.com/homebrew/) -- another package manager for
Max OSX.  Currently GalSim is available on homebrew, so it (plus dependencies)
should be installable via

    brew tap camphogg/science
    brew install gal-sim


5. More SCons options
=====================

Here is a fairly complete list of the options you can pass to SCons to control
the build process. The options are listed with their default value. You change
them simply by specifying a different value on the command line.

For example:

    scons CXX=icpc TMV_DIR=~

(Unlike autotools, SCons correctly expands ~ to your home directory.)
You can list these options from the command line with

    scons -h

### Basic flags about the C++ compilation (default values in parentheses) ###

* `CXX` (g++) specifies which C++ compiler to use.

* `FLAGS` ('') specifies the basic flags to pass to the compiler.  The default 
   behavior is to automatically choose good flags to use according to which 
   kind of compiler you are using. This option overrides that and lets you 
   specify exactly what flags to use.

* `EXTRA_FLAGS` ('') specifies some extra flags that you want to use in addition
   to the defaults that SCons determines on its own. Unlike the above option, 
   this do not override the defaults, it just adds to them.

* `DEBUG` (True) specifies whether to keep the debugging assert statements in
   the compiled library code. They are not much of a performance hit, so it is
   generally worth keeping them in, but if you need to squeeze out every last 
   bit of performance, you can set this to False.

* `WARN` (False) specifies whether to add warning compiler flags such as 
   `-Wall`.  Developers should set this to True and fix everything that comes 
   up as a warning (on both `g++` and `icpc`).  However, end users can leave
   it as False in case their compiler is a stickler for something that did not
   get caught in development.  TODO: Move this prescription that developers 
   should set it to True somewhere else. Maybe in the credo.txt file?

* `PYTHON` (/usr/bin/env python) specifies which version of Python you are 
   planning to use GalSim with.  If you choose not to use the default here, 
   then you need to remember to use the correct Python version 

### Flags about where to install the library and modules ###

* `PREFIX` (/usr/local) specifies where to install the library when running 
   `scons install`.

* `PYPREFIX` ([your python dir]/site-packages) specifies where to install the
   Python modules when running `scons install`.

* `FINAL_PREFIX` (`PREFIX`) specifies the final installation prefix if different
   from PREFIX.  (This is only needed for things like fink, where they install 
   into a staging area first before copying over to the final location.)

### Flags that specify where to look for external libraries ###

* `TMV_DIR` ('') specifies the location of TMV if it is not in a standard
   location. This should be the same value as you used for PREFIX when 
   installing TMV.

* `TMV_LINK` ('') specifies the location of the tmv-link file. Normally, this is
   in `TMV_DIR/share`, but if not, you can specify the correct location here.

* `FFTW_DIR` ('') specifies the root location of FFTW. The header files should
   be in `FFTW_DIR/include` and the library files in `FFTW_DIR/lib`.

* `BOOST_DIR` ('') specifies the root location of BOOST The header files should
   be in `BOOST_DIR/include/boost` and the library files in `BOOST_DIR/lib`.

* `EXTRA_INCLUDE_PATH` ('') specifies extra directories in which to search for
   header files in addition to the standard locations such as `/usr/include` and
   `/usr/local/include` and the ones derived from the above options.  Sometimes
   the above options do not quite work, so you may need to specify other 
   locations, which is what this option is for.  These directories are specified
   as `-I` flags to the compiler.  If you are giving multiple directories, they
   should be separated by colons.

* `EXTRA_LIB_PATH` ('') specifies extra directories in which to search for
   libraries in addition to the standard locations such as `/usr/lib` and 
   `/usr/local/lib`.  These directories are specified as `-L` flags to the 
   linker. If you are giving multiple directories, they should be separated by 
   colons.  To add the library `/blah/libfoo.a`, specify 
   `EXTRA_LIB_PATH=/blah/ EXTRA_LIBS=foo`.

* `EXTRA_PATH` ('') specifies directories in which to search for executables
   (notably the compiler, although you can also just give the full path in the 
   CXX parameter) in addition to the standard locations such as `/usr/bin` and 
   `/usr/local/bin`.  If you are giving multiple directories, they should be 
   separated by colons.

* `IMPORT_PATHS` (False) specifies whether to import extra path directories
   from the environment variables: `PATH`, `C_INCLUDE_PATH`, `LD_LIBRARY_PATH` 
   and `LIBRARY_PATH`.  If you have a complicated setup in which you use these 
   environment variables to control everything, this can be an easy way to let 
   SCons know about these locations.

* `IMPORT_ENV` (True) specifies whether to import the entire environment from
   the calling shell.  The default is for SCons to use the same environment as 
   the shell from which it is called.  However, sometimes it can be useful to 
   start with a clean environment and manually add paths for various things, in
   which case you would want to set this to False.

* `EXTRA_LIBS` ('') specifies libraries to use in addition to what SCons finds
   on its own. This might be useful if you have a non-standard name for one of 
   the external libraries. e.g. If you want to use the Intel MKL library for the
   FFTW library, SCons will not automatically try that, so you could add those
   libraries here.  If there is more than one, they should be quoted with spaces
   between the different libraries. e.g. 
   `EXTRA_LIBS="mkl_intel mkl_intel_thread mkl_core"`

* `IMPORT_PREFIX` (True) specifies whether to include the directories
   `PREFIX/include`, `PREFIX/lib` and `PREFIX/bin` as part of the standard
   path lists.  Normally, you install everything in the same place, so it is 
   useful to search those locations for some of the prerequisite packages, so 
   the default is True.  But occasionally, this might be inconvenient, so you 
   can turn this feature off.

### Miscellaneous flags ###

* `NOSETESTS` (nosetests) specifies which version of nosetests you want to use
   for running the unit tests.  If you specified a non-default Python, then 
   there is a possibility that the standard nosetests executable in your path 
   will not work (since it might be for a different version of Python).  In 
   that case, specify the correct nosetests here.

* `CACHE_LIB` (True) specifies whether to cache the results of the library 
   checks.  While you are working one getting the prerequisites installed
   properly, it can be useful to set this to False to force SCons to redo all of
   its library checks each time. Once you have a successful build, you should 
   set it back to True so that later builds can skip those checks.

* `WITH_PROF` (False) specifies whether to use the compiler flag `-pg` to
   include profiling info for `gprof`.

* `MEM_TEST` (False) specifies whether to test the code for memory leaks.

* `TMV_DEBUG` (False) specifies whether to turn on extra (slower) debugging
   statements within the TMV library.

* `WITH_OPENMP` (False) specifies whether to use OpenMP to parallelize some 
   parts of the code. (Note: We do not actually use OpenMP currently, so this
   does not do anything.)

* `USE_UNKNOWN_VARS` (False) specifies whether to accept scons parameters other
   than the ones listed here.  Normally, another name would indicate a typo, so
   we catch it and let you know.  But if you want to use other scons options 
   that we did not list here, you would want to also set this to True.

