C++ Layer
#########

While GalSim is primarily a Python package, much of the implementation is done in C++
for improved speed for the hard-core numerical calculations.

If you would like to use the C++-layer functions in your own C++ project, you can do
so with the caveat that these are officially implementation details, so we don't
strictly enforce semantic versioning for the C++-layer API.  That is, function signatures
may change on minor version updates (e.g. 2.3.x to 2.4.0).  We don't often make huge
changes to the C++ API, so most of the time you will be fine upgrading, but you should
be prepared that you may need to update your code when upgrading GalSim.  (We do
guarantee that we won't change the C++ API for bugfix updates, e.g. 2.4.1 to 2.4.2.)

The other caveat is that we haven't put much energy into documenting the C++ layer
functions.  The following comes from a combination of Doxygen and Breathe to shoehorn
it into the Sphinx structure.  But the docs are pretty bare bones in places.  Sorry
about that.  If you use these and want to pretty up these docs, a PR doing so would be
much appreciated.  :)

When compiling your code, all of the public functionality should be included simply
by using ``#include "GalSim.h"`` with the appropriate ``-I`` directive when compiling
to find the GalSim include directory.  The appropriate directory name is accessible
from python by running the command:

.. code::

    python -c "import galsim; print(galsim.include_dir)"


.. toctree::
    :maxdepth: 1

    cpp_image
    cpp_sb
    cpp_bounds
    cpp_noise
    cpp_photon
    cpp_interp
    cpp_hsm
    cpp_math


Linking Your Code
=================

If you install GalSim using conda (see `Installing With Conda`), then the appropriate
C++ library file is included in the conda packaging.  It should be installed into either
the main conda lib directory (e.g. ``/anaconda/lib``) or the one for your conda environment
(e.g. ``/anaconda/envs/myenv/lib``).

If you don't use conda, then you will need to build the lib file yourself, since we don't
include it in the pip package. 
See `Installing the C++ Shared Library` for instructions on installing it.

There are both versioned and unversioned copies of the library.  On OSX, these are
``libgalsim.M.m.dylib`` and ``libgalsim.dylib``.  On Linux, they are
``libgalsim.M.m.so`` and ``libgalsim.so``.

You should link by specifying the appropriate directory with ``-L`` and link with ``-lgalsim``.

Version control
===============

We provide a number of functions to help you ensure that your code remains compatible with
updates to GalSim.

First, there are 3 MACROS that you can use to make a compile-time assert that you are
coding to the right version:

.. doxygendefine:: GALSIM_MAJOR

.. doxygendefine:: GALSIM_MINOR

.. doxygendefine:: GALSIM_REVISION

Then there are three functions that return the compiled versions of those same numbers:

.. doxygenfunction:: galsim::major_version

.. doxygenfunction:: galsim::minor_version

.. doxygenfunction:: galsim::revision

One can also get the full three-number version as a string (e.g. "1.4.2")

.. doxygenfunction:: galsim::version

And finally, we provide a fucntion that checks that the header file being included matches
the compiled values in the library being linked to:

.. doxygenfunction:: galsim::check_version



