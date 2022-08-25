Bessel Functions
================

These are probably not super useful for most users.  They should all be equivalent to
the scipy bessel functions.  However, in C++ we wanted to avoid dependencies that would
have given us these Bessel functions, so we implemented our own.  The Python interface
is mostly to enable unit tests that these C++ function are correct.

.. autofunction:: galsim.bessel.j0
.. autofunction:: galsim.bessel.j1
.. autofunction:: galsim.bessel.jv
.. autofunction:: galsim.bessel.jn
.. autofunction:: galsim.bessel.kv
.. autofunction:: galsim.bessel.kn
.. autofunction:: galsim.bessel.yv
.. autofunction:: galsim.bessel.yn
.. autofunction:: galsim.bessel.iv
.. autofunction:: galsim.bessel.j0_root
.. autofunction:: galsim.bessel.jv_root

The next few are not really related to Bessel functions, but they are also exposed
from the C++ layer.

.. autofunction:: galsim.bessel.si
.. autofunction:: galsim.bessel.ci
.. autofunction:: galsim.bessel.sinc
.. autofunction:: galsim.bessel.gammainc
