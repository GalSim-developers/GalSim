
The GSObject base class
=======================

This class defines most of the public API methods for how to use one of the various
surface brightness profiles like transforming it, drawing it, etc.

Note that not all methods are allowed to be called for all subclasses.  For instance,
some classes only define the profile in Fourier space, so methods which need to access
the profile in real space may not be implemented.  In such cases, a NotImplementedError
will be raised.


.. autoclass:: galsim.GSObject
    :members:
    :special-members:


