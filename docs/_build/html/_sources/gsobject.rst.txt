
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

    .. automethod:: galsim.GSObject.__add__
    .. automethod:: galsim.GSObject.__sub__
    .. automethod:: galsim.GSObject.__mul__
    .. automethod:: galsim.GSObject.__rmul__
    .. automethod:: galsim.GSObject.__div__
    .. automethod:: galsim.GSObject._xValue
    .. automethod:: galsim.GSObject._kValue
    .. automethod:: galsim.GSObject._shear
    .. automethod:: galsim.GSObject._shift
    .. automethod:: galsim.GSObject._drawReal
    .. automethod:: galsim.GSObject._calculate_nphotons
    .. automethod:: galsim.GSObject._shoot
    .. automethod:: galsim.GSObject._drawKImage

