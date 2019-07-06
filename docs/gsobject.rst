
The GSObject base class
=======================

There are many possible ways to define a surface brightness profile of an astronomical
object (stars, galaxies, etc.) in GalSim.  We have classes for common PSF models (e.g. 
`Moffat` and `Airy`), analytic galaxy profiles (e.g. `Exponential` and `Sersic`), interpolation
of an arbitrary input image (`InterpolatedImage`), some other more complicated models,
and ways to combine models as sums or convolutions.  Models can also be arbitrarily
stretched, rotated, and dilated in various ways.

The classes for these various surface brightness profiles are all subclasses of the
`GSObject` base class.  Thus, we will first describe the methods associated with this
base class, and then below we will describe the individual subclasses you would actually
use to define your object.

Note that not all methods are allowed to be called for all subclasses.  For instance,
some classes only define the profile in Fourier space, so methods which need to access
the profile in real space may not be implemented.  In such cases, a NotImplementedError
will be raised.


.. autoclass:: galsim.GSObject
    :members:


