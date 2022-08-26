
Surface Brightness Profiles
###########################

There are many possible ways to define a surface brightness profile of an astronomical
object (stars, galaxies, etc.) in GalSim.  We have classes for common PSF models (e.g.
`Moffat` and `Airy`), analytic galaxy profiles (e.g. `Exponential` and `Sersic`), interpolation
of an arbitrary input image (`InterpolatedImage`), some other more complicated models,
and ways to combine models as sums or convolutions.  Models can also be arbitrarily
stretched, rotated, and dilated in various ways.

The classes for these various surface brightness profiles are all subclasses of the
`GSObject` base class.  See that section for details about most of the public API
methods that are defined for all (or almost all) of these classes.

Next, we list the subclasses of `GSObject` organized by their intended use.  These are
the classes you would actually use when building the profile for an  astronomical object.


.. toctree::
    :maxdepth: 2

    gsobject
    simple
    psf
    gal
    arbitrary
    phase_psf
    real_gal
    composite
    transform
    gsparams
