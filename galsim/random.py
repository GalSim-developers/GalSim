"""
Addition of docstrings to the Random deviate classes at the Python layer.
"""
from . import _galsim

# BaseDeviate docstring
_galsim.BaseDeviate.__doc__ = """
Base class for all the various random deviates.

This holds the essential random number generator that all the other classes use.

All deviates have three constructors that define different ways of setting up the random number
generator.

  1) Only the arguments particular to the derived class (e.g. mean and sigma for GaussianDeviate).
     In this case, a new random number generator is created and it is seeded using the computer's
     microsecond counter.

  2) Using a particular seed as the first argument to the constructor.
     This will also create a new random number generator, but seed it with the provided value.

  3) Passing another BaseDeviate as the first arguemnt to the constructor.
     This will make the new Deviate share the same underlying random number generator with the other
     Deviate.  So you can make one Deviate (of any type), and seed it with a particular
     deterministic value.  Then if you pass that Deviate to any other one you make, they will all be
     using the same rng and have a particular deterministic series of values.  (It doesn't have to
     be the first one -- any one you've made later can also be used to seed a new one.)
     
There is not much you can do with something that is only known to be a BaseDeviate rather than one
of the derived classes other than construct it and change the seed, and use it as an argument to
pass to other Deviate constructors.
"""

# UniformDeviate docstrings
_galsim.UniformDeviate.__doc__ = """
Pseudo-random number generator with uniform distribution in interval [0.,1.).

Initialization
--------------
>>> u = UniformDeviate()       # Initializes u to be a UniformDeviate instance, and seeds the PRNG
                               # using current time.

>>> u = UniformDeviate(lseed)  # Initializes u to be a UniformDeviate instance, and seeds the PRNG
                               # using specified long integer lseed.

>>> u = UniformDeviate(dev)    # Initializes u to be a UniformDeviate instance, and use the same RNG
                               # as dev.

Calling
-------
Taking the instance from the above examples, successive calls to u() then generate pseudo-random
numbers distributed uniformly in the interval [0., 1.).
"""

_galsim.UniformDeviate.applyTo.__doc__ = """
Add Uniform deviates to every element in a supplied Image.

Calling
-------
>>> UniformDeviate.applyTo(image)

On output each element of the input Image will have a pseudo-random UniformDeviate return value
added to it.
"""

# GaussianDeviate docstrings
_galsim.GaussianDeviate.__doc__ = """
Pseudo-random number generator with Gaussian distribution.

Initialization
--------------

>>> g = GaussianDeviate(mean=0., sigma=1.)

Initializes g to be a GaussianDeviate instance using the current time for the seed.

>>> g = GaussianDeviate(lseed, mean=0., sigma=1.)

Initializes g using the specified seed.

>>> g = GaussianDeviate(dev, mean=0., sigma=1.)

Initializes g to share the same underlying random number generator as dev.

Parameters:

    mean     optional mean for Gaussian distribution (default = 0.).
    sigma    optional sigma for Gaussian distribution (default = 1.).

Calling
-------
Taking the instance from the above examples, successive calls to g() then generate pseudo-random
numbers Gaussian-distributed with the provided mean, sigma.
"""

_galsim.GaussianDeviate.applyTo.__doc__ = """
Add Gaussian deviates to every element in a supplied Image.

Calling
-------
>>> GaussianDeviate.applyTo(image)

On output each element of the input Image will have a pseudo-random GaussianDeviate return value
added to it, with current values of mean and sigma.
"""
