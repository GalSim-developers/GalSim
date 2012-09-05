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

Methods
-------
To add deviates to every element of an image, see the docstring for the .applyTo() method of each
instance.

This docstring can be found using the Python interpreter or in pysrc/Random.cpp.
"""

#_galsim.UniformDeviate.applyTo.__doc__ = """
#Add Uniform deviates to every element in a supplied Image.
#
#Calling
#-------
#>>> UniformDeviate.applyTo(image)
#
#On output each element of the input Image will have a pseudo-random UniformDeviate return value
#added to it.
#"""

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
numbers which Gaussian-distributed with the provided mean, sigma.

Methods
-------
To add deviates to every element of an image, see the docstring for the .applyTo() method of each
instance.

To get and set the deviate parameters, see the docstrings for the .getN(), .setN(), .getSigma() and
.setSigma() methods of each instance.

These docstrings can be found using the Python interpreter or in pysrc/Random.cpp.
"""

_galsim.BinomialDeviate.__doc__ = """
Pseudo-random Binomial deviate for N trials each of probability p.

N is number of 'coin flips,' p is probability of 'heads,' and each call returns an integer value
where 0 <= value <= N giving number of heads.

Initialization
--------------

>>> b = BinomialDeviate(N=1., p=0.5)

Initializes b to be a BinomialDeviate instance using the current time for the seed.

>>> b = BinomialDeviate(lseed, N=1., p=0.5)

Initializes b using the specified seed.

>>> b = BinomialDeviate(dev, N=1., p=0.5)

Initializes b to share the same underlying random number generator as dev.

Parameters:

    N   optional number of 'coin flips' per trial (default `N = 1`).
    p   optional probability of success per coin flip (default `p = 0.5`).

Calling
-------
Taking the instance from the above examples, successive calls to b() then generate pseudo-random
numbers binomial-distributed with the provided N, p, which must both be > 0.

Methods
-------
To add deviates to every element of an image, see the docstring for the .applyTo() method of each
instance.

To get and set the deviate parameters, see the docstrings for the .getN(), .setN(), .getP() and
.setP() methods of each instance.

These docstrings can be found using the Python interpreter or in pysrc/Random.cpp.
"""

_galsim.PoissonDeviate.__doc__ = """
Pseudo-random Poisson deviate with specified mean.

The input mean sets the mean and variance of the Poisson deviate.  An integer deviate with this
distribution is returned after each call.

Initialization
--------------

>>> p = PoissonDeviate(mean=1.)

Initializes g to be a PoissonDeviate instance using the current time for the seed.

>>> p = PoissonDeviate(lseed, mean=1.)

Initializes g using the specified seed.

>>> p = PoissonDeviate(dev, mean=1.)

Initializes g to share the same underlying random number generator as dev.

Parameters:

    mean     optional mean of the distribution (default `mean = 1`).

Calling
-------
Taking the instance from the above examples, successive calls to p() will return successive,
pseudo-random Poisson deviates with specified mean, which must be > 0.

Methods
-------
To add deviates to every element of an image, see the docstring for the .applyTo() method of each
instance.

To get and set the deviate parameter, see the docstrings for the .getMean(), method of each
instance.

These docstrings can be found using the Python interpreter or in pysrc/Random.cpp.
"""

_galsim.CCDNoise.__doc__ = """
Pseudo-random number generator with a basic CCD noise model.

A CCDNoise instance is initialized given a gain level in Electrons per ADU used for the Poisson
noise term, and a Gaussian read noise in electrons (if gain > 0.) or ADU (if gain <= 0.).  With 
these parameters set, the CCDNoise operates on an Image, adding noise to each pixel following this 
model. 

Initialization
--------------

>>> ccd_noise = CCDNoise(gain=1., read_noise=0.)

Initializes ccd_noise to be a CCDNoise instance using the current time for the seed.

>>> ccd_noise = CCDNoise(lseed, gain=1., read_noise=0.)

Initializes ccd_noise to be a CCDNoise instance using the specified seed.

>>> ccd_noise = CCDNoise(dev, gain=1., read_noise=0.)

Initializes ccd_noise to share the same underlying random number generator as dev.

Parameters:

    gain        the gain for each pixel in electrons per ADU; setting gain <=0 will shut off the
                Poisson noise, and the Gaussian rms will take the value read_noise as being in units
                of ADU rather than electrons [default=1.].
    read_noise  the read noise on each pixel in electrons (gain > 0.) or ADU (gain <= 0.)
                setting read_noise=0. will shut off the Gaussian noise [default=0.].

Calling
-------
Taking the instance from the above examples, successive calls to ccd_noise() will generate noise 
following this model.

Methods
-------
To add deviates to every element of an image, see the docstring for the .applyTo() method of each
instance.

To get and set the deviate parameters, see the docstrings for the .getGain(), .setGain(), 
.getReadNoise() and .setReadNoise() methods of each instance.

These docstrings can be found using the Python interpreter or in pysrc/Random.cpp.
"""

_galsim.WeibullDeviate.__doc__ = """
Pseudo-random Weibull-distributed deviate for shape parameter a & scale parameter b.

The Weibull distribution is related to a number of other probability distributions;  in particular,
it interpolates between the exponential distribution (a=1) and the Rayleigh distribution (a=2). 
See http://en.wikipedia.org/wiki/Weibull_distribution (a=k and b=lambda in the notation adopted in 
the Wikipedia article) for more details.  The Weibull distribution is real valued and produces 
deviates >= 0.

Initialization
--------------

>>> w = WeibullDeviate(a=1., b=1.)

Initializes w to be a WeibullDeviate instance using the current time for the seed.

>>> w = WeibullDeviate(lseed, a=1., b=1.)

Initializes w using the specified seed.

>>> w = WeibullDeviate(dev, a=1., b=1.)

Initializes w to share the same underlying random number generator as dev.

Parameters:

    a        shape parameter of the distribution (default a = 1).
    b        scale parameter of the distribution (default b = 1).

a and b must both be > 0.

Calling
-------
Taking the instance from the above examples, successive calls to w() then generate pseudo-random 
numbers Weibull-distributed with shape and scale parameters a and b.

Methods
-------
To add deviates to every element of an image, see the docstring for the .applyTo() method of each
instance.

To get and set the deviate parameters, see the docstrings for the .getA(), .setA(), .getB() and 
.setB() methods of each instance.

These docstrings can be found using the Python interpreter or in pysrc/Random.cpp.
"""

_galsim.GammaDeviate.__doc__ = """
Pseudo-random Gamma-distributed deviate for parameters alpha & beta.

See http://en.wikipedia.org/wiki/Gamma_distribution (although note that in the Boost random routine
this class calls the notation is alpha=k and beta=theta).  The Gamma distribution is a real valued
distribution producing deviates >= 0.

Initialization
--------------

>>> gam = GammaDeviate(alpha=1., beta=1.)

Initializes gam to be a GammaDeviate instance using the current time for the seed.

>>> gam = GammaDeviate(lseed, alpha=1., beta=1.)

Initializes gam using the specified seed.

>>> gam = GammaDeviate(dev alpha=1., beta=1.)

Initializes gam to share the same underlying random number generator as dev.

Parameters:

    alpha    shape parameter of the distribution (default alpha = 1).
    beta     scale parameter of the distribution (default beta = 1).

alpha and beta must both be > 0.

Calling
-------
Taking the instance from the above examples, successive calls to g() will return successive, 
pseudo-random Gamma-distributed deviates with shape and scale parameters alpha and beta. 

Methods
-------
To add deviates to every element of an image, see the docstring for the .applyTo() method of each
instance.

To get and set the deviate parameters, see the docstrings for the .getAlpha(), .setAlpha(), 
.getBeta() and .setBeta() methods of each instance.

These docstrings can be found using the Python interpreter or in pysrc/Random.cpp.
"""



