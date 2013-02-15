# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file random.py 
Addition of docstrings to the Random deviate classes at the Python layer.
"""


from . import _galsim


def permute(rng, *args):
    """Randomly permute one or more lists.

    If more than one list is given, then all lists will have the same random permutation 
    applied to it.

    @param rng    The random number generator to use. (This will be converted to a UniformDeviate.)
    @param args   Any number of lists to be permuted.
    """
    ud = _galsim.UniformDeviate(rng)
    if len(args) == 0: return

    # We use an algorithm called the Knuth shuffle, which is based on the Fisher-Yates shuffle.
    # See http://en.wikipedia.org/wiki/Fisher-Yates_shuffle for more information.
    n = len(args[0])
    for i in range(n-1,1,-1):
        j = int((i+1) * ud())
        if j == i+1: j = i  # I'm not sure if this is possible, but just in case...
        for list in args:
            list[i], list[j] = list[j], list[i]



# BaseDeviate docstrings
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
     using the same RNG and have a particular deterministic series of values.  (It doesn't have to
     be the first one -- any one you've made later can also be used to seed a new one.)
     
There is not much you can do with something that is only known to be a BaseDeviate rather than one
of the derived classes other than construct it and change the seed, and use it as an argument to
pass to other Deviate constructors.

Examples
--------

    >>> rng = galsim.BaseDeviate(215324)    
    >>> rng()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: 'BaseDeviate' object is not callable
    >>> ud = galsim.UniformDeviate(rng)
    >>> ud()
    0.58736140513792634
    >>> ud2 = galsim.UniformDeviate(215324)
    >>> ud2()
    0.58736140513792634

"""

_galsim.BaseDeviate.seed.__func__.__doc__ = """
Seed the pseudo-random number generator.

Multiple Calling Options
------------------------

    >>> galsim.BaseDeviate.seed()       # Re-seed the PRNG using current time.

    >>> galsim.BaseDeviate.seed(lseed)  # Re-seed the PRNG using specified seed, where lseed is a
                                        # long int.

"""

_galsim.BaseDeviate.reset.__func__.__doc__ = """
Reset the pseudo-random number generator, severing connections to any other deviates.

Multiple Calling Options
------------------------

    >>> galsim.BaseDeviate.reset()        # Re-seed the PRNG using current time, and sever the
                                          # connection to any other Deviate.

    >>> galsim.BaseDeviate.reset(lseed)   # Re-seed the PRNG using specified seed, where lseed is a
                                          # long int, and sever the connection to any other Deviate.

    >>> galsim.BaseDeviate.reset(dev)     # Re-connect this Deviate with the same underlying random
                                          # number generator supplied in dev.

"""


# UniformDeviate docstrings
_galsim.UniformDeviate.__doc__ = """
Pseudo-random number generator with uniform distribution in interval [0.,1.).

Initialization
--------------

    >>> u = galsim.UniformDeviate()       # Initializes u to be a UniformDeviate instance, and seeds
                                          # the PRNG using current time.

    >>> u = galsim.UniformDeviate(lseed)  # Initializes u to be a UniformDeviate instance, and seeds
                                          # the PRNG using specified long integer lseed.

    >>> u = galsim.UniformDeviate(dev)    # Initializes u to be a UniformDeviate instance, and share
                                          # the same underlying random number generator as dev.

Calling
-------
Taking the instance from the above examples, successive calls to u() then generate pseudo-random
numbers distributed uniformly in the interval [0., 1.).

    >>> u = galsim.UniformDeviate()
    >>> u()
    0.35068059829063714
    >>> u()            
    0.56841182382777333

Methods
-------
To add deviates to every element of an image, use the syntax image.addNoise(u).

This docstring can be found using the Python interpreter or in pysrc/Random.cpp.
"""

_galsim.UniformDeviate.__call__.__func__.__doc__= "Draw a new random number from the distribution."


# GaussianDeviate docstrings
_galsim.GaussianDeviate.__doc__ = """
Pseudo-random number generator with Gaussian distribution.

See http://en.wikipedia.org/wiki/Gaussian_distribution for further details.

Initialization
--------------

    >>> g = galsim.GaussianDeviate(mean=0., sigma=1.)          # Initializes g to be a
                                                               # GaussianDeviate instance using the
                                                               # current time for the seed.

    >>> g = galsim.GaussianDeviate(lseed, mean=0., sigma=1.)   # Initializes g using the specified
                                                               # seed, where lseed is a long int.

    >>> g = galsim.GaussianDeviate(dev, mean=0., sigma=1.)     # Initializes g to share the same
                                                               # underlying random number generator
                                                               # as dev.

Parameters:

    mean     optional mean for Gaussian distribution [default `mean = 0.`].
    sigma    optional sigma for Gaussian distribution [default `sigma = 1.`].  Must be > 0.

Calling
-------
Taking the instance from the above examples, successive calls to g() then generate pseudo-random
numbers which Gaussian-distributed with the provided mean, sigma.

    >>> g = galsim.GaussianDeviate()
    >>> g()
    1.398768034960607
    >>> g()
    -0.8456136323830128

Methods
-------
To add deviates to every element of an image, use the syntax image.addNoise(g).

To get and set the deviate parameters, see the docstrings for the .getMean(), .setMean(), 
.getSigma() and .setSigma() methods of each instance.
"""

_galsim.GaussianDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Gaussian deviate with current mean and sigma.
"""
_galsim.GaussianDeviate.getMean.__func__.__doc__ = "Get current distribution mean."
_galsim.GaussianDeviate.setMean.__func__.__doc__ = "Set current distribution mean."
_galsim.GaussianDeviate.getSigma.__func__.__doc__ = "Get current distribution sigma."
_galsim.GaussianDeviate.setSigma.__func__.__doc__ = "Set current distribution sigma."


# BinomialDeviate docstrings
_galsim.BinomialDeviate.__doc__ = """
Pseudo-random Binomial deviate for N trials each of probability p.

N is number of 'coin flips,' p is probability of 'heads,' and each call returns an integer value
where 0 <= value <= N giving number of heads.  See http://en.wikipedia.org/wiki/Binomial_distribution
for more information.

Initialization
--------------

    >>> b = galsim.BinomialDeviate(N=1., p=0.5)          # Initializes b to be a BinomialDeviate
                                                         # instance using the current time for the
                                                         # seed.

    >>> b = galsim.BinomialDeviate(lseed, N=1., p=0.5)   # Initializes b using the specified seed,
                                                         # where lseed is a long int.

    >>> b = galsim.BinomialDeviate(dev, N=1., p=0.5)     # Initializes b to share the same
                                                         # underlying random number generator as dev.

Parameters:

    N   optional number of 'coin flips' per trial [default `N = 1`].  Must be > 0.
    p   optional probability of success per coin flip [default `p = 0.5`].  Must be > 0.

Calling
-------
Taking the instance from the above examples, successive calls to b() then generate pseudo-random
numbers binomial-distributed with the provided N, p.

    >>> b = galsim.BinomialDeviate()
    >>> b()
    0
    >>> b()
    1

Methods
-------
To add deviates to every element of an image, use the syntax image.addNoise(b).

To get and set the deviate parameters, see the docstrings for the .getN(), .setN(), .getP() and
.setP() methods of each instance.
"""

_galsim.BinomialDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Binomial deviate with current N and p.
"""
_galsim.BinomialDeviate.getN.__func__.__doc__ = "Get current distribution N."
_galsim.BinomialDeviate.setN.__func__.__doc__ = "Set current distribution N."
_galsim.BinomialDeviate.getP.__func__.__doc__ = "Get current distribution p."
_galsim.BinomialDeviate.setP.__func__.__doc__ = "Set current distribution p."


# PoissonDeviate docstrings
_galsim.PoissonDeviate.__doc__ = """
Pseudo-random Poisson deviate with specified mean.

The input mean sets the mean and variance of the Poisson deviate.  An integer deviate with this
distribution is returned after each call.  See http://en.wikipedia.org/wiki/Poisson_distribution
for more details.

Initialization
--------------

    >>> p = galsim.PoissonDeviate(mean=1.)         # Initializes g to be a PoissonDeviate instance
                                                   # using the current time for the seed.

    >>> p = galsim.PoissonDeviate(lseed, mean=1.)  # Initializes g using the specified seed, where
                                                   # lseed is a long int.

    >>> p = galsim.PoissonDeviate(dev, mean=1.)    # Initializes g to share the same underlying
                                                   # random number generator as dev.

Parameters:

    mean     optional mean of the distribution [default `mean = 1`].  Must be > 0.

Calling
-------
Taking the instance from the above examples, successive calls to p() will return successive,
pseudo-random Poisson deviates with specified mean.

    >>> p = galsim.PoissonDeviate()
    >>> p()
    1
    >>> p()
    2

Methods
-------
To add deviates to every element of an image, use the syntax image.addNoise(p).

To get and set the deviate parameter, see the docstrings for the .getMean(), .setMean() method of 
each instance.
"""

_galsim.PoissonDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Poisson deviate with current mean.
"""
_galsim.PoissonDeviate.getMean.__func__.__doc__ = "Get current distribution mean."
_galsim.PoissonDeviate.setMean.__func__.__doc__ = "Set current distribution mean."



# WeibullDeviate docstrings
_galsim.WeibullDeviate.__doc__ = """
Pseudo-random Weibull-distributed deviate for shape parameter a & scale parameter b.

The Weibull distribution is related to a number of other probability distributions;  in particular,
it interpolates between the exponential distribution (a=1) and the Rayleigh distribution (a=2). 
See http://en.wikipedia.org/wiki/Weibull_distribution (a=k and b=lambda in the notation adopted in 
the Wikipedia article) for more details.  The Weibull distribution is real valued and produces 
deviates >= 0.

Initialization
--------------

    >>> w = galsim.WeibullDeviate(a=1., b=1.)         # Initializes w to be a WeibullDeviate
                                                      # instance using the current time for the seed.

    >>> w = galsim.WeibullDeviate(lseed, a=1., b=1.)  # Initializes w using the specified seed,
                                                      # where lseed is a long int.

    >>> w = galsim.WeibullDeviate(dev, a=1., b=1.)    # Initializes w to share the same underlying
                                                      # random number generator as dev.

Parameters:

    a        shape parameter of the distribution [default `a = 1`].  Must be > 0.
    b        scale parameter of the distribution [default `b = 1`].  Must be > 0.

Calling
-------
Taking the instance from the above examples, successive calls to w() then generate pseudo-random 
numbers Weibull-distributed with shape and scale parameters a and b.

    >>> w = galsim.WeibullDeviate()
    >>> w()
    2.152873075208731
    >>> w()
    2.0826856212853846

Methods
-------
To add deviates to every element of an image, use the syntax image.addNoise(w).

To get and set the deviate parameters, see the docstrings for the .getA(), .setA(), .getB() and 
.setB() methods of each instance.
"""

_galsim.WeibullDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Weibull-distributed deviate with current a and b.
"""
_galsim.WeibullDeviate.getA.__func__.__doc__ = "Get current distribution shape parameter a."
_galsim.WeibullDeviate.setA.__func__.__doc__ = "Set current distribution shape parameter a."
_galsim.WeibullDeviate.getB.__func__.__doc__ = "Get current distribution shape parameter b."
_galsim.WeibullDeviate.setB.__func__.__doc__ = "Set current distribution shape parameter b."


# GammaDeviate docstrings
_galsim.GammaDeviate.__doc__ = """
Pseudo-random Gamma-distributed deviate for parameters alpha & beta.

See http://en.wikipedia.org/wiki/Gamma_distribution (note that alpha=k and beta=theta in the
notation adopted in the Boost.Random routine called by this class).  The Gamma distribution is a 
real-valued distribution producing deviates >= 0.

Initialization
--------------

    >>> gam = galsim.GammaDeviate(alpha=1., beta=1.)         # Initializes gam to be a GammaDeviate
                                                             # instance using the current time for
                                                             # the seed.

    >>> gam = galsim.GammaDeviate(lseed, alpha=1., beta=1.)  # Initializes gam using the specified
                                                             # seed, where lseed is a long int.

    >>> gam = galsim.GammaDeviate(dev alpha=1., beta=1.)     # Initializes gam to share the same
                                                             # underlying random number generator as
                                                             # dev.

Parameters:

    alpha    shape parameter of the distribution [default `alpha = 1`].  Must be > 0.
    beta     scale parameter of the distribution [default `beta = 1`].  Must be > 0.

Calling
-------
Taking the instance from the above examples, successive calls to g() will return successive, 
pseudo-random Gamma-distributed deviates with shape and scale parameters alpha and beta. 

    >>> gam = galsim.GammaDeviate()
    >>> gam()
    0.020092014608829315
    >>> gam()
    0.5062533114685395

Methods
-------
To add deviates to every element of an image, use the syntax image.addNoise(gam).

To get and set the deviate parameters, see the docstrings for the .getAlpha(), .setAlpha(), 
.getBeta() and .setBeta() methods of each instance.

To add deviates to every element of an image, the syntax image.addNoise() is preferred.
"""

_galsim.GammaDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Gamma-distributed deviate with current alpha and beta.
"""
_galsim.GammaDeviate.getAlpha.__func__.__doc__ = "Get current distribution shape parameter alpha."
_galsim.GammaDeviate.setAlpha.__func__.__doc__ = "Set current distribution shape parameter alpha."
_galsim.GammaDeviate.getBeta.__func__.__doc__ = "Get current distribution shape parameter beta."
_galsim.GammaDeviate.setBeta.__func__.__doc__ = "Set current distribution shape parameter beta."


# Chi2Deviate docstrings
_galsim.Chi2Deviate.__doc__ = """
Pseudo-random Chi^2-distributed deviate for degrees-of-freedom parameter n.

See http://en.wikipedia.org/wiki/Chi-squared_distribution (note that k=n in the notation adopted in
the Boost.Random routine called by this class).  The Chi^2 distribution is a real-valued 
distribution producing deviates >= 0.

Initialization
--------------

    >>> chis = galsim.Chi2Deviate(n=1.)          # Initializes chis to be a Chi2Deviate instance
                                                 # using the current time for the seed.

    >>> chis = galsim.Chi2Deviate(lseed, n=1.)   # Initializes chis using the specified seed, where
                                                 # lseed is a long int.

    >>> chis = galsim.Chi2Deviate(dev, n=1.)     # Initializes chis to share the same underlying
                                                 # random number generator as dev.

Parameters:
    n   number of degrees of freedom for the output distribution [default `n = 1`].  Must be > 0.

Calling
-------
Taking the instance from the above examples, successive calls to g() will return successive, 
pseudo-random Chi^2-distributed deviates with degrees-of-freedom parameter n.

    >>> chis = galsim.Chi2Deviate()
    >>> chis()
    0.35617890086874854
    >>> chis()
    0.17269982670901735

Methods
-------
To add deviates to every element of an image, use the syntax image.addNoise(chis).

To get and set the deviate parameter, see the docstrings for the .getN(), .setN() methods of each
instance.
"""

_galsim.Chi2Deviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Chi2-distributed deviate with current n degrees of freedom.
"""
_galsim.Chi2Deviate.getN.__func__.__doc__ = "Get current distribution n degrees of freedom."
_galsim.Chi2Deviate.setN.__func__.__doc__ = "Set current distribution n degrees of freedom."


#
# The rest of these are from Noise.h in the C++ layer, not Randome.h.
#


# GaussianNoise docstrings
_galsim.GaussianNoise.__doc__ = """
Class implementing simple Gaussian noise.

This is a simple noise model where each pixel in the image gets Gaussian noise with a 
given sigma.

Initialization
--------------

    >>> gaussian_noise = galsim.GaussianNoise(rng, sigma=1.)

Parameters:

    rng       A BaseDeviate instance to use for generating the random numbers.
    sigma     The rms noise on each pixel [default `sigma = 1.`].

Methods
-------
To add noise to every element of an image, use the syntax image.addNoise(gaussian_noise).
"""

_galsim.GaussianNoise.applyTo.__func__.__doc__ = """
Add Gaussian noise to an input Image.

Calling
-------

    >>> gaussian_noise.applyTo(image)

On output the Image instance image will have been given additional Gaussian noise according 
to the given GaussianNoise instance.

Note: The syntax image.addNoise(gaussian_noise) is preferred.
"""
_galsim.GaussianNoise.getSigma.__func__.__doc__ = "Get sigma in current noise model."
_galsim.GaussianNoise.setSigma.__func__.__doc__ = "Set sigma in current noise model."


# PoissonNoise docstrings
_galsim.PoissonNoise.__doc__ = """
Class implementing Poisson noise according to the flux (and sky level) present in the image.

The PoissonNoise class encapsulates a simple version of the noise model of a normal CCD image
where each pixel has Poisson noise corresponding to the number of electrons in each pixel
(including an optional extra sky level).

Note that if the image to which you are adding noise already has a sky level on it,
then you should not provide the sky level here as well.  The sky level here corresponds
to a level is taken to be already subtracted from the image, but which was present
for the Poisson noise.

Initialization
--------------

    >>> poisson_noise = galsim.PoissonNoise(rng, sky_level=0.)

Parameters:

    rng         A BaseDeviate instance to use for generating the random numbers.
    sky_level   The sky level in electrons per pixel that was originally in the input image, 
                but which is taken to have already been subtracted off [default `sky_level = 0.`].

Methods
-------
To add noise to every element of an image, use the syntax image.addNoise(poisson_noise).
"""

_galsim.PoissonNoise.applyTo.__func__.__doc__ = """
Add Poisson noise to an input Image.

Calling
-------

    >>> galsim.PoissonNoise.applyTo(image)

On output the Image instance image will have been given additional Poisson noise according 
to the given PoissonNoise instance.

Note: the syntax image.addNoise(poisson_noise) is preferred.
"""
_galsim.PoissonNoise.getSkyLevel.__func__.__doc__ = "Get sky level in current noise model."
_galsim.PoissonNoise.setSkyLevel.__func__.__doc__ = "Set sky level in current noise model."


# CCDNoise docstrings
_galsim.CCDNoise.__doc__ = """
Class implementing a basic CCD noise model.

The CCDNoise class encapsulates the noise model of a normal CCD image.  The noise has two
components: first, Poisson noise corresponding to the number of electrons in each pixel
(including an optional extra sky level); second, Gaussian read noise.

Note that if the image to which you are adding noise already has a sky level on it,
then you should not provide the sky level here as well.  The sky level here corresponds
to a level is taken to be already subtracted from the image, but which was present
for the Poisson noise.

Initialization
--------------

    >>> ccd_noise = galsim.CCDNoise(rng, sky_level=0., gain=1., read_noise=0.)  

Parameters:

    rng         A BaseDeviate instance to use for generating the random numbers.
    sky_level   The sky level in electrons per pixel that was originally in the input image, 
                but which is taken to have already been subtracted off [default `sky_level = 0.`].
    gain        The gain for each pixel in electrons per ADU; setting gain<=0 will shut off the
                Poisson noise, and the Gaussian rms will take the value read_noise as being in 
                units of ADU rather than electrons [default `gain = 1.`].
    read_noise  The read noise on each pixel in electrons (gain > 0.) or ADU (gain <= 0.)
                setting read_noise=0. will shut off the Gaussian noise [default `read_noise = 0.`].

Methods
-------
To add noise to every element of an image, use the syntax image.addNoise(ccd_noise).
"""

_galsim.CCDNoise.applyTo.__func__.__doc__ = """
Add noise to an input Image.

Calling
-------

    >>> ccd_noise.applyTo(image)

On output the Image instance image will have been given additional stochastic noise according to 
the gain and read noise settings of the given CCDNoise instance.

Note: the syntax image.addNoise(ccd_noise) is preferred.
"""
_galsim.CCDNoise.getSkyLevel.__func__.__doc__ = "Get sky level in current noise model."
_galsim.CCDNoise.getGain.__func__.__doc__ = "Get gain in current noise model."
_galsim.CCDNoise.getReadNoise.__func__.__doc__ = "Get read noise in current noise model."
_galsim.CCDNoise.setSkyLevel.__func__.__doc__ = "Set sky level in current noise model."
_galsim.CCDNoise.setGain.__func__.__doc__ = "Set gain in current noise model."
_galsim.CCDNoise.setReadNoise.__func__.__doc__ = "Set read noise in current noise model."


# DeviateNoise docstrings
_galsim.DeviateNoise.__doc__ = """
Class implementing noise with an arbitrary BaseDeviate object.

The DeviateNoise class provides a way to treat an arbitrary deviate as the noise model for 
each pixel in an image.

Initialization
--------------

    >>> dev_noise = galsim.DeviateNoise(dev)

Parameters:

    dev         A BaseDeviate subclass to use as the noise deviate for each pixel.

Methods
-------
To add noise to every element of an image, use the syntax image.addNoise(dev_noise).
"""

_galsim.DeviateNoise.applyTo.__func__.__doc__ = """
Add noise to an input Image.

Calling
-------

    >>> dev_noise.applyTo(image)

On output the Image instance image will have been given additional noise according to 
the given DeviateNoise instance.

To add deviates to every element of an image, the syntax image.addNoise() is preferred.
"""

