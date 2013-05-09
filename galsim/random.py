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
Addition of docstrings to the Random deviate classes at the Python layer and definition of the 
DistDeviate class.
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


class DistDeviate(_galsim.BaseDeviate):
    """A class to draw random numbers from a user-defined probability distribution.
    
    DistDeviate is a BaseDeviate class that can be used to draw from an arbitrary probability
    distribution.  The probability distribution passed to DistDeviate can be given one of three 
    ways: as the name of a file containing a 2d ASCII array of x and P(x) or as a callable 
    function.
    
    Once given a probability, DistDeviate creates a table of x value versus cumulative probability 
    and draws from it using a UniformDeviate.  The precision of its outputs can be controlled with
    the keyword npoints, which sets the number of points DistDeviate creates for its internal table
    of x vs CDF(x).  To prevent errors due to non-monotonicity, the interpolant for this internal
    table is always linear.
    
    Two keywords, x_min and x_max, define the support of the function.  They must be passed if a 
    callable function is given to DistDeviate, unless the function is a galsim.LookupTable, which 
    has its own defined endpoints.  If a filename or LookupTable is passed to DistDeviate, the use
    of x_min or x_max will result in an error.
        
    If given a table in a file, DistDeviate will construct an interpolated LookupTable to obtain 
    more finely gridded probabilities for generating the cumulative probability table.  The default
    interpolant is linear, but any interpolant understood by LookupTable may be used.  We caution 
    against the use of splines because they can cause non-monotonic behavior.  Passing the 
    interpolant keyword next to anything but a table in a file will result in an error.
    
    Initialization
    --------------
    
    Some sample initialization calls:
    
    >>> d = galsim.DistDeviate(function=f, x_min=x_min, x_max=x_max)
    
    Initializes d to be a DistDeviate instance with a distribution given by the callable function
    f(x) from x=x_min to x=x_max and seeds the PRNG using current time.  
    
    >>> d = galsim.DistDeviate(rng=1062533, function=file_name, interpolant='floor')
    
    Initializes d to be a DistDeviate instance with a distribution given by the data in file
    file_name, which must be a 2-column ASCII table, and seeds the PRNG using the long int
    seed 1062533. It generates probabilities from file_name using the interpolant 'floor'.
    
    >>> d = galsim.DistDeviate(rng, function=galsim.LookupTable(x,p))
    
    Initializes d to be a DistDeviate instance with a distribution given by P(x), defined as two
    arrays x and p which are used to make a callable galsim.LookupTable, and links the DistDeviate
    PRNG to the already-existing random number generator rng.
    
    @param rng          Something that can seed a BaseDeviate: a long int seed or another 
                        BaseDeviate.  Using 0 means to use the time of day as a seed.
                        (default: 0)
    @param function     A callable function giving a probability distribution or the name of a 
                        file containing a probability distribution as a 2-column ASCII table.
    @param x_min        The minimum desired return value (required for non-galsim.LookupTable
                        callable functions; will raise an error if not passed in that case, or if
                        passed in any other case)
    @param x_min        The maximum desired return value (required for non-galsim.LookupTable
                        callable functions; will raise an error if not passed in that case, or if
                        passed in any other case)
    @param interpolant  Type of interpolation used for interpolating a file (causes an error if 
                        passed alongside a callable function).  Options are given in the 
                        documentation for galsim.LookupTable. (default: 'linear')
    @param npoints      Number of points DistDeviate should create for its internal interpolation
                        tables. (default: 256)

    Calling
    -------
    Taking the instance from the above examples, successive calls to d() then generate pseudo-random
    numbers distributed according to the initialized distribution.

    >>> d()
    1.396015204978437
    >>> d()
    1.6481898771717463
    >>> d()
    2.108800962574702
    """    
    def __init__(self, rng=0, function=None, x_min=None, 
                 x_max=None, interpolant=None, npoints=256):
        """Initializes a DistDeviate instance.
        
        The rng, if given, must be something that can initialize a BaseDeviate instance, such as 
        another BaseDeviate or a long int seed.  See the documentation for the DistDeviate class 
        for more information on this and other options.
        """
        
        import numpy
        import galsim
 
        # Set up the PRNG
        _galsim.BaseDeviate.__init__(self,rng)
        self._ud = galsim.UniformDeviate(self)

        # Basic input checking and setups
        if not function:
            raise TypeError('You must pass a function to DistDeviate!')
        # Figure out if a string is a filename or something we should be using in an eval call
        if isinstance(function, str):
            input_function = function
            function = eval('lambda x: ' + function)
            try:
                if x_min is not None: # is not None in case x_min=0.
                    function(x_min)
                else: 
                    # Somebody would be silly to pass a string for evaluation without x_min,
                    # but we'd like to throw reasonable errors in that case anyway
                    function(0.6) # A value unlikely to be a singular point of a function
            except: # Okay, maybe it's a file name after all
                function = input_function
                import os.path
                if not os.path.isfile(function):
                    raise ValueError('String passed with function keyword to DistDeviate does '
                                     'not point to an existing file and cannot be evaluated via '
                                     'an eval call with lambda x: %s'%function)
        # Check that the function is actually a function
        if not (isinstance(function, galsim.LookupTable) or isinstance(function, str) or
                hasattr(function,'__call__')):
            raise TypeError('Keyword function passed to DistDeviate must be a callable function or '
                            'a string: %s'%function)

        # Set up the probability function & min and max values for any inputs
        if hasattr(function,'__call__'):
            if interpolant:
                raise TypeError('Cannot pass an interpolant with a callable '
                                'function to DistDeviate')
            if isinstance(function,galsim.LookupTable):
                if x_min or x_max:
                    raise TypeError('Cannot pass x_min or x_max alongside a LookupTable '
                                    'in arguments to DistDeviate')
                x_min = function.x_min
                x_max = function.x_max
            else:
                if x_min is None or x_max is None:
                    raise TypeError('Must pass x_min and x_max alongside non-galsim.LookupTable '
                                    'callable functions in arguments to DistDeviate')
        else: # must be a filename
            if interpolant is None:
                interpolant='linear'
            if x_min or x_max:
                raise TypeError('Cannot pass x_min or x_max alongside a '
                                'filename in arguments to DistDeviate')
            function = galsim.LookupTable(file=function, interpolant=interpolant)
            x_min = function.x_min
            x_max = function.x_max

        # Compute the cumulative distribution function
        xarray = x_min+(1.*x_max-x_min)/(npoints-1)*numpy.array(range(npoints),float)
        # cdf is the cumulative distribution function--just easier to type!
        dcdf = [galsim.integ.int1d(function, xarray[i], xarray[i+1]) for i in range(npoints - 1)]
        cdf = [sum(dcdf[0:i]) for i in range(npoints)]
        # Quietly renormalize the probability if it wasn't already normalized
        totalprobability = cdf[-1]
        cdf = numpy.array(cdf)/totalprobability
        dcdf = numpy.array(dcdf)/totalprobability
        # Check that the probability is nonnegative
        if not numpy.all(dcdf >= 0):
            raise ValueError('Negative probability passed to DistDeviate: %s'%function)
        # Now get rid of points with dcdf == 0
        elif not numpy.all(dcdf > 0.):
            # Remove consecutive dx=0 points, except endpoints
            zeroindex = numpy.where(dcdf==0)[0]
            # numpy.where returns a tuple containing 1 array, which tends to be annoying for
            # indexing, so the [0] returns the actual array of interest (indices of dcdf==0).
            # Now, we want to remove consecutive dcdf=0 points, leaving the lower end.
            # Zeroindex contains the indices of all the dcdf=0 points, so we look for ones that are
            # only 1 apart; this tells us the *lower* of the two points, but we want to remove the
            # *upper*, so we add 1 to the resultant array.
            dindex = numpy.where(numpy.diff(zeroindex)==1)[0]+1 
            # So dindex contains the indices of the elements of array zeroindex, which tells us the 
            # indices that we might want to delete from cdf and xarray, so we delete 
            # zeroindex[dindex].
            cdf = numpy.delete(cdf,zeroindex[dindex])
            xarray = numpy.delete(xarray,zeroindex[dindex])
            dcdf = numpy.diff(cdf)
            # Tweak the edges of dx=0 regions so function is always increasing
            for index in numpy.where(dcdf == 0)[0]:
                if index+1 < len(cdf):
                    cdf[index+1] += 2.23E-16
                else:
                    cdf = cdf[:-1]
                    xarray = xarray[:-1]
            dcdf = numpy.diff(cdf)
            if not (numpy.all(dcdf>0)):
                raise RuntimeError(
                    'Cumulative probability in DistDeviate is too flat for program to fix')
                        
        self._inverseprobabilitytable = galsim.LookupTable(cdf, xarray, interpolant='linear')
        self.x_min = x_min
        self.x_max = x_max

        
    def val(self):
        return self._inverseprobabilitytable(self._ud())
    
    def __call__(self):
        return self.val()

    def reset(self, rng=0):
        _galsim.BaseDeviate.reset(self,rng)
        # Make sure the stored _ud object stays in sync with self.
        self._ud.reset(self)


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

    >>> galsim.BaseDeviate.seed()       # Re-seed the PRNG using current time

    >>> galsim.BaseDeviate.seed(lseed)  # Re-seed the PRNG using specified seed, where lseed
                                        # is a long int

"""

_galsim.BaseDeviate.reset.__func__.__doc__ = """
Reset the pseudo-random number generator, severing connections to any other deviates.

Multiple Calling Options
------------------------

    >>> galsim.BaseDeviate.reset()        # Re-seed the PRNG using current time, and sever the
                                          # connection to any other Deviate

    >>> galsim.BaseDeviate.reset(lseed)   # Re-seed the PRNG using specified seed, where lseed is a
                                          # long int, and sever the connection to any other Deviate

    >>> galsim.BaseDeviate.reset(dev)     # Re-connect this Deviate with the same underlying random
                                          # number generator supplied in dev

"""


# UniformDeviate docstrings
_galsim.UniformDeviate.__doc__ = """
Pseudo-random number generator with uniform distribution in interval [0.,1.).

Initialization
--------------

    >>> u = galsim.UniformDeviate()       # Initializes u to be a UniformDeviate instance, and seeds
                                          # the PRNG using current time

    >>> u = galsim.UniformDeviate(lseed)  # Initializes u to be a UniformDeviate instance, and seeds
                                          # the PRNG using specified long integer lseed

    >>> u = galsim.UniformDeviate(dev)    # Initializes u to be a UniformDeviate instance, and share
                                          # the same underlying random number generator as dev

Calling
-------
Taking the instance from the above examples, successive calls to u() then generate pseudo-random
numbers distributed uniformly in the interval [0., 1.).

    >>> u = galsim.UniformDeviate()
    >>> u()
    0.35068059829063714
    >>> u()            
    0.56841182382777333
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
                                                               # current time for the seed

    >>> g = galsim.GaussianDeviate(lseed, mean=0., sigma=1.)   # Initializes g using the specified
                                                               # seed, where lseed is a long int

    >>> g = galsim.GaussianDeviate(dev, mean=0., sigma=1.)     # Initializes g to share the same
                                                               # underlying random number generator
                                                               # as dev

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
                                                         # seed

    >>> b = galsim.BinomialDeviate(lseed, N=1., p=0.5)   # Initializes b using the specified seed,
                                                         # where lseed is a long int

    >>> b = galsim.BinomialDeviate(dev, N=1., p=0.5)     # Initializes b to share the same
                                                         # underlying random number generator as dev

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
                                                   # using the current time for the seed

    >>> p = galsim.PoissonDeviate(lseed, mean=1.)  # Initializes g using the specified seed, where
                                                   # lseed is a long int

    >>> p = galsim.PoissonDeviate(dev, mean=1.)    # Initializes g to share the same underlying
                                                   # random number generator as dev

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
                                                      # instance using the current time for the seed

    >>> w = galsim.WeibullDeviate(lseed, a=1., b=1.)  # Initializes w using the specified seed,
                                                      # where lseed is a long int

    >>> w = galsim.WeibullDeviate(dev, a=1., b=1.)    # Initializes w to share the same underlying
                                                      # random number generator as dev

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
A Gamma-distributed deviate with shape parameter k and scale parameter theta.
See http://en.wikipedia.org/wiki/Gamma_distribution.  
(Note: we use the k, theta notation. If you prefer alpha, beta, use k=alpha, theta=1/beta.)
The Gamma distribution is a real valued distribution producing deviates >= 0.

Initialization
--------------

    >>> gam = galsim.GammaDeviate(k=1., theta=1.)         # Initializes gam to be a GammaDeviate
                                                          # instance using the current time for
                                                          # the seed

    >>> gam = galsim.GammaDeviate(lseed, k=1., theta=1.)  # Initializes gam using the specified
                                                          # seed, where lseed is a long int

    >>> gam = galsim.GammaDeviate(dev, k=1., theta=1.)    # Initializes gam to share the same
                                                          # underlying random number generator as
                                                          # dev

Parameters:

    k       shape parameter of the distribution [default `k = 1`].  Must be > 0.
    theta   scale parameter of the distribution [default `theta = 1`].  Must be > 0.

Calling
-------
Taking the instance from the above examples, successive calls to g() will return successive, 
pseudo-random Gamma-distributed deviates with shape and scale parameters k and theta. 

    >>> gam = galsim.GammaDeviate()
    >>> gam()
    0.020092014608829315
    >>> gam()
    0.5062533114685395
"""

_galsim.GammaDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Gamma-distributed deviate with current k and theta.
"""
_galsim.GammaDeviate.getK.__func__.__doc__ = "Get current distribution shape parameter k."
_galsim.GammaDeviate.setK.__func__.__doc__ = "Set current distribution shape parameter k."
_galsim.GammaDeviate.getTheta.__func__.__doc__ = "Get current distribution shape parameter theta."
_galsim.GammaDeviate.setTheta.__func__.__doc__ = "Set current distribution shape parameter theta."


# Chi2Deviate docstrings
_galsim.Chi2Deviate.__doc__ = """
Pseudo-random Chi^2-distributed deviate for degrees-of-freedom parameter n.

See http://en.wikipedia.org/wiki/Chi-squared_distribution (note that k=n in the notation adopted in
the Boost.Random routine called by this class).  The Chi^2 distribution is a real-valued 
distribution producing deviates >= 0.

Initialization
--------------

    >>> chis = galsim.Chi2Deviate(n=1.)          # Initializes chis to be a Chi2Deviate instance
                                                 # using the current time for the seed

    >>> chis = galsim.Chi2Deviate(lseed, n=1.)   # Initializes chis using the specified seed, where
                                                 # lseed is a long int

    >>> chis = galsim.Chi2Deviate(dev, n=1.)     # Initializes chis to share the same underlying
                                                 # random number generator as dev

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
"""

_galsim.Chi2Deviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Chi2-distributed deviate with current n degrees of freedom.
"""
_galsim.Chi2Deviate.getN.__func__.__doc__ = "Get current distribution n degrees of freedom."
_galsim.Chi2Deviate.setN.__func__.__doc__ = "Set current distribution n degrees of freedom."
