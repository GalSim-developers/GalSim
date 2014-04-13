# Copyright 2012-2014 The GalSim developers:
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
from ._galsim import BaseDeviate, UniformDeviate, GaussianDeviate, PoissonDeviate
from ._galsim import BinomialDeviate, Chi2Deviate, GammaDeviate, WeibullDeviate

# BaseDeviate docstrings
_galsim.BaseDeviate.__doc__ = """
Base class for all the various random deviates.

This holds the essential random number generator that all the other classes use.

Initialization
--------------

All deviates take an initial `seed` argument that is used to seed the underlying random number
generator.  It has three different kinds of behavior.

  1. An integer value can be provided to explicitly seed the random number generator with a 
     particular value.  This is useful to have deterministic behavior.  If you seed with an
     integer value, the subsequent series of "random" values will be the same each time you
     run the program.

  2. A special value of 0 means to pick some arbitrary value that will be different each time
     you run the program.  Currently, this is taken from the current time, but this behavior
     may change in the future.  You can also get this behavior by omitting the seed argument
     entirely.  (i.e. the default is 0.)

  3. Providing another BaseDeviate object as the seed will make the new Deviate share the same
     underlying random number generator as the other Deviate.  So you can make one Deviate (of
     any type), and seed it with a particular deterministic value.  Then if you pass that Deviate
     to any other one you make, they will both be using the same RNG and the series of "random"
     values will be deterministic.
     
Usage
-----

There is not much you can do with something that is only known to be a BaseDeviate rather than one
of the derived classes other than construct it and change the seed, and use it as an argument to
pass to other Deviate constructors.

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

Methods
-------

There are a few methods that are common to all BaseDeviate classes, so we describe them here.

    dev.seed(seed)      Set a new (integer) seed value for the underlying RNG.
    dev.reset(seed)     Sever the connection to the current RNG and seed a new one (either
                        creating a new RNG if seed is an integer or connecting to an existing
                        RNG if seed is a BaseDeviate instance)
    dev.clearCache()    Clear the internal cache of the Deviate, if there is any.
    dev.duplicate()     Create a duplicate of the current Deviate, which will produce an identical
                        series of values as the original.
"""

_galsim.BaseDeviate.seed.__func__.__doc__ = """
Seed the pseudo-random number generator with a given integer value.

@param seed         An int value to be used to seed the random number generator.  Using 0
                    means to use the time of day as a seed. [default: 0]
"""

_galsim.BaseDeviate.reset.__func__.__doc__ = """
Reset the pseudo-random number generator, severing connections to any other deviates.
Providing another BaseDeviate object as the seed connects this deviate with the other
one, so they will both use the same underlying random number generator.

@param seed         Something that can seed a BaseDeviate: a long int seed or another 
                    BaseDeviate.  Using 0 means to use the time of day as a seed. [default: 0]
"""

_galsim.BaseDeviate.duplicate.__func__.__doc__ = """
Create a duplicate of the current Deviate object.  The subsequent series from each copy
of the Deviate will produce identical values.

Example
_______

    >>> u = galsim.UniformDeviate(31415926)
    >>> u()
    0.17100770119577646
    >>> u2 = u.duplicate()
    >>> u()
    0.49095047544687986
    >>> u()
    0.10306670609861612
    >>> u2()
    0.49095047544687986
    >>> u2()
    0.10306670609861612
    >>> u2()
    0.13129289541393518
    >>> u()
    0.13129289541393518
"""

_galsim.BaseDeviate.clearCache.__func__.__doc__ = """
Clear the internal cache of the Deviate, if any.  This is currently only relevant for
GaussianDeviate, since it generates two values at a time, saving the second one to use for the
next output value.
"""


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
    ways: as the name of a file containing a 2d ASCII array of `x` and `P(x)` or as a callable 
    function.
    
    Once given a probability, DistDeviate creates a table of `x` value versus cumulative probability
    and draws from it using a UniformDeviate.  The precision of its outputs can be controlled with
    the keyword npoints, which sets the number of points DistDeviate creates for its internal table
    of `x` vs `CDF(x)`.  To prevent errors due to non-monotonicity, the interpolant for this
    internal table is always linear.
    
    Two keywords, `x_min` and `x_max`, define the support of the function.  They must be passed if a
    callable function is given to DistDeviate, unless the function is a LookupTable, which has its
    own defined endpoints.  If a filename or LookupTable is passed to DistDeviate, the use of
    `x_min` or `x_max` will result in an error.
        
    If given a table in a file, DistDeviate will construct an interpolated LookupTable to obtain
    more finely gridded probabilities for generating the cumulative probability table.  The default
    `interpolant` is linear, but any interpolant understood by LookupTable may be used.  We caution
    against the use of splines because they can cause non-monotonic behavior.  Passing the
    `interpolant` keyword next to anything but a table in a file will result in an error.
    
    Initialization
    --------------
    
    Some sample initialization calls:
    
    >>> d = galsim.DistDeviate(function=f, x_min=x_min, x_max=x_max)
    
    Initializes d to be a DistDeviate instance with a distribution given by the callable function
    `f(x)` from `x=x_min` to `x=x_max` and seeds the PRNG using current time.  
    
    >>> d = galsim.DistDeviate(1062533, function=file_name, interpolant='floor')
    
    Initializes d to be a DistDeviate instance with a distribution given by the data in file
    `file_name`, which must be a 2-column ASCII table, and seeds the PRNG using the long int
    seed 1062533. It generates probabilities from `file_name` using the interpolant 'floor'.
    
    >>> d = galsim.DistDeviate(rng, function=galsim.LookupTable(x,p))
    
    Initializes d to be a DistDeviate instance with a distribution given by `P(x)`, defined as two
    arrays `x` and `p` which are used to make a callable LookupTable, and links the DistDeviate
    PRNG to the already-existing random number generator `rng`.
    
    @param seed         Something that can seed a BaseDeviate: a long int seed or another 
                        BaseDeviate.  Using 0 means to use the time of day as a seed. [default: 0]
    @param function     A callable function giving a probability distribution or the name of a 
                        file containing a probability distribution as a 2-column ASCII table.
                        [default: None]
    @param x_min        The minimum desired return value (required for non-LookupTable
                        callable functions; will raise an error if not passed in that case, or if
                        passed in any other case) [default: None]
    @param x_min        The maximum desired return value (required for non-LookupTable
                        callable functions; will raise an error if not passed in that case, or if
                        passed in any other case) [default: None]
    @param interpolant  Type of interpolation used for interpolating a file (causes an error if 
                        passed alongside a callable function).  Options are given in the 
                        documentation for LookupTable. [default: 'linear']
    @param npoints      Number of points DistDeviate should create for its internal interpolation
                        tables. [default: 256]

    Calling
    -------

    Successive calls to d() generate pseudo-random values with the given probability distribution.

    >>> d = galsim.DistDeviate(31415926, function=lambda x: 1-abs(x), x_min=-1, x_max=1)
    >>> d()
    -0.4151921102709466
    >>> d()
    -0.00909781188974034
    """    
    def __init__(self, seed=0, function=None, x_min=None, 
                 x_max=None, interpolant=None, npoints=256, _init=True, lseed=None):
        # lseed is an obsolete synonym for seed
        # I think this was the only place that the name lseed was actually used in the docs.
        # so we keep it for now for backwards compatibility.
        if lseed is not None: seed = lseed

        import numpy
        import galsim

        # Special internal "private" constructor option that doesn't do any initialization.
        if not _init: return
 
        # Set up the PRNG
        _galsim.BaseDeviate.__init__(self,seed)
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

        
    def val(self,p):
        """
        Return the value `x` of the input function to DistDeviate such that 
        `p` = cumulative probability(x).
        """
        if p<0 or p>1:
            raise ValueError('Cannot request cumulative probability value from DistDeviate for '
                             'p<0 or p>1!  You entered: %f'%p)
        return self._inverseprobabilitytable(p)
    
    # This is the private function that is required to make DistDeviate work as a derived 
    # class of BaseDeviate.  See pysrc/Random.cpp.
    def _val(self):
        return self.val(self._ud())

    def __call__(self):
        return self._val()

    def seed(self, seed=0):
        _galsim.BaseDeviate.seed(self,seed)
        # Make sure the stored _ud object stays in sync with self.
        self._ud.reset(self)

    def reset(self, seed=0):
        _galsim.BaseDeviate.reset(self,seed)
        # Make sure the stored _ud object stays in sync with self.
        self._ud.reset(self)

    def duplicate(self):
        dup = DistDeviate(_init=False)
        dup.__dict__.update(self.__dict__)
        dup._ud = self._ud.duplicate()
        return dup



# UniformDeviate docstrings
_galsim.UniformDeviate.__doc__ = """
Pseudo-random number generator with uniform distribution in interval [0.,1.).

Initialization
--------------

@param seed         Something that can seed a BaseDeviate: a long int seed or another 
                    BaseDeviate.  Using 0 means to use the time of day as a seed. [default: 0]

Calling
-------

Successive calls to u() generate pseudo-random values distributed uniformly in the interval 
[0., 1.).

    >>> u = galsim.UniformDeviate(31415926)
    >>> u()
    0.17100770119577646
    >>> u()
    0.49095047544687986
"""

_galsim.UniformDeviate.__call__.__func__.__doc__= "Draw a new random number from the distribution."


# GaussianDeviate docstrings
_galsim.GaussianDeviate.__doc__ = """
Pseudo-random number generator with Gaussian distribution.

See http://en.wikipedia.org/wiki/Gaussian_distribution for further details.

Initialization
--------------

@param seed         Something that can seed a BaseDeviate: a long int seed or another 
                    BaseDeviate.  Using 0 means to use the time of day as a seed. [default: 0]
@param mean         Mean of Gaussian distribution. [default: 0.]
@param sigma        Sigma of Gaussian distribution. [default: 1.; Must be > 0]

Calling
-------

Successive calls to g() generate pseudo-random values distributed according to a Gaussian
distribution with the provided `mean`, `sigma`.

    >>> g = galsim.GaussianDeviate(31415926)
    >>> g()
    0.5533754000847082
    >>> g()
    1.0218588970190354
"""

_galsim.GaussianDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Gaussian deviate with current `mean` and `sigma`.
"""
_galsim.GaussianDeviate.getMean.__func__.__doc__ = "Get current distribution `mean`."
_galsim.GaussianDeviate.setMean.__func__.__doc__ = "Set current distribution `mean`. Discouraged."
_galsim.GaussianDeviate.getSigma.__func__.__doc__ = "Get current distribution `sigma`."
_galsim.GaussianDeviate.setSigma.__func__.__doc__ = "Set current distribution `sigma`. Discouraged."


# BinomialDeviate docstrings
_galsim.BinomialDeviate.__doc__ = """
Pseudo-random Binomial deviate for `N` trials each of probability `p`.

`N` is number of 'coin flips,' `p` is probability of 'heads,' and each call returns an integer value
where 0 <= value <= N gives the number of heads.  See
http://en.wikipedia.org/wiki/Binomial_distribution for more information.

Initialization
--------------

@param seed         Something that can seed a BaseDeviate: a long int seed or another 
                    BaseDeviate.  Using 0 means to use the time of day as a seed. [default: 0]
@param N            The number of 'coin flips' per trial. [default: 1; Must be > 0]
@param p            The probability of success per coin flip. [default: 0.5; Must be > 0]

Calling
-------

Successive calls to b() generate pseudo-random integer values distributed according to a binomial
distribution with the provided `N`, `p`.

    >>> b = galsim.BinomialDeviate(31415926, N=10, p=0.3)
    >>> b()
    2
    >>> b()
    3
"""

_galsim.BinomialDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Binomial deviate with current `N` and `p`.
"""
_galsim.BinomialDeviate.getN.__func__.__doc__ = "Get current distribution `N`."
_galsim.BinomialDeviate.setN.__func__.__doc__ = "Set current distribution `N`. Discouraged."
_galsim.BinomialDeviate.getP.__func__.__doc__ = "Get current distribution `p`."
_galsim.BinomialDeviate.setP.__func__.__doc__ = "Set current distribution `p`. Discouraged."


# PoissonDeviate docstrings
_galsim.PoissonDeviate.__doc__ = """
Pseudo-random Poisson deviate with specified `mean`.

The input `mean` sets the mean and variance of the Poisson deviate.  An integer deviate with this
distribution is returned after each call.  See http://en.wikipedia.org/wiki/Poisson_distribution for
more details.

Initialization
--------------

@param seed         Something that can seed a BaseDeviate: a long int seed or another 
                    BaseDeviate.  Using 0 means to use the time of day as a seed. [default: 0]
@param mean         Mean of the distribution. [default: 1; Must be > 0]

Calling
-------

Successive calls to p() generate pseudo-random integer values distributed according to a Poisson
distribution with the specified `mean`.

    >>> p = galsim.PoissonDeviate(31415926, mean=100)
    >>> p()
    94
    >>> p()
    106
"""

_galsim.PoissonDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Poisson deviate with current `mean`.
"""
_galsim.PoissonDeviate.getMean.__func__.__doc__ = "Get current distribution `mean`."
_galsim.PoissonDeviate.setMean.__func__.__doc__ = "Set current distribution `mean`. Discouraged."



# WeibullDeviate docstrings
_galsim.WeibullDeviate.__doc__ = """
Pseudo-random Weibull-distributed deviate for shape parameter `a` and scale parameter `b`.

The Weibull distribution is related to a number of other probability distributions;  in particular,
it interpolates between the exponential distribution (a=1) and the Rayleigh distribution (a=2). 
See http://en.wikipedia.org/wiki/Weibull_distribution (a=k and b=lambda in the notation adopted in 
the Wikipedia article) for more details.  The Weibull distribution is real valued and produces 
deviates >= 0.

Initialization
--------------

@param seed         Something that can seed a BaseDeviate: a long int seed or another 
                    BaseDeviate.  Using 0 means to use the time of day as a seed. [default: 0]
@param a            Shape parameter of the distribution. [default: 1; Must be > 0]
@param b            Scale parameter of the distribution. [default: 1; Must be > 0]

Calling
-------

Successive calls to p() generate pseudo-random values distributed according to a Weibull
distribution with the specified shape and scale parameters `a` and `b`.

    >>> w = galsim.WeibullDeviate(31415926, a=1.3, b=4)
    >>> w()
    1.1038481241018219
    >>> w()
    2.957052966368049
"""

_galsim.WeibullDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Weibull-distributed deviate with current `a` and `b`.
"""
_galsim.WeibullDeviate.getA.__func__.__doc__ = "Get current distribution shape parameter `a`."
_galsim.WeibullDeviate.setA.__func__.__doc__ = "Set current distribution shape parameter `a`. Discouraged."
_galsim.WeibullDeviate.getB.__func__.__doc__ = "Get current distribution shape parameter `b`."
_galsim.WeibullDeviate.setB.__func__.__doc__ = "Set current distribution shape parameter `b`. Discouraged."


# GammaDeviate docstrings
_galsim.GammaDeviate.__doc__ = """
A Gamma-distributed deviate with shape parameter `k` and scale parameter `theta`.
See http://en.wikipedia.org/wiki/Gamma_distribution.  
(Note: we use the k, theta notation. If you prefer alpha, beta, use k=alpha, theta=1/beta.)
The Gamma distribution is a real valued distribution producing deviates >= 0.

Initialization
--------------

@param seed         Something that can seed a BaseDeviate: a long int seed or another 
                    BaseDeviate.  Using 0 means to use the time of day as a seed. [default: 0]
@param k            Shape parameter of the distribution. [default: 1; Must be > 0]
@param theta        Scale parameter of the distribution. [default: 1; Must be > 0]

Calling
-------

Successive calls to p() generate pseudo-random values distributed according to a gamma
distribution with the specified shape and scale parameters `k` and `theta`.

    >>> gam = galsim.GammaDeviate(31415926, k=1, theta=2)
    >>> gam()
    0.37508882726316
    >>> gam()
    1.3504199388358704
"""

_galsim.GammaDeviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Gamma-distributed deviate with current k and theta.
"""
_galsim.GammaDeviate.getK.__func__.__doc__ = "Get current distribution shape parameter `k`."
_galsim.GammaDeviate.setK.__func__.__doc__ = "Set current distribution shape parameter `k`. Discouraged."
_galsim.GammaDeviate.getTheta.__func__.__doc__ = "Get current distribution shape parameter `theta`."
_galsim.GammaDeviate.setTheta.__func__.__doc__ = "Set current distribution shape parameter `theta`. Discouraged."


# Chi2Deviate docstrings
_galsim.Chi2Deviate.__doc__ = """
Pseudo-random Chi^2-distributed deviate for degrees-of-freedom parameter `n`.

See http://en.wikipedia.org/wiki/Chi-squared_distribution (note that k=n in the notation adopted in
the Boost.Random routine called by this class).  The Chi^2 distribution is a real-valued 
distribution producing deviates >= 0.

Initialization
--------------

@param seed         Something that can seed a BaseDeviate: a long int seed or another 
                    BaseDeviate.  Using 0 means to use the time of day as a seed. [default: 0]
@param n            Number of degrees of freedom for the output distribution. [default: 1; 
                    Must be > 0]

Calling
-------

Successive calls to chi2() generate pseudo-random values distributed according to a chi-square
distribution with the specified degrees of freedom, `n`.

    >>> chi2 = galsim.Chi2Deviate(31415926, n=7)
    >>> chi2()
    7.9182211987712385
    >>> chi2()
    6.644121724269535
"""

_galsim.Chi2Deviate.__call__.__func__.__doc__ = """
Draw a new random number from the distribution.

Returns a Chi2-distributed deviate with current `n` degrees of freedom.
"""
_galsim.Chi2Deviate.getN.__func__.__doc__ = "Get current distribution `n` degrees of freedom."
_galsim.Chi2Deviate.setN.__func__.__doc__ = "Set current distribution `n` degrees of freedom. Discouraged."


# Some functions to enable pickling of deviates
def BaseDeviate_getinitargs(self):
    return self.serialize(), 
_galsim.BaseDeviate.__getinitargs__ = BaseDeviate_getinitargs

def UniformDeviate_getinitargs(self):
    return self.serialize(),
_galsim.UniformDeviate.__getinitargs__ = UniformDeviate_getinitargs

def GaussianDeviate_getinitargs(self):
    return self.serialize(), self.getMean(), self.getSigma()
_galsim.GaussianDeviate.__getinitargs__ = GaussianDeviate_getinitargs

def BinomialDeviate_getinitargs(self):
    return self.serialize(), self.getN(), self.getP()
_galsim.BinomialDeviate.__getinitargs__ = BinomialDeviate_getinitargs

def PoissonDeviate_getinitargs(self):
    return self.serialize(), self.getMean()
_galsim.PoissonDeviate.__getinitargs__ = PoissonDeviate_getinitargs

def WeibullDeviate_getinitargs(self):
    return self.serialize(), self.getA(), self.getB()
_galsim.WeibullDeviate.__getinitargs__ = WeibullDeviate_getinitargs

def GammaDeviate_getinitargs(self):
    return self.serialize(), self.getK(), self.getTheta()
_galsim.GammaDeviate.__getinitargs__ = GammaDeviate_getinitargs

def Chi2Deviate_getinitargs(self):
    return self.serialize(), self.getN()
_galsim.Chi2Deviate.__getinitargs__ = Chi2Deviate_getinitargs

