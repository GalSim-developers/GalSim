# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from __future__ import print_function
import numpy as np
import os
import sys

import galsim
from galsim_test_helpers import *


#
# Note: all tests below were generated using the python interface to the RNG.  Eventually need tests
# for comparison against the C++!
#

precision = 10
# decimal point at which agreement is required for all double precision tests

precisionD = precision
precisionF = 5  # precision=10 does not make sense at single precision
precisionS = 1  # "precision" also a silly concept for ints, but allows all 4 tests to run in one go
precisionI = 1

# The number of values to generate when checking the mean and variance calculations.
# This is currently low enough to not dominate the time of the unit tests, but when changing
# something, it may be useful to add a couple zeros while testing.
nvals = 100000

testseed = 1000 # seed used for UniformDeviate for all tests
# Warning! If you change testseed, then all of the *Result variables below must change as well.

# the right answer for the first three uniform deviates produced from testseed
uResult = (0.11860922840423882, 0.21456799632869661, 0.43088198406621814)

# mean, sigma to use for Gaussian tests
gMean = 4.7
gSigma = 3.2
# the right answer for the first three Gaussian deviates produced from testseed
gResult = (6.3344979808161215, 6.2082355273987861, -0.069894693358302007)

# N, p to use for binomial tests
bN = 10
bp = 0.7
# the right answer for the first three binomial deviates produced from testseed
bResult = (9, 8, 7)

# mean to use for Poisson tests
pMean = 7
# the right answer for the first three Poisson deviates produced from testseed
pResult = (4, 5, 6)

# a & b to use for Weibull tests
wA = 4.
wB = 9.
# Tabulated results for Weibull
wResult = (5.3648053017485591, 6.3093033550873878, 7.7982696798921074)

# k & theta to use for Gamma tests
gammaK = 1.5
gammaTheta = 4.5
# Tabulated results for Gamma
gammaResult = (4.7375613139927157, 15.272973580418618, 21.485016362839747)

# n to use for Chi2 tests
chi2N = 30
# Tabulated results for Chi2
chi2Result = (32.209933900954049, 50.040002656028513, 24.301442486313896)

#function and min&max to use for DistDeviate function call tests
dmin=0.0
dmax=2.0
def dfunction(x):
    return x*x
# Tabulated results for DistDeviate function call
dFunctionResult = (0.9826461346196363, 1.1973307331701328, 1.5105900949284945)

# x and p arrays and interpolant to use for DistDeviate LookupTable tests
dx=[0.0, 1.0, 1.000000001, 2.999999999, 3.0, 4.0]
dp=[0.1, 0.1, 0.0    , 0.0    , 0.1, 0.1]
dLookupTable=galsim.LookupTable(x=dx,f=dp,interpolant='linear')
# Tabulated results for DistDeviate LookupTable call
dLookupTableResult = (0.23721845680847731, 0.42913599265739233, 0.86176396813243539)
# File with the same values
dLookupTableFile = os.path.join('random_data','dLookupTable.dat')


@timer
def test_uniform():
    """Test uniform random number generator
    """
    u = galsim.UniformDeviate(testseed)
    u2 = u.duplicate()
    u3 = galsim.UniformDeviate(u.serialize())
    testResult = (u(), u(), u())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(uResult), precision,
            err_msg='Wrong uniform random number sequence generated')
    testResult = (u2(), u2(), u2())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(uResult), precision,
            err_msg='Wrong uniform random number sequence generated with duplicate')
    testResult = (u3(), u3(), u3())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(uResult), precision,
            err_msg='Wrong uniform random number sequence generated from serialize')

    # Check that the mean and variance come out right
    vals = [u() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = 1./2.
    v = 1./12.
    print('mean = ',mean,'  true mean = ',mu)
    print('var = ',var,'   true var = ',v)
    np.testing.assert_almost_equal(mean, mu, 1,
            err_msg='Wrong mean from UniformDeviate')
    np.testing.assert_almost_equal(var, v, 1,
            err_msg='Wrong variance from UniformDeviate')

    # Check seed, reset
    u.seed(testseed)
    testResult2 = (u(), u(), u())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong uniform random number sequence generated after seed')

    u.reset(testseed)
    testResult2 = (u(), u(), u())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong uniform random number sequence generated after reset(seed)')

    rng = galsim.BaseDeviate(testseed)
    u.reset(rng)
    testResult2 = (u(), u(), u())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong uniform random number sequence generated after reset(rng)')

    # Check dump, raw
    np.testing.assert_equal(u.raw(), u2.raw(),
            err_msg='Uniform deviates generate different raw values')

    rng2 = galsim.BaseDeviate(testseed)
    rng2.discard(4)
    np.testing.assert_equal(rng.raw(), rng2.raw(),
            err_msg='BaseDeviates generate different raw values after discard')

    # Check that two connected uniform deviates work correctly together.
    u2 = galsim.UniformDeviate(testseed)
    u.reset(u2)
    testResult2 = (u(), u2(), u())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong uniform random number sequence generated using two uds')
    u.seed(testseed)
    testResult2 = (u2(), u(), u2())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong uniform random number sequence generated using two uds after seed')

    # Check that seeding with the time works (although we cannot check the output).
    # We're mostly just checking that this doesn't raise an exception.
    # The output could be anything.
    u.seed()
    testResult2 = (u(), u(), u())
    assert testResult2 != testResult
    u.reset()
    testResult3 = (u(), u(), u())
    assert testResult3 != testResult
    assert testResult3 != testResult2
    u.reset()
    testResult4 = (u(), u(), u())
    assert testResult4 != testResult
    assert testResult4 != testResult2
    assert testResult4 != testResult3
    u = galsim.UniformDeviate()
    testResult5 = (u(), u(), u())
    assert testResult5 != testResult
    assert testResult5 != testResult2
    assert testResult5 != testResult3
    assert testResult5 != testResult4

    # Test generate
    u.seed(testseed)
    test_array = np.empty(3)
    u.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(uResult), precision,
            err_msg='Wrong uniform random number sequence from generate.')

    # Test add_generate
    u.seed(testseed)
    u.add_generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, 2.*np.array(uResult), precision,
            err_msg='Wrong uniform random number sequence from generate.')

    # Test generate with a float32 array
    u.seed(testseed)
    test_array = np.empty(3, dtype=np.float32)
    u.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(uResult), precisionF,
            err_msg='Wrong uniform random number sequence from generate.')

    # Test add_generate
    u.seed(testseed)
    u.add_generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, 2.*np.array(uResult), precisionF,
            err_msg='Wrong uniform random number sequence from generate.')

    # Check picklability
    do_pickle(u, lambda x: x.serialize())
    do_pickle(u, lambda x: (x(), x(), x(), x()))
    do_pickle(u)
    do_pickle(rng)
    assert 'UniformDeviate' in repr(u)
    assert 'UniformDeviate' in str(u)
    assert isinstance(eval(repr(u)), galsim.UniformDeviate)
    assert isinstance(eval(str(u)), galsim.UniformDeviate)
    assert isinstance(eval(repr(rng)), galsim.BaseDeviate)
    assert isinstance(eval(str(rng)), galsim.BaseDeviate)

    # Check that we can construct a UniformDeviate from None, and that it depends on dev/random.
    u1 = galsim.UniformDeviate(None)
    u2 = galsim.UniformDeviate(None)
    assert u1 != u2, "Consecutive UniformDeviate(None) compared equal!"
    # We shouldn't be able to construct a UniformDeviate from anything but a BaseDeviate, int, str,
    # or None.
    assert_raises(TypeError, galsim.UniformDeviate, dict())
    assert_raises(TypeError, galsim.UniformDeviate, list())
    assert_raises(TypeError, galsim.UniformDeviate, set())

    assert_raises(TypeError, u.seed, '123')
    assert_raises(TypeError, u.seed, 12.3)


@timer
def test_gaussian():
    """Test Gaussian random number generator
    """
    g = galsim.GaussianDeviate(testseed, mean=gMean, sigma=gSigma)
    g2 = g.duplicate()
    g3 = galsim.GaussianDeviate(g.serialize(), mean=gMean, sigma=gSigma)
    testResult = (g(), g(), g())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(gResult), precision,
            err_msg='Wrong Gaussian random number sequence generated')
    testResult = (g2(), g2(), g2())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(gResult), precision,
            err_msg='Wrong Gaussian random number sequence generated with duplicate')
    testResult = (g3(), g3(), g3())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(gResult), precision,
            err_msg='Wrong Gaussian random number sequence generated from serialize')

    # Check that the mean and variance come out right
    vals = [g() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = gMean
    v = gSigma**2
    print('mean = ',mean,'  true mean = ',mu)
    print('var = ',var,'   true var = ',v)
    np.testing.assert_almost_equal(mean, mu, 1,
            err_msg='Wrong mean from GaussianDeviate')
    np.testing.assert_almost_equal(var, v, 0,
            err_msg='Wrong variance from GaussianDeviate')

    # Check seed, reset
    g.seed(testseed)
    testResult2 = (g(), g(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Gaussian random number sequence generated after seed')

    g.reset(testseed)
    testResult2 = (g(), g(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Gaussian random number sequence generated after reset(seed)')

    rng = galsim.BaseDeviate(testseed)
    g.reset(rng)
    testResult2 = (g(), g(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Gaussian random number sequence generated after reset(rng)')

    ud = galsim.UniformDeviate(testseed)
    g.reset(ud)
    testResult = (g(), g(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Gaussian random number sequence generated after reset(ud)')

    # Check that two connected Gaussian deviates work correctly together.
    g2 = galsim.GaussianDeviate(testseed, mean=gMean, sigma=gSigma)
    g.reset(g2)
    # Note: GaussianDeviate generates two values at a time, so we have to compare them in pairs.
    testResult2 = (g(), g(), g2())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Gaussian random number sequence generated using two gds')
    g.seed(testseed)
    # For the same reason, after seeding one, we need to manually clear the other's cache:
    g2.clearCache()
    testResult2 = (g2(), g2(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Gaussian random number sequence generated using two gds after seed')

    # Check that seeding with the time works (although we cannot check the output).
    # We're mostly just checking that this doesn't raise an exception.
    # The output could be anything.
    g.seed()
    testResult2 = (g(), g(), g())
    assert testResult2 != testResult
    g.reset()
    testResult3 = (g(), g(), g())
    assert testResult3 != testResult
    assert testResult3 != testResult2
    g.reset()
    testResult4 = (g(), g(), g())
    assert testResult4 != testResult
    assert testResult4 != testResult2
    assert testResult4 != testResult3
    g = galsim.GaussianDeviate(mean=gMean, sigma=gSigma)
    testResult5 = (g(), g(), g())
    assert testResult5 != testResult
    assert testResult5 != testResult2
    assert testResult5 != testResult3
    assert testResult5 != testResult4

    # Test generate
    g.seed(testseed)
    test_array = np.empty(3)
    g.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(gResult), precision,
            err_msg='Wrong Gaussian random number sequence from generate.')

    # Test generate_from_variance.
    g2 = galsim.GaussianDeviate(testseed)
    test_array.fill(gSigma**2)
    g2.generate_from_variance(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(gResult)-gMean, precision,
            err_msg='Wrong Gaussian random number sequence from generate_from_variance.')

    # Test generate with a float32 array.
    g.seed(testseed)
    test_array = np.empty(3, dtype=np.float32)
    g.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(gResult), precisionF,
            err_msg='Wrong Gaussian random number sequence from generate.')

    # Test generate_from_variance.
    g2.seed(testseed)
    test_array.fill(gSigma**2)
    g2.generate_from_variance(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(gResult)-gMean, precisionF,
            err_msg='Wrong Gaussian random number sequence from generate_from_variance.')

    # Check picklability
    do_pickle(g, lambda x: (x.serialize(), x.mean, x.sigma))
    do_pickle(g, lambda x: (x(), x(), x(), x()))
    do_pickle(g)
    assert 'GaussianDeviate' in repr(g)
    assert 'GaussianDeviate' in str(g)
    assert isinstance(eval(repr(g)), galsim.GaussianDeviate)
    assert isinstance(eval(str(g)), galsim.GaussianDeviate)

    # Check that we can construct a GaussianDeviate from None, and that it depends on dev/random.
    g1 = galsim.GaussianDeviate(None)
    g2 = galsim.GaussianDeviate(None)
    assert g1 != g2, "Consecutive GaussianDeviate(None) compared equal!"
    # We shouldn't be able to construct a GaussianDeviate from anything but a BaseDeviate, int, str,
    # or None.
    assert_raises(TypeError, galsim.GaussianDeviate, dict())
    assert_raises(TypeError, galsim.GaussianDeviate, list())
    assert_raises(TypeError, galsim.GaussianDeviate, set())

    assert_raises(ValueError, galsim.GaussianDeviate, testseed, mean=1, sigma=-1)


@timer
def test_binomial():
    """Test binomial random number generator
    """
    b = galsim.BinomialDeviate(testseed, N=bN, p=bp)
    b2 = b.duplicate()
    b3 = galsim.BinomialDeviate(b.serialize(), N=bN, p=bp)
    testResult = (b(), b(), b())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(bResult), precision,
            err_msg='Wrong binomial random number sequence generated')
    testResult = (b2(), b2(), b2())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(bResult), precision,
            err_msg='Wrong binomial random number sequence generated with duplicate')
    testResult = (b3(), b3(), b3())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(bResult), precision,
            err_msg='Wrong binomial random number sequence generated from serialize')

    # Check that the mean and variance come out right
    vals = [b() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = bN*bp
    v = bN*bp*(1.-bp)
    print('mean = ',mean,'  true mean = ',mu)
    print('var = ',var,'   true var = ',v)
    np.testing.assert_almost_equal(mean, mu, 1,
            err_msg='Wrong mean from BinomialDeviate')
    np.testing.assert_almost_equal(var, v, 1,
            err_msg='Wrong variance from BinomialDeviate')

    # Check seed, reset
    b.seed(testseed)
    testResult2 = (b(), b(), b())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong binomial random number sequence generated after seed')

    b.reset(testseed)
    testResult2 = (b(), b(), b())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong binomial random number sequence generated after reset(seed)')

    rng = galsim.BaseDeviate(testseed)
    b.reset(rng)
    testResult2 = (b(), b(), b())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong binomial random number sequence generated after reset(rng)')

    ud = galsim.UniformDeviate(testseed)
    b.reset(ud)
    testResult = (b(), b(), b())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong binomial random number sequence generated after reset(ud)')

    # Check that two connected binomial deviates work correctly together.
    b2 = galsim.BinomialDeviate(testseed, N=bN, p=bp)
    b.reset(b2)
    testResult2 = (b(), b2(), b())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong binomial random number sequence generated using two bds')
    b.seed(testseed)
    testResult2 = (b2(), b(), b2())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong binomial random number sequence generated using two bds after seed')

    # Check that seeding with the time works (although we cannot check the output).
    # We're mostly just checking that this doesn't raise an exception.
    # The output could be anything.  However, in this case, there are few enough options
    # for the output that occasionally two of these match.  So we don't do the normal
    # testResult2 != testResult, etc.
    b.seed()
    testResult2 = (b(), b(), b())
    #assert testResult2 != testResult
    b.reset()
    testResult3 = (b(), b(), b())
    #assert testResult3 != testResult
    #assert testResult3 != testResult2
    b.reset()
    testResult4 = (b(), b(), b())
    #assert testResult4 != testResult
    #assert testResult4 != testResult2
    #assert testResult4 != testResult3
    b = galsim.BinomialDeviate(N=bN, p=bp)
    testResult5 = (b(), b(), b())
    #assert testResult5 != testResult
    #assert testResult5 != testResult2
    #assert testResult5 != testResult3
    #assert testResult5 != testResult4

    # Test generate
    b.seed(testseed)
    test_array = np.empty(3)
    b.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(bResult), precision,
            err_msg='Wrong binomial random number sequence from generate.')

    # Test generate with an int array
    b.seed(testseed)
    test_array = np.empty(3, dtype=np.int)
    b.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(bResult), precisionI,
            err_msg='Wrong binomial random number sequence from generate.')

    # Check picklability
    do_pickle(b, lambda x: (x.serialize(), x.n, x.p))
    do_pickle(b, lambda x: (x(), x(), x(), x()))
    do_pickle(b)
    assert 'BinomialDeviate' in repr(b)
    assert 'BinomialDeviate' in str(b)
    assert isinstance(eval(repr(b)), galsim.BinomialDeviate)
    assert isinstance(eval(str(b)), galsim.BinomialDeviate)

    # Check that we can construct a BinomialDeviate from None, and that it depends on dev/random.
    b1 = galsim.BinomialDeviate(None)
    b2 = galsim.BinomialDeviate(None)
    assert b1 != b2, "Consecutive BinomialDeviate(None) compared equal!"
    # We shouldn't be able to construct a BinomialDeviate from anything but a BaseDeviate, int, str,
    # or None.
    assert_raises(TypeError, galsim.BinomialDeviate, dict())
    assert_raises(TypeError, galsim.BinomialDeviate, list())
    assert_raises(TypeError, galsim.BinomialDeviate, set())


@timer
def test_poisson():
    """Test Poisson random number generator
    """
    p = galsim.PoissonDeviate(testseed, mean=pMean)
    p2 = p.duplicate()
    p3 = galsim.PoissonDeviate(p.serialize(), mean=pMean)
    testResult = (p(), p(), p())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(pResult), precision,
            err_msg='Wrong Poisson random number sequence generated')
    testResult = (p2(), p2(), p2())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(pResult), precision,
            err_msg='Wrong Poisson random number sequence generated with duplicate')
    testResult = (p3(), p3(), p3())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(pResult), precision,
            err_msg='Wrong Poisson random number sequence generated from serialize')

    # Check that the mean and variance come out right
    vals = [p() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = pMean
    v = pMean
    print('mean = ',mean,'  true mean = ',mu)
    print('var = ',var,'   true var = ',v)
    np.testing.assert_almost_equal(mean, mu, 1,
            err_msg='Wrong mean from PoissonDeviate')
    np.testing.assert_almost_equal(var, v, 1,
            err_msg='Wrong variance from PoissonDeviate')

    # Check seed, reset
    p.seed(testseed)
    testResult2 = (p(), p(), p())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong poisson random number sequence generated after seed')

    p.reset(testseed)
    testResult2 = (p(), p(), p())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong poisson random number sequence generated after reset(seed)')

    rng = galsim.BaseDeviate(testseed)
    p.reset(rng)
    testResult2 = (p(), p(), p())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong poisson random number sequence generated after reset(rng)')

    ud = galsim.UniformDeviate(testseed)
    p.reset(ud)
    testResult = (p(), p(), p())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong poisson random number sequence generated after reset(ud)')

    # Check that two connected poisson deviates work correctly together.
    p2 = galsim.PoissonDeviate(testseed, mean=pMean)
    p.reset(p2)
    testResult2 = (p(), p2(), p())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong poisson random number sequence generated using two pds')
    p.seed(testseed)
    testResult2 = (p2(), p(), p2())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong poisson random number sequence generated using two pds after seed')

    # Check that seeding with the time works (although we cannot check the output).
    # We're mostly just checking that this doesn't raise an exception.
    # The output could be anything.  However, in this case, there are few enough options
    # for the output that occasionally two of these match.  So we don't do the normal
    # testResult2 != testResult, etc.
    p.seed()
    testResult2 = (p(), p(), p())
    #assert testResult2 != testResult
    p.reset()
    testResult3 = (p(), p(), p())
    #assert testResult3 != testResult
    #assert testResult3 != testResult2
    p.reset()
    testResult4 = (p(), p(), p())
    #assert testResult4 != testResult
    #assert testResult4 != testResult2
    #assert testResult4 != testResult3
    p = galsim.PoissonDeviate(mean=pMean)
    testResult5 = (p(), p(), p())
    #assert testResult5 != testResult
    #assert testResult5 != testResult2
    #assert testResult5 != testResult3
    #assert testResult5 != testResult4

    # Test generate
    p.seed(testseed)
    test_array = np.empty(3)
    p.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(pResult), precision,
            err_msg='Wrong poisson random number sequence from generate.')

    # Test generate_from_expectation
    p.seed(testseed)
    test_array = np.array([pMean]*3)
    p.generate_from_expectation(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(pResult), precision,
            err_msg='Wrong poisson random number sequence from generate_from_expectation.')

    # Test generate with an int array
    p.seed(testseed)
    test_array = np.empty(3, dtype=int)
    p.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(pResult), precisionI,
            err_msg='Wrong poisson random number sequence from generate.')

    # Test generate_from_expectation
    p.seed(testseed)
    test_array = np.array([pMean]*3, dtype=int)
    p.generate_from_expectation(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(pResult), precisionI,
            err_msg='Wrong poisson random number sequence from generate_from_expectation.')

    # Check picklability
    do_pickle(p, lambda x: (x.serialize(), x.mean))
    do_pickle(p, lambda x: (x(), x(), x(), x()))
    do_pickle(p)
    assert 'PoissonDeviate' in repr(p)
    assert 'PoissonDeviate' in str(p)
    assert isinstance(eval(repr(p)), galsim.PoissonDeviate)
    assert isinstance(eval(str(p)), galsim.PoissonDeviate)

    # Check that we can construct a PoissonDeviate from None, and that it depends on dev/random.
    p1 = galsim.PoissonDeviate(None)
    p2 = galsim.PoissonDeviate(None)
    assert p1 != p2, "Consecutive PoissonDeviate(None) compared equal!"
    # We shouldn't be able to construct a PoissonDeviate from anything but a BaseDeviate, int, str,
    # or None.
    assert_raises(TypeError, galsim.PoissonDeviate, dict())
    assert_raises(TypeError, galsim.PoissonDeviate, list())
    assert_raises(TypeError, galsim.PoissonDeviate, set())


@timer
def test_poisson_highmean():
    """Test Poisson random number generator with high (>2^30) mean (cf. Issue #881)

    It turns out that the boost poisson deviate class that we use maxes out at 2^31 and wraps
    around to -2^31.  We have code to automatically switch over to using a Gaussian deviate
    instead if the mean > 2^30 (factor of 2 from the problem to be safe).  Check that this
    works properly.
    """
    mean_vals =[ 2**30 + 50,  # Uses Gaussian
                 2**30 - 50,  # Uses Poisson
                 2**30,       # Uses Poisson (highest value of mean that does)
                 2**31,       # This is where problems happen if not using Gaussian
                 5.e20,       # Definitely would have problems with normal implementation.
               ]

    if __name__ == '__main__':
        nvals = 10000000
        rtol_var = 1.e-3
    else:
        nvals = 100000
        rtol_var = 1.e-2

    for mean in mean_vals:
        print('Test PoissonDeviate with mean = ',mean)
        p = galsim.PoissonDeviate(testseed, mean=mean)
        p2 = p.duplicate()
        p3 = galsim.PoissonDeviate(p.serialize(), mean=mean)
        testResult = (p(), p(), p())
        testResult2 = (p2(), p2(), p2())
        testResult3 = (p3(), p3(), p3())
        np.testing.assert_allclose(
                testResult2, testResult, rtol=1.e-8,
                err_msg='PoissonDeviate.duplicate not equivalent for mean=%s'%mean)
        np.testing.assert_allclose(
                testResult3, testResult, rtol=1.e-8,
                err_msg='PoissonDeviate from serialize not equivalent for mean=%s'%mean)

        # Check that the mean and variance come out right
        vals = [p() for i in range(nvals)]
        mu = np.mean(vals)
        var = np.var(vals)
        print('mean = ',mu,'  true mean = ',mean)
        print('var = ',var,'   true var = ',mean)
        np.testing.assert_allclose(mu, mean, rtol=1.e-5,
                err_msg='Wrong mean from PoissonDeviate with mean=%s'%mean)
        np.testing.assert_allclose(var, mean, rtol=rtol_var,
                err_msg='Wrong variance from PoissonDeviate with mean=%s'%mean)

        # Check that two connected poisson deviates work correctly together.
        p2 = galsim.PoissonDeviate(testseed, mean=mean)
        p.reset(p2)
        testResult2 = (p(), p(), p2())
        np.testing.assert_array_equal(
                testResult2, testResult,
                err_msg='Wrong poisson random number sequence generated using two pds')
        p.seed(testseed)
        p2.clearCache()
        testResult2 = (p2(), p2(), p())
        np.testing.assert_array_equal(
                testResult2, testResult,
                err_msg='Wrong poisson random number sequence generated using two pds after seed')

        # Test filling an image
        p.seed(testseed)
        testimage = galsim.ImageD(np.zeros((3, 1)))
        testimage.addNoise(galsim.DeviateNoise(p))
        np.testing.assert_array_equal(
                testimage.array.flatten(), testResult,
                err_msg='Wrong poisson random number sequence generated when applied to image.')

        # The PoissonNoise version also subtracts off the mean value
        rng = galsim.BaseDeviate(testseed)
        pn = galsim.PoissonNoise(rng, sky_level=mean)
        testimage.fill(0)
        testimage.addNoise(pn)
        np.testing.assert_array_equal(
                testimage.array.flatten(), np.array(testResult)-mean,
                err_msg='Wrong poisson random number sequence generated using PoissonNoise')

        # Check PoissonNoise variance:
        np.testing.assert_allclose(
                pn.getVariance(), mean, rtol=1.e-8,
                err_msg="PoissonNoise getVariance returns wrong variance")
        np.testing.assert_allclose(
                pn.sky_level, mean, rtol=1.e-8,
                err_msg="PoissonNoise sky_level returns wrong value")

        # Check that the noise model really does produce this variance.
        big_im = galsim.Image(2048,2048,dtype=float)
        big_im.addNoise(pn)
        var = np.var(big_im.array)
        print('variance = ',var)
        print('getVar = ',pn.getVariance())
        np.testing.assert_allclose(
                var, pn.getVariance(), rtol=rtol_var,
                err_msg='Realized variance for PoissonNoise did not match getVariance()')


@timer
def test_poisson_zeromean():
    """Make sure Poisson Deviate behaves sensibly when mean=0.
    """
    p = galsim.PoissonDeviate(testseed, mean=0)
    p2 = p.duplicate()
    p3 = galsim.PoissonDeviate(p.serialize(), mean=0)
    do_pickle(p)

    # Test direct draws
    testResult = (p(), p(), p())
    testResult2 = (p2(), p2(), p2())
    testResult3 = (p3(), p3(), p3())
    np.testing.assert_array_equal(testResult, 0)
    np.testing.assert_array_equal(testResult2, 0)
    np.testing.assert_array_equal(testResult3, 0)

    # Test generate
    test_array = np.empty(3, dtype=int)
    p.generate(test_array)
    np.testing.assert_array_equal(test_array, 0)
    p2.generate(test_array)
    np.testing.assert_array_equal(test_array, 0)
    p3.generate(test_array)
    np.testing.assert_array_equal(test_array, 0)

    # Test generate_from_expectation
    test_array = np.array([0,0,0])
    np.testing.assert_allclose(test_array, 0)
    test_array = np.array([1,0,4])
    assert test_array[0] != 0
    assert test_array[1] == 0
    assert test_array[2] != 0

    # Error raised if mean<0
    with assert_raises(ValueError):
        p = galsim.PoissonDeviate(testseed, mean=-0.1)
    with assert_raises(ValueError):
        p = galsim.PoissonDeviate(testseed, mean=-10)
    test_array = np.array([-1,1,4])
    with assert_raises(ValueError):
        p.generate_from_expectation(test_array)
    test_array = np.array([1,-1,-4])
    with assert_raises(ValueError):
        p.generate_from_expectation(test_array)

@timer
def test_weibull():
    """Test Weibull random number generator
    """
    w = galsim.WeibullDeviate(testseed, a=wA, b=wB)
    w2 = w.duplicate()
    w3 = galsim.WeibullDeviate(w.serialize(), a=wA, b=wB)
    testResult = (w(), w(), w())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(wResult), precision,
            err_msg='Wrong Weibull random number sequence generated')
    testResult = (w2(), w2(), w2())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(wResult), precision,
            err_msg='Wrong Weibull random number sequence generated with duplicate')
    testResult = (w3(), w3(), w3())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(wResult), precision,
            err_msg='Wrong Weibull random number sequence generated from serialize')

    # Check that the mean and variance come out right
    vals = [w() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    try:
        import math
        gammaFactor1 = math.gamma(1.+1./wA)
        gammaFactor2 = math.gamma(1.+2./wA)
    except:
        # gamma was introduced in python 2.7, so these are the correct answers for the
        # current wA.  Need to change this if wA is chagned.
        # (Values obtained from Wolfram Alpha.)
        gammaFactor1 = 0.906402477055477
        gammaFactor2 = 0.886226925452758
    mu = wB * gammaFactor1
    v = wB**2 * gammaFactor2 - mu**2
    print('mean = ',mean,'  true mean = ',mu)
    print('var = ',var,'   true var = ',v)
    np.testing.assert_almost_equal(mean, mu, 1,
            err_msg='Wrong mean from WeibullDeviate')
    np.testing.assert_almost_equal(var, v, 1,
            err_msg='Wrong variance from WeibullDeviate')

    # Check seed, reset
    w.seed(testseed)
    testResult2 = (w(), w(), w())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong weibull random number sequence generated after seed')

    w.reset(testseed)
    testResult2 = (w(), w(), w())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong weibull random number sequence generated after reset(seed)')

    rng = galsim.BaseDeviate(testseed)
    w.reset(rng)
    testResult2 = (w(), w(), w())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong weibull random number sequence generated after reset(rng)')

    ud = galsim.UniformDeviate(testseed)
    w.reset(ud)
    testResult = (w(), w(), w())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong weibull random number sequence generated after reset(ud)')

    # Check that two connected weibull deviates work correctly together.
    w2 = galsim.WeibullDeviate(testseed, a=wA, b=wB)
    w.reset(w2)
    testResult2 = (w(), w2(), w())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong weibull random number sequence generated using two wds')
    w.seed(testseed)
    testResult2 = (w2(), w(), w2())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong weibull random number sequence generated using two wds after seed')

    # Check that seeding with the time works (although we cannot check the output).
    # We're mostly just checking that this doesn't raise an exception.
    # The output could be anything.
    w.seed()
    testResult2 = (w(), w(), w())
    assert testResult2 != testResult
    w.reset()
    testResult3 = (w(), w(), w())
    assert testResult3 != testResult
    assert testResult3 != testResult2
    w.reset()
    testResult4 = (w(), w(), w())
    assert testResult4 != testResult
    assert testResult4 != testResult2
    assert testResult4 != testResult3
    w = galsim.WeibullDeviate(a=wA, b=wB)
    testResult5 = (w(), w(), w())
    assert testResult5 != testResult
    assert testResult5 != testResult2
    assert testResult5 != testResult3
    assert testResult5 != testResult4

    # Test generate
    w.seed(testseed)
    test_array = np.empty(3)
    w.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(wResult), precision,
            err_msg='Wrong weibull random number sequence from generate.')

    # Test generate with a float32 array
    w.seed(testseed)
    test_array = np.empty(3, dtype=np.float32)
    w.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(wResult), precisionF,
            err_msg='Wrong weibull random number sequence from generate.')

    # Check picklability
    do_pickle(w, lambda x: (x.serialize(), x.a, x.b))
    do_pickle(w, lambda x: (x(), x(), x(), x()))
    do_pickle(w)
    assert 'WeibullDeviate' in repr(w)
    assert 'WeibullDeviate' in str(w)
    assert isinstance(eval(repr(w)), galsim.WeibullDeviate)
    assert isinstance(eval(str(w)), galsim.WeibullDeviate)

    # Check that we can construct a WeibullDeviate from None, and that it depends on dev/random.
    w1 = galsim.WeibullDeviate(None)
    w2 = galsim.WeibullDeviate(None)
    assert w1 != w2, "Consecutive WeibullDeviate(None) compared equal!"
    # We shouldn't be able to construct a WeibullDeviate from anything but a BaseDeviate, int, str,
    # or None.
    assert_raises(TypeError, galsim.WeibullDeviate, dict())
    assert_raises(TypeError, galsim.WeibullDeviate, list())
    assert_raises(TypeError, galsim.WeibullDeviate, set())


@timer
def test_gamma():
    """Test Gamma random number generator
    """
    g = galsim.GammaDeviate(testseed, k=gammaK, theta=gammaTheta)
    g2 = g.duplicate()
    g3 = galsim.GammaDeviate(g.serialize(), k=gammaK, theta=gammaTheta)
    testResult = (g(), g(), g())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(gammaResult), precision,
            err_msg='Wrong Gamma random number sequence generated')
    testResult = (g2(), g2(), g2())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(gammaResult), precision,
            err_msg='Wrong Gamma random number sequence generated with duplicate')
    testResult = (g3(), g3(), g3())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(gammaResult), precision,
            err_msg='Wrong Gamma random number sequence generated from serialize')

    # Check that the mean and variance come out right
    vals = [g() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = gammaK*gammaTheta
    v = gammaK*gammaTheta**2
    print('mean = ',mean,'  true mean = ',mu)
    print('var = ',var,'   true var = ',v)
    np.testing.assert_almost_equal(mean, mu, 1,
            err_msg='Wrong mean from GammaDeviate')
    np.testing.assert_almost_equal(var, v, 0,
            err_msg='Wrong variance from GammaDeviate')

    # Check seed, reset
    g.seed(testseed)
    testResult2 = (g(), g(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong gamma random number sequence generated after seed')

    g.reset(testseed)
    testResult2 = (g(), g(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong gamma random number sequence generated after reset(seed)')

    rng = galsim.BaseDeviate(testseed)
    g.reset(rng)
    testResult2 = (g(), g(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong gamma random number sequence generated after reset(rng)')

    ud = galsim.UniformDeviate(testseed)
    g.reset(ud)
    testResult = (g(), g(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong gamma random number sequence generated after reset(ud)')

    # Check that two connected gamma deviates work correctly together.
    g2 = galsim.GammaDeviate(testseed, k=gammaK, theta=gammaTheta)
    g.reset(g2)
    testResult2 = (g(), g2(), g())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong gamma random number sequence generated using two gds')
    g.seed(testseed)
    testResult2 = (g2(), g(), g2())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong gamma random number sequence generated using two gds after seed')

    # Check that seeding with the time works (although we cannot check the output).
    # We're mostly just checking that this doesn't raise an exception.
    # The output could be anything.
    g.seed()
    testResult2 = (g(), g(), g())
    assert testResult2 != testResult
    g.reset()
    testResult3 = (g(), g(), g())
    assert testResult3 != testResult
    assert testResult3 != testResult2
    g.reset()
    testResult4 = (g(), g(), g())
    assert testResult4 != testResult
    assert testResult4 != testResult2
    assert testResult4 != testResult3
    g = galsim.GammaDeviate(k=gammaK, theta=gammaTheta)
    testResult5 = (g(), g(), g())
    assert testResult5 != testResult
    assert testResult5 != testResult2
    assert testResult5 != testResult3
    assert testResult5 != testResult4

    # Test generate
    g.seed(testseed)
    test_array = np.empty(3)
    g.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(gammaResult), precision,
            err_msg='Wrong gamma random number sequence from generate.')

    # Test generate with a float32 array
    g.seed(testseed)
    test_array = np.empty(3, dtype=np.float32)
    g.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(gammaResult), precisionF,
            err_msg='Wrong gamma random number sequence from generate.')

    # Check picklability
    do_pickle(g, lambda x: (x.serialize(), x.k, x.theta))
    do_pickle(g, lambda x: (x(), x(), x(), x()))
    do_pickle(g)
    assert 'GammaDeviate' in repr(g)
    assert 'GammaDeviate' in str(g)
    assert isinstance(eval(repr(g)), galsim.GammaDeviate)
    assert isinstance(eval(str(g)), galsim.GammaDeviate)

    # Check that we can construct a GammaDeviate from None, and that it depends on dev/random.
    g1 = galsim.GammaDeviate(None)
    g2 = galsim.GammaDeviate(None)
    assert g1 != g2, "Consecutive GammaDeviate(None) compared equal!"
    # We shouldn't be able to construct a GammaDeviate from anything but a BaseDeviate, int, str,
    # or None.
    assert_raises(TypeError, galsim.GammaDeviate, dict())
    assert_raises(TypeError, galsim.GammaDeviate, list())
    assert_raises(TypeError, galsim.GammaDeviate, set())


@timer
def test_chi2():
    """Test Chi^2 random number generator
    """
    c = galsim.Chi2Deviate(testseed, n=chi2N)
    c2 = c.duplicate()
    c3 = galsim.Chi2Deviate(c.serialize(), n=chi2N)
    testResult = (c(), c(), c())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(chi2Result), precision,
            err_msg='Wrong Chi^2 random number sequence generated')
    testResult = (c2(), c2(), c2())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(chi2Result), precision,
            err_msg='Wrong Chi^2 random number sequence generated with duplicate')
    testResult = (c3(), c3(), c3())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(chi2Result), precision,
            err_msg='Wrong Chi^2 random number sequence generated from serialize')

    # Check that the mean and variance come out right
    vals = [c() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = chi2N
    v = 2.*chi2N
    print('mean = ',mean,'  true mean = ',mu)
    print('var = ',var,'   true var = ',v)
    np.testing.assert_almost_equal(mean, mu, 1,
            err_msg='Wrong mean from Chi2Deviate')
    np.testing.assert_almost_equal(var, v, 0,
            err_msg='Wrong variance from Chi2Deviate')

    # Check seed, reset
    c.seed(testseed)
    testResult2 = (c(), c(), c())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Chi^2 random number sequence generated after seed')

    c.reset(testseed)
    testResult2 = (c(), c(), c())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Chi^2 random number sequence generated after reset(seed)')

    rng = galsim.BaseDeviate(testseed)
    c.reset(rng)
    testResult2 = (c(), c(), c())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Chi^2 random number sequence generated after reset(rng)')

    ud = galsim.UniformDeviate(testseed)
    c.reset(ud)
    testResult = (c(), c(), c())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Chi^2 random number sequence generated after reset(ud)')

    # Check that two connected Chi^2 deviates work correctly together.
    c2 = galsim.Chi2Deviate(testseed, n=chi2N)
    c.reset(c2)
    testResult2 = (c(), c2(), c())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Chi^2 random number sequence generated using two cds')
    c.seed(testseed)
    testResult2 = (c2(), c(), c2())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong Chi^2 random number sequence generated using two cds after seed')

    # Check that seeding with the time works (although we cannot check the output).
    # We're mostly just checking that this doesn't raise an exception.
    # The output could be anything.
    c.seed()
    testResult2 = (c(), c(), c())
    assert testResult2 != testResult
    c.reset()
    testResult3 = (c(), c(), c())
    assert testResult3 != testResult
    assert testResult3 != testResult2
    c.reset()
    testResult4 = (c(), c(), c())
    assert testResult4 != testResult
    assert testResult4 != testResult2
    assert testResult4 != testResult3
    c = galsim.Chi2Deviate(n=chi2N)
    testResult5 = (c(), c(), c())
    assert testResult5 != testResult
    assert testResult5 != testResult2
    assert testResult5 != testResult3
    assert testResult5 != testResult4

    # Test generate
    c.seed(testseed)
    test_array = np.empty(3)
    c.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(chi2Result), precision,
            err_msg='Wrong Chi^2 random number sequence from generate.')

    # Test generate with a float32 array
    c.seed(testseed)
    test_array = np.empty(3, dtype=np.float32)
    c.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(chi2Result), precisionF,
            err_msg='Wrong Chi^2 random number sequence from generate.')

    # Check picklability
    do_pickle(c, lambda x: (x.serialize(), x.n))
    do_pickle(c, lambda x: (x(), x(), x(), x()))
    do_pickle(c)
    assert 'Chi2Deviate' in repr(c)
    assert 'Chi2Deviate' in str(c)
    assert isinstance(eval(repr(c)), galsim.Chi2Deviate)
    assert isinstance(eval(str(c)), galsim.Chi2Deviate)

    # Check that we can construct a Chi2Deviate from None, and that it depends on dev/random.
    c1 = galsim.Chi2Deviate(None)
    c2 = galsim.Chi2Deviate(None)
    assert c1 != c2, "Consecutive Chi2Deviate(None) compared equal!"
    # We shouldn't be able to construct a Chi2Deviate from anything but a BaseDeviate, int, str,
    # or None.
    assert_raises(TypeError, galsim.Chi2Deviate, dict())
    assert_raises(TypeError, galsim.Chi2Deviate, list())
    assert_raises(TypeError, galsim.Chi2Deviate, set())


@timer
def test_distfunction():
    """Test distribution-defined random number generator with a function
    """
    # Make sure it requires an input function in order to work.
    assert_raises(TypeError, galsim.DistDeviate)
    # Make sure it does appropriate input sanity checks.
    assert_raises(TypeError, galsim.DistDeviate,
                  function='../examples/data/cosmo-fid.zmed1.00_smoothed.out',
                  x_min=1.)
    assert_raises(TypeError, galsim.DistDeviate, function=1.0)
    assert_raises(ValueError, galsim.DistDeviate, function='foo.dat')
    assert_raises(TypeError, galsim.DistDeviate, function = lambda x : x*x, interpolant='linear')
    assert_raises(TypeError, galsim.DistDeviate, function = lambda x : x*x)
    assert_raises(TypeError, galsim.DistDeviate, function = lambda x : x*x, x_min=1.)
    test_vals = range(10)
    assert_raises(TypeError, galsim.DistDeviate,
                  function=galsim.LookupTable(test_vals, test_vals),
                  x_min = 1.)
    foo = galsim.DistDeviate(10, galsim.LookupTable(test_vals, test_vals))
    assert_raises(ValueError, foo.val, -1.)
    assert_raises(ValueError, galsim.DistDeviate, function = lambda x : -1, x_min=dmin, x_max=dmax)
    assert_raises(ValueError, galsim.DistDeviate, function = lambda x : x**2-1, x_min=dmin, x_max=dmax)

    d = galsim.DistDeviate(testseed, function=dfunction, x_min=dmin, x_max=dmax)
    d2 = d.duplicate()
    d3 = galsim.DistDeviate(d.serialize(), function=dfunction, x_min=dmin, x_max=dmax)
    testResult = (d(), d(), d())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate random number sequence generated')
    testResult = (d2(), d2(), d2())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate random number sequence generated with duplicate')
    testResult = (d3(), d3(), d3())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate random number sequence generated from serialize')

    # Check val() method
    # pdf(x) = x^2
    # cdf(x) = (x/2)^3
    # val(y) = 2 y^(1/3)
    np.testing.assert_almost_equal(d.val(0), 0, 4)
    np.testing.assert_almost_equal(d.val(1), 2, 4)
    np.testing.assert_almost_equal(d.val(0.125), 1, 4)
    np.testing.assert_almost_equal(d.val(0.027), 0.6, 4)
    np.testing.assert_almost_equal(d.val(0.512), 1.6, 4)
    u = galsim.UniformDeviate(testseed)
    testResult = (d.val(u()), d.val(u()), d.val(u()))
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate sequence using d.val(u())')

    # Check that the mean and variance come out right
    vals = [d() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = 3./2.
    v = 3./20.
    print('mean = ',mean,'  true mean = ',mu)
    print('var = ',var,'   true var = ',v)
    np.testing.assert_almost_equal(mean, mu, 1,
            err_msg='Wrong mean from DistDeviate random numbers using function')
    np.testing.assert_almost_equal(var, v, 1,
            err_msg='Wrong variance from DistDeviate random numbers using function')

    # Check seed, reset
    d.seed(testseed)
    testResult2 = (d(), d(), d())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong DistDeviate random number sequence generated after seed')

    d.reset(testseed)
    testResult2 = (d(), d(), d())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong DistDeviate random number sequence generated after reset(seed)')

    rng = galsim.BaseDeviate(testseed)
    d.reset(rng)
    testResult2 = (d(), d(), d())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong DistDeviate random number sequence generated after reset(rng)')

    ud = galsim.UniformDeviate(testseed)
    d.reset(ud)
    testResult = (d(), d(), d())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong DistDeviate random number sequence generated after reset(ud)')

    # Check that two connected DistDeviate deviates work correctly together.
    d2 = galsim.DistDeviate(testseed, function=dfunction, x_min=dmin, x_max=dmax)
    d.reset(d2)
    testResult2 = (d(), d2(), d())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong DistDeviate random number sequence generated using two dds')
    d.seed(testseed)
    testResult2 = (d2(), d(), d2())
    np.testing.assert_array_equal(
            np.array(testResult), np.array(testResult2),
            err_msg='Wrong DistDeviate random number sequence generated using two dds after seed')

    # Check that seeding with the time works (although we cannot check the output).
    # We're mostly just checking that this doesn't raise an exception.
    # The output could be anything.
    d.seed()
    testResult2 = (d(), d(), d())
    assert testResult2 != testResult
    d.reset()
    testResult3 = (d(), d(), d())
    assert testResult3 != testResult
    assert testResult3 != testResult2
    d.reset()
    testResult4 = (d(), d(), d())
    assert testResult4 != testResult
    assert testResult4 != testResult2
    assert testResult4 != testResult3
    d = galsim.DistDeviate(function=dfunction, x_min=dmin, x_max=dmax)
    testResult5 = (d(), d(), d())
    assert testResult5 != testResult
    assert testResult5 != testResult2
    assert testResult5 != testResult3
    assert testResult5 != testResult4

    # Check with lambda function
    d = galsim.DistDeviate(testseed, function=lambda x: x*x, x_min=dmin, x_max=dmax)
    testResult = (d(), d(), d())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate random number sequence generated with lambda function')

    # Check auto-generated lambda function
    d = galsim.DistDeviate(testseed, function='x*x', x_min=dmin, x_max=dmax)
    testResult = (d(), d(), d())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate random number sequence generated with auto-lambda function')

    # Test generate
    d.seed(testseed)
    test_array = np.empty(3)
    d.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate random number sequence from generate.')

    # Test add_generate
    d.seed(testseed)
    d.add_generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, 2*np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate random number sequence from add_generate.')

    # Test generate with a float32 array
    d.seed(testseed)
    test_array = np.empty(3, dtype=np.float32)
    d.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(dFunctionResult), precisionF,
            err_msg='Wrong DistDeviate random number sequence from generate.')

    # Test add_generate with a float32 array
    d.seed(testseed)
    d.add_generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, 2*np.array(dFunctionResult), precisionF,
            err_msg='Wrong DistDeviate random number sequence from add_generate.')

    # Check picklability
    do_pickle(d, lambda x: (x(), x(), x(), x()))
    do_pickle(d)
    assert 'DistDeviate' in repr(d)
    assert 'DistDeviate' in str(d)
    assert isinstance(eval(repr(d)), galsim.DistDeviate)
    assert isinstance(eval(str(d)), galsim.DistDeviate)

    # Check that we can construct a DistDeviate from None, and that it depends on dev/random.
    c1 = galsim.DistDeviate(None, lambda x:1, 0, 1)
    c2 = galsim.DistDeviate(None, lambda x:1, 0, 1)
    assert c1 != c2, "Consecutive DistDeviate(None) compared equal!"
    # We shouldn't be able to construct a DistDeviate from anything but a BaseDeviate, int, str,
    # or None.
    assert_raises(TypeError, galsim.DistDeviate, dict(), lambda x:1, 0, 1)
    assert_raises(TypeError, galsim.DistDeviate, list(), lambda x:1, 0, 1)
    assert_raises(TypeError, galsim.DistDeviate, set(), lambda x:1, 0, 1)


@timer
def test_distLookupTable():
    """Test distribution-defined random number generator with a LookupTable
    """
    precision = 9
    # Note: 256 used to be the default, so this is a regression test
    # We check below that it works with the default npoints=None
    d = galsim.DistDeviate(testseed, function=dLookupTable, npoints=256)
    d2 = d.duplicate()
    d3 = galsim.DistDeviate(d.serialize(), function=dLookupTable, npoints=256)
    np.testing.assert_equal(
            d.x_min, dLookupTable.x_min,
            err_msg='DistDeviate and the LookupTable passed to it have different lower bounds')
    np.testing.assert_equal(
            d.x_max, dLookupTable.x_max,
            err_msg='DistDeviate and the LookupTable passed to it have different upper bounds')

    testResult = (d(), d(), d())
    print('testResult = ',testResult)
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence using LookupTable')
    testResult = (d2(), d2(), d2())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence using LookupTable with duplicate')
    testResult = (d3(), d3(), d3())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence using LookupTable from serialize')

    # Check that the mean and variance come out right
    vals = [d() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = 2.
    v = 7./3.
    print('mean = ',mean,'  true mean = ',mu)
    print('var = ',var,'   true var = ',v)
    np.testing.assert_almost_equal(mean, mu, 1,
            err_msg='Wrong mean from DistDeviate random numbers using LookupTable')
    np.testing.assert_almost_equal(var, v, 1,
            err_msg='Wrong variance from DistDeviate random numbers using LookupTable')

    # This should give the same values with only 5 points because of the particular nature
    # of these arrays.
    d = galsim.DistDeviate(testseed, function=dLookupTable, npoints=5)
    testResult = (d(), d(), d())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence for LookupTable with 5 points')

    # And it should also work if npoints is None
    d = galsim.DistDeviate(testseed, function=dLookupTable)
    testResult = (d(), d(), d())
    assert len(dLookupTable.x) == len(d._inverse_cdf.x)
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence for LookupTable with npoints=None')

    # Also read these values from a file
    d = galsim.DistDeviate(testseed, function=dLookupTableFile, interpolant='linear')
    testResult = (d(), d(), d())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence for LookupTable from file')

    d = galsim.DistDeviate(testseed, function=dLookupTableFile)
    testResult = (d(), d(), d())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence for LookupTable with default '
            'interpolant')

    # Test generate
    d.seed(testseed)
    test_array = np.empty(3)
    d.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence from generate.')

    # Test filling an image
    d.seed(testseed)
    testimage = galsim.ImageD(np.zeros((3, 1)))
    testimage.addNoise(galsim.DeviateNoise(d))
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence generated when applied to image.')

    # Test generate
    d.seed(testseed)
    test_array = np.empty(3)
    d.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence from generate.')

    # Test generate with a float32 array
    d.seed(testseed)
    test_array = np.empty(3, dtype=np.float32)
    d.generate(test_array)
    np.testing.assert_array_almost_equal(
            test_array, np.array(dLookupTableResult), precisionF,
            err_msg='Wrong DistDeviate random number sequence from generate.')

    # Test a case with nearly flat probabilities
    # x and p arrays with and without a small (epsilon) step
    dx_eps = np.arange(6)
    dp1_eps = np.zeros(dx_eps.shape)
    dp2_eps = np.zeros(dx_eps.shape)
    eps = np.finfo(dp1_eps[0].dtype).eps
    dp1_eps[0] = 0.5
    dp2_eps[0] = 0.5
    dp1_eps[-1] = 0.5
    dp2_eps[-2] = eps
    dp2_eps[-1] = 0.5-eps
    dLookupTableEps1 = galsim.LookupTable(x=dx_eps, f=dp1_eps, interpolant='linear')
    dLookupTableEps2 = galsim.LookupTable(x=dx_eps, f=dp2_eps, interpolant='linear')
    d1 = galsim.DistDeviate(testseed, function=dLookupTableEps1, npoints=len(dx_eps))
    d2 = galsim.DistDeviate(testseed, function=dLookupTableEps2, npoints=len(dx_eps))
    # If these were successfully created everything is probably fine, but check they create the same
    # internal LookupTable
    np.testing.assert_array_almost_equal(
            d1._inverse_cdf.getArgs(), d2._inverse_cdf.getArgs(), precision,
            err_msg='DistDeviate with near-flat probabilities incorrectly created '
                    'a monotonic version of the CDF')
    np.testing.assert_array_almost_equal(
            d1._inverse_cdf.getVals(), d2._inverse_cdf.getVals(), precision,
            err_msg='DistDeviate with near-flat probabilities incorrectly created '
                    'a monotonic version of the CDF')

    # And that they generate the same values
    ar1 = np.empty(100); d1.generate(ar1)
    ar2 = np.empty(100); d2.generate(ar2)
    np.testing.assert_array_almost_equal(ar1, ar2, precision,
            err_msg='Two DistDeviates with near-flat probabilities generated different values.')

    # Check picklability
    do_pickle(d, lambda x: (x(), x(), x(), x()))
    do_pickle(d)
    assert 'DistDeviate' in repr(d)
    assert 'DistDeviate' in str(d)
    assert isinstance(eval(repr(d)), galsim.DistDeviate)
    assert isinstance(eval(str(d)), galsim.DistDeviate)


@timer
def test_multiprocess():
    """Test that the same random numbers are generated in single-process and multi-process modes.
    """
    from multiprocessing import Process, Queue, current_process

    def generate_list(seed):
        """Given a particular seed value, generate a list of random numbers.
           Should be deterministic given the input seed value.
        """
        rng = galsim.UniformDeviate(seed)
        out = []
        for i in range(20):
            out.append(rng())
        return out

    def worker(input, output):
        """input is a queue with seed values
           output is a queue storing the results of the tasks along with the process name,
           and which args the result is for.
        """
        for args in iter(input.get, 'STOP'):
            result = generate_list(*args)
            output.put( (result, current_process().name, args) )

    # Use sequential numbers.
    # On inspection, can see that even the first value in each list is random with
    # respect to the other lists.  i.e. "nearby" inputs do not produce nearby outputs.
    # I don't know of an actual assert to do for this, but it is clearly true.
    seeds = [ 1532424 + i for i in range(16) ]

    nproc = 4  # Each process will do 4 lists (typically)

    # First make lists in the single process:
    ref_lists = dict()
    for seed in seeds:
        list = generate_list(seed)
        ref_lists[seed] = list

    # Now do this with multiprocessing
    # Put the seeds in a queue
    task_queue = Queue()
    for seed in seeds:
        task_queue.put( [seed] )

    # Run the tasks:
    done_queue = Queue()
    for k in range(nproc):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Check the results in the order they finished
    for i in range(len(seeds)):
        list, proc, args = done_queue.get()
        seed = args[0]
        np.testing.assert_array_equal(
                list, ref_lists[seed],
                err_msg="Random numbers are different when using multiprocessing")

    # Stop the processes:
    for k in range(nproc):
        task_queue.put('STOP')


@timer
def test_permute():
    """Simple tests of the permute() function."""
    # Make a fake list, and another list consisting of indices.
    my_list = [3.7, 4.1, 1.9, 11.1, 378.3, 100.0]
    import copy
    my_list_copy = copy.deepcopy(my_list)
    n_list = len(my_list)
    ind_list = list(range(n_list))

    # Permute both at the same time.
    galsim.random.permute(312, my_list, ind_list)

    # Make sure that everything is sensible
    for ind in range(n_list):
        assert my_list_copy[ind_list[ind]] == my_list[ind]

    # Repeat with same seed, should do same permutation.
    my_list = copy.deepcopy(my_list_copy)
    galsim.random.permute(312, my_list)
    for ind in range(n_list):
        assert my_list_copy[ind_list[ind]] == my_list[ind]

    # permute with no lists should raise TypeError
    with assert_raises(TypeError):
        galsim.random.permute(312)


@timer
def test_ne():
    """ Check that inequality works as expected for corner cases where the reprs of two
    unequal BaseDeviates may be the same due to truncation.
    """
    a = galsim.BaseDeviate(seed='1 2 3 4 5 6 7 8 9 10')
    b = galsim.BaseDeviate(seed='1 2 3 7 6 5 4 8 9 10')
    assert repr(a) == repr(b)
    assert a != b

    # Check DistDeviate separately, since it overrides __repr__ and __eq__
    d1 = galsim.DistDeviate(seed=a, function=galsim.LookupTable([1, 2, 3], [4, 5, 6]))
    d2 = galsim.DistDeviate(seed=b, function=galsim.LookupTable([1, 2, 3], [4, 5, 6]))
    assert repr(d1) == repr(d2)
    assert d1 != d2

if __name__ == "__main__":
    test_uniform()
    test_gaussian()
    test_binomial()
    test_poisson()
    test_poisson_highmean()
    test_poisson_zeromean()
    test_weibull()
    test_gamma()
    test_chi2()
    test_distfunction()
    test_distLookupTable()
    test_multiprocess()
    test_permute()
    test_ne()
