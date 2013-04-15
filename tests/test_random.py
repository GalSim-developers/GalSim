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
import numpy as np
import os
import sys

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

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
# This is currenly low enough to not dominate the time of the unit tests, but when changing
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

# gain, read noise to use for CCD noise tests
cGain = 3.
cReadNoise = 5.

# types to use in CCD tests
types = (np.int16, np.int32, np.float32, np.float64)
typestrings = ("S", "I", "F", "D")

# constant sky background level to use for CCD noise test image
sky = 50

# Tabulated results
cResultS = np.array([[44, 47], [50, 49]], dtype=np.int16)
cResultI = np.array([[44, 47], [50, 49]], dtype=np.int32)
cResultF = np.array([[44.45332718, 47.79725266], [50.67744064, 49.58272934]], dtype=np.float32)
cResultD = np.array([[44.453328440057618, 47.797254142519577], 
                     [50.677442088335162, 49.582730949808081]],dtype=np.float64)

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
dx=[0.0, 1.0, 1.00001, 2.99999, 3.0, 4.0]
dp=[0.1, 0.1, 0.0    , 0.0    , 0.1, 0.1]
dLookupTable=galsim.LookupTable(x=dx,f=dp,interpolant='linear')
# Tabulated results for DistDeviate LookupTable call
dLookupTableResult = (0.23721845680847731, 0.42913599265739233, 0.86176396813243539)
# File with the same values
dLookupTableFile = os.path.join('random_data','dLookupTable.dat')


def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_uniform():
    """Test uniform random number generator
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    testResult = (u(), u(), u())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(uResult), precision, 
            err_msg='Wrong uniform random number sequence generated')

    # Check that the mean and variance come out right
    vals = [u() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = 1./2.
    v = 1./12.
    print 'mean = ',mean,'  true mean = ',mu
    print 'var = ',var,'   true var = ',v
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
    
    # Test filling an image
    u.seed(testseed)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(galsim.DeviateNoise(u))
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(uResult), precision,
            err_msg='Wrong uniform random number sequence generated when applied to image.')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_gaussian():
    """Test Gaussian random number generator 
    """
    import time
    t1 = time.time()
    g = galsim.GaussianDeviate(testseed, mean=gMean, sigma=gSigma)
    testResult = (g(), g(), g())
    np.testing.assert_array_almost_equal(   
            np.array(testResult), np.array(gResult), precision,
            err_msg='Wrong Gaussian random number sequence generated')

    # Check that the mean and variance come out right
    vals = [g() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = gMean
    v = gSigma**2
    print 'mean = ',mean,'  true mean = ',mu
    print 'var = ',var,'   true var = ',v
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
    
    # Test filling an image
    g.seed(testseed)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(galsim.DeviateNoise(g))
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(gResult), precision,
            err_msg='Wrong Gaussian random number sequence generated when applied to image.')

    # GaussianNoise is equivalent, but no mean allowed.
    rng.seed(testseed)
    gn = galsim.GaussianNoise(rng, sigma=gSigma)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(gn)
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(gResult)-gMean, precision,
            err_msg="GaussianNoise applied to Images does not reproduce expected sequence")

    # Check changing GaussianNoise variance:
    np.testing.assert_almost_equal(
            gn.getVariance(), gSigma**2, precision, 
            err_msg="GaussianNoise getVariance returns wrong variance")
    np.testing.assert_almost_equal(
            gn.getSigma(), gSigma, precision, 
            err_msg="GaussianNoise getSigma returns wrong value")
    gn.scaleVariance(0.5)
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.5 * gSigma**2, precision, 
            err_msg="GaussianNoise scaleVariance results in wrong variance")
    np.testing.assert_almost_equal(
            gn.getSigma(), np.sqrt(0.5) * gSigma, precision, 
            err_msg="GaussianNoise scaleVariance results in wrong sigma")
    gn.setSigma(0.5)
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.25, precision, 
            err_msg="GaussianNoise setSigma results in wrong variance")
    np.testing.assert_almost_equal(
            gn.getSigma(), 0.5, precision, 
            err_msg="GaussianNoise setSigma results in wrong sigma")
    gn.setVariance(0.5)
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.5, precision, 
            err_msg="GaussianNoise setVariance results in wrong variance")
    np.testing.assert_almost_equal(
            gn.getSigma(), np.sqrt(0.5), precision, 
            err_msg="GaussianNoise setVariance results in wrong sigma")
    rng.seed(testseed)
    testimage.setZero()
    testimage.addNoise(gn)
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), (np.array(gResult)-gMean)*np.sqrt(0.5)/gSigma, precision,
            err_msg="GaussianNoise after setVariance does not reproduce expected sequence")
    gn2 = gn * 3
    np.testing.assert_almost_equal(
            gn2.getVariance(), 1.5, precision, 
            err_msg="GaussianNoise gn*3 results in wrong variance")
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.5, precision, 
            err_msg="GaussianNoise gn*3 results in wrong variance for original gn")
    gn2 = 5 * gn
    np.testing.assert_almost_equal(
            gn2.getVariance(), 2.5, precision, 
            err_msg="GaussianNoise 5*gn results in wrong variance")
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.5, precision, 
            err_msg="GaussianNoise 5*gn results in wrong variance for original gn")
    gn2 = gn/2
    np.testing.assert_almost_equal(
            gn2.getVariance(), 0.25, precision, 
            err_msg="GaussianNoise gn/2 results in wrong variance")
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.5, precision, 
            err_msg="GaussianNoise 5*gn results in wrong variance for original gn")
    gn *= 3
    np.testing.assert_almost_equal(
            gn.getVariance(), 1.5, precision, 
            err_msg="GaussianNoise gn*=3 results in wrong variance")
    gn /= 2
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.75, precision, 
            err_msg="GaussianNoise gn/=2 results in wrong variance")
 
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_binomial():
    """Test binomial random number generator
    """
    import time
    t1 = time.time()
    b = galsim.BinomialDeviate(testseed, N=bN, p=bp)
    testResult = (b(), b(), b())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(bResult), precision,
            err_msg='Wrong binomial random number sequence generated')
 
    # Check that the mean and variance come out right
    vals = [b() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = bN*bp
    v = bN*bp*(1.-bp)
    print 'mean = ',mean,'  true mean = ',mu
    print 'var = ',var,'   true var = ',v
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
    
    # Test filling an image
    b.seed(testseed)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(galsim.DeviateNoise(b))
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(bResult), precision,
            err_msg='Wrong binomial random number sequence generated when applied to image.')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_poisson():
    """Test Poisson random number generator
    """
    import time
    t1 = time.time()
    p = galsim.PoissonDeviate(testseed, mean=pMean)
    testResult = (p(), p(), p())
    np.testing.assert_array_almost_equal(   
            np.array(testResult), np.array(pResult), precision, 
            err_msg='Wrong Poisson random number sequence generated')

    # Check that the mean and variance come out right
    vals = [p() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = pMean
    v = pMean
    print 'mean = ',mean,'  true mean = ',mu
    print 'var = ',var,'   true var = ',v
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
    
    # Test filling an image
    p.seed(testseed)
    testimage = galsim.ImageViewI(np.zeros((3, 1), dtype=np.int32))
    testimage.addNoise(galsim.DeviateNoise(p))
    np.testing.assert_array_equal(
            testimage.array.flatten(), np.array(pResult),
            err_msg='Wrong poisson random number sequence generated when applied to image.')

    # The PoissonNoise version also subtracts off the mean value
    rng.seed(testseed)
    pn = galsim.PoissonNoise(rng, sky_level=pMean)
    testimage.fill(0)
    testimage.addNoise(pn)
    np.testing.assert_array_equal(
            testimage.array.flatten(), np.array(pResult)-pMean,
            err_msg='Wrong poisson random number sequence generated using PoissonNoise')

    # Check changing PoissonNoise variance:
    np.testing.assert_almost_equal(
            pn.getVariance(), pMean, precision, 
            err_msg="PoissonNoise getVariance returns wrong variance")
    np.testing.assert_almost_equal(
            pn.getSkyLevel(), pMean, precision, 
            err_msg="PoissonNoise getSkyLevel returns wrong value")
    pn.scaleVariance(0.5)
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.5 * pMean, precision, 
            err_msg="PoissonNoise scaleVariance results in wrong variance")
    np.testing.assert_almost_equal(
            pn.getSkyLevel(), 0.5 * pMean, precision, 
            err_msg="PoissonNoise scaleVariance results in wrong skyLevel")
    pn.setVariance(0.5)
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.5, precision, 
            err_msg="PoissonNoise setVariance results in wrong variance")
    np.testing.assert_almost_equal(
            pn.getSkyLevel(), 0.5, precision, 
            err_msg="PoissonNoise setVariance results in wrong skyLevel")
    pn2 = pn * 3
    np.testing.assert_almost_equal(
            pn2.getVariance(), 1.5, precision, 
            err_msg="PoissonNoise pn*3 results in wrong variance")
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.5, precision, 
            err_msg="PoissonNoise pn*3 results in wrong variance for original pn")
    pn2 = 5 * pn
    np.testing.assert_almost_equal(
            pn2.getVariance(), 2.5, precision, 
            err_msg="PoissonNoise 5*pn results in wrong variance")
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.5, precision, 
            err_msg="PoissonNoise 5*pn results in wrong variance for original pn")
    pn2 = pn/2
    np.testing.assert_almost_equal(
            pn2.getVariance(), 0.25, precision, 
            err_msg="PoissonNoise pn/2 results in wrong variance")
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.5, precision, 
            err_msg="PoissonNoise 5*pn results in wrong variance for original pn")
    pn *= 3
    np.testing.assert_almost_equal(
            pn.getVariance(), 1.5, precision, 
            err_msg="PoissonNoise pn*=3 results in wrong variance")
    pn /= 2
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.75, precision, 
            err_msg="PoissonNoise pn/=2 results in wrong variance")
 
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_weibull():
    """Test Weibull random number generator
    """
    import time
    t1 = time.time()
    w = galsim.WeibullDeviate(testseed, a=wA, b=wB)
    testResult = (w(), w(), w())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(wResult), precision, 
            err_msg='Wrong Weibull random number sequence generated')

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
    print 'mean = ',mean,'  true mean = ',mu
    print 'var = ',var,'   true var = ',v
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
    
    # Test filling an image
    w.seed(testseed)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(galsim.DeviateNoise(w))
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(wResult), precision,
            err_msg='Wrong weibull random number sequence generated when applied to image.')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_gamma():
    """Test Gamma random number generator
    """
    import time
    t1 = time.time()
    g = galsim.GammaDeviate(testseed, k=gammaK, theta=gammaTheta)
    testResult = (g(), g(), g())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(gammaResult), precision, 
            err_msg='Wrong Gamma random number sequence generated')

    # Check that the mean and variance come out right
    vals = [g() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = gammaK*gammaTheta
    v = gammaK*gammaTheta**2
    print 'mean = ',mean,'  true mean = ',mu
    print 'var = ',var,'   true var = ',v
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
    
    # Test filling an image
    g.seed(testseed)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(galsim.DeviateNoise(g))
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(gammaResult), precision,
            err_msg='Wrong gamma random number sequence generated when applied to image.')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_chi2():
    """Test Chi^2 random number generator
    """
    import time
    t1 = time.time()
    c = galsim.Chi2Deviate(testseed, n=chi2N)
    testResult = (c(), c(), c())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(chi2Result), precision, 
            err_msg='Wrong Chi^2 random number sequence generated')

    # Check that the mean and variance come out right
    vals = [c() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = chi2N
    v = 2.*chi2N
    print 'mean = ',mean,'  true mean = ',mu
    print 'var = ',var,'   true var = ',v
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
    
    # Test filling an image
    c.seed(testseed)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(galsim.DeviateNoise(c))
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(chi2Result), precision,
            err_msg='Wrong Chi^2 random number sequence generated when applied to image.')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_distfunction():
    """Test distribution-defined random number generator with a function
    """
    import time
    t1 = time.time()

    d = galsim.DistDeviate(testseed, function=dfunction, x_min=dmin, x_max=dmax)
    testResult = (d(), d(), d())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate random number sequence generated')

    # Check that the mean and variance come out right
    vals = [d() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = 3./2.
    v = 3./20.
    print 'mean = ',mean,'  true mean = ',mu
    print 'var = ',var,'   true var = ',v
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

    # Test filling an image
    d.seed(testseed)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(galsim.DeviateNoise(d))
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(dFunctionResult), precision,
            err_msg='Wrong DistDeviate random number sequence generated when applied to image.')
 
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_distLookupTable():
    """Test distribution-defined random number generator with a LookupTable
    """
    import time
    t1 = time.time()

    d = galsim.DistDeviate(testseed, function=dLookupTable)
    np.testing.assert_equal(
            d.x_min, dLookupTable.x_min,
            err_msg='DistDeviate and the LookupTable passed to it have different lower bounds')
    np.testing.assert_equal(
            d.x_max, dLookupTable.x_max,
            err_msg='DistDeviate and the LookupTable passed to it have different upper bounds')

    testResult = (d(), d(), d())
    np.testing.assert_array_almost_equal(
            np.array(testResult), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence using LookupTable')

    # Check that the mean and variance come out right
    vals = [d() for i in range(nvals)]
    mean = np.mean(vals)
    var = np.var(vals)
    mu = 2.
    v = 7./3.
    print 'mean = ',mean,'  true mean = ',mu
    print 'var = ',var,'   true var = ',v
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

    # Test filling an image
    d.seed(testseed)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(galsim.DeviateNoise(d))
    np.testing.assert_array_almost_equal(
            testimage.array.flatten(), np.array(dLookupTableResult), precision,
            err_msg='Wrong DistDeviate random number sequence generated when applied to image.')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_ccdnoise():
    """Test CCD Noise generator
    """
    import time
    t1 = time.time()
    for i in xrange(4):
        prec = eval("precision"+typestrings[i])
        cResult = eval("cResult"+typestrings[i])

        rng = galsim.BaseDeviate(testseed)
        ccdnoise = galsim.CCDNoise(rng, gain=cGain, read_noise=cReadNoise)
        testImage = galsim.ImageView[types[i]]((np.zeros((2, 2))+sky).astype(types[i]))
        ccdnoise.applyTo(testImage)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+".")

        # Check that reseeding the rng reseeds the internal deviate in CCDNoise
        rng.seed(testseed)
        testImage.fill(sky)
        ccdnoise.applyTo(testImage)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+
                " after seed")

        # Check using addNoise
        rng.seed(testseed)
        testImage.fill(sky)
        testImage.addNoise(ccdnoise)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+
                " using addNoise")

        # Now include sky_level in ccdnoise
        rng.seed(testseed)
        ccdnoise = galsim.CCDNoise(rng, sky_level=sky, gain=cGain, read_noise=cReadNoise)
        testImage.fill(0)
        ccdnoise.applyTo(testImage)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult-sky, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+
                " with sky_level included in noise")

        rng.seed(testseed)
        testImage.fill(0)
        testImage.addNoise(ccdnoise)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult-sky, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+
                " using addNoise with sky_level included in noise")

    # Check changing CCDNoise variance:
    var1 = (sky+cReadNoise**2)/cGain
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), var1, precision, 
            err_msg="CCDNoise getVariance returns wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.getSkyLevel(), sky, precision, 
            err_msg="CCDNoise getSkyLevel returns wrong value")
    np.testing.assert_almost_equal(
            ccdnoise.getGain(), cGain, precision, 
            err_msg="CCDNoise getGain returns wrong value")
    np.testing.assert_almost_equal(
            ccdnoise.getReadNoise(), cReadNoise, precision, 
            err_msg="CCDNoise getReadNoise returns wrong value")
    ccdnoise.scaleVariance(0.5)
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.5 * var1, precision, 
            err_msg="CCDNoise scaleVariance results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.getSkyLevel(), 0.5 * sky, precision, 
            err_msg="CCDNoise scaleVariance results in wrong SkyLevel")
    np.testing.assert_almost_equal(
            ccdnoise.getGain(), cGain, precision, 
            err_msg="CCDNoise scaleVariance results in wrong Gain")
    np.testing.assert_almost_equal(
            ccdnoise.getReadNoise(), np.sqrt(0.5) * cReadNoise, precision, 
            err_msg="CCDNoise scaleVariance results in wrong ReadNoise")
    ccdnoise.setVariance(0.5)
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.5, precision, 
            err_msg="CCDNoise setVariance results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.getSkyLevel(), sky * 0.5/var1, precision, 
            err_msg="CCDNoise setVariance results in wrong SkyLevel")
    np.testing.assert_almost_equal(
            ccdnoise.getGain(), cGain, precision, 
            err_msg="CCDNoise setVariance results in wrong Gain")
    np.testing.assert_almost_equal(
            ccdnoise.getReadNoise(), cReadNoise * np.sqrt(0.5/var1), precision, 
            err_msg="CCDNoise setVariance results in wrong ReadNoise")
    ccdnoise2 = ccdnoise * 3
    np.testing.assert_almost_equal(
            ccdnoise2.getVariance(), 1.5, precision, 
            err_msg="CCDNoise ccdnoise*3 results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.5, precision, 
            err_msg="CCDNoise ccdnoise*3 results in wrong variance for original ccdnoise")
    ccdnoise2 = 5 * ccdnoise
    np.testing.assert_almost_equal(
            ccdnoise2.getVariance(), 2.5, precision, 
            err_msg="CCDNoise 5*ccdnoise results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.5, precision, 
            err_msg="CCDNoise 5*ccdnoise results in wrong variance for original ccdnoise")
    ccdnoise2 = ccdnoise/2
    np.testing.assert_almost_equal(
            ccdnoise2.getVariance(), 0.25, precision, 
            err_msg="CCDNoise ccdnoise/2 results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.5, precision, 
            err_msg="CCDNoise 5*ccdnoise results in wrong variance for original ccdnoise")
    ccdnoise *= 3
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 1.5, precision, 
            err_msg="CCDNoise ccdnoise*=3 results in wrong variance")
    ccdnoise /= 2
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.75, precision, 
            err_msg="CCDNoise ccdnoise/=2 results in wrong variance")

    ccdnoise = galsim.CCDNoise(rng, gain=cGain, read_noise=cReadNoise)
    ccdnoise.setSkyLevel(3000)
    np.testing.assert_almost_equal(
            ccdnoise.getSkyLevel(), 3000, precision, 
            err_msg="CCDNoise setSkyLevel results in wrong SkyLevel")
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), (3000+cReadNoise**2)/cGain, precision, 
            err_msg="CCDNoise setSkyLevel results in wrong variance")
    ccdnoise.setGain(5.8)
    np.testing.assert_almost_equal(
            ccdnoise.getGain(), 5.8, precision, 
            err_msg="CCDNoise setGain results in wrong Gain")
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), (3000+cReadNoise**2)/5.8, precision, 
            err_msg="CCDNoise setGain results in wrong variance")
    ccdnoise.setReadNoise(10.2)
    np.testing.assert_almost_equal(
            ccdnoise.getReadNoise(), 10.2, precision, 
            err_msg="CCDNoise setReadNoise results in wrong ReadNoise")
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), (3000+10.2**2)/5.8, precision, 
            err_msg="CCDNoise setReadNoise results in wrong variance")
    
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_multiprocess():
    """Test that the same random numbers are generated in single-process and multi-process modes.
    """
    from multiprocessing import Process, Queue, current_process
    import time
    t1 = time.time()

    # Workaround for a bug in python 2.6.  We apply it always, just in case, but I think this
    # bit is unnecessary in python 2.7.  The bug is that sys.stdin can be double closed if
    # multiprocessing is used within something that already uses multiprocessing.
    # Specifically, if we are using nosetests with multiple processes.
    # See http://bugs.python.org/issue5313 for more info.
    sys.stdin.close()
    sys.stdin = open(os.devnull)

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
        #print 'list for %d was calculated to be %s'%(seed, list)

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
        #print 'list for %d was calculated by process %s to be %s'%(seed, proc, list)
        np.testing.assert_array_equal(
                list, ref_lists[seed], 
                err_msg="Random numbers are different when using multiprocessing")

    # Stop the processes:
    for k in range(nproc):
        task_queue.put('STOP')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_uniform()
    test_gaussian()
    test_binomial()
    test_poisson()
    test_weibull()
    test_gamma()
    test_chi2()
    test_distfunction()
    test_distLookupTable()
    test_ccdnoise()
    test_multiprocess()

