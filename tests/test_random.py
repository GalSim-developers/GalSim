import galsim
import numpy as np

#
# Note: all tests below were generated using the python interface to the RNG.  Eventually need tests
# for comparison against the C++!
#

precision = 10
# decimal point at which agreement is required for all tests

testseed = 1000 # seed used for UniformDeviate for all tests
# Warning! If you change testseed, then all of the *Result variables below must change as well.

# the right answer for the first three uniform deviates produced from testseed
uResult = (0.6535895883571357, 0.20552172302268445, 0.11500694020651281)

# mean, sigma to use for Gaussian tests
gMean = 4.7
gSigma = 3.2
# the right answer for the first three Gaussian deviates produced from testseed
gResult = (3.464038348789816, 2.9155603332579436, 7.995607564277979)

# N, p to use for binomial tests
bN = 10
bp = 0.7
# the right answer for the first three binomial deviates produced from testseed
bResult = (6, 8, 9)

# mean to use for Poisson tests
pMean = 7
# the right answer for the first three Poisson deviates produced from testseed
pResult = (8, 5, 4)

def test_uniform_rand():
    """Test uniform random number generator for expected result given the above seed.
    """
    u = galsim.UniformDeviate(testseed)
    testResult = (u(), u(), u())
    np.testing.assert_almost_equal(testResult, uResult, precision, 
                                   err_msg='Wrong uniform random number sequence generated')

def test_uniform_rand_reset():
    """Testing ability to reset uniform random number generator and reproduce sequence.
    """
    u = galsim.UniformDeviate(testseed)
    testResult1 = (u(), u(), u())
    u = galsim.UniformDeviate(testseed)
    testResult2 = (u(), u(), u())
# note this one is still equal, not almost_equal, because we should be able to achieve complete
# equality for the same seed and the same exact system
    np.testing.assert_equal(testResult1, testResult2,
                            err_msg='Cannot reset generator with same seed to reproduce a sequence')

def test_gaussian_rand():
    """Test Gaussian random number generator for expected result given the above seed.
    """
    u = galsim.UniformDeviate(testseed)
    g = galsim.GaussianDeviate(u, mean=gMean, sigma=gSigma)
    testResult = (g(), g(), g())
    np.testing.assert_almost_equal(testResult, gResult, precision,
                                   err_msg='Wrong Gaussian random number sequence generated')

def test_binomial_rand():
    """Test binomial random number generator for expected result given the above seed.
    """
    u = galsim.UniformDeviate(testseed)
    b = galsim.BinomialDeviate(u, N=bN, p=bp)
    testResult = (b(), b(), b())
    np.testing.assert_almost_equal(testResult, bResult, precision,
                                   err_msg='Wrong binomial random number sequence generated')

def test_poisson_rand():
    """Test Poisson random number generator for expected result given the above seed.
    """
    u = galsim.UniformDeviate(testseed)
    p = galsim.PoissonDeviate(u, mean=pMean)
    testResult = (p(), p(), p())
    np.testing.assert_almost_equal(testResult, pResult, precision, 
                                   err_msg='Wrong Poisson random number sequence generated')
