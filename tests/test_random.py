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

# gain, read noise to use for CCD noise tests
cGain = 3.
cReadNoise = 5.

# types to use in CCD tests
types = (np.int16, np.int32, np.float32, np.float64)
typestrings = ("S", "I", "F", "D")

# constant sky background level to use for CCD noise test image
sky = 50.

# Tabulated results
cResultS = np.array([[55, 47], [44, 54]], dtype=np.int16)
cResultI = np.array([[55, 47], [44, 54]], dtype=np.int32)
cResultF = np.array([[55.894428, 47.926445], [44.766380, 54.509399]], dtype=np.float32)
cResultD = np.array([[55.894425874146336, 47.926443828684981],
                     [44.766381648303245, 54.509402357022211]], dtype=np.float64)

# a & b to use for Weibull tests
wA = 4.
wB = 9.
# Tabulated results for Weibull
wResult = (9.13234684792985, 6.233146513996428, 5.320941931556709)

# alpha & beta to use for Gamma tests
gammaAlpha = 1.5
gammaBeta = 4.5
# Tabulated results for Weibull
gammaResult = (7.044528184548506, 1.2478440729556932, 7.261520614904038)

# n to use for Chi2 tests
chi2N = 30
# Tabulated results for Chi2
chi2Result = (32.070036783568476, 26.303957298429747, 23.438338438325008)

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_uniform_rand():
    """Test uniform random number generator for expected result given the above seed.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    testResult = (u(), u(), u())
    np.testing.assert_array_almost_equal(np.array(testResult), np.array(uResult), precision, 
                                         err_msg='Wrong uniform random number sequence generated')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_uniform_rand_reset():
    """Testing ability to reset uniform random number generator and reproduce sequence.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    testResult1 = (u(), u(), u())
    u = galsim.UniformDeviate(testseed)
    testResult2 = (u(), u(), u())
# note this one is still equal, not almost_equal, because we should be able to achieve complete
# equality for the same seed and the same exact system
    np.testing.assert_array_equal(np.array(testResult1), np.array(testResult2),
                               err_msg='Cannot reset generator (same seed) to reproduce sequence')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_uniform_image():
    """Testing ability to apply uniform random numbers to images using their addNoise method, 
    and reproduce sequence.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(u)
    np.testing.assert_array_almost_equal(testimage.array.flatten(), np.array(uResult),
                               err_msg="UniformDeviate generator applied to Images does not "
                                       "reproduce expected sequence")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_gaussian_rand():
    """Test Gaussian random number generator for expected result given the above seed.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    g = galsim.GaussianDeviate(u, mean=gMean, sigma=gSigma)
    testResult = (g(), g(), g())
    np.testing.assert_array_almost_equal(np.array(testResult), np.array(gResult), precision,
                                         err_msg='Wrong Gaussian random number sequence generated')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_gaussian_image():
    """Testing ability to apply Gaussian random numbers to images using their addNoise method, 
    and reproduce sequence.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    g = galsim.GaussianDeviate(u, mean=gMean, sigma=gSigma)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(g)
    np.testing.assert_array_almost_equal(testimage.array.flatten(), np.array(gResult), precision,
                               err_msg="GaussianDeviate generator applied to Images does not "
                                       "reproduce expected sequence")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_binomial_rand():
    """Test binomial random number generator for expected result given the above seed.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    b = galsim.BinomialDeviate(u, N=bN, p=bp)
    testResult = (b(), b(), b())
    np.testing.assert_array_almost_equal(np.array(testResult), np.array(bResult), precision,
                                         err_msg='Wrong binomial random number sequence generated')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_binomial_image():
    """Testing ability to apply Binomial random numbers to images using their addNoise method, 
    and reproduce sequence.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    b = galsim.BinomialDeviate(u, N=bN, p=bp)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(b)
    np.testing.assert_array_almost_equal(testimage.array.flatten(), np.array(bResult), precision,
                               err_msg="BinomialDeviate generator applied to Images does not "
                                       "reproduce expected sequence")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_poisson_rand():
    """Test Poisson random number generator for expected result given the above seed.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    p = galsim.PoissonDeviate(u, mean=pMean)
    testResult = (p(), p(), p())
    np.testing.assert_array_almost_equal(np.array(testResult), np.array(pResult), precision, 
                                         err_msg='Wrong Poisson random number sequence generated')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_poisson_image():
    """Testing ability to apply Poisson random numbers to images using their addNoise method, 
    and reproduce sequence.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    p = galsim.PoissonDeviate(u, mean=pMean)
    testimage = galsim.ImageViewI(np.zeros((3, 1), dtype=np.int32))
    testimage.addNoise(p)
    np.testing.assert_array_almost_equal(testimage.array.flatten(), np.array(pResult)-pMean,
                               err_msg="PoissonDeviate generator applied to Images does not "
                                       "reproduce expected sequence")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_ccdnoise_rand():
    """Test CCD Noise generator on a 2x2 image against the expected result given the above seed.
    """
    import time
    t1 = time.time()
    for i in xrange(4):
        u = galsim.UniformDeviate(testseed)
        ccdnoise = galsim.CCDNoise(u, gain=cGain, read_noise=cReadNoise)
        testImage = galsim.ImageView[types[i]]((np.zeros((2, 2)) + sky).astype(types[i]))
        ccdnoise.applyTo(testImage)
        np.testing.assert_array_almost_equal(testImage.array, eval("cResult"+typestrings[i]),
                                             eval("precision"+typestrings[i]),
                                             err_msg="Wrong CCD noise random sequence generated "+
                                                     "for Image"+typestrings[i]+" images.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_ccdnoise_image():
    """Test CCD Noise generator on a 2x2 image against the expected result given the above seed,
    and using the image method version of the CCD Noise generator.
    """
    import time
    t1 = time.time()
    for i in xrange(4):
        u = galsim.UniformDeviate(testseed)
        ccdnoise = galsim.CCDNoise(u, gain=cGain, read_noise=cReadNoise)
        testImage = galsim.ImageView[types[i]]((np.zeros((2, 2)) + sky).astype(types[i]))
        testImage.addNoise(ccdnoise)
        np.testing.assert_array_almost_equal(testImage.array, eval("cResult"+typestrings[i]),
                                             eval("precision"+typestrings[i]),
                                             err_msg="Wrong CCD noise random sequence generated "+
                                                     "for Image"+typestrings[i]+" images.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_weibull_rand():
    """Test Weibull random number generator for expected result given the above seed.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    w = galsim.WeibullDeviate(u, a=wA, b=wB)
    testResult = (w(), w(), w())
    np.testing.assert_array_almost_equal(np.array(testResult), np.array(wResult), precision, 
                                         err_msg='Wrong Weibull random number sequence generated')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_weibull_image():
    """Testing ability to apply Weibull random numbers to images using their addNoise method, 
    and reproduce sequence.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    w = galsim.WeibullDeviate(u, a=wA, b=wB)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(w)
    np.testing.assert_array_almost_equal(testimage.array.flatten(), np.array(wResult), precision, 
                                         err_msg='Wrong Weibull random number sequence generated '
                                                +'using image method.')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_gamma_rand():
    """Test Gamma random number generator for expected result given the above seed.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    gam = galsim.GammaDeviate(u, alpha=gammaAlpha, beta=gammaBeta)
    testResult = (gam(), gam(), gam())
    np.testing.assert_array_almost_equal(np.array(testResult), np.array(gammaResult), precision, 
                                         err_msg='Wrong Gamma random number sequence generated')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_gamma_image():
    """Testing ability to apply Gamma random numbers to images using their addNoise method, 
    and reproduce sequence.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    gam = galsim.GammaDeviate(u, alpha=gammaAlpha, beta=gammaBeta)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(gam)
    np.testing.assert_array_almost_equal(testimage.array.flatten(), np.array(gammaResult),
                                         precision,
                                         err_msg='Wrong Gamma random number sequence generated '
                                                +'using image method.')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_chi2_rand():
    """Test Chi^2 random number generator for expected result given the above seed.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    chi2 = galsim.Chi2Deviate(u, n=chi2N)
    testResult = (chi2(), chi2(), chi2())
    np.testing.assert_array_almost_equal(np.array(testResult), np.array(chi2Result), precision, 
                                         err_msg='Wrong Chi^2 random number sequence generated')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_chi2_image():
    """Testing ability to apply Chi^2 random numbers to images using their addNoise method, 
    and reproduce sequence.
    """
    import time
    t1 = time.time()
    u = galsim.UniformDeviate(testseed)
    chi2 = galsim.Chi2Deviate(u, n =chi2N)
    testimage = galsim.ImageViewD(np.zeros((3, 1)))
    testimage.addNoise(chi2)
    np.testing.assert_array_almost_equal(testimage.array.flatten(), np.array(chi2Result),
                                         precision,
                                         err_msg='Wrong Chi^2 random number sequence generated '
                                                +'using image method.')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_multiprocess():
    """Test that the same random numbers are generated in single-process and multi-process modes.
    """
    from multiprocessing import Process, Queue, current_process
    import time
    t1 = time.time()

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
        print 'list for %d was calculated by process %s to be %s'%(seed, proc, list)
        np.testing.assert_array_equal(
                list, ref_lists[seed], 
                err_msg="Random numbers are different when using multiprocessing")

    # Stop the processes:
    for k in range(nproc):
        task_queue.put('STOP')

if __name__ == "__main__":
    test_uniform_rand()
    test_uniform_rand_reset()
    test_uniform_image()
    test_gaussian_rand()
    test_gaussian_image()
    test_binomial_rand()
    test_binomial_image()
    test_poisson_rand()
    test_poisson_image()
    test_ccdnoise_rand()
    test_ccdnoise_image()
    test_weibull_rand()
    test_weibull_image()
    test_gamma_rand()
    test_gamma_image()
    test_chi2_rand()
    test_chi2_image()
    test_multiprocess()

