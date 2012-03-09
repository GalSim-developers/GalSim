import galsim.optics
import numpy as np

testshape = (128, 128)  # shape of image arrays for all tests
decimal = 10 # Last decimal place used for checking equality of float arrays, see
             # np.testing.assert_array_almost_equal()

def test_roll2d_circularity():
    # Make heterogenous 2D array, integers first, test that a full roll gives the same as the inputs
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.optics.roll2d(int_image, (int_image.shape[0],
                                                                   int_image.shape[1])),
                                  err_msg='galsim.optics.roll2D failed int array circularity test')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.optics.roll2d(flt_image, (flt_image.shape[0],
                                                                   flt_image.shape[1])),
                                  err_msg='galsim.optics.roll2D failed flt array circularity test')

def test_roll2d_fwdbck():
    # Make heterogenous 2D array, integers first, test that a +1, -1 roll gives the same as initial
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.optics.roll2d(galsim.optics.roll2d(int_image, (+1, +1)),
                                                       (-1, -1)),
                                  err_msg='galsim.optics.roll2D failed int array fwd/back test')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.optics.roll2d(galsim.optics.roll2d(flt_image, (+1, +1)),
                                                       (-1, -1)),
                                  err_msg='galsim.optics.roll2D failed flt array fwd/back test')

def test_kxky():
    # Test that the basic properties of kx and ky are right, by reference to np.fft.fftfreq
    kx, ky = galsim.optics.kxky(testshape)
    kxref = np.fft.fftfreq(testshape[1]) * np.pi
    kyref = np.fft.fftfreq(testshape[0]) * np.pi
    for i in xrange(testshape[0]):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in xrange(testshape[1]):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))

def test_kxky_plusone():
    # Test that the basic properties of kx and ky are right, by reference to np.fft.fftfreq,
    # increments testshape used in test_kxky by one to test both odd and even cases
    kx, ky = galsim.optics.kxky((testshape[0] + 1, testshape[1] + 1))
    kxref = np.fft.fftfreq(testshape[1] + 1) * np.pi
    kyref = np.fft.fftfreq(testshape[0] + 1) * np.pi
    for i in xrange(testshape[0] + 1):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in xrange(testshape[1] + 1):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))
