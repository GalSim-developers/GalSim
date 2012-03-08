import galsim.optics
import numpy as np

testshape = (128, 128)  # shape of image arrays for all tests

def test_roll2d_circularity():
    # Make heterogenous 2D array, integers first, test that a full roll gives the same as the inputs
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.optics.roll2d(int_image, (int_image.shape[0],
                                                                   int_image.shape[1])))
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.optics.roll2d(flt_image, (flt_image.shape[0],
                                                                   flt_image.shape[1])))

def test_roll2d_fwdbck():
    # Make heterogenous 2D array, integers first, test that a +1, -1 roll gives the same as initial
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.optics.roll2d(galsim.optics.roll2d(int_image, (+1, +1)),
                                                       (-1, -1)))
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.optics.roll2d(galsim.optics.roll2d(flt_image, (+1, +1)),
                                                       (-1, -1)))

                                      #def test_kxky_axes():
    
