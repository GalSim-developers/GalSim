import galsim.optics
import numpy as np

testshape = (128, 128)  # shape of image arrays for all tests
decimal = 10 # Last decimal place used for checking equality of float arrays, see
             # np.testing.assert_array_almost_equal()

def test_roll2d_circularity():
    """Test both integer and float arrays are unchanged by full circular roll.
    """
    # Make heterogenous 2D array, integers first, test that a full roll gives the same as the inputs
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.optics.roll2d(int_image, int_image.shape),
                                  err_msg='galsim.optics.roll2D failed int array circularity test')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.optics.roll2d(flt_image, flt_image.shape),
                                  err_msg='galsim.optics.roll2D failed flt array circularity test')

def test_roll2d_fwdbck():
    """Test both integer and float arrays are unchanged by unit forward and backward roll.
    """
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

def test_roll2d_join():
    """Test both integer and float arrays are equivalent if rolling +1/-1 or -/+(shape[i/j] - 1).
    """
    # Make heterogenous 2D array, integers first
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(galsim.optics.roll2d(int_image, (+1, -1)),
                                  galsim.optics.roll2d(int_image, (-(int_image.shape[0] - 1),
                                                                   +(int_image.shape[1] - 1))),
                                  err_msg='galsim.optics.roll2D failed int array +/- join test')
    np.testing.assert_array_equal(galsim.optics.roll2d(int_image, (-1, +1)),
                                  galsim.optics.roll2d(int_image, (+(int_image.shape[0] - 1),
                                                                   -(int_image.shape[1] - 1))),
                                  err_msg='galsim.optics.roll2D failed int array -/+ join test')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(galsim.optics.roll2d(flt_image, (+1, -1)),
                                  galsim.optics.roll2d(flt_image, (-(flt_image.shape[0] - 1),
                                                                   +(flt_image.shape[1] - 1))),
                                  err_msg='galsim.optics.roll2D failed flt array +/- join test')
    np.testing.assert_array_equal(galsim.optics.roll2d(flt_image, (-1, +1)),
                                  galsim.optics.roll2d(flt_image, (+(flt_image.shape[0] - 1),
                                                                   -(flt_image.shape[1] - 1))),
                                  err_msg='galsim.optics.roll2D failed flt array -/+ join test')

def test_kxky():
    """Test that the basic properties of kx and ky are right.
    """
    kx, ky = galsim.optics.kxky((4, 4))
    kxref = np.array([0., 0.25, -0.5, -0.25]) * 2. * np.pi
    kyref = np.array([0., 0.25, -0.5, -0.25]) * 2. * np.pi
    for i in xrange(4):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in xrange(4):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))

def test_kxky_plusone():
    """Test that the basic properties of kx and ky are right...
    But increment testshape used in test_kxky by one to test both odd and even cases.
    """
    kx, ky = galsim.optics.kxky((4 + 1, 4 + 1))
    kxref = np.array([0., 0.2, 0.4, -0.4, -0.2]) * 2. * np.pi
    kyref = np.array([0., 0.2, 0.4, -0.4, -0.2]) * 2. * np.pi
    for i in xrange(4 + 1):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in xrange(4 + 1):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))

def test_check_all_contiguous():
    """Test all galsim.optics outputs are C-contiguous as required by the galsim.Image class.
    """
    #Check that roll2d outputs contiguous arrays whatever the input
    imcstyle = np.random.random(size=testshape)
    rolltest = galsim.optics.roll2d(imcstyle, (+1, -1))
    assert rolltest.flags.c_contiguous
    imfstyle = np.random.random(size=testshape).T
    rolltest = galsim.optics.roll2d(imfstyle, (+1, -1))
    assert rolltest.flags.c_contiguous
    # Check kx, ky
    kx, ky = galsim.optics.kxky(testshape)
    assert kx.flags.c_contiguous
    assert ky.flags.c_contiguous
    # Check basic outputs from wavefront, psf and mtf (array contents won't matter, so we'll use
    # a pure circular pupil)
    assert galsim.optics.wavefront(shape=testshape).flags.c_contiguous
    assert galsim.optics.psf(shape=testshape).flags.c_contiguous
    assert galsim.optics.otf(shape=testshape).flags.c_contiguous
    assert galsim.optics.mtf(shape=testshape).flags.c_contiguous
    assert galsim.optics.ptf(shape=testshape).flags.c_contiguous
