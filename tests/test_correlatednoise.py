import numpy as np

try:
    import galsim
except ImportError:
    import os
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# Use a deterministic random number generator so we don't fail tests because of rare flukes
# in the random numbers.
glob_ud = galsim.UniformDeviate(12345)

smallim_size = 10 # size of image when we test correlation function properties using small inputs
largeim_size = 30 * smallim_size # ditto, but when we need a larger image

# then make a small image of uncorrelated, unit variance noise for later tests
gd = galsim.GaussianDeviate(glob_ud, mean=0., sigma=1.)
uncorr_noise_small = galsim.ImageD(smallim_size, smallim_size)
uncorr_noise_small.addNoise(gd)

# then make a large image of uncorrelated, unit variance noise, also for later tests
uncorr_noise_large = galsim.ImageD(largeim_size, largeim_size)
uncorr_noise_large.addNoise(gd)

# make some x-correlated noise using shift and add
xnoise_large = galsim.ImageViewD(
    uncorr_noise_large.array + np.roll(uncorr_noise_large.array, 1, axis=1)) # note NumPy thus [y,x]
xnoise_large *= (np.sqrt(2.) / 2.) # make unit variance
xnoise_small = galsim.ImageViewD(
    uncorr_noise_small.array + np.roll(uncorr_noise_small.array, 1, axis=1)) # note NumPy thus [y,x]
xnoise_small *= (np.sqrt(2.) / 2.) # make unit variance
 
# make some y-correlated noise using shift and add
ynoise_large = galsim.ImageViewD(
    uncorr_noise_large.array + np.roll(uncorr_noise_large.array, 1, axis=0)) # note NumPy thus [y,x]
ynoise_large *= (np.sqrt(2.) / 2.) # make unit variance
ynoise_small = galsim.ImageViewD(
    uncorr_noise_small.array + np.roll(uncorr_noise_small.array, 1, axis=0)) # note NumPy thus [y,x]
ynoise_small *= (np.sqrt(2.) / 2.) # make unit variance

# relative tolerance for large image comparisons (still needs to include some stochasticity margin)
rtol_large = 0.01
# decimal place for absolute comparisons with large images (ditto)
decimal_large = 2

# number of positions to test in nonzero lag uncorrelated tests
npos_test = 33


def test_uncorrelated_noise_zero_lag():
    """Test that the zero lag correlation of an input uncorrelated noise field matches its variance.
    """
    sigmas = [3.e-9, 49., 1.11e11, 3.4e30]  # some wide ranging sigma values for the noise field
    # loop through the sigmas
    for sigma in sigmas:
        noise_test = uncorr_noise_large * sigma
        ncf = galsim.correlatednoise.CorrFunc(noise_test, dx=1.)
        cf_zero = ncf.xValue(galsim.PositionD(0., 0.))
        # Then test this estimated value is good to 1% of the input variance; we expect this!
        np.testing.assert_allclose(
            cf_zero, sigma**2, rtol=rtol_large, atol=0.,
            err_msg="Zero distance noise correlation value does not match input noise variance.")

def test_uncorrelated_noise_nonzero_lag():
    """Test that the non-zero lag correlation of an input uncorrelated noise field is zero at some
    randomly chosen positions.
    """
    ncf = galsim.correlatednoise.CorrFunc(uncorr_noise_large, dx=1.)
    # set up some random positions (within and outside) the bounds of the table inside the corrfunc
    # then test
    for i in range(npos_test):
        # generate the test position at least one pixel away from the origin
        rpos = 1. + glob_ud() * (largeim_size - 1.) # this can go outside table bounds
        tpos = 2. * np.pi * glob_ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test_value = ncf.xValue(pos)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_almost_equal(
            cf_test_value, 0., decimal=decimal_large,
            err_msg="Non-zero distance noise correlation value not sufficiently close to target "+
            "value of zero.")

def test_uncorrelated_noise_symmetry():
    """Test that the non-zero lag correlation of an input uncorrelated noise field has two-fold
    rotational symmetry.
    """
    ncf = galsim.correlatednoise.CorrFunc(uncorr_noise_small, dx=1.) # the small image is fine here
    # set up some random positions (within and outside) the bounds of the table inside the corrfunc
    # then test
    for i in range(npos_test):
        rpos = glob_ud() * smallim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * glob_ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = ncf.xValue(pos)
        cf_test2 = ncf.xValue(-pos)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_allclose(
            cf_test1, cf_test2, rtol=1.e-15, # this should be good to machine precision
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric.")

def test_xcorr_noise_basics():
    """Test the basic properties of a noise field, correlated in the x direction, generated using
    a simple shift-add prescription.
    """
    # use the xnoise defined above to make the x correlation function
    xncf = galsim.correlatednoise.CorrFunc(xnoise_large, dx=1.)
    # Then test the zero-lag value is good to 1% of the input variance; we expect this!
    cf_zero = xncf.xValue(galsim.PositionD(0., 0.))
    np.testing.assert_allclose(
        cf_zero, 1., rtol=rtol_large, atol=0.,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (1, 0) value is good to 1% of the input variance (0.5); we expect this!
    cf_10 = xncf.xValue(galsim.PositionD(1., 0.))
    np.testing.assert_allclose(
        cf_10, .5, rtol=rtol_large, atol=0.,
        err_msg="Noise correlation value at (1, 0) does not match input covariance.")

def test_ycorr_noise_basics():
    """Test the basic properties of a noise field, correlated in the y direction, generated using
    a simple shift-add prescription.
    """
    # use the ynoise defined above to make the y correlation function
    yncf = galsim.correlatednoise.CorrFunc(ynoise_large, dx=1.)
    # Then test the zero-lag value is good to 1% of the input variance; we expect this!
    cf_zero = yncf.xValue(galsim.PositionD(0., 0.))
    np.testing.assert_allclose(
        cf_zero, 1., rtol=rtol_large, atol=0.,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (1, 0) value is good to 1% of the input variance (0.5); we expect this!
    cf_01 = yncf.xValue(galsim.PositionD(0., 1.))
    np.testing.assert_allclose(
        cf_01, .5, rtol=rtol_large, atol=0.,
        err_msg="Noise correlation value at (0, 1) does not match input covariance.")

def test_xcorr_noise_symmetry():
    """Test that the non-zero lag correlation of an input x correlated noise field has two-fold
    rotational symmetry.
    """
    ncf = galsim.correlatednoise.CorrFunc(xnoise_large, dx=1.) # the small image is fine here
    # set up some random positions (within and outside) the bounds of the table inside the corrfunc
    # then test
    for i in range(npos_test):
        rpos = glob_ud() *largeim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * glob_ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = ncf.xValue(pos)
        cf_test2 = ncf.xValue(-pos)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_allclose(
            cf_test1, cf_test2, rtol=1.e-15, # this should be good to machine precision
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric for x correlated noise field.")

def test_ycorr_noise_symmetry():
    """Test that the non-zero lag correlation of an input y correlated noise field has two-fold
    rotational symmetry.
    """
    ncf = galsim.correlatednoise.CorrFunc(ynoise_large, dx=1.) # the small image is fine here
    # set up some random positions (within and outside) the bounds of the table inside the corrfunc
    # then test
    for i in range(npos_test):
        rpos = glob_ud() *largeim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * glob_ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = ncf.xValue(pos)
        cf_test2 = ncf.xValue(-pos)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_allclose(
            cf_test1, cf_test2, rtol=1.e-15, # this should be good to machine precision
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric for y correlated noise field.")


