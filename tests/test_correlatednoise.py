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

smallim_size = 32 # size of image when we test correlation function properties using small inputs
largeim_size = 16 * smallim_size # ditto, but when we need a larger image

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
rtol_test = 0.01
# decimal place for absolute comparisons with large images (ditto)
decimal_test = 2

# number of positions to test in nonzero lag uncorrelated tests
npos_test = 25


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
            cf_zero, sigma**2, rtol=rtol_test, atol=0.,
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
            cf_test_value, 0., decimal=decimal_test,
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
        np.testing.assert_equal(
            cf_test1, cf_test2,
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric.")

def test_uncorrelated_noise_90degree_rotation():
    """Test that the CorrFunc rotation methods produces the same output as initializing with a 90
    degree-rotated input field.
    """
    ncf = galsim.correlatednoise.CorrFunc(uncorr_noise_large, dx=1.)
    ks = [1, 2, 3, 4]
    angles = [
        90. * galsim.degrees, 180. * galsim.degrees, 270. * galsim.degrees, 360. * galsim.degrees]
    # loop over rotation angles and check
    for k, angle in zip(ks, angles):
        noise_ref = galsim.ImageViewD(np.ascontiguousarray(np.rot90(uncorr_noise_large.array, k=k)))
        ncf_ref = galsim.correlatednoise.CorrFunc(noise_ref, dx=1.)
        # first we'll check the createRotated() method
        ncf_test1 = ncf.createRotated(angle)
        # then we'll check the createRotation() method
        ncf_test2 = ncf.copy()
        ncf_test2.applyRotation(angle) 
        # then check some positions
        for i in range(npos_test):
            rpos = glob_ud() * smallim_size
            tpos = 2. * np.pi * glob_ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            cf_ref = ncf_ref.xValue(pos)
            cf_test1 = ncf_test1.xValue(pos)
            cf_test2 = ncf_test2.xValue(pos) 
            # Then test these estimated value is good to within our chosen decimal place
            np.testing.assert_allclose(
                cf_ref, cf_test1, rtol=1.e-7, # these should be good at high, FFT-dependent acc.
                err_msg="Uncorrelated noise failed 90 degree createRotated() method test.")
            np.testing.assert_allclose(
                cf_ref, cf_test2, rtol=1.e-7,
                err_msg="Uncorrelated noise failed 90 degree applyRotation() method test.")

def test_xcorr_noise_basics():
    """Test the basic properties of a noise field, correlated in the x direction, generated using
    a simple shift-add prescription.
    """
    # use the xnoise defined above to make the x correlation function
    xncf = galsim.correlatednoise.CorrFunc(xnoise_large, dx=1.)
    # Then test the zero-lag value is good to 1% of the input variance; we expect this!
    cf_zero = xncf.xValue(galsim.PositionD(0., 0.))
    np.testing.assert_allclose(
        cf_zero, 1., rtol=rtol_test, atol=0.,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (1, 0) value is good to 1% of the input variance (0.5); we expect this!
    cf_10 = xncf.xValue(galsim.PositionD(1., 0.))
    np.testing.assert_allclose(
        cf_10, .5, rtol=rtol_test, atol=0.,
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
        cf_zero, 1., rtol=rtol_test, atol=0.,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (1, 0) value is good to 1% of the input variance (0.5); we expect this!
    cf_01 = yncf.xValue(galsim.PositionD(0., 1.))
    np.testing.assert_allclose(
        cf_01, .5, rtol=rtol_test, atol=0.,
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

def test_90degree_rotation(): # probably only need to do the x direction for this test if the 
                              # previous tests have passed OK
    """Test that the CorrFunc rotation methods produces the same output as initializing with a 90
    degree-rotated input field, this time with a noise field that's correlated in the x direction..
    """
    ncf = galsim.correlatednoise.CorrFunc(xnoise_large, dx=1.)
    ks = [1, 2, 3, 4]
    angles = [
        90. * galsim.degrees, 180. * galsim.degrees, 270. * galsim.degrees, 360. * galsim.degrees]
    # loop over rotation angles and check
    for k, angle in zip(ks, angles):
        noise_ref = galsim.ImageViewD(np.ascontiguousarray(np.rot90(xnoise_large.array, k=k)))
        ncf_ref = galsim.correlatednoise.CorrFunc(noise_ref, dx=1.)
        # first we'll check the createRotated() method
        ncf_test1 = ncf.createRotated(angle)
        # then we'll check the createRotation() method
        ncf_test2 = ncf.copy()
        ncf_test2.applyRotation(angle) 
        # then check some positions
        for i in range(npos_test):
            rpos = glob_ud() * smallim_size # look in the vicinity of the action near the centre
            tpos = 2. * np.pi * glob_ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            cf_ref = ncf_ref.xValue(pos)
            cf_test1 = ncf_test1.xValue(pos)
            cf_test2 = ncf_test2.xValue(pos) 
            np.testing.assert_allclose(
                cf_ref, cf_test1, rtol=1.e-7, # these should be good at high, FFT-dependent acc. 
                err_msg="Noise correlated in the x direction failed 90 degree createRotated() "+
                "method test.")
            np.testing.assert_allclose(
                cf_ref, cf_test2, rtol=1.e-7,
                err_msg="Noise correlated in the x direction failed 90 degree applyRotation() "+
                "method test.")

def test_arbitrary_rotation():
    """Check that rotated correlation function xValues() are correct for a correlation function with
    something in it.
    """
    ncf = galsim.correlatednoise.CorrFunc(ynoise_small, dx=1.) # use something not purely zero r>0
    for i in range(npos_test):
        rot_angle = 2. * np.pi * glob_ud()
        rpos = glob_ud() * smallim_size # look in the vicinity of the action near the centre
        tpos = 2. * np.pi * glob_ud()
        # get reference test position
        pos_ref = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        # then a rotated version
        pos_rot = galsim.PositionD(pos_ref.x * np.cos(rot_angle) + pos_ref.y * np.sin(rot_angle),
                                   -pos_ref.x * np.sin(rot_angle) + pos_ref.y * np.cos(rot_angle))
        # then create rotated ncfs for comparison
        ncf_rot1 = ncf.createRotated(rot_angle * galsim.radians)
        ncf_rot2 = ncf.copy()
        ncf_rot2.applyRotation(rot_angle * galsim.radians)
        np.testing.assert_allclose(
            ncf.xValue(pos_rot), ncf_rot1.xValue(pos_ref), 
            rtol=1.e-7, # this should be good at very high accuracy 
            err_msg="Noise correlated in the y direction failed createRotated() "+
            "method test for arbitrary rotations.")
        np.testing.assert_allclose(
            ncf.xValue(pos_rot), ncf_rot2.xValue(pos_ref), 
            rtol=1.e-7, # ditto
            err_msg="Noise correlated in the y direction failed applyRotation() "+
            "method test for arbitrary rotations.")

def test_scaling_magnification():
    """Test the scaling and magnification of correlation functions.
    """
    ncf = galsim.correlatednoise.CorrFunc(ynoise_small, dx=1.)
    scalings = [7.e-13, 424., 7.9e23]
    for scale in scalings:
       ncf_test1 = ncf.createMagnified(scale)
       ncf_test2 = ncf.copy() 
       ncf_test2.applyMagnification(scale)
       for i in range(npos_test):
           rpos = glob_ud() * 0.1 * smallim_size * scale # vicinity of the centre
           tpos = 2. * np.pi * glob_ud()
           pos_ref = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
           np.testing.assert_allclose(
               ncf_test1.xValue(pos_ref), ncf.xValue(pos_ref / scale), rtol=1.e-7,
               err_msg="Noise correlated in the y direction failed createMagnified() scaling test.")
           np.testing.assert_allclose(
               ncf_test2.xValue(pos_ref), ncf.xValue(pos_ref / scale), rtol=1.e-7,
               err_msg="Noise correlated in the y direction failed applyMagnification() scaling "+
               "test.")

def test_draw():
    """Test that the CorrFunc draw() method matches its internal, NumPy-derived estimate of the
    correlation function, and an independent calculation of the same thing.
    """
    from galsim import utilities
    # Build a noise correlation function using DFTs
    ft_array = np.fft.fft2(xnoise_small.array)
    # Calculate the power spectrum then correlation function
    ps_array = (ft_array * ft_array.conj()).real
    cf_array = (np.fft.ifft2(ps_array)).real / float(np.product(np.shape(ft_array)))
    cf_array = utilities.roll2d(cf_array, (cf_array.shape[0] / 2, cf_array.shape[1] / 2))
    # Then use the CorrFunc class for comparison
    ncf = galsim.correlatednoise.CorrFunc(xnoise_small, dx=1.)
    testim1 = galsim.ImageD(smallim_size, smallim_size)
    ncf.draw(testim1, dx=1.)
    # Commented stuff below makes some plots showing slightly weird edge behaviour of draw
    #
    import matplotlib.pyplot as plt
    plt.pcolor(testim1.array); plt.colorbar()
    plt.title('CF drawn by CorrFunc')
    plt.savefig('cf_drawn.pdf')
    plt.figure()
    plt.pcolor(ncf.original_cf_image.array); plt.colorbar()
    plt.title('Original CF image')
    plt.savefig('cf_original.pdf')
    plt.figure()
    plt.pcolor(cf_array); plt.colorbar()
    plt.title('Independently calculated original CF image')
    plt.figure()
    plt.pcolor(testim1.array - cf_array); plt.colorbar()
    plt.title('Difference between CorrFunc drawn CF and original')
    plt.savefig('cf_drawn_residual.pdf')
    plt.show()
    
    # then compare the arrays: note the need to take a border, this is illustrated by the commented
    # code above (something weird on left edge of the drawn image)
    np.testing.assert_array_almost_equal(
        testim1.array[1:-1, 1:-1], ncf.original_cf_image.array[1:-1, 1:-1], decimal=decimal_test, 
        err_msg="Drawn image does not match internal correlation function.")
    np.testing.assert_array_almost_equal(
        testim1.array[1:-1, 1:-1], cf_array[1:-1, 1:-1], decimal=decimal_test, 
        err_msg="Drawn image does not match independently calculated correlation function.")

def test_output_generation_basic():
#    """Test that noise generated by a CorrFunc matches the correlation function.
#    """
#    ncf = galsim.correlatednoise.CorrFunc(xnoise_small, dx=1.)
#    outimage = galsim.ImageD(xnoise_large.bounds)
#    ncf.applyNoiseTo(outimage, dx=1., dev=glob_ud)
#    ncf_test = galsim.correlatednoise.CorrFunc(outimage, dx=1.)
#    outim1 = galsim.ImageD(smallim_size, smallim_size)
#    outim2 = galsim.ImageD(smallim_size, smallim_size)
#    ncf.draw(outim1, dx=1.)
#    import matplotlib.pyplot as plt
#    plt.pcolor(outim1.array, vmin=-.1, vmax=1.1); plt.colorbar()
#    plt.figure()
#    ncf_test.draw(outim2, dx=1.)
#    plt.pcolor(outim2.array, vmin=-.1, vmax=1.1); plt.colorbar()
#    print np.std(outimage.array), np.std(xnoise_small.array)
#    plt.show()
    pass


def test_output_generation_rotated():
    pass

def test_output_generation_magnified():
    pass


