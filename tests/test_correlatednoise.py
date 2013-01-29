import time
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

smallim_size = 16 # size of image when we test correlation function properties using small inputs
smallim_size_odd = 17 # odd-sized version of the above for odd/even relevant tests (e.g. draw)
largeim_size = 12 * smallim_size # ditto, but when we need a larger image
xlargeim_size =long(np.ceil(1.41421356 * largeim_size)) # sometimes, for precision tests, we 
                                                        # need a very large image that will 
                                                        # fit a large image within it, even if 
                                                        # rotated

# then make a small image of uncorrelated, unit variance noise for later tests
gd = galsim.GaussianDeviate(glob_ud, mean=0., sigma=1.)
uncorr_noise_small = galsim.ImageD(smallim_size, smallim_size)
uncorr_noise_small.addNoise(gd)

# then make a large image of uncorrelated, unit variance noise, also for later tests
uncorr_noise_large = galsim.ImageD(largeim_size, largeim_size)
uncorr_noise_large.addNoise(gd)
# make an extra large image here for rotation generation tests
uncorr_noise_xlarge = galsim.ImageD(xlargeim_size, xlargeim_size)
uncorr_noise_xlarge.addNoise(gd)

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
# make an extra large image here for rotation generation tests
ynoise_xlarge = galsim.ImageViewD(
    uncorr_noise_xlarge.array + np.roll(uncorr_noise_xlarge.array, 1, axis=0))
ynoise_xlarge *= (np.sqrt(2.) / 2.) # make unit variance

# decimals for comparison (one for fine detail, another for comparing stochastic quantities)
decimal_approx = 2
decimal_precise = 7

# number of positions to test in nonzero lag uncorrelated tests
npos_test = 3

# number of ImageCorrFuncs to sum over to get slightly better statistics for noise generation test
nsum_test = 5

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_uncorrelated_noise_zero_lag():
    """Test that the zero lag correlation of an input uncorrelated noise field matches its variance.
    """
    t1 = time.time()
    sigmas = [3.e-9, 49., 1.11e11, 3.4e30]  # some wide ranging sigma values for the noise field
    # loop through the sigmas
    cf_zero = 0.
    for sigma in sigmas:
        # Test the estimated value is good to 1% of the input variance; we expect this!
        # Note we make multiple correlation funcs and average their zero lag to beat down noise
        for i in range(nsum_test):
            uncorr_noise_large_extra = galsim.ImageD(largeim_size, largeim_size)
            uncorr_noise_large_extra.addNoise(gd)
            noise_test = uncorr_noise_large_extra * sigma
            ncf = galsim.ImageCorrFunc(noise_test, dx=1.)
            cf_zero += ncf._profile.xValue(galsim.PositionD(0., 0.))
        cf_zero /= float(nsum_test)
        np.testing.assert_almost_equal(
            cf_zero / sigma**2, 1., decimal=decimal_approx,
            err_msg="Zero distance noise correlation value does not match input noise variance.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_uncorrelated_noise_nonzero_lag():
    """Test that the non-zero lag correlation of an input uncorrelated noise field is zero at some
    randomly chosen positions.
    """
    t1 = time.time()
    # Set up some random positions (within and outside) the bounds of the table inside the
    # ImageCorrFunc then test
    for i in range(npos_test):
        # Note we make multiple correlation funcs and average their zero lag to beat down noise
        cf_test_value = 0.
        for i in range(nsum_test):
            uncorr_noise_large_extra = galsim.ImageD(largeim_size, largeim_size)
            uncorr_noise_large_extra.addNoise(gd)
            noise_test = uncorr_noise_large_extra
            ncf = galsim.ImageCorrFunc(noise_test, dx=1.)
            # generate the test position at least one pixel away from the origin
            rpos = 1. + glob_ud() * (largeim_size - 1.) # this can go outside table bounds
            tpos = 2. * np.pi * glob_ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            cf_test_value += ncf._profile.xValue(pos)
        cf_test_value /= float(nsum_test)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_almost_equal(
            cf_test_value, 0., decimal=decimal_approx,
            err_msg="Non-zero distance noise correlation value not sufficiently close to target "+
            "value of zero.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_uncorrelated_noise_symmetry():
    """Test that the non-zero lag correlation of an input uncorrelated noise field has two-fold
    rotational symmetry.
    """
    t1 = time.time()
    ncf = galsim.ImageCorrFunc(uncorr_noise_small, dx=1.) # small image is fine here
    # Set up some random positions (within and outside) the bounds of the table inside the corrfunc
    # then test
    for i in range(npos_test):
        rpos = glob_ud() * smallim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * glob_ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = ncf._profile.xValue(pos)
        cf_test2 = ncf._profile.xValue(-pos)
        np.testing.assert_almost_equal(
            cf_test1, cf_test2,
            decimal=decimal_precise,
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_uncorrelated_noise_90degree_rotation():
    """Test that the ImageCorrFunc rotation methods produces the same output as initializing with a
    90 degree-rotated input field.
    """
    t1 = time.time()
    ncf = galsim.ImageCorrFunc(uncorr_noise_large, dx=1.)
    ks = [1, 2, 3, 4]
    angles = [
        90. * galsim.degrees, 180. * galsim.degrees, 270. * galsim.degrees, 360. * galsim.degrees]
    # loop over rotation angles and check
    for k, angle in zip(ks, angles):
        noise_ref = galsim.ImageViewD(np.ascontiguousarray(np.rot90(uncorr_noise_large.array, k=k)))
        ncf_ref = galsim.ImageCorrFunc(noise_ref, dx=1.)
        # first we'll check the createRotated() method
        ncf_test1 = ncf.createRotated(angle)
        # then we'll check the applyRotation() method
        ncf_test2 = ncf.copy()
        ncf_test2.applyRotation(angle)
        # then check some positions
        for i in range(npos_test):
            rpos = glob_ud() * smallim_size
            tpos = 2. * np.pi * glob_ud()
            pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
            cf_ref = ncf_ref._profile.xValue(pos)
            cf_test1 = ncf_test1._profile.xValue(pos)
            cf_test2 = ncf_test2._profile.xValue(pos)
            # Then test these estimated value is good to within our chosen decimal place
            np.testing.assert_almost_equal(
                cf_ref, cf_test1, decimal=decimal_precise, # slightly FFT-dependent accuracy
                err_msg="Uncorrelated noise failed 90 degree createRotated() method test.")
            np.testing.assert_almost_equal(
                cf_ref, cf_test2, decimal=decimal_precise,
                err_msg="Uncorrelated noise failed 90 degree applyRotation() method test.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_xcorr_noise_basics():
    """Test the basic properties of a noise field, correlated in the x direction, generated using
    a simple shift-add prescription.
    """
    t1 = time.time()
    # Use the xnoise defined above to make the x correlation function
    # Note we make multiple correlation funcs and average their zero lag to beat down noise
    cf_zero = 0.
    cf_10 = 0.
    for i in range(nsum_test):
        uncorr_noise_large_extra = galsim.ImageD(largeim_size, largeim_size)
        uncorr_noise_large_extra.addNoise(gd)
        xnoise_large_extra = galsim.ImageViewD(
            uncorr_noise_large_extra.array + np.roll(
                uncorr_noise_large_extra.array, 1, axis=1)) # note NumPy thus [y,x]
        xnoise_large_extra *= (np.sqrt(2.) / 2.) # make unit variance
        xncf = galsim.ImageCorrFunc(xnoise_large_extra, dx=1.)
        cf_zero += xncf._profile.xValue(galsim.PositionD(0., 0.))
        cf_10 += xncf._profile.xValue(galsim.PositionD(1., 0.))
    cf_zero /= float(nsum_test)
    cf_10 /= float(nsum_test)
    # Then test the zero-lag value is good to 1% of the input variance; we expect this!
    np.testing.assert_almost_equal(
        cf_zero, 1., decimal=decimal_approx,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (1, 0) value is good to 1% of the input variance (0.5); we expect this!
    np.testing.assert_almost_equal(
        cf_10, .5, decimal=decimal_approx,
        err_msg="Noise correlation value at (1, 0) does not match input covariance.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_ycorr_noise_basics():
    """Test the basic properties of a noise field, correlated in the y direction, generated using
    a simple shift-add prescription.
    """
    t1 = time.time()
    # use the ynoise defined above to make the y correlation function
    # Note we make multiple correlation funcs and average their zero lag to beat down noise
    cf_zero = 0.
    cf_01 = 0.
    for i in range(nsum_test):
        uncorr_noise_large_extra = galsim.ImageD(largeim_size, largeim_size)
        uncorr_noise_large_extra.addNoise(gd)
        ynoise_large_extra = galsim.ImageViewD(
            uncorr_noise_large_extra.array + np.roll(
                uncorr_noise_large_extra.array, 1, axis=0)) # note NumPy thus [y,x]
        ynoise_large_extra *= (np.sqrt(2.) / 2.) # make unit variance
        yncf = galsim.ImageCorrFunc(ynoise_large_extra, dx=1.)
        cf_zero += yncf._profile.xValue(galsim.PositionD(0., 0.))
        cf_01 += yncf._profile.xValue(galsim.PositionD(0., 1.))
    cf_zero /= float(nsum_test)
    cf_01 /= float(nsum_test)
    # Then test the zero-lag value is good to 1% of the input variance; we expect this!
    np.testing.assert_almost_equal(
        cf_zero, 1., decimal=decimal_approx,
        err_msg="Zero distance noise correlation value does not match input noise variance.")
    # Then test the (0, 1) value is good to 1% of the input variance (0.5); we expect this!
    np.testing.assert_almost_equal(
        cf_01, .5, decimal=decimal_approx,
        err_msg="Noise correlation value at (0, 1) does not match input covariance.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_xcorr_noise_symmetry():
    """Test that the non-zero lag correlation of an input x correlated noise field has two-fold
    rotational symmetry.
    """
    t1 = time.time()
    ncf = galsim.ImageCorrFunc(xnoise_small, dx=1.) # the small image is fine here
    # set up some random positions (within and outside) the bounds of the table inside the corrfunc
    # then test
    for i in range(npos_test):
        rpos = glob_ud() * smallim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * glob_ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = ncf._profile.xValue(pos)
        cf_test2 = ncf._profile.xValue(-pos)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_almost_equal(
            cf_test1, cf_test2, decimal=decimal_precise, # should be good to machine precision
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric for x correlated noise field.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_ycorr_noise_symmetry():
    """Test that the non-zero lag correlation of an input y correlated noise field has two-fold
    rotational symmetry.
    """
    t1 = time.time()
    ncf = galsim.ImageCorrFunc(ynoise_small, dx=1.) # the small image is fine here
    # set up some random positions (within and outside) the bounds of the table inside the corrfunc
    # then test
    for i in range(npos_test):
        rpos = glob_ud() * smallim_size # this can go outside lookup table bounds
        tpos = 2. * np.pi * glob_ud()
        pos = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
        cf_test1 = ncf._profile.xValue(pos)
        cf_test2 = ncf._profile.xValue(-pos)
        # Then test this estimated value is good to within our chosen decimal place of zero
        np.testing.assert_almost_equal(
            cf_test1, cf_test2, decimal=decimal_precise, # should be good to machine precision
            err_msg="Non-zero distance noise correlation values not two-fold rotationally "+
            "symmetric for y correlated noise field.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_90degree_rotation(): # probably only need to do the x direction for this test if the 
                              # previous tests have passed OK
    """Test that the ImageCorrFunc rotation methods produces the same output as initializing with a
    90 degree-rotated input field, this time with a noise field that's correlated in the x
    direction.
    """
    t1 = time.time()
    ncf = galsim.ImageCorrFunc(xnoise_large, dx=1.)
    ks = [1, 2, 3, 4]
    angles = [
        90. * galsim.degrees, 180. * galsim.degrees, 270. * galsim.degrees, 360. * galsim.degrees]
    # loop over rotation angles and check
    for k, angle in zip(ks, angles):
        noise_ref = galsim.ImageViewD(np.ascontiguousarray(np.rot90(xnoise_large.array, k=k)))
        ncf_ref = galsim.ImageCorrFunc(noise_ref, dx=1.)
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
            cf_ref = ncf_ref._profile.xValue(pos)
            cf_test1 = ncf_test1._profile.xValue(pos)
            cf_test2 = ncf_test2._profile.xValue(pos) 
            np.testing.assert_almost_equal(
                cf_ref, cf_test1, decimal=decimal_precise, # should be accurate, but FFT-dependent 
                err_msg="Noise correlated in the x direction failed 90 degree createRotated() "+
                "method test.")
            np.testing.assert_almost_equal(
                cf_ref, cf_test2, decimal=decimal_precise,
                err_msg="Noise correlated in the x direction failed 90 degree applyRotation() "+
                "method test.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_arbitrary_rotation():
    """Check that rotated correlation function xValues() are correct for a correlation function with
    something in it.
    """
    t1 = time.time()
    ncf = galsim.ImageCorrFunc(ynoise_small, dx=1.) # use something >0
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
        np.testing.assert_almost_equal(
            ncf._profile.xValue(pos_rot), ncf_rot1._profile.xValue(pos_ref), 
            decimal=decimal_precise, # this should be good at very high accuracy 
            err_msg="Noise correlated in the y direction failed createRotated() "+
            "method test for arbitrary rotations.")
        np.testing.assert_almost_equal(
            ncf._profile.xValue(pos_rot), ncf_rot2._profile.xValue(pos_ref), 
            decimal=decimal_precise, # ditto
            err_msg="Noise correlated in the y direction failed applyRotation() "+
            "method test for arbitrary rotations.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_scaling():
    """Test the scaling of correlation functions.
    """
    t1 = time.time()
    ncf = galsim.ImageCorrFunc(ynoise_small, dx=1.)
    scalings = [7.e-13, 424., 7.9e23]
    for scale in scalings:
       ncf_test1 = ncf.createMagnified(scale)
       ncf_test2 = ncf.copy() 
       ncf_test2.applyMagnification(scale)
       for i in range(npos_test):
           rpos = glob_ud() * 0.1 * smallim_size * scale # look in vicinity of the centre
           tpos = 2. * np.pi * glob_ud()
           pos_ref = galsim.PositionD(rpos * np.cos(tpos), rpos * np.sin(tpos))
           np.testing.assert_almost_equal(
               ncf_test1._profile.xValue(pos_ref), ncf._profile.xValue(pos_ref / scale),
               decimal=decimal_precise,
               err_msg="Noise correlated in the y direction failed createMagnified() scaling test.")
           np.testing.assert_almost_equal(
               ncf_test2._profile.xValue(pos_ref), ncf._profile.xValue(pos_ref / scale),
               decimal=decimal_precise,
               err_msg="Noise correlated in the y direction failed applyMagnification() scaling "+
               "test.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_draw():
    """Test that the ImageCorrFunc draw() method matches its internal, NumPy-derived estimate of the
    correlation function, and an independent calculation of the same thing.
    """
    t1 = time.time()
    from galsim import utilities
    # We have slightly different expectations for how the ImageCorrFunc will represent and store CFs
    # from even and odd sized noise fields, so we will test both here.  
    #
    # First let's do odd (an uncorrelated noise field is fine for the tests we want to do):
    uncorr_noise_small_odd = galsim.ImageD(smallim_size_odd, smallim_size_odd)
    uncorr_noise_small.addNoise(gd)
    # Build a noise correlation function using DFTs
    ft_array = np.fft.fft2(uncorr_noise_small_odd.array)
    # Calculate the power spectrum then correlation function
    ps_array = (ft_array * ft_array.conj()).real
    cf_array = (np.fft.ifft2(ps_array)).real / float(np.product(np.shape(ft_array)))
    cf_array = utilities.roll2d(cf_array, (cf_array.shape[0] / 2, cf_array.shape[1] / 2))
    # Then use the ImageCorrFunc class for comparison
    ncf = galsim.ImageCorrFunc(uncorr_noise_small_odd, dx=1.)
    testim1 = galsim.ImageD(smallim_size_odd, smallim_size_odd)
    ncf.draw(testim1, dx=1.)
    # Then compare the odd-sized arrays:
    np.testing.assert_array_almost_equal(
        testim1.array, cf_array, decimal=decimal_precise, 
        err_msg="Drawn image does not match independently calculated correlation function.")
    # Now we do even
    ft_array = np.fft.fft2(uncorr_noise_small.array)
    # Calculate the power spectrum then correlation function
    ps_array = (ft_array * ft_array.conj()).real
    cf_array = (np.fft.ifft2(ps_array)).real / float(np.product(np.shape(ft_array)))
    cf_array = utilities.roll2d(cf_array, (cf_array.shape[0] / 2, cf_array.shape[1] / 2))
    # Then use the ImageCorrFunc class for comparison
    ncf = galsim.ImageCorrFunc(uncorr_noise_small, dx=1.)
    testim1 = galsim.ImageD(smallim_size, smallim_size)
    ncf.draw(testim1, dx=1.)
    # Then compare the even-sized arrays:
    np.testing.assert_array_almost_equal(
        testim1.array, cf_array, decimal=decimal_precise, 
        err_msg="Drawn image does not match independently calculated correlation function.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_output_generation_basic():
    """Test that noise generated by a ImageCorrFunc matches the correlation function.  Averages over
    a ImageCorrFuncs from a number of realizations.
    """
    t1 = time.time()
    # Get the noise correlation function from an image
    ncf = galsim.ImageCorrFunc(xnoise_large, dx=1.)
    refim = galsim.ImageD(smallim_size, smallim_size)
    # Draw this for reference
    ncf.draw(refim, dx=1.)
    # Generate a large image containing noise according to this function
    outimage = galsim.ImageD(xnoise_large.bounds)
    ncf.applyNoiseTo(outimage, dx=1., dev=glob_ud)
    # Summed (average) ImageCorrFuncs should be approximately equal to the input, so average
    # multiple CFs
    ncf_2ndlevel = galsim.ImageCorrFunc(outimage, dx=1.)
    for i in range(nsum_test - 1):
        # Then repeat
        outimage.setZero()
        ncf.applyNoiseTo(outimage, dx=1., dev=glob_ud)
        ncf_2ndlevel += galsim.ImageCorrFunc(outimage, dx=1.)
    ncf_2ndlevel /= float(nsum_test)
    # Then draw the summed CF to an image for comparison 
    testim = galsim.ImageD(smallim_size, smallim_size)
    ncf_2ndlevel.draw(testim, dx=1.)
    np.testing.assert_array_almost_equal(
        testim.array, refim.array, decimal=decimal_approx,
        err_msg="Generated noise field (basic) does not match input correlation properties.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_output_generation_rotated():
    """Test that noise generated by a rotated ImageCorrFunc matches the correlation function.
    """
    t1 = time.time()
    # Get the noise correlation function
    # Note that here we use an extra large image: this is because rotating the noise correlation
    # function (CF) brings in beyond-edge regions (imagine rotating a square but trimming within a
    # fixed square border of the same size).  These seem to add excess variance, perhaps due to
    # interpolant behaviour across transition to formal zero in the CF, which ruins agreement at
    # 2dp (still OK to 1dp or better).  This behaviour is quite strongly dependent on interpolant,
    # with Linear seeming to provide the best performance.  This is also likely to be related to
    # the fact that we do not zero-pad while generating the noise field as we might while generating
    # a galaxy in an empty patch of sky: the Linear interpolatant has a limited real space support.
    #
    # Therefore, we rotate a CF with a support larger than the output region we simulate: this works
    # well at 2dp.
    #
    # TODO: It would be good to understand more about the detailed interpolant behaviour though...
    ncf = galsim.ImageCorrFunc(ynoise_xlarge, dx=1.)
    # Then loop over some angles
    angles = [28.7 * galsim.degrees, 135. * galsim.degrees]
    for angle in angles:
        ncf_rot = ncf.createRotated(angle)
        refim = galsim.ImageD(smallim_size * 2, smallim_size * 2)
        # Draw this for reference
        ncf_rot.draw(refim, dx=1.)
        # Generate a large image containing noise according to this function
        outimage = galsim.ImageD(largeim_size, largeim_size)
        ncf_rot.applyNoiseTo(outimage, dx=1., dev=glob_ud)
        # Summed (average) ImageCorrFuncs should be approximately equal to the input, so avg
        # multiple CFs
        ncf_2ndlevel = galsim.ImageCorrFunc(outimage, dx=1.)
        for i in range(nsum_test - 1):
            # Then repeat
            outimage.setZero()
            ncf_rot.applyNoiseTo(outimage, dx=1., dev=glob_ud)
            ncf_2ndlevel += galsim.ImageCorrFunc(outimage, dx=1.)
        ncf_2ndlevel /= float(nsum_test)
        # Then draw the summed CF to an image for comparison 
        testim = galsim.ImageD(smallim_size * 2, smallim_size * 2)
        ncf_2ndlevel.draw(testim, dx=1.)
        np.testing.assert_array_almost_equal(
            testim.array, refim.array, decimal=decimal_approx,
            err_msg="Generated noise field (rotated) does not match input correlation properties.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)

def test_output_generation_magnified():
    """Test that noise generated by a magnified ImageCorrFunc matches the correlation function.
    """
    t1 = time.time()
    # Get the noise correlation function
    ncf = galsim.ImageCorrFunc(ynoise_large, dx=1.)
    refim = galsim.ImageD(smallim_size, smallim_size)
    # Draw this for reference
    ncf.draw(refim, dx=1.)
    # Then loop over some scales, using `applyNoiseTo` with the relevant scaling in the `dx` to
    # argument check that the underlying correlation function is preserved when both `dx` and
    # a magnification factor `scale` change in the same sense
    scales = [0.03, 11.]
    for scale in scales:
        ncf_scl = ncf.createMagnified(scale)
        # Generate a large image containing noise according to this function
        outimage = galsim.ImageD(largeim_size, largeim_size)
        ncf_scl.applyNoiseTo(outimage, dx=scale, dev=glob_ud)
        # Summed (average) ImageCorrFuncs should be approximately equal to the input, so avg
        # multiple CFs
        ncf_2ndlevel = galsim.ImageCorrFunc(outimage, dx=1.)
        for i in range(nsum_test + 3 - 1): # Need to add here to nsum_test to beat down noise
            # Then repeat
            outimage.setZero()
            ncf_scl.applyNoiseTo(outimage, dx=scale, dev=glob_ud) # apply noise using scale
            ncf_2ndlevel += galsim.ImageCorrFunc(outimage, dx=1.)
        # Divide by nsum_test to get average quantities
        ncf_2ndlevel /= float(nsum_test + 3)
        # Then draw the summed CF to an image for comparison 
        testim = galsim.ImageD(smallim_size, smallim_size)
        ncf_2ndlevel.draw(testim, dx=1.)
        np.testing.assert_array_almost_equal(
            testim.array, refim.array, decimal=decimal_approx,
            err_msg="Generated noise does not match (magnified) input correlation properties.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(), t2 - t1)


if __name__ == "__main__":
    test_uncorrelated_noise_zero_lag()
    test_uncorrelated_noise_nonzero_lag()
    test_uncorrelated_noise_symmetry()
    test_uncorrelated_noise_90degree_rotation()
    test_xcorr_noise_basics()
    test_ycorr_noise_basics()
    test_xcorr_noise_symmetry()
    test_ycorr_noise_symmetry()
    test_90degree_rotation()
    test_arbitrary_rotation()
    test_scaling()
    test_draw()
    test_output_generation_basic()
    test_output_generation_rotated()
    test_output_generation_magnified()
