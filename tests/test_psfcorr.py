import os
import sys
import pyfits
import numpy as np
import math

"""Unit tests for the PSF correction and shear estimation routines.

There are two types of tests: tests that use Gaussian profiles, for which the ideal results are
known; and tests that use real galaxies in SDSS for which results were tabulated using the same code
before it was integrated into GalSim (so we can make sure we are not breaking anything as we modify
the code).
"""

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# define a range of input parameters for the Gaussians that we are testing
gaussian_sig_values = [0.5, 1.0, 2.0]
shear_values = [0.01, 0.03, 0.10, 0.30]
pixel_scale = 0.2
decimal = 2 # decimal place at which to require equality in sizes
decimal_shape = 3 # decimal place at which to require equality in shapes

# define inputs and expected results for tests that use real SDSS galaxies
img_dir = os.path.join(".","HSM_precomputed")
gal_file_prefix = "image."
psf_file_prefix = "psf."
img_suff = ".fits"
file_indices = [0, 2, 4, 6, 8]
x_centroid = [35.888, 19.44, 8.74, 20.193, 57.94]
y_centroid = [19.845, 25.047, 11.92, 38.93, 27.73]
sky_var = [35.01188, 35.93418, 35.15456, 35.11146, 35.16454]
correction_methods = ["KSB", "BJ", "LINEAR", "REGAUSS"]
# Note: expected results give shear for KSB and distortion for others, but the results below have
# converted KSB expected results to distortion for the sake of consistency
e1_expected = np.array([
        [0.467603106752, 0.381211727, 0.398856937, 0.401755571],
        [0.28618443944, 0.199222784, 0.233883543, 0.234257525],
        [0.271533794146, 0.158049396, 0.183517068, 0.184893412],
        [-0.293754156071, -0.457024541, 0.123946584, -0.609233462],
        [0.557720893779, 0.374143023, 0.714147448, 0.435404409] ])
e2_expected = np.array([
        [-0.867225166489, -0.734855778, -0.777027588, -0.774684891],
        [-0.469354341577, -0.395520479, -0.502540961, -0.464466257],
        [-0.519775291311, -0.471589061, -0.574750641, -0.529664935],
        [0.345688365839, -0.342047099, 0.120603755, -0.446743913],
        [0.525728304099, 0.370691830, 0.702724807, 0.433999442] ])
resolution_expected = np.array([
        [0.796144249, 0.835624917, 0.835624917, 0.827796187],
        [0.685023735, 0.699602704, 0.699602704, 0.659457638],
        [0.634736458, 0.651040481, 0.651040481, 0.614663396],
        [0.477027015, 0.477210752, 0.477210752, 0.423157447],
        [0.595205998, 0.611824797, 0.611824797, 0.563582092] ])
sigma_e_expected = np.array([
        [0.016924826, 0.014637648, 0.014637648, 0.014465546],
        [0.075769504, 0.073602324, 0.073602324, 0.064414520],
        [0.110253112, 0.106222900, 0.106222900, 0.099357106],
        [0.185276702, 0.184300955, 0.184300955, 0.173478300],
        [0.073020065, 0.070270966, 0.070270966, 0.061856263] ])

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_moments_basic():
    """Test that we can properly recover adaptive moments for Gaussians."""
    import time
    t1 = time.time()
    for sig in gaussian_sig_values:
        for g1 in shear_values:
            for g2 in shear_values:
                total_shear = np.sqrt(g1**2 + g2**2)
                conversion_factor = np.tanh(2.0*math.atanh(total_shear))/total_shear
                distortion_1 = g1*conversion_factor
                distortion_2 = g2*conversion_factor
                gal = galsim.Gaussian(flux = 1.0, sigma = sig)
                gal.applyShear(g1=g1, g2=g2)
                gal_image = gal.draw(dx = pixel_scale)
                result = gal_image.FindAdaptiveMom()
                # make sure we find the right Gaussian sigma
                np.testing.assert_almost_equal(np.fabs(result.moments_sigma-sig/pixel_scale), 0.0,
                                               err_msg = "- incorrect dsigma", decimal = decimal)
                # make sure we find the right e
                np.testing.assert_almost_equal(result.observed_shape.e1,
                                               distortion_1, err_msg = "- incorrect e1",
                                               decimal = decimal_shape)
                np.testing.assert_almost_equal(result.observed_shape.e2,
                                               distortion_2, err_msg = "- incorrect e2",
                                               decimal = decimal_shape)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shearest_basic():
    """Test that we can recover shears for Gaussian galaxies and PSFs."""
    import time
    t1 = time.time()
    for sig in gaussian_sig_values:
        for g1 in shear_values:
            for g2 in shear_values:
                total_shear = np.sqrt(g1**2 + g2**2)
                conversion_factor = np.tanh(2.0*math.atanh(total_shear))/total_shear
                distortion_1 = g1*conversion_factor
                distortion_2 = g2*conversion_factor
                gal = galsim.Gaussian(flux = 1.0, sigma = sig)
                psf = galsim.Gaussian(flux = 1.0, sigma = sig)
                gal.applyShear(g1=g1, g2=g2)
                final = galsim.Convolve([gal, psf])
                final_image = final.draw(dx = pixel_scale)
                epsf_image = psf.draw(dx = pixel_scale)
                result = galsim.EstimateShearHSM(final_image, epsf_image)
                # make sure we find the right e after PSF correction
                # with regauss, which returns a distortion
                np.testing.assert_almost_equal(result.corrected_e1,
                                               distortion_1, err_msg = "- incorrect e1",
                                               decimal = decimal_shape)
                np.testing.assert_almost_equal(result.corrected_e2,
                                               distortion_2, err_msg = "- incorrect e2",
                                               decimal = decimal_shape)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shearest_precomputed():
    """Test that we can recover shears the same as before the code was put into GalSim."""
    import time
    t1 = time.time()
    # loop over real galaxies
    for index in range(len(file_indices)):
        # define input filenames
        img_file = os.path.join(img_dir, gal_file_prefix + str(file_indices[index]) + img_suff)
        psf_file = os.path.join(img_dir, psf_file_prefix + str(file_indices[index]) + img_suff)

        # read in information for objects and expected results
        img = galsim.fits.read(img_file)
        img -= 1000
        psf = galsim.fits.read(psf_file)
        psf -= 1000

        # loop over methods
        for method_index in range(len(correction_methods)):
            # call PSF correction
            result = galsim.EstimateShearHSM(img, psf, sky_var = sky_var[index], shear_est =
                                             correction_methods[method_index],
                                             guess_x_centroid = x_centroid[index], guess_y_centroid
                                             = y_centroid[index])

            # compare results with precomputed
            print result.meas_type, correction_methods[method_index]
            if result.meas_type == 'e':
                np.testing.assert_almost_equal(result.corrected_e1,
                                               e1_expected[index][method_index], decimal =
                                               decimal_shape)
                np.testing.assert_almost_equal(result.corrected_e2,
                                               e2_expected[index][method_index], decimal =
                                               decimal_shape)
            else:
                gval = np.sqrt(result.corrected_g1**2 + result.corrected_g2**2)
                if gval <= 1.0:
                    s = galsim.Shear(g1=result.corrected_g1, g2=result.corrected_g2)
                    np.testing.assert_almost_equal(s.e1,
                                                   e1_expected[index][method_index], decimal =
                                                   decimal_shape)
                    np.testing.assert_almost_equal(s.e2,
                                                   e2_expected[index][method_index], decimal =
                                                   decimal_shape)
            # also compare resolutions and estimated errors
            np.testing.assert_almost_equal(result.resolution_factor,
                                           resolution_expected[index][method_index], decimal =
                                           decimal_shape)
            np.testing.assert_almost_equal(result.corrected_shape_err,
                                           sigma_e_expected[index][method_index], decimal =
                                           decimal_shape)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_masks():
    """Test that moments and shear estimation routines respond appropriately to masks."""
    # set up some toy galaxy and PSF
    my_sigma = 1.0
    my_pixscale = 0.1
    my_g1 = 0.15
    my_g2 = -0.4
    imsize = 256
    g = galsim.Gaussian(sigma = my_sigma)
    p = galsim.Gaussian(sigma = my_sigma) # the ePSF is Gaussian (kind of silly but it means we can
                                     # predict results exactly)
    g.applyShear(g1=my_g1, g2=my_g2)
    obj = galsim.Convolve(g, p)
    im = galsim.ImageF(imsize, imsize)
    p_im = galsim.ImageF(imsize, imsize)
    im = obj.draw(image = im, dx = my_pixscale)
    p_im = p.draw(image = p_im, dx = my_pixscale)

    # make some screwy masks that should cause issues, and check that the exception is thrown
    ## mask of floats
    mask_im = galsim.ImageF(imsize, imsize)
    np.testing.assert_raises(ValueError, galsim.FindAdaptiveMom, im, mask_im)
    np.testing.assert_raises(ValueError, galsim.EstimateShearHSM, im, p_im, mask_im)
    ## mask different size from image
    mask_im = galsim.ImageI(imsize, 2*imsize)
    np.testing.assert_raises(ValueError, galsim.FindAdaptiveMom, im, mask_im)
    np.testing.assert_raises(ValueError, galsim.EstimateShearHSM, im, p_im, mask_im)
    ## mask has weird values
    mask_im = galsim.ImageI(imsize, imsize)-3
    np.testing.assert_raises(ValueError, galsim.FindAdaptiveMom, im, mask_im)
    np.testing.assert_raises(ValueError, galsim.EstimateShearHSM, im, p_im, mask_im)
    ## mask excludes all pixels
    mask_im = galsim.ImageI(imsize, imsize)
    np.testing.assert_raises(RuntimeError, galsim.FindAdaptiveMom, im, mask_im)
    np.testing.assert_raises(RuntimeError, galsim.EstimateShearHSM, im, p_im, mask_im)

    # check moments, shear without mask
    resm = im.FindAdaptiveMom()
    ress = galsim.EstimateShearHSM(im, p_im)

    # check moments, shear with mask that includes all pixels
    maskall1 = galsim.ImageI(imsize, imsize) + 1
    resm_maskall1 = im.FindAdaptiveMom(maskall1)
    ress_maskall1 = galsim.EstimateShearHSM(im, p_im, maskall1)
    np.testing.assert_equal(resm.observed_shape.e1, resm_maskall1.observed_shape.e1,
        err_msg="e1 from FindAdaptiveMom changes when using inclusive mask")
    np.testing.assert_equal(resm.observed_shape.e2, resm_maskall1.observed_shape.e2,
        err_msg="e2 from FindAdaptiveMom changes when using inclusive mask")
    np.testing.assert_equal(resm.moments_sigma, resm_maskall1.moments_sigma,
        err_msg="sigma from FindAdaptiveMom changes when using inclusive mask")
    np.testing.assert_equal(ress.observed_shape.e1, ress_maskall1.observed_shape.e1,
        err_msg="observed e1 from EstimateShearHSM changes when using inclusive mask")
    np.testing.assert_equal(ress.observed_shape.e2, ress_maskall1.observed_shape.e2,
        err_msg="observed e2 from EstimateShearHSM changes when using inclusive mask")
    np.testing.assert_equal(ress.moments_sigma, ress_maskall1.moments_sigma,
        err_msg="observed sigma from EstimateShearHSM changes when using inclusive mask")
    np.testing.assert_equal(ress.corrected_e1, ress_maskall1.corrected_e1,
        err_msg="corrected e1 from EstimateShearHSM changes when using inclusive mask")
    np.testing.assert_equal(ress.corrected_e2, ress_maskall1.corrected_e2,
        err_msg="corrected e2 from EstimateShearHSM changes when using inclusive mask")
    np.testing.assert_equal(ress.resolution_factor, ress_maskall1.resolution_factor,
        err_msg="resolution factor from EstimateShearHSM changes when using inclusive mask")

    # check moments and shears with mask of edges, should be nearly the same
    # (this seems dumb, but it's helpful for keeping track of whether the pointers in the C++ code
    # are being properly updated despite the masks.  If we monkey in that code again, it will be a
    # useful check.)
    maskedge = galsim.ImageI(imsize, imsize) + 1
    xmin = maskedge.xmin
    xmax = maskedge.xmax
    ymin = maskedge.ymin
    ymax = maskedge.ymax
    edgenum = 3
    for ind1 in range(xmin, xmax+1):
        for ind2 in range(ymin, ymax+1):
            if (ind1 <= (xmin+edgenum)) or (ind1 >= (xmax-edgenum)) or (ind2 <= (ymin+edgenum)) or (ind2 >= (ymax-edgenum)):
                maskedge.setValue(ind1, ind2, 0)
    resm_maskedge = im.FindAdaptiveMom(maskedge)
    ress_maskedge = galsim.EstimateShearHSM(im, p_im, maskedge)
    test_decimal = 4
    np.testing.assert_almost_equal(resm.observed_shape.e1, resm_maskedge.observed_shape.e1,
        decimal=test_decimal, err_msg="e1 from FindAdaptiveMom changes when masking edge")
    np.testing.assert_almost_equal(resm.observed_shape.e2, resm_maskedge.observed_shape.e2,
        decimal=test_decimal, err_msg="e2 from FindAdaptiveMom changes when masking edge")
    np.testing.assert_almost_equal(resm.moments_sigma, resm_maskedge.moments_sigma,
        decimal=test_decimal, err_msg="sigma from FindAdaptiveMom changes when masking edge")
    np.testing.assert_almost_equal(ress.observed_shape.e1, ress_maskedge.observed_shape.e1,
        decimal=test_decimal, err_msg="observed e1 from EstimateShearHSM changes when masking edge")
    np.testing.assert_almost_equal(ress.observed_shape.e2, ress_maskedge.observed_shape.e2,
        decimal=test_decimal, err_msg="observed e2 from EstimateShearHSM changes when masking edge")
    np.testing.assert_almost_equal(ress.moments_sigma, ress_maskedge.moments_sigma,
        decimal=test_decimal,
        err_msg="observed sigma from EstimateShearHSM changes when masking edge")
    np.testing.assert_almost_equal(ress.corrected_e1, ress_maskedge.corrected_e1,
        decimal=test_decimal,
        err_msg="corrected e1 from EstimateShearHSM changes when masking edge")
    np.testing.assert_almost_equal(ress.corrected_e2, ress_maskedge.corrected_e2,
        decimal=test_decimal,
        err_msg="corrected e2 from EstimateShearHSM changes when masking edge")
    np.testing.assert_almost_equal(ress.resolution_factor, ress_maskedge.resolution_factor,
        decimal=test_decimal,
        err_msg="resolution factor from EstimateShearHSM changes when masking edge")

if __name__ == "__main__":
    test_moments_basic()
    test_shearest_basic()
    test_shearest_precomputed()
    test_masks()
