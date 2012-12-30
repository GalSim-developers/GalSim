import numpy as np
import os
import sys

"""Unit tests for the InterpolatedImage class.
"""

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def funcname():
    import inspect
    return inspect.stack()[1][3]

# For photon shooting, we calculate the number of photons to use based on the target
# accuracy we are shooting for.  (Pun intended.)
# For each pixel,
# uncertainty = sqrt(N_pix) * flux_photon = sqrt(N_tot * flux_pix / flux_tot) * flux_tot / N_tot
#             = sqrt(flux_pix) * sqrt(flux_tot) / sqrt(N_tot)
# This is largest for the brightest pixel.  So we use:
# N = flux_max * flux_tot / photon_shoot_accuracy^2
photon_shoot_accuracy = 2.e-3
# The number of decimal places at which to test the photon shooting
photon_decimal_test = 2
# for flux normalization tests
test_flux = 0.7
# for dx tests - avoid 1.0 because factors of dx^2 won't show up!
test_dx = 2.0
# Use a deterministic random number generator so we don't fail tests because of rare flukes
# in the random numbers.
glob_ud = galsim.UniformDeviate(12345)

# do_shoot utility taken from test_SBProfile.py
def do_shoot(prof, img, name):
    print 'Start do_shoot'
    # Test photon shooting for a particular profile (given as prof). 
    # Since shooting implicitly convolves with the pixel, we need to compare it to 
    # the given profile convolved with a pixel.
    pix = galsim.Pixel(xw=img.getScale())
    compar = galsim.Convolve(prof,pix)
    compar.draw(img)
    flux_max = img.array.max()
    print 'prof.getFlux = ',prof.getFlux()
    print 'compar.getFlux = ',compar.getFlux()
    print 'flux_max = ',flux_max
    flux_tot = img.array.sum()
    print 'flux_tot = ',flux_tot
    if flux_max > 1.:
        # Since the number of photons required for a given accuracy level (in terms of 
        # number of decimal places), we rescale the comparison by the flux of the 
        # brightest pixel.
        compar /= flux_max
        img /= flux_max
        prof /= flux_max
        # The formula for number of photons needed is:
        # nphot = flux_max * flux_tot / photon_shoot_accuracy**2
        # But since we rescaled the image by 1/flux_max, it becomes
        nphot = flux_tot / flux_max / photon_shoot_accuracy**2
    elif flux_max < 0.1:
        # If the max is very small, at least bring it up to 0.1, so we are testing something.
        scale = 0.1 / flux_max;
        print 'scale = ',scale
        compar *= scale
        img *= scale
        prof *= scale
        nphot = flux_max * flux_tot * scale * scale / photon_shoot_accuracy**2
    else:
        nphot = flux_max * flux_tot / photon_shoot_accuracy**2
    print 'prof.getFlux => ',prof.getFlux()
    print 'compar.getFlux => ',compar.getFlux()
    print 'img.sum => ',img.array.sum()
    print 'img.max => ',img.array.max()
    print 'nphot = ',nphot
    img2 = img.copy()
    prof.drawShoot(img2, n_photons=nphot, poisson_flux=False, rng=glob_ud)
    print 'img2.sum => ',img2.array.sum()
    np.testing.assert_array_almost_equal(
            img2.array, img.array, photon_decimal_test,
            err_msg="Photon shooting for %s disagrees with expected result"%name)

    # Test normalization
    dx = img.getScale()
    # Test with a large image to make sure we capture enough of the flux
    # even for slow convergers like Airy (which needs a _very_ large image) or Sersic.
    if 'Airy' in name:
        img = galsim.ImageD(2048,2048)
    elif 'Sersic' in name or 'DeVauc' in name:
        img = galsim.ImageD(512,512)
    else:
        img = galsim.ImageD(128,128)
    img.setScale(dx)
    compar.setFlux(test_flux)
    compar.draw(img, normalization="surface brightness")
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux/(dx*dx)
    np.testing.assert_almost_equal(img.array.sum() * dx*dx, test_flux, 5,
            err_msg="Surface brightness normalization for %s disagrees with expected result"%name)
    compar.draw(img, normalization="flux")
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux
    np.testing.assert_almost_equal(img.array.sum(), test_flux, 5,
            err_msg="Flux normalization for %s disagrees with expected result"%name)

    prof.setFlux(test_flux)
    scale = test_flux / flux_tot # from above
    nphot *= scale * scale
    print 'nphot -> ',nphot
    if 'InterpolatedImage' in name:
        nphot *= 10
        print 'nphot -> ',nphot
    prof.drawShoot(img, n_photons=nphot, normalization="surface brightness", poisson_flux=False,
                   rng=glob_ud)
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux/(dx*dx)
    np.testing.assert_almost_equal(img.array.sum() * dx*dx, test_flux, photon_decimal_test,
            err_msg="Photon shooting SB normalization for %s disagrees with expected result"%name)
    prof.drawShoot(img, n_photons=nphot, normalization="flux", poisson_flux=False,
                   rng=glob_ud)
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux
    np.testing.assert_almost_equal(img.array.sum(), test_flux, photon_decimal_test,
            err_msg="Photon shooting flux normalization for %s disagrees with expected result"%name)

def test_roundtrip():
    """Test round trip from Image to InterpolatedImage back to Image.
    """
    # Based heavily on test_sbinterpolatedimage() in test_SBProfile.py!
    import time
    t1 = time.time()

    # for each type, try to make an SBInterpolatedImage, and check that when we draw an image from
    # that SBInterpolatedImage that it is the same as the original
    ftypes = [np.float32, np.float64]
    ref_array = np.array([
        [0.01, 0.08, 0.07, 0.02],
        [0.13, 0.38, 0.52, 0.06],
        [0.09, 0.41, 0.44, 0.09],
        [0.04, 0.11, 0.10, 0.01] ]) 

    for array_type in ftypes:
        image_in = galsim.ImageView[array_type](ref_array.astype(array_type))
        np.testing.assert_array_equal(
                ref_array.astype(array_type),image_in.array,
                err_msg="Array from input Image differs from reference array for type %s"%
                        array_type)
        interp = galsim.InterpolatedImage(image_in, dx=test_dx)
        test_array = np.zeros(ref_array.shape, dtype=array_type)
        image_out = galsim.ImageView[array_type](test_array)
        image_out.setScale(test_dx)
        interp.draw(image_out)
        np.testing.assert_array_equal(
                ref_array.astype(array_type),image_out.array,
                err_msg="Array from output Image differs from reference array for type %s"%
                        array_type)
 
        # Lanczos doesn't quite get the flux right.  Wrong at the 5th decimal place.
        # Gary says that's expected -- Lanczos isn't technically flux conserving.  
        # He applied the 1st order correction to the flux, but expect to be wrong at around
        # the 10^-5 level.
        # Anyway, Quintic seems to be accurate enough.
        quint = galsim.Quintic(1.e-4)
        quint_2d = galsim.InterpolantXY(quint)
        interp = galsim.InterpolatedImage(image_in, interpolant=quint_2d, dx=test_dx, flux=1.)
        do_shoot(interp,image_out,"InterpolatedImage")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_fluxnorm():
    """Test that InterpolatedImage class responds properly to instructions about flux normalization.
    """
    import time
    t1 = time.time()

    # define values
    im_lin_scale = 5 # make an image with this linear scale
    im_fill_value = 3. # fill it with this number
    im_scale = 1.3

    # First, make some Image with some total flux value (sum of pixel values) and scale
    im = galsim.ImageF(im_lin_scale, im_lin_scale)
    im.fill(im_fill_value)
    im.setScale(im_scale)
    total_flux = im_fill_value*(im_lin_scale**2)
    np.testing.assert_equal(total_flux, im.array.sum(),
                            err_msg='Created array with wrong total flux')

    # Check that if we make an InterpolatedImage with flux normalization, it keeps that flux
    interp = galsim.InterpolatedImage(im) # note, flux normalization is the default
    np.testing.assert_equal(total_flux, interp.getFlux(),
                            err_msg='Did not keep flux normalization')
    # Check that this is preserved when drawing
    im2 = interp.draw(dx = im_scale)
    np.testing.assert_equal(total_flux, im2.array.sum(),
                            err_msg='Drawn image does not have expected flux normalization')

    # Now make an InterpolatedImage but tell it sb normalization
    interp_sb = galsim.InterpolatedImage(im, normalization = 'sb')
    # Check that when drawing, the sum is equal to what we expect if the original image had been
    # surface brightness
    im3 = interp_sb.draw(dx = im_scale)
    np.testing.assert_almost_equal(total_flux*(im_scale**2), im3.array.sum(), decimal=6,
                                   err_msg='Did not use surface brightness normalization')
    # Check that when drawing with sb normalization, the sum is the same as the original
    im4 = interp_sb.draw(dx = im_scale, normalization = 'sb')
    np.testing.assert_almost_equal(total_flux, im4.array.sum(), decimal=6,
                                   err_msg='Failed roundtrip for sb normalization')

    # Finally make an InterpolatedImage but give it some other flux value
    interp_flux = galsim.InterpolatedImage(im, flux=test_flux)
    # Check that it has that flux
    np.testing.assert_equal(test_flux, interp_flux.getFlux(),
                            err_msg = 'InterpolatedImage did not use flux keyword')
    # Check that this is preserved when drawing
    im5 = interp_flux.draw(dx = im_scale)
    np.testing.assert_almost_equal(test_flux, im5.array.sum(), decimal=6,
                                   err_msg = 'Drawn image does not reflect flux keyword')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_exceptions():
    """Test failure modes for InterpolatedImage class.
    """
    import time
    t1 = time.time()

    # What if it receives as input something that is not an Image? Give it a GSObject to check.
    g = galsim.Gaussian(sigma=1.)
    np.testing.assert_raises(ValueError, galsim.InterpolatedImage, g)
    # What if Image does not have a scale set, but dx keyword is not specified?
    im = galsim.ImageF(5, 5)
    np.testing.assert_raises(ValueError, galsim.InterpolatedImage, im)
    # Image must have bounds defined
    im = galsim.ImageF()
    im.setScale(1.)
    np.testing.assert_raises(ValueError, galsim.InterpolatedImage, im)
    # Weird flux normalization
    im = galsim.ImageF(5, 5)
    im.setScale(1.)
    np.testing.assert_raises(ValueError, galsim.InterpolatedImage, im, normalization = 'foo')
    # Weird interpolant - give it something random like a GSObject
    np.testing.assert_raises(RuntimeError, galsim.InterpolatedImage, im, interpolant = g)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

#def test_operations():
# To test: +, *, /, scaleFlux, setFlux, applyTransformation, applyDilation, applyMagnification,
# applyShear, applyRotation, applyShift
#
#def test_real():
# To test: nyquistDx, centroid, xValue, isAnalyticX, isAxisymmetric, hasHardEdges
#
#def test_fourier()
# To test: maxK, stepK, isAnalyticK, kValue

if __name__ == "__main__":
    test_roundtrip()
    test_fluxnorm()
    test_exceptions()
#    test_operations()
#    test_real()
#    test_fourier()
