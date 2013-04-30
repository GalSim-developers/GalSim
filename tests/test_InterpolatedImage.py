# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
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
        interp = galsim.InterpolatedImage(image_in, x_interpolant=quint_2d, dx=test_dx, flux=1.)
        do_shoot(interp,image_out,"InterpolatedImage")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_fluxnorm():
    """Test that InterpolatedImage class responds properly to instructions about flux normalization.
    """
    import time
    t1 = time.time()

    # define values
    # Note that im_lin_scale should be even, since the auto-sized draw() command always produces
    # an even-sized image.  If the even/odd-ness doesn't match then the interpolant will come 
    # into play, and the exact checks will fail.
    im_lin_scale = 6 # make an image with this linear scale
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
    np.testing.assert_almost_equal(total_flux, interp.getFlux(), decimal=9,
                            err_msg='Did not keep flux normalization')
    # Check that this is preserved when drawing
    im2 = interp.draw(dx = im_scale)
    np.testing.assert_almost_equal(total_flux, im2.array.sum(), decimal=9,
                                   err_msg='Drawn image does not have expected flux normalization')

    # Now make an InterpolatedImage but tell it sb normalization
    interp_sb = galsim.InterpolatedImage(im, normalization = 'sb')
    # Check that when drawing, the sum is equal to what we expect if the original image had been
    # surface brightness
    im3 = interp_sb.draw(dx = im_scale)
    np.testing.assert_almost_equal(total_flux*(im_scale**2)/im3.array.sum(), 1.0, decimal=6,
                                   err_msg='Did not use surface brightness normalization')
    # Check that when drawing with sb normalization, the sum is the same as the original
    im4 = interp_sb.draw(dx = im_scale, normalization = 'sb')
    np.testing.assert_almost_equal(total_flux/im4.array.sum(), 1.0, decimal=6,
                                   err_msg='Failed roundtrip for sb normalization')

    # Finally make an InterpolatedImage but give it some other flux value
    interp_flux = galsim.InterpolatedImage(im, flux=test_flux)
    # Check that it has that flux
    np.testing.assert_equal(test_flux, interp_flux.getFlux(),
                            err_msg = 'InterpolatedImage did not use flux keyword')
    # Check that this is preserved when drawing
    im5 = interp_flux.draw(dx = im_scale)
    np.testing.assert_almost_equal(test_flux/im5.array.sum(), 1.0, decimal=6,
                                   err_msg = 'Drawn image does not reflect flux keyword')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_exceptions():
    """Test failure modes for InterpolatedImage class.
    """
    import time
    t1 = time.time()

    try:
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
        np.testing.assert_raises(Exception, galsim.InterpolatedImage, im, x_interpolant = g)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_operations_simple():
    """Simple test of operations on InterpolatedImage: shear, magnification, rotation, shifting."""
    import time
    t1 = time.time()

    # Make some nontrivial image that can be described in terms of sums and convolutions of
    # GSObjects.  We want this to be somewhat hard to describe, but should be at least
    # critically-sampled, so put in an Airy PSF.
    gal_flux = 1000.
    pix_scale = 0.03 # arcsec
    bulge_frac = 0.3
    bulge_hlr = 0.3 # arcsec
    bulge_e = 0.15
    bulge_pos_angle = 30.*galsim.degrees
    disk_hlr = 0.6 # arcsec
    disk_e = 0.5
    disk_pos_angle = 60.*galsim.degrees
    lam = 800              # nm    NB: don't use lambda - that's a reserved word.
    tel_diam = 2.4         # meters
    lam_over_diam = lam * 1.e-9 / tel_diam # radians
    lam_over_diam *= 206265  # arcsec
    im_size = 512

    bulge = galsim.Sersic(4, half_light_radius=bulge_hlr)
    bulge.applyShear(e=bulge_e, beta=bulge_pos_angle)
    disk = galsim.Exponential(half_light_radius = disk_hlr)
    disk.applyShear(e=disk_e, beta=disk_pos_angle)
    gal = bulge_frac*bulge + (1.-bulge_frac)*disk
    gal.setFlux(gal_flux)
    psf = galsim.Airy(lam_over_diam)
    pix = galsim.Pixel(pix_scale)
    obj = galsim.Convolve(gal, psf, pix)
    im = obj.draw(dx=pix_scale)

    # Turn it into an InterpolatedImage with default param settings
    int_im = galsim.InterpolatedImage(im)

    # Shear it, and compare with expectations from GSObjects directly
    test_g1=-0.07
    test_g2=0.1
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    comp_region=30 # compare the central region of this linear size
    test_int_im = int_im.createSheared(g1=test_g1, g2=test_g2)
    ref_obj = obj.createSheared(g1=test_g1, g2=test_g2)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.draw(image=im, dx=pix_scale)
    ref_obj.draw(image=ref_im, dx=pix_scale)
    # define subregion for comparison
    new_bounds = galsim.BoundsI(1,comp_region,1,comp_region)
    new_bounds.shift((im_size-comp_region)/2, (im_size-comp_region)/2)
    im_sub = im.subImage(new_bounds)
    ref_im_sub = ref_im.subImage(new_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Sheared InterpolatedImage disagrees with reference')

    # Magnify it, and compare with expectations from GSObjects directly
    test_mag = 1.08
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    comp_region=30 # compare the central region of this linear size
    test_int_im = int_im.createMagnified(test_mag)
    ref_obj = obj.createMagnified(test_mag)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.draw(image=im, dx=pix_scale)
    ref_obj.draw(image=ref_im, dx=pix_scale)
    # define subregion for comparison
    new_bounds = galsim.BoundsI(1,comp_region,1,comp_region)
    new_bounds.shift((im_size-comp_region)/2, (im_size-comp_region)/2)
    im_sub = im.subImage(new_bounds)
    ref_im_sub = ref_im.subImage(new_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Magnified InterpolatedImage disagrees with reference')

    # Lens it (shear and magnify), and compare with expectations from GSObjects directly
    test_g1 = -0.03
    test_g2 = -0.04
    test_mag = 0.74
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    comp_region=30 # compare the central region of this linear size
    test_int_im = int_im.createLensed(test_g1, test_g2, test_mag)
    ref_obj = obj.createLensed(test_g1, test_g2, test_mag)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.draw(image=im, dx=pix_scale)
    ref_obj.draw(image=ref_im, dx=pix_scale)
    # define subregion for comparison
    new_bounds = galsim.BoundsI(1,comp_region,1,comp_region)
    new_bounds.shift((im_size-comp_region)/2, (im_size-comp_region)/2)
    im_sub = im.subImage(new_bounds)
    ref_im_sub = ref_im.subImage(new_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Lensed InterpolatedImage disagrees with reference')

    # Rotate it, and compare with expectations from GSObjects directly
    test_rot_angle = 32.*galsim.degrees
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    comp_region=30 # compare the central region of this linear size
    test_int_im = int_im.createRotated(test_rot_angle)
    ref_obj = obj.createRotated(test_rot_angle)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.draw(image=im, dx=pix_scale)
    ref_obj.draw(image=ref_im, dx=pix_scale)
    # define subregion for comparison
    new_bounds = galsim.BoundsI(1,comp_region,1,comp_region)
    new_bounds.shift((im_size-comp_region)/2, (im_size-comp_region)/2)
    im_sub = im.subImage(new_bounds)
    ref_im_sub = ref_im.subImage(new_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Rotated InterpolatedImage disagrees with reference')

    # Shift it, and compare with expectations from GSObjects directly
    x_shift = -0.31
    y_shift = 0.87
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    comp_region=30 # compare the central region of this linear size
    test_int_im = int_im.createShifted(x_shift, y_shift)
    ref_obj = obj.createShifted(x_shift, y_shift)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.draw(image=im, dx=pix_scale)
    ref_obj.draw(image=ref_im, dx=pix_scale)
    # define subregion for comparison
    new_bounds = galsim.BoundsI(1,comp_region,1,comp_region)
    new_bounds.shift((im_size-comp_region)/2, (im_size-comp_region)/2)
    im_sub = im.subImage(new_bounds)
    ref_im_sub = ref_im.subImage(new_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Shifted InterpolatedImage disagrees with reference')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_operations():
    """Test of operations on complicated InterpolatedImage: shear, magnification, rotation, shifting."""
    import time
    t1 = time.time()
    test_decimal = 3

    # Make some nontrivial image
    im = galsim.fits.read('./real_comparison_images/test_images.fits') # read in first real galaxy
                                                                       # in test catalog
    int_im = galsim.InterpolatedImage(im)
    orig_mom = im.FindAdaptiveMom()

    # Magnify by some amount and make sure change is as expected
    mag_scale = 0.92
    new_int_im = int_im.createMagnified(mag_scale)
    test_im = galsim.ImageF(im.bounds)
    new_int_im.draw(image = test_im, dx = im.getScale())
    new_mom = test_im.FindAdaptiveMom()
    np.testing.assert_almost_equal(new_mom.moments_sigma/np.sqrt(mag_scale),
        orig_mom.moments_sigma, test_decimal,
        err_msg = 'Size of magnified InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.observed_shape.e1, orig_mom.observed_shape.e1,
        test_decimal,
        err_msg = 'e1 of magnified InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.observed_shape.e2, orig_mom.observed_shape.e2,
        test_decimal,
        err_msg = 'e2 of magnified InterpolatedImage from HST disagrees with expectations')

    # Shift, make sure change in moments is as expected
    x_shift = 0.92
    y_shift = -0.16
    new_int_im = int_im.createShifted(x_shift, y_shift)
    test_im = galsim.ImageF(im.bounds)
    new_int_im.draw(image = test_im, dx = im.getScale())
    new_mom = test_im.FindAdaptiveMom()
    np.testing.assert_almost_equal(new_mom.moments_sigma, orig_mom.moments_sigma,
        test_decimal,
        err_msg = 'Size of shifted InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.moments_centroid.x-x_shift, orig_mom.moments_centroid.x,
        test_decimal,
        err_msg = 'x centroid of shifted InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.moments_centroid.y-y_shift, orig_mom.moments_centroid.y,
        test_decimal,
        err_msg = 'y centroid of shifted InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.observed_shape.e1, orig_mom.observed_shape.e1,
        test_decimal,
        err_msg = 'e1 of shifted InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.observed_shape.e2, orig_mom.observed_shape.e2,
        test_decimal,
        err_msg = 'e2 of shifted InterpolatedImage from HST disagrees with expectations')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_uncorr_padding():
    """Test for uncorrelated noise padding of InterpolatedImage."""
    import time
    t1 = time.time()

    # Set up some defaults: use weird image sizes / shapes and noise variances.
    decimal_precise=5
    decimal_coarse=2
    orig_nx = 147
    orig_ny = 174
    noise_var = 1.73
    big_nx = 259
    big_ny = 282
    orig_seed = 151241

    # first, make a noise image
    orig_img = galsim.ImageF(orig_nx, orig_ny)
    orig_img.setScale(1.)
    gd = galsim.GaussianDeviate(orig_seed, mean=0., sigma=np.sqrt(noise_var))
    orig_img.addNoise(galsim.DeviateNoise(gd))

    # make it into an InterpolatedImage with some zero-padding
    # (note that default is zero-padding, by factors of several)
    int_im = galsim.InterpolatedImage(orig_img)
    # draw into a larger image
    big_img = galsim.ImageF(big_nx, big_ny)
    int_im.draw(big_img, dx=1.)
    # check that variance is diluted by expected amount - should be exact, so check precisely!
    # Note that this only works if the big image has the same even/odd-ness in the two sizes.
    # Otherwise the center of the original image will fall between pixels in the big image.
    # Then the variance will be smoothed somewhat by the interpolant.
    big_var_expected = np.var(orig_img.array)*float(orig_nx*orig_ny)/(big_nx*big_ny)
    np.testing.assert_almost_equal(
        np.var(big_img.array), big_var_expected, decimal=decimal_precise,
        err_msg='Variance not diluted by expected amount when zero-padding')

    # make it into an InterpolatedImage with noise-padding
    int_im = galsim.InterpolatedImage(orig_img, noise_pad=noise_var,
                                      rng = galsim.GaussianDeviate(orig_seed))
    # draw into a larger image
    big_img = galsim.ImageF(big_nx, big_ny)
    int_im.draw(big_img, dx=1.)
    # check that variance is same as original - here, we cannot be too precise because the padded
    # region is not huge and the comparison will be, well, noisy.
    np.testing.assert_almost_equal(
        np.var(big_img.array), noise_var, decimal=decimal_coarse,
        err_msg='Variance not correct after padding image with noise')

    # check that if we pass in a RNG, it is actually used to pad with the same noise field
    # basically, redo all of the above steps and draw into a new image, make sure it's the same as
    # previous.
    int_im = galsim.InterpolatedImage(orig_img, noise_pad=noise_var,
                                      rng = galsim.GaussianDeviate(orig_seed))
    big_img_2 = galsim.ImageF(big_nx, big_ny)
    int_im.draw(big_img_2, dx=1.)
    np.testing.assert_array_almost_equal(
        big_img_2.array, big_img.array, decimal=decimal_precise,
        err_msg='Cannot reproduce noise-padded image with same choice of seed')

    # Finally check inputs: what if we give it an input variance that is neg?  A list?
    try:
        np.testing.assert_raises(ValueError,galsim.InterpolatedImage,orig_img,noise_pad=-1.)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_corr_padding():
    """Test for correlated noise padding of InterpolatedImage."""
    import time
    t1 = time.time()

    # Set up some defaults for tests.
    decimal_precise=4
    decimal_coarse=2
    imgfile = 'blankimg.fits'
    orig_nx = 187
    orig_ny = 164
    big_nx = 319
    big_ny = 322
    orig_seed = 151241

    # Read in some small image of a noise field from HST.
    # Rescale it to have a decently large amplitude for the purpose of doing these tests.
    im = 1.e2*galsim.fits.read(imgfile)
    # Make a CorrrlatedNoise out of it.
    cn = galsim.CorrelatedNoise(galsim.BaseDeviate(orig_seed), im)

    # first, make a noise image
    orig_img = galsim.ImageF(orig_nx, orig_ny)
    orig_img.setScale(1.)
    orig_img.addNoise(cn)

    # make it into an InterpolatedImage with some zero-padding
    # (note that default is zero-padding, by factors of several)
    int_im = galsim.InterpolatedImage(orig_img)
    # draw into a larger image
    big_img = galsim.ImageF(big_nx, big_ny)
    int_im.draw(big_img, dx=1.)
    # check that variance is diluted by expected amount - should be exact, so check precisely!
    big_var_expected = np.var(orig_img.array)*float(orig_nx*orig_ny)/(big_nx*big_ny)
    np.testing.assert_almost_equal(np.var(big_img.array), big_var_expected, decimal=decimal_precise,
        err_msg='Variance not diluted by expected amount when zero-padding')

    # make it into an InterpolatedImage with noise-padding
    int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                      noise_pad = im)

    # draw into a larger image
    big_img = galsim.ImageF(big_nx, big_ny)
    int_im.draw(big_img, dx=1.)
    # check that variance is same as original - here, we cannot be too precise because the padded
    # region is not huge and the comparison will be, well, noisy.
    np.testing.assert_almost_equal(np.var(big_img.array), np.var(orig_img.array),
        decimal=decimal_coarse,
        err_msg='Variance not correct after padding image with correlated noise')

    # check that if we pass in a RNG, it is actually used to pad with the same noise field
    # basically, redo all of the above steps and draw into a new image, make sure it's the same as
    # previous.
    int_im = galsim.InterpolatedImage(
        orig_img, rng=galsim.GaussianDeviate(orig_seed), noise_pad=cn)
    big_img_2 = galsim.ImageF(big_nx, big_ny)
    int_im.draw(big_img_2, dx=1.)
    np.testing.assert_array_almost_equal(big_img_2.array, big_img.array, decimal=decimal_precise,
        err_msg='Cannot reproduce correlated noise-padded image with same choice of seed')

    # Finally, check inputs:
    # what if we give it a screwy way of defining the image padding?
    try:
        np.testing.assert_raises(ValueError,galsim.InterpolatedImage,orig_img,noise_pad=-1.)
    except ImportError:
        print 'The assert_raises tests require nose'
    # also, check that whether we give it a string, image, or cn, it gives the same noise field
    # (given the same random seed)
    infile = 'blankimg.fits'
    inimg = galsim.fits.read(infile)
    incf = galsim.CorrelatedNoise(galsim.GaussianDeviate(), inimg) # input RNG will be ignored below
    int_im2 = galsim.InterpolatedImage(orig_img, rng=galsim.GaussianDeviate(orig_seed),
                                       noise_pad=inimg)
    int_im3 = galsim.InterpolatedImage(orig_img, rng=galsim.GaussianDeviate(orig_seed),
                                       noise_pad=incf)
    big_img2 = galsim.ImageF(big_nx, big_ny)
    big_img3 = galsim.ImageF(big_nx, big_ny)
    int_im2.draw(big_img2, dx=1.)
    int_im3.draw(big_img3, dx=1.)
    np.testing.assert_equal(big_img2.array, big_img3.array,
                            err_msg='Diff ways of specifying correlated noise give diff answers')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_roundtrip()
    test_fluxnorm()
    test_exceptions()
    test_operations_simple()
    test_operations()
    test_uncorr_padding()
    test_corr_padding()
