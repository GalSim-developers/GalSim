# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import time
import numpy as np

import galsim
from galsim_test_helpers import *

VAR_NDECIMAL=4
CHECKNOISE_NDECIMAL=2

@timer
def test_metacal_tracking():
    """Test that the noise tracking works for the metacal use case involving deconvolution and
    reconvolution by almost the same PSF.

    This test is similar to the above test_uncorrelated_noise_tracking, except the modifications
    are based on what is done for the metacal procedure.
    """
    import math
    def check_noise(noise_image, noise, msg):
        # A helper function to check that the current noise in the image is properly described
        # by the given CorrelatedNoise object
        noise2 = galsim.CorrelatedNoise(noise_image)
        print('noise = ',noise)
        print('noise2 = ',noise2)
        np.testing.assert_almost_equal(noise.getVariance(), noise2.getVariance(),
                                       decimal=CHECKNOISE_NDECIMAL,
                                       err_msg=msg + ': variance does not match.')
        cf_im1 = galsim.Image(8,8, wcs=noise_image.wcs)
        cf_im2 = cf_im1.copy()
        noise.drawImage(image=cf_im1)
        noise2.drawImage(image=cf_im2)
        np.testing.assert_almost_equal(cf_im1.array, cf_im2.array,
                                       decimal=CHECKNOISE_NDECIMAL,
                                       err_msg=msg + ': image of cf does not match.')

    def check_symm_noise(noise_image, msg):
        # A helper funciton to see if a noise image has 4-fold symmetric noise.
        im2 = noise_image.copy()
        # Clear out any wcs to make the test simpler
        im2.wcs = galsim.PixelScale(1.)
        noise = galsim.CorrelatedNoise(im2)
        cf = noise.drawImage(galsim.Image(bounds=galsim.BoundsI(-1,1,-1,1), scale=1))
        # First check the variance
        print('variance: ',cf(0,0), noise.getVariance())
        np.testing.assert_almost_equal(cf(0,0)/noise.getVariance(), 1.0, decimal=VAR_NDECIMAL,
                                       err_msg=msg + ':: noise variance is wrong.')
        cf_plus = np.array([ cf(1,0), cf(-1,0), cf(0,1), cf(0,-1) ])
        cf_cross = np.array([ cf(1,1), cf(-1,-1), cf(-1,1), cf(1,-1) ])
        print('plus pattern: ',cf_plus)
        print('diff relative to dc: ',(cf_plus-np.mean(cf_plus))/cf(0,0))
        print('cross pattern: ',cf_cross)
        print('diff relative to dc: ',(cf_cross-np.mean(cf_cross))/cf(0,0))
        # For now, don't make these asserts.  Just print whether they will pass or fail.
        if True:
            if np.all(np.abs((cf_plus-np.mean(cf_plus))/cf(0,0)) < 0.01):
                print('plus test passes')
            else:
                print('*** FAIL ***')
                print(msg + ': plus pattern is not constant')
            if np.all(np.abs((cf_cross-np.mean(cf_cross))/cf(0,0)) < 0.01):
                print('cross test passes')
            else:
                print('*** FAIL ***')
                print(msg + ': cross pattern is not constant')
        else:
            np.testing.assert_almost_equal((cf_plus-np.mean(cf_plus))/cf(0,0), 0.0, decimal=2,
                                           err_msg=msg + ': plus pattern is not constant')
            np.testing.assert_almost_equal((cf_cross-np.mean(cf_cross))/cf(0,0), 0.0, decimal=2,
                                           err_msg=msg + ': cross pattern is not constant')

    seed = 1234567  # For use as a unit test, we need a specific seed
    #seed = 0  # During testing, it's useful to see how numbers flop around to know if
               # something is systematic or random.
    rng = galsim.BaseDeviate(seed)

    noise_var = 1.3
    im_size = 1024
    dg = 0.1  # This is bigger than metacal would use, but it makes the test easier.

    # Use a non-trivial wcs...
    #wcs = galsim.JacobianWCS(0.26, 0.04, -0.04, -0.26)  # Rotation + flip.  No shear.
    wcs = galsim.JacobianWCS(0.26, 0.03, 0.08, -0.21)   # Fully complex
    #dudx =  0.12*0.26
    #dudy =  1.10*0.26
    #dvdx = -0.915*0.26
    #dvdy = -0.04*0.26
    #wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)  # Even more extreme

    # And an asymmetric PSF
    #orig_psf = galsim.Gaussian(fwhm=0.9).shear(g1=0.05, g2=0.03)
    # This one is small enough not to be fully Nyquist sampled, which makes things harder.
    orig_psf = galsim.Gaussian(fwhm=0.7).shear(g1=0.05, g2=0.03)

    # pixel is the pixel in world coords
    pixel = wcs.toWorld(galsim.Pixel(scale=1))
    pixel_inv = galsim.Deconvolve(pixel)

    psf_image = orig_psf.drawImage(nx=im_size, ny=im_size, wcs=wcs)

    # Metacal only has access to the PSF as an image, so use this from here on.
    psf = galsim.InterpolatedImage(psf_image)
    psf_nopix = galsim.Convolve([psf, pixel_inv])
    psf_inv = galsim.Deconvolve(psf)

    # Not what is done currently, but using a smoother target PSF helps make sure the
    # zeros in the deconvolved PSF get adequately zeroed out.
    def get_target_psf(psf):
        dk = 0.1              # The resolution in k space for the KImage
        small_kval = 1.e-2    # Find the k where the given psf hits this kvalue
        smaller_kval = 3.e-3  # Target PSF will have this kvalue at the same k

        kim = psf.drawKImage(scale=dk)
        karr_r = kim.real.array
        # Find the smallest r where the kval < small_kval
        nk = karr_r.shape[0]
        kx, ky = np.meshgrid(np.arange(-nk/2,nk/2), np.arange(-nk/2,nk/2))
        ksq = (kx**2 + ky**2) * dk**2
        ksq_max = np.min(ksq[karr_r < small_kval * psf.flux])

        # We take our target PSF to be the (round) Gaussian that is even smaller at this ksq
        # exp(-0.5 * ksq_max * sigma_sq) = smaller_kval
        sigma_sq = -2. * np.log(smaller_kval) / ksq_max
        return galsim.Gaussian(sigma = np.sqrt(sigma_sq))

    # The target PSF dilates the part without the pixel, but reconvolve by the real pixel.
    #psf_target_nopix = psf_nopix.dilate(1. + 2.*dg)
    #psf_target_nopix = orig_psf.dilate(1. + 4.*dg)
    psf_target_nopix = get_target_psf(psf_nopix.shear(g1=dg))
    print('PSF target HLR = ',psf_target_nopix.calculateHLR())
    print('PSF target FWHM = ',psf_target_nopix.calculateFWHM())
    psf_target = galsim.Convolve([psf_target_nopix, pixel])

    # Make an image of pure (white) Gaussian noise
    # Normally, there would be a galaxy in this image, but for the tests, we just have noise.
    obs_image = galsim.Image(im_size, im_size, init_value=0, wcs=wcs)
    obs_image.addNoise(galsim.GaussianNoise(rng=rng, sigma=math.sqrt(noise_var)))

    # The noise on this image should be describable as an UncorrelatedNoise object:
    noise = galsim.UncorrelatedNoise(variance=noise_var, wcs=wcs)
    check_noise(obs_image, noise, 'initial UncorrelatedNoise model is wrong')

    # Make an InterpolatedImage profile to use for manipulating this image
    # We can get away with no padding here, since our image is so large, but normally, you would
    # probably want to pad this with noise padding.
    ii = galsim.InterpolatedImage(obs_image, pad_factor=1)
    ii.noise = noise

    # If we draw it back, the attached noise attribute should still be correct
    check_noise(ii.drawImage(obs_image.copy(),method='no_pixel'), ii.noise,
                'noise model is wrong for InterpolatedImage')

    # Here is the metacal process for which we want to understand the noise.
    # We'll try a few different methods.
    shear = galsim.Shear(g1=dg)
    sheared_obj = galsim.Convolve(ii, psf_inv).shear(shear)
    final_obj = galsim.Convolve(psf_target, sheared_obj)
    final_image = final_obj.drawImage(obs_image.copy(), method='no_pixel')

    try:
        check_symm_noise(final_image, 'Initial image')
        #didnt_fail = True
        # This bit doesn't work while we are not actually raising exceptions in check_symm_noise
        # So we expect to see **FAIL** text at this point.
        print('The above tests are expected to **FAIL**.  This is not a problem.')
        didnt_fail = False
    except AssertionError as e:
        print('As expected initial image fails symmetric noise test:')
        print(e)
        didnt_fail = False
    if didnt_fail:
        assert False, 'Initial image was expected to fail symmetric noise test, but passed.'

    if True:
        print('\n\nStrategy 1:')
        # Strategy 1: Use the noise attribute attached to ii and use it to either whiten or
        #             symmetrize the noise in the final image.
        # Note: The check_noise tests fail.  I think because the convolve and deconvolve impose
        #       a maxk = that of the psf.  Which is too small for an accurate rendering of the
        #       correlation function (here just an autocorrelation of a Pixel.
        # The whiten tests kind of work, but they add a lot of extra noise.  Much more than
        # strategy 4 below.  So the level of correlation remaining is pretty well below the
        # dc variance.  Symmetrize doesn't add much noise, but the residual correlation is about
        # the same, which means it doesn't pass the test relative to the lower dc variance.

        # First, deconvolve and reconvolve by the same PSF:
        test_obj = galsim.Convolve([ii, psf, psf_inv])
        # This fails...
        if False:
            check_noise(test_obj.drawImage(obs_image.copy(), method='no_pixel'), test_obj.noise,
                        'noise model is wrong after convolve/deconvolve by psf')

        # Now use a slightly dilated PSF for the reconvolution:
        test_obj = galsim.Convolve([ii, psf_target, psf_inv])
        if False:
            check_noise(test_obj.drawImage(obs_image.copy(), method='no_pixel'), test_obj.noise,
                        'noise model is wrong for dilated target psf')

        # Finally, include the shear step.  This was done above with sheared_obj, final_obj.
        if False:
            check_noise(final_image, final_obj.noise,
                        'noise model is wrong when including small shear')

        # If we whiten using this noise model, we should get back to white noise.
        t3 = time.time()
        final_image2 = final_image.copy()  # Don't clobber the original
        final_var = final_image2.whitenNoise(final_obj.noise)
        t4 = time.time()
        print('Noise tracking method with whiten: final_var = ',final_var)
        print('Check: direct variance = ',np.var(final_image2.array))
        check_symm_noise(final_image2, 'noise whitening does not work')
        print('Time for noise tracking with whiten = ',t4-t3)

        # Using symmetrizeNoise should add less noise, but also work.
        t3 = time.time()
        final_image2 = final_image.copy()
        final_var = final_image2.symmetrizeNoise(final_obj.noise)
        t4 = time.time()
        print('Noise tracking method with symmetrize: final_var = ',final_var)
        print('Check: direct variance = ',np.var(final_image2.array))
        check_symm_noise(final_image2, 'noise symmetrizing does not work')
        print('Time for noise tracking with symmetrize = ',t4-t3)


    if True:
        print('\n\nStrategy 2:')
        # Strategy 2: Don't trust the noise tracking. Track a noise image through the same process
        #             and then measure the noise from that image.  Use it to either whiten or
        #             symmetrize the noise in the final image.
        # Note: This method doesn't work any better.  The added noise for whitening is even more
        # than strategy 1.  And the remaining correlations are still similarly significant for the
        # symmetrize version.  A little smaller than strategy 1, but not enough to pass our tests.

        # Make another noise image, since we don't actually have access to a pure noise image
        # for real objects.  But we should be able to estimate the variance in the image.
        t3 = time.time()
        noise_image = galsim.Image(im_size,im_size, init_value=0, wcs=wcs)
        noise_image.addNoise(galsim.GaussianNoise(rng=rng, sigma=math.sqrt(noise_var)))
        noise_ii = galsim.InterpolatedImage(noise_image, pad_factor=1)
        sheared_noise_obj = galsim.Convolve(noise_ii, psf_inv).shear(shear)
        final_noise_obj = galsim.Convolve(psf_target, sheared_noise_obj)
        final_noise_image = final_noise_obj.drawImage(obs_image.copy(), method='no_pixel')

        # Use this to construct an appropriate CorrelatedNoise object
        noise = galsim.CorrelatedNoise(final_noise_image)
        t4 = time.time()
        final_image2 = final_image.copy()
        final_var = final_image2.whitenNoise(noise)
        t5 = time.time()

        check_noise(final_noise_image, noise,
                    'noise model is wrong when direct measuring the final noise image')

        print('Direct noise method with whiten: final_var = ',final_var)
        # Neither of these work currently, so maybe a bug in the whitening code?
        # Or possibly in my logic here.
        check_symm_noise(final_image2, 'whitening the noise using direct noise model failed')
        print('Time for direct noise with whitening = ',t5-t3)

        t6 = time.time()
        final_image2 = final_image.copy()
        final_var = final_image2.symmetrizeNoise(noise)
        t7 = time.time()

        print('Direct noise method with symmetrize: final_var = ',final_var)
        check_symm_noise(final_image2, 'symmetrizing the noise using direct noise model failed')
        print('Time for direct noise with symmetrizing = ',t7-t6 + t4-t3)

    if False:
        print('\n\nStrategy 3:')
        # Strategy 3: Make a noise field and do the same operations as we do to the main image
        #             except use the opposite shear value.  Add this noise field to the final
        #             image to get a symmetric noise field.
        # Note: This method works!  But only for square pixels.  However, they may be rotated
        # or flipped. Just not sheared.
        # Update: I think this method won't ever work for non-square pixels.  The reason it works
        # for square pixels is that in that case, it is equivalent to strategy 4.
        t3 = time.time()

        # Make another noise image
        rev_image = galsim.Image(im_size,im_size, init_value=0, wcs=wcs)
        rev_image.addNoise(galsim.GaussianNoise(rng=rng, sigma=math.sqrt(noise_var)))
        rev_ii = galsim.InterpolatedImage(rev_image, pad_factor=1)

        rev_sheared_obj = galsim.Convolve(rev_ii, psf_inv).shear(-shear)
        rev_final_obj = galsim.Convolve(psf_target, rev_sheared_obj)
        rev_final_image = rev_final_obj.drawImage(obs_image.copy(), method='no_pixel')

        # Add the reverse-sheared noise image to the original image.
        final_image2 = final_image + rev_final_image
        t4 = time.time()

        # The noise variance in the end should be 2x as large as the original
        final_var = np.var(final_image2.array)
        print('Reverse shear method: final_var = ',final_var)
        check_symm_noise(final_image2, 'using reverse shear does not work')
        print('Time for reverse shear method = ',t4-t3)

    if True:
        print('\n\nStrategy 4:')
        # Strategy 4: Make a noise field and do the same operations as we do to the main image,
        #             then rotate it by 90 degress and add it to the final image.
        # This method works!  Even for an arbitrarily sheared wcs.
        t3 = time.time()

        # Make another noise image
        noise_image = galsim.Image(im_size,im_size, init_value=0, wcs=wcs)
        noise_image.addNoise(galsim.GaussianNoise(rng=rng, sigma=math.sqrt(noise_var)))
        noise_ii = galsim.InterpolatedImage(noise_image, pad_factor=1)

        noise_sheared_obj = galsim.Convolve(noise_ii, psf_inv).shear(shear)
        noise_final_obj = galsim.Convolve(psf_target, noise_sheared_obj)
        noise_final_image = noise_final_obj.drawImage(obs_image.copy(), method='no_pixel')

        # Rotate the image by 90 degrees
        rot_noise_final_image = galsim.Image(np.ascontiguousarray(np.rot90(noise_final_image.array)))

        # Add the rotated noise image to the original image.
        final_image2 = final_image + rot_noise_final_image
        t4 = time.time()

        # The noise variance in the end should be 2x as large as the original
        final_var = np.var(final_image2.array)
        print('Rotate image method: final_var = ',final_var)
        check_symm_noise(final_image2, 'using rotated noise image does not work')
        print('Time for rotate image method = ',t4-t3)

    if False:
        print('\n\nStrategy 5:')
        # Strategy 5: The same as strategy 3, except we target the effective net transformation
        #             done by strategy 4.
        # I think this strategy probably can't work for non-square pixels, because the shear
        # happens before the convolution by the PSF.  And if the wcs is non-square, then the
        # PSF is sheared relative to the pixels.  That shear isn't being accounted for here,
        # so the net result isn't equivalent to rotating by 90 degrees at the end.
        t3 = time.time()

        # Make another noise image
        rev_image = galsim.Image(im_size,im_size, init_value=0, wcs=wcs)
        rev_image.addNoise(galsim.GaussianNoise(rng=rng, sigma=math.sqrt(noise_var)))
        rev_ii = galsim.InterpolatedImage(rev_image, pad_factor=1)

        # Find the effective transformation to apply in sky coordinates that matches what
        # you would get by applying the shear in sky coords, going to image coords and then
        # rotating by 90 degrees.
        #
        # If J is the jacobian of the wcs, and S1 is the applied shear, then we want to find
        # S2 such that J^-1 S2 = R90 J^-1 S1
        jac = wcs.jacobian()
        J = jac.getMatrix()
        Jinv = jac.inverse().getMatrix()
        S1 = shear.getMatrix()
        R90 = np.array([[0,-1],[1,0]])
        S2 = J.dot(R90).dot(Jinv).dot(S1)
        scale, rev_shear, rev_theta, flip = galsim.JacobianWCS(*S2.flatten()).getDecomposition()
        # Flip should be False, and scale should be essentially 1.0.
        assert flip == False
        assert abs(scale - 1.) < 1.e-8

        rev_sheared_obj = galsim.Convolve(rev_ii, psf_inv).rotate(rev_theta).shear(rev_shear)
        rev_final_obj = galsim.Convolve(psf_target, rev_sheared_obj)
        rev_final_image = rev_final_obj.drawImage(obs_image.copy(), method='no_pixel')

        # Add the reverse-sheared noise image to the original image.
        final_image2 = final_image + rev_final_image
        t4 = time.time()

        # The noise variance in the end should be 2x as large as the original
        final_var = np.var(final_image2.array)
        print('Alternate reverse shear method: final_var = ',final_var)
        check_symm_noise(final_image2, 'using alternate reverse shear does not work')
        print('Time for alternate reverse shear method = ',t4-t3)

@timer
def test_wcs():
    """Reproduce an error Erin found and reported in #834.  This was never an error in a released
    version, just temporarily on master, but the mistake hadn't been caught by any of our unit
    tests, so this test catches the error.
    """
    wcs = galsim.JacobianWCS(0.01, -0.26, -0.28, -0.03)
    gal = galsim.Exponential(half_light_radius=1.1, flux=237)
    psf = galsim.Moffat(beta=3.5, half_light_radius=0.9)
    obs = galsim.Convolve(gal,psf)

    obs_im = obs.drawImage(nx=32, ny=32, offset=(0.3,-0.2), wcs=wcs)
    psf_im = psf.drawImage(nx=32, ny=32, wcs=wcs)

    ii = galsim.InterpolatedImage(obs_im)
    psf_ii = galsim.InterpolatedImage(psf_im)
    psf_inv = galsim.Deconvolve(psf_ii)

    ii_nopsf = galsim.Convolve(ii, psf_inv)

    newpsf=galsim.Moffat(beta=3.5, half_light_radius=0.95)
    newpsf=newpsf.dilate(1.02)

    new_ii = ii_nopsf.shear(g1=0.01,g2=0.0)
    new_ii = galsim.Convolve(new_ii, newpsf)

    new_im = new_ii.drawImage(image=obs_im.copy(), method='no_pixel')
    np.testing.assert_almost_equal(new_im.array.sum()/237, obs_im.array.sum()/237, decimal=1)

if __name__ == "__main__":
    runtests(__file__)
