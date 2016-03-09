# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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

# A couple helper functions that should be in the BaseWCS class, but it's not.
# So for now implement it here, and mokey patch it in.
def _transformShear(shear, jac):
    # We're basically trying to commute the matrices J and S.
    # J S -> S' J
    # S' = J S J^-1
    J = jac.getMatrix()
    Jinv = jac.inverse().getMatrix()
    S = J.dot(shear.getMatrix()).dot(Jinv)
    Sjac = galsim.JacobianWCS(*S.flatten())
    scale, new_shear, theta, flip = Sjac.getDecomposition()
    return theta, new_shear

def shearToWorld(self, shear, image_pos=None, world_pos=None):
    """Convert a shear in image coordinates into the corresponding effect in world coordinates.

    If the input shear is applied to an object and then drawn onto an image with unit pixel scale,
    then the result is equivalent to converting both to world coordinates, applying the 
    transformation there, and then drawing onto an image with this wcs.

    The subtlety here is that the appropriate transformation is not just a shear.  Now it is
    both a rotation and a shear.  Hence the return is a tuple (theta, shear) which are the 
    transformations to apply in world coordinates (in that order).

        >>> profile = ...
        >>> shear = ...
        >>> wcs = ... 
        >>> im1 = profile.shear(shear).drawImage(scale=1.)
        >>> world_profile = wcs.toWorld(profile)
        >>> world_theta, world_shear = shearToWorld(wcs, shear)
        >>> im2 = world_profile.rotate(world_theta).shear(world_shear).drawImage(wcs=wcs)
        >>> assert im1 == im2   # At least within numerical precision.
    """
    # Use a helper function, since the other direction will be almost the same.
    jac = self.jacobian(image_pos, world_pos)
    return _transformShear(shear, jac)

def shearToImage(self, shear, image_pos=None, world_pos=None):
    """Convert a shear in world coordinates into the corresponding effect in image coordinates.

    If the input shear is applied to an object and then drawn onto an image with this wcs,
    then the result is equivalent to converting both to image coordinates, applying the 
    shear there, and then drawing onto an image with unit pixel scale.

        >>> profile = ...
        >>> shear = ...
        >>> wcs = ... 
        >>> im1 = profile.shear(shear).drawImage(wcs=wcs)
        >>> image_profile = wcs.toImage(profile)
        >>> image_theta, image_shear = shearToImage(wcs, shear)
        >>> im2 = image_profile.rotate(image_theta).shear(image_shear).drawImage(scale=1.)
        >>> assert im1 == im2   # At least within numerical precision.
    """
    # This one is the same method, but using the inverse jacobian.
    jac = self.jacobian(image_pos, world_pos).inverse()
    return _transformShear(shear, jac)


galsim.BaseWCS.shearToImage = shearToImage
galsim.BaseWCS.shearToWorld = shearToWorld

def test_metacal_tracking():
    """Test that the noise tracking works for the metacal use case involving deconvolution and
    reconvolution by almost the same PSF.

    This test is similar to the above test_uncorrelated_noise_tracking, except the modifications 
    are based on what is done for the metacal procedure.
    """
    t1 = time.time()
    import math

    def check_noise(noise_image, noise, msg):
        # A helper function to check that the current noise in the image is properly described
        # by the given CorrelatedNoise object
        noise2 = galsim.CorrelatedNoise(noise_image)
        print 'noise = ',noise
        print 'noise2 = ',noise2
        np.testing.assert_almost_equal(noise.getVariance(), noise2.getVariance(), decimal=2,
                                       err_msg=msg + ': variance does not match.')
        cf_im1 = galsim.Image(8,8, wcs=noise_image.wcs)
        cf_im2 = cf_im1.copy()
        noise.drawImage(image=cf_im1)
        noise2.drawImage(image=cf_im2)
        #print 'cf_im1 = ',cf_im1.array
        #print 'cf_im2 = ',cf_im2.array
        np.testing.assert_almost_equal(cf_im1.array, cf_im2.array, decimal=2,
                                       err_msg=msg + ': image of cf does not match.')

    def check_symm_noise(noise_image, msg):
        # A helper funciton to see if a noise image has 4-fold symmetric noise.
        im2 = noise_image.copy()
        # Clear out any wcs to make the test simpler
        im2.wcs = galsim.PixelScale(1.)
        noise = galsim.CorrelatedNoise(im2)
        cf = noise.drawImage(galsim.Image(bounds=galsim.BoundsI(-1,1,-1,1), scale=1))
        #print 'noise cf = ',cf
        # First check the variance
        print 'variance: ',cf(0,0), noise.getVariance()
        np.testing.assert_almost_equal(cf(0,0), noise.getVariance(), decimal=4,
                                       err_msg=msg + ':: noise variance is wrong.')
        cf_plus = [ cf(1,0), cf(-1,0), cf(0,1), cf(0,-1) ]
        cf_cross = [ cf(1,1), cf(-1,-1), cf(-1,1), cf(1,-1) ]
        print 'plus pattern: ',cf_plus
        print 'relative to mean: ',cf_plus/np.mean(cf_plus)
        print 'cross pattern: ',cf_cross
        print 'relative to mean: ',cf_cross/np.mean(cf_cross)
        # For now, don't make these asserts.  Just print whether they will pass or fail.
        if True:
            if np.all(np.abs(cf_plus/np.mean(cf_plus)-1) < 0.01):
                print 'plus test passes'
            else:
                print '*** FAIL ***'
                print msg + ': plus pattern is not constant'
            if np.all(np.abs(cf_cross/np.mean(cf_cross)-1) < 0.01):
                print 'cross test passes'
            else:
                print '*** FAIL ***'
                print msg + ': cross pattern is not constant'
        else:
            np.testing.assert_almost_equal(cf_plus/np.mean(cf_plus), 1.0, decimal=2,
                                           err_msg=msg + ': plus pattern is not constant')
            np.testing.assert_almost_equal(cf_cross/np.mean(cf_cross), 1.0, decimal=2,
                                           err_msg=msg + ': cross pattern is not constant')
 
    noise_var = 1.3
    #seed = 1234567  # For use as a unit test, we need a specific seed
    seed = 0  # During testing, it's useful to see how numbers flop around to know if 
              # something is systematic or random.
    im_size = 1024
    dg = 0.1  # This is bigger than metacal would use, but it makes the test easier.
    rng = galsim.BaseDeviate(seed)
    # Use a non-trivial wcs...
    # The first two don't work yet (even for strategy 3)
    # The last two are working.
    #wcs = galsim.JacobianWCS(0.26, 0.03, 0.08, -0.21)
    #wcs = galsim.JacobianWCS(0.26, 0.03, 0.03, 0.26)
    wcs = galsim.JacobianWCS(0.26, 0.03, -0.03, 0.26)
    #wcs = galsim.PixelScale(0.26)
    psf = galsim.Gaussian(fwhm=0.9)
    psf_target = psf.dilate(1. + 2.*dg)

    # First make an image of pure (white) Gaussian noise
    # Normally, there would be a galaxy in this image, but for the tests, we just have noise.
    obs_image = galsim.Image(im_size,im_size, init_value=0, wcs=wcs)
    obs_image.addNoise(galsim.GaussianNoise(rng=rng, sigma=math.sqrt(noise_var)))

    # The noise on this image should be describable as an UncorrelatedNoise object:
    noise = galsim.UncorrelatedNoise(variance=noise_var, wcs=wcs)
    check_noise(obs_image, noise, 'initial UncorrelatedNoise model is wrong')

    # Make an InterpolatedImage profile to use for manipulating this image
    #noise_pad_size = 4.*im_size / wcs.minLinearScale()
    #ii = galsim.InterpolatedImage(obs_image, noise_pad_size=noise_pad_size, noise_pad=noise_var, pad_factor=1)
    ii = galsim.InterpolatedImage(obs_image, pad_factor=1)
    ii.noise = noise

    # If we draw it back, the attached noise attribute should still be correct
    check_noise(ii.drawImage(obs_image.copy(),method='no_pixel'), ii.noise,
                'noise model is wrong for InterpolatedImage')

    # Here is the metacal process for which we want to understand the noise.
    # We'll try a few different methods.
    shear = galsim.Shear(g1=dg)
    sheared_obj = galsim.Convolve(ii, galsim.Deconvolve(psf)).shear(shear)
    final_obj = galsim.Convolve(psf_target, sheared_obj)
    final_image = final_obj.drawImage(obs_image.copy(), method='no_pixel')

    try:
        check_symm_noise(final_image, 'Initial image')
        #didnt_fail = True
        # This bit doesn't work while we are not actually raising exceptions in check_symm_noise
        # So we expect to see **FAIL** text at this point.
        print 'The above tests are expected to **FAIL**.  This is not a problem.'
        didnt_fail = False
    except AssertionError as e:
        print 'As expected initial image fails symmetric noise test:'
        print e
        didnt_fail = False
    if didnt_fail:
        assert False, 'Initial image was expected to fail symmetric noise test, but passed.'

    if True:
        print '\n\nStrategy 1:'
        # Strategy 1: Use the noise attribute attached to ii and use it to either whiten or
        #             symmetrize the noise in the final image.
        # Note: The check_noise tests fail.  I think because the convolve and deconvolve impose
        #       a maxk = that of the psf.  Which is too small for an accurate rendering of the
        #       correlation function (here just an autocorrelation of a Pixel.
        #       Strangely, even with a manifestly inaccurate description of the correlated noise,
        #       the symmetrize step does successfully symmetrize the noise.  At least to the 
        #       accuracy I'm testing it here.  Whitening on the other hand doesn't work.

        # First, deconvolve and reconvolve by the same PSF:
        test_obj = galsim.Convolve([ii, psf, galsim.Deconvolve(psf)])
        # This fails...
        #check_noise(test_obj.drawImage(obs_image.copy(), method='no_pixel'), test_obj.noise,
                    #'noise model is wrong after convolve/deconvolve by psf')

        # Now use a slightly dilated PSF for the reconvolution:
        test_obj = galsim.Convolve([ii, psf_target, galsim.Deconvolve(psf)])
        #check_noise(test_obj.drawImage(obs_image.copy(), method='no_pixel'), test_obj.noise,
                    #'noise model is wrong for dilated target psf')

        # Finally, include the shear step.  This was done above with sheared_obj, final_obj.
        #check_noise(final_image, final_obj.noise,
                    #'noise model is wrong when including small shear')

        # If we whiten using this noise model, we should get back to white noise.
        t3 = time.time()
        final_image2 = final_image.copy()  # Don't clobber the original
        final_var = final_image2.whitenNoise(final_obj.noise)
        t4 = time.time()
        print 'Noise tracking method with whiten: final_var = ',final_var
        print 'Check: direct variance = ',np.var(final_image2.array)
        check_symm_noise(final_image2, 'noise whitening does not work')
        print 'Time for noise tracking with whiten = ',t4-t3

        # Using symmetrizeNoise should add less noise, but also work.
        t3 = time.time()
        final_image2 = final_image.copy()
        final_var = final_image2.symmetrizeNoise(final_obj.noise)
        t4 = time.time()
        print 'Noise tracking method with symmetrize: final_var = ',final_var
        print 'Check: direct variance = ',np.var(final_image2.array)
        # Oddly, this one works, despite the previous ones failing...
        check_symm_noise(final_image2, 'noise symmetrizing does not work')
        print 'Time for noise tracking with symmetrize = ',t4-t3


    if True:
        print '\n\nStrategy 2:'
        # Strategy 2: Don't trust the noise tracking. Track a noise image through the same process
        #             and then measure the noise from that image.  Use it to either whiten or
        #             symmetrize the noise in the final image.
        # Note: This method doesn't work.  Even the symmetrize option.  Also, it seems to always
        #       add more noise than the noise tracking method above.

        # Make another noise image, since we don't actually have access to a pure noise image
        # for real objects.  But we should be able to estimate the variance in the image.
        t3 = time.time()
        noise_image = galsim.Image(im_size,im_size, init_value=0, wcs=wcs)
        noise_image.addNoise(galsim.GaussianNoise(rng=rng, sigma=math.sqrt(noise_var)))
        noise_ii = galsim.InterpolatedImage(noise_image, pad_factor=1)
        sheared_noise_obj = galsim.Convolve(noise_ii, galsim.Deconvolve(psf)).shear(shear)
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

        print 'Direct noise method with whiten: final_var = ',final_var
        # Neither of these work currently, so maybe a bug in the whitening code?
        # Or possibly in my logic here.
        check_symm_noise(final_image2, 'whitening the noise using direct noise model failed')
        print 'Time for direct noise with whitening = ',t5-t3

        t6 = time.time()
        final_image2 = final_image.copy()
        final_var = final_image2.symmetrizeNoise(noise)
        t7 = time.time()

        print 'Direct noise method with symmetrize: final_var = ',final_var
        check_symm_noise(final_image2, 'symmetrizing the noise using direct noise model failed')
        print 'Time for direct noise with symmetrizing = ',t7-t6 + t4-t3

    if True:
        print '\n\nStrategy 3:'
        # Strategy 3: Make a noise field and do the same operations as we do to the main image
        #             except use the opposite shear value.  Add this noise field to the final
        #             image to get a white noise field.
        # Note: This method works!  But not for non-trivial wcs yet.
        t3 = time.time()

        # Make another noise image, since we don't actually have access to a pure noise image
        # for real objects.  But we should be able to estimate the variance in the image.
        rev_image = galsim.Image(im_size,im_size, init_value=0, wcs=wcs)
        rev_image.addNoise(galsim.GaussianNoise(rng=rng, sigma=math.sqrt(noise_var)))
        rev_ii = galsim.InterpolatedImage(rev_image, pad_factor=1)

        # Note: the "reverse" shear needs to take into account the wcs.  The shear is applied
        #       in sky coordinates, but we want the opposite shear in image coordinates.
        #local_theta, local_shear = wcs.shearToImage(galsim.Shear(shear))
        #rev_local_shear = -local_shear
        #rev_theta, rev_shear = wcs.shearToWorld(rev_local_shear)
        # I thought this would be simply transforming the shearToImage, taking -shear, 
        # and then going back with shearToWorld.  However, the rotations mess that up.
        # This result does not give the inverse shear in image coordinates.
        #
        # If J is the jacobian of the wcs, and S1 is the applied shear, then we want to find
        # S2 such that the net application in image coords has the same net rotation, but the 
        # opposite shear as S1.
        # i.e. if J^-1 S1 = m G R F
        # then we want to find S2, such that J^-1 S2 = m G^-1 R F
        # S2 = J G^-1 G^-1 (m G R F) = J G^-1 G^-1 J^-1 S1
        jac = wcs.jacobian()
        J = jac.getMatrix()
        Jinv = jac.inverse().getMatrix()
        print 'shear = ',shear
        S1 = shear.getMatrix()
        JinvS1 = Jinv.dot(S1)
        print 'JinvS1 = ',JinvS1
        print '= ',galsim.JacobianWCS(*JinvS1.flatten()).getDecomposition()
        _, net_shear, _, _ = galsim.JacobianWCS(*JinvS1.flatten()).getDecomposition()
        print 'net_shear = ',net_shear
        Ginv = (-net_shear).getMatrix()
        print 'S1 = ',S1
        print 'J = ',J
        print 'JinvS1 = ',JinvS1
        print 'Ginv = ',Ginv
        S2 = J.dot(Ginv).dot(Ginv).dot(JinvS1)
        print 'S2 = ',S2
        scale, rev_shear, rev_theta, flip = galsim.JacobianWCS(*S2.flatten()).getDecomposition()
        print '= ',scale,rev_theta,rev_shear,flip
        print 'rev_shear = ',rev_shear
        print 'JinvS2 = ',Jinv.dot(S2)
        print '= ',galsim.JacobianWCS(*Jinv.dot(S2).flatten()).getDecomposition()
        # Flip should be False, and scale should be essentially 1.0.
        assert flip == False
        assert np.abs(scale - 1.) < 1.e-8

        rev_sheared_obj = galsim.Convolve(rev_ii, galsim.Deconvolve(psf)).rotate(rev_theta).shear(rev_shear)
        #rev_sheared_obj = galsim.Convolve(rev_ii, galsim.Deconvolve(psf)).shear(rev_shear)
        rev_final_obj = galsim.Convolve(psf_target, rev_sheared_obj)
        rev_final_image = rev_final_obj.drawImage(obs_image.copy(), method='no_pixel')

        # Add the reverse-sheared noise image to the original image.
        final_image2 = final_image - rev_final_image
        t4 = time.time()

        # The noise variance in the end should be 2x as large as the original
        final_var = np.var(final_image2.array)
        print 'Reverse shear method: final_var = ',final_var
        check_symm_noise(final_image2, 'using reverse shear does not work')
        print 'Time for reverse shear method = ',t4-t3

    if False:
        print '\n\nStrategy 4:'
        # Strategy 4: Make a noise field and do the same operations as we do to the main image
        #             Subtract off the difference between this and the original noise field.
        # Note: I don't think this method should ever work (i.e. I don't think the failure
        #       here points to a bug in anything), but it's included here since I tried it,
        #       and Eric had thought it should work.
        t3 = time.time()

        noise_image = galsim.Image(im_size,im_size, init_value=0, wcs=wcs)
        noise_image.addNoise(galsim.GaussianNoise(rng=rng, sigma=math.sqrt(noise_var)))
        noise_ii = galsim.InterpolatedImage(noise_image, pad_factor=1)
        deconv_noise_obj = galsim.Convolve(noise_ii, galsim.Deconvolve(psf))
        sheared_noise_obj = deconv_noise_obj.shear(shear)
        final_noise_obj = galsim.Convolve(psf_target, sheared_noise_obj)
        noshear_noise_obj = galsim.Convolve(psf_target, deconv_noise_obj)
        final_noise_image = final_noise_obj.drawImage(obs_image.copy(), method='no_pixel')
        noshear_noise_image = noshear_noise_obj.drawImage(obs_image.copy(), method='no_pixel')

        # Add the difference between the sheared and unsheared noise fields.
        final_image2 = final_image.copy()
        final_image2 += (noshear_noise_image - final_noise_image)
        t4 = time.time()

        final_var = np.var(final_image2.array)
        print 'd(noise) method: final_var = ',final_var
        check_symm_noise(final_image2, 'subtracting d(noise) does not work')
        print 'Time for d(noise) method = ',t4-t3

    t2 = time.time()
    print 'total time for tests = %.2f'%(t2-t1)

def test_wcs_convert_shear():
    """Test the routines wcs.shearToImage and wcs.shearToWorld"""

    # Code taken directly from the doc string of wcs.shearToImage:
    # Simple, but with ellipticity and a shift.
    profile = galsim.Exponential(half_light_radius=0.2).shear(e1=0.3, e2=-0.4).shift(0.01,-0.03)
    # Largish to make sure we notice if operations are done in the wrong order.
    shear = galsim.Shear(g1=-0.3, g2=0.2)
    # Something complicated.  Including a flip.
    wcs = galsim.JacobianWCS(0.26, 0.04, -0.09, -0.21)

    nx=48
    ny=48
    im1 = profile.shear(shear).drawImage(nx=nx, ny=ny, wcs=wcs)
    print 'world profile = ',profile.shear(shear)
    print 'world->image = ',wcs.toImage(profile.shear(shear))
    image_profile = wcs.toImage(profile)
    image_theta, image_shear = shearToImage(wcs, shear)
    print 'local = ',image_profile.rotate(image_theta).shear(image_shear)
    im2 = image_profile.rotate(image_theta).shear(image_shear).drawImage(nx=nx, ny=ny, scale=1.)
    np.testing.assert_almost_equal(im1.array, im2.array, decimal=12,
                                   err_msg="wcs.shearToImage didn't work correctly")

    # Now check the reverse process, shearToWorld.
    im1 = profile.shear(shear).drawImage(nx=nx, ny=ny, scale=1.)
    print 'image profile = ',profile.shear(shear)
    print 'image->world = ',wcs.toWorld(profile.shear(shear))
    world_profile = wcs.toWorld(profile)
    world_theta, world_shear = shearToWorld(wcs, shear)
    print 'world = ',world_profile.rotate(world_theta).shear(world_shear)
    im2 = world_profile.rotate(world_theta).shear(world_shear).drawImage(nx=nx, ny=ny, wcs=wcs)
    np.testing.assert_almost_equal(im1.array, im2.array, decimal=12,
                                   err_msg="wcs.shearToWorld didn't work correctly")
 
if __name__ == "__main__":
    test_metacal_tracking()
    test_wcs_convert_shear()
