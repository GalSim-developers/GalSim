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

import numpy as np
import os
import sys
import logging
import math

import galsim
from galsim_test_helpers import *


@timer
def test_gaussian():
    """Test the Gaussian noise builder
    """
    scale = 0.3
    sigma = 17.3

    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
            'pixel_scale' : scale,
            'size' : 32,

            'noise' : {
                'type' : 'Gaussian',
                'sigma' : sigma,
            }
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 1.1,
            'flux' : 100,
        },
    }

    # First build by hand
    first_seed = galsim.BaseDeviate(1234).raw()
    rng = galsim.BaseDeviate(first_seed + 1)
    gal = galsim.Gaussian(sigma=1.1, flux=100)
    im1a = gal.drawImage(nx=32, ny=32, scale=scale)
    var = sigma**2
    im1a.addNoise(galsim.GaussianNoise(rng, sigma))

    # Compare to what config builds
    im1b = galsim.config.BuildImage(config)
    np.testing.assert_equal(im1b.array, im1a.array)

    # Check noise variance
    var = sigma**2
    var1 = galsim.config.CalculateNoiseVariance(config)
    np.testing.assert_equal(var1, var)
    var2 = galsim.Image(3,3, dtype=float)
    galsim.config.AddNoiseVariance(config, var2)
    np.testing.assert_almost_equal(var2.array, var)

    # Check include_obj_var=True, which shouldn't do anything different in this case
    var3 = galsim.Image(32,32, dtype=float)
    galsim.config.AddNoiseVariance(config, var3, include_obj_var=True)
    np.testing.assert_almost_equal(var3.array, var)

    # Gaussian noise can also be given the variance directly, rather than sigma
    galsim.config.RemoveCurrent(config)
    del config['image']['noise']['sigma']
    del config['image']['noise']['_get']
    config['image']['noise']['variance'] = var
    im1c = galsim.config.BuildImage(config)
    np.testing.assert_equal(im1c.array, im1a.array)

    # Base class usage is invalid
    builder = galsim.config.noise.NoiseBuilder()
    assert_raises(NotImplementedError, builder.addNoise, config, config, im1a, rng, var,
                 draw_method='auto', logger=None)
    assert_raises(NotImplementedError, builder.getNoiseVariance, config, config)


@timer
def test_poisson():
    """Test the Poisson noise builder
    """
    scale = 0.3
    sky = 200

    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
            'pixel_scale' : scale,
            'size' : 32,

            'noise' : {
                'type' : 'Poisson',
                'sky_level' : sky,
            }
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 1.1,
            'flux' : 100,
        },
    }

    # First build by hand
    first_seed = galsim.BaseDeviate(1234).raw()
    rng = galsim.BaseDeviate(first_seed + 1)
    gal = galsim.Gaussian(sigma=1.1, flux=100)
    im1a = gal.drawImage(nx=32, ny=32, scale=scale)
    sky_pixel = sky * scale**2
    im1a.addNoise(galsim.PoissonNoise(rng, sky_level=sky_pixel))

    # Compare to what config builds
    im1b = galsim.config.BuildImage(config)
    np.testing.assert_equal(im1b.array, im1a.array)

    # Check noise variance
    var1 = galsim.config.CalculateNoiseVariance(config)
    np.testing.assert_equal(var1, sky_pixel)
    var2 = galsim.Image(3,3)
    galsim.config.AddNoiseVariance(config, var2)
    np.testing.assert_almost_equal(var2.array, sky_pixel)

    # Check include_obj_var=True
    var3 = galsim.Image(32,32)
    galsim.config.AddNoiseVariance(config, var3, include_obj_var=True)
    np.testing.assert_almost_equal(var3.array, sky_pixel + im1a.array)

    # Repeat using photon shooting, which needs to do something slightly different, since the
    # signal photons already have shot noise.
    rng.seed(first_seed + 1)
    im2a = gal.drawImage(nx=32, ny=32, scale=scale, method='phot', rng=rng)
    # Need to add Poisson noise for the sky, but not the signal (which already has shot noise)
    im2a.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=sky_pixel)))
    im2a -= sky_pixel

    # Compare to what config builds
    galsim.config.RemoveCurrent(config)
    config['image']['draw_method'] = 'phot'  # Make sure it gets copied over to stamp properly.
    del config['stamp']['draw_method']
    del config['stamp']['_done']
    im2b = galsim.config.BuildImage(config)
    np.testing.assert_equal(im2b.array, im2a.array)

    # Check non-trivial sky image
    galsim.config.RemoveCurrent(config)
    config['image']['sky_level'] = sky
    config['image']['wcs'] =  {
        'type' : 'UVFunction',
        'ufunc' : '0.05*x + 0.001*x**2',
        'vfunc' : '0.05*y + 0.001*y**2',
    }
    del config['image']['pixel_scale']
    del config['wcs']
    rng.seed(first_seed+1)
    wcs = galsim.UVFunction(ufunc='0.05*x + 0.001*x**2', vfunc='0.05*y + 0.001*y**2')
    im3a = gal.drawImage(nx=32, ny=32, wcs=wcs, method='phot', rng=rng)
    sky_im = galsim.Image(im3a.bounds, wcs=wcs)
    wcs.makeSkyImage(sky_im, sky)
    im3a += sky_im  # Add 1 copy of the raw sky image for image[sky]
    noise_im = sky_im.copy()
    noise_im *= 2.  # Now 2x because the noise includes both in image[sky] and noise[sky]
    noise_im.addNoise(galsim.PoissonNoise(rng))
    noise_im -= 2.*sky_im
    im3a += noise_im
    im3b = galsim.config.BuildImage(config)
    np.testing.assert_almost_equal(im3b.array, im3a.array, decimal=6)

    # With tree rings, the sky includes them as well.
    config['image']['sensor'] = {
        'type' : 'Silicon',
        'treering_func' : {
            'type' : 'File',
            'file_name' : 'tree_ring_lookup.dat',
            'amplitude' : 0.5
        },
        'treering_center' : {
            'type' : 'XY',
            'x' : 0,
            'y' : -500
        }
    }
    galsim.config.RemoveCurrent(config)
    config = galsim.config.CleanConfig(config)
    rng.seed(first_seed+1)
    trfunc = galsim.LookupTable.from_file('tree_ring_lookup.dat', amplitude=0.5)
    sensor = galsim.SiliconSensor(treering_func=trfunc, treering_center=galsim.PositionD(0,-500),
                                  rng=rng)
    im4a = gal.drawImage(nx=32, ny=32, wcs=wcs, method='phot', rng=rng, sensor=sensor)
    sky_im = galsim.Image(im3a.bounds, wcs=wcs)
    wcs.makeSkyImage(sky_im, sky)
    areas = sensor.calculate_pixel_areas(sky_im, use_flux=False)
    sky_im *= areas
    im4a += sky_im
    noise_im = sky_im.copy()
    noise_im *= 2.
    noise_im.addNoise(galsim.PoissonNoise(rng))
    noise_im -= 2.*sky_im
    im4a += noise_im
    im4b = galsim.config.BuildImage(config)
    np.testing.assert_almost_equal(im4b.array, im4a.array, decimal=6)

    # Can't have both sky_level and sky_level_pixel
    config['image']['noise']['sky_level_pixel'] = 2000.
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # Must have a valid noise type
    del config['image']['noise']['sky_level_pixel']
    config['image']['noise']['type'] = 'Invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # noise must be a dict
    config['image']['noise'] = 'Invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # Can't have signal_to_noise and  flux
    config['image']['noise'] = { 'type' : 'Poisson', 'sky_level' : sky }
    config['gal']['signal_to_noise'] = 100
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # This should work
    del config['gal']['flux']
    galsim.config.BuildImage(config)

    # These now hit the errors in CalculateNoiseVariance rather than AddNoise
    config['image']['noise']['type'] = 'Invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['noise'] = 'Invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    del config['image']['noise']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # If rather than signal_to_noise, we have an extra_weight output, then it hits
    # a different error.
    config['gal']['flux'] = 100
    del config['gal']['signal_to_noise']
    config['output'] = { 'weight' : {} }
    config['image']['noise'] = { 'type' : 'Poisson', 'sky_level' : sky }
    galsim.config.SetupExtraOutput(config)
    galsim.config.SetupConfigFileNum(config, 0, 0, 0)
    # This should work again.
    galsim.config.BuildImage(config)
    config['image']['noise']['type'] = 'Invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['noise'] = 'Invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)


@timer
def test_ccdnoise():
    """Test that the config layer CCD noise adds noise consistent with using a CCDNoise object.
    """
    import logging

    gain = 4
    sky = 50
    rn = 5
    size = 2048

    # Use this to turn on logging, but more info than we noramlly need, so generally leave it off.
    #logging.basicConfig(format="%(message)s", level=logging.DEBUG, stream=sys.stdout)
    #logger = logging.getLogger()
    logger = None

    config = {}
    config['gal'] = { 'type' : 'None' }
    config['image'] = {
        'type' : 'Single',
        'size' : size,
        'pixel_scale' : 0.3,
        'random_seed' : 123
    }
    config['image']['noise'] = {
        'type' : 'CCD',
        'sky_level_pixel' : sky,
        'gain' : gain,
        'read_noise' : rn
    }
    image = galsim.config.BuildImage(config,logger=logger)

    print('config-built image: ',np.mean(image.array),np.var(image.array.astype(float)))
    test_var = np.var(image.array.astype(float))

    # Build another image that should have equivalent noise properties.
    image2 = galsim.Image(size, size, scale=0.3, dtype=float)
    first_seed = galsim.BaseDeviate(123).raw()
    rng = galsim.BaseDeviate(first_seed+1)
    noise = galsim.CCDNoise(rng=rng, gain=gain, read_noise=rn)
    image2 += sky
    image2.addNoise(noise)
    image2 -= sky

    print('manual sky: ',np.mean(image2.array),np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var,
                                   err_msg="CCDNoise with manual sky failed variance test.")

    # So far this isn't too stringent of a test, since the noise module will use a CCDNoise
    # object for this.  In fact, it should do precisely the same calculation.
    # This should be equivalent to letting CCDNoise take the sky level:
    image2.fill(0)
    rng.reset(first_seed+1)
    noise = galsim.CCDNoise(rng=rng, sky_level=sky, gain=gain, read_noise=rn)
    image2.addNoise(noise)

    print('sky done by CCDNoise: ',np.mean(image2.array),np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var,
                                   err_msg="CCDNoise using sky failed variance test.")

    # Check that the CCDNoiseBuilder calculates the same variance as CCDNoise
    var1 = noise.getVariance()
    var2 = galsim.config.CalculateNoiseVariance(config)
    np.testing.assert_almost_equal(var1, var2,
                                   err_msg="CCDNoiseBuilder calculates the wrong variance")

    # Finally, the config layer also includes its own manual implementation of CCD noise that
    # it uses when there is already some noise in the image.  We want to check that this is
    # consistent with the regular CCDNoise object.

    # This time, we just set the current_var to 1.e-20 to trigger the alternate path, but
    # without any real noise there yet.
    image2.fill(0)
    rng.reset(first_seed+1)
    config['image_num_rng'] = rng
    galsim.config.AddNoise(config, image2, current_var=1.e-20, logger=logger)

    print('with negligible current_var: ',np.mean(image2.array),np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var,
                                   err_msg="CCDNoise with current_var failed variance test.")

    # Here we pre-load the full read noise and tell it it's there with current_var
    image2.fill(0)
    gn = galsim.GaussianNoise(rng=rng, sigma=rn/gain)
    image2.addNoise(gn)
    galsim.config.AddNoise(config, image2, current_var=(rn/gain)**2, logger=logger)

    print('current_var == read_noise: ',np.mean(image2.array),np.var(image2.array))
    # So far we've done this to very high accuracy, since we've been using the same rng seed,
    # so the results should be identical, not just close.  However, hereon the values are just
    # close, since they are difference noise realizations.  So check to 1 decimal place.
    np.testing.assert_almost_equal(np.var(image2.array),test_var, decimal=1,
                                   err_msg="CCDNoise w/ current_var==rn failed variance test.")

    # Now we pre-load part of the read-noise, but not all.  It should add the rest as read_noise.
    image2.fill(0)
    gn = galsim.GaussianNoise(rng=rng, sigma=0.5*rn/gain)
    image2.addNoise(gn)
    galsim.config.AddNoise(config, image2, current_var=(0.5*rn/gain)**2, logger=logger)

    print('current_var < read_noise: ',np.mean(image2.array),np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var, decimal=1,
                                   err_msg="CCDNoise w/ current_var < rn failed variance test.")

    # Last, we go beyond the read-noise, so it should remove some of the sky level to compensate.
    image2.fill(0)
    gn = galsim.GaussianNoise(rng=rng, sigma=2.*rn/gain)
    image2.addNoise(gn)
    galsim.config.AddNoise(config, image2, current_var=(2.*rn/gain)**2, logger=logger)

    print('current_var > read_noise',np.mean(image2.array),np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var, decimal=1,
                                   err_msg="CCDNoise w/ current_var > rn failed variance test.")

@timer
def test_ccdnoise_phot():
    """CCDNoise has some special code for photon shooting, so check that it works correctly.
    """
    scale = 0.3
    sky = 200
    gain = 1.8
    rn = 2.3

    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
            'pixel_scale' : scale,
            'size' : 32,
            'draw_method' : 'phot',

            'noise' : {
                'type' : 'CCD',
                'gain' : gain,
                'read_noise' : rn,
                'sky_level' : sky,
            }
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 1.1,
            'flux' : 100,
        },
    }

    # First build by hand
    first_seed = galsim.BaseDeviate(1234).raw()
    rng = galsim.BaseDeviate(first_seed + 1)
    gal = galsim.Gaussian(sigma=1.1, flux=100)
    im1a = gal.drawImage(nx=32, ny=32, scale=scale, method='phot', rng=rng)
    sky_pixel = sky * scale**2
    # Need to add Poisson noise for the sky, but not the signal (which already has shot noise)
    im1a *= gain
    im1a.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=sky_pixel * gain)))
    im1a /= gain
    im1a -= sky_pixel
    im1a.addNoise(galsim.GaussianNoise(rng, sigma=rn/gain))

    # Compare to what config builds
    im1b = galsim.config.BuildImage(config)
    np.testing.assert_equal(im1b.array, im1a.array)

    # Check noise variance
    var = sky_pixel / gain + rn**2 / gain**2
    var1 = galsim.config.CalculateNoiseVariance(config)
    np.testing.assert_equal(var1, var)
    var2 = galsim.Image(3,3)
    galsim.config.AddNoiseVariance(config, var2)
    np.testing.assert_almost_equal(var2.array, var)

    # Check include_obj_var=True
    var3 = galsim.Image(32,32)
    galsim.config.AddNoiseVariance(config, var3, include_obj_var=True)
    np.testing.assert_almost_equal(var3.array, var + im1a.array/gain)

    # Some slightly different code paths if rn = 0 or gain = 1:
    del config['image']['noise']['gain']
    del config['image']['noise']['read_noise']
    del config['image']['noise']['_get']
    rng.seed(first_seed + 1)
    im2a = gal.drawImage(nx=32, ny=32, scale=scale, method='phot', rng=rng)
    im2a.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=sky_pixel)))
    im2a -= sky_pixel
    im2b = galsim.config.BuildImage(config)
    np.testing.assert_equal(im2b.array, im2a.array)
    var5 = galsim.config.CalculateNoiseVariance(config)
    np.testing.assert_equal(var5, sky_pixel)
    var6 = galsim.Image(3,3)
    galsim.config.AddNoiseVariance(config, var6)
    np.testing.assert_almost_equal(var6.array, sky_pixel)
    var7 = galsim.Image(32,32)
    galsim.config.AddNoiseVariance(config, var7, include_obj_var=True)
    np.testing.assert_almost_equal(var7.array, sky_pixel + im2a.array)

    # Check non-trivial sky image
    galsim.config.RemoveCurrent(config)
    config['image']['sky_level'] = sky
    config['image']['wcs'] =  {
        'type' : 'UVFunction',
        'ufunc' : '0.05*x + 0.001*x**2',
        'vfunc' : '0.05*y + 0.001*y**2',
    }
    del config['image']['pixel_scale']
    del config['wcs']
    rng.seed(first_seed+1)
    wcs = galsim.UVFunction(ufunc='0.05*x + 0.001*x**2', vfunc='0.05*y + 0.001*y**2')
    im3a = gal.drawImage(nx=32, ny=32, wcs=wcs, method='phot', rng=rng)
    sky_im = galsim.Image(im3a.bounds, wcs=wcs)
    wcs.makeSkyImage(sky_im, sky)
    im3a += sky_im  # Add 1 copy of the raw sky image for image[sky]
    noise_im = sky_im.copy()
    noise_im *= 2.  # Now 2x because the noise includes both in image[sky] and noise[sky]
    noise_im.addNoise(galsim.PoissonNoise(rng))
    noise_im -= 2.*sky_im
    im3a += noise_im
    im3b = galsim.config.BuildImage(config)
    np.testing.assert_almost_equal(im3b.array, im3a.array, decimal=6)

    # And again with the rn and gain put back in.
    galsim.config.RemoveCurrent(config)
    config['image']['noise']['gain'] = gain
    config['image']['noise']['read_noise'] = rn
    del config['image']['noise']['_get']
    rng.seed(first_seed+1)
    im4a = gal.drawImage(nx=32, ny=32, wcs=wcs, method='phot', rng=rng)
    wcs.makeSkyImage(sky_im, sky)
    im4a += sky_im
    noise_im = sky_im.copy()
    noise_im *= 2. * gain
    noise_im.addNoise(galsim.PoissonNoise(rng))
    noise_im /= gain
    noise_im -= 2. * sky_im
    im4a += noise_im
    im4a.addNoise(galsim.GaussianNoise(rng, sigma=rn/gain))
    im4b = galsim.config.BuildImage(config)
    np.testing.assert_almost_equal(im4b.array, im4a.array, decimal=6)


@timer
def test_cosmosnoise():
    """Test that the config layer COSMOS noise works with keywords.
    """
    import logging

    logger = None

    pix_scale = 0.03
    random_seed = 123

    # First make the image using COSMOSNoise without kwargs.
    config = {}
    # Either gal or psf is required, but it can be type = None, which means don't draw anything.
    config['gal'] = { 'type' : 'None' }
    config['stamp'] = {
        'type' : 'Basic',
        'size' : 64
    }
    config['image'] = {
        'pixel_scale' : pix_scale,
        'random_seed' : 123
    }
    config['image']['noise'] = {
        'type' : 'COSMOS'
    }
    image = galsim.config.BuildStamp(config,logger=logger)[0]

    # Then make it using explicit kwargs to make sure they are getting passed through properly.
    config2 = {}
    config2['gal'] = config['gal']
    config2['stamp'] = {
        'type' : 'Basic',
        'xsize' : 64,  # Same thing, but cover the xsize, ysize options
        'ysize' : 64
    }
    config2['image'] = config['image']
    config2['image']['noise'] = {
        'type' : 'COSMOS',
        'file_name' : os.path.join(galsim.meta_data.share_dir,'acs_I_unrot_sci_20_cf.fits'),
        'cosmos_scale' : pix_scale
    }
    image2 = galsim.config.BuildStamp(config2,logger=logger)[0]

    # We used the same RNG and noise file / properties, so should get the same exact noise field.
    np.testing.assert_allclose(
        image.array, image2.array, rtol=1.e-5,
        err_msg='Config COSMOS noise does not reproduce results given kwargs')

    # Use the more generic Correlated noise type
    config3 = galsim.config.CopyConfig(config2)
    config3 = galsim.config.CleanConfig(config3)
    config3['image']['noise'] = {
        'type': 'Correlated',
        'file_name' : os.path.join(galsim.meta_data.share_dir,'acs_I_unrot_sci_20_cf.fits'),
        'pixel_scale' : pix_scale
    }
    image3 = galsim.config.BuildStamp(config3,logger=logger)[0]
    np.testing.assert_allclose(image3.array, image2.array,
        err_msg='Config Correlated noise not the same as COSMOS')

    # Use a RealGalaxy with whitening to make sure that it properly handles any current_var
    # in the image already.
    # Detects bug Rachel found in issue #792
    config['gal'] = {
        'type' : 'RealGalaxy',
        'index' : 79,
        # Use a small flux to make sure that whitening doesn't add more noise than we will
        # request from the COSMOS noise.  (flux = 0.1 is too high.)
        'flux' : 0.01
    }
    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = 'real_galaxy_catalog_23.5_example.fits'
    config['input'] = {
        'real_catalog' : {
            'dir' : real_gal_dir ,
            'file_name' : real_gal_cat
        }
    }
    config['image']['noise']['whiten'] = True
    galsim.config.ProcessInput(config)
    image3, current_var3 = galsim.config.BuildStamp(config, logger=logger)
    print('From BuildStamp, current_var = ',current_var3)

    # Build the same image by hand to make sure it matches what config drew.
    first_seed = galsim.BaseDeviate(123).raw()
    rng = galsim.BaseDeviate(first_seed+1)
    rgc = galsim.RealGalaxyCatalog(os.path.join(real_gal_dir, real_gal_cat))
    gal = galsim.RealGalaxy(rgc, index=79, flux=0.01, rng=rng)
    image4 = gal.drawImage(image=image3.copy())
    current_var4 = gal.noise.whitenImage(image4)
    print('After whitening, current_var = ',current_var4)
    noise = galsim.correlatednoise.getCOSMOSNoise(
            rng=rng,
            file_name=os.path.join(galsim.meta_data.share_dir,'acs_I_unrot_sci_20_cf.fits'),
            cosmos_scale=pix_scale)

    # Check that the COSMOSNoiseBuilder calculates the right variance.
    var1 = noise.getVariance()
    var2 = galsim.config.CalculateNoiseVariance(config)
    print('Full noise variance = ',noise.getVariance())
    print('From config.CalculateNoiseVar = ',var2)
    np.testing.assert_almost_equal(var2, var1, err_msg="COSMOSNoise calculated the wrong variance")

    # Finish whitening steps.
    np.testing.assert_equal(
        current_var3, noise.getVariance(),
        err_msg='Config COSMOS noise with whitening does not return the correct current_var')
    noise -= galsim.UncorrelatedNoise(current_var4, rng=rng, wcs=image4.wcs)
    print('After subtract current_var, noise variance = ',noise.getVariance())
    image4.addNoise(noise)
    np.testing.assert_equal(
        image3.array, image4.array,
        err_msg='Config COSMOS noise with whitening does not reproduce manually drawn image')

    # If CalculateNoiseVar happens before using the noise, there is a slightly different code
    # path, but it should return the same answer of course.
    del config['_current_cn_tag']
    var3 = galsim.config.CalculateNoiseVariance(config)
    print('From config.CalculateNoiseVar = ',var3)
    np.testing.assert_almost_equal(var3, var1, err_msg="COSMOSNoise calculated the wrong variance")
    del config3['_current_cn_tag']
    var3b = galsim.config.CalculateNoiseVariance(config3)
    np.testing.assert_almost_equal(var3b, var1, err_msg="CorrelatedNoise calculated the wrong variance")

    # AddNoiseVariance should add the variance to an image
    image5 = galsim.Image(32,32)
    galsim.config.AddNoiseVariance(config, image5)
    np.testing.assert_almost_equal(image5.array, var1,
                                   err_msg="AddNoiseVariance added the wrong variance")


@timer
def test_whiten():
    """Test the options in config to whiten images
    """
    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = 'real_galaxy_catalog_23.5_example.fits'
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
            'pixel_scale' : 0.05,
            'noise' : { 'type' : 'Gaussian', 'variance' : 50, },
        },
        'stamp' : {
            'type' : 'Basic',
            'size' : 32,
        },
        'gal' : {
            'type' : 'RealGalaxy',
            'index' : 79,
            'flux' : 100,
        },
        'psf' : {  # This is really slow if we don't convolve by a PSF.
            'type' : 'Gaussian',
            'sigma' : 0.05
        },
        'input' : {
            'real_catalog' : {
                'dir' : real_gal_dir ,
                'file_name' : real_gal_cat
            }
        }
    }

    # First build by hand (no whitening yet)
    first_seed = galsim.BaseDeviate(1234).raw()
    rng = galsim.BaseDeviate(first_seed + 1)
    rgc = galsim.RealGalaxyCatalog(os.path.join(real_gal_dir, real_gal_cat))
    gal = galsim.RealGalaxy(rgc, index=79, flux=100, rng=rng)
    psf = galsim.Gaussian(sigma=0.05)
    final = galsim.Convolve(gal,psf)
    im1a = final.drawImage(nx=32, ny=32, scale=0.05)

    # Compare to what config builds
    galsim.config.ProcessInput(config)
    im1b, cv1b = galsim.config.BuildStamp(config, do_noise=False)
    np.testing.assert_equal(cv1b, 0.)
    np.testing.assert_equal(im1b.array, im1a.array)

    # Now add whitening, but no noise yet.
    cv1a = final.noise.whitenImage(im1a)
    print('From whiten, current_var = ',cv1a)
    galsim.config.RemoveCurrent(config)
    config['image']['noise'] =  { 'whiten' : True, }
    im1c, cv1c = galsim.config.BuildStamp(config, do_noise=False)
    print('From BuildStamp, current_var = ',cv1c)
    # Occasionally these aren't precisely equal due to slight numerical rounding differences.
    np.testing.assert_allclose(cv1c, cv1a, rtol=1.e-15)
    np.testing.assert_allclose(im1c.array, im1a.array, rtol=1.e-15)
    rng1 = rng.duplicate()  # Save current state of rng

    # 1. Gaussian noise
    #####
    config['image']['noise'] =  {
        'type' : 'Gaussian',
        'variance' : 50,
        'whiten' : True,
    }
    galsim.config.RemoveCurrent(config)
    im2a = im1a.copy()
    im2a.addNoise(galsim.GaussianNoise(sigma=math.sqrt(50-cv1a), rng=rng))
    im2b, cv2b = galsim.config.BuildStamp(config)
    np.testing.assert_almost_equal(cv2b, 50)
    np.testing.assert_almost_equal(im2b.array, im2a.array, decimal=5)
    np.testing.assert_allclose(cv2b, np.var(im2b.array - im1a.array), rtol=0.1)

    # If whitening already added too much noise, raise an exception
    config['image']['noise']['variance'] = 1.e-5
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)

    # Can't have both whiten and symmetrize
    config['image']['noise']['variance'] = 50
    config['image']['noise']['symmetrize'] = 4
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)
    config['image']['noise']['symmetrize'] = False  # OK if false though.
    galsim.config.BuildStamp(config)

    # 2. Poisson noise
    #####
    config['image']['noise'] =  {
        'type' : 'Poisson',
        'sky_level_pixel' : 50,
        'whiten' : True,
    }
    galsim.config.RemoveCurrent(config)
    im3a = im1a.copy()
    sky = 50 - cv1a
    rng.reset(rng1.duplicate())
    im3a.addNoise(galsim.PoissonNoise(sky_level=sky, rng=rng))
    im3b, cv3b = galsim.config.BuildStamp(config)
    np.testing.assert_almost_equal(cv3b, 50, decimal=5)
    np.testing.assert_almost_equal(im3b.array, im3a.array, decimal=5)
    np.testing.assert_allclose(cv3b, np.var(im3b.array - im1a.array), rtol=0.1)

    # It's more complicated if the sky is quoted per arcsec and the wcs is not uniform.
    config2 = galsim.config.CopyConfig(config)
    galsim.config.RemoveCurrent(config2)
    config2['image']['sky_level'] = 100
    config2['image']['wcs'] =  {
        'type' : 'UVFunction',
        'ufunc' : '0.05*x + 0.001*x**2',
        'vfunc' : '0.05*y + 0.001*y**2',
    }
    del config2['image']['pixel_scale']
    del config2['wcs']
    config2['image']['noise']['symmetrize'] = 4 # Also switch to symmetrize, just to mix it up.
    del config2['image']['noise']['whiten']
    rng.reset(first_seed+1) # Start fresh, since redoing the whitening/symmetrizing
    wcs = galsim.UVFunction(ufunc='0.05*x + 0.001*x**2', vfunc='0.05*y + 0.001*y**2')
    im3c = galsim.Image(32,32, wcs=wcs)
    im3c = final.drawImage(im3c)
    cv3c = final.noise.symmetrizeImage(im3c,4)
    sky = galsim.Image(im3c.bounds, wcs=wcs)
    wcs.makeSkyImage(sky, 100)
    mean_sky = np.mean(sky.array)
    im3c += sky
    extra_sky = 50 - cv3c
    im3c.addNoise(galsim.PoissonNoise(sky_level=extra_sky, rng=rng))
    im3d, cv3d = galsim.config.BuildStamp(config2)
    np.testing.assert_almost_equal(cv3d, 50 + mean_sky, decimal=4)
    np.testing.assert_almost_equal(im3d.array, im3c.array, decimal=5)
    np.testing.assert_allclose(cv3d, np.var(im3d.array - im1a.array), rtol=0.1)

    config['image']['noise']['sky_level_pixel'] = 1.e-5
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)

    # 3. CCDNoise
    #####
    config['image']['noise'] =  {
        'type' : 'CCD',
        'sky_level_pixel' : 25,
        'read_noise' : 5,
        'gain' : 1,
        'whiten' : True,
    }
    galsim.config.RemoveCurrent(config)
    im4a = im1a.copy()
    rn = math.sqrt(25-cv1a)
    rng.reset(rng1.duplicate())
    im4a.addNoise(galsim.CCDNoise(sky_level=25, read_noise=rn, gain=1, rng=rng))
    im4b, cv4b = galsim.config.BuildStamp(config)
    np.testing.assert_almost_equal(cv4b, 50, decimal=5)
    np.testing.assert_almost_equal(im4b.array, im4a.array, decimal=5)
    np.testing.assert_allclose(cv4b, np.var(im4b.array - im1a.array), rtol=0.1)

    # Repeat with gain != 1
    gain = 3.7
    config['image']['noise']['gain'] = gain
    galsim.config.RemoveCurrent(config)
    im5a = im1a.copy()
    rn = math.sqrt(25 - cv1a*gain**2)
    rng.reset(rng1.duplicate())
    im5a.addNoise(galsim.CCDNoise(sky_level=25, read_noise=rn, gain=gain, rng=rng))
    im5b, cv5b = galsim.config.BuildStamp(config)
    np.testing.assert_almost_equal(cv5b, 25/gain + 5**2/gain**2, decimal=5)
    np.testing.assert_almost_equal(im5b.array, im5a.array, decimal=5)
    np.testing.assert_allclose(cv5b, np.var(im5b.array - im1a.array), rtol=0.1)

    # And again with a non-trivial sky image
    galsim.config.RemoveCurrent(config2)
    config2['image']['noise'] = config['image']['noise']
    config2['image']['noise']['symmetrize'] = 4
    del config2['image']['noise']['whiten']
    rng.reset(first_seed+1)
    im5c = galsim.Image(32,32, wcs=wcs)
    im5c = final.drawImage(im5c)
    cv5c = final.noise.symmetrizeImage(im5c, 4)
    sky = galsim.Image(im5c.bounds, wcs=wcs)
    wcs.makeSkyImage(sky, 100)
    mean_sky = np.mean(sky.array)
    im5c += sky
    rn = math.sqrt(25 - cv5c*gain**2)
    im5c.addNoise(galsim.CCDNoise(sky_level=25, read_noise=rn, gain=gain, rng=rng))
    im5d, cv5d = galsim.config.BuildStamp(config2)
    np.testing.assert_almost_equal(cv5d, (25+mean_sky)/gain + 5**2/gain**2, decimal=4)
    np.testing.assert_almost_equal(im5d.array, im5c.array, decimal=5)
    np.testing.assert_allclose(cv5d, np.var(im5d.array - im1a.array), rtol=0.1)

    config['image']['noise']['sky_level_pixel'] = 1.e-5
    config['image']['noise']['read_noise'] = 0
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)

    # 4. COSMOSNoise
    #####
    file_name = os.path.join(galsim.meta_data.share_dir,'acs_I_unrot_sci_20_cf.fits')
    config['image']['noise'] =  {
        'type' : 'COSMOS',
        'file_name' : file_name,
        'variance' : 50,
        'whiten' : True,
    }
    galsim.config.RemoveCurrent(config)
    im6a = im1a.copy()
    rng.reset(rng1.duplicate())
    noise = galsim.getCOSMOSNoise(file_name=file_name, variance=50, rng=rng)
    noise -= galsim.UncorrelatedNoise(cv1a, rng=rng, wcs=noise.wcs)
    im6a.addNoise(noise)
    im6b, cv6b = galsim.config.BuildStamp(config)
    np.testing.assert_almost_equal(cv6b, 50, decimal=5)
    np.testing.assert_almost_equal(im6b.array, im6a.array, decimal=5)
    np.testing.assert_allclose(cv6b, np.var(im6b.array - im1a.array), rtol=0.1)

    config['image']['noise']['variance'] = 1.e-5
    del config['_current_cn_tag']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)


@timer
def test_no_noise():
    """Test if there is no noise field in image.
    """
    scale = 0.3

    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
            'pixel_scale' : scale,
            'size' : 32,
        },
        'gal': { 'type': 'None' }
    }

    im = galsim.config.BuildImage(config)
    np.testing.assert_equal(im.array, 0)

    assert galsim.config.CalculateNoiseVariance(config) == 0
    assert galsim.config.AddNoise(config, im) == 0
    assert galsim.config.AddNoiseVariance(config, im) == None  # No return value.
    np.testing.assert_equal(im.array, 0)


if __name__ == "__main__":
    runtests(__file__)
