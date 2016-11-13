# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

from __future__ import print_function
import numpy as np
import os
import sys
import logging

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


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
    # Either gal or psf is required, so just give it a Gaussian with 0 flux.
    config['gal'] = {
        'type' : 'Gaussian',
        'sigma' : 1,
        'flux' : 0
    }
    config['image'] = {
       'type' : 'Single',
        'size' : size,
        'pixel_scale' : 0.3,
        'random_seed' : 123 # Note: this means the seed for the noise will really be 124
                            # since it is applied at the stamp level, so uses seed + obj_num
    }
    config['image']['noise'] = {
        'type' : 'CCD',
        'sky_level_pixel' : sky,
        'gain' : gain,
        'read_noise' : rn
    }
    image = galsim.config.BuildImage(config,logger=logger)

    print('config-built image: ')
    print('mean = ',np.mean(image.array))
    print('var = ',np.var(image.array.astype(float)))
    test_var = np.var(image.array.astype(float))

    # Build another image that should have equivalent noise properties.
    image2 = galsim.Image(size, size, scale=0.3, dtype=float)
    rng = galsim.BaseDeviate(124)
    noise = galsim.CCDNoise(rng=rng, gain=gain, read_noise=rn)
    image2 += sky
    image2.addNoise(noise)
    image2 -= sky

    print('manual sky:')
    print('mean = ',np.mean(image2.array))
    print('var = ',np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var,
                                   err_msg="CCDNoise with manual sky failed variance test.")

    # So far this isn't too stringent of a test, since the noise module will use a CCDNoise
    # object for this.  In fact, it should do precisely the same calculation.
    # This should be equivalent to letting CCDNoise take the sky level:
    image2.fill(0)
    rng.reset(124)
    noise = galsim.CCDNoise(rng=rng, sky_level=sky, gain=gain, read_noise=rn)
    image2.addNoise(noise)

    print('sky done by CCDNoise:')
    print('mean = ',np.mean(image2.array))
    print('var = ',np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var,
                                   err_msg="CCDNoise using sky failed variance test.")

    # Check that the CCDNoiseBuilder calculates the same variance as CCDNoise
    var1 = noise.getVariance()
    var2 = galsim.config.noise.CCDNoiseBuilder().getNoiseVariance(config['image']['noise'],config)
    print('CCDNoise variance = ',var1)
    print('CCDNoiseBuilder variance = ',var2)
    np.testing.assert_almost_equal(var1, var2,
                                   err_msg="CCDNoiseBuidler calculates the wrong variance")

    # Finally, the config layer also includes its own manual implementation of CCD noise that
    # it uses when there is already some noise in the image.  We want to check that this is
    # consistent with the regular CCDNoise object.

    # This time, we just set the current_var to 1.e-20 to trigger the alternate path, but
    # without any real noise there yet.
    image2.fill(0)
    rng.reset(124)
    galsim.config.noise.CCDNoiseBuilder().addNoise(
            config['image']['noise'], config, image2, rng,
            current_var = 1.e-20, draw_method='fft', logger=logger)

    print('Use CCDNoiseBuilder with negligible current_var')
    print('mean = ',np.mean(image2.array))
    print('var = ',np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var,
                                   err_msg="CCDNoise with current_var failed variance test.")

    # Here we pre-load the full read noise and tell it it's there with current_var
    image2.fill(0)
    gn = galsim.GaussianNoise(rng=rng, sigma=rn/gain)
    image2.addNoise(gn)
    galsim.config.noise.CCDNoiseBuilder().addNoise(
            config['image']['noise'], config, image2, rng,
            current_var = (rn/gain)**2, draw_method='fft', logger=logger)

    print('Use CCDNoiseBuilder with current_var == read_noise')
    print('mean = ',np.mean(image2.array))
    print('var = ',np.var(image2.array))
    # So far we've done this to very high accuracy, since we've been using the same rng seed,
    # so the results should be identical, not just close.  However, hereon the values are just
    # close, since they are difference noise realizations.  So check to 1 decimal place.
    np.testing.assert_almost_equal(np.var(image2.array),test_var, decimal=1,
                                   err_msg="CCDNoise w/ current_var==rn failed variance test.")

    # Now we pre-load part of the read-noise, but not all.  It should add the rest as read_noise.
    image2.fill(0)
    gn = galsim.GaussianNoise(rng=rng, sigma=0.5*rn/gain)
    image2.addNoise(gn)
    galsim.config.noise.CCDNoiseBuilder().addNoise(
            config['image']['noise'], config, image2, rng,
            current_var = (0.5*rn/gain)**2, draw_method='fft', logger=logger)

    print('Use CCDNoiseBuilder with current_var < read_noise')
    print('mean = ',np.mean(image2.array))
    print('var = ',np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var, decimal=1,
                                   err_msg="CCDNoise w/ current_var < rn failed variance test.")

    # Last, we go beyond the read-noise, so it should remove some of the sky level to compensate.
    image2.fill(0)
    gn = galsim.GaussianNoise(rng=rng, sigma=2.*rn/gain)
    image2.addNoise(gn)
    galsim.config.noise.CCDNoiseBuilder().addNoise(
            config['image']['noise'], config, image2, rng,
            current_var = (2.*rn/gain)**2, draw_method='fft', logger=logger)

    print('Use CCDNoiseBuilder with current_var > read_noise')
    print('mean = ',np.mean(image2.array))
    print('var = ',np.var(image2.array))
    np.testing.assert_almost_equal(np.var(image2.array),test_var, decimal=1,
                                   err_msg="CCDNoise w/ current_var > rn failed variance test.")


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
        'random_seed' : 123 # Note: this means the seed for the noise will really be 124
                            # since it is applied at the stamp level, so uses seed + obj_num
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
    rng = galsim.BaseDeviate(124)
    rgc = galsim.RealGalaxyCatalog(os.path.join(real_gal_dir, real_gal_cat))
    gal = galsim.RealGalaxy(rgc, index=79, flux=0.01, rng=rng)
    image4 = gal.drawImage(image=image3.copy())
    current_var4 = gal.noise.whitenImage(image4)
    print('After whitening, current_var = ',current_var4)
    noise = galsim.correlatednoise.getCOSMOSNoise(
            rng=rng,
            file_name=os.path.join(galsim.meta_data.share_dir,'acs_I_unrot_sci_20_cf.fits'),
            cosmos_scale=pix_scale)
    print('Full noise variance = ',noise.getVariance())
    np.testing.assert_equal(
        current_var3, noise.getVariance(),
        err_msg='Config COSMOS noise with whitening does not return the correct current_var')
    noise -= galsim.UncorrelatedNoise(current_var4, rng=rng, wcs=image4.wcs)
    print('After subtract current_var, noise variance = ',noise.getVariance())
    image4.addNoise(noise)
    np.testing.assert_equal(
        image3.array, image4.array,
        err_msg='Config COSMOS noise with whitening does not reproduce manually drawn image')


if __name__ == "__main__":
    test_ccdnoise()
    test_cosmosnoise()
