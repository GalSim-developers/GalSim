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

# TODO: Add more tests of the higher level config items.
# So far, I only added two tests related to bugs that David Kirkby found in issues
# #380 and #391.  But clearly more deserve to be added to our test suite.


@timer
def test_scattered():
    """Test aspects of building an Scattered image
    """
    import copy

    # Name some variables to make it easier to be sure they are the same value in the config dict
    # as when we build the image manually.
    size = 48
    stamp_size = 20
    scale = 0.45
    flux = 17
    sigma = 0.7
    x1 = 23.1
    y1 = 27.3
    x2 = 13.4
    y2 = 31.9
    x3 = 39.8
    y3 = 19.7

    # This part of the config will be the same for all tests
    base_config = {
        'gal' : { 'type' : 'Gaussian',
                  'sigma' : sigma,
                  'flux' : flux
                }
    }

    # Check that the stamps are centered at the correct location for both odd and even stamp size.
    base_config['image'] = {
        'type' : 'Scattered',
        'size' : size,
        'pixel_scale' : scale,
        'stamp_size' : stamp_size,
        'image_pos' : { 'type' : 'XY', 'x' : x1, 'y' : y1 },
        'nobjects' : 1
    }
    for convention in [ 0, 1 ]:
        for test_stamp_size in [ stamp_size, stamp_size + 1 ]:
            # Deep copy to make sure we don't have any "current_val" caches present.
            config = copy.deepcopy(base_config)
            config['image']['stamp_size'] = test_stamp_size
            config['image']['index_convention'] = convention

            image = galsim.config.BuildImage(config)
            np.testing.assert_equal(image.getXMin(), convention)
            np.testing.assert_equal(image.getYMin(), convention)

            xgrid, ygrid = np.meshgrid(np.arange(size) + image.getXMin(),
                                       np.arange(size) + image.getYMin())
            obs_flux = np.sum(image.array)
            cenx = np.sum(xgrid * image.array) / flux
            ceny = np.sum(ygrid * image.array) / flux
            ixx = np.sum((xgrid-cenx)**2 * image.array) / flux
            ixy = np.sum((xgrid-cenx)*(ygrid-ceny) * image.array) / flux
            iyy = np.sum((ygrid-ceny)**2 * image.array) / flux
            np.testing.assert_almost_equal(obs_flux/flux, 1, decimal=3)
            np.testing.assert_almost_equal(cenx, x1, decimal=3)
            np.testing.assert_almost_equal(ceny, y1, decimal=3)
            np.testing.assert_almost_equal(ixx / (sigma/scale)**2, 1, decimal=1)
            np.testing.assert_almost_equal(ixy, 0., decimal=3)
            np.testing.assert_almost_equal(iyy / (sigma/scale)**2, 1, decimal=1)


    # Check that stamp_xsize, stamp_ysize, image_pos use the object count, rather than the
    # image count.
    config = copy.deepcopy(base_config)
    config['image'] = {
        'type' : 'Scattered',
        'size' : size,
        'pixel_scale' : scale,
        'stamp_xsize' : { 'type': 'Sequence', 'first' : stamp_size },
        'stamp_ysize' : { 'type': 'Sequence', 'first' : stamp_size },
        'image_pos' : { 'type' : 'List',
                        'items' : [ galsim.PositionD(x1,y1),
                                    galsim.PositionD(x2,y2),
                                    galsim.PositionD(x3,y3) ]
                      },
        'nobjects' : 3
    }

    image = galsim.config.BuildImage(config)

    image2 = galsim.ImageF(size,size, scale=scale)
    image2.setZero()
    gal = galsim.Gaussian(sigma=sigma, flux=flux)

    for (i,x,y) in [ (0,x1,y1), (1,x2,y2), (2,x3,y3) ]:
        stamp = galsim.ImageF(stamp_size+i,stamp_size+i, scale=scale)
        if (stamp_size+i) % 2 == 0:
            x += 0.5
            y += 0.5
        ix = int(np.floor(x+0.5))
        iy = int(np.floor(y+0.5))
        stamp.setCenter(ix,iy)
        dx = x-ix
        dy = y-iy
        gal.drawImage(stamp, offset=(dx, dy))
        b = image2.bounds & stamp.bounds
        image2[b] += stamp[b]

    np.testing.assert_almost_equal(image.array, image2.array)


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
    # Either gal or psf is required, so just give it a Gaussian with 0 flux.
    config['gal'] = {
        'type' : 'Gaussian',
        'sigma' : 0.1,
        'flux' : 0
    }
    config['image'] = {
        'type' : 'Single',
        'pixel_scale' : pix_scale,
        'random_seed' : 123 # Note: this means the seed for the noise will really be 124
                            # since it is applied at the stamp level, so uses seed + obj_num
    }
    config['image']['noise'] = {
        'type' : 'COSMOS'
    }
    image = galsim.config.BuildImage(config,logger=logger)

    # Then make using kwargs explicitly, to make sure they are getting passed through properly.
    config2 = {}
    # Either gal or psf is required, so just give it a Gaussian with 0 flux.
    config2['gal'] = config['gal']
    config2['image'] = config['image']
    config2['image']['noise'] = {
        'type' : 'COSMOS',
        'file_name' : os.path.join(galsim.meta_data.share_dir,'acs_I_unrot_sci_20_cf.fits'),
        'cosmos_scale' : pix_scale
    }
    image2 = galsim.config.BuildImage(config2,logger=logger)

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
    image3 = galsim.config.BuildImage(config, logger=logger)

    # Build the same image by hand to make sure it matches what config drew.
    rng = galsim.BaseDeviate(124)
    rgc = galsim.RealGalaxyCatalog(os.path.join(real_gal_dir, real_gal_cat))
    gal = galsim.RealGalaxy(rgc, index=79, flux=0.01, rng=rng)
    image4 = gal.drawImage(image=image3.copy())
    current_var = gal.noise.whitenImage(image4)
    noise = galsim.correlatednoise.getCOSMOSNoise(
            rng=rng,
            file_name=os.path.join(galsim.meta_data.share_dir,'acs_I_unrot_sci_20_cf.fits'),
            cosmos_scale=pix_scale)
    noise -= galsim.UncorrelatedNoise(current_var, rng=rng, wcs=image4.wcs)
    image4.addNoise(noise)
    image3.write('image3.fits')
    image4.write('image4.fits')
    np.testing.assert_equal(
        image3.array, image4.array,
        err_msg='Config COSMOS noise with whiting does not reproduce manually drawn image')


@timer
def test_njobs():
    """Test that splitting up jobs works correctly.

    Note that there is also an implicit test of this feature in the check_yaml script.
    It runs demo9.yaml using 3 jobs and compares the result to demo9.py that does everything
    in a single run.

    However, Josh caught a very subtle bug when splitting up cgc.yaml in the examples/great3
    directory.  So this test explicitly checks for that.
    """
    # The bug was related to using a Current specification in the input field that accessed
    # a value that should have used the index_key = image_num, rather than the default when
    # processing the input field, being index_key = file_num.  Here is a fairly minimal
    # example that reproduces the error.
    config = {
        'psf' : {
            'index_key' : 'image_num',
            'type' : 'Convolve',
            'items' : [
                { 'type' : 'Gaussian', 'sigma' : 0.3 },
                {
                    'type' : 'Gaussian',
                    'sigma' : { 'type' : 'Random', 'min' : 0.3, 'max' : 1.1 },
                },
            ],
        },
        'gal' : {
            'type' : 'COSMOSGalaxy',
            'gal_type' : 'parametric',
            'index' : { 'type': 'Random' },
        },
        'image' : {
            'pixel_scale' : 0.2,
            'size' : 64,
            'random_seed' : 31415,
        },
        'input' : {
            'cosmos_catalog' : {
                'min_hlr' : '@psf.items.1.sigma',
                'dir' : os.path.join('..','examples','data'),
                'file_name' : 'real_galaxy_catalog_23.5_example.fits',
            },
        },
        'output' : {
            'nfiles' : 2,
            'dir' : 'output',
            'file_name' : {
                'type' : 'NumberedFile',
                'root' : 'test_one_job_',
                'digits' : 2,
            },
        },
    }
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger()
    galsim.config.Process(config, logger=logger)

    # Repeat with 2 jobs
    config['output']['file_name']['root'] = 'test_two_jobs_'
    galsim.config.Process(config, njobs=2, job=1, logger=logger)
    galsim.config.Process(config, njobs=2, job=2, logger=logger)

    # Check that the images are equal:
    one00 = galsim.fits.read('test_one_job_00.fits', dir='output')
    one01 = galsim.fits.read('test_one_job_01.fits', dir='output')
    two00 = galsim.fits.read('test_two_jobs_00.fits', dir='output')
    two01 = galsim.fits.read('test_two_jobs_01.fits', dir='output')

    np.testing.assert_equal(one00.array, two00.array,
                            err_msg="00 image was different for one job vs two jobs")
    np.testing.assert_equal(one01.array, two01.array,
                            err_msg="01 image was different for one job vs two jobs")


if __name__ == "__main__":
    test_scattered()
    test_ccdnoise()
    test_cosmosnoise()
    test_njobs()
