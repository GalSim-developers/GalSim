# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
import math
import re
import warnings

import galsim
from galsim_test_helpers import *


@timer
def test_single():
    """Test the default image type = Single and stamp type = Basic
    """
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
        },
        'stamp' : {
            'type' : 'Basic',
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        }
    }

    logger = logging.getLogger('test_single')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    #logger.setLevel(logging.DEBUG)

    im1_list = []
    nimages = 6
    for k in range(nimages):
        ud = galsim.UniformDeviate(1234 + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        print('gal = ',gal)
        im1 = gal.drawImage(scale=1)
        im1_list.append(im1)

        im2 = galsim.config.BuildImage(config, obj_num=k, logger=logger)
        np.testing.assert_array_equal(im2.array, im1.array)

        # Can also use BuildStamp.  In this case, it will used the cached value
        # of sigma, so we don't need to worry about resetting the rng in the config dict.
        im3 = galsim.config.BuildStamp(config, obj_num=k, logger=logger)[0]
        np.testing.assert_array_equal(im3.array, im1.array)

        # Users can write their own custom stamp builders, in which case they may want
        # to call DrawBasic directly as part of the draw method in their builder.
        im4 = galsim.config.DrawBasic(gal, im3.copy(), 'auto', galsim.PositionD(0,0),
                                      config['stamp'], config, logger)
        np.testing.assert_array_equal(im4.array, im1.array)

        # The user is allowed to to add extra kwarg to the end, which would be passed on
        # to the drawImage command.
        im5 = galsim.config.DrawBasic(gal, im3.copy(), 'auto', galsim.PositionD(0,0),
                                      config['stamp'], config, logger, scale=1.0)
        np.testing.assert_array_equal(im5.array, im1.array)

    # Use BuildStamps to build them all at once:
    im6_list = galsim.config.BuildStamps(nimages, config)[0]
    for k in range(nimages):
        im1 = im1_list[k]
        im6 = im6_list[k]
        np.testing.assert_array_equal(im6.array, im1.array)

    # In this case, BuildImages does essentially the same thing
    im7_list = galsim.config.BuildImages(nimages, config)
    for k in range(nimages):
        im1 = im1_list[k]
        im7 = im7_list[k]
        np.testing.assert_array_equal(im7.array, im1.array)

    # Check some errors
    config['stamp'] = 'Invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['stamp'] = { 'type' : 'Invalid' }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['stamp'] = { 'draw_method' : 'invalid' }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['stamp'] = { 'n_photons' : 200 }    # These next few require draw_method = phot
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['stamp'] = { 'poisson_flux' : False }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['stamp'] = { 'max_extra_noise' : 20. }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    del config['stamp']
    config['image'] = 'Invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image'] = { 'type' : 'Invalid' }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImages(3,config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImages(0,config)
    config['image'] = { 'type' : 'Single', 'xsize' : 32 }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image'] = { 'type' : 'Single', 'xsize' : 0, 'ysize' : 32 }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image'] = { 'type' : 'Single' }
    del config['gal']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)


@timer
def test_positions():
    """Test various ways to set the object position
    """
    # Start with a configuration that puts a single galaxy somewhere off-center on an image
    config = {
        'image' : {
            'type' : 'Single',
        },
        'stamp' : {
            'type' : 'Basic',
            'xsize' : 21,
            'ysize' : 21,
            'image_pos' : { 'type' : 'XY', 'x' : 39, 'y' : 43 },
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 1.7,
            'flux' : 100,
        }
    }

    logger = logging.getLogger('test_positions')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    #logger.setLevel(logging.DEBUG)

    gal = galsim.Gaussian(sigma=1.7, flux=100)
    im1= gal.drawImage(nx=21, ny=21, scale=1)
    im1.setCenter(39,43)

    im2 = galsim.config.BuildImage(config, logger=logger)
    np.testing.assert_array_equal(im2.array, im1.array)
    assert im2.bounds == im1.bounds  # This is really the main test.

    # image_pos could also be in image
    config['image']['image_pos'] = config['stamp']['image_pos']
    del config['stamp']['image_pos']
    im3 = galsim.config.BuildImage(config, logger=logger)
    np.testing.assert_array_equal(im3.array, im1.array)
    assert im3.bounds == im1.bounds

    # since our wcs is just a pixel scale, we can also specify world_pos instead.
    config['stamp']['world_pos'] = config['image']['image_pos']
    del config['image']['image_pos']
    im4 = galsim.config.BuildImage(config, logger=logger)
    np.testing.assert_array_equal(im4.array, im1.array)
    assert im4.bounds == im1.bounds

    # world_pos in image works slightly differently for image type = Single.
    # The intent there is just to give the object a world position for values that might depend
    # on it (e.g. NFWHalo shears)
    config['image']['world_pos'] = config['stamp']['world_pos']
    del config['stamp']['world_pos']
    im5 = galsim.config.BuildImage(config, logger=logger)
    np.testing.assert_array_equal(im5.array, im1.array)
    assert im5.bounds == galsim.BoundsI(-10,10,-10,10)

    # It is also valid to give both world_pos and image_pos in the image field for Single.
    config['image']['image_pos'] = config['image']['world_pos']
    im6 = galsim.config.BuildImage(config, logger=logger)
    np.testing.assert_array_equal(im6.array, im1.array)
    assert im6.bounds == im1.bounds

    del config['image']['image_pos']
    del config['image']['world_pos']
    config['stamp']['world_pos'] = { 'type' : 'Random' }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)


@timer
def test_phot():
    """Test draw_method=phot, which has extra allowed kwargs
    """
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
            'draw_method' : 'phot'
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 1.7,
            'flux' : 100,
        }
    }

    # First the simple config written above
    ud = galsim.UniformDeviate(1234 + 1)
    gal = galsim.Gaussian(sigma=1.7, flux=100)
    im1a = gal.drawImage(scale=1, method='phot', rng=ud)
    im1b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im1b.array, im1a.array)

    # Use a non-default number of photons
    del config['_copied_image_keys_to_stamp']
    config['image']['n_photons'] = 300
    ud.seed(1234 + 1)
    im2a = gal.drawImage(scale=1, method='phot', n_photons=300, rng=ud)
    im2b = galsim.config.BuildImage(config)
    print('image = ',config['image'])
    print('stamp = ',config['stamp'])
    np.testing.assert_array_equal(im2b.array, im2a.array)

    # Allow the flux to vary as a Poisson deviate even though n_photons is given
    del config['_copied_image_keys_to_stamp']
    config['image']['poisson_flux'] = True
    ud.seed(1234 + 1)
    im3a = gal.drawImage(scale=1, method='phot', n_photons=300, rng=ud, poisson_flux=True)
    im3b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im3b.array, im3a.array)

    # If max_extra_noise is given with n_photons, then ignore it.
    del config['_copied_image_keys_to_stamp']
    config['stamp']['max_extra_noise'] = 0.1
    im3c = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im3c.array, im3a.array)

    # Although with a logger, it should give a warning.
    with CaptureLog() as cl:
        im3d = galsim.config.BuildImage(config, logger=cl.logger)
    assert "ignoring 'max_extra_noise'" in cl.output

    # Without n_photons, it should work.  But then, we also need a noise field
    # So without the noise field, it will raise an exception.
    del config['image']['n_photons']
    del config['stamp']['n_photons']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # Using this much extra noise with a sky noise variance of 50 cuts the number of photons
    # approximately in half.
    print('N,g without extra_noise: ',gal._calculate_nphotons(0, False, 0, None))
    print('N,g with extra_noise: ',gal._calculate_nphotons(0, False, 5, None))
    config['image']['noise'] = { 'type' : 'Gaussian', 'variance' : 50 }
    ud.seed(1234 + 1)
    im4a = gal.drawImage(scale=1, method='phot', max_extra_noise=5, rng=ud, poisson_flux=True)
    im4a.addNoise(galsim.GaussianNoise(sigma=math.sqrt(50), rng=ud))
    im4b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im4b.array, im4a.array)

    # max_extra noise < 0 is invalid
    config['stamp']['max_extra_noise'] = -1.
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # Also negative variance noise is invalid.
    config['stamp']['max_extra_noise'] = 0.1
    config['image']['noise'] = { 'type' : 'Gaussian', 'variance' : -50 }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)


@timer
def test_reject():
    """Test various ways that objects can be rejected.
    """
    # Make a custom function for rejecting COSMOSCatalog objects that use Sersics with n > 2.
    def HighN(config, base, value_type):
        # GetCurrentValue returns the constructed 'gal' object for this pass
        gal = galsim.config.GetCurrentValue('gal',base)
        # First undo the two bits we did to the galaxy that COSMOSCatalog made
        assert isinstance(gal, galsim.Transformation)
        gal = gal.original
        assert isinstance(gal, galsim.Convolution)
        gal = gal._obj_list[0]
        # Now gal is the object COSMOSCatalog produced.
        if isinstance(gal, galsim.Sum):
            # Reject all B+D galaxies (which are a minority)
            reject = True
        else:
            # For pure Sersics, reject those with n > 2.
            assert isinstance(gal, galsim.Transformation)
            gal = gal.original
            assert isinstance(gal, galsim.Sersic)
            reject = gal.n > 2
        # The second item in the return tuple is "safe", which means is this value safe to
        # cache and reuse (potentially saving calculation time for complex objects).
        # Not really applicable here, so return False.
        return reject, False
    # You need to register this function with a "type" name so config knows about it.
    galsim.config.RegisterValueType('HighN', HighN, [bool])

    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 12345,
            'noise' : { 'type' : 'Gaussian', 'variance' : 10 },
            'pixel_scale' : 0.1,
        },
        'stamp' : {
            'type' : 'Basic',
            'size' : 32,
            'retry_failures' : 50,
            'reject' : { 'type' : 'HighN' },
            'min_flux_frac' : 0.95,  # This rejects around 1/4 of the objects
            'max_snr' : 50, # This just rejects a few objects
            'image_pos': {
                # This will raise an exception about 1/4 the time (when inner_radius > radius)
                'type' : 'RandomCircle',
                'radius' : { 'type' : 'Random', 'min' : 0, 'max': 20 },
                'inner_radius' : { 'type' : 'Random', 'min' : 0, 'max': 10 },
            },
            'skip' : '$obj_num == 9',
        },
        'gal' : {
            'type' : 'Convolve',
            'items' : [
                {
                    'type' : 'COSMOSGalaxy',
                    'gal_type' : 'parametric',
                    # This is invalid about 1/3 of the time. (There are only 100 items in the
                    # catalog.)
                    'index' : { 'type' : 'Random', 'min' : 0, 'max' : 150 },
                },
                # This is essentially the PSF, but doing it this way covers a branch in the reject
                # function that wouldn't be covered if we had a psf field.
                { 'type' : 'Gaussian', 'sigma' : 0.15 }
            ],
            'scale_flux' : {
                'type' : 'Eval',
                # This will raise an exception about half the time
                'str' : 'math.sqrt(x)',
                'fx' : { 'type' : 'Random', 'min' : -10000, 'max' : 10000 },
            },
            'skip' : {
                # Skip doesn't count as an error for the recount.  Rather it returns None
                # for the image rather than making it.
                # 1/20 will be skipped.  (Although with all the other rejections and exceptions
                # a much higher fraction of the returned images will be None.)
                'type' : 'RandomBinomial',
                'p' : 0.05,
            },
        },
        'input' : {
            'cosmos_catalog' : {
                'dir' : '../examples/data',
                'file_name' : 'real_galaxy_catalog_23.5_example_fits.fits',
                'use_real' : False,
            }
        }
    }
    galsim.config.ProcessInput(config)

    if False:
        logger = logging.getLogger('test_reject')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        #logger.setLevel(logging.DEBUG)
    else:
        logger = galsim.config.LoggerWrapper(None)

    nimages = 10
    im_list = galsim.config.BuildStamps(nimages, config, do_noise=False, logger=logger)[0]
    # For this particular config, only 6 of them are real images.  The others were skipped.
    # The skipped ones are present in the list, but their flux is 0
    fluxes = [im.array.sum(dtype=float) if im is not None else 0 for im in im_list]
    expected_fluxes = [1289, 0, 1993, 1398, 0, 1795, 0, 0, 458, 0]
    np.testing.assert_almost_equal(fluxes, expected_fluxes, decimal=0)

    # Check for a few of the logging outputs that explain why things were rejected.
    with CaptureLog() as cl:
        im_list2 = galsim.config.BuildStamps(nimages, config, do_noise=False, logger=cl.logger)[0]
    for im1,im2 in zip(im_list, im_list2):
        assert im1 == im2
    #print(cl.output)
    # Note: I'm testing for specific error messages here, which could change if we change
    # the order of operations somewhere.  The point here is that we hit at least one of each
    # kind of skip/rejection/exception that we intended in the config dict above.
    assert "obj 1: Skipping because field skip=True" in cl.output
    assert "obj 1: Caught SkipThisObject: e = None" in cl.output
    assert "Skipping object 1" in cl.output
    assert "Object 0: Caught exception index=105 has gone past the number of entries" in cl.output
    assert "Object 0: Caught exception inner_radius must be less than radius (3.931733)" in cl.output
    assert "Object 0: Caught exception Unable to evaluate string 'math.sqrt(x)'" in cl.output
    assert "obj 0: reject evaluated to True" in cl.output
    assert "Object 0: Rejecting this object and rebuilding" in cl.output
    # This next one can end up with slightly different numerical values depending on numpy version
    #assert "Object 0: Measured flux = 3253.173584 < 0.95 * 3457.712670." in cl.output
    #assert "Object 0: Measured snr = 60.992197 > 50.0." in cl.output
    assert re.search(r"Object 0: Measured flux = 3253.17[0-9]* < 0.95 \* 3457.712670.", cl.output)
    assert re.search(r"Object 0: Measured snr = 60.992[0-9]* > 50.0.", cl.output)

    # For test coverage to get all branches, do min_snr and max_snr separately.
    del config['stamp']['max_snr']
    config['stamp']['min_snr'] = 20
    with CaptureLog() as cl:
        im_list2 = galsim.config.BuildStamps(nimages, config, do_noise=False, logger=cl.logger)[0]
    #print(cl.output)
    assert re.search(r"Object 8: Measured snr = 9.1573[0-9]* < 20.0.", cl.output)

    # If we lower the number of retries, we'll max out and abort the image
    config['stamp']['retry_failures'] = 10
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamps(nimages, config, do_noise=False)
    try:
        with CaptureLog() as cl:
            galsim.config.BuildStamps(nimages, config, do_noise=False, logger=cl.logger)
    except (galsim.GalSimConfigError):
        pass
    #print(cl.output)
    assert "Object 0: Too many exceptions/rejections for this object. Aborting." in cl.output
    assert "Exception caught when building stamp" in cl.output

    # We can also do this with BuildImages which runs through a different code path.
    galsim.config.RemoveCurrent(config)
    try:
        with CaptureLog() as cl:
            galsim.config.BuildImages(nimages, config, logger=cl.logger)
    except (ValueError,IndexError,galsim.GalSimError):
        pass
    #print(cl.output)
    assert "Exception caught when building image" in cl.output

    # When in nproc > 1 mode, the error message is slightly different.
    config['image']['nproc'] = 2
    try:
        with CaptureLog() as cl:
            galsim.config.BuildStamps(nimages, config, do_noise=False, logger=cl.logger)
    except (ValueError,IndexError,galsim.GalSimError):
        pass
    #print(cl.output)
    assert re.search("Process-.: Exception caught when building stamp",cl.output)

    try:
        with CaptureLog() as cl:
            galsim.config.BuildImages(nimages, config, logger=cl.logger)
    except (ValueError,IndexError,galsim.GalSimError):
        pass
    #print(cl.output)
    assert re.search("Process-.: Exception caught when building image",cl.output)

    # Finally, if all images give errors, BuildFiles will not raise an exception, but will just
    # report that no files were written.
    config['stamp']['max_snr'] = 20 # If nothing else failed, min or max snr will reject.
    config['root'] = 'test_reject'  # This lets the code generate a file name automatically.
    del config['stamp']['size']     # Otherwise skipped images will still build an empty image.
    config = galsim.config.CleanConfig(config)
    with CaptureLog() as cl:
        galsim.config.BuildFiles(nimages, config, logger=cl.logger)
    #print(cl.output)
    assert "No files were written.  All were either skipped or had errors." in cl.output

    # If we skip all objects, and don't have a definite size for them, then we get to a message
    # that no stamps were built.
    config['gal']['skip'] = True
    galsim.config.RemoveCurrent(config)
    im_list3 = galsim.config.BuildStamps(nimages, config, do_noise=False)[0]
    assert all (im is None for im in im_list3)
    with CaptureLog() as cl:
        im_list3 = galsim.config.BuildStamps(nimages, config, do_noise=False, logger=cl.logger)[0]
    #print(cl.output)
    assert "No stamps were built.  All objects were skipped." in cl.output

    # Likewise with BuildImages, but with a slightly different message.
    with CaptureLog() as cl:
        im_list4 = galsim.config.BuildImages(nimages, config, logger=cl.logger)
    assert "No images were built.  All were either skipped or had errors." in cl.output

    # And BuildFiles
    with CaptureLog() as cl:
        galsim.config.BuildFiles(nimages, config, logger=cl.logger)
    assert "No files were written.  All were either skipped or had errors." in cl.output

    # Finally, with a fake logger, this covers the LoggerWrapper functionality.
    logger = galsim.config.LoggerWrapper(None)
    galsim.config.BuildFiles(nimages, config, logger=logger)


@timer
def test_snr():
    """Test signal-to-noise option for setting the flux
    """
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
            'noise' : { 'type' : 'Gaussian', 'variance' : 50 },
            'pixel_scale' : 0.4,
        },
        'stamp' : {
            'type' : 'Basic',
            'size' : 32,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 1.7,
            'signal_to_noise' : 70,
        }
    }

    # Do the S/N calculation by hand.
    ud = galsim.UniformDeviate(1234 + 1)
    gal = galsim.Gaussian(sigma=1.7)
    im1a = gal.drawImage(nx=32, ny=32, scale=0.4)
    sn_meas = math.sqrt( np.sum(im1a.array**2, dtype=float) / 50 )
    print('measured s/n = ',sn_meas)
    im1a *= 70 / sn_meas
    im1a.addNoise(galsim.GaussianNoise(sigma=math.sqrt(50), rng=ud))

    # Compare to what config does
    im1b = galsim.config.BuildImage(config)
    np.testing.assert_equal(im1b.array, im1a.array)

    # Also works on psf images
    config['psf'] = config['gal']
    del config['gal']
    im1c = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im1c.array, im1a.array)

    # Photon shooting can do snr scaling, but will give a warning.
    config['stamp']['draw_method'] = 'phot'
    config['stamp']['n_photons'] = 100  # Need something here otherwise it will shoot 1 photon.
    ud.seed(1234 + 1)
    im2a = gal.drawImage(nx=32, ny=32, scale=0.4, method='phot', n_photons=100, rng=ud)
    sn_meas = math.sqrt( np.sum(im2a.array**2, dtype=float) / 50 )
    print('measured s/n = ',sn_meas)
    im2a *= 70 / sn_meas
    im2a.addNoise(galsim.GaussianNoise(sigma=math.sqrt(50), rng=ud))
    im2b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im2b.array, im2a.array)

    with CaptureLog() as cl:
        im2c = galsim.config.BuildImage(config, logger=cl.logger)
    np.testing.assert_array_equal(im2c.array, im2a.array)
    assert 'signal_to_noise calculation is not accurate for draw_method = phot' in cl.output


@timer
def test_ring():
    """Test the stamp type = Ring
    """
    config = {
        'stamp' : {
            'type' : 'Ring',
            'num' : 2,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 2,
            'ellip' : {
                'type' : 'E1E2',
                'e1' : { 'type' : 'List',
                            'items' : [ 0.3, 0.2, 0.8 ],
                            'index' : { 'type' : 'Sequence', 'repeat' : 2 }
                       },
                'e2' : 0.1
            }
        }
    }

    gauss = galsim.Gaussian(sigma=2)
    e1_list = [ 0.3, -0.3, 0.2, -0.2, 0.8, -0.8 ]
    e2_list = [ 0.1, -0.1, 0.1, -0.1, 0.1, -0.1 ]

    galsim.config.SetupConfigImageNum(config, 0, 0)
    ignore = galsim.config.stamp_ignore
    ring_builder = galsim.config.stamp_ring.RingBuilder()
    for k in range(6):
        galsim.config.SetupConfigObjNum(config, k)
        ring_builder.setup(config['stamp'], config, None, None, ignore, None)
        gal1a = ring_builder.buildProfile(config['stamp'], config, None, {}, None)
        gal1b = gauss.shear(e1=e1_list[k], e2=e2_list[k])
        print('gal1a = ',gal1a)
        print('gal1b = ',gal1b)
        gsobject_compare(gal1a, gal1b)

    # Make sure it all runs using the normal syntax
    stamp = galsim.config.BuildStamp(config)

    # num <= 0 is invalid
    config['stamp']['num'] = 0
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)
    config['stamp']['num'] = -1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)
    del config['stamp']['num']
    del config['stamp']['_get']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamps(10, config)   # Different error is making multiple stamps.
    # Check invalid index.  (Ususally this is automatic and can't be wrong, but it is
    # permissible to set it by hand.)
    config['stamp']['num'] = 2
    config['stamp']['index'] = 2
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)
    config['stamp']['index'] = -1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)
    # Starting on an odd index is an error.  It's hard to make this happen in practice,
    # but her we can do it by manually deleting the first attribute.
    del galsim.config.stamp.valid_stamp_types['Ring'].first
    config['stamp']['index'] = 1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)
    # Invalid to just have a psf.
    config['psf'] = config['gal']
    del config['gal']
    config['stamp']['index'] = 0
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config)

    config = {
        'stamp' : {
            'type' : 'Ring',
            'num' : 10,
        },
        'gal' : {
            'type' : 'Exponential', 'half_light_radius' : 2,
            'ellip' : galsim.Shear(e2=0.3)
        },
    }

    disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)

    galsim.config.SetupConfigImageNum(config, 0, 0)
    for k in range(25):
        galsim.config.SetupConfigObjNum(config, k)
        ring_builder.setup(config['stamp'], config, None, None, ignore, None)
        gal2a = ring_builder.buildProfile(config['stamp'], config, None, {}, None)
        gal2b = disk.rotate(theta = k * 18 * galsim.degrees)
        gsobject_compare(gal2a, gal2b)

    config = {
        'stamp' : {
            'type' : 'Ring',
            'num' : 5,
            'full_rotation' : 360. * galsim.degrees,
            'index' : { 'type' : 'Sequence', 'repeat' : 4 }
        },
        'gal' : {
            'type' : 'Sum',
            'items' : [
                { 'type' : 'Exponential', 'half_light_radius' : 2,
                    'ellip' : galsim.Shear(e2=0.3)
                },
                { 'type' : 'Sersic', 'n' : 3, 'half_light_radius' : 1.3,
                    'ellip' : galsim.Shear(e1=0.12,e2=-0.08)
                }
            ]
        },
    }

    disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)
    bulge = galsim.Sersic(n=3, half_light_radius=1.3).shear(e1=0.12,e2=-0.08)
    sum = disk + bulge

    galsim.config.SetupConfigImageNum(config, 0, 0)
    for k in range(25):
        galsim.config.SetupConfigObjNum(config, k)
        index = k // 4  # make sure we use integer division
        ring_builder.setup(config['stamp'], config, None, None, ignore, None)
        gal3a = ring_builder.buildProfile(config['stamp'], config, None, {}, None)
        gal3b = sum.rotate(theta = index * 72 * galsim.degrees)
        gsobject_compare(gal3a, gal3b)

    # Check that the ring items correctly inherit their gsparams from the top level
    config = {
        'stamp' : {
            'type' : 'Ring',
            'num' : 20,
            'full_rotation' : 360. * galsim.degrees,
            'gsparams' : { 'maxk_threshold' : 1.e-2,
                           'folding_threshold' : 1.e-2,
                           'stepk_minimum_hlr' : 3 }
        },
        'gal' : {
            'type' : 'Sum',
            'items' : [
                { 'type' : 'Exponential', 'half_light_radius' : 2,
                    'ellip' : galsim.Shear(e2=0.3)
                },
                { 'type' : 'Sersic', 'n' : 3, 'half_light_radius' : 1.3,
                    'ellip' : galsim.Shear(e1=0.12,e2=-0.08)
                }
            ]
        },
    }

    galsim.config.SetupConfigImageNum(config, 0, 0)
    galsim.config.SetupConfigObjNum(config, 0)
    ring_builder.setup(config['stamp'], config, None, None, ignore, None)
    gal4a = ring_builder.buildProfile(config['stamp'], config, None, config['stamp']['gsparams'],
                                      None)
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, folding_threshold=1.e-2, stepk_minimum_hlr=3)
    disk = galsim.Exponential(half_light_radius=2, gsparams=gsparams).shear(e2=0.3)
    bulge = galsim.Sersic(n=3,half_light_radius=1.3, gsparams=gsparams).shear(e1=0.12,e2=-0.08)
    gal4b = disk + bulge
    gsobject_compare(gal4a, gal4b, conv=galsim.Gaussian(sigma=1))

    # Make sure they don't match when using the default GSParams
    disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)
    bulge = galsim.Sersic(n=3,half_light_radius=1.3).shear(e1=0.12,e2=-0.08)
    gal4c = disk + bulge
    with assert_raises(AssertionError):
        gsobject_compare(gal4a, gal4c, conv=galsim.Gaussian(sigma=1))


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
            # Deep copy to make sure we don't have any "current" caches present.
            config = copy.deepcopy(base_config)
            config['image']['stamp_size'] = test_stamp_size
            config['image']['index_convention'] = convention

            image = galsim.config.BuildImage(config)
            np.testing.assert_equal(image.xmin, convention)
            np.testing.assert_equal(image.ymin, convention)

            xgrid, ygrid = np.meshgrid(np.arange(size) + image.xmin,
                                       np.arange(size) + image.ymin)
            obs_flux = np.sum(image.array, dtype=float)
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

    config['image']['index_convention'] = 'invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

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

    # Check error message for missing nobjects
    del config['image']['nobjects']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    # Also if there is an input field that doesn't have nobj capability
    config['input'] = { 'dict' : { 'dir' : 'config_input', 'file_name' : 'dict.p' } }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    # However, an input field that does have nobj will return something for nobjects.
    # This catalog has 3 rows, so equivalent to nobjects = 3
    config['input'] = { 'catalog' : { 'dir' : 'config_input', 'file_name' : 'catalog.txt' } }
    config = galsim.config.CleanConfig(config)
    image = galsim.config.BuildImage(config)
    np.testing.assert_almost_equal(image.array, image2.array)

    del config['image']['size']
    del config['image']['_get']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['xsize'] = size
    del config['image']['_get']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['ysize'] = size
    del config['image']['_get']

    # If doing datacube, sizes have to be consistent.
    config['image_force_xsize'] = size
    config['image_force_ysize'] = size
    galsim.config.BuildImage(config)  # This works

    # These don't.
    config['image']['xsize'] = size-1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['xsize'] = size
    config['image']['ysize'] = size+1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['ysize'] = size

    # Can't have both image_pos and world_pos
    config['image']['world_pos'] = {
        'type' : 'List',
        'items' : [ galsim.PositionD(x1*scale, y1*scale),
                    galsim.PositionD(x2*scale, y2*scale),
                    galsim.PositionD(x3*scale, y3*scale) ]
    }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    del config['image']['image_pos']
    image = galsim.config.BuildImage(config)  # But just world_pos is fine.
    np.testing.assert_almost_equal(image.array, image2.array)

    # When starting from the file state, there is some extra code to test about this, so
    # check that here.
    config['output'] = { 'type' : 'MultiFits', 'file_name' : 'output/test_scattered.fits',
                         'nimages' : 2 }
    config = galsim.config.CleanConfig(config)
    galsim.config.BuildFile(config)
    image = galsim.fits.read('output/test_scattered.fits')
    np.testing.assert_almost_equal(image.array, image2.array)

@timer
def test_scattered_whiten():
    """Test whitening with the image type Scattered.  In particular getting the noise flattened
    across overlapping stamps and stamps that are partially off the image.
    """
    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = 'real_galaxy_catalog_23.5_example.fits'
    scale = 0.05
    index = 79
    flux = 10000
    variance = 10
    skip_prob = 0.2
    nobjects = 30
    config = {
        'image' : {
            'type' : 'Scattered',
            'random_seed' : 12345,
            'pixel_scale' : scale,
            'size' : 100,
            'image_pos' : { 'type' : 'XY',
                            # Some of these will be completely off the main image.
                            # They will be ignored.
                            'x' : { 'type' : 'Random', 'min': -50, 'max': 150 },
                            'y' : { 'type' : 'Random', 'min': -50, 'max': 150 },
                          },
            'nobjects' : nobjects,
            'noise' : {
                'type' : 'Gaussian',
                'variance' : variance,
                'whiten' : True,
            },
        },
        'gal' : {
            'type' : 'RealGalaxy',
            'index' : index,  # It's a bit faster if they all use the same index.
            'flux' : flux,

            # This tests a special case in FlattenNoiseVariance
            'skip' : { 'type': 'RandomBinomial', 'p': skip_prob }
        },
        'psf' : {
            'type' : 'Gaussian',
            'sigma' : 0.1,
        },
        'input' : {
            'real_catalog' : {
                'dir' : real_gal_dir ,
                'file_name' : real_gal_cat,
            }
        }
    }

    # First build by hand
    rgc = galsim.RealGalaxyCatalog(os.path.join(real_gal_dir, real_gal_cat))
    gal = galsim.RealGalaxy(rgc, index=index, flux=flux)
    psf = galsim.Gaussian(sigma=0.1)
    final = galsim.Convolve(gal,psf)
    im1 = galsim.Image(100,100, scale=scale)
    cv_im = galsim.Image(100,100)

    for k in range(nobjects):
        ud = galsim.UniformDeviate(12345 + k + 1)

        x = ud() * 200. - 50.
        y = ud() * 200. - 50.

        skip_dev = galsim.BinomialDeviate(ud, N=1, p=skip_prob)
        if skip_dev() > 0: continue

        ix = int(math.floor(x+1))
        iy = int(math.floor(y+1))
        dx = x-ix+0.5
        dy = y-iy+0.5
        stamp = final.drawImage(offset=(dx, dy), scale=scale)
        stamp.setCenter(ix,iy)

        final.noise.rng.reset(ud)
        cv = final.noise.whitenImage(stamp)

        b = im1.bounds & stamp.bounds
        if not b.isDefined(): continue

        im1[b] += stamp[b]
        cv_im[b] += cv

    print('max cv = ',cv_im.array.max())
    print('min cv = ',cv_im.array.min())
    max_cv = cv_im.array.max()
    noise_im = max_cv - cv_im
    rng = galsim.BaseDeviate(12345)
    im1.addNoise(galsim.VariableGaussianNoise(rng, noise_im))
    im1.addNoise(galsim.GaussianNoise(rng, sigma=math.sqrt(variance-max_cv)))

    # Compare to what config builds
    im2 = galsim.config.BuildImage(config)
    np.testing.assert_almost_equal(im2.array, im1.array)

    # Should give a warning for the objects that fall off the edge
    with CaptureLog() as cl:
        im3 = galsim.config.BuildImage(config, logger=cl.logger)
    #print(cl.output)
    assert "skip drawing object because its image will be entirely off the main image." in cl.output
    im2 = galsim.config.BuildImage(config)


@timer
def test_tiled():
    """Test image type = Tiled
    """
    nx = 5
    ny = 7
    xsize = 32
    ysize = 25
    xborder = 2
    yborder = 3
    scale = 0.3
    config = {
        'image' : {
            'type' : 'Tiled',
            'nx_tiles' : nx,
            'ny_tiles' : ny,
            'stamp_xsize' : xsize,
            'stamp_ysize' : ysize,
            'pixel_scale' : scale,

            'xborder' : xborder,
            'yborder' : yborder,

            'random_seed' : 1234,

            'noise' : { 'type': 'Gaussian', 'sigma': 0.5 }
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : '$image_pos.x + image_pos.y',
        },
    }

    seed = 1234
    im1a = galsim.Image(nx * (xsize+xborder) - xborder, ny * (ysize+yborder) - yborder, scale=scale)
    for j in range(ny):
        for i in range(nx):
            seed += 1
            ud = galsim.UniformDeviate(seed)
            xorigin = i * (xsize+xborder) + 1
            yorigin = j * (ysize+yborder) + 1
            x = xorigin + (xsize-1)/2.
            y = yorigin + (ysize-1)/2.
            stamp = galsim.Image(xsize,ysize, scale=scale)
            stamp.setOrigin(xorigin,yorigin)

            sigma = ud() + 1
            flux = x + y
            gal = galsim.Gaussian(sigma=sigma, flux=flux)
            gal.drawImage(stamp)
            stamp.addNoise(galsim.GaussianNoise(sigma=0.5, rng=ud))
            im1a[stamp.bounds] = stamp

    # Compare to what config builds
    im1b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im1b.array, im1a.array)

    # Switch to column ordering.  Also make the stamps overlap slightly, which changes how the
    # noise is applied.
    galsim.config.RemoveCurrent(config)
    config['image']['order'] = 'col'
    config['image']['xborder'] = -xborder
    config['image']['yborder'] = -yborder
    im2a = galsim.Image(nx * (xsize-xborder) + xborder, ny * (ysize-yborder) + yborder, scale=scale)
    seed = 1234
    for i in range(nx):
        for j in range(ny):
            seed += 1
            ud = galsim.UniformDeviate(seed)
            xorigin = i * (xsize-xborder) + 1
            yorigin = j * (ysize-yborder) + 1
            x = xorigin + (xsize-1)/2.
            y = yorigin + (ysize-1)/2.
            stamp = galsim.Image(xsize,ysize, scale=scale)
            stamp.setOrigin(xorigin,yorigin)

            sigma = ud() + 1
            flux = x + y
            gal = galsim.Gaussian(sigma=sigma, flux=flux)
            gal.drawImage(stamp)
            im2a[stamp.bounds] += stamp
    im2a.addNoise(galsim.GaussianNoise(sigma=0.5, rng=galsim.BaseDeviate(1234)))

    # Compare to what config builds
    im2b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im2b.array, im2a.array)

    # Finally, random ordering.  And also add some skips, so some of the stamps come back as None.
    # Switch to column ordering.  Also make the stamps overlap slightly, which changes how the
    # noise is applied.
    galsim.config.RemoveCurrent(config)
    config['image']['order'] = 'rand'
    config['gal']['skip'] = { 'type': 'RandomBinomial', 'p': 0.2 }
    im3a = galsim.Image(nx * (xsize-xborder) + xborder, ny * (ysize-yborder) + yborder, scale=scale)
    seed = 1234
    i_list = []
    j_list = []
    for i in range(nx):
        for j in range(ny):
            i_list.append(i)
            j_list.append(j)
    rng = galsim.BaseDeviate(seed)
    galsim.random.permute(rng, i_list, j_list)
    for i,j in zip(i_list,j_list):
        seed += 1
        ud = galsim.UniformDeviate(seed)

        skip_dev = galsim.BinomialDeviate(ud, N=1, p=0.2)
        if skip_dev() > 0: continue

        xorigin = i * (xsize-xborder) + 1
        yorigin = j * (ysize-yborder) + 1
        x = xorigin + (xsize-1)/2.
        y = yorigin + (ysize-1)/2.
        stamp = galsim.Image(xsize,ysize, scale=scale)
        stamp.setOrigin(xorigin,yorigin)

        sigma = ud() + 1
        flux = x + y
        gal = galsim.Gaussian(sigma=sigma, flux=flux)
        gal.drawImage(stamp)
        im3a[stamp.bounds] += stamp
    im3a.addNoise(galsim.GaussianNoise(sigma=0.5, rng=rng))

    # Compare to what config builds
    im3b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im3b.array, im3a.array)

    # Check errors
    # sizes need to be > 0
    config['image']['stamp_xsize'] = 0
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['stamp_xsize'] = xsize
    config['image']['stamp_ysize'] = -30
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['stamp_ysize'] = ysize
    config['image']['order'] = 'invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['order'] = 'col'
    del config['image']['nx_tiles']
    del config['image']['_get']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImages(2,config)
    config['image']['nx_tiles'] = nx

    # If doing datacube, sizes have to be consistent.
    config['image']['stamp_xsize'] = xsize
    config['image']['stamp_ysize'] = ysize
    config['image_force_xsize'] = im3b.array.shape[1]
    config['image_force_ysize'] = im3b.array.shape[0]
    galsim.config.BuildImage(config)  # This works.

    # These don't.
    config['image']['stamp_xsize'] = xsize-1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['stamp_xsize'] = xsize
    config['image']['stamp_ysize'] = ysize+1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image']['stamp_ysize'] = ysize
    config['image']['yborder'] = xborder
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # If tile with a square grid, then PowerSpectrum can omit grid_spacing and ngrid.
    size = 32
    config = {
        'image' : {
            'type' : 'Tiled',
            'nx_tiles' : nx,
            'ny_tiles' : ny,
            'stamp_size' : size,
            'pixel_scale' : scale,

            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : '$image_pos.x + image_pos.y',
            'shear' : { 'type' : 'PowerSpectrumShear' },
        },
        'input' : {
            'power_spectrum' : { 'e_power_function' : 'np.exp(-k**0.2)' },
        },
    }

    seed = 1234
    ps = galsim.PowerSpectrum(e_power_function=lambda k: np.exp(-k**0.2))
    rng = galsim.BaseDeviate(seed)
    im4a = galsim.Image(nx*size, ny*size, scale=scale)
    center = im4a.true_center * scale
    ps.buildGrid(grid_spacing=size*scale, ngrid=max(nx,ny)+1, rng=rng, center=center)
    for j in range(ny):
        for i in range(nx):
            seed += 1
            ud = galsim.UniformDeviate(seed)
            xorigin = i * size + 1
            yorigin = j * size + 1
            x = xorigin + (size-1)/2.
            y = yorigin + (size-1)/2.
            stamp = galsim.Image(size,size, scale=scale)
            stamp.setOrigin(xorigin,yorigin)

            sigma = ud() + 1
            flux = x + y
            gal = galsim.Gaussian(sigma=sigma, flux=flux)
            g1, g2 = ps.getShear(galsim.PositionD(x*scale,y*scale))
            gal = gal.shear(g1=g1, g2=g2)
            gal.drawImage(stamp)
            im4a[stamp.bounds] = stamp

    # Compare to what config builds
    im4b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im4b.array, im4a.array)

    # If grid sizes aren't square, it also works properly, but with more complicated ngrid calc.
    config = galsim.config.CleanConfig(config)
    del config['image']['stamp_size']
    config['image']['stamp_xsize'] = xsize
    config['image']['stamp_ysize'] = ysize
    seed = 1234
    rng = galsim.BaseDeviate(seed)
    im5a = galsim.Image(nx*xsize, ny*ysize, scale=scale)
    center = im5a.true_center * scale
    grid_spacing = min(xsize,ysize) * scale
    ngrid = int(math.ceil(max(nx*xsize, ny*ysize) * scale / grid_spacing))+1
    ps.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid, rng=rng, center=center)
    for j in range(ny):
        for i in range(nx):
            seed += 1
            ud = galsim.UniformDeviate(seed)
            xorigin = i * xsize + 1
            yorigin = j * ysize + 1
            x = xorigin + (xsize-1)/2.
            y = yorigin + (ysize-1)/2.
            stamp = galsim.Image(xsize,ysize, scale=scale)
            stamp.setOrigin(xorigin,yorigin)

            sigma = ud() + 1
            flux = x + y
            gal = galsim.Gaussian(sigma=sigma, flux=flux)
            g1, g2 = ps.getShear(galsim.PositionD(x*scale,y*scale))
            gal = gal.shear(g1=g1, g2=g2)
            gal.drawImage(stamp)
            im5a[stamp.bounds] = stamp

    im5b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im5b.array, im5a.array)

    # Finally, if the image type isn't tiled, then grid_spacing is required.
    config = {
        'image' : {
            'type' : 'Scattered',
            'size': nx*size,
            'nobjects' : nx*ny,
            'pixel_scale' : scale,
            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : '$image_pos.x + image_pos.y',
            'shear' : { 'type' : 'PowerSpectrumShear' },
        },
        'input' : {
            'power_spectrum' : { 'e_power_function' : 'np.exp(-k**0.2)' },
        },
    }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)


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
    config1 = galsim.config.CopyConfig(config)

    logger = logging.getLogger('test_njobs')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    galsim.config.Process(config, logger=logger)

    # Repeat with 2 jobs
    config = galsim.config.CopyConfig(config1)
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

def test_wcs():
    """Test various wcs options"""
    config = {
        # We'll need this for some of the items below.
        'image_center' : galsim.PositionD(1024, 1024)
    }
    config['image'] = {
        'pixel_scale' : 0.34,
        'scale2' : { 'type' : 'PixelScale', 'scale' : 0.43 },
        'scale3' : {
            'type' : 'PixelScale',
            'scale' : 0.43,
            'origin' : galsim.PositionD(32,24),
        },
        'scale4' : {
            'type' : 'PixelScale',
            'scale' : 0.43,
            'origin' : { 'type' : 'XY', 'x' : 32, 'y' : 24 },
            'world_origin' : { 'type' : 'XY', 'x' : 15, 'y' : 90 }
        },
        'scale5' : galsim.PixelScale(0.49),
        'scale6' : '$galsim.PixelScale(0.56)',
        'scale7' : '@image.scale6',
        'scale8' : { 'type': 'Eval', 'str': 'galsim.PixelScale(0.56)' },
        'scale9' : { 'type': 'Current', 'key': 'image.scale6' },
        'shear1' : {
            'type' : 'Shear',
            'scale' : 0.43,
            'shear' : galsim.Shear(g1=0.2, g2=0.3)
        },
        'shear2' : {
            'type' : 'Shear',
            'scale' : 0.43,
            'shear' : { 'type' : 'G1G2', 'g1' : 0.2, 'g2' : 0.3 },
            'origin' : { 'type' : 'XY', 'x' : 32, 'y' : 24 },
            'world_origin' : { 'type' : 'XY', 'x' : 15, 'y' : 90 }
        },
        'jac1' : {
            'type' : 'Jacobian',
            'dudx' : 0.2,
            'dudy' : 0.02,
            'dvdx' : -0.04,
            'dvdy' : 0.21
        },
        'jac2' : {
            'type' : 'Affine',  # Affine is really just a synonym for Jacobian here.
            'dudx' : 0.2,
            'dudy' : 0.02,
            'dvdx' : -0.04,
            'dvdy' : 0.21
        },
        'jac3' : {
            'type' : 'Jacobian',
            'dudx' : 0.2,
            'dudy' : 0.02,
            'dvdx' : -0.04,
            'dvdy' : 0.21,
            'origin' : { 'type' : 'XY', 'x' : 32, 'y' : 24 },
            'world_origin' : { 'type' : 'XY', 'x' : 15, 'y' : 90 }
        },
        'jac4' : {
            'type' : 'Affine',
            'dudx' : 0.2,
            'dudy' : 0.02,
            'dvdx' : -0.04,
            'dvdy' : 0.21,
            'origin' : { 'type' : 'XY', 'x' : 32, 'y' : 24 },
            'world_origin' : { 'type' : 'XY', 'x' : 15, 'y' : 90 }
        },
        'uv' : {
            'type' : 'UVFunction',
            'ufunc' : '0.05 * numpy.exp(1. + x/100.)',
            'vfunc' : '0.05 * np.exp(1. + y/100.)',
            'xfunc' : '100. * (np.log(u*20.) - 1.)',
            'yfunc' : '100. * (np.log(v*20.) - math.sqrt(1.))',
            'origin' : 'center',
        },
        'radec' : {
            'type' : 'RaDecFunction',
            'ra_func' : '0.05 * numpy.exp(1. + x/100.) * galsim.hours / galsim.radians',
            'dec_func' : '0.05 * np.exp(math.sqrt(1.) + y/100.) * galsim.degrees / galsim.radians',
            'origin' : 'center',
        },
        'fits1' : {
            'type' : 'Fits',
            'file_name' : 'tpv.fits',
            'dir' : 'fits_files'
        },
        'fits2' : {
            'type' : 'Fits',
            'file_name' : 'fits_files/tpv.fits',
        },
        'tan1' : {
            'type' : 'Tan',
            'dudx' : 0.2,
            'dudy' : 0.02,
            'dvdx' : -0.04,
            'dvdy' : 0.21,
            'units' : 'arcsec',
            'origin' : 'center',
            'ra' : '19.3 hours',
            'dec' : '-33.1 degrees',
        },
        'tan2' : {
            'type' : 'Tan',
            'dudx' : 0.2,
            'dudy' : 0.02,
            'dvdx' : -0.04,
            'dvdy' : 0.21,
            'ra' : '19.3 hours',
            'dec' : '-33.1 degrees',
        },
        'list' : {
            'type' : 'List',
            'items' : [
                galsim.PixelScale(0.12),
                { 'type': 'PixelScale', 'scale' : 0.23 },
                { 'type': 'PixelScale', 'scale' : '0.34' }
            ]
        },
        # This needs to be done after 'scale2', so call it zref to make sure it happens
        # alphabetically after scale2 in a sorted list.
        'zref' : '$(@image.scale2).withOrigin(galsim.PositionD(22,33))',
        'bad1' : 34,
        'bad2' : { 'type' : 'Invalid' },
        'bad3' : { 'type' : 'List', 'items' : galsim.PixelScale(0.12), },
        'bad4' : { 'type' : 'List', 'items' : "galsim.PixelScale(0.12)", },
        'bad5' : {
            'type' : 'List',
            'items' : [ galsim.PixelScale(0.12), galsim.PixelScale(0.23) ],
            'index' : -1
        },
        'bad6' : {
            'type' : 'List',
            'items' : [ galsim.PixelScale(0.12), galsim.PixelScale(0.23) ],
            'index' : 2
        },
    }

    reference = {
        'scale1' : galsim.PixelScale(0.34),  # Missing from config.  Uses pixel_scale
        'scale2' : galsim.PixelScale(0.43),
        'scale3' : galsim.OffsetWCS(scale=0.43, origin=galsim.PositionD(32,24)),
        'scale4' : galsim.OffsetWCS(scale=0.43, origin=galsim.PositionD(32,24),
                                    world_origin=galsim.PositionD(15,90)),
        'scale5' : galsim.PixelScale(0.49),
        'scale6' : galsim.PixelScale(0.56),
        'scale7' : galsim.PixelScale(0.56),
        'scale8' : galsim.PixelScale(0.56),
        'scale9' : galsim.PixelScale(0.56),
        'shear1' : galsim.ShearWCS(scale=0.43, shear=galsim.Shear(g1=0.2, g2=0.3)),
        'shear2' : galsim.OffsetShearWCS(scale=0.43, shear=galsim.Shear(g1=0.2, g2=0.3),
                                         origin=galsim.PositionD(32,24),
                                         world_origin=galsim.PositionD(15,90)),
        'jac1' : galsim.JacobianWCS(0.2, 0.02, -0.04, 0.21),
        'jac2' : galsim.JacobianWCS(0.2, 0.02, -0.04, 0.21),
        'jac3' : galsim.AffineTransform(0.2, 0.02, -0.04, 0.21,
                                  origin=galsim.PositionD(32,24),
                                  world_origin=galsim.PositionD(15,90)),
        'jac4' : galsim.AffineTransform(0.2, 0.02, -0.04, 0.21,
                                  origin=galsim.PositionD(32,24),
                                  world_origin=galsim.PositionD(15,90)),
        'uv' : galsim.UVFunction(
                ufunc = lambda x,y: 0.05 * np.exp(1. + x/100.),
                vfunc = lambda x,y: 0.05 * np.exp(1. + y/100.),
                xfunc = lambda u,v: 100. * (np.log(u*20.) - 1.),
                yfunc = lambda u,v: 100. * (np.log(v*20.) - 1.),
                origin = config['image_center']),
        'radec' : galsim.RaDecFunction(
                ra_func = lambda x,y: 0.05 * np.exp(1. + x/100.) * galsim.hours / galsim.radians,
                dec_func = lambda x,y: 0.05 * np.exp(1. + y/100.) * galsim.degrees / galsim.radians,
                origin = config['image_center']),
        'fits1' : galsim.FitsWCS('fits_files/tpv.fits'),
        'fits2' : galsim.FitsWCS('fits_files/tpv.fits'),
        'tan1' : galsim.TanWCS(affine=galsim.AffineTransform(0.2, 0.02, -0.04, 0.21,
                                                             origin=config['image_center']),
                               world_origin=galsim.CelestialCoord(19.3*galsim.hours,
                                                                  -33.1*galsim.degrees),
                               units=galsim.arcsec),
        'tan2' : galsim.TanWCS(affine=galsim.AffineTransform(0.2, 0.02, -0.04, 0.21),
                               world_origin=galsim.CelestialCoord(19.3*galsim.hours,
                                                                  -33.1*galsim.degrees)),
        'list' : galsim.PixelScale(0.12),
        'zref' : galsim.PixelScale(0.43).withOrigin(galsim.PositionD(22,33)),
    }

    for key in sorted(reference.keys()):
        wcs = galsim.config.BuildWCS(config['image'], key, config)
        ref = reference[key]

        print(key,'=',wcs)
        #print('ref =',ref)

        p = galsim.PositionD(23,12)
        #print(wcs.toWorld(p), ref.toWorld(p))
        if ref.isCelestial():
            np.testing.assert_almost_equal(wcs.toWorld(p).rad, ref.toWorld(p).rad)
        else:
            np.testing.assert_almost_equal(wcs.toWorld(p).x, ref.toWorld(p).x)
            np.testing.assert_almost_equal(wcs.toWorld(p).y, ref.toWorld(p).y)
            #print(wcs.toImage(p), ref.toImage(p))
            np.testing.assert_almost_equal(wcs.toImage(p).x, ref.toImage(p).x)
            np.testing.assert_almost_equal(wcs.toImage(p).y, ref.toImage(p).y)

    # If we build something again with the same index, it should get the current value
    wcs = galsim.config.BuildWCS(config['image'], 'shear2', config)
    ref = reference['shear2']
    assert wcs == ref
    assert wcs is config['image']['shear2']['current'][0]

    # List should return different wcs when indexed differently.
    config['image_num'] = 1
    wcs = galsim.config.BuildWCS(config['image'], 'list', config)
    assert wcs == galsim.PixelScale(0.23)
    config['image_num'] = 2
    wcs = galsim.config.BuildWCS(config['image'], 'list', config)
    assert wcs == galsim.PixelScale(0.34)

    # Finally, check the default if there is no wcs or pixel_scale item
    wcs = galsim.config.BuildWCS(config, 'wcs', config)
    assert wcs == galsim.PixelScale(1.0)

    for bad in ['bad1', 'bad2', 'bad3', 'bad4', 'bad5', 'bad6']:
        with assert_raises(galsim.GalSimConfigError):
            galsim.config.BuildWCS(config['image'], bad, config)

    # Base class usage is invalid
    builder = galsim.config.wcs.WCSBuilder()
    assert_raises(NotImplementedError, builder.buildWCS, config, config, logger=None)

@timer
def test_index_key():
    """Test some aspects of setting non-default index_key values
    """
    nfiles = 3
    nimages = 3
    nx = 3
    ny = 3
    n_per_file = nimages * nx * ny
    n_per_image = nx * ny

    # First generate using the config layer.
    config = galsim.config.ReadConfig('config_input/index_key.yaml')[0]

    # Normal sequential
    config1 = galsim.config.CopyConfig(config)
    # Note: Using BuildFiles(config1) would normally work, but it has an extra copy internally,
    # which messes up some of the current checks later.
    for n in range(nfiles):
        galsim.config.BuildFile(config1, file_num=n, image_num=n*nimages, obj_num=n*n_per_file)
    images1 = [ galsim.fits.readMulti('output/index_key%02d.fits'%n) for n in range(nfiles) ]

    if __name__ == '__main__':
        # For nose tests skip these 3 to save some time.
        # images5 is really the hardest test, and images1 is the easiest, so those two will
        # give good diagnostics for any errors.

        # Multiprocessing files
        config2 = galsim.config.CopyConfig(config)
        config2['output']['nproc'] = nfiles
        for n in range(nfiles):
            galsim.config.BuildFile(config2, file_num=n, image_num=n*nimages, obj_num=n*n_per_file)
        images2 = [ galsim.fits.readMulti('output/index_key%02d.fits'%n) for n in range(nfiles) ]

        # Multiprocessing images
        config3 = galsim.config.CopyConfig(config)
        config3['image']['nproc'] = nfiles
        for n in range(nfiles):
            galsim.config.BuildFile(config3, file_num=n, image_num=n*nimages, obj_num=n*n_per_file)
        images3 = [ galsim.fits.readMulti('output/index_key%02d.fits'%n) for n in range(nfiles) ]

        # New config for each file
        config4 = [ galsim.config.CopyConfig(config) for n in range(nfiles) ]
        for n in range(nfiles):
            galsim.config.SetupConfigFileNum(config4[n], n, n*nimages, n*n_per_file)
            galsim.config.SetupConfigRNG(config4[n])
        images4 = [ galsim.config.BuildImages(nimages, config4[n],
                                              image_num=n*nimages, obj_num=n*n_per_file)
                    for n in range(nfiles) ]

    # New config for each image
    config5 = [ galsim.config.CopyConfig(config) for n in range(nfiles) ]
    for n in range(nfiles):
        galsim.config.SetupConfigFileNum(config5[n], n, n*nimages, n*n_per_file)
        galsim.config.SetupConfigRNG(config5[n])

    images5 = [ [ galsim.config.BuildImage(galsim.config.CopyConfig(config5[n]),
                                           image_num=n*nimages+i,
                                           obj_num=n*n_per_file + i*n_per_image)
                  for i in range(nimages) ]
                for n in range(nfiles) ]

    # Now generate by hand
    for n in range(nfiles):
        seed = 12345 + n*n_per_file
        file_rng = galsim.UniformDeviate(seed)
        fwhm = file_rng() * 0.2 + 0.9
        e = 0.2 + 0.05 * n
        beta = file_rng() * 2 * np.pi * galsim.radians
        kolm = galsim.Kolmogorov(fwhm=fwhm)
        psf_shear = galsim.Shear(e=e, beta=beta)
        kolm = kolm.shear(psf_shear)
        airy = galsim.Airy(lam=700, diam=4)
        psf = galsim.Convolve(kolm, airy)
        print('fwhm, shear = ',fwhm,psf_shear._g)
        ellip_e1 = file_rng() * 0.4 - 0.2

        for i in range(nimages):
            if i == 0:
                image_rng = file_rng
            else:
                seed = 12345 + n*n_per_file + i*n_per_image
                image_rng = galsim.UniformDeviate(seed)
            im = galsim.ImageF(32*3, 32*3, scale=0.3)
            ellip_e2 = image_rng() * 0.4 - 0.2
            ellip = galsim.Shear(e1=ellip_e1, e2=ellip_e2)
            shear_g2 = image_rng() * 0.04 - 0.02

            for k in range(nx*ny):
                seed = 12345 + n*n_per_file + i*n_per_image + k + 1
                obj_rng = galsim.UniformDeviate(seed)
                kx = k % 3
                ky = k // 3
                b = galsim.BoundsI(32*kx+1, 32*kx+32, 32*ky+1, 32*ky+32)
                stamp = im[b]
                flux = 100 + k*100
                hlr = 0.5 + i*0.5
                gal = galsim.Exponential(half_light_radius=hlr, flux=flux)
                while True:
                    shear_g1 = obj_rng() * 0.04 - 0.02
                    bd = galsim.BinomialDeviate(obj_rng, N=1, p=0.2)
                    if bd() == 0: break;
                shear = galsim.Shear(g1=shear_g1, g2=shear_g2)
                gal = gal.shear(ellip).shear(shear)
                print(n,i,k,flux,hlr,ellip._g,shear._g)
                final = galsim.Convolve(psf, gal)
                final.drawImage(stamp)

            if __name__ == '__main__':
                im.write('output/test_index_key%02d_%02d.fits'%(n,i))
                images5[n][i].write('output/test_index_key%02d_%02d_5.fits'%(n,i))
            np.testing.assert_array_equal(im.array, images1[n][i].array,
                                          "index_key parsing failed for sequential BuildFiles run")
            if __name__ == '__main__':
                np.testing.assert_array_equal(im.array, images2[n][i].array,
                                              "index_key parsing failed for output.nproc > 1")
                np.testing.assert_array_equal(im.array, images3[n][i].array,
                                              "index_key parsing failed for image.nproc > 1")
                np.testing.assert_array_equal(im.array, images4[n][i].array,
                                              "index_key parsing failed for BuildImages")
            np.testing.assert_array_equal(im.array, images5[n][i].array,
                                          "index_key parsing failed for BuildImage")

    # Check that current values get removed properly for various options to RemoveCurrent
    assert 'current' in config1['psf']
    assert 'current' in config1['psf']['items'][1]
    assert config1['psf']['items'][1]['current'][1]  # Index 1 in current is "safe"
    assert 'current' in config1['gal']
    assert 'current' in config1['gal']['ellip']
    assert 'current' in config1['gal']['shear']

    galsim.config.RemoveCurrent(config1, keep_safe=True, index_key='obj_num')
    assert 'current' in config1['psf']
    assert 'current' in config1['psf']['items'][1]
    assert 'current' not in config1['gal']
    assert 'current' in config1['gal']['ellip']
    assert 'current' not in config1['gal']['shear']

    galsim.config.RemoveCurrent(config1, keep_safe=True)
    assert 'current' not in config1['psf']
    assert 'current' in config1['psf']['items'][1]
    assert 'current' not in config1['gal']
    assert 'current' not in config1['gal']['ellip']
    assert 'current' not in config1['gal']['shear']

    galsim.config.RemoveCurrent(config1)
    assert 'current' not in config1['psf']
    assert 'current' not in config1['psf']['items'][1]
    assert 'current' not in config1['gal']
    assert 'current' not in config1['gal']['ellip']
    assert 'current' not in config1['gal']['shear']

    # Finally check for invalid index_key
    config['psf']['index_key'] = 'psf_num'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config)


@timer
def test_multirng():
    """Test using multiple rngs.

    This models a run where the galaxies are the same for 3 images, then a new set for the next
    3 images.  This includes both the galaxy positions and the shear from a power spectrum.
    The psf changes each image (also from a power spectrum) and the telescope pointing moves
    around a bit within the field being observed.

    Actually, this tests a few features.
    - Multiple rngs (random_seed being a list and using rng_num)
    - Multiple input fields (although tests in test_config_value.py also do this)
    - Using a non-default build_index for power_spectrum
    """
    if __name__ == '__main__':
        nimages = 6
        ngals = 20
        logger = logging.getLogger('test_multirng')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        #logger.setLevel(logging.DEBUG)
    else:
        nimages = 3
        ngals = 3
        logger = None

    # First generate using the config layer.
    config = galsim.config.ReadConfig('config_input/multirng.yaml')[0]
    config['image']['nobjects'] = ngals
    config1 = galsim.config.CopyConfig(config)  # Make sure the config dict is clean for each pass.
    config2 = galsim.config.CopyConfig(config)
    config3 = galsim.config.CopyConfig(config)

    images1 = galsim.config.BuildImages(nimages, config1, logger=logger)
    config2['image']['nproc'] = 6
    images2 = galsim.config.BuildImages(nimages, config2)
    images3 = [ galsim.config.BuildImage(galsim.config.CopyConfig(config),
                                         image_num=n, obj_num=n*ngals)
                for n in range(nimages) ]

    # Now generate by hand
    psf_ps = galsim.PowerSpectrum('(k**2 + (1./180)**2)**(-11./6.)',
                                  '(k**2 + (1./180)**2)**(-11./6.)',
                                  units=galsim.arcsec)
    gal_ps = galsim.PowerSpectrum('3.5e-8 * (k/10)**-1.4', units=galsim.radians)

    for n in range(nimages):
        seed = 12345 + n*ngals
        rng = galsim.UniformDeviate(seed)
        centeru = rng() * 10. - 5.
        centerv = rng() * 10. - 5.
        wcs = galsim.OffsetWCS(scale=0.1, world_origin=galsim.PositionD(centeru,centerv),
                               origin=galsim.PositionD(128.5,128.5))
        im = galsim.ImageF(256, 256, wcs=wcs)
        world_center = im.wcs.toWorld(im.true_center)
        psf_ps.buildGrid(grid_spacing=1.0, ngrid=30, rng=rng, center=world_center, variance=0.1)
        ps_rng = galsim.UniformDeviate(12345 + 31415 + (n//3))
        if n % 3 == 0:
            gal_ps.buildGrid(grid_spacing=1.0, ngrid=50, rng=ps_rng, center=galsim.PositionD(0,0))
        for i in range(ngals):
            seedb = 123456789 + (n//3)*ngals + i + 1
            rngb = galsim.UniformDeviate(seedb)
            u = rngb() * 50. - 25.
            v = rngb() * 50. - 25.
            world_pos = galsim.PositionD(u,v)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                psf_g1, psf_g2 = psf_ps.getShear(world_pos)
            if len(w) > 0:
                assert not psf_ps.bounds.includes(world_pos)
            psf = galsim.Moffat(fwhm=0.9, beta=2).shear(g1=psf_g1, g2=psf_g2)
            gal_g1, gal_g2 = gal_ps.getShear(world_pos)
            gal = galsim.Exponential(half_light_radius=1.3, flux=100).shear(g1=gal_g1, g2=gal_g2)
            print(n,i,u,v,psf_g1,psf_g2,gal_g1,gal_g2)
            image_pos = wcs.toImage(world_pos)
            ix = int(math.floor(image_pos.x+1))
            iy = int(math.floor(image_pos.y+1))
            offset = galsim.PositionD(image_pos.x-ix+0.5, image_pos.y-iy+0.5)
            stamp = galsim.Convolve(psf,gal).drawImage(scale=0.1, offset=offset)
            stamp.setCenter(ix,iy)
            b = stamp.bounds & im.bounds
            if b.isDefined():
                im[b] += stamp[b]
        im.addNoise(galsim.GaussianNoise(sigma=0.001, rng=rng))
        if __name__ == '__main__':
            im.write('output/test_multirng%02d.fits'%n)
        np.testing.assert_array_equal(im.array, images1[n].array)
        np.testing.assert_array_equal(im.array, images2[n].array)
        np.testing.assert_array_equal(im.array, images3[n].array)

    # Finally, test invalid rng_num
    config4 = galsim.config.CopyConfig(config)
    config4['image']['world_pos']['rng_num'] = -1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config4)
    config5 = galsim.config.CopyConfig(config)
    config5['image']['world_pos']['rng_num'] = 20
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config5)
    config6 = galsim.config.CopyConfig(config)
    config6['image']['world_pos']['rng_num'] = 1.3
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config6)
    config7 = galsim.config.CopyConfig(config)
    config7['image']['world_pos']['rng_num'] = 1
    config7['image']['random_seed'] = 12345
    del config7['input']
    del config7['psf']['ellip']
    del config7['gal']['shear']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config7)


@timer
def test_template():
    """Test various uses of the template keyword
    """
    # Use the multirng.yaml config file from the above test as a convenient template source
    config = {
        # This copies everything, but we'll override a few things
        "template" : "config_input/multirng.yaml",

        # Specific fields can be overridden
        "output" : { "file_name" : "test_template.fits" },

        # Can override non-top-level fields
        "gal.shear" : { "type" : "E1E2", "e1" : 0.2, "e2" : 0 },

        # Including within a list
        "input.power_spectrum.1.grid_spacing" : 2,

        # Can add new fields to an existing dict
        "gal.magnify" : { "type" : "PowerSpectrumMagnification", "num" : 1 },

        # Can use specific fields from the template file rather than the whole thing using :
        "gal.ellip" : { "template" : "config_input/multirng.yaml:psf.ellip" },

        # Using a zero-length string deletes an item
        "gal.half_light_radius" : "",
        "gal.scale_radius" : 1.6,

        # Check that template items work inside a list.
        "psf" : {
            "type" : "List",
            "items" : [
                { "template" : "config_input/multirng.yaml:psf" },
                { "type" : "Gaussian", "sigma" : 0.3 },
                # Omitting the file name before : means use the current config file instead.
                { "template" : ":psf.items.1", "sigma" : 0.4 },
            ]
        }
    }
    config2 = galsim.config.ReadConfig('config_input/multirng.yaml')[0]

    galsim.config.ProcessAllTemplates(config)

    assert config['image'] == config2['image']
    assert config['output'] != config2['output']
    assert config['output'] == { "file_name" : "test_template.fits" }

    assert config['gal']['type'] == 'Exponential'
    assert config['gal']['flux'] == 100
    assert config['gal']['shear'] == { "type" : "E1E2", "e1" : 0.2, "e2" : 0 }
    assert config['gal']['magnify'] == { "type" : "PowerSpectrumMagnification", "num" : 1 }
    assert config['gal']['ellip'] == { "type" : "PowerSpectrumShear", "num" : 0 }
    assert 'half_light_radius' not in config['gal']
    assert config['gal']['scale_radius'] == 1.6

    assert config['psf']['type'] == 'List'
    assert config['psf']['items'][0] == { "type": "Moffat", "beta": 2, "fwhm": 0.9,
                                          "ellip": { "type" : "PowerSpectrumShear", "num" : 0 } }
    assert config['psf']['items'][1] == { "type": "Gaussian", "sigma" : 0.3 }
    assert config['psf']['items'][2] == { "type": "Gaussian", "sigma" : 0.4 }

@timer
def test_variable_cat_size():
    """Test that some automatic nitems calculations work with variable input catalog sizes
    """
    config = {
        'gal': {
            'type': 'Gaussian',
            'half_light_radius': { 'type': 'Catalog', 'col': 0 },
            'shear': {
                'type': 'G1G2',
                'g1': { 'type': 'Catalog', 'col': 1 },
                'g2': { 'type': 'Catalog', 'col': 2 }
            },
            'flux': 1.7
        },
        'stamp': {
            'size': 33   # Use odd to avoid all the even-sized image centering complications
        },
        'image': {
            'type': 'Scattered',
            'size': 256,
            'image_pos': {
                'type': 'XY',
                'x': { 'type': 'Catalog', 'col': 3 },
                'y': { 'type': 'Catalog', 'col': 4 }
            }
        },
        'input': {
            'catalog': {
                'dir': 'config_input',
                'file_name': [ 'cat_3.txt', 'cat_5.txt' ]
            }
        }
    }

    logger = logging.getLogger('test_single')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    #logger.setLevel(logging.DEBUG)
    cfg_images = []
    galsim.config.SetupConfigFileNum(config, 0, 0, 0)
    galsim.config.ProcessInput(config, logger=logger)
    cfg_images.append(galsim.config.BuildImage(config, 0, 0, logger=logger))
    galsim.config.SetupConfigFileNum(config, 1, 1, 3)
    galsim.config.ProcessInput(config, logger=logger)
    cfg_images.append(galsim.config.BuildImage(config, 1, 3, logger=logger))

    # Build by hand to compare
    ref_images = []
    for cat_name in ['cat_3.txt', 'cat_5.txt']:
        cat = np.genfromtxt(os.path.join('config_input',cat_name), names=True, skip_header=1)
        im = galsim.ImageF(256,256,scale=1)
        for row in cat:
            gal = galsim.Gaussian(half_light_radius=row['hlr'], flux=1.7)
            gal = gal.shear(g1=row['e1'], g2=row['e2'])
            stamp = galsim.ImageF(33,33,scale=1)
            gal.drawImage(stamp)
            stamp.setCenter(row['x'],row['y'])
            im[stamp.bounds] += stamp
        ref_images.append(im)

    np.testing.assert_array_equal(cfg_images[0], ref_images[0])
    np.testing.assert_array_equal(cfg_images[1], ref_images[1])


class BlendSetBuilder(galsim.config.StampBuilder):
    """This is a stripped-down version of the BlendSetBuilder in examples/des/blend.py.
    Use this to test the validity of having a StampBuilder that doesn't use a simple
    GSObject for its "prof".
    """

    def setup(self, config, base, xsize, ysize, ignore, logger):
        """Do the appropriate setup for a Blend stamp.
        """
        self.ngal = galsim.config.ParseValue(config, 'n_neighbors', base, int)[0] + 1
        self.sep = galsim.config.ParseValue(config, 'sep', base, float)[0]
        ignore = ignore + ['n_neighbors', 'sep']
        return super(self.__class__, self).setup(config, base, xsize, ysize, ignore, logger)

    def buildProfile(self, config, base, psf, gsparams, logger):
        """
        Build a list of galaxy profiles, each convolved with the psf, to use for the blend image.
        """
        if (base['obj_num'] % self.ngal != 0):
            return None
        else:
            self.neighbor_gals = []
            for i in range(self.ngal-1):
                gal = galsim.config.BuildGSObject(base, 'gal', gsparams=gsparams, logger=logger)[0]
                self.neighbor_gals.append(gal)
                galsim.config.RemoveCurrent(base['gal'], keep_safe=True)

            rng = galsim.config.GetRNG(config, base, logger, 'BlendSet')
            ud = galsim.UniformDeviate(rng)
            self.neighbor_pos = [galsim.PositionI(int(ud()*2*self.sep-self.sep),
                                                  int(ud()*2*self.sep-self.sep))
                                 for i in range(self.ngal-1)]
            #print('neighbor positions = ',self.neighbor_pos)

            self.main_gal = galsim.config.BuildGSObject(base, 'gal', gsparams=gsparams,
                                                        logger=logger)[0]

            self.profiles = [ self.main_gal ]
            self.profiles += [ g.shift(p) for g, p in zip(self.neighbor_gals, self.neighbor_pos) ]
            if psf:
                self.profiles = [ galsim.Convolve(gal, psf) for gal in self.profiles ]
            return self.profiles

    def draw(self, profiles, image, method, offset, config, base, logger):
        nx = base['stamp_xsize']
        ny = base['stamp_ysize']
        wcs = base['wcs']

        if profiles is not None:
            bounds = galsim.BoundsI(galsim.PositionI(0,0))
            for pos in self.neighbor_pos:
                bounds += pos
            bounds = bounds.withBorder(max(nx,ny)//2 + 1)

            self.full_images = []
            for prof in profiles:
                im = galsim.ImageF(bounds=bounds, wcs=wcs)
                galsim.config.DrawBasic(prof, im, method, offset-im.true_center, config, base,
                                        logger)
                self.full_images.append(im)

        k = base['obj_num'] % self.ngal
        if k == 0:
            center_pos = galsim.PositionI(0,0)
        else:
            center_pos = self.neighbor_pos[k-1]
        xmin = int(center_pos.x) - nx//2 + 1
        ymin = int(center_pos.y) - ny//2 + 1
        self.bounds = galsim.BoundsI(xmin, xmin+nx-1, ymin, ymin+ny-1)

        image.setZero()
        image.wcs = wcs
        for full_im in self.full_images:
            assert full_im.bounds.includes(self.bounds)
            image += full_im[self.bounds]

        return image

@timer
def test_blend():
    """Test the functionality used by the BlendSet stamp type in examples/des/blend.py.
    Especially that it's internal "prof" is not just a single GSObject.
    """
    galsim.config.RegisterStampType('BlendSet', BlendSetBuilder())
    config = {
        'stamp' : {
            'type' : 'BlendSet',
            'n_neighbors' : 3,
            'sep' : 10,
            'size' : 64,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type' : 'Random', 'min': 1, 'max': 3 },
            'flux' : { 'type' : 'Random', 'min': 20, 'max': 300 },
        },
        'image' : {
            'type' : 'Single',
            'pixel_scale' : 0.5,
            'random_seed' : 1234,
        },
    }

    # First just check that this works correctly as is.
    galsim.config.SetupConfigImageNum(config, 0, 0)
    images = galsim.config.BuildImages(8,config)
    for i, im in enumerate(images):
        im.write('output/blend%02d.fits'%i)
    # Within each blendset, the images are shifted copies of each other.
    np.testing.assert_array_equal(images[1].array[3:64,16:64], images[0].array[0:61,9:57])
    np.testing.assert_array_equal(images[2].array[1:62,6:54], images[0].array[0:61,9:57])
    np.testing.assert_array_equal(images[3].array[0:61,0:48], images[0].array[0:61,9:57])

    np.testing.assert_array_equal(images[5].array[0:55,9:64], images[4].array[9:64,9:64])
    np.testing.assert_array_equal(images[6].array[5:60,1:56], images[4].array[9:64,9:64])
    np.testing.assert_array_equal(images[7].array[0:55,0:55], images[4].array[9:64,9:64])

    # If there is a current_image, then updateSkip requires special handling here.
    config['current_image'] = galsim.Image(64,64)
    im8 = galsim.config.BuildStamp(config, obj_num=8)

    # Some reject items are invalid when using this kind of stamp.
    config['stamp']['min_flux_frac'] = 0.3
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config, obj_num=8)
    del config['stamp']['min_flux_frac']
    config['stamp']['min_snr'] = 20
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config, obj_num=8)
    del config['stamp']['min_snr']
    config['stamp']['max_snr'] = 200
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamp(config, obj_num=8)


if __name__ == "__main__":
    test_single()
    test_positions()
    test_phot()
    test_reject()
    test_snr()
    test_ring()
    test_scattered()
    test_scattered_whiten()
    test_tiled()
    test_njobs()
    test_wcs()
    test_index_key()
    test_multirng()
    test_template()
    test_variable_cat_size()
    test_blend()
