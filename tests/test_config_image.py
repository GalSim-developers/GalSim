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
import re
import warnings
import astropy.units as u

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
    logger.setLevel(logging.INFO)

    # Test a little bit of the LoggerWrapper functionality
    logger_wrapper = galsim.config.LoggerWrapper(logger)
    assert logger_wrapper.level == logger.getEffectiveLevel()
    assert logger_wrapper.getEffectiveLevel() == logger.getEffectiveLevel()
    assert logger_wrapper.isEnabledFor(logging.WARNING)
    assert logger_wrapper.isEnabledFor(logging.CRITICAL)
    assert not logger_wrapper.isEnabledFor(logging.DEBUG)

    # smoke test for critical calls
    # these are not normally used by galsim so a test here is needed
    logger_wrapper.critical("blah blah")
    none_logger_wrapper = galsim.config.LoggerWrapper(None)
    none_logger_wrapper.critical("blah blah")

    im1_list = []
    nimages = 6
    first_seed = galsim.BaseDeviate(1234).raw()
    for k in range(nimages):
        ud = galsim.UniformDeviate(first_seed + k + 1)
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
                                      config['stamp'], config, logger,
                                      scale=1.0, dtype=np.float32)
        np.testing.assert_array_equal(im5.array, im1.array)

        # Both scale and float are valid options in config too, with these defaults.
        config['stamp']['scale'] = 1.0
        config['stamp']['dtype'] = 'np.float32'
        im6 = galsim.config.DrawBasic(gal, im3.copy(), 'auto', galsim.PositionD(0,0),
                                      config['stamp'], config, logger)
        np.testing.assert_array_equal(im6.array, im1.array)
        config['stamp'].pop('scale')
        config['stamp'].pop('dtype')

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

    # Can make images in double precision if desired.  Here use fact that Basic is default.
    config['stamp'] = { 'dtype': 'np.float64' }
    im8_list = galsim.config.BuildImages(nimages, config)
    for k in range(nimages):
        im1 = im1_list[k]
        im8 = im8_list[k]
        assert im1.dtype is np.float32
        assert im8.dtype is np.float64
        np.testing.assert_allclose(im8.array/gal.flux, im1.array/gal.flux, atol=1.e-8)

    # dtype is also allowed in image field.  Either np or numpy prefix works.
    config['stamp'] = {}
    config['image']['dtype'] = 'numpy.float64'
    im8_list = galsim.config.BuildImages(nimages, config)
    for k in range(nimages):
        im1 = im1_list[k]
        im8 = im8_list[k]
        assert im1.dtype is np.float32
        assert im8.dtype is np.float64
        np.testing.assert_allclose(im8.array/gal.flux, im1.array/gal.flux, atol=1.e-8)
    config['image'].pop('dtype')

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
    config['stamp'] = { 'dtype': 'invalid' }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['stamp'] = { 'dtype': 'np.float100' }
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
    config['stamp'] = { 'n_photons' : -1, 'draw_method' : 'phot' }
    with assert_raises(galsim.GalSimRangeError):
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
    config['image'] = { 'type' : 'Single', 'xsize' : 32 }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image'] = { 'type' : 'Single', 'xsize' : 0, 'ysize' : 32 }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image'] = { 'type' : 'Single', 'dtype' : 'invalid' }
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
    im1 = gal.drawImage(nx=21, ny=21, scale=1)
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

    # Can also set world_pos in image.
    config['image']['world_pos'] = config['stamp']['world_pos']
    del config['stamp']['world_pos']
    im5 = galsim.config.BuildImage(config, logger=logger)
    np.testing.assert_array_equal(im5.array, im1.array)
    assert im5.bounds == im1.bounds

    # It is also valid to give both world_pos and image_pos in the image field for Single.
    config['image']['image_pos'] = config['image']['world_pos']
    im6 = galsim.config.BuildImage(config, logger=logger)
    np.testing.assert_array_equal(im6.array, im1.array)
    assert im6.bounds == im1.bounds

    # Single is the default image type, so in this case, don't need the image field.
    config['stamp']['image_pos'] = config['image']['image_pos']
    del config['image']
    del config['stamp']['_done']
    im7, _ = galsim.config.BuildStamp(config, logger=logger)
    np.testing.assert_array_equal(im7.array, im1.array)
    assert im7.bounds == im1.bounds

    del config['image']
    del config['stamp']['_done']
    im8, _ = galsim.config.BuildStamps(1, config, logger=logger)
    im8 = im8[0]
    np.testing.assert_array_equal(im8.array, im1.array)
    assert im8.bounds == im1.bounds

    del config['image']
    del config['stamp']['_done']
    im9 = galsim.config.BuildImage(config, logger=logger)
    np.testing.assert_array_equal(im9.array, im1.array)
    assert im9.bounds == im1.bounds

    del config['image']
    del config['stamp']['_done']
    im10 = galsim.config.BuildImages(1, config, logger=logger)
    im10 = im10[0]
    np.testing.assert_array_equal(im10.array, im1.array)
    assert im10.bounds == im1.bounds

    config['stamp']['world_pos'] = { 'type' : 'Random' }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    del config['stamp']['world_pos']

    # If the size is set in image, then image_pos will put the object offset in this image.
    config['image'] = {
        'type' : 'Single',
        'size' : 64,
    }
    im11 = galsim.config.BuildImage(config)
    assert im11.bounds == galsim.BoundsI(1,64,1,64)
    assert im11[im2.bounds] == im2

    # If the offset is larger, then only the overlap is included
    config['image']['size'] = 50
    im12 = galsim.config.BuildImage(config)
    assert im12.bounds == galsim.BoundsI(1,50,1,50)
    b = im12.bounds & im2.bounds
    assert im12[b] == im2[b]

    # If the offset is large enough, none of the stamp is included.
    config['image']['size'] = 32
    im13 = galsim.config.BuildImage(config)
    assert im13.bounds == galsim.BoundsI(1,32,1,32)
    np.testing.assert_array_equal(im13.array, 0.)


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
    first_seed = galsim.BaseDeviate(1234).raw()
    ud = galsim.UniformDeviate(first_seed + 1)
    gal = galsim.Gaussian(sigma=1.7, flux=100)
    im1a = gal.drawImage(scale=1, method='phot', rng=ud)
    im1b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im1b.array, im1a.array)

    # Use a non-default number of photons
    del config['stamp']['_done']
    config['image']['n_photons'] = 300
    ud.seed(first_seed + 1)
    im2a = gal.drawImage(scale=1, method='phot', n_photons=300, rng=ud)
    im2b = galsim.config.BuildImage(config)
    print('image = ',config['image'])
    print('stamp = ',config['stamp'])
    np.testing.assert_array_equal(im2b.array, im2a.array)

    # Allow the flux to vary as a Poisson deviate even though n_photons is given
    del config['stamp']['_done']
    config['image']['poisson_flux'] = True
    ud.seed(first_seed + 1)
    im3a = gal.drawImage(scale=1, method='phot', n_photons=300, rng=ud, poisson_flux=True)
    im3b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im3b.array, im3a.array)

    # If max_extra_noise is given with n_photons, then ignore it.
    del config['stamp']['_done']
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
    ud.seed(first_seed + 1)
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
            'quick_skip' : '$obj_num == 10',
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
    orig_config = config.copy()

    if False:
        logger = logging.getLogger('test_reject')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        #logger.setLevel(logging.DEBUG)
    else:
        logger = galsim.config.LoggerWrapper(None)

    nimages = 11
    im_list = galsim.config.BuildStamps(nimages, config, do_noise=False, logger=logger)[0]
    # For this particular config, only 3 of them are real images.  The others were skipped.
    # The skipped ones are present in the list, but their flux is 0
    fluxes = [im.array.sum(dtype=float) if im is not None else 0 for im in im_list]
    expected_fluxes = [0, 2320, 0, 0, 0, 0, 1156, 2667, 0, 0, 0]
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
    assert "obj 8: Skipping because field skip=True" in cl.output
    assert "obj 8: Caught SkipThisObject: e = None" in cl.output
    assert "Skipping object 8" in cl.output
    assert "Object 6: Caught exception index=97 has gone past the number of entries" in cl.output
    assert "Object 5: Caught exception inner_radius must be less than radius (1.193147)" in cl.output
    assert "Object 4: Caught exception Unable to evaluate string 'math.sqrt(x)'" in cl.output
    assert "obj 0: reject evaluated to True" in cl.output
    assert "Object 0: Rejecting this object and rebuilding" in cl.output
    # This next two can end up with slightly different numerical values depending on numpy version
    # So use re.search rather than require an exact match.
    assert re.search(r"Object 0: Measured flux = 3086.30[0-9]* < 0.95 \* 3265.226572.", cl.output)
    assert re.search(r"Object 4: Measured snr = 79.888[0-9]* > 50.0.", cl.output)

    # 10 is quick skipped, so we don't even get a debug line for it.
    assert 'Stamp 10' not in cl.output
    assert 'obj 10' not in cl.output
    # Others all get at least that:
    for i in range(10):
        assert 'Stamp %d'%i in cl.output
        assert 'obj %d'%i in cl.output

    # For test coverage to get all branches, do min_snr and max_snr separately.
    del config['stamp']['max_snr']
    config['stamp']['min_snr'] = 26
    with CaptureLog() as cl:
        im_list2 = galsim.config.BuildStamps(nimages, config, do_noise=False, logger=cl.logger)[0]
    #print(cl.output)
    assert re.search(r"Object 6: Measured snr = 25.2741[0-9]* < 26.0.", cl.output)

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

    # Even lower, and we get a different limiting error.
    config['stamp']['retry_failures'] = 4
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildStamps(nimages, config, do_noise=False)
    try:
        with CaptureLog() as cl:
            galsim.config.BuildStamps(nimages, config, do_noise=False, logger=cl.logger)
    except (galsim.GalSimConfigError):
        pass
    #print(cl.output)
    assert "Rejected an object 5 times. If this is expected," in cl.output
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
    if galsim.config.UpdateNProc(2, nimages, config) > 1:
        assert re.search("Process-.: Exception caught when building stamp",cl.output)

    try:
        with CaptureLog() as cl:
            galsim.config.BuildImages(nimages, config, logger=cl.logger)
    except (ValueError,IndexError,galsim.GalSimError):
        pass
    #print(cl.output)
    if galsim.config.UpdateNProc(2, nimages, config) > 1:
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

    # There is a different path if all files raise an exception, rather than are rejected.
    config['stamp']['type'] = 'hello'
    config = galsim.config.CleanConfig(config)
    with CaptureLog() as cl:
        galsim.config.BuildFiles(nimages, config, logger=cl.logger)
    #print(cl.output)
    assert "No files were written.  All were either skipped or had errors." in cl.output

    # If we skip all objects, and don't have a definite size for them, then we get to a message
    # that no stamps were built.
    config['stamp']['type'] = 'Basic'
    config['gal']['skip'] = True
    galsim.config.RemoveCurrent(config)
    im_list3 = galsim.config.BuildStamps(nimages, config, do_noise=False)[0]
    assert all (im is None for im in im_list3)
    with CaptureLog() as cl:
        im_list3 = galsim.config.BuildStamps(nimages, config, do_noise=False, logger=cl.logger)[0]
    #print(cl.output)
    assert "No stamps were built.  All objects were skipped." in cl.output

    # Different message if nstamps=0, rather than all failures.
    with CaptureLog() as cl:
        galsim.config.BuildStamps(0, config, do_noise=False, logger=cl.logger)[0]
    assert "No stamps were built, since nstamps == 0." in cl.output

    # Likewise with BuildImages, but with a slightly different message.
    with CaptureLog() as cl:
        im_list4 = galsim.config.BuildImages(nimages, config, logger=cl.logger)
    assert "No images were built.  All were either skipped or had errors." in cl.output

    # Different message if nimages=0, rather than all failures.
    with CaptureLog() as cl:
        galsim.config.BuildImages(0, config, logger=cl.logger)
    assert "No images were built, since nimages == 0." in cl.output

    # And BuildFiles
    with CaptureLog() as cl:
        galsim.config.BuildFiles(nimages, config, logger=cl.logger)
    assert "No files were written.  All were either skipped or had errors." in cl.output

    # Different message if nfiles=0, rather than all failures.
    with CaptureLog() as cl:
        galsim.config.BuildFiles(0, config, logger=cl.logger)
    assert "No files were made, since nfiles == 0." in cl.output

    # Finally, with a fake logger, this covers the LoggerWrapper functionality.
    logger = galsim.config.LoggerWrapper(None)
    galsim.config.BuildFiles(nimages, config, logger=logger)

    # Now go back to the original config, and switch to skip_failures rather than retry.
    config = orig_config
    config['stamp']['skip_failures'] = True

    # With this and retry_failures, we get an error.
    with assert_raises(galsim.GalSimConfigValueError):
        galsim.config.BuildStamps(nimages, config, do_noise=False, logger=logger)

    del config['stamp']['retry_failures']
    im_list = galsim.config.BuildStamps(nimages, config, do_noise=False, logger=logger)[0]
    fluxes = [im.array.sum(dtype=float) if im is not None else 0 for im in im_list]
    # Everything gets skipped here.
    np.testing.assert_almost_equal(fluxes, 0, decimal=0)

    # Dial back some of the rejections and skips to get some images drawn.
    del config['stamp']['reject']
    del config['stamp']['min_flux_frac']
    del config['stamp']['max_snr']
    del config['stamp']['skip']
    del config['stamp']['quick_skip']
    im_list = galsim.config.BuildStamps(nimages, config, do_noise=False, logger=logger)[0]
    fluxes = [im.array.sum(dtype=float) if im is not None else 0 for im in im_list]
    expected_fluxes = [0, 76673, 0, 0, 24074, 0, 0, 9124, 0, 0, 0]
    np.testing.assert_almost_equal(fluxes, expected_fluxes, decimal=0)


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
            'dtype' : 'float'
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 1.7,
            'signal_to_noise' : 70,
        }
    }

    # Do the S/N calculation by hand.
    first_seed = galsim.BaseDeviate(1234).raw()
    ud = galsim.UniformDeviate(first_seed + 1)
    gal = galsim.Gaussian(sigma=1.7)
    im1a = gal.drawImage(nx=32, ny=32, scale=0.4, dtype=float)
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
    ud.seed(first_seed + 1)
    im2a = gal.drawImage(nx=32, ny=32, scale=0.4, method='phot', n_photons=100, rng=ud, dtype=float)
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
    stamps = galsim.config.BuildStamps(6, config)

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
            'type' : 'Exponential',
            'half_light_radius' : 2,
            'ellip' : galsim.Shear(e2=0.3)
        },
        # This time use a psf too to check where that gets applied.
        'psf' : {
            'type' : 'Gaussian',
            'fwhm' : 0.7,
        },
    }

    disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)
    psf = galsim.Gaussian(fwhm=0.7)

    galsim.config.SetupConfigImageNum(config, 0, 0)
    for k in range(25):
        galsim.config.SetupConfigObjNum(config, k)
        ring_builder.setup(config['stamp'], config, None, None, ignore, None)
        gal2a = ring_builder.buildProfile(config['stamp'], config, psf, {}, None)
        gal2b = disk.rotate(theta = k * 18 * galsim.degrees)
        gal2b = galsim.Convolve(gal2b,psf)
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

    # Check using BuildStamps with multiple processes gives the same answer as single proc.
    # This is somewhat non-trivial because each ring set needs to be done on a single proc.
    config['stamp']['num'] = 4
    config['gal']['flux'] = { 'type' : 'Random', 'min' : 10, 'max' : 100 }
    config['gal']['items'][0]['ellip'] = { 'type' : 'ETheta', 'e' : 0.3, 'theta' : { 'type' : 'Random' }}
    config['gal']['items'][1]['ellip'] = { 'type' : 'ETheta', 'e' : 0.1, 'theta' : { 'type' : 'Random' }}

    stamps1, _ = galsim.config.BuildStamps(20, config)
    config['image'] = { 'nproc' : 3 }
    stamps2, _ = galsim.config.BuildStamps(20, config)
    for s1,s2 in zip(stamps1, stamps2):
        np.testing.assert_array_equal(s1.array, s2.array)


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
            config = galsim.config.CopyConfig(base_config)
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

            # Check that image_pos can be in a stamp field, rather than image.
            config = galsim.config.CleanConfig(config)
            config['stamp'] = { 'image_pos' : base_config['image']['image_pos'] }
            del config['image']['image_pos']
            image2 = galsim.config.BuildImage(config)
            assert image == image2

            # Can also use world_pos instead.
            config = galsim.config.CleanConfig(config)
            del config['stamp']['image_pos']
            config['stamp']['world_pos'] = [ galsim.PositionD(x1*scale, y1*scale),
                                             galsim.PositionD(x2*scale, y2*scale),
                                             galsim.PositionD(x3*scale, y3*scale) ]
            image2 = galsim.config.BuildImage(config)
            assert image == image2

    config['image']['index_convention'] = 'invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['image'].pop('index_convention')
    config['image']['dtype'] = 'np.float100'  # Invalid dtype
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # Check that stamp_xsize, stamp_ysize, image_pos use the object count, rather than the
    # image count.
    config = galsim.config.CopyConfig(base_config)
    config['image'] = {
        'type' : 'Scattered',
        'size' : size,
        'pixel_scale' : scale,
        'index_convention' : 0,
        'stamp_xsize' : { 'type': 'Sequence', 'first' : stamp_size },
        'stamp_ysize' : { 'type': 'Sequence', 'first' : stamp_size },
        'image_pos' : { 'type' : 'List',
                        'items' : [ galsim.PositionD(x1,y1),
                                    galsim.PositionD(x2,y2),
                                    galsim.PositionD(x3,y3) ]
                      },
        'dtype' : 'float64',  # Can also use the type name without np if it's a valid numpy type.
        'nobjects' : 3
    }

    image = galsim.config.BuildImage(config)

    image2 = galsim.ImageD(size,size, scale=scale)
    image2.setZero()
    image2.setOrigin(0,0)
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
def test_scattered_noskip():
    """The default StampBuilder will automatically skip objects whose stamps are fully
    off the image.  But if someone uses a custom StampBuilder that doesn't do this, then
    the Scattered ImageBuilder has a guard to handle this appropriately.
    """
    class NoSkipStampBuilder(galsim.config.StampBuilder):
        def quickSkip(self, config, base):
            return False
        def getSkip(self, config, base, logger):
            return False
        def updateSkip(self, prof, image, method, offset, config, base, logger):
            return False
    galsim.config.RegisterStampType('NoSkip', NoSkipStampBuilder())

    config = {
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 1.2,
            'flux' : 100,
        },
        'image' : {
            'type' : 'Scattered',
            'size' : 200,
            'pixel_scale' : 0.2,
            'stamp_size' : 51,
            'image_pos' : {
                # Some will miss the image.
                'type' : 'XY',
                'x' : { 'type': 'Random', 'min': -100, 'max': 300 },
                'y' : { 'type': 'Random', 'min': -100, 'max': 300 },
            },
            'nobjects' : 50,
            'random_seed' : 1234,
            'obj_rng' : False,
        },
        'stamp' : {
            'type' : 'NoSkip',
        }
    }
    gal = galsim.Gaussian(flux=100, sigma=1.2)
    image = galsim.Image(200,200, scale=0.2)
    first_seed = galsim.BaseDeviate(1234).raw()
    ud = galsim.UniformDeviate(first_seed)
    for i in range(50):
        x = ud() * 400 - 100
        y = ud() * 400 - 100
        ix = int(np.floor(x+0.5))
        iy = int(np.floor(y+0.5))
        dx = x-ix
        dy = y-iy
        offset = galsim.PositionD(dx, dy)
        center = galsim.PositionI(ix, iy)

        stamp = gal.drawImage(nx=51, ny=51, offset=offset, scale=0.2)
        stamp.setCenter(center)
        print('gal %d: bounds = '%i,stamp.bounds)
        b = stamp.bounds & image.bounds
        if b.isDefined():
            image[b] += stamp[b]
        else:
            print('  gal %d is off the image'%i)

    # Repeat with config
    image2 = galsim.config.BuildImage(config)
    np.testing.assert_equal(image2.array, image.array)


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
            'nproc' : 2,
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

    first_seed = galsim.BaseDeviate(12345).raw()
    for k in range(nobjects):
        ud = galsim.UniformDeviate(first_seed + k + 1)

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
    first_seed = galsim.BaseDeviate(12345).raw()
    rng = galsim.BaseDeviate(first_seed)
    im1.addNoise(galsim.VariableGaussianNoise(rng, noise_im))
    im1.addNoise(galsim.GaussianNoise(rng, sigma=math.sqrt(variance-max_cv)))

    # Compare to what config builds
    im2 = galsim.config.BuildImage(config)
    np.testing.assert_almost_equal(im2.array, im1.array)

    # Should give a warning for the objects that fall off the edge
    # Note: CaptureLog doesn't work correctly in multiprocessing for some reason.
    # I haven't figured out what about the implementation fails, but it prints these
    # just fine when using a regular logger with nproc=2.  Oh well.
    config['image']['nproc'] = 1
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
            'obj_rng' : False,

            'noise' : { 'type': 'Gaussian', 'sigma': 0.5 }
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : '$image_pos.x + image_pos.y',
        },
    }

    seed = galsim.BaseDeviate(1234).raw()
    im1a = galsim.Image(nx * (xsize+xborder) - xborder, ny * (ysize+yborder) - yborder, scale=scale)
    ud = galsim.UniformDeviate(seed)  # Test obj_rng=False -- one ud for all.
    for j in range(ny):
        for i in range(nx):
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
    seed = galsim.BaseDeviate(1234).raw()
    ud = galsim.UniformDeviate(seed)
    for i in range(nx):
        for j in range(ny):
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
    im2a.addNoise(galsim.GaussianNoise(sigma=0.5, rng=ud))

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
    seed = galsim.BaseDeviate(1234).raw()
    i_list = []
    j_list = []
    for i in range(nx):
        for j in range(ny):
            i_list.append(i)
            j_list.append(j)
    ud = galsim.UniformDeviate(seed)
    galsim.random.permute(ud, i_list, j_list)
    for i,j in zip(i_list,j_list):
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
    im3a.addNoise(galsim.GaussianNoise(sigma=0.5, rng=ud))

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
    config['image']['yborder'] = -yborder

    # Test invalid dtype
    config['image']['dtype'] = 'np.float100'
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
            'dtype' : 'np.float64',

            'random_seed' : 1234,
            'obj_rng' : True,     # Gratuitous branch check.  This is the default.
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

    seed = galsim.BaseDeviate(1234).raw()
    rng = galsim.BaseDeviate(seed)
    ps = galsim.PowerSpectrum(e_power_function=lambda k: np.exp(-k**0.2))
    im4a = galsim.ImageD(nx*size, ny*size, scale=scale)
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
            stamp = galsim.ImageD(size,size, scale=scale)
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
    np.testing.assert_allclose(im4b.array, im4a.array, atol=1.e-14)

    # Also when built with multiprocessing.
    config['image']['nproc'] = 3
    im4c = galsim.config.BuildImage(config)
    np.testing.assert_allclose(im4c.array, im4a.array, atol=1.e-14)

    # If grid sizes aren't square, it also works properly, but with more complicated ngrid calc.
    config = galsim.config.CleanConfig(config)
    del config['image']['stamp_size']
    config['image']['stamp_xsize'] = xsize
    config['image']['stamp_ysize'] = ysize
    seed = galsim.BaseDeviate(1234).raw()
    rng = galsim.BaseDeviate(seed)
    im5a = galsim.ImageD(nx*xsize, ny*ysize, scale=scale)
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
            stamp = galsim.ImageD(xsize,ysize, scale=scale)
            stamp.setOrigin(xorigin,yorigin)

            sigma = ud() + 1
            flux = x + y
            gal = galsim.Gaussian(sigma=sigma, flux=flux)
            g1, g2 = ps.getShear(galsim.PositionD(x*scale,y*scale))
            gal = gal.shear(g1=g1, g2=g2)
            gal.drawImage(stamp)
            im5a[stamp.bounds] = stamp

    im5b = galsim.config.BuildImage(config)
    # Not sure why this isn't always exact, but GHA macos started failing with abs err=4e-16 here.
    np.testing.assert_allclose(im5b.array, im5a.array, atol=1.e-15)

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

    # For coverage purposes, check that if we try to ProcessInput without safe_only=True,
    # then the exception is raised.
    config = galsim.config.CopyConfig(config1)
    config['rng'] = object()
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ProcessInput(config, logger=logger, safe_only=False)


@timer
def test_wcs():
    """Test various wcs options"""
    config = {
        # We'll need this for some of the items below.
        'image_center' : galsim.PositionD(1024, 1024)
    }
    config['image'] = {
        'pixel_scale' : 0.34,
        'scale2' : { 'scale' : 0.43 },  # Default type is PixelScale
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
            # Default scale is the image.pixel_scale
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
        'shear2' : galsim.OffsetShearWCS(scale=0.34, shear=galsim.Shear(g1=0.2, g2=0.3),
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

        # Check actually using the config to draw an image.
        if key == 'scale1': continue
        config1 = {
            'gal': {'type': 'Gaussian', 'sigma': 3, 'flux': 100},
            'image': {
                'size': 64,
                'wcs': config['image'][key],
                'draw_method':'sb'
            }
        }
        image = galsim.config.BuildImage(config1)
        assert image.wcs == wcs

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

    # Check the various positions that get calculated and stored in the base config.
    for key in sorted(reference.keys()):
        wcs = galsim.config.BuildWCS(config['image'], key, config)
        image_pos = galsim.PositionD(23,12)
        world_pos = wcs.toWorld(image_pos)
        config['wcs'] = wcs
        world_center = wcs.toWorld(config['image_center'])

        # Test providing image_pos
        config['sky_pos'] = None
        galsim.config.SetupConfigStampSize(config, 0, 0, image_pos, None)
        assert config['image_pos'] == image_pos
        assert config['world_pos'] == world_pos
        if isinstance(wcs, galsim.wcs.CelestialWCS):
            assert config['sky_pos'] == world_pos
            print('uv_pos = ',config['uv_pos'])
            print('calculated uv_pos = ',world_center.project(world_pos, projection='gnomonic'))
            u,v = world_center.project(world_pos, projection='gnomonic')
            np.testing.assert_allclose((config['uv_pos'].x, config['uv_pos'].y),
                                       (u/galsim.arcsec, v/galsim.arcsec))
        else:
            assert config['sky_pos'] is None
            assert config['uv_pos'] == world_pos

        if key == 'radec': continue  # Can't do world->image for radec

        # Test providing world_pos
        config['sky_pos'] = None
        galsim.config.SetupConfigStampSize(config, 0, 0, None, world_pos)
        np.testing.assert_allclose((config['image_pos'].x, config['image_pos'].y),
                                    (image_pos.x, image_pos.y))
        assert config['world_pos'] == world_pos
        if isinstance(wcs, galsim.wcs.CelestialWCS):
            assert config['sky_pos'] == world_pos
            u,v = world_center.project(world_pos, projection='gnomonic')
            np.testing.assert_allclose((config['uv_pos'].x, config['uv_pos'].y),
                                       (u/galsim.arcsec, v/galsim.arcsec))
        else:
            assert config['sky_pos'] is None
            assert config['uv_pos'] == world_pos

    # Finally, check the default if there is no wcs or pixel_scale item
    del config['wcs']
    wcs = galsim.config.BuildWCS(config, 'wcs', config)
    assert wcs == galsim.PixelScale(1.0)

    for bad in ['bad1', 'bad2', 'bad3', 'bad4', 'bad5', 'bad6']:
        with assert_raises(galsim.GalSimConfigError):
            galsim.config.BuildWCS(config['image'], bad, config)

    # Base class usage is invalid
    builder = galsim.config.wcs.WCSBuilder()
    assert_raises(NotImplementedError, builder.buildWCS, config, config, logger=None)


@timer
def test_bandpass():
    """Test various bandpass options"""
    config = {
        'bp1' : {
            'file_name' : 'chromatic_reference_images/simple_bandpass.dat',
            'wave_type' : 'nm',
        },
        'bp2' : {
            'type' : 'FileBandpass',
            'file_name' : 'ACS_wfc_F814W.dat',
            'wave_type' : u.nm,
            'thin' : [1.e-4, 1.e-5, 1.e-6],
            'blue_limit': 7000*u.Angstrom,  # Try mismatched units
            'red_limit': 9500*u.Angstrom,
        },
        'bp3' : galsim.Bandpass('LSST_g.dat', 'nm'),

        'current1' : '@bp1',
        'current2' : {
            'type' : 'Current',
            'key' : 'bp3',
        },

        'eval1' : '$@bp2 * 0.5',
        'eval2' : {
            'type' : 'Eval',
            'str' : '@bp1 * @bp3'
        },

        'bpz' : {
            'file_name' : 'chromatic_reference_images/simple_bandpass.dat',
            'wave_type' : 'nm',
            'zeropoint' : 'Vega',
        },

        'bad1' : 34,
        'bad2' : { 'type' : 'Invalid' },
    }
    config['index_key'] = 'obj_num'
    config['obj_num'] = 0

    bp1 = galsim.config.BuildBandpass(config, 'bp1', config)[0]
    assert bp1 == galsim.Bandpass('chromatic_reference_images/simple_bandpass.dat', wave_type='nm')

    bp2 = galsim.config.BuildBandpass(config, 'bp2', config)[0]
    bp2b = galsim.Bandpass('ACS_wfc_F814W.dat', 'nm', blue_limit=700, red_limit=950)
    bp2c = bp2b.thin(1.e-4)
    assert bp2 == bp2c

    bp3 = galsim.config.BuildBandpass(config, 'bp3', config)[0]
    assert bp3 == galsim.Bandpass('LSST_g.dat', 'nm')

    bp4 = galsim.config.BuildBandpass(config, 'current1', config)[0]
    assert bp4 == bp1

    bp5 = galsim.config.BuildBandpass(config, 'current2', config)[0]
    assert bp5 == bp3

    bp6 = galsim.config.BuildBandpass(config, 'eval1', config)[0]
    assert bp6 == bp2 * 0.5

    bp7 = galsim.config.BuildBandpass(config, 'eval2', config)[0]
    bp7b = bp1 * bp3
    # These have lambdas, so == doesn't work.  Check at a range of wavelengths.
    for wave in range(int(bp1.blue_limit)+1,int(bp1.red_limit)):
        assert bp7(wave) == bp7b(wave)

    # If we build something again with the same index, it should get the current value
    bp8 = galsim.config.BuildBandpass(config, 'bp2', config)[0]
    assert bp8 is bp2

    # But not if the index has changed.
    config['obj_num'] = 1
    bp9 = galsim.config.BuildBandpass(config, 'bp2', config)[0]
    assert bp9 is not bp2
    assert bp9 == bp2b.thin(1.e-5)

    bpz = galsim.config.BuildBandpass(config, 'bpz', config)[0]
    assert bpz == bp1.withZeropoint('Vega')

    for bad in ['bad1', 'bad2']:
        with assert_raises(galsim.GalSimConfigError):
            galsim.config.BuildBandpass(config, bad, config)

    # Base class usage is invalid
    builder = galsim.config.bandpass.BandpassBuilder()
    assert_raises(NotImplementedError, builder.buildBandpass, config, config, logger=None)


@timer
def test_index_key(run_slow):
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
    if run_slow:
        logger = logging.getLogger('test_index_key')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.DEBUG)
    else:
        logger = None

    # Normal sequential
    config1 = galsim.config.CopyConfig(config)
    # Note: Using BuildFiles(config1) would normally work, but it has an extra copy internally,
    # which messes up some of the current checks later.
    for n in range(nfiles):
        galsim.config.BuildFile(config1, file_num=n, image_num=n*nimages, obj_num=n*n_per_file,
                                logger=logger)
    images1 = [ galsim.fits.readMulti('output/index_key%02d.fits'%n) for n in range(nfiles) ]

    if run_slow:
        # For pytest tests, skip these 3 to save some time.
        # images5 is really the hardest test, and images1 is the easiest, so those two will
        # give good diagnostics for any errors.

        # Multiprocessing files
        config2 = galsim.config.CopyConfig(config)
        config2['output']['nproc'] = nfiles
        for n in range(nfiles):
            galsim.config.BuildFile(config2, file_num=n, image_num=n*nimages, obj_num=n*n_per_file,
                                    logger=logger)
        images2 = [ galsim.fits.readMulti('output/index_key%02d.fits'%n) for n in range(nfiles) ]

        # Multiprocessing images
        config3 = galsim.config.CopyConfig(config)
        config3['image']['nproc'] = nfiles
        for n in range(nfiles):
            galsim.config.BuildFile(config3, file_num=n, image_num=n*nimages, obj_num=n*n_per_file,
                                    logger=logger)
        images3 = [ galsim.fits.readMulti('output/index_key%02d.fits'%n) for n in range(nfiles) ]

        # New config for each file
        config4 = [ galsim.config.CopyConfig(config) for n in range(nfiles) ]
        for n in range(nfiles):
            galsim.config.SetupConfigFileNum(config4[n], n, n*nimages, n*n_per_file)
            galsim.config.SetupConfigRNG(config4[n])
        images4 = [ galsim.config.BuildImages(nimages, config4[n],
                                              image_num=n*nimages, obj_num=n*n_per_file,
                                              logger=logger)
                    for n in range(nfiles) ]

    # New config for each image
    config5 = [ galsim.config.CopyConfig(config) for n in range(nfiles) ]
    for n in range(nfiles):
        galsim.config.SetupConfigFileNum(config5[n], n, n*nimages, n*n_per_file)
        galsim.config.SetupConfigRNG(config5[n])

    images5 = [ [ galsim.config.BuildImage(galsim.config.CopyConfig(config5[n]),
                                           image_num=n*nimages+i,
                                           obj_num=n*n_per_file + i*n_per_image,
                                           logger=logger)
                  for i in range(nimages) ]
                for n in range(nfiles) ]

    # Now generate by hand
    first_seed = galsim.BaseDeviate(12345).raw()
    for n in range(nfiles):
        seed = first_seed + n*n_per_file
        file_rng = galsim.UniformDeviate(seed)
        fwhm = file_rng() * 0.2 + 0.9
        e = 0.2 + 0.05 * n
        beta = file_rng() * 2 * np.pi * galsim.radians
        kolm = galsim.Kolmogorov(fwhm=fwhm)
        psf_shear = galsim.Shear(e=e, beta=beta)
        kolm = kolm.shear(psf_shear)
        airy = galsim.Airy(lam=700, diam=4)
        psf = galsim.Convolve(kolm, airy)
        assert np.isclose(psf.flux, 1.0, rtol=1.e-15)
        print('fwhm, shear = ',fwhm,psf_shear._g)
        ellip_e1 = file_rng() * 0.4 - 0.2

        for i in range(nimages):
            if i == 0:
                image_rng = file_rng
            else:
                seed = first_seed + n*n_per_file + i*n_per_image
                image_rng = galsim.UniformDeviate(seed)
            im = galsim.ImageF(32*3, 32*3, scale=0.3)

            for k in range(nx*ny):
                seed = first_seed + n*n_per_file + i*n_per_image + k + 1
                obj_rng = galsim.UniformDeviate(seed)
                kx = k % 3
                ky = k // 3
                b = galsim.BoundsI(32*kx+1, 32*kx+32, 32*ky+1, 32*ky+32)
                stamp = im[b]
                flux = 100 + k*100
                hlr = 0.5 + i*0.5
                ellip_e2 = image_rng() * 0.4 - 0.2
                if k == 0:
                    shear_g2 = image_rng() * 0.04 - 0.02

                gal = galsim.Exponential(half_light_radius=hlr, flux=flux)

                while True:
                    shear_g1 = obj_rng() * 0.04 - 0.02
                    bd = galsim.BinomialDeviate(image_rng, N=1, p=0.2)
                    if bd() == 0:
                        break;
                    else:
                        ellip_e2 = image_rng() * 0.4 - 0.2

                ellip = galsim.Shear(e1=ellip_e1, e2=ellip_e2)
                shear = galsim.Shear(g1=shear_g1, g2=shear_g2)
                gal = gal.shear(ellip).shear(shear)
                print(n,i,k,flux,hlr,ellip._g,shear._g)
                final = galsim.Convolve(gal, psf)
                final.drawImage(stamp)

            if run_slow:
                im.write('output/test_index_key%02d_%02d.fits'%(n,i))
                images5[n][i].write('output/test_index_key%02d_%02d_5.fits'%(n,i))
            np.testing.assert_array_equal(im.array, images1[n][i].array,
                                          "index_key parsing failed for sequential BuildFiles run")
            if run_slow:
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
    assert 'current' in config1['gal']['ellip']['e1']
    assert 'current' in config1['gal']['ellip']['e2']
    assert 'current' in config1['gal']['shear']

    galsim.config.RemoveCurrent(config1, keep_safe=True, index_key='obj_num')
    assert 'current' in config1['psf']
    assert 'current' in config1['psf']['items'][1]
    assert 'current' not in config1['gal']
    assert 'current' not in config1['gal']['ellip']
    assert 'current' in config1['gal']['ellip']['e1']
    assert 'current' not in config1['gal']['ellip']['e2']
    assert 'current' not in config1['gal']['shear']

    galsim.config.RemoveCurrent(config1, keep_safe=True)
    assert 'current' not in config1['psf']
    assert 'current' in config1['psf']['items'][1]
    assert 'current' not in config1['gal']
    assert 'current' not in config1['gal']['ellip']
    assert 'current' not in config1['gal']['ellip']['e1']
    assert 'current' not in config1['gal']['ellip']['e2']
    assert 'current' not in config1['gal']['shear']

    galsim.config.RemoveCurrent(config1)
    assert 'current' not in config1['psf']
    assert 'current' not in config1['psf']['items'][1]
    assert 'current' not in config1['gal']
    assert 'current' not in config1['gal']['ellip']
    assert 'current' not in config1['gal']['ellip']['e1']
    assert 'current' not in config1['gal']['ellip']['e2']
    assert 'current' not in config1['gal']['shear']

    # Finally check for invalid index_key
    config2 = galsim.config.CopyConfig(config)
    config2['psf']['index_key'] = 'psf_num'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config2)
    config3 = galsim.config.CopyConfig(config)
    config3['gal']['shear']['g1']['rng_index_key'] = 'gal_num'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config3)


@timer
def test_multirng(run_slow):
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
    if run_slow:
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

    first_seed = galsim.BaseDeviate(12345).raw()
    ps_first_seed = galsim.BaseDeviate(12345 + 31415).raw()
    for n in range(nimages):
        seed = first_seed + n*ngals
        rng = galsim.UniformDeviate(seed)
        centeru = rng() * 10. - 5.
        centerv = rng() * 10. - 5.
        wcs = galsim.OffsetWCS(scale=0.1, world_origin=galsim.PositionD(centeru,centerv),
                               origin=galsim.PositionD(128.5,128.5))
        im = galsim.ImageF(256, 256, wcs=wcs)
        world_center = im.wcs.toWorld(im.true_center)
        psf_ps.buildGrid(grid_spacing=1.0, ngrid=30, rng=rng, center=world_center, variance=0.1)
        ps_rng = galsim.UniformDeviate(ps_first_seed + (n//3))
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
        if run_slow:
            im.write('output/test_multirng%02d.fits'%n)
        np.testing.assert_array_equal(im.array, images1[n].array)
        np.testing.assert_array_equal(im.array, images2[n].array)
        np.testing.assert_array_equal(im.array, images3[n].array)

    # Test invalid rng_num
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

    # Check that a warning is given if the user uses a Sequence for the first random seed.
    config8 = galsim.config.CopyConfig(config)
    config8['image']['random_seed'] = {
        'type': 'Sequence',
        'repeat': 3
    }
    with assert_warns(galsim.GalSimWarning):
        galsim.config.BuildImage(config8)


@timer
def test_sequential_seeds(run_slow):
    """Test using sequential seeds for successive images.

    Our old (<=2.3) way of setting rng seeds involved using the nominal seed value for
    the image rng, used for noise and other things that apply to the full image, and then
    a series of sequential seed values following the nominal seed for each galaxy in the image.
    This way the galaxies could be done asynchronously over multiple processes and still be
    deterministic.

    The problem with this is that a reasonable user choice is to set a new random seed for
    each of several successive images, expecting all different galaxies to be built on each.
    However, if the nominal seeds are too close together, you might get exactly the same
    random values for galaxies on successive images.  The worst case is when the image seeds
    are only incremented by 1 from one image to the next (not an unreasonable choice for a
    user to make -- indeed one that tripped up multiple users).

    Now we pick the galaxy seeds from a list generated from the image rng, so using sequential
    seeds for multiple images is completely fine.  This test confirms that.  (It fails
    for the old way of doing the seed sequence.)
    """
    if run_slow:
        nimages = 6
        ngals = 20
        logger = logging.getLogger('test_sequential_seeds')
        logger.addHandler(logging.StreamHandler(sys.stdout))
    else:
        nimages = 3
        ngals = 3
        logger = None

    config = galsim.config.ReadConfig('config_input/sequential_seeds.yaml')[0]
    config['image']['nobjects'] = ngals

    all_stamps = []
    for n in range(nimages):
        # Check this at the stamp level.  Prior to the fix all but one of the stamps was
        # the same as for the previous image.
        config1 = galsim.config.CopyConfig(config)
        config1['eval_variables']['iid'] = n
        stamps, current_vars = galsim.config.BuildStamps(ngals, config1, logger=logger)
        assert len(stamps) == ngals
        all_stamps.append(stamps)

    for n in range(1,nimages):
        for i,stampi in enumerate(all_stamps[n]):
            for j,stampj in enumerate(all_stamps[n-1]):
                print(i,j,stampi==stampj)
            assert stampi not in all_stamps[n-1]


@timer
def test_template():
    """Test various uses of the template keyword
    """
    # Use the multirng.yaml config file from the above test as a convenient template source
    config1 = {
        # This copies everything, but we'll override a few things
        "template" : "config_input/multirng.yaml",

        # Modules works differently from the others.  Here, we want to concatenate the lists.
        "modules" : ["astropy.time"],

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
        # Gratuitous use of astropy.time to test submodule inclusions.
        "gal.scale_radius" : "$astropy.time.Time(1.6, format='jd').to_value('jd')",

        # Check that template items work inside a list.
        "psf" : {
            "type" : "List",
            "items" : [
                { "template" : "config_input/multirng.yaml:psf" },
                { "type" : "Gaussian", "sigma" : 0.3 },
                # Omitting the file name before : means use the current config file instead.
                { "template" : ":psf.items.1", "sigma" : 0.4 },
            ]
        },

        # Check setting a list element
        "input.power_spectrum.0": {
            "e_power_function": "(k**2 + (1./180)**2)**(-11./6.)",
            "units": "arcsec",
            "grid_spacing": 1,
            "ngrid": 50,
            "variance": 0.2,
        },
    }
    config = config1.copy()  # Leave config1 as the original given above.
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
    assert config['gal']['scale_radius'] == "$astropy.time.Time(1.6, format='jd').to_value('jd')"

    # Make sure that parses correctly.
    sr = galsim.config.ParseValue(config['gal'].copy(), 'scale_radius', config.copy(), float)[0]
    assert sr == 1.6

    assert config['psf']['type'] == 'List'
    assert config['psf']['items'][0] == { "type": "Moffat", "beta": 2, "fwhm": 0.9,
                                          "ellip": { "type" : "PowerSpectrumShear", "num" : 0 } }
    assert config['psf']['items'][1] == { "type": "Gaussian", "sigma" : 0.3 }
    assert config['psf']['items'][2] == { "type": "Gaussian", "sigma" : 0.4 }

    assert config['input']['power_spectrum'][1]['grid_spacing'] == 2
    assert config['input']['power_spectrum'][0]['grid_spacing'] == 1
    assert config['input']['power_spectrum'][0]['ngrid'] == 50
    assert config['input']['power_spectrum'][0]['variance'] == 0.2

    assert config['modules'] == ['numpy', 'astropy.time']

    # Test registering the template.
    galsim.config.RegisterTemplate('multirng', 'config_input/multirng.yaml')
    config3 = config1.copy()
    config3['template'] = 'multirng'

    galsim.config.ProcessAllTemplates(config3)
    assert config3 == config

    # Make sure template works when registered in a user module
    del galsim.config.process.valid_templates['multirng']
    config4 = config1.copy()
    config4['template'] = 'multirng'
    config4['modules'] = ['template_register', 'astropy.time']
    galsim.config.ImportModules(config4)
    galsim.config.ProcessAllTemplates(config4)
    for field in ['image', 'output', 'gal', 'psf']:
        assert config4[field] == config[field]

    # Check a simple Eval string
    config5 = config1.copy()
    config5['template'] = '$"gnritlum"[::-1]'
    galsim.config.ProcessAllTemplates(config5)
    for key in config:
        assert config5[key] == config[key]

    # Check deleting a list element
    config6 = config1.copy()
    config6['image.random_seed.1'] = ""
    galsim.config.ProcessAllTemplates(config6)
    assert config6['image']['random_seed'] == [config2['image']['random_seed'][0]]

    # Further adjustments must use the new index.
    config7 = config1.copy()
    config7['image.random_seed.0'] = ""
    config7['image.random_seed.0.str'] = '123 + (image_num//3) * @image.nobjects'
    with CaptureLog() as cl:
        galsim.config.ProcessAllTemplates(config7, cl.logger)
    assert config7['image']['random_seed'][0]['type'] == 'Eval'
    assert config7['image']['random_seed'][0]['str'] == '123 + (image_num//3) * @image.nobjects'
    print(cl.output)
    assert "Removing item 0 from image.random_seed." in cl.output

    # Read a template config from a file
    config8 = galsim.config.ReadConfig('config_input/template.yaml')[0]
    galsim.config.ProcessAllTemplates(config8)
    for key in config3:
        print(key)
        print(config4[key])
        print(config8[key])
        assert config8[key] == config4[key]
    assert config8 == config4

    # Make sure nested templating works
    config9 = {
        "template" : "config_input/template.yaml",
        "gal.ellip.num" : 1,
        "psf.items.0.beta" : 2.5,
        "psf.items.2" : { "template" : "multirng:image.noise" },
    }
    galsim.config.ProcessAllTemplates(config9)
    config8['gal']['ellip']['num'] = 1
    config8['psf']['items'][0]['beta'] = 2.5
    config8['psf']['items'][2] = config4['image']['noise']
    assert config9 == config8

    # Make sure evals work correltly for template string when not at top level.
    # (This used to give an error about the config dict changing size during iteration.)
    config10 = {
        "eval_variables" : {
            "iabc": 123,
            "ddict": {
                "template": "$os.path.join('config_input','dict.yaml')"
            }
        }
    }
    galsim.config.ProcessAllTemplates(config10)
    assert config10['eval_variables']['ddict']['b'] == False
    assert config10['eval_variables']['ddict']['s'] == "Brian"
    assert config10['eval_variables']['ddict']['noise']['models'][0]['variance'] == 0.12


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
                'file_name': [ 'cat_3.txt', 'cat_5.txt' ],
                'index_key': 'image_num',
            }
        },
        'output': {
            'type' : 'MultiFits',
            'file_name' : 'output/test_variable_input.fits',
            'nimages' : 2,
        }
    }

    config1 = galsim.config.CopyConfig(config)

    # This input isn't safe, so it can't load when doing the safe_only load.
    with CaptureLog() as cl:
        galsim.config.ProcessInput(config, safe_only=True, logger=cl.logger)
    assert "Skip catalog 0, since not safe" in cl.output

    logger = logging.getLogger('test_variable_input')
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

    assert cfg_images[0] == ref_images[0]
    assert cfg_images[1] == ref_images[1]

    # Now run with full Process function
    galsim.config.Process(config1, logger=logger)
    cfg_images2 = galsim.fits.readMulti('output/test_variable_input.fits')
    assert cfg_images2[0] == ref_images[0]
    assert cfg_images2[1] == ref_images[1]


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
    Especially that its internal "prof" is not just a single GSObject.
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
    print('0: ',np.unravel_index(np.argmax(images[0].array),images[0].array.shape))
    print('1: ',np.unravel_index(np.argmax(images[1].array),images[0].array.shape))
    print('2: ',np.unravel_index(np.argmax(images[2].array),images[0].array.shape))
    print('3: ',np.unravel_index(np.argmax(images[3].array),images[0].array.shape))
    print('4: ',np.unravel_index(np.argmax(images[4].array),images[0].array.shape))
    print('5: ',np.unravel_index(np.argmax(images[5].array),images[0].array.shape))
    print('6: ',np.unravel_index(np.argmax(images[6].array),images[0].array.shape))
    print('7: ',np.unravel_index(np.argmax(images[7].array),images[0].array.shape))
    # With these random numbers relative to image 0, the offsets are:
    # 1: +9, +4
    # 2: -9, +4
    # 3: +5, +9
    np.testing.assert_array_equal(images[1].array[19:63,14:58], images[0].array[10:54,10:54])
    np.testing.assert_array_equal(images[2].array[1:45,14:58], images[0].array[10:54,10:54])
    np.testing.assert_array_equal(images[3].array[15:59,19:63], images[0].array[10:54,10:54])

    # For the next set, the offset are (relative to image 4):
    # 5: -5, +1
    # 6: -1, +7
    # 7: -5, -1
    np.testing.assert_array_equal(images[5].array[5:49,11:55], images[4].array[10:54,10:54])
    np.testing.assert_array_equal(images[6].array[9:53,17:61], images[4].array[10:54,10:54])
    np.testing.assert_array_equal(images[7].array[5:49,9:53], images[4].array[10:54,10:54])

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


@timer
def test_chromatic(run_slow):
    """Test drawing a chromatic object on an image with a bandpass
    """
    if run_slow:
        bp_file = 'LSST_r.dat'
    else:
        # In pytest, use a simple bandpass to go faster.
        bp_file = 'chromatic_reference_images/simple_bandpass.dat'

    # First check a chromatic galaxy with a regular PSF.
    config = {
        'image': {
            'type': 'Single',
            'size': 64,
            'pixel_scale': 0.2,

            'bandpass': {
                'file_name': bp_file,
                'wave_type': 'nm',
                'thin': 1.e-4,
            }
        },
        'gal': {
            'type': 'Exponential',
            'half_light_radius': 0.5,

            'sed': {
                'file_name': 'CWW_E_ext.sed',
                'wave_type': u.Angstrom,
                'flux_type': u.erg/u.Angstrom/u.cm**2/u.s,
                'norm_flux_density': 1.0,
                'norm_wavelength': 500,
                'redshift': 0.8,
            },
        },
        'psf' : {
            'type': 'Moffat',
            'fwhm': 0.5,
            'beta': 2.5,
        },
    }
    image = galsim.config.BuildImage(config)

    bandpass = galsim.Bandpass(bp_file, 'nm').thin(1.e-4)
    sed = galsim.SED('CWW_E_ext.sed', 'Ang', 'flambda').withFluxDensity(1.0, 500).atRedshift(0.8)

    gal = galsim.Exponential(half_light_radius=0.5) * sed
    psf1 = galsim.Moffat(fwhm=0.5, beta=2.5)
    final = galsim.Convolve(gal, psf1)
    image1 = final.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    print('image.sum = ',image.array.sum())
    print('image1.sum = ',image1.array.sum())
    np.testing.assert_allclose(image.array, image1.array)

    # Can also set a flux rather than normalize the sed.
    config['gal']['sed'] = {
        'file_name': 'CWW_E_ext.sed',
        'wave_type': 'Ang',
        'flux_type': 'flambda',
    }
    config['gal']['flux'] = 500
    del config['gal']['_get']
    galsim.config.RemoveCurrent(config)
    image = galsim.config.BuildImage(config)
    sed = galsim.SED('CWW_E_ext.sed', 'Ang', 'flambda')
    gal = (galsim.Exponential(half_light_radius=0.5, flux=500) * sed).withFlux(500, bandpass)
    final = galsim.Convolve(gal, psf1)
    image1 = final.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    print('image.sum = ',image.array.sum())
    print('image1.sum = ',image1.array.sum())
    np.testing.assert_allclose(image.array, image1.array)

    # Now check ChromaticAtmosphere
    galsim.config.RemoveCurrent(config)
    config['psf'] =  {
        'type': 'ChromaticAtmosphere',
        'base_profile': {
            'type': 'Moffat',
            'fwhm': 0.5,
            'beta': 2.5,
        },
        'base_wavelength': 500,
        'zenith_angle' : '13.1 deg',
        'parallactic_angle' : '98 deg',
    }
    image = galsim.config.BuildImage(config)

    psf2 = galsim.ChromaticAtmosphere(psf1, base_wavelength=500,
                                      zenith_angle = 13.1 * galsim.degrees,
                                      parallactic_angle = 98 * galsim.degrees)
    final = galsim.Convolve(gal, psf2)
    image2 = final.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    np.testing.assert_allclose(image.array, image2.array)

    galsim.config.RemoveCurrent(config)
    del config['psf']['base_profile']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    config['psf'] =  {
        'type': 'ChromaticAtmosphere',
        'base_profile': {
            'type': 'Moffat',
            'fwhm': 0.5,
            'beta': 2.5,
        },
        'base_wavelength': 500,
        'latitude': '19.8207 deg',
        'HA': '-1.0 hour',
    }
    config['stamp'] = {
        'sky_pos' : {
            'type' : 'RADec',
            'ra' : '35 deg',
            'dec' : '12 deg',
        },
    }
    image = galsim.config.BuildImage(config)

    sky_pos = galsim.CelestialCoord(35 * galsim.degrees, 12 * galsim.degrees)
    psf3 = galsim.ChromaticAtmosphere(psf1, base_wavelength=500,
                                      obj_coord = sky_pos,
                                      latitude = 19.8207 * galsim.degrees,
                                      HA = -1.0 * galsim.hours)
    final = galsim.Convolve(gal, psf3)
    image3 = final.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    np.testing.assert_allclose(image.array, image3.array)

    galsim.config.RemoveCurrent(config)
    del config['stamp']
    del config['sky_pos']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # ChromaticAiry
    config['psf'] =  {
        'type': 'ChromaticAiry',
        'lam' : 550,
        'diam' : 6.5,
    }
    galsim.config.RemoveCurrent(config)
    image = galsim.config.BuildImage(config)
    psf4 = galsim.ChromaticAiry(lam=550, diam=6.5)
    final = galsim.Convolve(gal, psf4)
    image4 = final.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    np.testing.assert_allclose(image.array, image4.array)

    # ChromaticOpticalPSF
    config['psf'] =  {
        'type' : 'ChromaticOpticalPSF',
        'lam' : 700,
        'lam_over_diam' : 0.6,
        'defocus' : 0.23,
        'astig1' : -0.12,
        'astig2' : 0.11,
        'coma1' : -0.09,
        'coma2' : 0.03,
        'spher' : 0.19,
    }
    galsim.config.RemoveCurrent(config)
    image = galsim.config.BuildImage(config)
    psf5 = galsim.ChromaticOpticalPSF(lam=700, lam_over_diam=0.6, defocus=0.23,
                                      astig1=-0.12, astig2=0.11,
                                      coma1=-0.09, coma2=0.03, spher=0.19)
    final = galsim.Convolve(gal, psf5)
    image5 = final.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    np.testing.assert_allclose(image.array, image5.array)

    # ChromaticOpticalPSF with aberrations
    config['psf'] =  {
        'type' : 'ChromaticOpticalPSF',
        'lam' : 700,
        'diam' : 8.4,
        'aberrations' : [0.06, 0.12, -0.08, 0.07, 0.04, 0.0, 0.0, -0.13],
    }
    galsim.config.RemoveCurrent(config)
    image = galsim.config.BuildImage(config)
    psf6 = galsim.ChromaticOpticalPSF(lam=700, diam=8.4,
                                      aberrations=[0,0,0,0,
                                                   0.06, 0.12, -0.08, 0.07, 0.04, 0.0, 0.0, -0.13])
    final = galsim.Convolve(gal, psf6)
    image6 = final.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    np.testing.assert_allclose(image.array, image6.array)

    # ChromaticRealGalaxy
    # This is pretty slow, so switch back to an achromatic PSF to help keep things reasonable.
    config['psf'] = { 'type': 'Moffat', 'fwhm': 0.5, 'beta': 2.5, }
    config['gal'] = {
        'type' : 'ChromaticRealGalaxy',
        # Default is to go in order, so start with index=0
    }
    config['input'] = {
        'real_catalog' : [
            { 'dir' : 'real_comparison_images',
              'file_name' : 'AEGIS_F606w_catalog.fits',
            },
            { 'dir' : 'real_comparison_images',
              'file_name' : 'AEGIS_F814w_catalog.fits',
            },
        ]
    }
    galsim.config.RemoveCurrent(config)
    image = galsim.config.BuildImage(config)

    image_dir = 'real_comparison_images'
    f606w_cat = galsim.RealGalaxyCatalog('AEGIS_F606w_catalog.fits', dir=image_dir)
    print('ident = ',f606w_cat.ident)
    f814w_cat = galsim.RealGalaxyCatalog('AEGIS_F814w_catalog.fits', dir=image_dir)
    gal7 = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], index=0)
    psf = galsim.Moffat(fwhm=0.5, beta=2.5)
    final = galsim.Convolve(gal7, psf)
    image7 = final.drawImage(nx=64, ny=64, scale=0.2, bandpass=bandpass)
    np.testing.assert_allclose(image.array, image7.array)

    galsim.config.RemoveCurrent(config)
    config['gal']['index'] = 5
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['gal']['index'] = -1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # Some equivalent ways to get this same galaxy
    config['gal'] = {
        'type' : 'ChromaticRealGalaxy',
        'index' : 0,
        'gsparams' : { 'folding_threshold' : 5.e-3 }
    }
    galsim.config.RemoveCurrent(config)
    image = galsim.config.BuildImage(config)
    np.testing.assert_allclose(image.array, image7.array)

    config['gal'] = {
        'type' : 'ChromaticRealGalaxy',
        'id' : '23409',
    }
    galsim.config.RemoveCurrent(config)
    image = galsim.config.BuildImage(config)
    np.testing.assert_allclose(image.array, image7.array)

    config['gal'] = {
        'type' : 'ChromaticRealGalaxy',
        'random' : False
    }
    galsim.config.RemoveCurrent(config)
    image = galsim.config.BuildImage(config)
    np.testing.assert_allclose(image.array, image7.array)

    config['gal'] = {
        'type' : 'ChromaticRealGalaxy',
        'random' : True
    }
    config['image']['random_seed'] = 12341 # This seed happens to get index=0 first.
    galsim.config.RemoveCurrent(config)
    image = galsim.config.BuildImage(config)
    np.testing.assert_allclose(image.array, image7.array)

    # Finally check that without bandpass, we get an error.
    del config['image']['bandpass']
    del config['bandpass']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)


@timer
def test_photon_ops():
    # Test photon ops in config
    pupil_plane_im = os.path.join("Optics_comparison_images", "sample_pupil_rolled.fits")
    config = {
        'stamp' : {
            'photon_ops' : [
                {
                    'type' : 'FRatioAngles',
                    'fratio' : 1.234,
                    'obscuration' : 0.606,
                },
                {
                    'type' : 'WavelengthSampler',
                    'sed': {
                        'file_name': 'CWW_E_ext.sed',
                        'wave_type': 'Ang',
                        'flux_type': 'flambda',
                        'norm_flux_density': 1.0,
                        'norm_wavelength': 500,
                        'redshift' : '@gal.redshift',
                    },
                },
                {
                    'type' : 'PhotonDCR',
                    'base_wavelength' : '$bandpass.effective_wavelength',
                    'latitude' : '-30.24463 degrees',
                    # Give each object a different HA. Could do this better by basing it
                    # on image_pos and the wcs, etc.  But here I just want to check that
                    # the photon_ops are different for a different obj_num, so do something
                    # easy if not particularly realistic.
                    'HA' : '$-1.48 * galsim.hours + 150 * galsim.arcsec * obj_num',
                },
                {
                    'type' : 'FocusDepth',
                    'depth' : -0.6,  # pixels.  Negative means intrafocal (focus is in silicon)
                },
                {
                    'type' : 'Refraction',
                    'index_ratio' : 3.9,
                },
                {
                    'type' : 'PupilImageSampler',
                    'diam' : 2.4,
                    'lam' : 900,
                    'pupil_plane_im' : pupil_plane_im,
                },
                {
                    'type' : 'PupilAnnulusSampler',
                    'R_outer' : 1.0,
                    'R_inner' : 0.3,
                },
                {
                    'type' : 'TimeSampler',
                    't0' : 10,
                    'exptime' : 30,
                },
            ],
            'sky_pos' : {
                'type' : 'RADec',
                'ra' : '13 hr',
                'dec' : '-17 deg'
            }
        },
        'image' : {
            'type' : 'Single',
            'pixel_scale' : 0.2,
            'bandpass' : {
                'file_name': 'LSST_g.dat',
                'wave_type': 'nm',
            },
            'random_seed' : 1234,
            'draw_method' : 'phot',
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 2.3,
            'flux' : 10000,
            'redshift' : 0.8,
        },
        'psf' : {
            'type' : 'Moffat',
            'fwhm' : 0.7,
            'beta' : 3.5,
        },
    }

    first_seed = galsim.BaseDeviate(1234).raw()
    rng = galsim.BaseDeviate(first_seed+1)
    gal = galsim.Gaussian(sigma=2.3, flux=10000)
    psf = galsim.Moffat(beta=3.5, fwhm=0.7)
    obj = galsim.Convolve(gal, psf)
    bp = galsim.Bandpass('LSST_g.dat', wave_type='nm')
    sed = galsim.SED('CWW_E_ext.sed', wave_type='Ang', flux_type='flambda')
    sed = sed.withFluxDensity(1.0, 500).atRedshift(0.8)
    sky_pos = galsim.CelestialCoord(ra=13*galsim.hours, dec=-17*galsim.degrees)

    frat = galsim.FRatioAngles(fratio=1.234, obscuration=0.606)
    wave = galsim.WavelengthSampler(sed=sed, bandpass=bp)
    dcr = galsim.PhotonDCR(base_wavelength=bp.effective_wavelength,
                           latitude=-30.24463 * galsim.degrees,
                           obj_coord=sky_pos, HA=-1.48 * galsim.hours)
    depth = galsim.FocusDepth(-0.6)
    ref = galsim.Refraction(3.9)
    pupil_image = galsim.PupilImageSampler(diam=2.4, lam=900.,
                                           pupil_plane_im=pupil_plane_im)
    pupil_annulus = galsim.PupilAnnulusSampler(R_outer=1.0, R_inner=0.3)
    time = galsim.TimeSampler(t0=10., exptime=30.)
    photon_ops = [frat, wave, dcr, depth, ref, pupil_image, pupil_annulus, time]

    im1 = obj.drawImage(scale=0.2, method='phot', rng=rng, photon_ops=photon_ops)
    im2 = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im2.array, im1.array)

    # If we do it again, it uses the current values
    ops2 = galsim.config.BuildPhotonOps(config['stamp'], 'photon_ops', config)
    assert ops2 == photon_ops

    # But not if we are on the next object
    galsim.config.SetupConfigObjNum(config, obj_num=1)
    ops3 = galsim.config.BuildPhotonOps(config['stamp'], 'photon_ops', config)
    assert ops3 != photon_ops

    # Check some alternate ways to get the same photon_ops
    config['photon_ops_orig'] = config['stamp']['photon_ops']
    config['stamp']['photon_ops'] = [
        {
            'type' : 'List',
            'items' : ['$galsim.FRatioAngles(fratio=1.234, obscuration=0.606)']
        },
        {
            'type' : 'Eval',
            'str' : 'galsim.WavelengthSampler(sed=@photon_ops_orig.1.sed, bandpass=bandpass)',
        },
        '@photon_ops_orig.2',
        {
            'type' : 'Current',
            'key' : 'photon_ops_orig.3',
        },
        galsim.Refraction(3.9)
    ]
    galsim.config.SetupConfigObjNum(config, obj_num=0)
    galsim.config.SetupConfigRNG(config, seed_offset=1)
    galsim.config.RemoveCurrent(config)
    ops4 = galsim.config.BuildPhotonOps(config, 'photon_ops_orig', config)
    ops5 = galsim.config.BuildPhotonOps(config['stamp'], 'photon_ops', config)
    assert ops4[:5] == ops5

    galsim.config.RemoveCurrent(config)
    galsim.config.BuildPhotonOps(config, 'photon_ops_orig', config)
    im5 = galsim.config.BuildStamp(config)[0]
    np.testing.assert_array_equal(im5.array, im2.array)

    # Test various errors
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildPhotonOps(config['stamp']['photon_ops'], 0, config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildPhotonOp(config['stamp'], 'photon_ops', config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildPhotonOp(config, 'gal', config)
    config['photon_op'] = { 'type' : 'Invalid' }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildPhotonOp(config, 'photon_op', config)
    with assert_raises(NotImplementedError):
        galsim.config.photon_ops.PhotonOpBuilder().buildPhotonOp(config,config,None)
    del config['photon_ops_orig'][1]['sed']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildPhotonOp(config['photon_ops_orig'], 1, config)
    config['photon_ops_orig'][1]['sed'] = sed
    del config['bandpass']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildPhotonOp(config['photon_ops_orig'], 1, config)
    config['stamp']['photon_ops'][0]['index'] = 1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildPhotonOp(config['stamp']['photon_ops'], 0, config)
    config['stamp']['photon_ops'][0]['index'] = -1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildPhotonOp(config['stamp']['photon_ops'], 0, config)
    config['stamp']['photon_ops'][0]['index'] = 0
    config['stamp']['photon_ops'][0]['items'] = galsim.Refraction(3.9)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildPhotonOp(config['stamp']['photon_ops'], 0, config)


@timer
def test_sensor():
    # Test sensor option in config
    config = {
        'stamp' : {
            'photon_ops' : [
                {
                    'type' : 'FRatioAngles',
                    'fratio' : 1.234,
                    'obscuration' : 0.606,
                },
                {
                    'type' : 'WavelengthSampler',
                    'sed': {
                        'file_name': 'CWW_E_ext.sed',
                        'wave_type': 'Ang',
                        'flux_type': 'flambda',
                        'norm_flux_density': 1.0*u.erg/u.s/u.cm**2/u.nm,
                        'norm_wavelength': 500*u.nm,
                        'redshift' : '@gal.redshift',
                    },
                },
            ],
            'sky_pos' : {
                'type' : 'RADec',
                'ra' : '13 hr',
                'dec' : '-17 deg'
            },
        },
        'image' : {
            'type' : 'Single',
            'pixel_scale' : 0.2,
            'bandpass' : {
                'file_name': 'LSST_g.dat',
                'wave_type': 'nm',
            },
            'random_seed' : 1234,
            'draw_method' : 'phot',
            'sensor' : {
                'type' : 'Silicon'
            }
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 2.3,
            'flux' : 10000,
            'redshift' : 0.8,
        },
        'psf' : {
            'type' : 'Moffat',
            'fwhm' : 0.7,
            'beta' : 3.5,
        },
    }

    first_seed = galsim.BaseDeviate(1234).raw()
    rng = galsim.BaseDeviate(first_seed+1)
    gal = galsim.Gaussian(sigma=2.3, flux=10000)
    psf = galsim.Moffat(beta=3.5, fwhm=0.7)
    obj = galsim.Convolve(gal, psf)
    bp = galsim.Bandpass('LSST_g.dat', wave_type='nm')
    sed = galsim.SED('CWW_E_ext.sed', wave_type='Ang', flux_type='flambda')
    sed = sed.withFluxDensity(1.0, 500).atRedshift(0.8)
    sky_pos = galsim.CelestialCoord(ra=13*galsim.hours, dec=-17*galsim.degrees)

    frat = galsim.FRatioAngles(fratio=1.234, obscuration=0.606)
    wave = galsim.WavelengthSampler(sed=sed, bandpass=bp)
    photon_ops = [frat, wave]
    sensor = galsim.SiliconSensor(rng=rng)

    im1 = obj.drawImage(scale=0.2, method='phot', rng=rng, photon_ops=photon_ops, sensor=sensor)
    im2 = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im2.array, im1.array)

    # If we do it again, it uses the current values
    sensor2 = galsim.config.BuildSensor(config['image'], 'sensor', config)
    assert sensor2 == sensor

    # Can access this as a Current value, but only if it's already been evaluated normally.
    config['sensor3'] = '@image.sensor'
    sensor3 = galsim.config.BuildSensor(config, 'sensor3', config)
    assert sensor3 == sensor
    config['sensor4'] = { 'type' : 'Current', 'key' : 'image.sensor' }
    sensor4 = galsim.config.BuildSensor(config, 'sensor4', config)
    assert sensor4 == sensor

    # Can be an Eval string
    config['sensor5'] = '$galsim.SiliconSensor(rng=rng)'
    sensor5 = galsim.config.BuildSensor(config, 'sensor5', config)
    assert sensor5 == sensor
    config['sensor6'] = { 'type' : 'Eval', 'str' : 'galsim.SiliconSensor(rng=rng)' }
    sensor6 = galsim.config.BuildSensor(config, 'sensor6', config)
    assert sensor6 == sensor

    # Can be a Sensor instance
    config['sensor7'] = sensor
    sensor7 = galsim.config.BuildSensor(config, 'sensor7', config)
    assert sensor7 == sensor

    # Can be a List
    config['sensor8'] = { 'type' : 'List', 'items' : [ sensor ] }
    sensor8 = galsim.config.BuildSensor(config, 'sensor8', config)
    assert sensor8 == sensor

    # Won't use current if we are on the next object
    galsim.config.SetupConfigObjNum(config, obj_num=1)
    galsim.config.SetupConfigRNG(config, seed_offset=1)
    sensor9 = galsim.config.BuildSensor(config['image'], 'sensor', config)
    assert sensor9 != sensor

    # Default sensor is equivalent to not using one.
    galsim.config.SetupConfigRNG(config, seed_offset=1)
    galsim.config.RemoveCurrent(config)
    rng.reset(first_seed+1)
    config['image']['sensor'] = {}
    im3 = obj.drawImage(scale=0.2, method='phot', rng=rng, photon_ops=photon_ops)
    im4 = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im4.array, im3.array)

    # Test one with all the optional bits to SiliconSensor
    config['image']['sensor'] = {
        'type' : 'Silicon',
        'name' : 'lsst_e2v_50_8',
        'strength' : 0.8,
        'diffusion_factor' : 0.2,
        'qdist' : 2,
        'nrecalc' : 3000,
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
    galsim.config.SetupConfigRNG(config, seed_offset=1)
    galsim.config.RemoveCurrent(config)
    rng.reset(first_seed+1)
    trfunc = galsim.LookupTable.from_file('tree_ring_lookup.dat', amplitude=0.5)
    sensor = galsim.SiliconSensor(name='lsst_e2v_50_8', rng=rng,
                                  strength=0.8, diffusion_factor=0.2, qdist=2, nrecalc=3000,
                                  treering_func=trfunc, treering_center=galsim.PositionD(0,-500))
    im5 = obj.drawImage(scale=0.2, method='phot', rng=rng, photon_ops=photon_ops, sensor=sensor)
    im6 = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im6.array, im5.array)

    # Test various errors
    galsim.config.RemoveCurrent(config)
    config['sensor'] = { 'type' : 'Invalid' }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildSensor(config, 'sensor', config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildSensor(config, 'gal', config)
    with assert_raises(NotImplementedError):
        galsim.config.sensor.SensorBuilder().buildSensor(config,config,None)
    config['sensor8']['index'] = 1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildSensor(config, 'sensor8', config)
    config['sensor8']['index'] = -1
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildSensor(config, 'sensor8', config)
    config['sensor8']['index'] = 0
    config['sensor8']['items'] = sensor
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildSensor(config, 'sensor8', config)
    config['sensor'] = [sensor]
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildSensor(config, 'sensor', config)


@timer
def test_initial_image():
    # This test simulates a time series of a supernova going off near a big galaxy.
    # It first makes a reference image of the galaxy alone.
    # Then it makes 12 observations of the same scene with a SN.
    # The SN appears between the 3rd and 4th images, and then decays exponentially.

    # Note: this can be run from the command line as just `galsim sn.yaml`, which does
    # all the parts in one execution.
    configs = galsim.config.ReadConfig('config_input/sn.yaml')

    # In Python, we have to run each config dict separately.
    # First make the reference image.
    galsim.config.Process(configs[0])
    ref_image = galsim.fits.read('output/ref.fits')

    # Now the supernovae.
    galsim.config.ProcessInput(configs[1])
    sn_images = galsim.config.BuildImages(12, configs[1])

    for i, sn_image in enumerate(sn_images):
        print(i, sn_image.array.max(), sn_image.array.sum())
        t = (i - 2.3) * 7  # Time in day
        if t > 0:
            flux = np.exp(-t/50) * 1.e5
        else:
            flux = 0.

        # The diff image is basically perfect here, since we didn't add noise.
        # The only imprecision is that Moffats have flux off the edge of the stamp.
        # It's pretty noticable with beta=2.  But here with beta=3, it's pretty slight.
        diff_image = sn_image - ref_image
        print(i, t, flux, diff_image.array.sum())
        np.testing.assert_allclose(diff_image.array.sum(), flux, rtol=1.e-6)


if __name__ == "__main__":
    runtests(__file__)
