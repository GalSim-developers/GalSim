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
import shutil
import sys
import logging
import math
import yaml
import json
import re
import glob
import platform
from collections import OrderedDict
from unittest import mock

import galsim
from galsim_test_helpers import *


@timer
def test_fits():
    """Test the default output type = Fits
    """
    # Most of the tests in this file write to the 'output' directory.  Here we write to a different
    # directory and make sure that it properly creates the directory if necessary.
    if os.path.exists('output_fits'):
        shutil.rmtree('output_fits')
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'output' : {
            'type' : 'Fits',
            'nfiles' : 6,
            'file_name' : "$'output_fits/test_fits_%d.fits'%file_num",
        },
    }

    logger = logging.getLogger('test_fits')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)
    config1 = galsim.config.CopyConfig(config)

    im1_list = []
    nfiles = 6
    first_seed = galsim.BaseDeviate(1234).raw()
    for k in range(nfiles):
        ud = galsim.UniformDeviate(first_seed + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        im1 = gal.drawImage(scale=1)
        im1_list.append(im1)

        galsim.config.BuildFile(config, file_num=k, image_num=k, obj_num=k, logger=logger)
        file_name = 'output_fits/test_fits_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1.array)

    # Build all files at once
    config = galsim.config.CopyConfig(config1)
    galsim.config.BuildFiles(nfiles, config)
    for k in range(nfiles):
        file_name = 'output_fits/test_fits_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1_list[k].array)

    # Can also use Process to do this
    config = galsim.config.Process(config1)
    for k in range(nfiles):
        file_name = 'output_fits/test_fits_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1_list[k].array)

    # The returned config is modified relative to the original.
    assert config['image']['type'] == 'Single' # It has the items from the input.
    assert config1['image']['type'] == 'Single'
    assert config['image']['random_seed']['type'] == 'Sequence'  # Some things are modified.
    assert config1['image']['random_seed'] == 1234
    assert isinstance(config['rng'], galsim.BaseDeviate)  # And some new things
    assert 'rng' not in config1

    # For the first file, you don't need the file_num.
    os.remove('output_fits/test_fits_0.fits')
    config = galsim.config.CopyConfig(config1)
    galsim.config.BuildFile(config)
    im2 = galsim.fits.read('output_fits/test_fits_0.fits')
    np.testing.assert_array_equal(im2.array, im1_list[0].array)

    # nproc < 0 should automatically determine nproc from ncpu
    config = galsim.config.CopyConfig(config1)
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger, new_params={'output.nproc' : -1})
    assert 'ncpu = ' in cl.output

    # nproc > njobs should drop back to nproc = njobs
    config = galsim.config.CopyConfig(config1)
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger, new_params={'output.nproc' : 10})
    if galsim.config.UpdateNProc(10, 6, config) > 1:
        assert 'There are only 6 jobs to do.  Reducing nproc to 6' in cl.output

    # There is a feature that we reduce the number of tasks to be < 32767 to avoid problems
    # with the multiprocessing.Queue overflowing.  That 32767 number is a settable paramter,
    # mostly so we can test this without requiring a crazy huge simultation run.
    # So set it to 4 here to test it.
    galsim.config.util.max_queue_size = 4
    config = galsim.config.CopyConfig(config1)
    config['output']['nproc'] = 2
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger, new_params={'output.nproc' : 2})
    print(cl.output)
    if galsim.config.UpdateNProc(10, 6, config) > 1:
        assert 'len(tasks) = 6 is more than max_queue_size = 4' in cl.output
    for k in range(nfiles):
        file_name = 'output_fits/test_fits_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1_list[k].array)
    galsim.config.util.max_queue_size = 32767  # Set it back.

    # Check that profile outputs something appropriate for multiprocessing.
    # (The single-thread profiling is handled by the galsim executable, which we don't
    # bother testing here.)
    config = galsim.config.CopyConfig(config1)
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger,
                              new_params={'profile':True, 'output.nproc': 4})
    #print(cl.output)
    # Unfortunately, the LoggerProxy doesn't really work right with the string logger used
    # by CaptureLog.  I tried for a while to figure out how to get it to capture the proxied
    # logs and couldn't get it working.  So this just checks for an info log before the
    # multithreading starts.  But with a regular logger, there really is profiling output.
    if galsim.config.UpdateNProc(10, 6, config) > 1:
        assert "Starting separate profiling for each of the" in cl.output
    for p in range(4):
        pstats_file = f'galsim-Process-{p+1}.pstats'
        assert os.path.exists(pstats_file)
        os.remove(pstats_file)

    # Check some public API utility functions
    assert galsim.config.GetNFiles(config) == 6
    assert galsim.config.GetNImagesForFile(config, 0) == 1
    assert galsim.config.GetNObjForFile(config, 0, 0) == [1]

    # Check invalid output type
    config['output']['type'] = 'invalid'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.Process(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.GetNImagesForFile(config, 0)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.GetNObjForFile(config, 0, 0)

    # Invalid output file
    config = galsim.config.CopyConfig(config1)
    config['output']['file_name'] = "$'output_fits/test_fits_%d.fits/test_fits.fits'%file_num"
    with assert_raises(OSError):
        galsim.config.BuildFile(config)

    # If there is no output field, it raises an error when trying to do BuildFile.
    os.remove('output_fits/test_fits_0.fits')
    config = galsim.config.CopyConfig(config1)
    del config['output']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config)

    # However, when run from a real config file, the processing will write a 'root' field,
    # which it will use for the default behavior to write to root.fits.
    config['root'] = 'output_fits/test_fits_0'
    galsim.config.BuildFile(config)
    im2 = galsim.fits.read('output_fits/test_fits_0.fits')
    np.testing.assert_array_equal(im2.array, im1_list[0].array)

    # Check invalid input field
    config['input'] = { 'invalid' : {} }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ProcessInput(config)

    # Not sure if this is possible, but we have a check in case cpu_count fails, so
    # mock this up to make sure we handle it properly (by reverting to nproc = 1.
    with mock.patch('galsim.config.util.cpu_count', side_effect=RuntimeError()):
        config = galsim.config.CopyConfig(config1)
        with CaptureLog() as cl:
            galsim.config.Process(config, logger=cl.logger, new_params={'output.nproc' : -1})
        assert 'Using single process' in cl.output


@timer
def test_multifits():
    """Test the output type = MultiFits
    """
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'output' : {
            'type' : 'MultiFits',
            'nimages' : 6,
            'file_name' : 'output/test_multifits.fits'
        },
    }

    im1_list = []
    nimages = 6
    first_seed = galsim.BaseDeviate(1234).raw()
    for k in range(nimages):
        ud = galsim.UniformDeviate(first_seed + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        im1 = gal.drawImage(scale=1)
        im1_list.append(im1)
    print('multifit image shapes = ',[im.array.shape for im in im1_list])

    assert galsim.config.GetNFiles(config) == 1
    assert galsim.config.GetNImagesForFile(config, 0) == 6
    assert galsim.config.GetNObjForFile(config, 0, 0) == [1, 1, 1, 1, 1, 1]

    galsim.config.Process(config)
    im2_list = galsim.fits.readMulti('output/test_multifits.fits')
    for k in range(nimages):
        np.testing.assert_array_equal(im2_list[k].array, im1_list[k].array)

    # nimages = 1 is allowed
    config['output']['nimages'] = 1
    galsim.config.Process(config)
    im3_list = galsim.fits.readMulti('output/test_multifits.fits')
    assert len(im3_list) == 1
    np.testing.assert_array_equal(im3_list[0].array, im1_list[0].array)

    # Check error message for missing nimages
    del config['output']['nimages']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config)
    # Also if there is an input field that doesn't have nobj capability
    config['input'] = { 'dict' : { 'dir' : 'config_input', 'file_name' : 'dict.p' } }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config)

    # However, an input field that does have nobj will return something for nobjects.
    # This catalog has 3 rows, so equivalent to nobjects = 3
    config = galsim.config.CleanConfig(config)
    config['input'] = { 'catalog' : { 'dir' : 'config_input', 'file_name' : 'catalog.txt' } }
    galsim.config.BuildFile(config)
    im4_list = galsim.fits.readMulti('output/test_multifits.fits')
    assert len(im4_list) == 3
    for k in range(3):
        np.testing.assert_array_equal(im4_list[k].array, im1_list[k].array)


@timer
def test_datacube():
    """Test the output type = DataCube
    """
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'output' : {
            'type' : 'DataCube',
            'nimages' : 6,
            'file_name' : 'output/test_datacube.fits'
        },
    }

    im1_list = []
    nimages = 6
    b = None
    first_seed = galsim.BaseDeviate(1234).raw()
    for k in range(nimages):
        ud = galsim.UniformDeviate(first_seed + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        if b is None:
            im1 = gal.drawImage(scale=1)
            b = im1.bounds
        else:
            im1 = gal.drawImage(bounds=b, scale=1)
        im1_list.append(im1)
    print('datacube image shapes = ',[im.array.shape for im in im1_list])

    assert galsim.config.GetNFiles(config) == 1
    assert galsim.config.GetNImagesForFile(config, 0) == 6
    assert galsim.config.GetNObjForFile(config, 0, 0) == [1, 1, 1, 1, 1, 1]

    galsim.config.Process(config)
    im2_list = galsim.fits.readCube('output/test_datacube.fits')
    for k in range(nimages):
        np.testing.assert_array_equal(im2_list[k].array, im1_list[k].array)

    # nimages = 1 is allowed
    config['output']['nimages'] = 1
    galsim.config.Process(config)
    im3_list = galsim.fits.readCube('output/test_datacube.fits')
    assert len(im3_list) == 1
    np.testing.assert_array_equal(im3_list[0].array, im1_list[0].array)

    # Check error message for missing nimages
    del config['output']['nimages']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config)
    # Also if there is an input field that doesn't have nobj capability
    config['input'] = { 'dict' : { 'dir' : 'config_input', 'file_name' : 'dict.p' } }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config)

    # However, an input field that does have nobj will return something for nobjects.
    # This catalog has 3 rows, so equivalent to nobjects = 3
    config = galsim.config.CleanConfig(config)
    config['input'] = { 'catalog' : { 'dir' : 'config_input', 'file_name' : 'catalog.txt' } }
    galsim.config.BuildFile(config)
    im4_list = galsim.fits.readCube('output/test_datacube.fits')
    assert len(im4_list) == 3
    for k in range(3):
        np.testing.assert_array_equal(im4_list[k].array, im1_list[k].array)

    # DataCubes cannot include weight (or any other) extra outputs as additional hdus.
    # It should raise an exception if you try.
    config['output']['weight'] = { 'hdu' : 1 }
    config['output']['badpix'] = { 'file_name' : 'output/test_datacube_bp.fits' }
    config['image']['noise'] = { 'type' : 'Gaussian', 'variance' : 0.1 }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(config)

    # But if both weight and badpix are files, then it should work.
    config['output']['weight'] = { 'file_name' : 'output/test_datacube_wt.fits' }
    galsim.config.BuildFile(config)
    im5_list = galsim.fits.readCube('output/test_datacube.fits')
    assert len(im5_list) == 3
    for k in range(3):
        rng = galsim.UniformDeviate(first_seed + k + 1)
        rng.discard(1)
        im1_list[k].addNoise(galsim.GaussianNoise(sigma=0.1**0.5, rng=rng))
        np.testing.assert_array_equal(im5_list[k].array, im1_list[k].array)
    im5_wt = galsim.fits.read('output/test_datacube_wt.fits')
    im5_bp = galsim.fits.read('output/test_datacube_bp.fits')
    np.testing.assert_array_equal(im5_wt.array, 10)
    np.testing.assert_array_equal(im5_bp.array, 0)


@timer
def test_skip():
    """Test the skip and noclobber options
    """
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'output' : {
            'nfiles' : 6,
            'file_name' : "$'output/test_skip_%d.fits'%file_num",
            'skip' : { 'type' : 'Random', 'p' : 0.4 }
        },
    }

    im1_list = []
    skip_list = []
    nfiles = 6
    first_seed = galsim.BaseDeviate(1234).raw()
    for k in range(nfiles):
        file_name = 'output/test_skip_%d.fits'%k
        if os.path.exists(file_name):
            os.remove(file_name)
        ud_file = galsim.UniformDeviate(first_seed + k)
        if ud_file() < 0.4:
            print('skip k = ',k)
            skip_list.append(True)
        else:
            skip_list.append(False)
        ud = galsim.UniformDeviate(first_seed + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        im1 = gal.drawImage(scale=1)
        im1_list.append(im1)

    galsim.config.Process(config)
    for k in range(nfiles):
        file_name = 'output/test_skip_%d.fits'%k
        if skip_list[k]:
            assert not os.path.exists(file_name)
        else:
            im2 = galsim.fits.read(file_name)
            np.testing.assert_array_equal(im2.array, im1_list[k].array)

    # Build the ones we skipped using noclobber option
    del config['output']['skip']
    config['output']['noclobber'] = True
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    assert "Skipping file 1 = output/test_skip_1.fits because output.noclobber" in cl.output
    assert "Skipping file 3 = output/test_skip_3.fits because output.noclobber" in cl.output
    assert "Skipping file 5 = output/test_skip_5.fits because output.noclobber" in cl.output
    for k in range(nfiles):
        file_name = 'output/test_skip_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1_list[k].array)

    # Another way to skip files is to split the work into several jobs
    config['output']['noclobber'] = False
    for k in range(nfiles):
        file_name = 'output/test_skip_%d.fits'%k
        if os.path.exists(file_name): os.remove(file_name)
    galsim.config.Process(config, njobs=3, job=3)
    for k in range(nfiles):
        file_name = 'output/test_skip_%d.fits'%k
        if k <= 3:
            assert not os.path.exists(file_name)
        else:
            im2 = galsim.fits.read(file_name)
            np.testing.assert_array_equal(im2.array, im1_list[k].array)

    with CaptureLog() as cl:
        galsim.config.Process(config, njobs=3, job=3, logger=cl.logger)
    assert "Splitting work into 3 jobs.  Doing job 3" in cl.output
    assert "Building 2 out of 6 total files: file_num = 4 .. 5" in cl.output

    # job < 1 or job > njobs is invalid
    with assert_raises(galsim.GalSimValueError):
        galsim.config.Process(config, njobs=3, job=0)
    with assert_raises(galsim.GalSimValueError):
        galsim.config.Process(config, njobs=3, job=4)
    # Also njobs < 1 is invalid
    with assert_raises(galsim.GalSimValueError):
        galsim.config.Process(config, njobs=0)


@timer
def test_extra_wt():
    """Test the extra weight and badpix fields
    """
    nfiles = 6
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
            'pixel_scale' : 0.4,
            'noise' : { 'type' : 'Poisson', 'sky_level_pixel' : '$0.7 + image_num' }
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'output' : {
            'nfiles' : nfiles,
            'file_name' : "$'output/test_main_%d.fits'%file_num",

            'weight' : { 'file_name' : "$'output/test_wt_%d.fits'%file_num" },
            'badpix' : { 'file_name' : "$'output/test_bp_%d.fits'%file_num" },
        },
    }

    galsim.config.Process(config)

    main_im = [ galsim.fits.read('output/test_main_%d.fits'%k) for k in range(nfiles) ]
    for k in range(nfiles):
        im_wt = galsim.fits.read('output/test_wt_%d.fits'%k)
        np.testing.assert_almost_equal(im_wt.array, 1./(0.7 + k))
        im_bp = galsim.fits.read('output/test_bp_%d.fits'%k)
        np.testing.assert_array_equal(im_bp.array, 0)
        os.remove('output/test_main_%d.fits'%k)

    # If noclobber = True, don't overwrite existing file.
    config['noise'] = { 'type' : 'Poisson', 'sky_level_pixel' : 500 }
    config['output']['noclobber'] = True
    galsim.config.RemoveCurrent(config)
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    assert 'Not writing weight file 0 = output/test_wt_0.fits' in cl.output
    for k in range(nfiles):
        im = galsim.fits.read('output/test_main_%d.fits'%k)
        np.testing.assert_equal(im.array, main_im[k].array)
        im_wt = galsim.fits.read('output/test_wt_%d.fits'%k)
        np.testing.assert_almost_equal(im_wt.array, 1./(0.7 + k))

    # Can also add these as extra hdus rather than separate files.
    config['output']['noclobber'] = False
    config['output']['weight'] = { 'hdu' : 1 }
    config['output']['badpix'] = { 'hdu' : 2 }
    galsim.config.RemoveCurrent(config)
    galsim.config.Process(config)
    for k in range(nfiles):
        im_wt = galsim.fits.read('output/test_main_%d.fits'%k, hdu=1)
        np.testing.assert_almost_equal(im_wt.array, 1./(0.7 + k))
        im_bp = galsim.fits.read('output/test_main_%d.fits'%k, hdu=2)
        np.testing.assert_array_equal(im_bp.array, 0)

    config['output']['badpix'] = { 'hdu' : 0 }
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.Process(config, except_abort=True)
    config['output']['badpix'] = { 'hdu' : 1 }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.Process(config, except_abort=True)
    config['output']['badpix'] = { 'hdu' : 3 }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.Process(config, except_abort=True)

    # If include_obj_var = True, then weight image includes signal.
    config['output']['weight']['include_obj_var'] = True
    config['output']['badpix'] = { 'hdu' : 2 }
    config['output']['nproc'] = 2
    galsim.config.RemoveCurrent(config)
    galsim.config.Process(config)
    first_seed = galsim.BaseDeviate(1234).raw()
    for k in range(nfiles):
        ud = galsim.UniformDeviate(first_seed + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        im = gal.drawImage(scale=0.4)
        im_wt = galsim.fits.read('output/test_main_%d.fits'%k, hdu=1)
        np.testing.assert_almost_equal(im_wt.array, 1./(0.7 + k + im.array))

    # It is permissible for weight, badpix to have no output.  Some use cases require building
    # the weight and/or badpix information even if it is not associated with any output.
    config['output']['weight'] = {}
    config['output']['badpix'] = {}
    galsim.config.RemoveCurrent(config)
    galsim.config.Process(config)
    for k in range(nfiles):
        assert_raises(OSError, galsim.fits.read, 'output/test_main_%d.fits'%k, hdu=1)
        os.remove('output/test_wt_%d.fits'%k)
        os.remove('output/test_main_%d.fits'%k)

    # Can also have both outputs
    config['output']['weight'] = { 'file_name': "$'output/test_wt_%d.fits'%file_num", 'hdu': 1 }
    galsim.config.RemoveCurrent(config)
    galsim.config.Process(config, except_abort=True)
    for k in range(nfiles):
        im_wt1 = galsim.fits.read('output/test_wt_%d.fits'%k)
        np.testing.assert_almost_equal(im_wt1.array, 1./(0.7 + k))
        im_wt2 = galsim.fits.read('output/test_main_%d.fits'%k, hdu=1)
        np.testing.assert_almost_equal(im_wt2.array, 1./(0.7 + k))

    # Other such use cases would access the final weight or badpix image using GetFinalExtraOutput
    galsim.config.BuildFile(config)
    wt = galsim.config.extra.GetFinalExtraOutput('weight', config)
    np.testing.assert_almost_equal(wt[0].array, 1./0.7)

    # If the image is a Scattered type, then the weight and badpix images are built by a
    # different code path.
    config = {
        'image' : {
            'type' : 'Scattered',
            'random_seed' : 1234,
            'pixel_scale' : 0.4,
            'size' : 64,
            'noise' : { 'type' : 'Poisson', 'sky_level_pixel' : '$0.7 + image_num' },
            'nobjects' : 1,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'output' : {
            'nfiles' : nfiles,
            'file_name' : "$'output/test_main_%d.fits'%file_num",

            'weight' : { 'file_name' : "$'output/test_wt_%d.fits'%file_num" },
            'badpix' : { 'file_name' : "$'output/test_bp_%d.fits'%file_num" },
        },
    }

    galsim.config.Process(config)

    for k in range(nfiles):
        im_wt = galsim.fits.read('output/test_wt_%d.fits'%k)
        np.testing.assert_almost_equal(im_wt.array, 1./(0.7 + k))
        im_bp = galsim.fits.read('output/test_bp_%d.fits'%k)
        np.testing.assert_array_equal(im_bp.array, 0)

    # If include_obj_var = True, then weight image includes signal.
    config['output']['weight']['include_obj_var'] = True
    config['output']['nproc'] = 2
    galsim.config.RemoveCurrent(config)
    galsim.config.Process(config)
    first_seed = galsim.BaseDeviate(1234).raw()
    for k in range(nfiles):
        ud = galsim.UniformDeviate(first_seed + k + 1)
        x = ud() * 63 + 1
        y = ud() * 63 + 1
        ix = int(math.floor(x+1))
        iy = int(math.floor(y+1))
        dx = x-ix+0.5
        dy = y-iy+0.5

        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        im = galsim.ImageF(64,64)
        stamp = gal.drawImage(scale=0.4, offset=(dx,dy))
        stamp.setCenter(ix,iy)
        b = im.bounds & stamp.bounds
        im[b] = stamp[b]
        im_wt = galsim.fits.read('output/test_wt_%d.fits'%k)
        np.testing.assert_almost_equal(im_wt.array, 1./(0.7 + k + im.array))

    # If both output.nproc and image.nproc, then only use output.nproc
    config['image']['nproc' ] = -1
    config['image']['nobjects'] = 5
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    #print(cl.output)
    #assert 'Already multiprocessing.  Ignoring image.nproc' in cl.output
    # Note: This doesn't show up because cl.logger doesn't get through the multiprocessing,
    #       but it does ignore image.nproc > 1.
    # Do it manually to confirm.
    config['current_nproc'] = 2
    with CaptureLog() as cl:
        nproc = galsim.config.UpdateNProc(2, 5, config, logger=cl.logger)
    assert 'Already multiprocessing.  Ignoring image.nproc' in cl.output
    assert nproc == 1


@timer
def test_extra_psf():
    """Test the extra psf field
    """
    nfiles = 6
    config = {
        'image' : {
            'type' : 'Scattered',
            'random_seed' : 1234,
            'nobjects' : 1,
            'pixel_scale' : 0.4,
            'size' : 64,
            'stamp_size' : 25,
            'image_pos' : { 'type' : 'XY',  # Some of these are intentionally off the imgae.
                            'x' : { 'type': 'Random', 'min': -30, 'max': 100 },
                            'y' : { 'type': 'Random', 'min': -30, 'max': 100 } },
            'offset' : { 'type' : 'XY',
                         'x' : { 'type': 'Random', 'min': -0.5, 'max': 0.5 },
                         'y' : { 'type': 'Random', 'min': -0.5, 'max': 0.5 } },
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'shift' : { 'type' : 'XY',
                        'x' : { 'type': 'Random', 'min': -1, 'max': 1 },
                        'y' : { 'type': 'Random', 'min': -1, 'max': 1 } },
            'flux' : 100,
        },
        'psf' : {
            'type' : 'Moffat',
            'beta' : 3.5,
            'fwhm' : { 'type': 'Random', 'min': 0.5, 'max': 0.9 },
        },
        'output' : {
            'nfiles' : nfiles,
            'file_name' : "$'output/test_gal_%d.fits'%file_num",

            'psf' : { 'file_name' : "$'output/test_psf_%d.fits'%file_num", }
        },
    }

    for f in glob.glob('output/test_psf_*.fits'): os.remove(f)
    for f in glob.glob('output/test_gal_*.fits'): os.remove(f)
    galsim.config.Process(config)

    gal_center = []
    gal_dxy = []
    gal_shift = []
    gal_offset = []
    psf_fwhm = []

    first_seed = galsim.BaseDeviate(1234).raw()
    for k in range(nfiles):
        ud = galsim.UniformDeviate(first_seed + k + 1)
        x = ud() * 130 - 30
        y = ud() * 130 - 30
        ix = int(math.floor(x+0.5))
        iy = int(math.floor(y+0.5))
        dx = x-ix
        dy = y-iy

        fwhm = ud() * 0.4 + 0.5
        psf = galsim.Moffat(beta=3.5, fwhm=fwhm)

        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)

        shift_x = ud() * 2. - 1.
        shift_y = ud() * 2. - 1.
        gal = gal.shift(shift_x, shift_y)

        offset_x = ud() - 0.5
        offset_y = ud() - 0.5

        # Store values for later loops
        gal_center.append( (ix,iy) )
        gal_dxy.append( (dx,dy) )
        gal_shift.append( (shift_x, shift_y) )
        gal_offset.append( (offset_x, offset_y) )
        psf_fwhm.append(fwhm)

        final = galsim.Convolve(gal, psf)
        im = galsim.ImageF(64,64)
        stamp = final.drawImage(scale=0.4, nx=25, ny=25, offset=(offset_x+dx,offset_y+dy))
        stamp.setCenter(ix,iy)
        b = im.bounds & stamp.bounds
        if b.isDefined():
            im[b] = stamp[b]
        im2 = galsim.fits.read('output/test_gal_%d.fits'%k)
        np.testing.assert_almost_equal(im2.array, im.array)

        # Default is for the PSF to be centered at (x,y).  No shift, no offset. (But still dx,dy)
        im.setZero()
        stamp = psf.drawImage(scale=0.4, nx=25, ny=25, offset=(dx,dy))
        stamp.setCenter(ix,iy)
        if b.isDefined():
            im[b] = stamp[b]
        im2 = galsim.fits.read('output/test_psf_%d.fits'%k)
        np.testing.assert_almost_equal(im2.array, im.array)

    # Now have the psf shift and offset match the galaxy
    config['output']['psf']['shift'] = 'galaxy'
    config['output']['psf']['offset'] = 'galaxy'
    galsim.config.RemoveCurrent(config)
    galsim.config.Process(config)
    for k in range(nfiles):
        ix, iy = gal_center[k]
        dx, dy = gal_dxy[k]
        sx, sy = gal_shift[k]
        ox, oy = gal_offset[k]
        psf = galsim.Moffat(beta=3.5, fwhm=psf_fwhm[k])
        psf = psf.shift(sx,sy)
        stamp = psf.drawImage(scale=0.4, nx=25, ny=25, offset=(ox+dx,oy+dy))
        stamp.setCenter(ix,iy)
        im = galsim.ImageF(64,64)
        b = im.bounds & stamp.bounds
        if b.isDefined():
            im[b] = stamp[b]
        im2 = galsim.fits.read('output/test_psf_%d.fits'%k)
        np.testing.assert_almost_equal(im2.array, im.array)

    # Can also define custom shift and/or offset for the psf sepatately from the galaxy.
    config['output']['psf']['shift'] = {
        'type' : 'XY',
        'x' : { 'type': 'Random', 'min': -1, 'max': 1 },
        'y' : { 'type': 'Random', 'min': -1, 'max': 1 }
    }
    config['output']['psf']['offset'] = {
        'type' : 'XY',
        'x' : { 'type': 'Random', 'min': -0.5, 'max': 0.5 },
        'y' : { 'type': 'Random', 'min': -0.5, 'max': 0.5 }
    }
    # Also, let's test the ability of the extra fields to be in a different directory.
    if os.path.exists('output_psf'):
        shutil.rmtree('output_psf')
    config['output']['psf']['dir'] = 'output_psf'
    config['output']['psf']['file_name'] = "$'test_psf_%d.fits'%file_num"
    galsim.config.RemoveCurrent(config)
    galsim.config.Process(config)
    for k in range(nfiles):
        ud = galsim.UniformDeviate(first_seed + k + 1)
        ud.discard(8)  # The ud() calls for the galaxy precede the extra_output calls.
        ix, iy = gal_center[k]
        dx, dy = gal_dxy[k]

        psf = galsim.Moffat(beta=3.5, fwhm=psf_fwhm[k])

        shift_x = ud() * 2. - 1.
        shift_y = ud() * 2. - 1.
        psf = psf.shift(shift_x, shift_y)

        offset_x = ud() - 0.5
        offset_y = ud() - 0.5

        stamp = psf.drawImage(scale=0.4, nx=25, ny=25, offset=(offset_x+dx,offset_y+dy))
        stamp.setCenter(ix,iy)
        im = galsim.ImageF(64,64)
        b = im.bounds & stamp.bounds
        if b.isDefined():
            im[b] = stamp[b]
        im2 = galsim.fits.read('output_psf/test_psf_%d.fits'%k)
        np.testing.assert_almost_equal(im2.array, im.array)

    # Finally, another mode that is allowed is to only write a single PSF file to correspond to
    # multiple image files
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
            'size' : 32,
            'pixel_scale' : 0.4,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'psf' : {
            'type' : 'Moffat',
            'beta' : 3.5,
            'fwhm' : 0.9,
        },
        'output' : {
            'nfiles' : nfiles,
            'file_name' : "$'output/test_gal_%d.fits'%file_num",
            'psf' : { 'file_name' : 'output_psf/test_psf.fits' }
        },
    }
    galsim.config.Process(config)

    psf = galsim.Moffat(beta=3.5, fwhm=0.9)
    im = psf.drawImage(scale=0.4, nx=32, ny=32)
    im2 = galsim.fits.read('output_psf/test_psf.fits')
    np.testing.assert_almost_equal(im2.array, im.array)

    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    assert "Not writing psf file 1 = output_psf/test_psf.fits because already written" in cl.output
    assert "Not writing psf file 2 = output_psf/test_psf.fits because already written" in cl.output
    assert "Not writing psf file 3 = output_psf/test_psf.fits because already written" in cl.output
    assert "Not writing psf file 4 = output_psf/test_psf.fits because already written" in cl.output
    assert "Not writing psf file 5 = output_psf/test_psf.fits because already written" in cl.output

@timer
def test_extra_psf_sn():
    """Test the signal_to_noise option of the extra psf field
    """
    config = {
        'image' : {
            'random_seed' : 1234,
            'pixel_scale' : 0.4,
            'size' : 64,
            'dtype': 'float',
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 2.3,
            'flux' : 100,
        },
        'psf' : {
            'type' : 'Moffat',
            'beta' : 3.5,
            'fwhm' : 0.7,
            'gsparams' : { 'maxk_threshold': 3.e-4 }
        },
        'output' : {
            'psf' : {}
        },
    }
    # First pure psf image with no noise.
    gal_image = galsim.config.BuildImage(config)
    pure_psf_image = galsim.config.extra.GetFinalExtraOutput('psf', config)[0]
    assert gal_image.dtype is np.float64
    assert pure_psf_image.dtype is np.float64  # PSF gets dtype from main image
    np.testing.assert_almost_equal(pure_psf_image.array.sum(), 1., decimal=6)

    # Draw PSF at S/N = 100
    # (But first check that an error is raised if noise is missing.
    config['output']['psf']['signal_to_noise'] = 100
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    noise_var = 20.
    config['image']['noise'] = { 'type' : 'Gaussian', 'variance' : noise_var, }
    gal_image = galsim.config.BuildImage(config)
    sn100_psf_image = galsim.config.extra.GetFinalExtraOutput('psf', config)[0]
    sn100_flux = sn100_psf_image.array.sum()
    psf_noise = sn100_psf_image - sn100_flux * pure_psf_image
    print('psf_noise.var = ',psf_noise.array.var(), noise_var)
    np.testing.assert_allclose(psf_noise.array.var(), noise_var, rtol=0.02)
    snr = np.sqrt( np.sum(sn100_psf_image.array**2, dtype=float) / noise_var )
    print('snr = ',snr, 100)
    np.testing.assert_allclose(snr, 100, rtol=0.25)  # Not super accurate for any single image.

    # Can also specify different draw_methods.
    config['output']['psf']['draw_method'] = 'real_space'
    galsim.config.RemoveCurrent(config)
    gal_image = galsim.config.BuildImage(config)
    real_psf_image = galsim.config.extra.GetFinalExtraOutput('psf', config)[0]
    print('real flux = ', real_psf_image.array.sum(), sn100_flux)
    np.testing.assert_allclose(real_psf_image.array.sum(), sn100_flux, rtol=1.e-4)

    # phot is invalid with signal_to_noise
    config['output']['psf']['draw_method'] = 'phot'
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # Check for other invalid input
    config['output']['psf']['draw_method'] = 'input'
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)
    config['output']['psf']['draw_method'] = 'auto'
    config['output']['psf']['flux'] = sn100_flux
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # OK to use phot with flux.
    config['output']['psf']['draw_method'] = 'phot'
    del config['output']['psf']['signal_to_noise']
    gal_image = galsim.config.BuildImage(config)
    phot_psf_image = galsim.config.extra.GetFinalExtraOutput('psf', config)[0]
    print('phot flux = ', phot_psf_image.array.sum(), sn100_flux)
    np.testing.assert_allclose(phot_psf_image.array.sum(), sn100_flux, rtol=1.e-4)


@timer
def test_extra_truth():
    """Test the extra truth field
    """
    nobjects = 6
    config = {
        # Custom type in this dir.  Lets us use HSM_Shape
        'modules' : ['hsm_shape'],
        'image' : {
            'type' : 'Tiled',
            'nx_tiles' : nobjects,
            'ny_tiles' : 1,
            'stamp_xsize' : 32,
            'stamp_ysize' : 32,
            'random_seed' : 1234,
            'pixel_scale' : 0.2,
            'nproc' : 2,
        },
        'psf' : {
            'type': 'Gaussian',
            'sigma': 0.5,
        },
        'gal' : {
            'type' : 'List',
            'items' : [
                {
                    'type': 'Gaussian',
                    'sigma': 1.e-6,
                    # Notably, this has no ellip field.
                    # The workaround below for setting an effective ellip value in the truth
                    # catalog to deal with this used to not work.
                },
                {
                    'type': 'Gaussian',
                    'sigma': {
                                'type': 'Random_float',
                                'min': 1,
                                'max': 2,
                             },
                    'ellip': {'type': 'EBeta', 'e': 0.2, 'beta': {'type': 'Random'} },
                },
            ],
            'flux': { 'type': 'Random', 'min': '$obj_num+1', 'max': '$(obj_num+1) * 4' },
            # 1/3 of objects are stars.
            'index': '$0 if obj_num % 3 == 0 else 1',
        },
        'output' : {
            'type' : 'Fits',
            'file_name' : 'output/test_truth.fits',
            'truth' : {
                'hdu' : 1,
                'columns' : OrderedDict([
                    ('object_id' , 'obj_num'),
                    ('index' , 'gal.index'),
                    ('flux' , '@gal.flux'), # The @ is not required, but allowed.
                    # Check several different ways to do calculations
                    ('sigma' , '$@gal.items.0.sigma if @gal.index==0 else @gal.items.1.sigma'),
                    ('g' , {
                        'type': 'Eval',
                        'str': '0. if @gal.index==0 else (@gal.items.1.ellip).g',
                    }),
                    ('beta' , '$0. if @gal.index==0 else (@gal.items.1.ellip).beta.rad'),
                    ('hlr' , '$@output.truth.columns.sigma * np.sqrt(2.*math.log(2))'),
                    ('fwhm' , '$(@gal).original.fwhm if @gal.index == 1 else (@gal).fwhm'),
                    ('pos' , 'image_pos'),
                    # slightly gratuitous here.  Use int16 to force a check that np.integer works.
                    ('obj_type_i' , '$np.int16(@gal.index)'),
                    ('obj_type_s' , '$"gal" if @gal.index else "star"'),
                    # Can also just be a constant value.
                    ('run_num' , 17),
                    ('shape' , { 'type' : 'HSM_Shape' }),
                ])
            }
        }
    }

    galsim.config.ImportModules(config)
    galsim.config.Process(config)

    sigma = np.empty(nobjects)
    flux = np.empty(nobjects)
    g = np.empty(nobjects)
    beta = np.empty(nobjects)
    meas_g1 = np.empty(nobjects)
    meas_g2 = np.empty(nobjects)
    obj_type_i = np.empty(nobjects, dtype=int)
    obj_type_s = [None] * nobjects
    first_seed = galsim.BaseDeviate(1234).raw()
    for k in range(nobjects):
        ud = galsim.UniformDeviate(first_seed + k + 1)
        if k%3 == 0:
            sigma[k] = 1.e-6
            g[k] = 0.
            beta[k] = 0.
            obj_type_i[k] = 0
            obj_type_s[k] = 'star'
            gal = galsim.Gaussian(sigma=sigma[k])
        else:
            sigma[k] = ud() + 1
            shear = galsim.Shear(e=0.2, beta=ud() * 2*np.pi * galsim.radians)
            g[k] = shear.g
            beta[k] = shear.beta.rad
            obj_type_i[k] = 1
            obj_type_s[k] = 'gal'
            gal = galsim.Gaussian(sigma=sigma[k]).shear(shear)
        flux[k] = (k+1) * (ud() * 3 + 1)
        gal = gal.withFlux(flux[k])
        psf = galsim.Gaussian(sigma=0.5)
        obj = galsim.Convolve(psf,gal)
        meas_shape = obj.drawImage(nx=32,ny=32,scale=0.2).FindAdaptiveMom().observed_shape
        meas_g1[k] = meas_shape.g1
        meas_g2[k] = meas_shape.g2

    file_name = 'output/test_truth.fits'
    cat = galsim.Catalog(file_name, hdu=1)
    obj_num = np.array(range(nobjects))
    np.testing.assert_almost_equal(cat.data['object_id'], obj_num)
    np.testing.assert_equal(cat.data['index'], obj_type_i)
    np.testing.assert_almost_equal(cat.data['flux'], flux)
    np.testing.assert_almost_equal(cat.data['sigma'], sigma)
    np.testing.assert_almost_equal(cat.data['g'], g)
    np.testing.assert_almost_equal(cat.data['beta'], beta)
    np.testing.assert_equal(cat.data['obj_type_i'], obj_type_i)
    np.testing.assert_equal(cat.data['obj_type_s'], obj_type_s)
    np.testing.assert_almost_equal(cat.data['hlr'], sigma * galsim.Gaussian._hlr_factor)
    np.testing.assert_almost_equal(cat.data['fwhm'], sigma * galsim.Gaussian._fwhm_factor)
    np.testing.assert_almost_equal(cat.data['pos.x'], obj_num * 32 + 16.5)
    np.testing.assert_almost_equal(cat.data['pos.y'], 16.5)
    np.testing.assert_almost_equal(cat.data['run_num'], 17)
    np.testing.assert_almost_equal(cat.data['shape.g1'], meas_g1)
    np.testing.assert_almost_equal(cat.data['shape.g2'], meas_g2)

    # If types are not consistent for all objects, raise an error.
    # Here it's a float for stars and Angle for galaxies.
    config['output']['truth']['columns']['beta'] = (
        '$0. if @gal.index==0 else (@gal.items.1.ellip).beta')
    del config['image']['nproc']
    with CaptureLog(level=1) as cl:
        with assert_raises(galsim.GalSimConfigError):
            galsim.config.Process(config, logger=cl.logger)
    assert "beta has type Angle, but previously had type float" in cl.output
    config['output']['truth']['columns']['beta'] = (
        '$0. if @gal.index==0 else (@gal.items.1.ellip).beta.rad')

    # If we don't use Random_float, the truth catalog can't figure out the type of gal.sigma
    # when it's used as @gal.items.1.sigma before being calculated.
    # This gives and error, but also a suggestion for how it might be remedied.
    config['gal']['items'][1]['sigma'] = {
        'type': 'Random', 'min': 1, 'max': 2, 'rng_index_key': 'image_num' }
    try:
        galsim.config.Process(config)
        # This is effectively doing assert_raises, but we want to check the error string.
        assert False
    except galsim.GalSimConfigError as e:
        print(e)
        assert 'Consider using an explicit value-typed type name like Random_float' in str(e)

@timer
def test_retry_io():
    """Test the retry_io option
    """
    # Make a class that mimics writeMulti, except that it fails about 1/3 of the time.
    class FlakyWriter:
        def __init__(self, rng):
            self.ud = galsim.UniformDeviate(rng)
        def writeFile(self, *args, **kwargs):
            p = self.ud()
            if p < 0.33:
                raise OSError("p = %f"%p)
            else:
                galsim.fits.writeMulti(*args, **kwargs)

    # Now make a copy of Fits and ExtraWeight using this writer.
    class FlakyFits(galsim.config.OutputBuilder):
        def writeFile(self, data, file_name, config, base, logger):
            flaky_writer = FlakyWriter(galsim.config.GetRNG(config,base))
            flaky_writer.writeFile(data, file_name)
    galsim.config.RegisterOutputType('FlakyFits', FlakyFits())

    class FlakyWeight(galsim.config.extra_weight.WeightBuilder):
        def writeFile(self, file_name, config, base, logger):
            flaky_writer = FlakyWriter(galsim.config.GetRNG(config,base))
            flaky_writer.writeFile(self.final_data, file_name)
    galsim.config.RegisterExtraOutput('flaky_weight', FlakyWeight())

    galsim.config.output._sleep_mult = 1.e-10  # Don't take forever testing this.

    nfiles = 6
    config = {
        'image' : {
            'type' : 'Single',
            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : 100,
        },
        'output' : {
            'type' : 'FlakyFits',
            'nfiles' : nfiles,
            'retry_io': 5,
            'file_name' : "$'output/test_flaky_fits_%d.fits'%file_num",
            'flaky_weight' : { 'file_name' : "$'output/test_flaky_wt_%d.fits'%file_num" },
        },
    }

    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    #print(cl.output)
    assert "File output/test_flaky_fits_0.fits: Caught OSError" in cl.output
    assert "This is try 2/6, so sleep for 2 sec and try again." in cl.output
    assert "file 0: Wrote FlakyFits to file 'output/test_flaky_fits_0.fits'" in cl.output
    assert "File output/test_flaky_wt_4.fits: Caught OSError: " in cl.output
    assert "This is try 1/6, so sleep for 1 sec and try again." in cl.output
    assert "file 0: Wrote flaky_weight to 'output/test_flaky_wt_0.fits'" in cl.output

    # Now the regular versions.
    config2 = galsim.config.CopyConfig(config)
    config2['output'] = {
        'type' : 'Fits',
        'nfiles' : nfiles,
        'file_name' : "$'output/test_nonflaky_fits_%d.fits'%file_num",
        'weight' : { 'file_name' : "$'output/test_nonflaky_wt_%d.fits'%file_num" },
    }
    galsim.config.Process(config2)

    for k in range(nfiles):
        im1 = galsim.fits.read('output/test_flaky_fits_%d.fits'%k)
        im2 = galsim.fits.read('output/test_nonflaky_fits_%d.fits'%k)
        np.testing.assert_allclose(im1.array, im2.array, atol=1.e-15)
        wt1 = galsim.fits.read('output/test_flaky_wt_%d.fits'%k)
        wt2 = galsim.fits.read('output/test_nonflaky_wt_%d.fits'%k)
        np.testing.assert_allclose(wt1.array, wt2.array, atol=1.e-15)

    # Without retry_io, it will fail, but keep going
    del config['output']['retry_io']
    galsim.config.RemoveCurrent(config)
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    #print(cl.output)
    assert "Exception caught for file 0 = output/test_flaky_fits_0.fits" in cl.output
    assert "File output/test_flaky_fits_0.fits not written! Continuing on..." in cl.output
    assert "file 1: Wrote FlakyFits to file 'output/test_flaky_fits_1.fits'" in cl.output
    assert "File 1 = output/test_flaky_fits_1.fits" in cl.output
    assert "File 2 = output/test_flaky_fits_2.fits" in cl.output
    assert "File 3 = output/test_flaky_fits_3.fits" in cl.output
    assert "Exception caught for file 4 = output/test_flaky_fits_4.fits" in cl.output
    assert "File output/test_flaky_fits_4.fits not written! Continuing on..." in cl.output
    assert "File 5 = output/test_flaky_fits_5.fits" in cl.output

    # Also works in nproc > 1 mode
    config['output']['nproc'] = 2
    galsim.config.RemoveCurrent(config)
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    #print(cl.output)
    if galsim.config.UpdateNProc(2, nfiles, config) > 1:
        assert re.search("Process-.: Exception caught for file 0 = output/test_flaky_fits_0.fits",
                         cl.output)
        assert "File output/test_flaky_fits_0.fits not written! Continuing on..." in cl.output
        assert re.search("Process-.: File 1 = output/test_flaky_fits_1.fits", cl.output)
        assert re.search("Process-.: File 2 = output/test_flaky_fits_2.fits", cl.output)
        assert re.search("Process-.: File 3 = output/test_flaky_fits_3.fits", cl.output)
        assert re.search("Process-.: Exception caught for file 4 = output/test_flaky_fits_4.fits",
                         cl.output)
        assert "File output/test_flaky_fits_4.fits not written! Continuing on..." in cl.output
        assert re.search("Process-.: File 5 = output/test_flaky_fits_5.fits", cl.output)

    # But with except_abort = True, it will stop after the first failure
    del config['output']['nproc']  # Otherwise which file fails in non-deterministic.
    with CaptureLog() as cl:
        try:
            galsim.config.Process(config, logger=cl.logger, except_abort=True)
        except OSError as e:
            assert str(e) == "p = 0.285159"
    #print(cl.output)
    assert "File output/test_flaky_fits_0.fits not written." in cl.output


@timer
def test_config():
    """Test that configuration files are read, copied, and merged correctly.
    """
    config = {
        'gal' : { 'type' : 'Gaussian', 'sigma' : 2.3,
                  'flux' : { 'type' : 'List', 'items' : [ 100, 500, 1000 ] } },
        'psf' : { 'type' : 'Convolve',
                  'items' : [
                    {'type' : 'Moffat', 'beta' : 3.5, 'fwhm' : 0.9 },
                    {'type' : 'Airy', 'obscuration' : 0.3, 'lam' : 900, 'diam' : 4. } ] },
        'image' : { 'type' : 'Single', 'random_seed' : 1234, },
        'output' : { 'type' : 'Fits', 'file_name' : "test.fits", 'dir' : 'None' },
        'input' : { 'dict' : { 'dir' : 'config_input', 'file_name' : 'dict.p' } },
        'eval_variables' : { 'fpixel_scale' : 0.3 }
    }

    # Test yaml
    yaml_file_name = "output/test_config.yaml"
    with open(yaml_file_name, 'w') as fout:
        yaml.dump(config, fout, default_flow_style=True)
    # String None will be coverted to a real None.  Convert here in the comparison dict
    config['output']['dir'] = None

    config1 = galsim.config.ReadConfig(yaml_file_name)[0]
    assert config == dict(config1)
    config2 = galsim.config.ReadConfig(yaml_file_name, file_type='yaml')[0]
    assert config == dict(config2)
    config3 = galsim.config.ReadYaml(yaml_file_name)[0]
    assert config == dict(config2)

    # Test json
    json_file_name = "output/test_config.json"
    with open(json_file_name, 'w') as fout:
        json.dump(config, fout)

    config4 = galsim.config.ReadConfig(json_file_name)[0]
    assert config == dict(config4)
    config5 = galsim.config.ReadConfig(json_file_name, file_type='json')[0]
    assert config == dict(config5)
    config6 = galsim.config.ReadJson(json_file_name)[0]
    assert config == dict(config6)

    # Merging identical dicts, should do nothing
    galsim.config.MergeConfig(config1,config2)
    assert config == dict(config1)
    with CaptureLog() as cl:
        galsim.config.MergeConfig(config1,config2,logger=cl.logger)
    assert "Not merging key type from the base config" in cl.output
    assert "Not merging key items from the base config" in cl.output

    # Merging different configs does something, with the first taking precedence on conflicts
    del config5['gal']
    del config6['psf']
    config6['image']['random_seed'] = 1337
    galsim.config.MergeConfig(config5, config6)
    assert config == config5

    # Copying deep copies and removes any existing input_manager
    config4['_input_manager'] = 'an input manager'
    config7 = galsim.config.CopyConfig(config4)
    assert config == config7

    # It also works on empty config dicts (gratuitous, but adds some test coverage)
    config8 = {}
    config9 = galsim.config.CopyConfig(config8)
    assert config9 == config8

    # Check ParseExtendedKey functionality
    d,k = galsim.config.ParseExtendedKey(config,'gal.sigma')
    assert d[k] == 2.3
    d,k = galsim.config.ParseExtendedKey(config,'gal.flux.items.0')
    assert d[k] == 100
    d,k = galsim.config.ParseExtendedKey(config,'psf.items.1.diam')
    assert d[k] == 4

    # Check GetFromConfig functionality
    v = galsim.config.GetFromConfig(config,'gal.sigma')
    assert v == 2.3
    v = galsim.config.GetFromConfig(config,'gal.flux.items.0')
    assert v == 100
    v = galsim.config.GetFromConfig(config,'psf.items.1.diam')
    assert v == 4

    # Check SetInConfig functionality
    galsim.config.SetInConfig(config,'gal.sigma', 2.8)
    assert galsim.config.GetFromConfig(config,'gal.sigma') == 2.8
    galsim.config.SetInConfig(config,'gal.flux.items.0', 120)
    assert galsim.config.GetFromConfig(config,'gal.flux.items.0') == 120
    galsim.config.SetInConfig(config,'psf.items.1.diam', 8)
    assert galsim.config.GetFromConfig(config,'psf.items.1.diam') == 8

    assert_raises(ValueError, galsim.config.GetFromConfig, config, 'psf.items.lam')
    assert_raises(ValueError, galsim.config.GetFromConfig, config, 'psf.items.4')
    assert_raises(ValueError, galsim.config.GetFromConfig, config, 'psf.itms.1.lam')
    assert_raises(ValueError, galsim.config.SetInConfig, config, 'psf.items.lam', 700)
    assert_raises(ValueError, galsim.config.SetInConfig, config, 'psf.items.4', 700)
    assert_raises(ValueError, galsim.config.SetInConfig, config, 'psf.itms.1.lam', 700)

    # Check the yaml multiple document option.
    # Easiest to just read demo6 with both Yaml and Json.
    yaml_config_file = os.path.join('..','examples','demo6.yaml')
    json_config_file_1 = os.path.join('..','examples','json','demo6a.json')
    json_config_file_2 = os.path.join('..','examples','json','demo6b.json')

    configs = galsim.config.ReadConfig(yaml_config_file)
    config1 = galsim.config.ReadConfig(json_config_file_1)
    config2 = galsim.config.ReadConfig(json_config_file_2)
    assert len(configs) == 2
    assert len(config1) == 1
    assert len(config2) == 1

    # A few adjustments are required before checking that they are equal.
    # json files use '#' for comments
    del config1[0]['#']
    del config2[0]['#']
    # remove the output dirs
    del configs[0]['output']['dir']
    del configs[1]['output']['dir']
    del config1[0]['output']['dir']
    del config2[0]['output']['dir']
    # They have different parsing for 1e5, 1e6 to either string or float
    configs[1]['gal']['flux'] = eval(configs[1]['gal']['flux'])
    configs[1]['image']['sky_level'] = eval(configs[1]['image']['sky_level'])
    # Now serialize with json to force the same ordering, etc.
    s_yaml = json.dumps(configs[0], sort_keys=True)
    s_json = json.dumps(config1[0], sort_keys=True)
    assert s_yaml == s_json
    s_yaml = json.dumps(configs[1], sort_keys=True)
    s_json = json.dumps(config2[0], sort_keys=True)
    assert s_yaml == s_json

@timer
def test_no_output():
    """Technically, it is permissible to not have an output field.

    This is pretty contrived, but make sure it works as intended.
    """
    config = {
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 1.7,
            'flux' : 100,
        },
        'root' : 'output/test_no_output'  # The galsim executable sets this to the base name of
                                          # the config file.
    }
    file_name = 'output/test_no_output.fits'
    if os.path.exists(file_name):
        os.remove(file_name)
    galsim.config.Process(config)
    assert os.path.exists(file_name)
    im1 = galsim.fits.read(file_name)
    im2 = galsim.Gaussian(sigma=1.7,flux=100).drawImage(scale=1)
    np.testing.assert_equal(im1.array,im2.array)

@timer
def test_eval_full_word():
    """This test duplicates a bug that was found when using the galsim_extra FocalPlane type.
    It's a bit subtle.  The FocalPlane builder sets up some eval_variables with extra things
    that can be used in Eval items like the center of the exposure, the min/max RA and Dec,
    the distance of an object from the center of the exposure, etc.

    Two of these are focal_r and focal_rmax.  The former is calculated for any given object
    and gives the radial distance from the center of the focal plane.  The latter gives the
    maximum possible radial distance of any possible object (based on the outermost chip
    corners).

    The bug that turned up was that focal_rmax was accessed when loading an input power_spectrum,
    which would also trigger the evaluation of focal_r, since that string was also located in
    the eval string.  But this led to problems, since focal_r was based on world_pos, but that
    was intended to be used with obj_num rngs, which wasn't set up set at the time time input
    stuff is processed.

    So there are two fixes to this, which this test checks.  First, the setup of the file-level
    RNG also sets up the object-level RNG properly, so it doesn't matter if focal_r is accessed
    at this point.  And second, the eval code now matches to the full word, not just any portion
    of a word, so shorter eval_variables (focal_r in this case) won't get evaluated gratuitously.

    In additon to testing that issue, we also include another feature where we originally ran into
    trouble.  Namely having the number of objects be random in each exposure, but have the random
    number seed for most things repeat for all images in each exposure, which needs to know the
    number of objects in the exposure.  The salient aspects of this are duplicated here by
    using MultiFits with the objects being identical for each image in the file.
    """

    # Much of this is copied from the FocalPlane implementation or the focal_quick.yaml file
    # in the galsim_extra repo.
    config = {
        'eval_variables': {
            # focal_r is a useful calculation that galaxy/PSF properties might want to depend on.
            # It is intended to be accessed as an object property.
            'ffocal_r' : {
                'type' : 'Eval',
                'str' : "math.sqrt(pos.x**2 + pos.y**2)",
                'ppos' : {
                    'type' : 'Eval',
                    'str' : "galsim.PositionD((uv/galsim.arcsec for uv in world_center.project(world_pos)))",
                    'cworld_pos' : "@image.world_pos"
                }
            },
            # FocalPlane calculates the below values, including particularly focal_rmax, based on
            # the WCS's and sets the value in the config dict for each exposure.
            # They may be used by objects in conjunction with focal_r, but in this case it is also
            # used by the input PSF power spectrum (below) to set the overall scale of the fft
            # grid. This is where the bug related to full words in the Eval code came into play.
            'ffocal_rmax' : 25.,
            'afov_minra' : '-15 arcsec',
            'afov_maxra' : '15 arcsec',
            'afov_mindec' : '-15 arcsec',
            'afov_maxdec' : '15 arcsec',

            'fseeing' : {
                'type' : 'RandomGaussian',
                'mean' : 0.7,
                'sigma' : 0.1,
                'index_key' : 'image_num'  # Seeing changes each exposure
            }
        },

        'input' :  {
            'power_spectrum' : {
                'e_power_function': '(k**2 + (1./180)**2)**(-11./6.)',
                'b_power_function': '@input.power_spectrum.e_power_function',
                'units': 'arcsec',
                'grid_spacing': 10,
                'ngrid': '$math.ceil(2*focal_rmax / @input.power_spectrum.grid_spacing)',
            },
        },

        'image' : {
            'type' : 'Scattered',
            'xsize' : 100,
            'ysize' : 100,
            # This is similar to the tricky random number generation issue that we ran into in
            # FocalPlane.  That repeated for each exp_num, rather than file_num, but the issue
            # is basically the same.
            'random_seed' : [
                # Used for noise and nobjects.
                12345,
                # Used for objects.  Repeats sequence for each image in file
                {
                    'type' : 'Eval',
                    'index_key' : 'obj_num',
                    'str' : '314159 + start_obj_num + (obj_num - start_obj_num) % nobjects',
                    'inobjects' : { 'type' : 'Current', 'key' : 'image.nobjects' }
                },
            ],

            # We also used to have problems with this being a random value, so keep that feature
            # here as well.
            'nobjects' : {
                'type' : 'RandomPoisson',
                'index_key' : 'file_num',
                'mean' : 10  # Normally much more of course.
            },

            'noise' : { 'type' : 'Gaussian', 'sigma' : 10 },

            # FocalPlane sets this for each exposure. We'll use the same thing for all files here.
            'world_center' : galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees),

            # focal_r depends on world_pos, so let's copy that as is from the galsim_extra
            # config file, focal_quick.yaml, where we used to have problems.
            'world_pos': {
                'rng_num' : 1,
                'type': 'RADec',
                'ra': {
                    'type': 'Radians',
                    'theta': { 'type': 'Random', 'min': "$fov_minra.rad", 'max': "$fov_maxra.rad" }
                },
                'dec': {
                    'type': 'Radians',
                    'theta': {
                        'type': 'RandomDistribution',
                        'function': "math.cos(x)",
                        'x_min': "$fov_mindec.rad",
                        'x_max': "$fov_maxdec.rad",
                    }
                }
            },

            # We have to have a CelestialWCS to use CelestialCoords for world_pos.
            # This one is about as simple as it gets.
            'wcs': {
                'type': 'Tan',
                'dudx': 0.26, 'dudy': 0., 'dvdx': 0., 'dvdy': 0.26,
                'origin' : galsim.PositionD(50,50),
                'ra' : '0 deg', 'dec' : '0 deg',
            }

        },

        'output' : {
            # Not using the FocalPlane type, since that's a galsim_extra thing.  But we can
            # get the same complications in terms of the random number of objects by using
            # MultiFits output, and have the random_seed repeat for each image in a file.
            'type' : 'MultiFits',
            'nimages' : 2,
            'nfiles' : 2,
            'file_name' : "$'output/test_eval_full_word_{0}.fits'.format(file_num)",
            'truth' : {
                'file_name' : "$'output/test_eval_full_word_{0}.dat'.format(file_num)",
                'columns' : {
                    'num' : 'obj_num',
                    'exposure' : 'image_num',
                    'pos' : 'image_pos',
                    'ra' : 'image.world_pos.ra',
                    'dec' : 'image.world_pos.dec',
                    'flux' : 'gal.flux',
                    'size' : 'gal.sigma',
                    'psf_fwhm' : 'psf.fwhm',
                }
            }
        },

        'psf' : {
            'type' : 'Moffat',
            'beta' : 3.0,
            # Size of PSF ranges from 0.7 to 0.9 over the focal plane
            'fwhm' : '$seeing + 0.2 * (focal_r / focal_rmax)**2',
        },

        'gal' : {
            'rng_num' : 1,
            # Keep the galaxy simple, but with random components.
            'type' : 'Gaussian',
            'sigma' : { 'type' : 'Random', 'min': 0.5, 'max': 1.5 },
            'flux' : { 'type' : 'Random', 'min': 5000, 'max': 25000 },
        }
    }

    logger = logging.getLogger('test_eval_full_word')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)
    galsim.config.Process(config, logger=logger, except_abort=True)

    # First check the truth catalogs
    data0 = np.genfromtxt('output/test_eval_full_word_0.dat', names=True, deletechars='')
    data1 = np.genfromtxt('output/test_eval_full_word_1.dat', names=True, deletechars='')

    n1 = 11  # 11 obj each for first two exposures
    n2 = 14  # 14 obj each for next two exposures
    assert len(data0) == 2*n1
    assert len(data1) == 2*n2
    data00 = data0[:n1]
    data01 = data0[n1:]
    data10 = data1[:n2]
    data11 = data1[n2:]

    # Check exposure = image_num
    np.testing.assert_array_equal(data00['exposure'], 0)
    np.testing.assert_array_equal(data01['exposure'], 1)
    np.testing.assert_array_equal(data10['exposure'], 2)
    np.testing.assert_array_equal(data11['exposure'], 3)

    # Check obj_num
    np.testing.assert_array_equal(data00['num'], range(0,n1))
    np.testing.assert_array_equal(data01['num'], range(n1,2*n1))
    np.testing.assert_array_equal(data10['num'], range(2*n1,2*n1+n2))
    np.testing.assert_array_equal(data11['num'], range(2*n1+n2,2*n1+2*n2))

    # Check that galaxy properties are identical within exposures, but different across exposures
    for key in ['pos.x', 'pos.y', 'ra.rad', 'dec.rad', 'flux', 'size']:
        np.testing.assert_array_equal(data00[key], data01[key])
        np.testing.assert_array_equal(data10[key], data11[key])
        assert np.all(np.not_equal(data00[key], data10[key][:n1]))

    # PSFs should all be different, but only in the mean
    assert np.all(np.not_equal(data00['psf_fwhm'], data01['psf_fwhm']))
    assert np.all(np.not_equal(data10['psf_fwhm'], data11['psf_fwhm']))
    assert np.all(np.not_equal(data00['psf_fwhm'], data10['psf_fwhm'][:n1]))
    np.testing.assert_array_almost_equal(data00['psf_fwhm'] - np.mean(data00['psf_fwhm']),
                                         data01['psf_fwhm'] - np.mean(data01['psf_fwhm']))
    np.testing.assert_array_almost_equal(data10['psf_fwhm'] - np.mean(data10['psf_fwhm']),
                                         data11['psf_fwhm'] - np.mean(data11['psf_fwhm']))

    # Finally the images should be different, but almost equal, since the different should only
    # be in the Gaussian noise.
    im00, im01 = galsim.fits.readMulti('output/test_eval_full_word_0.fits')
    assert np.all(np.not_equal(im00.array, im01.array))
    assert abs(np.mean(im00.array - im01.array)) < 0.5
    assert 13.5 < np.std(im00.array - im01.array) < 15  # should be ~10 * sqrt(2)
    assert np.max(np.abs(im00.array)) > 200  # Just verify that many values are quite large

    im10, im11 = galsim.fits.readMulti('output/test_eval_full_word_1.fits')
    assert np.all(np.not_equal(im10.array, im11.array))
    assert abs(np.mean(im10.array - im11.array)) < 0.5
    assert 13.5 < np.std(im10.array - im11.array) < 15
    assert np.max(np.abs(im10.array)) > 200

@timer
def test_timeout():
    """Test the timeout option
    """
    config = {
        'image' : {
            'type' : 'Scattered',
            'random_seed' : 1234,
            'nobjects' : 5,
            'pixel_scale' : 0.3,
            'size' : 128,
            'image_pos' : { 'type' : 'XY',
                            'x' : { 'type': 'Random', 'min': 10, 'max': 54 },
                            'y' : { 'type': 'Random', 'min': 10, 'max': 54 } },
        },
        'gal' : {
            'type' : 'Sersic',
            'flux' : 100,
            # Note: Making n random means the image creation is moderately slow
            # (since a new Hankel transform is done for each one in SersicInfo)
            # But don't let max be too large so it's not very slow!
            'n' : { 'type': 'Random', 'min': 3, 'max': 4 },
            'half_light_radius' : { 'type': 'Random', 'min': 1, 'max': 2 },
        },
        'psf' : {
            'type' : 'Moffat',
            'fwhm' : { 'type': 'Random', 'min': 0.3, 'max': 0.6 },
            'beta' : { 'type': 'Random', 'min': 1.5, 'max': 6 },
        },
        'output' : {
            'type' : 'Fits',
            'nfiles' : 6,
            'file_name' : "$'output/test_timeout_%d.fits'%file_num",
        },
    }

    logger = logging.getLogger('test_timeout')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    #logger.setLevel(logging.DEBUG)

    # Single proc:
    config1 = galsim.config.CopyConfig(config)
    galsim.config.Process(config1, logger=logger)

    # nproc in output field.
    config2 = galsim.config.CopyConfig(config)
    config2['output']['nproc'] = 3
    config2['output']['timeout'] = 30  # Still plenty large enough not to timeout.
    config2['output']['file_name'] = "$'output/test_timeout_nproc1_%d.fits'%file_num"
    galsim.config.Process(config2, logger=logger)
    for n in range(6):
        im1 = galsim.fits.read('output/test_timeout_%d.fits'%n)
        im2 = galsim.fits.read('output/test_timeout_nproc1_%d.fits'%n)
        assert im1 == im2

    # Check that it behaves sensibly if it hits timeout limit.
    # (PyPy doesn't seem to timeout, so skip this on PyPy.)
    if platform.python_implementation() != 'PyPy':
        config2 = galsim.config.CleanConfig(config2)
        config2['output']['timeout'] = 0.0001
        with CaptureLog() as cl:
            with assert_raises(galsim.GalSimError):
                galsim.config.Process(config2, logger=cl.logger)
        assert 'Multiprocessing timed out waiting for a task to finish.' in cl.output

    # nproc in image field.
    config2 = galsim.config.CopyConfig(config)
    config2['image']['nproc'] = 3
    config2['image']['timeout'] = 30
    config2['output']['file_name'] = "$'output/test_timeout_nproc2_%d.fits'%file_num"
    galsim.config.Process(config2, logger=logger)
    for n in range(6):
        im1 = galsim.fits.read('output/test_timeout_%d.fits'%n)
        im2 = galsim.fits.read('output/test_timeout_nproc2_%d.fits'%n)
        assert im1 == im2

    # If you use BuildImages, it uses the image nproc and timeout specs, but parallelizes
    # over images rather than stamps.  So check that.
    config2 = galsim.config.CleanConfig(config2)
    images = galsim.config.BuildImages(6, config2, logger=logger)
    for n, im in enumerate(images):
        im1 = galsim.fits.read('output/test_timeout_%d.fits'%n)
        assert im1 == im

    if platform.python_implementation() != 'PyPy':
        # Check that it behaves sensibly if it hits timeout limit.
        # This time, it will continue on after each error, but report the error in the log.
        config2 = galsim.config.CleanConfig(config2)
        config2['image']['timeout'] = 0.001
        with CaptureLog() as cl:
            galsim.config.Process(config2, logger=cl.logger)
        assert 'Multiprocessing timed out waiting for a task to finish.' in cl.output
        # Note: Usually they all fail, and the two lines below are in the logging output, but
        #       it's possible for one of them to finish, so these asserts occasionally fail.
        #assert 'File output/test_timeout_nproc2_1.fits not written! Continuing on...' in cl.output
        #assert 'No files were written.  All were either skipped or had errors.' in cl.output

        # If you want this to abort, use except_abort=True
        config2 = galsim.config.CleanConfig(config2)
        with CaptureLog() as cl:
            with assert_raises(galsim.GalSimError):
                galsim.config.Process(config2, logger=cl.logger, except_abort=True)
        assert 'Multiprocessing timed out waiting for a task to finish.' in cl.output

@timer
def test_direct_extra_output():
    # Test the ability to get extra output directly after calling BuildImage, but
    # not the usual higher level functions (Process or BuildFile).
    # Thanks to Sid Mau for finding this problem in version 2.4.2.
    config = {
        'gal': {
            'type': 'Exponential',
            'half_light_radius': 0.5,
            'signal_to_noise': 100,
        },
        'psf': {
            'type': 'Gaussian',
            'fwhm': 0.7,
        },
        'image': {
            'type': 'Tiled',
            'nx_tiles': 10,
            'ny_tiles': 10,
            'stamp_size': 32,

            'pixel_scale': 0.2,
            'noise': {
                'type': 'Gaussian',
                'sigma': 0.02,
            },
            'random_seed': 1234,
        },
        'output': {
            'dir': 'output',
            'file_name': 'test_direct_extra.fits',
            'weight': {
                'hdu': 1
            },
            'badpix': {
                'hdu': 2
            },
            'psf': {
                'hdu': 3
            },
        },
    }

    # First, get the extras without running the whole file processing.
    image = galsim.config.BuildImage(config)
    weight = galsim.config.GetFinalExtraOutput('weight', config)[0]
    badpix = galsim.config.GetFinalExtraOutput('badpix', config)[0]
    psf = galsim.config.GetFinalExtraOutput('psf', config)[0]

    # These should be the same as what you get from running BuildFile.
    galsim.config.BuildFile(config)
    fname = os.path.join('output', 'test_direct_extra.fits')
    image1 = galsim.fits.read(fname, hdu=0)
    weight1 = galsim.fits.read(fname, hdu=1)
    badpix1 = galsim.fits.read(fname, hdu=2)
    psf1 = galsim.fits.read(fname, hdu=3)

    assert image == image1
    assert weight == weight1
    assert badpix == badpix1
    assert psf == psf1


if __name__ == "__main__":
    runtests(__file__)
