# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
import shutil
import sys
import logging
import math
import yaml
import json
import re
import glob

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


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
    for k in range(nfiles):
        ud = galsim.UniformDeviate(1234 + k + 1)
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
    config = galsim.config.CopyConfig(config1)
    galsim.config.Process(config)
    for k in range(nfiles):
        file_name = 'output_fits/test_fits_%d.fits'%k
        im2 = galsim.fits.read(file_name)
        np.testing.assert_array_equal(im2.array, im1_list[k].array)

    # For the first file, you don't need the file_num.
    os.remove('output_fits/test_fits_0.fits')
    config = galsim.config.CopyConfig(config1)
    galsim.config.BuildFile(config)
    im2 = galsim.fits.read('output_fits/test_fits_0.fits')
    np.testing.assert_array_equal(im2.array, im1_list[0].array)

    # nproc < 0 should automatically determine nproc from ncpu
    config = galsim.config.CopyConfig(config1)
    config['output']['nproc'] = -1
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    assert 'ncpu = ' in cl.output

    # nproc > njobs should drop back to nproc = njobs
    config = galsim.config.CopyConfig(config1)
    config['output']['nproc'] = 10
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    assert 'There are only 6 jobs to do.  Reducing nproc to 6' in cl.output

    # Check that profile outputs something appropriate for multithreading.
    # (The single-thread profiling is handled by the galsim executable, which we don't
    # bother testing here.)
    config = galsim.config.CopyConfig(config1)
    config['profile'] = True
    config['output']['nproc'] = -1
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    #print(cl.output)
    # Unfortunately, the LoggerProxy doesn't really work right with the string logger used
    # by CaptureLog.  I tried for a while to figure out how to get it to capture the proxied
    # logs and couldn't get it working.  So this just checks for an info log before the
    # multithreading starts.  But with a regular logger, there really is profiling output.
    assert "Starting separate profiling for each of the" in cl.output

    # If there is no output field, the default behavior is to write to root.fits.
    os.remove('output_fits/test_fits_0.fits')
    config = galsim.config.CopyConfig(config1)
    del config['output']
    config['root'] = 'output_fits/test_fits_0'
    galsim.config.BuildFile(config)
    im2 = galsim.fits.read('output_fits/test_fits_0.fits')
    np.testing.assert_array_equal(im2.array, im1_list[0].array)


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
    for k in range(nimages):
        ud = galsim.UniformDeviate(1234 + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        im1 = gal.drawImage(scale=1)
        im1_list.append(im1)
    print('multifit image shapes = ',[im.array.shape for im in im1_list])

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

    try:
        # Check error message for missing nimages
        del config['output']['nimages']
        np.testing.assert_raises(AttributeError, galsim.config.BuildFile,config)
        # Also if there is an input field that doesn't have nobj capability
        config['input'] = { 'dict' : { 'dir' : 'config_input', 'file_name' : 'dict.p' } }
        np.testing.assert_raises(AttributeError, galsim.config.BuildFile,config)
    except ImportError:
        pass
    # However, an input field that does have nobj will return something for nobjects.
    # This catalog has 3 rows, so equivalent to nobjects = 3
    del config['input_objs']
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
    for k in range(nimages):
        ud = galsim.UniformDeviate(1234 + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        if b is None:
            im1 = gal.drawImage(scale=1)
            b = im1.bounds
        else:
            im1 = gal.drawImage(bounds=b, scale=1)
        im1_list.append(im1)
    print('datacube image shapes = ',[im.array.shape for im in im1_list])

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

    try:
        # Check error message for missing nimages
        del config['output']['nimages']
        np.testing.assert_raises(AttributeError, galsim.config.BuildFile,config)
        # Also if there is an input field that doesn't have nobj capability
        config['input'] = { 'dict' : { 'dir' : 'config_input', 'file_name' : 'dict.p' } }
        np.testing.assert_raises(AttributeError, galsim.config.BuildFile,config)
    except ImportError:
        pass
    # However, an input field that does have nobj will return something for nobjects.
    # This catalog has 3 rows, so equivalent to nobjects = 3
    del config['input_objs']
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
    try:
        np.testing.assert_raises(AttributeError, galsim.config.BuildFile,config)
    except ImportError:
        pass

    # But if both weight and badpix are files, then it should work.
    config['output']['weight'] = { 'file_name' : 'output/test_datacube_wt.fits' }
    galsim.config.BuildFile(config)
    im5_list = galsim.fits.readCube('output/test_datacube.fits')
    assert len(im5_list) == 3
    for k in range(3):
        rng = galsim.UniformDeviate(1234 + k + 1)
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
    for k in range(nfiles):
        file_name = 'output/test_skip_%d.fits'%k
        if os.path.exists(file_name):
            os.remove(file_name)
        ud_file = galsim.UniformDeviate(1234 + k)
        if ud_file() < 0.4:
            print('skip k = ',k)
            skip_list.append(True)
        else:
            skip_list.append(False)
        ud = galsim.UniformDeviate(1234 + k + 1)
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
    assert "Skipping file 2 = output/test_skip_2.fits because output.noclobber" in cl.output
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
    for k in range(nfiles):
        ud = galsim.UniformDeviate(1234 + k + 1)
        sigma = ud() + 1.
        gal = galsim.Gaussian(sigma=sigma, flux=100)
        im = gal.drawImage(scale=0.4)
        im_wt = galsim.fits.read('output/test_wt_%d.fits'%k)
        np.testing.assert_almost_equal(im_wt.array, 1./(0.7 + k + im.array))

    # If the image is a Scattered type, then the weight adn badpix images are built by a
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
    for k in range(nfiles):
        ud = galsim.UniformDeviate(1234 + k + 1)
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

    for k in range(nfiles):
        ud = galsim.UniformDeviate(1234 + k + 1)
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
        ud = galsim.UniformDeviate(1234 + k + 1)
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
def test_extra_truth():
    """Test the extra truth field
    """
    nobjects = 6
    config = {
        'image' : {
            'type' : 'Tiled',
            'nx_tiles' : nobjects,
            'ny_tiles' : 1,
            'stamp_xsize' : 32,
            'stamp_ysize' : 32,
            'random_seed' : 1234,
        },
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : { 'type': 'Random', 'min': 1, 'max': 2 },
            'flux' : '$100 * obj_num',
        },
        'output' : {
            'type' : 'Fits',
            'file_name' : 'output/test_truth.fits',
            'truth' : {
                'hdu' : 1,
                'columns' : {
                    'object_id' : 'obj_num',
                    'flux' : 'gal.flux',
                    # Check several different ways to do calculations
                    'sigma' : '@gal.sigma',  # The @ is not required, but allowed.
                    'hlr' : '$(@gal.sigma) * np.sqrt(2.*math.log(2))',
                    'fwhm' : '$(@gal).fwhm',
                    'pos' : 'image_pos'
                }
            }
        }
    }

    galsim.config.Process(config)

    sigma_list = []
    for k in range(nobjects):
        ud = galsim.UniformDeviate(1234 + k + 1)
        sigma = ud() + 1.
        flux = k * 100
        gal = galsim.Gaussian(sigma=sigma, flux=flux)
        sigma_list.append(sigma)
    sigma = np.array(sigma_list)

    file_name = 'output/test_truth.fits'
    cat = galsim.Catalog(file_name, hdu=1)
    obj_num = np.array(range(nobjects))
    np.testing.assert_almost_equal(cat.data['object_id'], obj_num)
    np.testing.assert_almost_equal(cat.data['flux'], 100.*obj_num)
    np.testing.assert_almost_equal(cat.data['sigma'], sigma)
    np.testing.assert_almost_equal(cat.data['hlr'], sigma * galsim.Gaussian._hlr_factor)
    np.testing.assert_almost_equal(cat.data['fwhm'], sigma * galsim.Gaussian._fwhm_factor)
    np.testing.assert_almost_equal(cat.data['pos.x'], obj_num * 32 + 16.5)
    np.testing.assert_almost_equal(cat.data['pos.y'], 16.5)


@timer
def test_retry_io():
    """Test the retry_io option
    """
    # Make a class that mimics writeMulti, except that it fails about 1/3 of the time.
    class FlakyWriter(object):
        def __init__(self, rng):
            self.ud = galsim.UniformDeviate(rng)
        def writeFile(self, *args, **kwargs):
            p = self.ud()
            if p < 0.33:
                raise IOError("p = %f"%p)
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
    assert "File output/test_flaky_fits_0.fits: Caught IOError" in cl.output
    assert "This is try 2/6, so sleep for 2 sec and try again." in cl.output
    assert "file 0: Wrote FlakyFits to file 'output/test_flaky_fits_0.fits'" in cl.output
    assert "File output/test_flaky_wt_0.fits: Caught IOError: " in cl.output
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
        np.testing.assert_array_equal(im1.array, im2.array)
        wt1 = galsim.fits.read('output/test_flaky_wt_%d.fits'%k)
        wt2 = galsim.fits.read('output/test_nonflaky_wt_%d.fits'%k)
        np.testing.assert_array_equal(wt1.array, wt2.array)

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
    assert "File 4 = output/test_flaky_fits_4.fits" in cl.output
    assert "Exception caught for file 5 = output/test_flaky_fits_5.fits" in cl.output
    assert "File output/test_flaky_fits_5.fits not written! Continuing on..." in cl.output

    # Also works in nproc > 1 mode
    config['output']['nproc'] = 2
    galsim.config.RemoveCurrent(config)
    with CaptureLog() as cl:
        galsim.config.Process(config, logger=cl.logger)
    #print(cl.output)
    assert re.search("Process-.: Exception caught for file 0 = output/test_flaky_fits_0.fits",
                     cl.output)
    assert "File output/test_flaky_fits_0.fits not written! Continuing on..." in cl.output
    assert re.search("Process-.: File 1 = output/test_flaky_fits_1.fits", cl.output)
    assert re.search("Process-.: File 2 = output/test_flaky_fits_2.fits", cl.output)
    assert re.search("Process-.: File 3 = output/test_flaky_fits_3.fits", cl.output)
    assert re.search("Process-.: File 4 = output/test_flaky_fits_4.fits", cl.output)
    assert re.search("Process-.: Exception caught for file 5 = output/test_flaky_fits_5.fits",
                     cl.output)
    assert "File output/test_flaky_fits_5.fits not written! Continuing on..." in cl.output

    # But with except_abort = True, it will stop after the first failure
    del config['output']['nproc']  # Otherwise which file fails in non-deterministic.
    with CaptureLog() as cl:
        try:
            galsim.config.Process(config, logger=cl.logger, except_abort=True)
        except IOError as e:
            assert str(e) == "p = 0.126989"
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
        'output' : { 'type' : 'Fits', 'file_name' : "test.fits" },
        'input' : { 'dict' : { 'dir' : 'config_input', 'file_name' : 'dict.p' } },
        'eval_variables' : { 'fpixel_scale' : 0.3 }
    }

    # Test yaml
    yaml_file_name = "output/test_config.yaml"
    with open(yaml_file_name, 'w') as fout:
        yaml.dump(config, fout, default_flow_style=True)

    config1 = galsim.config.ReadConfig(yaml_file_name)[0]
    assert config == dict(config1)
    config2 = galsim.config.ReadConfig(yaml_file_name, file_type='yaml')[0]
    assert config == dict(config2)
    config3 = galsim.config.ReadYaml(yaml_file_name)[0]
    assert config == dict(config2)

    # Test json
    json_file_name = "output/test_config.yaml"
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
    config4['input_manager'] = 'an input manager'
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

    try:
        np.testing.assert_raises(ValueError,galsim.config.GetFromConfig,config,'psf.items.lam')
        np.testing.assert_raises(ValueError,galsim.config.GetFromConfig,config,'psf.items.4')
        np.testing.assert_raises(ValueError,galsim.config.GetFromConfig,config,'psf.itms.1.lam')
        np.testing.assert_raises(ValueError,galsim.config.SetInConfig,config,'psf.items.lam',700)
        np.testing.assert_raises(ValueError,galsim.config.SetInConfig,config,'psf.items.4',700)
        np.testing.assert_raises(ValueError,galsim.config.SetInConfig,config,'psf.itms.1.lam',700)
    except ImportError:
        pass

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


if __name__ == "__main__":
    test_fits()
    test_multifits()
    test_datacube()
    test_skip()
    test_extra_wt()
    test_extra_psf()
    test_extra_truth()
    test_retry_io()
    test_config()
    test_no_output()
