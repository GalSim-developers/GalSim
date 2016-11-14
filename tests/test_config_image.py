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
    logger.setLevel(logging.DEBUG)

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

    logger = logging.getLogger('test_single')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

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
    config['image']['max_extra_noise'] = 0.1
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
    try:
        np.testing.assert_raises(AttributeError, galsim.config.BuildImage, config)
    except ImportError:
        pass

    # Using this much extra noise with a sky noise variance of 50 cuts the number of photons
    # approximately in half.
    print('N,g without extra_noise: ',gal._calculate_nphotons(0, False, None, None))
    print('N,g with extra_noise: ',gal._calculate_nphotons(0, False, 5, None))
    config['image']['noise'] = { 'type' : 'Gaussian', 'variance' : 50 }
    ud.seed(1234 + 1)
    im4a = gal.drawImage(scale=1, method='phot', max_extra_noise=5, rng=ud, poisson_flux=True)
    im4a.addNoise(galsim.GaussianNoise(sigma=np.sqrt(50), rng=ud))
    im4b = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im4b.array, im4a.array)


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

    try:
        # Make sure they don't match when using the default GSParams
        disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)
        bulge = galsim.Sersic(n=3,half_light_radius=1.3).shear(e1=0.12,e2=-0.08)
        gal4c = disk + bulge
        np.testing.assert_raises(AssertionError,gsobject_compare, gal4a, gal4c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print('The assert_raises tests require nose')


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
    test_single()
    test_positions()
    test_phot()
    test_ring()
    test_scattered()
    test_njobs()
