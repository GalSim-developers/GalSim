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

import numpy as np
import os
import sys

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

def test_scattered():
    """Test aspects of building an Scattered image
    """
    import copy
    import time
    t1 = time.time()

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


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

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
                'file_name' : 'real_galaxy_catalog_example.fits',
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
    galsim.config.Process(config)

    # Repeat with 2 jobs
    config['output']['file_name']['root'] = 'test_two_jobs_'
    galsim.config.Process(config, njobs=2, job=1)
    galsim.config.Process(config, njobs=2, job=2)

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
    test_njobs()

