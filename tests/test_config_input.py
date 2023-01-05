# Copyright (c) 2012-2022 by the GalSim developers team on GitHub
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

import galsim
from galsim_test_helpers import timer, CaptureLog, assert_raises

@timer
def test_input_init():
    """Test using the init method for inputs.
    """
    # Most of the tests in this file write to the 'output' directory.
    # Here we write to a different directory and make sure that it properly
    # creates the directory if necessary.
    if os.path.exists('output_fits'):
        shutil.rmtree('output_fits')
    config = {
        'modules': ['config_input_test_modules'],
        'input': {
            'input_size_module': [{'size': 55}, {'size': 45}],
        },
        'image': {
            'type': 'Single',
            'random_seed': 1234,
            'xsize': "$input_size_0",
            'ysize': "$input_size_arr[1]",
        },
        'gal': {
            'type': 'Gaussian',
            'sigma': {'type': 'Random', 'min': 1, 'max': 2},
            'flux': 100,
        },
        'output': {
            'type': 'Fits',
            'file_name': "output_fits/test_fits.fits"
        },
    }

    logger = logging.getLogger('test_fits')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

    first_seed = galsim.BaseDeviate(1234).raw()
    ud = galsim.UniformDeviate(first_seed + 0 + 1)
    sigma = ud() + 1.
    gal = galsim.Gaussian(sigma=sigma, flux=100)
    im1 = gal.drawImage(scale=1, nx=55, ny=45)

    galsim.config.Process(config, logger=logger)
    file_name = 'output_fits/test_fits.fits'
    im2 = galsim.fits.read(file_name)
    np.testing.assert_array_equal(im2.array, im1.array)

@timer
def test_approx_nobjects():
    """Test the getApproxNObjects functionality.
    """
    class BigCatalog(galsim.Catalog):
        def getApproxNObjects(self):
            return 2*self.getNObjects()

    class SmallCatalog(galsim.Catalog):
        def getApproxNObjects(self):
            return self.getNObjects() // 2

    def GenerateFromBigCatalog(config, base, value_type):
        input_cat = galsim.config.GetInputObj('big_catalog', config, base, 'BigCatalog')
        galsim.config.SetDefaultIndex(config, input_cat.getNObjects())
        req = {'col': int, 'index': int}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
        col = kwargs['col']
        index = kwargs['index']
        return input_cat.getFloat(index, col), safe

    def GenerateFromSmallCatalog(config, base, value_type):
        input_cat = galsim.config.GetInputObj('small_catalog', config, base, 'SmallCatalog')
        galsim.config.SetDefaultIndex(config, input_cat.getNObjects())
        req = {'col': int, 'index': int}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
        col = kwargs['col']
        index = kwargs['index']
        return input_cat.getFloat(index, col), safe

    galsim.config.RegisterInputType('big_catalog',
                                    galsim.config.InputLoader(BigCatalog, has_nobj=True))
    galsim.config.RegisterInputType('small_catalog',
                                    galsim.config.InputLoader(SmallCatalog, has_nobj=True))
    galsim.config.RegisterValueType('BigCatalog', GenerateFromBigCatalog, [float],
                                    input_type='big_catalog')
    galsim.config.RegisterValueType('SmallCatalog', GenerateFromSmallCatalog, [float],
                                    input_type='small_catalog')

    # FWIW, this is the config dict from test_variable_cat_size in test_config_image.py.
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
            'size': 33
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
            'file_name' : 'output/test_approx_nobj.fits',
            'nimages' : 2,
        }
    }
    config1 = galsim.config.CopyConfig(config)
    config2 = galsim.config.CopyConfig(config)
    config3 = galsim.config.CopyConfig(config)

    # First regular with normal Catalog.
    galsim.config.Process(config1)
    images1 = galsim.fits.readMulti('output/test_approx_nobj.fits')

    # Now with catalogs whose approximate nobj are too big.
    config2['gal']['half_light_radius']['type'] = 'BigCatalog'
    config2['gal']['shear']['g1']['type'] = 'BigCatalog'
    config2['gal']['shear']['g2']['type'] = 'BigCatalog'
    config2['image']['image_pos']['x']['type'] = 'BigCatalog'
    config2['image']['image_pos']['y']['type'] = 'BigCatalog'
    config2['input']['big_catalog'] = galsim.config.CopyConfig(config['input']['catalog'])
    del config2['input']['catalog']
    galsim.config.Process(config2)
    images2 = galsim.fits.readMulti('output/test_approx_nobj.fits')
    assert images2[0] == images1[0]
    assert images2[1] == images1[1]

    # Lastly, one where the approx nobj is too small.
    config3['gal']['half_light_radius']['type'] = 'SmallCatalog'
    config3['gal']['shear']['g1']['type'] = 'SmallCatalog'
    config3['gal']['shear']['g2']['type'] = 'SmallCatalog'
    config3['image']['image_pos']['x']['type'] = 'SmallCatalog'
    config3['image']['image_pos']['y']['type'] = 'SmallCatalog'
    config3['input']['small_catalog'] = galsim.config.CopyConfig(config['input']['catalog'])
    del config3['input']['catalog']
    with CaptureLog() as cl:
        galsim.config.Process(config3, logger=cl.logger)
    # The real numbers are 3 and 5, but the small version guesses them to be 1 and 2 respectively.
    assert 'Input small_catalog has approximately 1' in cl.output
    assert 'Input small_catalog has approximately 2' in cl.output

    images3 = galsim.fits.readMulti('output/test_approx_nobj.fits')
    assert images3[0] == images1[0]
    assert images3[1] == images1[1]


@timer
def test_atm_input():
    """Test an imsim-like AtmosphericPSF as an input object.
    """
    import multiprocessing

    class AtmPSF:

        def __init__(self):
            ctx = multiprocessing.get_context('fork')
            rng = galsim.BaseDeviate(1234)
            atm = galsim.Atmosphere(mp_context=ctx, r0_500=0.2,
                                    altitude=[0.2, 2.58, 5.16],  # Same as imsim, but just 3 layers
                                    r0_weights=[0.652, 0.172, 0.055],
                                    rng=rng, screen_size=256, screen_scale=1)
            self.aper = galsim.Aperture(diam=2, lam=500, screen_list=atm)
            with galsim.utilities.single_threaded():
                with ctx.Pool(3, initializer=galsim.phase_screens.initWorker,
                              initargs=galsim.phase_screens.initWorkerArgs()) as pool:
                    atm.instantiate(pool=pool, check='FFT')
            self.atm = atm

        def getGSScreenShare(self):
            return self.atm[0].getGSScreenShare()

        def getPSF(self):
            return self.atm.makePSF(500, aper=self.aper)

    def BuildAtmPSF(config, base, ignore, gsparams, logger):
        atm = galsim.config.GetInputObj('atm_psf', config, base, 'AtmPSF')
        return atm.getPSF(), True

    galsim.config.RegisterInputType('atm_psf',
            galsim.config.InputLoader(AtmPSF, use_proxy=False,
                                      worker_init=galsim.phase_screens.initWorker,
                                      worker_initargs=galsim.phase_screens.initWorkerArgs))
    galsim.config.RegisterObjectType('AtmPSF', BuildAtmPSF, input_type='atm_psf')

    config = {
        'input': {
            'atm_psf': {}
        },
        'image': {
            'type': 'Single',
            'random_seed': 1234,
        },
        'gal': {
            'type': 'Gaussian',
            'sigma': {'type': 'Random', 'min': 1, 'max': 2},
            'flux': 100,
        },
        'psf': {
            'type': 'AtmPSF',
        },
        'output': {
            'type': 'MultiFits',
            'nimages': 3,
            'file_name': "output_fits/test_atm.fits",
        },
    }

    # Basically the check here is that multiprocessing works here
    # and that it produces the same images as serial processing

    # First nproc=1.
    config1 = galsim.config.CopyConfig(config)
    galsim.config.Process(config1)

    # Now nproc=3 in output field.
    # This doesn't actually do 3 processes, since there is only 1 output file,
    # but even this fails without the use_proxy=False bit above.
    config2 = galsim.config.CopyConfig(config)
    config2['output']['nproc'] = 3
    config2['output']['file_name'] = "output_fits/test_atm_output_nproc_3.fits"
    galsim.config.Process(config2)

    # Finally nproc=3 in image field.
    # This really does parallelize the run over the 3 images.
    config3 = galsim.config.CopyConfig(config)
    config3['image']['nproc'] = 3
    config3['output']['file_name'] = "output_fits/test_atm_image_nproc_3.fits"
    galsim.config.Process(config3)

    # Check that the results match
    ims1 = galsim.fits.readMulti("output_fits/test_atm.fits")
    ims2 = galsim.fits.readMulti("output_fits/test_atm_output_nproc_3.fits")
    ims3 = galsim.fits.readMulti("output_fits/test_atm_image_nproc_3.fits")

    for im1, im2, im3 in zip(ims1,ims2,ims3):
        assert im1 == im2
        assert im1 == im3

    # Both worker_init and worker_initargs are required when either is provided.
    with assert_raises(galsim.GalSimError):
        galsim.config.InputLoader(AtmPSF, use_proxy=False,
                                  worker_init=galsim.phase_screens.initWorker)
    with assert_raises(galsim.GalSimError):
        galsim.config.InputLoader(AtmPSF, use_proxy=False,
                                  worker_initargs=galsim.phase_screens.initWorkerArgs)

if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
