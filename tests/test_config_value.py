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
import astropy.units as u
import math

import galsim
from galsim_test_helpers import *


@timer
def test_float_value():
    """Test various ways to generate a float value
    """
    halo_mass = 1.e14
    halo_conc = 4
    halo_z = 0.3
    gal_z = 1.3
    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ],
                    'dict' : [
                        { 'dir' : 'config_input', 'file_name' : 'dict.p' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.json' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.yaml' } ],
                    'nfw_halo' : { 'mass' : halo_mass, 'conc' : halo_conc, 'redshift' : halo_z },
                    'power_spectrum' : { 'e_power_function' : 'np.exp(-k**0.2)',
                                         'grid_spacing' : 10, 'interpolant' : 'linear' },
                    'fits_header' : { 'dir' : 'fits_files', 'file_name' : 'tpv.fits' },
                  },

        'val1' : 9.9,
        'val2' : int(400),
        'val3' : None,
        'str1' : '8.73',
        'str2' : '2.33e-9',
        'str3' : '6.e-9',
        'cat1' : { 'type' : 'Catalog' , 'col' : 0 },
        'cat2' : { 'type' : 'Catalog' , 'col' : 1 },
        'cat3' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'float1' },
        'cat4' : { 'type' : 'Catalog_float' , 'num' : 1, 'col' : 'float2' },
        'ran1' : { 'type' : 'Random', 'min' : 0.5, 'max' : 3 },
        'ran2' : { 'type' : 'Random_float', 'min' : -5, 'max' : 0 },
        'gauss1' : { 'type' : 'RandomGaussian', 'sigma' : 1 },
        'gauss1b' : { 'type' : 'RandomGaussian', 'sigma' : 1 },
        'gauss2' : { 'type' : 'RandomGaussian', 'sigma' : 3, 'mean' : 4 },
        'gauss3' : { 'type' : 'RandomGaussian', 'sigma' : 1.5, 'min' : -2, 'max' : 2 },
        'gauss4' : { 'type' : 'RandomGaussian', 'sigma' : 0.5, 'min' : 0, 'max' : 0.8 },
        'gauss5' : { 'type' : 'RandomGaussian',
                     'sigma' : 0.3, 'mean' : 0.5, 'min' : 0, 'max' : 0.5 },
        'gauss6' : { 'type' : 'RandomGaussian',
                     'sigma' : 0.8, 'mean' : 0.3, 'min' : 2. },
        'gauss7' : { 'type' : 'RandomGaussian',
                     'sigma' : 1.3, 'mean' : 0.3, 'min' : -2., 'max' : 0. },
        'dist1' : { 'type' : 'RandomDistribution', 'function' : 'config_input/distribution.txt',
                    'interpolant' : 'linear' },
        'dist2' : { 'type' : 'RandomDistribution',
                    'x' : [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ],
                    'f' : [ 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1 ],
                    'interpolant' : 'linear' },
        'dist3' : { 'type' : 'RandomDistribution', 'function' : 'x*x',
                    'x_min' : 0., 'x_max' : 2.0 },
        'dev1' : { 'type' : 'RandomPoisson', 'mean' : 137 },
        'dev2' : { 'type' : 'RandomBinomial', 'N' : 17 },
        'dev3' : { 'type' : 'RandomBinomial', 'N' : 17, 'p' : 0.2 },
        'dev4' : { 'type' : 'RandomWeibull', 'a' : 1.7, 'b' : 4.3 },
        'dev5' : { 'type' : 'RandomGamma', 'k' : 1, 'theta' : 4 },
        'dev6' : { 'type' : 'RandomGamma', 'k' : 1.9, 'theta' : 4.1 },
        'dev7' : { 'type' : 'RandomChi2', 'n' : 17},
        'seq1' : { 'type' : 'Sequence' },
        'seq2' : { 'type' : 'Sequence', 'step' : 0.1 },
        'seq3' : { 'type' : 'Sequence', 'first' : 1.5, 'step' : 0.5 },
        'seq4' : { 'type' : 'Sequence', 'first' : 10, 'step' : -2 },
        'seq5' : { 'type' : 'Sequence_float', 'first' : 1, 'last' : 2.1, 'repeat' : 2 },
        'list1' : { 'type' : 'List', 'items' : [ 73, 8.9, 3.14 ] },
        'list2' : { 'type' : 'List_float',
                    'items' : [ 0.6, 1.8, 2.1, 3.7, 4.3, 5.5, 6.1, 7.0, 8.6, 9.3, 10.8, 11.2 ],
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -3 } },
        'list3' : { 'type' : 'List',
                    'items' : '$[i for i in range(7) if i not in [3, 5]]' },
        'dict1' : { 'type' : 'Dict', 'key' : 'f' },
        'dict2' : { 'type' : 'Dict', 'num' : 1, 'key' : 'f' },
        'dict3' : { 'type' : 'Dict', 'num' : 2, 'key' : 'f' },
        'dict4' : { 'type' : 'Dict_float', 'num' : 2, 'key' : 'noise.models.1.gain' },
        'fits1' : { 'type' : 'FitsHeader', 'key' : 'AIRMASS' },
        'fits2' : { 'type' : 'FitsHeader', 'key' : 'MJD-OBS' },
        'sum1' : { 'type' : 'Sum', 'items' : [ 72, '2.33', { 'type' : 'Dict', 'key' : 'f' } ] },
        'sum2' : { 'type' : 'Sum',
                   'items' : '$[i for i in range(7) if i not in [3, 5]]' },
        'nfw' : { 'type' : 'NFWHaloMagnification' },
        'ps' : { 'type' : 'PowerSpectrumMagnification' },
        'bad1' : { 'value' : 34. },
        'bad2' : { 'type' : 'RandomGaussian', 'sig' : 1 },
        'bad3' : { 'type' : 'RandomGaussian', 'sigma' : 'not a number' },
        'bad4' : { 'type' : 'Invalid', 'sig' : 1 },
        'bad5' : { 'type' : 'Sequence', 'first' : 1, 'last' : 2.1, 'repeat' : -2 },
        'bad6' : { 'type' : 'Sequence', 'first' : 1, 'last' : 2.1, 'nitems' : 12 },
        'bad7' : { 'type' : 'RandomDistribution',
                    'x' : [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ],
                    'interpolant' : 'linear' },
        'bad8' : { 'type' : 'RandomDistribution', 'function' : 'x*x',
                    'x' : [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ],
                    'f' : [ 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1 ],
                    'interpolant' : 'linear' },
        'bad9' : { 'type' : 'RandomDistribution', 'interpolant' : 'linear' },
        'bad10' : { 'type' : 'RandomDistribution', 'function' : 'x*x', 'x_log' : True },
        'bad11' : { 'type' : 'RandomDistribution', 'function' : 'x*x', 'f_log' : True },

        # Some items that would normally be set by the config processing
        'image_xsize' : 2000,
        'image_ysize' : 2000,
        'wcs' : galsim.PixelScale(0.1),
        'image_center' : galsim.PositionD(0,0),
    }

    galsim.config.ProcessInput(config)

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, float)[0]
    np.testing.assert_almost_equal(val1, 9.9)

    val2 = galsim.config.ParseValue(config,'val2',config, float)[0]
    np.testing.assert_almost_equal(val2, 400)

    # Even though None is not a float, it is valid to set any parameter to None
    val3 = galsim.config.ParseValue(config,'val3',config, float)[0]
    np.testing.assert_equal(val3, None)

    # You can also give None as the value type, which just returns whatever is in the dict.
    val1b  = galsim.config.ParseValue(config,'val1',config, None)[0]
    val2b  = galsim.config.ParseValue(config,'val2',config, None)[0]
    np.testing.assert_almost_equal(val1b, 9.9)
    np.testing.assert_almost_equal(val2b, 400)

    # Test conversions from strings
    str1 = galsim.config.ParseValue(config,'str1',config, float)[0]
    np.testing.assert_almost_equal(str1, 8.73)

    str2 = galsim.config.ParseValue(config,'str2',config, float)[0]
    np.testing.assert_almost_equal(str2, 2.33e-9)

    str3 = galsim.config.ParseValue(config,'str3',config, float)[0]
    np.testing.assert_almost_equal(str3, 6.0e-9)

    # Test values read from a Catalog
    cat1 = []
    cat2 = []
    cat3 = []
    cat4 = []
    config['index_key'] = 'file_num'
    for k in range(5):
        config['file_num'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, float)[0])
        cat2.append(galsim.config.ParseValue(config,'cat2',config, float)[0])
        cat3.append(galsim.config.ParseValue(config,'cat3',config, float)[0])
        cat4.append(galsim.config.ParseValue(config,'cat4',config, float)[0])

    np.testing.assert_array_almost_equal(cat1, [ 1.234, 2.345, 3.456, 1.234, 2.345 ])
    np.testing.assert_array_almost_equal(cat2, [ 4.131, -900, 8000, 4.131, -900 ])
    np.testing.assert_array_almost_equal(cat3, [ 1.234, 2.345, 3.456, 1.234, 2.345 ])
    np.testing.assert_array_almost_equal(cat4, [ 4.131, -900, 8000, 4.131, -900 ])

    # Test values generated from a uniform deviate
    del config['index_key']
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        config['obj_num'] = k  # The Random type doesn't use obj_num, but this keeps it
                               # from thinking "current" value is still current.
        ran1 = galsim.config.ParseValue(config,'ran1',config, float)[0]
        np.testing.assert_almost_equal(ran1, rng() * 2.5 + 0.5)

        ran2 = galsim.config.ParseValue(config,'ran2',config, float)[0]
        np.testing.assert_almost_equal(ran2, rng() * 5 - 5)

    # Test values generated from a Gaussian deviate
    for k in range(6):
        config['obj_num'] = k
        gauss1 = galsim.config.ParseValue(config,'gauss1',config, float)[0]
        gd = galsim.GaussianDeviate(rng,mean=0,sigma=1)
        np.testing.assert_almost_equal(gauss1, gd())

        # Repeating with the same sigma will use the same gd object
        gauss1b = galsim.config.ParseValue(config,'gauss1b',config, float)[0]
        np.testing.assert_almost_equal(gauss1b, gd())

        gauss2 = galsim.config.ParseValue(config,'gauss2',config, float)[0]
        gd = galsim.GaussianDeviate(rng,mean=4,sigma=3)
        np.testing.assert_almost_equal(gauss2, gd())

        gauss3 = galsim.config.ParseValue(config,'gauss3',config, float)[0]
        gd = galsim.GaussianDeviate(rng,mean=0,sigma=1.5)
        gd_val = gd()
        while math.fabs(gd_val) > 2:
            gd_val = gd()
        np.testing.assert_almost_equal(gauss3, gd_val)

        gauss4 = galsim.config.ParseValue(config,'gauss4',config, float)[0]
        gd = galsim.GaussianDeviate(rng,mean=0,sigma=0.5)
        gd_val = math.fabs(gd())
        while gd_val > 0.8:
            gd_val = math.fabs(gd())
        np.testing.assert_almost_equal(gauss4, gd_val)

        gauss5 = galsim.config.ParseValue(config,'gauss5',config, float)[0]
        gd = galsim.GaussianDeviate(rng,mean=0.5,sigma=0.3)
        gd_val = gd()
        if gd_val > 0.5:
            gd_val = 1-gd_val
        while gd_val < 0:
            gd_val = gd()
            if gd_val > 0.5:
                gd_val = 1-gd_val
        np.testing.assert_almost_equal(gauss5, gd_val)

        gauss6 = galsim.config.ParseValue(config,'gauss6',config, float)[0]
        gd = galsim.GaussianDeviate(rng,mean=0.,sigma=0.8)
        gd_val = abs(gd())
        while gd_val < 1.7:
            gd_val = abs(gd())
        gd_val += 0.3
        np.testing.assert_almost_equal(gauss6, gd_val)

        gauss7 = galsim.config.ParseValue(config,'gauss7',config, float)[0]
        gd = galsim.GaussianDeviate(rng,mean=0.,sigma=1.3)
        gd_val = abs(gd())
        while gd_val < 0.3 or gd_val > 2.3:
            gd_val = abs(gd())
        gd_val = -gd_val + 0.3
        np.testing.assert_almost_equal(gauss7, gd_val)

    # Test values generated from a distribution in a file
    dd=galsim.DistDeviate(rng,function='config_input/distribution.txt',interpolant='linear')
    for k in range(6):
        config['obj_num'] = k
        dist1 = galsim.config.ParseValue(config,'dist1',config, float)[0]
        np.testing.assert_almost_equal(dist1, dd())
    dd=galsim.DistDeviate(rng,function='config_input/distribution2.txt',interpolant='linear')
    for k in range(6):
        config['obj_num'] = k
        dist2 = galsim.config.ParseValue(config,'dist2',config, float)[0]
        np.testing.assert_almost_equal(dist2, dd())
    dd=galsim.DistDeviate(rng,function=lambda x: x*x,x_min=0.,x_max=2.)
    for k in range(6):
        config['obj_num'] = k
        dist3 = galsim.config.ParseValue(config,'dist3',config, float)[0]
        np.testing.assert_almost_equal(dist3, dd())

    # Test values generated from various other deviates
    for k in range(6):
        config['obj_num'] = k
        dev = galsim.PoissonDeviate(rng, mean=137)
        dev1 = galsim.config.ParseValue(config,'dev1',config, float)[0]
        np.testing.assert_almost_equal(dev1, dev())

        dev = galsim.BinomialDeviate(rng, N=17)
        dev2 = galsim.config.ParseValue(config,'dev2',config, float)[0]
        np.testing.assert_almost_equal(dev2, dev())

        dev = galsim.BinomialDeviate(rng, N=17, p=0.2)
        dev3 = galsim.config.ParseValue(config,'dev3',config, float)[0]
        np.testing.assert_almost_equal(dev3, dev())

        dev = galsim.WeibullDeviate(rng, a=1.7, b=4.3)
        dev4 = galsim.config.ParseValue(config,'dev4',config, float)[0]
        np.testing.assert_almost_equal(dev4, dev())

        dev = galsim.GammaDeviate(rng, k=1, theta=4)
        dev5 = galsim.config.ParseValue(config,'dev5',config, float)[0]
        np.testing.assert_almost_equal(dev5, dev())

        dev = galsim.GammaDeviate(rng, k=1.9, theta=4.1)
        dev6 = galsim.config.ParseValue(config,'dev6',config, float)[0]
        np.testing.assert_almost_equal(dev6, dev())

        dev = galsim.Chi2Deviate(rng, n=17)
        dev7 = galsim.config.ParseValue(config,'dev7',config, float)[0]
        np.testing.assert_almost_equal(dev7, dev())

    # Test values generated from a Sequence
    seq1 = []
    seq2 = []
    seq3 = []
    seq4 = []
    seq5 = []
    config['index_key'] = 'file_num'
    for k in range(6):
        config['file_num'] = k
        seq1.append(galsim.config.ParseValue(config,'seq1',config, float)[0])
    config['index_key'] = 'image_num'
    for k in range(6):
        config['image_num'] = k
        seq2.append(galsim.config.ParseValue(config,'seq2',config, float)[0])
    config['index_key'] = 'obj_num'
    for k in range(6):
        config['obj_num'] = k
        seq3.append(galsim.config.ParseValue(config,'seq3',config, float)[0])
    config['index_key'] = 'obj_num_in_file'
    config['start_obj_num'] = 10
    for k in range(6):
        config['obj_num'] = k+10
        seq4.append(galsim.config.ParseValue(config,'seq4',config, float)[0])
        seq5.append(galsim.config.ParseValue(config,'seq5',config, float)[0])
    del config['start_obj_num']

    np.testing.assert_array_almost_equal(seq1, [ 0, 1, 2, 3, 4, 5 ])
    np.testing.assert_array_almost_equal(seq2, [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ])
    np.testing.assert_array_almost_equal(seq3, [ 1.5, 2, 2.5, 3, 3.5, 4 ])
    np.testing.assert_array_almost_equal(seq4, [ 10, 8, 6, 4, 2, 0 ])
    np.testing.assert_array_almost_equal(seq5, [ 1, 1, 2, 2, 1, 1 ])

    # Test values taken from a List
    list1 = []
    list2 = []
    list3 = []
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, float)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, float)[0])
        list3.append(galsim.config.ParseValue(config,'list3',config, float)[0])

    np.testing.assert_array_almost_equal(list1, [ 73, 8.9, 3.14, 73, 8.9 ])
    np.testing.assert_array_almost_equal(list2, [ 10.8, 7.0, 4.3, 1.8, 10.8 ])
    np.testing.assert_array_almost_equal(list3, [ 0, 1, 2, 4, 6 ])

    # Test values read from a Dict
    dict = []
    dict.append(galsim.config.ParseValue(config,'dict1',config, float)[0])
    dict.append(galsim.config.ParseValue(config,'dict2',config, float)[0])
    dict.append(galsim.config.ParseValue(config,'dict3',config, float)[0])
    dict.append(galsim.config.ParseValue(config,'dict4',config, float)[0])
    np.testing.assert_array_almost_equal(dict, [ 23.17, -17.23, 0.1, 1.9 ])

    assert galsim.config.ParseValue(config,'fits1',config, float)[0] == 1.185
    assert galsim.config.ParseValue(config,'fits2',config, float)[0] == 54384.18627436

    sum1 = galsim.config.ParseValue(config,'sum1',config, float)[0]
    sum2 = galsim.config.ParseValue(config,'sum2',config, float)[0]
    np.testing.assert_almost_equal(sum1, 72 + 2.33 + 23.17)
    np.testing.assert_almost_equal(sum2, sum([ 0, 1, 2, 4, 6]))

    # Test NFWHaloMagnification
    galsim.config.SetupInputsForImage(config, None)
    # Raise an error because no uv_pos
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'nfw',config, float)
    config['uv_pos'] = galsim.PositionD(6,8)
    # Still raise an error because no redshift
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'nfw',config, float)
    # With this, it should work.
    config['gal'] = { 'redshift' : gal_z }
    nfw_halo = galsim.NFWHalo(mass=halo_mass, conc=halo_conc, redshift=halo_z)
    print("weak lensing mag = ",nfw_halo.getMagnification((6,8), gal_z))
    nfw1 = galsim.config.ParseValue(config,'nfw',config, float)[0]
    np.testing.assert_almost_equal(nfw1, nfw_halo.getMagnification((6,8), gal_z))

    # Too large magnification should max out at 25
    galsim.config.RemoveCurrent(config)
    config['uv_pos'] = galsim.PositionD(0.1,0.3)
    print("strong lensing mag = ",nfw_halo.getMagnification((0.1,0.3), gal_z))
    galsim.config.RemoveCurrent(config)
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, cl.logger)
        nfw2 = galsim.config.ParseValue(config, 'nfw', config, float)[0]
    print(cl.output)
    assert "Warning: NFWHalo mu = 249.374050 means strong lensing." in cl.output
    np.testing.assert_almost_equal(nfw2, 25.)

    # Or set a different maximum
    galsim.config.RemoveCurrent(config)
    config['nfw']['max_mu'] = 3000.
    del config['nfw']['_get']
    config['uv_pos'] = galsim.PositionD(0.1,0.3)
    nfw3 = galsim.config.ParseValue(config,'nfw',config, float)[0]
    np.testing.assert_almost_equal(nfw3, nfw_halo.getMagnification((0.1,0.3), gal_z))

    # Also, if it goes negative, it should report the max_mu value.
    galsim.config.RemoveCurrent(config)
    config['uv_pos'] = galsim.PositionD(0.1,0.2)
    print("very strong lensing mag = ",nfw_halo.getMagnification((0.1,0.2), gal_z))
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, cl.logger)
        nfw4 = galsim.config.ParseValue(config, 'nfw', config, float)[0]
    print(cl.output)
    assert "Warning: NFWHalo mu = -163.631846 means strong lensing." in cl.output
    np.testing.assert_almost_equal(nfw4, 3000.)

    # Negative max_mu is invalid.
    galsim.config.RemoveCurrent(config)
    config['nfw']['max_mu'] = -3.
    del config['nfw']['_get']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'nfw',config, float)

    # Test PowerSpectrumMagnification
    ps = galsim.PowerSpectrum(e_power_function='np.exp(-k**0.2)')
    galsim.config.RemoveCurrent(config)
    rng = galsim.BaseDeviate(31415)  # reset this so changes to tests above don't mess this up.
    config['rng'] = rng.duplicate()
    ps.buildGrid(grid_spacing=10, ngrid=21, interpolant='linear', rng=rng)
    print("ps mag = ",ps.getMagnification((0.1,0.2)))
    galsim.config.SetupInputsForImage(config, None)
    ps1 = galsim.config.ParseValue(config,'ps',config, float)[0]
    np.testing.assert_almost_equal(ps1, ps.getMagnification((0.1,0.2)))

    # Beef up the amplitude to get strong lensing.
    ps = galsim.PowerSpectrum(e_power_function='2000 * np.exp(-k**0.2)')
    ps.buildGrid(grid_spacing=10, ngrid=21, interpolant='linear', rng=rng)

    print("strong lensing mag = ",ps.getMagnification((0.1,0.2)))
    config = galsim.config.CleanConfig(config)
    config['input']['power_spectrum']['e_power_function'] = '2000 * np.exp(-k**0.2)'
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, logger=cl.logger)
        ps2a = galsim.config.ParseValue(config,'ps',config, float)[0]
    print(cl.output)
    assert 'PowerSpectrum mu = -4.335137 means strong lensing. Using mu=25.000000' in cl.output
    np.testing.assert_almost_equal(ps2a, 25.)

    # Need a different point that happens to have strong lensing, since the PS realization changed.
    ps.buildGrid(grid_spacing=10, ngrid=21, interpolant='linear', rng=rng)
    config['uv_pos'] = galsim.PositionD(55,-25)
    galsim.config.RemoveCurrent(config)
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, logger=cl.logger)
        ps2b = galsim.config.ParseValue(config, 'ps', config, float)[0]
    print(cl.output)
    assert "PowerSpectrum mu = 26.746296 means strong lensing. Using mu=25.000000" in cl.output
    np.testing.assert_almost_equal(ps2b, 25.)

    # Or set a different maximum
    galsim.config.RemoveCurrent(config)
    config['ps']['max_mu'] = 30.
    del config['ps']['_get']
    ps3 = galsim.config.ParseValue(config,'ps',config, float)[0]
    np.testing.assert_almost_equal(ps3, 26.7462955457826)

    # Negative max_mu is invalid.
    galsim.config.RemoveCurrent(config)
    config['ps']['max_mu'] = -3.
    del config['ps']['_get']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'ps',config, float)
    config['ps']['max_mu'] = 25.

    # Out of bounds results in shear = 0, and a warning.
    galsim.config.RemoveCurrent(config)
    config['uv_pos'] = galsim.PositionD(1000,2000)
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, cl.logger)
        ps2c = galsim.config.ParseValue(config, 'ps', config, float)[0]
    print(cl.output)
    assert ("Extrapolating beyond input range. galsim.PositionD(x=1000.0, y=2000.0) not in "
            "galsim.BoundsD") in cl.output
    np.testing.assert_almost_equal(ps2c, 1.)

    # Error if no uv_pos
    del config['uv_pos']
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'ps', config, float)

    # Should raise a GalSimConfigError if there is no type in the dict
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad1', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad2', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad3', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad4', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad5', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad6', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad7', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad8', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad9', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad10', config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'bad11', config, float)

    # Error if given the wrong type.  Should be float, not np.float16.
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'gauss1b',config, np.float16)
    # Different path to (different) error if already processed into a _gen_fn.
    galsim.config.ParseValue(config,'gauss1',config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'gauss1',config, np.float16)


@timer
def test_int_value():
    """Test various ways to generate an int value
    """
    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ],
                    'dict' : [
                        { 'dir' : 'config_input', 'file_name' : 'dict.p' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.json' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.yaml' } ],
                    'fits_header' : { 'dir' : 'fits_files', 'file_name' : 'tpv.fits' },
                  },

        'val1' : 9,
        'val2' : float(8.7),  # Reading as int will drop the fraction.
        'val3' : -400.8,      # Not floor - negatives will round up.
        'val4' : None,
        'str1' : '8',
        'str2' : '-2',
        'cat1' : { 'type' : 'Catalog' , 'col' : 2 },
        'cat2' : { 'type' : 'Catalog' , 'col' : 3 },
        'cat3' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'int1' },
        'cat4' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'int2' },
        'ran1' : { 'type' : 'Random', 'min' : 0, 'max' : 3 },
        'ran2' : { 'type' : 'Random_int', 'min' : -5, 'max' : 10 },
        'dev1' : { 'type' : 'RandomPoisson', 'mean' : 137 },
        'dev2' : { 'type' : 'RandomBinomial', 'N' : 17 },
        'dev3' : { 'type' : 'RandomBinomial', 'N' : 17, 'p' : 0.2 },
        'seq1' : { 'type' : 'Sequence' },
        'seq2' : { 'type' : 'Sequence', 'step' : 3 },
        'seq3' : { 'type' : 'Sequence', 'first' : 1, 'step' : 5 },
        'seq4' : { 'type' : 'Sequence', 'first' : 10, 'step' : -2 },
        'seq5' : { 'type' : 'Sequence_int', 'first' : 1, 'last' : 2, 'repeat' : 2 },
        'seq_file' : { 'type' : 'Sequence', 'index_key' : 'file_num' },
        'seq_image' : { 'type' : 'Sequence', 'index_key' : 'image_num' },
        'seq_obj' : { 'type' : 'Sequence', 'index_key' : 'obj_num' },
        'seq_obj2' : { 'type' : 'Sequence', 'index_key' : 'obj_num_in_file' },
        'list1' : { 'type' : 'List', 'items' : [ 73, 8, 3 ] },
        'list2' : { 'type' : 'List_int',
                    'items' : np.array([ 6, 8, 1, 7, 3, 5, 1, 0, 6, 3, 8, 2 ]),
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -3 } },
        'list3' : [ 1, 2, 3, 4 ],
        'list4' : [],
        'dict1' : { 'type' : 'Dict', 'key' : 'i' },
        'dict2' : { 'type' : 'Dict', 'num' : 1, 'key' : 'i' },
        'dict3' : { 'type' : 'Dict_int', 'num' : 2, 'key' : 'i' },
        'fits1' : { 'type' : 'FitsHeader', 'key' : 'CCDNUM' },
        'fits2' : { 'type' : 'FitsHeader', 'key' : 'FILPOS' },
        'sum1' : { 'type' : 'Sum', 'items' : [ 72.3, '2', { 'type' : 'Dict', 'key' : 'i' } ] },
        'cur1' : { 'type' : 'Current', 'key' : 'val1' },
        'cur2' : { 'type' : 'Current_int', 'key' : 'list2.index.step' },
        'bad1' : 'left',
        'bad2' : int,
        'bad3' : { 'type' : 'Current', 'key' : 'list2.index.type' },
        'bad4' : { 'type' : 'Catalog' , 'num' : 2, 'col' : 'int1' },
        'bad5' : { 'type' : 'Catalog' , 'num' : -1, 'col' : 'int1' },
    }

    galsim.config.ProcessInput(config)

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, int)[0]
    np.testing.assert_equal(val1, 9)

    val2 = galsim.config.ParseValue(config,'val2',config, int)[0]
    np.testing.assert_equal(val2, 8)

    val3 = galsim.config.ParseValue(config,'val3',config, int)[0]
    np.testing.assert_equal(val3, -400)

    val4 = galsim.config.ParseValue(config,'val4',config, int)[0]
    np.testing.assert_equal(val4, None)

    # Test conversions from strings
    str1 = galsim.config.ParseValue(config,'str1',config, int)[0]
    np.testing.assert_equal(str1, 8)

    str2 = galsim.config.ParseValue(config,'str2',config, int)[0]
    np.testing.assert_equal(str2, -2)

    # Test values read from a Catalog
    cat1 = []
    cat2 = []
    cat3 = []
    cat4 = []
    config['index_key'] = 'image_num'
    for k in range(5):
        config['image_num'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, int)[0])
        cat2.append(galsim.config.ParseValue(config,'cat2',config, int)[0])
        cat3.append(galsim.config.ParseValue(config,'cat3',config, int)[0])
        cat4.append(galsim.config.ParseValue(config,'cat4',config, int)[0])

    np.testing.assert_array_equal(cat1, [ 9, 0, -4, 9, 0 ])
    np.testing.assert_array_equal(cat2, [ -3, 8, 17, -3, 8 ])
    np.testing.assert_array_equal(cat3, [ 9, 0, -4, 9, 0 ])
    np.testing.assert_array_equal(cat4, [ -3, 8, 17, -3, 8 ])

    # Test values generated from a uniform deviate
    del config['index_key']
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        config['obj_num'] = k
        ran1 = galsim.config.ParseValue(config,'ran1',config, int)[0]
        np.testing.assert_equal(ran1, int(math.floor(rng() * 4)))

        ran2 = galsim.config.ParseValue(config,'ran2',config, int)[0]
        np.testing.assert_equal(ran2, int(math.floor(rng() * 16))-5)

    # Test values generated from various other deviates
    for k in range(6):
        config['obj_num'] = k
        dev = galsim.PoissonDeviate(rng, mean=137)
        dev1 = galsim.config.ParseValue(config,'dev1',config, int)[0]
        np.testing.assert_almost_equal(dev1, dev())

        dev = galsim.BinomialDeviate(rng, N=17)
        dev2 = galsim.config.ParseValue(config,'dev2',config, int)[0]
        np.testing.assert_almost_equal(dev2, dev())

        dev = galsim.BinomialDeviate(rng, N=17, p=0.2)
        dev3 = galsim.config.ParseValue(config,'dev3',config, int)[0]
        np.testing.assert_almost_equal(dev3, dev())

    # Test values generated from a Sequence
    seq1 = []
    seq2 = []
    seq3 = []
    seq4 = []
    seq5 = []
    config['index_key'] = 'obj_num'
    for k in range(6):
        config['obj_num'] = k
        seq1.append(galsim.config.ParseValue(config,'seq1',config, int)[0])
        seq2.append(galsim.config.ParseValue(config,'seq2',config, int)[0])
        seq3.append(galsim.config.ParseValue(config,'seq3',config, int)[0])
        seq4.append(galsim.config.ParseValue(config,'seq4',config, int)[0])
        seq5.append(galsim.config.ParseValue(config,'seq5',config, int)[0])

    np.testing.assert_array_equal(seq1, [ 0, 1, 2, 3, 4, 5 ])
    np.testing.assert_array_equal(seq2, [ 0, 3, 6, 9, 12, 15 ])
    np.testing.assert_array_equal(seq3, [ 1, 6, 11, 16, 21, 26 ])
    np.testing.assert_array_equal(seq4, [ 10, 8, 6, 4, 2, 0 ])
    np.testing.assert_array_equal(seq5, [ 1, 1, 2, 2, 1, 1 ])

    # This is more like how the indexing actually happens in a regular config run:
    seq_file = []
    seq_image = []
    seq_obj = []
    seq_obj2 = []
    config['file_num'] = 0
    config['image_num'] = 0
    config['obj_num'] = 0
    for file_num in range(3):
        config['start_obj_num'] = config['obj_num']
        for image_num in range(2):
            for obj_num in range(5):
                seq_file.append(galsim.config.ParseValue(config,'seq_file',config, int)[0])
                seq_image.append(galsim.config.ParseValue(config,'seq_image',config, int)[0])
                seq_obj.append(galsim.config.ParseValue(config,'seq_obj',config, int)[0])
                seq_obj2.append(galsim.config.ParseValue(config,'seq_obj2',config, int)[0])
                config['obj_num'] += 1
            config['image_num'] += 1
        config['file_num'] += 1
    del config['start_obj_num']

    np.testing.assert_array_equal(seq_file, [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                              2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ])
    np.testing.assert_array_equal(seq_image, [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                                               2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                               4, 4, 4, 4, 4, 5, 5, 5, 5, 5 ])
    np.testing.assert_array_equal(seq_obj, [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29 ])
    np.testing.assert_array_equal(seq_obj2, [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                              0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                              0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ])

    # Test values taken from a List
    list1 = []
    list2 = []
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, int)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, int)[0])

    np.testing.assert_array_equal(list1, [ 73, 8, 3, 73, 8 ])
    np.testing.assert_array_equal(list2, [ 8, 0, 3, 8, 8 ])

    # Test a direct list in the config file.
    list3  = galsim.config.ParseValue(config,'list3',config, list)[0]
    list4  = galsim.config.ParseValue(config,'list4',config, list)[0]
    np.testing.assert_array_equal(list3, [ 1, 2, 3, 4 ])
    np.testing.assert_array_equal(list4, [])

    # You can also give None as the value type, which just returns whatever is in the dict.
    list3b  = galsim.config.ParseValue(config,'list3',config, None)[0]
    list4b  = galsim.config.ParseValue(config,'list4',config, None)[0]
    np.testing.assert_array_equal(list3b, [ 1, 2, 3, 4 ])
    np.testing.assert_array_equal(list4b, [])

    # Test values read from a Dict
    dict = []
    dict.append(galsim.config.ParseValue(config,'dict1',config, int)[0])
    dict.append(galsim.config.ParseValue(config,'dict2',config, int)[0])
    dict.append(galsim.config.ParseValue(config,'dict3',config, int)[0])
    np.testing.assert_array_equal(dict, [ 17, -23, 1 ])

    assert galsim.config.ParseValue(config,'fits1',config, int)[0] == 1
    assert galsim.config.ParseValue(config,'fits2',config, int)[0] == 6

    sum1 = galsim.config.ParseValue(config,'sum1', config, int)[0]
    np.testing.assert_almost_equal(sum1, 72 + 2 + 17)

    cur1 = galsim.config.ParseValue(config,'cur1', config, int)[0]
    np.testing.assert_array_equal(cur1, 9)
    cur2 = galsim.config.ParseValue(config,'cur2', config, int)[0]
    np.testing.assert_array_equal(cur2, -3)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad1', config, int)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad2', config, int)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad3',config, int)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad4',config, int)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad5',config, int)
    config = galsim.config.CleanConfig(config)
    del config['input']['catalog']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'cat1',config, int)
    del config['input']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'cat1',config, int)


@timer
def test_bool_value():
    """Test various ways to generate a bool value
    """
    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ],
                    'dict' : [
                        { 'dir' : 'config_input', 'file_name' : 'dict.p' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.json' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.yaml' } ],
                    'fits_header' : { 'dir' : 'fits_files', 'file_name' : 'tpv.fits' },
                  },

        'val1' : True,
        'val2' : 1,
        'val3' : 0.0,
        'val4' : None,
        'str1' : 'true',
        'str2' : '0',
        'str3' : 'yes',
        'str4' : 'No',
        'cat1' : { 'type' : 'Catalog' , 'col' : 4 },
        'cat2' : { 'type' : 'Catalog' , 'col' : 5 },
        'cat3' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'bool1' },
        'cat4' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'bool2' },
        'ran1' : { 'type' : 'Random' },
        'ran2' : { 'type' : 'Random_bool', 'p' : 0.8 },
        'dev1' : { 'type' : 'RandomBinomial', 'N' : 1 },
        'dev2' : { 'type' : 'RandomBinomial', 'N' : 1, 'p' : 0.5 },
        'dev3' : { 'type' : 'RandomBinomial', 'p' : 0.2 },
        'seq1' : { 'type' : 'Sequence' },
        'seq2' : { 'type' : 'Sequence_bool', 'first' : True, 'repeat' : 2 },
        'list1' : { 'type' : 'List', 'items' : [ 'yes', 'no', 'no' ] },
        'list2' : { 'type' : 'List_bool',
                    'items' : [ 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0 ],
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -3 } },
        'dict1' : { 'type' : 'Dict', 'key' : 'b' },
        'dict2' : { 'type' : 'Dict', 'num' : 1, 'key' : 'b' },
        'dict3' : { 'type' : 'Dict_bool', 'num' : 2, 'key' : 'b' },
        'fits1' : { 'type' : 'FitsHeader', 'key' : 'PHOTFLAG' },
        'fits2' : { 'type' : 'FitsHeader', 'key' : 'SCAMPFLG' },
        'bad1' : 'left',
        'bad2' : 'nope',
        'bad3' : { 'type' : 'RandomBinomial', 'N' : 2 },
    }

    galsim.config.ProcessInput(config)

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, bool)[0]
    np.testing.assert_equal(val1, True)

    val2 = galsim.config.ParseValue(config,'val2',config, bool)[0]
    np.testing.assert_equal(val2, True)

    val3 = galsim.config.ParseValue(config,'val3',config, bool)[0]
    np.testing.assert_equal(val3, False)

    val4 = galsim.config.ParseValue(config,'val4',config, bool)[0]
    np.testing.assert_equal(val4, None)

    # Test conversions from strings
    str1 = galsim.config.ParseValue(config,'str1',config, bool)[0]
    np.testing.assert_equal(str1, True)

    str2 = galsim.config.ParseValue(config,'str2',config, bool)[0]
    np.testing.assert_equal(str2, False)

    str3 = galsim.config.ParseValue(config,'str3',config, bool)[0]
    np.testing.assert_equal(str3, True)

    str4 = galsim.config.ParseValue(config,'str4',config, bool)[0]
    np.testing.assert_equal(str4, False)

    # Test values read from a Catalog
    cat1 = []
    cat2 = []
    cat3 = []
    cat4 = []
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, bool)[0])
        cat2.append(galsim.config.ParseValue(config,'cat2',config, bool)[0])
        cat3.append(galsim.config.ParseValue(config,'cat3',config, bool)[0])
        cat4.append(galsim.config.ParseValue(config,'cat4',config, bool)[0])

    np.testing.assert_array_equal(cat1, [ 1, 0, 1, 1, 0 ])
    np.testing.assert_array_equal(cat2, [ 1, 0, 0, 1, 0 ])
    np.testing.assert_array_equal(cat3, [ 1, 0, 1, 1, 0 ])
    np.testing.assert_array_equal(cat4, [ 1, 0, 0, 1, 0 ])

    # Test values generated from a uniform deviate
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        config['obj_num'] = k
        ran1 = galsim.config.ParseValue(config,'ran1',config, bool)[0]
        np.testing.assert_equal(ran1, rng() < 0.5)

    for k in range(6):
        config['obj_num'] = k
        ran1 = galsim.config.ParseValue(config,'ran2',config, bool)[0]
        np.testing.assert_equal(ran1, rng() < 0.8)

    # Test values generated from binomial deviate
    for k in range(6):
        config['obj_num'] = k
        dev = galsim.BinomialDeviate(rng, N=1)
        dev1 = galsim.config.ParseValue(config,'dev1',config, bool)[0]
        np.testing.assert_almost_equal(dev1, dev())

        dev = galsim.BinomialDeviate(rng, N=1, p=0.5)
        dev2 = galsim.config.ParseValue(config,'dev2',config, bool)[0]
        np.testing.assert_almost_equal(dev2, dev())

        dev = galsim.BinomialDeviate(rng, N=1, p=0.2)
        dev3 = galsim.config.ParseValue(config,'dev3',config, bool)[0]
        np.testing.assert_almost_equal(dev3, dev())

    # Test values generated from a Sequence
    seq1 = []
    seq2 = []
    config['index_key'] = 'obj_num'
    for k in range(6):
        config['obj_num'] = k
        seq1.append(galsim.config.ParseValue(config,'seq1',config, bool)[0])
        seq2.append(galsim.config.ParseValue(config,'seq2',config, bool)[0])

    np.testing.assert_array_equal(seq1, [ 0, 1, 0, 1, 0, 1 ])
    np.testing.assert_array_equal(seq2, [ 1, 1, 0, 0, 1, 1 ])

    # Test values taken from a List
    list1 = []
    list2 = []
    config['index_key'] = 'file_num'
    for k in range(5):
        config['file_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, bool)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, bool)[0])

    np.testing.assert_array_equal(list1, [ 1, 0, 0, 1, 0 ])
    np.testing.assert_array_equal(list2, [ 0, 1, 1, 1, 0 ])

    # Test values read from a Dict
    dict = []
    dict.append(galsim.config.ParseValue(config,'dict1',config, bool)[0])
    dict.append(galsim.config.ParseValue(config,'dict2',config, bool)[0])
    dict.append(galsim.config.ParseValue(config,'dict3',config, bool)[0])
    np.testing.assert_array_equal(dict, [ True, False, False ])

    assert galsim.config.ParseValue(config,'fits1',config, bool)[0] == False
    assert galsim.config.ParseValue(config,'fits2',config, bool)[0] == False

    # Test bad values
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad1',config, bool)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad2',config, bool)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad3',config, bool)


@timer
def test_str_value():
    """Test various ways to generate a str value
    """
    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ],
                    'dict' : [
                        { 'dir' : 'config_input', 'file_name' : 'dict.p' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.json' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.yaml' } ],
                    'fits_header' : { 'dir' : 'fits_files', 'file_name' : 'tpv.fits' },
                  },

        'val1' : -93,
        'val2' : True,
        'val3' : 123.8,
        'val4' : None,
        'str1' : "Norwegian",
        'str2' : u"Blue",
        'cat1' : { 'type' : 'Catalog' , 'col' : 6 },
        'cat2' : { 'type' : 'Catalog' , 'col' : 7 },
        'cat3' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'str1' },
        'cat4' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'str2' },
        'list1' : { 'type' : 'List', 'items' : [ 'Beautiful', 'plumage!', 'Ay?' ] },
        'list2' : { 'type' : 'List_str', 'items' : [ 'Beautiful', 'plumage!', 'Ay?' ] },
        'file1' : { 'type' : 'NumberedFile', 'root' : 'file', 'num' : 5,
                    'ext' : '.fits.fz', 'digits' : 3 },
        'file2' : { 'type' : 'NumberedFile', 'root' : 'file', 'num' : 5 },
        'fs1' : { 'type' : 'FormattedStr',
                  'format' : 'realgal_type%02d_dilation%d.fits',
                  'items' : [
                      { 'type' : 'Sequence' , 'repeat' : 3 },
                      { 'type' : 'Sequence' , 'nitems' : 3 } ] },
        'fs2' : { 'type' : 'FormattedStr',
                  'format' : '%%%d %i %x %o%i %lf=%g=%e %hi%u %r%s %%',
                  'items' : [4, 5, 12, 9, 9, math.pi, math.pi, math.pi, 11, -11,
                             'Goodbye cruel world.', ', said Pink.'] },
        'dict1' : { 'type' : 'Dict', 'key' : 's' },
        'dict2' : { 'type' : 'Dict', 'num' : 1, 'key' : 's' },
        'dict3' : { 'type' : 'Dict', 'num' : 2, 'key' : 's' },
        'fits1' : { 'type' : 'FitsHeader', 'key' : 'FILTER' },
        'fits2' : { 'type' : 'FitsHeader', 'key' : 'DETECTOR' },
        'bad1' : { 'type' : 'FormattedStr', 'format' : 'realgal%02q.fits', 'items' : [4,5,6] },
        'bad2' : { 'type' : 'FormattedStr', 'format' : 'realgal%02', 'items' : [4,5,6] },
        'bad3' : { 'type' : 'FormattedStr', 'format' : 'realgal%02d_%d.fits', 'items' : [4,5,6] },
        'bad4' : { 'type' : 'List', 'items' : 'Beautiful plumage! Ay?' },
        'bad5' : { 'type' : 'List', 'items' : [ 'Beautiful', 'plumage!', 'Ay?' ], 'index' : 5 },
    }

    galsim.config.ProcessInput(config)

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, str)[0]
    np.testing.assert_equal(val1, '-93')

    val2 = galsim.config.ParseValue(config,'val2',config, str)[0]
    np.testing.assert_equal(val2, 'True')

    val3 = galsim.config.ParseValue(config,'val3',config, str)[0]
    np.testing.assert_equal(val3, '123.8')

    val4 = galsim.config.ParseValue(config,'val4',config, str)[0]
    np.testing.assert_equal(val4, None)

    # Test conversions from strings
    str1 = galsim.config.ParseValue(config,'str1',config, str)[0]
    np.testing.assert_equal(str1, 'Norwegian')

    str2 = galsim.config.ParseValue(config,'str2',config, str)[0]
    np.testing.assert_equal(str2, 'Blue')

    # Test values read from a Catalog
    cat1 = []
    cat2 = []
    cat3 = []
    cat4 = []
    config['index_key'] = 'obj_num'
    for k in range(3):
        config['obj_num'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, str)[0])
        cat2.append(galsim.config.ParseValue(config,'cat2',config, str)[0])
        cat3.append(galsim.config.ParseValue(config,'cat3',config, str)[0])
        cat4.append(galsim.config.ParseValue(config,'cat4',config, str)[0])

    np.testing.assert_array_equal(cat1, ["He's", "bleedin'", "demised!"])
    # Note: white space in the input catalog always separates columns. ' and " don't work.
    np.testing.assert_array_equal(cat2, ['"ceased', '"bereft', '"kicked'])
    np.testing.assert_array_equal(cat3, ["He's", "bleedin'", "demised!"])
    np.testing.assert_array_equal(cat4, ['"ceased', '"bereft', '"kicked'])

    # Test values taken from a List
    list1 = []
    list2 = []
    config['index_key'] = 'image_num'
    for k in range(5):
        config['image_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, str)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, str)[0])

    np.testing.assert_array_equal(list1, ['Beautiful', 'plumage!', 'Ay?', 'Beautiful', 'plumage!'])
    np.testing.assert_array_equal(list2, list1)

    # Test values built using NumberedFile
    file1 = galsim.config.ParseValue(config,'file1',config, str)[0]
    np.testing.assert_equal(file1, 'file005.fits.fz')
    file2 = galsim.config.ParseValue(config,'file2',config, str)[0]
    np.testing.assert_equal(file2, 'file5')

    # Test value built from FormattedStr
    config['index_key'] = 'obj_num'
    for k in range(9):
        config['obj_num'] = k
        type = k / 3
        dil = k % 3
        fs1 = galsim.config.ParseValue(config,'fs1',config, str)[0]
        np.testing.assert_equal(fs1, 'realgal_type%02d_dilation%d.fits'%(type,dil))

    fs2 = galsim.config.ParseValue(config,'fs2',config, str)[0]
    np.testing.assert_equal(fs2,
        "%4 5 c 119 3.141593=3.14159=3.141593e+00 11-11 'Goodbye cruel world.', said Pink. %")

    # Test values read from a Dict
    dict = []
    dict.append(galsim.config.ParseValue(config,'dict1',config, str)[0])
    dict.append(galsim.config.ParseValue(config,'dict2',config, str)[0])
    dict.append(galsim.config.ParseValue(config,'dict3',config, str)[0])
    np.testing.assert_array_equal(dict, [ 'Life', 'of', 'Brian' ])

    assert galsim.config.ParseValue(config,'fits1',config, str)[0] == 'I'
    assert galsim.config.ParseValue(config,'fits2',config, str)[0] == 'Mosaic2'

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad1',config, str)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad2',config, str)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad3',config, str)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad4',config, str)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad5',config, str)

@timer
def test_angle_value():
    """Test various ways to generate an Angle value
    """
    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ] },

        'val1' : 1.9 * galsim.radians,
        'val2' : -41 * galsim.degrees,
        'val3' : None,
        'str1' : '0.73 radians',
        'str2' : '240 degrees',
        'str3' : '1.2 rad',
        'str4' : '45:12:55.1 deg',
        'str5' : '6 hrs',
        'str6' : '21:31:05.3 hour',
        'str7' : '-240 arcmin',
        'str8' : '1800 arcsec',
        'cat1' : { 'type' : 'Radians' ,
                   'theta' : { 'type' : 'Catalog' , 'col' : 10 } },
        'cat2' : { 'type' : 'Degrees' ,
                   'theta' : { 'type' : 'Catalog' , 'col' : 11 } },
        'cat3' : { 'type' : 'Radians' ,
                   'theta' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'angle1' } },
        'cat4' : { 'type' : 'Degrees' ,
                   'theta' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'angle2' } },
        'ran1' : { 'type' : 'Random' },
        'ran2' : { 'type' : 'Random_Angle' },
        'seq1' : { 'type' : 'Rad', 'theta' : { 'type' : 'Sequence' } },
        'seq2' : { 'type' : 'Deg', 'theta' : { 'type' : 'Sequence', 'first' : 45, 'step' : 80 } },
        'list1' : { 'type' : 'List',
                    'items' : [ 73 * galsim.arcmin,
                                8.9 * galsim.arcmin,
                                3.14 * galsim.arcmin ] },
        'list2' : { 'type' : 'List_Angle',
                    'items' : [ 73 * galsim.arcmin,
                                8.9 * galsim.arcmin,
                                3.14 * galsim.arcmin ] },
        'sum1' : { 'type' : 'Sum', 'items' : [ 72 * galsim.degrees, '2.33 degrees' ] },
        'bad1' : '1.9 * galsim.rradds',
        'bad2' : { 'type' : 'Sum', 'items' : 72 * galsim.degrees },
    }

    galsim.config.ProcessInput(config)

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(val1.rad, 1.9)

    val2 = galsim.config.ParseValue(config,'val2',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(val2.rad, -41 * math.pi/180)

    val3 = galsim.config.ParseValue(config,'val3',config, galsim.Angle)[0]
    np.testing.assert_equal(val3, None)

    val1b = galsim.config.ParseValue(config,'val1',config, None)[0]
    val2b = galsim.config.ParseValue(config,'val2',config, None)[0]
    np.testing.assert_almost_equal(val1b.rad, 1.9)
    np.testing.assert_almost_equal(val2b.rad, -41 * math.pi/180)

    # Test conversions from strings
    str1 = galsim.config.ParseValue(config,'str1',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str1.rad, 0.73)

    str2 = galsim.config.ParseValue(config,'str2',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str2 / galsim.degrees, 240)

    str3 = galsim.config.ParseValue(config,'str3',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str3.rad, 1.2)

    str4 = galsim.config.ParseValue(config,'str4',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str4.rad, galsim.Angle.from_dms('45:12:55.1').rad)

    str5 = galsim.config.ParseValue(config,'str5',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str5.rad, math.pi/2)

    str6 = galsim.config.ParseValue(config,'str6',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str6.rad, galsim.Angle.from_hms('21:31:05.3').rad)

    str7 = galsim.config.ParseValue(config,'str7',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str7 / galsim.degrees, -4)

    str8 = galsim.config.ParseValue(config,'str8',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str8 / galsim.degrees, 0.5)

    # Test values read from a Catalog
    cat1 = []
    cat2 = []
    cat3 = []
    cat4 = []
    config['index_key'] = 'file_num'
    for k in range(5):
        config['file_num'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, galsim.Angle)[0].rad)
        cat2.append(galsim.config.ParseValue(config,'cat2',config, galsim.Angle)[0]/galsim.degrees)
        cat3.append(galsim.config.ParseValue(config,'cat3',config, galsim.Angle)[0].rad)
        cat4.append(galsim.config.ParseValue(config,'cat4',config, galsim.Angle)[0]/galsim.degrees)

    np.testing.assert_array_almost_equal(cat1, [ 1.2, 0.1, -0.9, 1.2, 0.1 ])
    np.testing.assert_array_almost_equal(cat2, [ 23, 15, 82, 23, 15 ])
    np.testing.assert_array_almost_equal(cat3, [ 1.2, 0.1, -0.9, 1.2, 0.1 ])
    np.testing.assert_array_almost_equal(cat4, [ 23, 15, 82, 23, 15 ])

    # Test values generated from a uniform deviate
    del config['index_key']
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        config['obj_num'] = k
        ran1 = galsim.config.ParseValue(config,'ran1',config, galsim.Angle)[0]
        ran2 = galsim.config.ParseValue(config,'ran2',config, galsim.Angle)[0]
        theta = rng() * 2 * math.pi
        np.testing.assert_almost_equal(ran1.rad, theta)
        theta = rng() * 2 * math.pi
        np.testing.assert_almost_equal(ran2.rad, theta)

    # Test values generated from a Sequence
    seq1 = []
    seq2 = []
    config['index_key'] = 'obj_num'
    for k in range(6):
        config['obj_num'] = k
        seq1.append(galsim.config.ParseValue(config,'seq1',config, galsim.Angle)[0].rad)
        seq2.append(galsim.config.ParseValue(config,'seq2',config, galsim.Angle)[0]/galsim.degrees)

    np.testing.assert_array_almost_equal(seq1, [ 0, 1, 2, 3, 4, 5 ])
    np.testing.assert_array_almost_equal(seq2, [ 45, 125, 205, 285, 365, 445 ])

    # Test values taken from a List
    list1 = []
    list2 = []
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, galsim.Angle)[0]/galsim.arcmin)
        list2.append(galsim.config.ParseValue(config,'list2',config, galsim.Angle)[0]/galsim.arcmin)

    np.testing.assert_array_almost_equal(list1, [ 73, 8.9, 3.14, 73, 8.9 ])
    np.testing.assert_equal(list2, list1)

    sum1 = galsim.config.ParseValue(config,'sum1', config, galsim.Angle)[0]
    np.testing.assert_almost_equal(sum1 / galsim.degrees, 72 + 2.33)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad1', config, galsim.Angle)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad2', config, galsim.Angle)


@timer
def test_shear_value():
    """Test various ways to generate a Shear value
    """
    halo_mass = 1.e14
    halo_conc = 4
    halo_z = 0.3
    gal_z = 1.3
    config = {
        'val1' : galsim.Shear(g1=0.2, g2=0.3),
        'val2' : galsim.Shear(e1=0.1),
        'val3' : None,
        's1' : { 'type' : 'E1E2', 'e1' : 0.5, 'e2' : -0.1 },
        's2' : { 'type' : 'EBeta', 'e' : 0.5, 'beta' : 0.1 * galsim.radians },
        's3' : { 'type' : 'G1G2', 'g1' : 0.5, 'g2' : -0.1 },
        's4' : { 'type' : 'GBeta', 'g' : 0.5, 'beta' : 0.1 * galsim.radians },
        's5' : { 'type' : 'Eta1Eta2', 'eta1' : 0.5, 'eta2' : -0.1 },
        's6' : { 'type' : 'EtaBeta', 'eta' : 0.5, 'beta' : 0.1 * galsim.radians },
        's7' : { 'type' : 'QBeta', 'q' : 0.5, 'beta' : 0.1 * galsim.radians },
        'list1' : { 'type' : 'List',
                    'items' : [ galsim.Shear(g1 = 0.2, g2 = -0.3),
                                galsim.Shear(g1 = -0.5, g2 = 0.2),
                                galsim.Shear(g1 = 0.1, g2 = 0.0) ] },
        'list2' : { 'type' : 'List_Shear',
                    'items' : [ galsim.Shear(g1 = 0.2, g2 = -0.3),
                                galsim.Shear(g1 = -0.5, g2 = 0.2),
                                galsim.Shear(g1 = 0.1, g2 = 0.0) ] },
        'sum1' : { 'type' : 'Sum',
                  'items' : [ galsim.Shear(g1 = 0.2, g2 = -0.3),
                              galsim.Shear(g1 = -0.5, g2 = 0.2),
                              galsim.Shear(g1 = 0.1, g2 = 0.0) ] },
        'nfw' : { 'type' : 'NFWHaloShear' },
        'ps' : { 'type' : 'PowerSpectrumShear' },
        'bad1' : { 'type' : 'G1G2', 'g1' : 0.5 },
        'bad2' : { 'type' : 'G1G2' },
        'bad3' : { 'type' : 'G1G2', 'g1' : 0.5, 'g2' : -0.1, 'g3' : 0.3 },

        'input' : { 'nfw_halo' : { 'mass' : halo_mass, 'conc' : halo_conc, 'redshift' : halo_z },
                    'power_spectrum' : { 'e_power_function' : 'np.exp(-k**0.2)',
                                         'grid_spacing' : 10, 'interpolant' : 'linear',
                                         'ngrid' : 40, 'center' : '5,5' },
                  },
    }

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(val1.g1, 0.2)
    np.testing.assert_almost_equal(val1.g2, 0.3)

    val2 = galsim.config.ParseValue(config,'val2',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(val2.e1, 0.1)
    np.testing.assert_almost_equal(val2.e2, 0.)

    val3 = galsim.config.ParseValue(config,'val3',config, galsim.Shear)[0]
    np.testing.assert_equal(val3, None)

    # Test various direct types
    s1 = galsim.config.ParseValue(config,'s1',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s1.e1, 0.5)
    np.testing.assert_almost_equal(s1.e2, -0.1)

    s2 = galsim.config.ParseValue(config,'s2',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s2.e, 0.5)
    np.testing.assert_almost_equal(s2.beta.rad, 0.1)

    s3 = galsim.config.ParseValue(config,'s3',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s3.g1, 0.5)
    np.testing.assert_almost_equal(s3.g2, -0.1)

    s4 = galsim.config.ParseValue(config,'s4',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s4.g, 0.5)
    np.testing.assert_almost_equal(s4.beta.rad, 0.1)

    s5 = galsim.config.ParseValue(config,'s5',config, galsim.Shear)[0]
    eta = s5.eta
    e = s5.e
    eta1 = s5.e1 * eta/e
    eta2 = s5.e2 * eta/e
    np.testing.assert_almost_equal(eta1, 0.5)
    np.testing.assert_almost_equal(eta2, -0.1)

    s6 = galsim.config.ParseValue(config,'s6',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s6.eta, 0.5)
    np.testing.assert_almost_equal(s6.beta.rad, 0.1)

    s7 = galsim.config.ParseValue(config,'s7',config, galsim.Shear)[0]
    g = s7.g
    q = (1-g)/(1+g)
    np.testing.assert_almost_equal(q, 0.5)
    np.testing.assert_almost_equal(s7.beta.rad, 0.1)

    # Test values taken from a List
    list1 = []
    list2 = []
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, galsim.Shear)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, galsim.Shear)[0])

    np.testing.assert_almost_equal(list1[0].g1, 0.2)
    np.testing.assert_almost_equal(list1[0].g2, -0.3)
    np.testing.assert_almost_equal(list1[1].g1, -0.5)
    np.testing.assert_almost_equal(list1[1].g2, 0.2)
    np.testing.assert_almost_equal(list1[2].g1, 0.1)
    np.testing.assert_almost_equal(list1[2].g2, 0.0)
    np.testing.assert_almost_equal(list1[3].g1, 0.2)
    np.testing.assert_almost_equal(list1[3].g2, -0.3)
    np.testing.assert_almost_equal(list1[4].g1, -0.5)
    np.testing.assert_almost_equal(list1[4].g2, 0.2)
    np.testing.assert_equal(list2, list1)

    sum1 = galsim.config.ParseValue(config,'sum1', config, galsim.Shear)[0]
    s = galsim.Shear(g1=0.2, g2=-0.3)
    s += galsim.Shear(g1=-0.5, g2=0.2)
    s += galsim.Shear(g1=0.1, g2=0.0)
    np.testing.assert_almost_equal(sum1.g1, s.g1)
    np.testing.assert_almost_equal(sum1.g2, s.g2)

    # Test NFWHaloShear
    galsim.config.ProcessInput(config)
    galsim.config.SetupInputsForImage(config, None)
    # Raise an error because no uv_pos
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'nfw', config, galsim.Shear)
    config['uv_pos'] = galsim.PositionD(6,8)
    # Still raise an error because no redshift
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'nfw', config, galsim.Shear)
    # With this, it should work.
    config['gal'] = { 'redshift' : gal_z }
    nfw_halo = galsim.NFWHalo(mass=halo_mass, conc=halo_conc, redshift=halo_z)
    nfw1a = galsim.config.ParseValue(config,'nfw',config, galsim.Shear)[0]
    nfw1b = nfw_halo.getShear((6,8), gal_z)
    print('nfw1a = ',nfw1a)
    print('nfw1b = ',nfw1b)
    np.testing.assert_almost_equal(nfw1a.g1, nfw1b[0])
    np.testing.assert_almost_equal(nfw1a.g2, nfw1b[1])

    # If shear is larger than 1, it raises a warning and returns 0,0
    galsim.config.RemoveCurrent(config)
    config['uv_pos'] = galsim.PositionD(0.1,0.2)
    print("strong lensing shear = ",nfw_halo.getShear((0.1,0.2), gal_z))
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, cl.logger)
        nfw2a = galsim.config.ParseValue(config, 'nfw', config, galsim.Shear)[0]
    print(cl.output)
    assert "Warning: NFWHalo shear (g1=1.148773, g2=-1.531697) is invalid." in cl.output
    np.testing.assert_almost_equal((nfw2a.g1, nfw2a.g2), (0,0))

    # Test PowerSpectrumShear
    rng = galsim.BaseDeviate(1234)
    config['rng'] = rng.duplicate()
    ps = galsim.PowerSpectrum(e_power_function='np.exp(-k**0.2)')
    ps.buildGrid(grid_spacing=10, ngrid=40, center=galsim.PositionD(5,5), interpolant='linear',
                 rng=rng)
    config['image_xsize'] = config['image_ysize'] = 2000
    config['wcs'] = galsim.PixelScale(0.1)
    config['image_center'] = galsim.PositionD(0,0)
    galsim.config.SetupInputsForImage(config, None)
    ps1a = galsim.config.ParseValue(config,'ps',config, galsim.Shear)[0]
    ps1b = ps.getShear((0.1,0.2))
    print("ps shear= ",ps1b)
    np.testing.assert_almost_equal(ps1a.g1, ps1b[0])
    np.testing.assert_almost_equal(ps1a.g2, ps1b[1])

    # Beef up the amplitude to get strong lensing.
    ps = galsim.PowerSpectrum(e_power_function='500 * np.exp(-k**0.2)')
    ps.buildGrid(grid_spacing=10, ngrid=40, center=galsim.PositionD(5,5), interpolant='linear',
                 rng=rng)
    print("strong lensing shear = ",ps.getShear((0.1,0.2)))
    config = galsim.config.CleanConfig(config)
    config['input']['power_spectrum']['e_power_function'] = '500 * np.exp(-k**0.2)'
    galsim.config.SetupInputsForImage(config, None)
    ps2b = ps.getShear((0.1,0.2))
    print("ps shear= ",ps2b)
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, logger=cl.logger)
        ps2a = galsim.config.ParseValue(config,'ps',config, galsim.Shear)[0]
    assert 'PowerSpectrum shear (g1=-1.626101, g2=0.287082) is invalid. Using shear = 0.' in cl.output
    np.testing.assert_almost_equal((ps2a.g1, ps2a.g2), (0,0))

    # Out of bounds results in shear = 0, and a warning.
    galsim.config.RemoveCurrent(config)
    config['uv_pos'] = galsim.PositionD(1000,2000)
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, cl.logger)
        ps2c = galsim.config.ParseValue(config, 'ps', config, galsim.Shear)[0]
    print(cl.output)
    assert ("Extrapolating beyond input range. galsim.PositionD(x=1000.0, y=2000.0) not in "
            "galsim.BoundsD(xmin=-190.00000000000023, xmax=200.00000000000023, "
            "ymin=-190.00000000000023, ymax=200.00000000000023)") in cl.output
    np.testing.assert_almost_equal((ps2c.g1, ps2c.g2), (0,0))

    # Error if no uv_pos
    del config['uv_pos']
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config, 'ps', config, galsim.Shear)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad1',config, galsim.Shear)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad2',config, galsim.Shear)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad3',config, galsim.Shear)


@timer
def test_pos_value():
    """Test various ways to generate a Position value
    """
    config = {
        'val1' : galsim.PositionD(0.1,0.2),
        'val2' : '0.1, 0.2',
        'val3' : None,
        'val4' : '123.4, 567.8',
        'xy1' : { 'type' : 'XY', 'x' : 1.3, 'y' : 2.4 },
        'ran1' : { 'type' : 'RandomCircle', 'radius' : 3 },
        'ran2' : { 'type' : 'RandomCircle', 'radius' : 1, 'center' : galsim.PositionD(3,7) },
        'ran3' : { 'type' : 'RandomCircle', 'radius' : 3.1, 'center' : galsim.PositionD(0.2,-0.9),
                   'inner_radius' : 1.3 },
        'ran4' : { 'type' : 'RTheta', 'r' : 1.3, 'theta' : { 'type': 'Random' } },
        'list1' : { 'type' : 'List',
                    'items' : [ galsim.PositionD(0.2, -0.3),
                                galsim.PositionD(-0.5, 0.2),
                                galsim.PositionD(0.1, 0.0) ] },
        'list2' : { 'type' : 'List_PositionD',
                    'items' : [ galsim.PositionD(0.2, -0.3),
                                galsim.PositionD(-0.5, 0.2),
                                galsim.PositionD(0.1, 0.0) ] },
        'sum1' : { 'type' : 'Sum',
                   'items' : [ galsim.PositionD(0.2, -0.3),
                               galsim.PositionD(-0.5, 0.2),
                               galsim.PositionD(0.1, 0.0) ] },
        'radec' : { 'type' : 'RADec', 'ra' : 13.4 * galsim.hours, 'dec' : -0.3 * galsim.degrees },
        'list_radec' : { 'type' : 'List_CelestialCoord',
                         'items' : [ { 'type': 'RADec', 'ra': '13.4 hours', 'dec': '-0.3 deg' } ],
                       },
        'cur1' : { 'type' : 'Current', 'key' : 'input.val1' },
        'cur2' : '@input.val2',
        'bad1' : '0.1, 0.2, 0.3',
        'bad2' : '0.1,',
        'bad3' : '0.1',
        'bad4' : 'red, blue',

        # This one tests @ with input, not used as a normal input field, but rather used as just
        # a regular field in the dict.
        'input' : { 'val1' : galsim.PositionD(0.1,0.2),
                    'val2' : '0.3, 0.4' },
    }
    # Also use this to check CopyConfig and CleanConfig.  Processing adds a lot to the
    # config dict for efficiency.  But CopyConfig should copy the current state, and
    # CleanConfig should get it back to a clean state after processing is done.
    # The one catch is that it needs to know what the top-level fields are, and we use non-standard
    # ones here.  So add them to top_level_fields.
    galsim.config.top_level_fields += config.keys()
    orig_config = galsim.config.CopyConfig(config)
    assert orig_config == config

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(val1.x, 0.1)
    np.testing.assert_almost_equal(val1.y, 0.2)

    val2 = galsim.config.ParseValue(config,'val2',config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(val1.x, 0.1)
    np.testing.assert_almost_equal(val1.y, 0.2)

    val3 = galsim.config.ParseValue(config,'val3',config, galsim.PositionD)[0]
    np.testing.assert_equal(val3, None)

    val4 = galsim.config.ParseValue(config,'val4',config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(val4.x, 123.4)
    np.testing.assert_almost_equal(val4.y, 567.8)

    xy1 = galsim.config.ParseValue(config,'xy1',config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(xy1.x, 1.3)
    np.testing.assert_almost_equal(xy1.y, 2.4)

    # Test Current
    cur1 = galsim.config.ParseValue(config,'cur1',config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(cur1.x, 0.1)
    np.testing.assert_almost_equal(cur1.y, 0.2)
    cur2 = galsim.config.ParseValue(config,'cur2',config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(cur2.x, 0.3)
    np.testing.assert_almost_equal(cur2.y, 0.4)

    # Test values generated in a random circle
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    config['index_key'] = 'image_num'
    for k in range(6):
        config['image_num'] = k
        ran1 = galsim.config.ParseValue(config,'ran1',config, galsim.PositionD)[0]
        # Emulate a do-while loop
        while True:
            x = (2*rng()-1) * 3
            y = (2*rng()-1) * 3
            rsq = x**2 + y**2
            if rsq <= 9: break
        np.testing.assert_almost_equal(ran1.x, x)
        np.testing.assert_almost_equal(ran1.y, y)

        ran2 = galsim.config.ParseValue(config,'ran2',config, galsim.PositionD)[0]
        while True:
            x = (2*rng()-1)
            y = (2*rng()-1)
            rsq = x**2 + y**2
            if rsq <= 1: break
        np.testing.assert_almost_equal(ran2.x, x+3)
        np.testing.assert_almost_equal(ran2.y, y+7)

        ran3 = galsim.config.ParseValue(config,'ran3',config, galsim.PositionD)[0]
        while True:
            x = (2*rng()-1) * 3.1
            y = (2*rng()-1) * 3.1
            rsq = x**2 + y**2
            if rsq >= 1.3**2 and rsq <= 3.1**2: break
        np.testing.assert_almost_equal(ran3.x, x+0.2)
        np.testing.assert_almost_equal(ran3.y, y-0.9)

        ran4 = galsim.config.ParseValue(config,'ran4',config, galsim.PositionD)[0]
        r = 1.3
        theta = rng() * 2. * math.pi
        np.testing.assert_almost_equal(ran4.x, r * math.cos(theta))
        np.testing.assert_almost_equal(ran4.y, r * math.sin(theta))

    # Test values taken from a List
    list1 = []
    list2 = []
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, galsim.PositionD)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, galsim.PositionD)[0])

    np.testing.assert_almost_equal(list1[0].x, 0.2)
    np.testing.assert_almost_equal(list1[0].y, -0.3)
    np.testing.assert_almost_equal(list1[1].x, -0.5)
    np.testing.assert_almost_equal(list1[1].y, 0.2)
    np.testing.assert_almost_equal(list1[2].x, 0.1)
    np.testing.assert_almost_equal(list1[2].y, 0.0)
    np.testing.assert_almost_equal(list1[3].x, 0.2)
    np.testing.assert_almost_equal(list1[3].y, -0.3)
    np.testing.assert_almost_equal(list1[4].x, -0.5)
    np.testing.assert_almost_equal(list1[4].y, 0.2)
    np.testing.assert_equal(list2, list1)

    sum1 = galsim.config.ParseValue(config,'sum1', config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(sum1.x, 0.2 - 0.5 + 0.1)
    np.testing.assert_almost_equal(sum1.y, -0.3 + 0.2 + 0.0)

    radec = galsim.config.ParseValue(config,'radec',config, galsim.CelestialCoord)[0]
    np.testing.assert_almost_equal(radec.ra / galsim.hours, 13.4)
    np.testing.assert_almost_equal(radec.dec / galsim.degrees, -0.3)
    radec2 = galsim.config.ParseValue(config,'list_radec',config, galsim.CelestialCoord)[0]
    np.testing.assert_almost_equal(radec2.ra / galsim.hours, 13.4)
    np.testing.assert_almost_equal(radec2.dec / galsim.degrees, -0.3)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad1',config, galsim.PositionD)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad2',config, galsim.PositionD)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad3',config, galsim.PositionD)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad4',config, galsim.PositionD)

    clean_config = galsim.config.CleanConfig(config)
    # Remove all current values
    galsim.config.RemoveCurrent(clean_config)
    # And a few extra things we added by hand.
    for key in ['obj_num', 'index_key', 'rng', 'image_num']:
        del clean_config[key]
    # And one extra thing that gets set as a default, but CleanConfig doesn't remove
    del clean_config['list1']['index']
    del clean_config['list2']['index']
    del clean_config['list_radec']  # this has a str->Angle conversion that isn't
    del orig_config['list_radec']   # worth trying to undo.  Just delete both.
    # Finally, these value got changed, so they won't match the original
    # unless we manually set them back to the original strings.
    clean_config['val2'] = '0.1, 0.2'
    clean_config['val4'] = '123.4, 567.8'
    clean_config['cur2'] = '@input.val2'
    clean_config['input']['val2'] = '0.3, 0.4'
    assert clean_config == orig_config

@timer
def test_table_value():
    """Test various ways to generate a LookupTable value
    """
    config = {
        'val1' : galsim.LookupTable([0,1,2,3], [0,10,10,0], interpolant='linear'),
        'file1' : { 'type' : 'File',
                    'file_name' : '../examples/data/cosmo-fid.zmed1.00.out' },
        'file2' : { 'type' : 'File',
                    'file_name' : '../examples/data/cosmo-fid.zmed1.00.out',
                    'interpolant' : 'linear', 'x_log' : True, 'f_log' : True },
        'file3' : { 'type' : 'File',
                    'file_name' : 'tree_ring_lookup.dat', 'amplitude' : 0.3 },
        'list1' : { 'type' : 'List',
                    'items' : [ galsim.LookupTable([0,1,2,3], [0,10,10,0], interpolant='linear'),
                                galsim.LookupTable([0,3], [0,10], interpolant='linear'),
                              ]
                  },
        'cur1' : { 'type' : 'Current', 'key' : 'file3' },
        'cur2' : { 'type' : 'Current', 'key' : 'list1.items.1' },
        'eval1' : '$galsim.LookupTable([0,1,2,3], [0,10,10,0], interpolant="linear")',
        'eval2' : '$galsim.LookupTable.from_file(@file1.file_name)',
    }

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, galsim.LookupTable)[0]
    assert val1 == config['val1']

    # You can also give None as the value type, which just returns whatever is in the dict.
    val1b  = galsim.config.ParseValue(config,'val1',config, None)[0]
    assert val1b == config['val1']

    # Test from file
    file1 = galsim.config.ParseValue(config,'file1',config, galsim.LookupTable)[0]
    assert file1 == galsim.LookupTable.from_file('../examples/data/cosmo-fid.zmed1.00.out')

    file2 = galsim.config.ParseValue(config,'file2',config, galsim.LookupTable)[0]
    assert file2 == galsim.LookupTable.from_file('../examples/data/cosmo-fid.zmed1.00.out',
                                                 interpolant='linear', x_log=True, f_log=True)

    file3 = galsim.config.ParseValue(config,'file3',config, galsim.LookupTable)[0]
    assert file3 == galsim.LookupTable.from_file('tree_ring_lookup.dat', amplitude=0.3)

    # Test values taken from a List
    list1 = []
    config['index_key'] = 'obj_num'
    for k in range(2):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, galsim.LookupTable)[0])
    assert list1 == config['list1']['items']

    # Test Current
    cur1 = galsim.config.ParseValue(config,'cur1',config, galsim.LookupTable)[0]
    assert cur1 == file3
    cur2 = galsim.config.ParseValue(config,'cur2',config, galsim.LookupTable)[0]
    assert cur2 == list1[1]

    # Test Eval
    eval1 = galsim.config.ParseValue(config,'eval1',config, galsim.LookupTable)[0]
    assert eval1 == val1
    eval2 = galsim.config.ParseValue(config,'eval2',config, galsim.LookupTable)[0]
    assert eval2 == file1

@timer
def test_eval():
    """Test various ways that we evaluate a string as a function or value
    """
    config = {
        # The basic calculation
        'eval1' : { 'type' : 'Eval', 'str' : 'np.exp(-0.5 * 1.8**2)' },
        # Different ways to get variables
        'eval2' : { 'type' : 'Eval', 'str' : 'np.exp(-0.5 * x**2)', 'fx' : 1.8 },
        'eval3' : { 'type' : 'Eval', 'str' : 'np.exp(-y**2 / two) if maybe else 0.' },
        # Make sure to use all valid letter prefixes here...
        'eval_variables' : { 'fy' : 1.8, 'bmaybe' : True, 'itwo' : 2, 'shalf' : '0.5',
                             'atheta' : 1.8 * galsim.radians,
                             'ppos' : galsim.PositionD(1.8,0),
                             'ccoord' : galsim.CelestialCoord(1.8*galsim.radians,0*galsim.radians),
                             'gshear' : galsim.Shear(g1=0.5, g2=0),
                             'ttable' : galsim.LookupTable([0,1,2,3], [0,1.8,1.8,0],
                                                           interpolant='linear'),
                             'ddct' : { 'a' : 1, 'b' : 2 },
                             'llst' : [ 1.5, 1.0, 0.5 ],
                             'xlit_two' : 2,
                             'qlength' : 1.8*u.m,
                             'upint' : u.imperial.pint,
                           },
        # Shorthand notation with $
        'eval4' : '$np.exp(-0.5 * y**2)',
        # math and numpy should also work
        'eval5' : '$numpy.exp(-0.5 * y**2)',
        'eval6' : '$math.exp(-0.5 * y**2)',
        # Use variables that are automatically defined
        'eval7' : '$np.exp(-0.5 * image_pos.x**2)',
        'eval8' : '$np.exp(-0.5 * world_pos.y**2)',
        'eval9' : '$np.exp(-0.5 * pixel_scale**2)',
        'eval10' : '$np.exp(-0.5 * (image_xsize / 100.)**2)',
        'eval11' : '$np.exp(-0.5 * (image_ysize / 200.)**2)',
        'eval12' : '$np.exp(-0.5 * (stamp_xsize / 20.)**2)',
        'eval13' : '$np.exp(-0.5 * (stamp_ysize / 20.)**2)',
        'eval14' : '$np.exp(-0.5 * (image_bounds.xmax / 100.)**2)',
        'eval15' : '$np.exp(-0.5 * ((image_center.y-0.5) / 100.)**2)',
        'eval16' : '$np.exp(-0.5 * wcs.scale**2)',
        # Shorthand notation with @
        'psf' : { 'type' : 'Gaussian', 'sigma' : 1.8 },
        'eval17' : '$np.exp(-@psf.sigma**2 / @eval_variables.itwo)',
        # A couple more to cover the other various letter prefixes.
        'eval18' : { 'type' : 'Eval', 'str' : 'np.exp(-eval(half) * theta.rad**lit_two)' },
        'eval19' : { 'type' : 'Eval', 'str' : 'np.exp(-shear.g1 * pos.x * coord.ra.rad)' },
        'eval20' : { 'type' : 'Eval', 'str' : 'np.exp(-lst[2] * table(1.5)**dct["b"])' },
        # Can access the input object as a current.
        'eval21' : { 'type' : 'Eval', 'str' : 'np.exp(-0.5 * ((@input.catalog).nobjects*0.6)**2)' },
        'eval22' : { 'type' : 'Eval', 'str' : 'np.exp(-0.5 * (@input.dict["f"]*18)**2)' },
        'eval23' : { 'type' : 'Eval', 'str' : 'np.exp(-pint/u.imperial.quart * length.to_value(u.m)**2)' },

        # Some that raise exceptions
        'bad1' : { 'type' : 'Eval', 'str' : 'npexp(-0.5)' },
        'bad2' : { 'type' : 'Eval', 'str' : 'np.exp(-0.5 * x**2)', 'x' : 1.8 },
        'bad3' : { 'type' : 'Eval', 'str' : 'np.exp(-0.5 * x**2)', 'wx' : 1.8 },
        'bad4' : { 'type' : 'Eval', 'str' : 'np.exp(-0.5 * q**2)', 'fx' : 1.8 },
        'bad5' : { 'type' : 'Eval', 'eval_str' : 'np.exp(-0.5 * x**2)', 'fx' : 1.8 },

        # Check that a list can be made using Eval
        'list0' : [0,1,2,3,4,5],
        'list1' : '$np.arange(6)',
        'list2' : (0,1,2,3,4,5),
        'list3' : '$(0,1,2,3,4,5)',

        # Check that a dict can be made using Eval
        'dict0' : {0:'h', 1:'e', 2:'l', 3:'l', 4:'o'},
        'dict1' : dict(enumerate("hello")),
        'dict2' : '$dict(enumerate("hello"))',
        'dict3' : '${ k:v for k,v in zip(np.arange(5), "hello") }',

        # These would be set by config in real runs, but just add them here for the tests.
        'image_pos' : galsim.PositionD(1.8,13),
        'world_pos' : galsim.PositionD(7.2,1.8),
        'uv_pos' : galsim.PositionD(7.2,1.8),
        'pixel_scale' : 1.8,
        'image_xsize' : 180,
        'image_ysize' : 360,
        'stamp_xsize' : 36,
        'stamp_ysize' : 36,
        'image_bounds' : galsim.BoundsI(1,180,1,360),
        'image_center' : galsim.PositionD(90.5, 180.5),
        'wcs' : galsim.PixelScale(1.8),

        'input' : { 'catalog' : { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                    'dict' : { 'dir' : 'config_input', 'file_name' : 'dict.yaml' },
                  },
     }

    galsim.config.ProcessInput(config)
    true_val = np.exp(-0.5 * 1.8**2)  # All of these should equal this value.
    for i in range(1,24):
        test_val = galsim.config.ParseValue(config, 'eval%d'%i, config, float)[0]
        print('i = ',i, 'val = ',test_val,true_val)
        np.testing.assert_almost_equal(test_val, true_val)

    # Doing it again uses saved _value and _fn
    galsim.config.RemoveCurrent(config)
    for i in range(1,24):
        test_val = galsim.config.ParseValue(config, 'eval%d'%i, config, float)[0]
        print('i = ',i, 'val = ',test_val,true_val)
        np.testing.assert_almost_equal(test_val, true_val)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad1',config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad2',config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad3',config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad4',config, float)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'bad5',config, float)
    config['eval_variables'] = 'itwo'
    config = galsim.config.CleanConfig(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'eval3',config, float)
    del config['eval_variables']
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.ParseValue(config,'eval3',config, float)

    # Check ways of making a list
    for i in range(4):
        test_list = galsim.config.ParseValue(config, 'list%d'%i, config, list)[0]
        print(test_list)
        np.testing.assert_array_equal(test_list, np.arange(6))

    # Check ways of making a dict
    for i in range(4):
        test_dict = galsim.config.ParseValue(config, 'dict%d'%i, config, dict)[0]
        print(test_dict)
        np.testing.assert_array_equal(test_dict, dict(enumerate('hello')))

    # Test the evaluation in RandomDistribution
    # Example config taken directly from Issue #776:
    config['shear'] = {
        'type': 'GBeta',
        'g': {
            'type': 'RandomDistribution',
            'function': "(1-x**2)**2 * np.exp( -0.5 * x**2 / 0.2**2 )",
            'x_min': 0.0,
            'x_max': 1.0,
        },
        'beta': {
            'type': 'Random'
        }
    }

    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234)

    dd = galsim.DistDeviate(rng, function=lambda x: (1-x**2)**2 * np.exp(-0.5*x**2/0.2**2),
                            x_min=0., x_max=1.)
    for k in range(6):
        config['obj_num'] = k
        shear1 = galsim.config.ParseValue(config,'shear',config, galsim.Shear)[0]
        shear2 = galsim.Shear(beta=rng()*360.*galsim.degrees, g=dd()) # order matters here.
        print('k = ',k,'shear1 = ',shear1,'shear2 = ',shear2)
        np.testing.assert_almost_equal(shear1.g1, shear2.g1)
        np.testing.assert_almost_equal(shear1.g2, shear2.g2)

    # Should also work using numpy or math instead of np
    config['shear']['g']['function'] = "(1-x**2)**2 * numpy.exp( -0.5 * x**2 / 0.2**2 )"
    for k in range(6):
        config['obj_num'] = k
        shear1 = galsim.config.ParseValue(config,'shear',config, galsim.Shear)[0]
        shear2 = galsim.Shear(beta=rng()*360.*galsim.degrees, g=dd()) # order matters here.
        print('k = ',k,'shear1 = ',shear1,'shear2 = ',shear2)
        np.testing.assert_almost_equal(shear1.g1, shear2.g1)
        np.testing.assert_almost_equal(shear1.g2, shear2.g2)

    config['shear']['g']['function'] = "(1-x**2)**2 * math.exp( -0.5 * x**2 / 0.2**2 )"
    for k in range(6):
        config['obj_num'] = k
        shear1 = galsim.config.ParseValue(config,'shear',config, galsim.Shear)[0]
        shear2 = galsim.Shear(beta=rng()*360.*galsim.degrees, g=dd()) # order matters here.
        print('k = ',k,'shear1 = ',shear1,'shear2 = ',shear2)
        np.testing.assert_almost_equal(shear1.g1, shear2.g1)
        np.testing.assert_almost_equal(shear1.g2, shear2.g2)

    # PowerSpectrum evaluates e_power_function and b_power_function, so check those.
    config['input'] = {
        'power_spectrum' :  [
            {
                'e_power_function' : 'np.exp(-k**0.2)',
                'b_power_function' : 'np.exp(-k**1.2)',
                'grid_spacing' : 10
            },
            {
                'e_power_function' : 'numpy.exp(-k**0.2)',
                'b_power_function' : 'numpy.exp(-k**1.2)',
                'grid_spacing' : 10,
                'index' : { 'type' : 'Sequence', 'repeat' : 3 }
            },
            # math doesn't work for acting on k, since it is a numpy array, but
            # we can check that math.sqrt works.
            {
                'e_power_function' : 'np.exp(-k ** math.sqrt(0.04))',
                'b_power_function' : 'np.exp(-k ** math.log(math.exp(1.2)))',
                'grid_spacing' : 10,
                'variance' : 0.05,
            },
        ]
    }
    config['ps_shear'] = { 'type' : 'PowerSpectrumShear' }
    config['ps_mu'] = { 'type' : 'PowerSpectrumMagnification' }
    config['ps_shear1'] = { 'type' : 'PowerSpectrumShear', 'num' : 1 }
    config['ps_mu1'] = { 'type' : 'PowerSpectrumMagnification', 'num' : 1 }
    config['ps_shear2'] = { 'type' : 'PowerSpectrumShear', 'num' : 2 }
    config['ps_mu2'] = { 'type' : 'PowerSpectrumMagnification', 'num' : 2 }

    config['index_key'] = 'file_num'
    config['file_num'] = 0
    config['image'] = { 'random_seed' : 1234 }
    rng = galsim.BaseDeviate(galsim.BaseDeviate(1234).raw())
    galsim.config.ProcessInput(config)
    galsim.config.SetupInputsForImage(config, None)
    ps = galsim.PowerSpectrum(e_power_function = lambda k: np.exp(-k**0.2),
                              b_power_function = lambda k: np.exp(-k**1.2))
    # ngrid is calculated from the image size by config, which was setup above.
    grid_spacing = 10.
    ngrid = int(math.ceil(config['image_ysize'] * config['pixel_scale'] / grid_spacing)) + 1
    center = config['wcs'].toWorld(config['image_center'])
    ps.buildGrid(grid_spacing=10, ngrid=ngrid, center=center, rng=rng)
    g1,g2,mu = ps.getLensing(pos = config['world_pos'])
    ps_shear = galsim.config.ParseValue(config, 'ps_shear', config, galsim.Shear)[0]
    ps_mu = galsim.config.ParseValue(config, 'ps_mu', config, float)[0]
    print('num = 0')
    print(g1,g2,mu)
    print(ps_shear,ps_mu)
    np.testing.assert_almost_equal(ps_shear.g1, g1)
    np.testing.assert_almost_equal(ps_shear.g2, g2)
    np.testing.assert_almost_equal(ps_mu, mu)

    # Check use of numpy in the evaluated string
    rng2 = galsim.BaseDeviate(galsim.BaseDeviate(1234 + 31415).raw())
    ps.buildGrid(grid_spacing=10, ngrid=ngrid, center=center, rng=rng2)
    g1,g2,mu = ps.getLensing(pos = config['world_pos'])
    ps_shear = galsim.config.ParseValue(config, 'ps_shear1', config, galsim.Shear)[0]
    ps_mu = galsim.config.ParseValue(config, 'ps_mu1', config, float)[0]
    print('num = 1')
    print(g1,g2,mu)
    print(ps_shear,ps_mu)
    np.testing.assert_almost_equal(ps_shear.g1, g1)
    np.testing.assert_almost_equal(ps_shear.g2, g2)
    np.testing.assert_almost_equal(ps_mu, mu)

    # Check use of math in the evaluated string
    ps.buildGrid(grid_spacing=10, ngrid=ngrid, center=center, rng=rng, variance=0.05)
    g1,g2,mu = ps.getLensing(pos = config['world_pos'])
    ps_shear = galsim.config.ParseValue(config, 'ps_shear2', config, galsim.Shear)[0]
    ps_mu = galsim.config.ParseValue(config, 'ps_mu2', config, float)[0]
    print('num = 2')
    print(g1,g2,mu)
    print(ps_shear,ps_mu)
    np.testing.assert_almost_equal(ps_shear.g1, g1)
    np.testing.assert_almost_equal(ps_shear.g2, g2)
    np.testing.assert_almost_equal(ps_mu, mu)


def test_quantity():
    import astropy.units as u
    config = {
        'length': 1.0 * u.m,
        'length2': '1.0 m',
        'length3': '100.0 cm',
        'length4': {
            'type': 'Quantity',
            'value': 0.001,
            'unit': 'km',
        },
        'length5': {
            'type': 'Quantity',
            'value': 0.001,
            'unit': u.km,
        },
        'length6': '$1.0 * u.m',
        'length7': '10 kg',  # Not a length!
        'length8': {
            'type': 'Quantity',
            'value': {
                'type': 'Random',
                'min': 0.0,
                'max': 1.0,
            },
            'unit': 'm',
        },
        'length9': {
            'type': 'Sum',
            'items': [
                {
                    'type': 'Quantity',
                    'value': 1.0,
                    'unit': 'm',
                },
                {
                    'type': 'Quantity',
                    'value': 1.0,
                    'unit': 'cm',
                }
            ]
        },
        'length10': {
            'type': 'Current',
            'key': 'length',
        },
    }

    value, _ = galsim.config.ParseValue(config, 'length', config, u.Quantity)
    assert value == 1.0 * u.m
    value, _ = galsim.config.ParseValue(config, 'length2', config, u.Quantity)
    assert value == 1.0 * u.m
    value, _ = galsim.config.ParseValue(config, 'length3', config, u.Quantity)
    assert value == 1.0 * u.m
    value, _ = galsim.config.ParseValue(config, 'length4', config, u.Quantity)
    assert value == 1.0 * u.m
    value, _ = galsim.config.ParseValue(config, 'length5', config, u.Quantity)
    assert value == 1.0 * u.m
    value, _ = galsim.config.ParseValue(config, 'length6', config, u.Quantity)
    assert value == 1.0 * u.m
    # We can demand a Quantity, but there's currently no way to demand a
    # particular dimensionality.  So this one just yields a mass.
    value, _ = galsim.config.ParseValue(config, 'length7', config, u.Quantity)
    assert value == 10 * u.kg
    value, _ = galsim.config.ParseValue(config, 'length8', config, u.Quantity)
    assert 0 <= value.value <= 1.0
    assert value.unit == u.m
    value, _ = galsim.config.ParseValue(config, 'length9', config, u.Quantity)
    assert value == 1.01 * u.m
    value, _ = galsim.config.ParseValue(config, 'length10', config, u.Quantity)
    assert value == 1.0 * u.m


def test_astropy_unit():
    import astropy.units as u
    config = {
        'mass1': u.kg,
        'mass2': 'kg',
        'mass3': '$u.kg',
        'mass4': {
            'type': 'Unit',
            'unit': 'kg',
        },
        'area1': 'm^2',
        'area2': '$u.m * u.m',
        'area3': '$u.m**2'
    }

    for k in ['mass1', 'mass2', 'mass3', 'mass4']:
        value, _ = galsim.config.ParseValue(config, k, config, u.Unit)
        assert value == u.kg

    for k in ['area1', 'area2', 'area3']:
        value, _ = galsim.config.ParseValue(config, k, config, u.Unit)
        assert value == u.m**2


if __name__ == "__main__":
    runtests(__file__)
