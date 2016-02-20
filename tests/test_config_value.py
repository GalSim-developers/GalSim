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
import math

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


def test_float_value():
    """Test various ways to generate a float value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ],
                    'dict' : [ 
                        { 'dir' : 'config_input', 'file_name' : 'dict.p' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.json' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.yaml' } ] },

        'val1' : 9.9,
        'val2' : int(400),
        'str1' : '8.73',
        'str2' : '2.33e-9',
        'str3' : '6.e-9', 
        'cat1' : { 'type' : 'Catalog' , 'col' : 0 },
        'cat2' : { 'type' : 'Catalog' , 'col' : 1 },
        'cat3' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'float1' },
        'cat4' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'float2' },
        'ran1' : { 'type' : 'Random', 'min' : 0.5, 'max' : 3 },
        'ran2' : { 'type' : 'Random', 'min' : -5, 'max' : 0 },
        'gauss1' : { 'type' : 'RandomGaussian', 'sigma' : 1 },
        'gauss2' : { 'type' : 'RandomGaussian', 'sigma' : 3, 'mean' : 4 },
        'gauss3' : { 'type' : 'RandomGaussian', 'sigma' : 1.5, 'min' : -2, 'max' : 2 },
        'gauss4' : { 'type' : 'RandomGaussian', 'sigma' : 0.5, 'min' : 0, 'max' : 0.8 },
        'gauss5' : { 'type' : 'RandomGaussian',
                     'sigma' : 0.3, 'mean' : 0.5, 'min' : 0, 'max' : 0.5 },
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
        'seq5' : { 'type' : 'Sequence', 'first' : 1, 'last' : 2.1, 'repeat' : 2 },
        'list1' : { 'type' : 'List', 'items' : [ 73, 8.9, 3.14 ] },
        'list2' : { 'type' : 'List',
                    'items' : [ 0.6, 1.8, 2.1, 3.7, 4.3, 5.5, 6.1, 7.0, 8.6, 9.3, 10.8, 11.2 ],
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -3 } },
        'dict1' : { 'type' : 'Dict', 'key' : 'f' },
        'dict2' : { 'type' : 'Dict', 'num' : 1, 'key' : 'f' },
        'dict3' : { 'type' : 'Dict', 'num' : 2, 'key' : 'f' },
        'dict4' : { 'type' : 'Dict', 'num' : 2, 'key' : 'noise.models.1.gain' },
        'sum1' : { 'type' : 'Sum', 'items' : [ 72, '2.33', { 'type' : 'Dict', 'key' : 'f' } ] }
    }

    test_yaml = True
    try:
        galsim.config.ProcessInput(config)
    except:
        # We don't require PyYAML as a dependency, so if this fails, just remove the YAML dict.
        del config['input']['dict'][2]
        galsim.config.ProcessInput(config)
        test_yaml = False

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, float)[0]
    np.testing.assert_almost_equal(val1, 9.9)

    val2 = galsim.config.ParseValue(config,'val2',config, float)[0]
    np.testing.assert_almost_equal(val2, 400)

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
                               # from thinking current_val is still current.
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
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, float)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, float)[0])

    np.testing.assert_array_almost_equal(list1, [ 73, 8.9, 3.14, 73, 8.9 ])
    np.testing.assert_array_almost_equal(list2, [ 10.8, 7.0, 4.3, 1.8, 10.8 ])

    # Test values read from a Dict
    dict = []
    dict.append(galsim.config.ParseValue(config,'dict1',config, float)[0])
    dict.append(galsim.config.ParseValue(config,'dict2',config, float)[0])
    if test_yaml:
        dict.append(galsim.config.ParseValue(config,'dict3',config, float)[0])
        dict.append(galsim.config.ParseValue(config,'dict4',config, float)[0])
    else:
        dict.append(0.1)
        dict.append(1.9)
    np.testing.assert_array_almost_equal(dict, [ 23.17, -17.23, 0.1, 1.9 ])

    sum1 = galsim.config.ParseValue(config,'sum1',config, float)[0]
    np.testing.assert_almost_equal(sum1, 72 + 2.33 + 23.17)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



def test_int_value():
    """Test various ways to generate an int value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ],
                    'dict' : [ 
                        { 'dir' : 'config_input', 'file_name' : 'dict.p' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.json' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.yaml' } ] },

        'val1' : 9,
        'val2' : float(8.7),  # Reading as int will drop the fraction.
        'val3' : -400.8,      # Not floor - negatives will round up.
        'str1' : '8',
        'str2' : '-2',
        'cat1' : { 'type' : 'Catalog' , 'col' : 2 },
        'cat2' : { 'type' : 'Catalog' , 'col' : 3 },
        'cat3' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'int1' },
        'cat4' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'int2' },
        'ran1' : { 'type' : 'Random', 'min' : 0, 'max' : 3 },
        'ran2' : { 'type' : 'Random', 'min' : -5, 'max' : 10 },
        'dev1' : { 'type' : 'RandomPoisson', 'mean' : 137 },
        'dev2' : { 'type' : 'RandomBinomial', 'N' : 17 },
        'dev3' : { 'type' : 'RandomBinomial', 'N' : 17, 'p' : 0.2 },
        'seq1' : { 'type' : 'Sequence' },
        'seq2' : { 'type' : 'Sequence', 'step' : 3 },
        'seq3' : { 'type' : 'Sequence', 'first' : 1, 'step' : 5 },
        'seq4' : { 'type' : 'Sequence', 'first' : 10, 'step' : -2 },
        'seq5' : { 'type' : 'Sequence', 'first' : 1, 'last' : 2, 'repeat' : 2 },
        'seq_file' : { 'type' : 'Sequence', 'index_key' : 'file_num' },
        'seq_image' : { 'type' : 'Sequence', 'index_key' : 'image_num' },
        'seq_obj' : { 'type' : 'Sequence', 'index_key' : 'obj_num' },
        'seq_obj2' : { 'type' : 'Sequence', 'index_key' : 'obj_num_in_file' },
        'list1' : { 'type' : 'List', 'items' : [ 73, 8, 3 ] },
        'list2' : { 'type' : 'List',
                    'items' : [ 6, 8, 1, 7, 3, 5, 1, 0, 6, 3, 8, 2 ],
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -3 } },
        'dict1' : { 'type' : 'Dict', 'key' : 'i' },
        'dict2' : { 'type' : 'Dict', 'num' : 1, 'key' : 'i' },
        'dict3' : { 'type' : 'Dict', 'num' : 2, 'key' : 'i' },
        'sum1' : { 'type' : 'Sum', 'items' : [ 72.3, '2', { 'type' : 'Dict', 'key' : 'i' } ] }
    }

    test_yaml = True
    try:
        galsim.config.ProcessInput(config)
    except:
        # We don't require PyYAML as a dependency, so if this fails, just remove the YAML dict.
        del config['input']['dict'][2]
        galsim.config.ProcessInput(config)
        test_yaml = False

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, int)[0]
    np.testing.assert_equal(val1, 9)

    val2 = galsim.config.ParseValue(config,'val2',config, int)[0]
    np.testing.assert_equal(val2, 8)

    val3 = galsim.config.ParseValue(config,'val3',config, int)[0]
    np.testing.assert_equal(val3, -400)

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

    # Test values read from a Dict
    dict = []
    dict.append(galsim.config.ParseValue(config,'dict1',config, int)[0])
    dict.append(galsim.config.ParseValue(config,'dict2',config, int)[0])
    if test_yaml:
        dict.append(galsim.config.ParseValue(config,'dict3',config, int)[0])
    else:
        dict.append(1)
    np.testing.assert_array_equal(dict, [ 17, -23, 1 ])
 
    sum1 = galsim.config.ParseValue(config,'sum1', config, int)[0]
    np.testing.assert_almost_equal(sum1, 72 + 2 + 17)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



def test_bool_value():
    """Test various ways to generate a bool value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ],
                    'dict' : [ 
                        { 'dir' : 'config_input', 'file_name' : 'dict.p' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.json' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.yaml' } ] },

        'val1' : True,
        'val2' : 1,
        'val3' : 0.0,
        'str1' : 'true',
        'str2' : '0',
        'str3' : 'yes',
        'str4' : 'No',
        'cat1' : { 'type' : 'Catalog' , 'col' : 4 },
        'cat2' : { 'type' : 'Catalog' , 'col' : 5 },
        'cat3' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'bool1' },
        'cat4' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'bool2' },
        'ran1' : { 'type' : 'Random' },
        'dev1' : { 'type' : 'RandomBinomial', 'N' : 1 },
        'dev2' : { 'type' : 'RandomBinomial', 'N' : 1, 'p' : 0.5 },
        'dev3' : { 'type' : 'RandomBinomial', 'p' : 0.2 },
        'seq1' : { 'type' : 'Sequence' },
        'seq2' : { 'type' : 'Sequence', 'first' : True, 'repeat' : 2 },
        'list1' : { 'type' : 'List', 'items' : [ 'yes', 'no', 'no' ] },
        'list2' : { 'type' : 'List',
                    'items' : [ 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0 ],
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -3 } },
        'dict1' : { 'type' : 'Dict', 'key' : 'b' },
        'dict2' : { 'type' : 'Dict', 'num' : 1, 'key' : 'b' },
        'dict3' : { 'type' : 'Dict', 'num' : 2, 'key' : 'b' }
    }

    test_yaml = True
    try:
        galsim.config.ProcessInput(config)
    except:
        # We don't require PyYAML as a dependency, so if this fails, just remove the YAML dict.
        del config['input']['dict'][2]
        galsim.config.ProcessInput(config)
        test_yaml = False

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, bool)[0]
    np.testing.assert_equal(val1, True)

    val2 = galsim.config.ParseValue(config,'val2',config, bool)[0]
    np.testing.assert_equal(val2, True)

    val3 = galsim.config.ParseValue(config,'val3',config, bool)[0]
    np.testing.assert_equal(val3, False)

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
    if test_yaml:
        dict.append(galsim.config.ParseValue(config,'dict3',config, bool)[0])
    else:
        dict.append(False)
    np.testing.assert_array_equal(dict, [ True, False, False ])
 
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



def test_str_value():
    """Test various ways to generate a str value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ],
                    'dict' : [ 
                        { 'dir' : 'config_input', 'file_name' : 'dict.p' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.json' },
                        { 'dir' : 'config_input', 'file_name' : 'dict.yaml' } ] },

        'val1' : -93,
        'val2' : True,
        'val3' : 123.8,
        'str1' : "Norwegian",
        'str2' : u"Blue",
        'cat1' : { 'type' : 'Catalog' , 'col' : 6 },
        'cat2' : { 'type' : 'Catalog' , 'col' : 7 },
        'cat3' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'str1' },
        'cat4' : { 'type' : 'Catalog' , 'num' : 1, 'col' : 'str2' },
        'list1' : { 'type' : 'List', 'items' : [ 'Beautiful', 'plumage!', 'Ay?' ] },
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
        'dict3' : { 'type' : 'Dict', 'num' : 2, 'key' : 's' }
    }

    test_yaml = True
    try:
        galsim.config.ProcessInput(config)
    except:
        # We don't require PyYAML as a dependency, so if this fails, just remove the YAML dict.
        del config['input']['dict'][2]
        galsim.config.ProcessInput(config)
        test_yaml = False

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, str)[0]
    np.testing.assert_equal(val1, '-93')

    val2 = galsim.config.ParseValue(config,'val2',config, str)[0]
    np.testing.assert_equal(val2, 'True')

    val3 = galsim.config.ParseValue(config,'val3',config, str)[0]
    np.testing.assert_equal(val3, '123.8')

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
    config['index_key'] = 'image_num'
    for k in range(5):
        config['image_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, str)[0])

    np.testing.assert_array_equal(list1, ['Beautiful', 'plumage!', 'Ay?', 'Beautiful', 'plumage!'])

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
    if test_yaml:
        dict.append(galsim.config.ParseValue(config,'dict3',config, str)[0])
    else:
        dict.append('Brian')
    np.testing.assert_array_equal(dict, [ 'Life', 'of', 'Brian' ])
 
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_angle_value():
    """Test various ways to generate an Angle value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : [
                        { 'dir' : 'config_input', 'file_name' : 'catalog.txt' },
                        { 'dir' : 'config_input', 'file_name' : 'catalog.fits' } ] },

        'val1' : 1.9 * galsim.radians,
        'val2' : -41 * galsim.degrees,
        'str1' : '0.73 radians',
        'str2' : '240 degrees',
        'str3' : '1.2 rad',
        'str4' : '45 deg',
        'str5' : '6 hrs',
        'str6' : '21 hour',
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
        'seq1' : { 'type' : 'Rad', 'theta' : { 'type' : 'Sequence' } },
        'seq2' : { 'type' : 'Deg', 'theta' : { 'type' : 'Sequence', 'first' : 45, 'step' : 80 } },
        'list1' : { 'type' : 'List',
                    'items' : [ 73 * galsim.arcmin,
                                8.9 * galsim.arcmin,
                                3.14 * galsim.arcmin ] },
        'sum1' : { 'type' : 'Sum', 'items' : [ 72 * galsim.degrees, '2.33 degrees' ] }
    }

    galsim.config.ProcessInput(config)

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(val1.rad(), 1.9)

    val2 = galsim.config.ParseValue(config,'val2',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(val2.rad(), -41 * math.pi/180)

    # Test conversions from strings
    str1 = galsim.config.ParseValue(config,'str1',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str1.rad(), 0.73)

    str2 = galsim.config.ParseValue(config,'str2',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str2 / galsim.degrees, 240)

    str3 = galsim.config.ParseValue(config,'str3',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str3.rad(), 1.2)

    str4 = galsim.config.ParseValue(config,'str4',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str4.rad(), math.pi/4)

    str5 = galsim.config.ParseValue(config,'str5',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str5.rad(), math.pi/2)

    str6 = galsim.config.ParseValue(config,'str6',config, galsim.Angle)[0]
    np.testing.assert_almost_equal(str6.rad(), 7*math.pi/4)

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
        cat1.append(galsim.config.ParseValue(config,'cat1',config, galsim.Angle)[0].rad())
        cat2.append(galsim.config.ParseValue(config,'cat2',config, galsim.Angle)[0]/galsim.degrees)
        cat3.append(galsim.config.ParseValue(config,'cat3',config, galsim.Angle)[0].rad())
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
        theta = rng() * 2 * math.pi
        np.testing.assert_almost_equal(ran1.rad(), theta)

    # Test values generated from a Sequence
    seq1 = []
    seq2 = []
    config['index_key'] = 'obj_num'
    for k in range(6):
        config['obj_num'] = k
        seq1.append(galsim.config.ParseValue(config,'seq1',config, galsim.Angle)[0].rad())
        seq2.append(galsim.config.ParseValue(config,'seq2',config, galsim.Angle)[0]/galsim.degrees)

    np.testing.assert_array_almost_equal(seq1, [ 0, 1, 2, 3, 4, 5 ])
    np.testing.assert_array_almost_equal(seq2, [ 45, 125, 205, 285, 365, 445 ])

    # Test values taken from a List
    list1 = []
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, galsim.Angle)[0]/galsim.arcmin)

    np.testing.assert_array_almost_equal(list1, [ 73, 8.9, 3.14, 73, 8.9 ])

    sum1 = galsim.config.ParseValue(config,'sum1', config, galsim.Angle)[0]
    np.testing.assert_almost_equal(sum1 / galsim.degrees, 72 + 2.33)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shear_value():
    """Test various ways to generate a Shear value
    """
    import time
    t1 = time.time()

    config = {
        'val1' : galsim.Shear(g1=0.2, g2=0.3),
        'val2' : galsim.Shear(e1=0.1),
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
        'sum1' : { 'type' : 'Sum', 
                  'items' : [ galsim.Shear(g1 = 0.2, g2 = -0.3),
                              galsim.Shear(g1 = -0.5, g2 = 0.2),
                              galsim.Shear(g1 = 0.1, g2 = 0.0) ] }
    }

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(val1.getG1(), 0.2)
    np.testing.assert_almost_equal(val1.getG2(), 0.3)

    val2 = galsim.config.ParseValue(config,'val2',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(val2.getE1(), 0.1)
    np.testing.assert_almost_equal(val2.getE2(), 0.)

    # Test various direct types
    s1 = galsim.config.ParseValue(config,'s1',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s1.getE1(), 0.5)
    np.testing.assert_almost_equal(s1.getE2(), -0.1)

    s2 = galsim.config.ParseValue(config,'s2',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s2.getE(), 0.5)
    np.testing.assert_almost_equal(s2.getBeta().rad(), 0.1)

    s3 = galsim.config.ParseValue(config,'s3',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s3.getG1(), 0.5)
    np.testing.assert_almost_equal(s3.getG2(), -0.1)

    s4 = galsim.config.ParseValue(config,'s4',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s4.getG(), 0.5)
    np.testing.assert_almost_equal(s4.getBeta().rad(), 0.1)

    s5 = galsim.config.ParseValue(config,'s5',config, galsim.Shear)[0]
    eta = s5.getEta()
    e = s5.getE()
    eta1 = s5.getE1() * eta/e
    eta2 = s5.getE2() * eta/e
    np.testing.assert_almost_equal(eta1, 0.5)
    np.testing.assert_almost_equal(eta2, -0.1)

    s6 = galsim.config.ParseValue(config,'s6',config, galsim.Shear)[0]
    np.testing.assert_almost_equal(s6.getEta(), 0.5)
    np.testing.assert_almost_equal(s6.getBeta().rad(), 0.1)

    s7 = galsim.config.ParseValue(config,'s7',config, galsim.Shear)[0]
    g = s7.getG()
    q = (1-g)/(1+g)
    np.testing.assert_almost_equal(q, 0.5)
    np.testing.assert_almost_equal(s7.getBeta().rad(), 0.1)

    # Test values taken from a List
    list1 = []
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, galsim.Shear)[0])

    np.testing.assert_almost_equal(list1[0].getG1(), 0.2)
    np.testing.assert_almost_equal(list1[0].getG2(), -0.3)
    np.testing.assert_almost_equal(list1[1].getG1(), -0.5)
    np.testing.assert_almost_equal(list1[1].getG2(), 0.2)
    np.testing.assert_almost_equal(list1[2].getG1(), 0.1)
    np.testing.assert_almost_equal(list1[2].getG2(), 0.0)
    np.testing.assert_almost_equal(list1[3].getG1(), 0.2)
    np.testing.assert_almost_equal(list1[3].getG2(), -0.3)
    np.testing.assert_almost_equal(list1[4].getG1(), -0.5)
    np.testing.assert_almost_equal(list1[4].getG2(), 0.2)

    sum1 = galsim.config.ParseValue(config,'sum1', config, galsim.Shear)[0]
    s = galsim.Shear(g1=0.2, g2=-0.3)
    s += galsim.Shear(g1=-0.5, g2=0.2)
    s += galsim.Shear(g1=0.1, g2=0.0)
    np.testing.assert_almost_equal(sum1.getG1(), s.getG1())
    np.testing.assert_almost_equal(sum1.getG2(), s.getG2())

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_pos_value():
    """Test various ways to generate a Position value
    """
    import time
    t1 = time.time()

    config = {
        'val1' : galsim.PositionD(0.1,0.2),
        'xy1' : { 'type' : 'XY', 'x' : 1.3, 'y' : 2.4 },
        'ran1' : { 'type' : 'RandomCircle', 'radius' : 3 },
        'list1' : { 'type' : 'List', 
                    'items' : [ galsim.PositionD(0.2, -0.3),
                                galsim.PositionD(-0.5, 0.2),
                                galsim.PositionD(0.1, 0.0) ] },
        'sum1' : { 'type' : 'Sum', 
                   'items' : [ galsim.PositionD(0.2, -0.3),
                               galsim.PositionD(-0.5, 0.2),
                               galsim.PositionD(0.1, 0.0) ] }
    }

    # Test direct values
    val1 = galsim.config.ParseValue(config,'val1',config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(val1.x, 0.1)
    np.testing.assert_almost_equal(val1.y, 0.2)

    xy1 = galsim.config.ParseValue(config,'xy1',config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(xy1.x, 1.3)
    np.testing.assert_almost_equal(xy1.y, 2.4)

    # Test values generated from a uniform deviate
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

    # Test values taken from a List
    list1 = []
    config['index_key'] = 'obj_num'
    for k in range(5):
        config['obj_num'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, galsim.PositionD)[0])

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

    sum1 = galsim.config.ParseValue(config,'sum1', config, galsim.PositionD)[0]
    np.testing.assert_almost_equal(sum1.x, 0.2 - 0.5 + 0.1)
    np.testing.assert_almost_equal(sum1.y, -0.3 + 0.2 + 0.0)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_float_value()
    test_int_value()
    test_bool_value()
    test_str_value()
    test_angle_value()
    test_shear_value()
    test_pos_value()


