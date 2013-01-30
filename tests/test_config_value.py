# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#

import numpy as np
import os
import sys
import math

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def funcname():
    import inspect
    return inspect.stack()[1][3]


def test_float_value():
    """Test various ways to generate a float value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : { 'dir' : 'config_input', 'file_name' : 'catalog.txt' } },

        'val1' : 9.9,
        'val2' : int(400),
        'str1' : '8.73',
        'str2' : '2.33e-9',
        'str3' : '6.e-9', 
        'cat1' : { 'type' : 'InputCatalog' , 'col' : 0 },
        'cat2' : { 'type' : 'InputCatalog' , 'col' : 1 },
        'ran1' : { 'type' : 'Random', 'min' : 0.5, 'max' : 3 },
        'ran2' : { 'type' : 'Random', 'min' : -5, 'max' : 0 },
        'gauss1' : { 'type' : 'RandomGaussian', 'sigma' : 1 },
        'gauss2' : { 'type' : 'RandomGaussian', 'sigma' : 3, 'mean' : 4 },
        'gauss3' : { 'type' : 'RandomGaussian', 'sigma' : 1.5, 'min' : -2, 'max' : 2 },
        'gauss4' : { 'type' : 'RandomGaussian', 'sigma' : 0.5, 'min' : 0, 'max' : 0.8 },
        'gauss5' : { 'type' : 'RandomGaussian',
                     'sigma' : 0.3, 'mean' : 0.5, 'min' : 0, 'max' : 0.5 },
        'seq1' : { 'type' : 'Sequence' },
        'seq2' : { 'type' : 'Sequence', 'step' : 0.1 },
        'seq3' : { 'type' : 'Sequence', 'first' : 1.5, 'step' : 0.5 },
        'seq4' : { 'type' : 'Sequence', 'first' : 10, 'step' : -2 },
        'seq5' : { 'type' : 'Sequence', 'first' : 1, 'last' : 2.1, 'repeat' : 2 },
        'list1' : { 'type' : 'List', 'items' : [ 73, 8.9, 3.14 ] },
        'list2' : { 'type' : 'List',
                    'items' : [ 0.6, 1.8, 2.1, 3.7, 4.3, 5.5, 6.1, 7.0, 8.6, 9.3, 10.8, 11.2 ],
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -3 } }
    }

    galsim.config.ProcessInput(config)

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

    # Test values read from an InputCatalog
    input_cat = galsim.InputCatalog(dir='config_input', file_name='catalog.txt')
    cat1 = []
    cat2 = []
    for k in range(5):
        config['seq_index'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, float)[0])
        cat2.append(galsim.config.ParseValue(config,'cat2',config, float)[0])

    np.testing.assert_array_almost_equal(cat1, [ 1.234, 2.345, 3.456, 1.234, 2.345 ])
    np.testing.assert_array_almost_equal(cat2, [ 4.131, -900, 8000, 4.131, -900 ])

    # Test values generated from a uniform deviate
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        ran1 = galsim.config.ParseValue(config,'ran1',config, float)[0]
        np.testing.assert_almost_equal(ran1, rng() * 2.5 + 0.5)

        ran2 = galsim.config.ParseValue(config,'ran2',config, float)[0]
        np.testing.assert_almost_equal(ran2, rng() * 5 - 5)

    # Test values generated from a Gaussian deviate
    gd = galsim.GaussianDeviate(rng)
    for k in range(6):
        gauss1 = galsim.config.ParseValue(config,'gauss1',config, float)[0]
        gd.setMean(0)
        gd.setSigma(1)
        np.testing.assert_almost_equal(gauss1, gd())

        gauss2 = galsim.config.ParseValue(config,'gauss2',config, float)[0]
        gd.setMean(4)
        gd.setSigma(3)
        np.testing.assert_almost_equal(gauss2, gd())

        gauss3 = galsim.config.ParseValue(config,'gauss3',config, float)[0]
        gd.setMean(0)
        gd.setSigma(1.5)
        gd_val = gd()
        while math.fabs(gd_val) > 2:
            gd_val = gd()
        np.testing.assert_almost_equal(gauss3, gd_val)

        gauss4 = galsim.config.ParseValue(config,'gauss4',config, float)[0]
        gd.setMean(0)
        gd.setSigma(0.5)
        gd_val = math.fabs(gd())
        while gd_val > 0.8:
            gd_val = math.fabs(gd())
        np.testing.assert_almost_equal(gauss4, gd_val)

        gauss5 = galsim.config.ParseValue(config,'gauss5',config, float)[0]
        gd.setMean(0.5)
        gd.setSigma(0.3)
        gd_val = gd()
        if gd_val > 0.5: 
            gd_val = 1-gd_val
        while gd_val < 0:
            gd_val = gd()
            if gd_val > 0.5: 
                gd_val = 1-gd_val
        np.testing.assert_almost_equal(gauss5, gd_val)

    # Test values generated from a Sequence
    seq1 = []
    seq2 = []
    seq3 = []
    seq4 = []
    seq5 = []
    for k in range(6):
        config['seq_index'] = k
        seq1.append(galsim.config.ParseValue(config,'seq1',config, float)[0])
        seq2.append(galsim.config.ParseValue(config,'seq2',config, float)[0])
        seq3.append(galsim.config.ParseValue(config,'seq3',config, float)[0])
        seq4.append(galsim.config.ParseValue(config,'seq4',config, float)[0])
        seq5.append(galsim.config.ParseValue(config,'seq5',config, float)[0])

    np.testing.assert_array_almost_equal(seq1, [ 0, 1, 2, 3, 4, 5 ])
    np.testing.assert_array_almost_equal(seq2, [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ])
    np.testing.assert_array_almost_equal(seq3, [ 1.5, 2, 2.5, 3, 3.5, 4 ])
    np.testing.assert_array_almost_equal(seq4, [ 10, 8, 6, 4, 2, 0 ])
    np.testing.assert_array_almost_equal(seq5, [ 1, 1, 2, 2, 1, 1 ])

    # Test values taken from a List
    list1 = []
    list2 = []
    for k in range(5):
        config['seq_index'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, float)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, float)[0])

    np.testing.assert_array_almost_equal(list1, [ 73, 8.9, 3.14, 73, 8.9 ])
    np.testing.assert_array_almost_equal(list2, [ 10.8, 7.0, 4.3, 1.8, 10.8 ])

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



def test_int_value():
    """Test various ways to generate an int value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : { 'dir' : 'config_input', 'file_name' : 'catalog.txt' } },

        'val1' : 9,
        'val2' : float(8.7),  # Reading as int will drop the fraction.
        'val3' : -400.8,      # Not floor - negatives will round up.
        'str1' : '8',
        'str2' : '-2',
        'cat1' : { 'type' : 'InputCatalog' , 'col' : 2 },
        'cat2' : { 'type' : 'InputCatalog' , 'col' : 3 },
        'ran1' : { 'type' : 'Random', 'min' : 0, 'max' : 3 },
        'ran2' : { 'type' : 'Random', 'min' : -5, 'max' : 10 },
        'seq1' : { 'type' : 'Sequence' },
        'seq2' : { 'type' : 'Sequence', 'step' : 3 },
        'seq3' : { 'type' : 'Sequence', 'first' : 1, 'step' : 5 },
        'seq4' : { 'type' : 'Sequence', 'first' : 10, 'step' : -2 },
        'seq5' : { 'type' : 'Sequence', 'first' : 1, 'last' : 2, 'repeat' : 2 },
        'list1' : { 'type' : 'List', 'items' : [ 73, 8, 3 ] },
        'list2' : { 'type' : 'List',
                    'items' : [ 6, 8, 1, 7, 3, 5, 1, 0, 6, 3, 8, 2 ],
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -3 } }
    }

    galsim.config.ProcessInput(config)

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

    # Test values read from an InputCatalog
    input_cat = galsim.InputCatalog(dir='config_input', file_name='catalog.txt')
    cat1 = []
    cat2 = []
    for k in range(5):
        config['seq_index'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, int)[0])
        cat2.append(galsim.config.ParseValue(config,'cat2',config, int)[0])

    np.testing.assert_array_equal(cat1, [ 9, 0, -4, 9, 0 ])
    np.testing.assert_array_equal(cat2, [ -3, 8, 17, -3, 8 ])

    # Test values generated from a uniform deviate
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        ran1 = galsim.config.ParseValue(config,'ran1',config, int)[0]
        np.testing.assert_equal(ran1, int(math.floor(rng() * 4)))

        ran2 = galsim.config.ParseValue(config,'ran2',config, int)[0]
        np.testing.assert_equal(ran2, int(math.floor(rng() * 16))-5)

    # Test values generated from a Sequence
    seq1 = []
    seq2 = []
    seq3 = []
    seq4 = []
    seq5 = []
    for k in range(6):
        config['seq_index'] = k
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

    # Test values taken from a List
    list1 = []
    list2 = []
    for k in range(5):
        config['seq_index'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, int)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, int)[0])

    np.testing.assert_array_equal(list1, [ 73, 8, 3, 73, 8 ])
    np.testing.assert_array_equal(list2, [ 8, 0, 3, 8, 8 ])

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



def test_bool_value():
    """Test various ways to generate a bool value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : { 'dir' : 'config_input', 'file_name' : 'catalog.txt' } },

        'val1' : True,
        'val2' : 1,
        'val3' : 0.0,
        'str1' : 'true',
        'str2' : '0',
        'str3' : 'yes',
        'str4' : 'No',
        'cat1' : { 'type' : 'InputCatalog' , 'col' : 4 },
        'cat2' : { 'type' : 'InputCatalog' , 'col' : 5 },
        'ran1' : { 'type' : 'Random' },
        'seq1' : { 'type' : 'Sequence' },
        'seq2' : { 'type' : 'Sequence', 'first' : True, 'repeat' : 2 },
        'list1' : { 'type' : 'List', 'items' : [ 'yes', 'no', 'no' ] },
        'list2' : { 'type' : 'List',
                    'items' : [ 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0 ],
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -3 } }
    }

    galsim.config.ProcessInput(config)

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

    # Test values read from an InputCatalog
    input_cat = galsim.InputCatalog(dir='config_input', file_name='catalog.txt')
    cat1 = []
    cat2 = []
    for k in range(5):
        config['seq_index'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, bool)[0])
        cat2.append(galsim.config.ParseValue(config,'cat2',config, bool)[0])

    np.testing.assert_array_equal(cat1, [ 1, 0, 1, 1, 0 ])
    np.testing.assert_array_equal(cat2, [ 1, 0, 0, 1, 0 ])

    # Test values generated from a uniform deviate
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        ran1 = galsim.config.ParseValue(config,'ran1',config, bool)[0]
        np.testing.assert_equal(ran1, rng() < 0.5)

    # Test values generated from a Sequence
    seq1 = []
    seq2 = []
    for k in range(6):
        config['seq_index'] = k
        seq1.append(galsim.config.ParseValue(config,'seq1',config, bool)[0])
        seq2.append(galsim.config.ParseValue(config,'seq2',config, bool)[0])

    np.testing.assert_array_equal(seq1, [ 0, 1, 0, 1, 0, 1 ])
    np.testing.assert_array_equal(seq2, [ 1, 1, 0, 0, 1, 1 ])

    # Test values taken from a List
    list1 = []
    list2 = []
    for k in range(5):
        config['seq_index'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, bool)[0])
        list2.append(galsim.config.ParseValue(config,'list2',config, bool)[0])

    np.testing.assert_array_equal(list1, [ 1, 0, 0, 1, 0 ])
    np.testing.assert_array_equal(list2, [ 0, 1, 1, 1, 0 ])

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



def test_str_value():
    """Test various ways to generate a str value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : { 'dir' : 'config_input', 'file_name' : 'catalog.txt' } },

        'val1' : -93,
        'val2' : True,
        'val3' : 123.8,
        'str1' : "Norwegian",
        'str2' : u"Blue",
        'cat1' : { 'type' : 'InputCatalog' , 'col' : 6 },
        'cat2' : { 'type' : 'InputCatalog' , 'col' : 7 },
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
    }

    galsim.config.ProcessInput(config)

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

    # Test values read from an InputCatalog
    input_cat = galsim.InputCatalog(dir='config_input', file_name='catalog.txt')
    cat1 = []
    cat2 = []
    for k in range(3):
        config['seq_index'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, str)[0])
        cat2.append(galsim.config.ParseValue(config,'cat2',config, str)[0])

    np.testing.assert_array_equal(cat1, ["He's", "bleedin'", "demised!"])
    # Note: white space in the input catalog always separates columns. ' and " don't work.
    np.testing.assert_array_equal(cat2, ['"ceased', '"bereft', '"kicked'])

    # Test values taken from a List
    list1 = []
    for k in range(5):
        config['seq_index'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, str)[0])

    np.testing.assert_array_equal(list1, ['Beautiful', 'plumage!', 'Ay?', 'Beautiful', 'plumage!'])

    # Test values built using NumberedFile
    file1 = galsim.config.ParseValue(config,'file1',config, str)[0]
    np.testing.assert_equal(file1, 'file005.fits.fz')
    file2 = galsim.config.ParseValue(config,'file2',config, str)[0]
    np.testing.assert_equal(file2, 'file5')

    # Test value built from FormattedStr
    for k in range(9):
        config['seq_index'] = k
        type = k / 3
        dil = k % 3
        fs1 = galsim.config.ParseValue(config,'fs1',config, str)[0]
        np.testing.assert_equal(fs1, 'realgal_type%02d_dilation%d.fits'%(type,dil))

    fs2 = galsim.config.ParseValue(config,'fs2',config, str)[0]
    np.testing.assert_equal(fs2, 
        "%4 5 c 119 3.141593=3.14159=3.141593e+00 11-11 'Goodbye cruel world.', said Pink. %")


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_angle_value():
    """Test various ways to generate an Angle value
    """
    import time
    t1 = time.time()

    config = {
        'input' : { 'catalog' : { 'dir' : 'config_input', 'file_name' : 'catalog.txt' } },

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
                   'theta' : { 'type' : 'InputCatalog' , 'col' : 10 } },
        'cat2' : { 'type' : 'Degrees' , 
                   'theta' : { 'type' : 'InputCatalog' , 'col' : 11 } },
        'ran1' : { 'type' : 'Random' },
        'seq1' : { 'type' : 'Rad', 'theta' : { 'type' : 'Sequence' } },
        'seq2' : { 'type' : 'Deg', 'theta' : { 'type' : 'Sequence', 'first' : 45, 'step' : 80 } },
        'list1' : { 'type' : 'List',
                    'items' : [ 73 * galsim.arcmin,
                                8.9 * galsim.arcmin,
                                3.14 * galsim.arcmin ] },
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

    # Test values read from an InputCatalog
    input_cat = galsim.InputCatalog(dir='config_input', file_name='catalog.txt')
    cat1 = []
    cat2 = []
    for k in range(5):
        config['seq_index'] = k
        cat1.append(galsim.config.ParseValue(config,'cat1',config, galsim.Angle)[0].rad())
        cat2.append(galsim.config.ParseValue(config,'cat2',config, galsim.Angle)[0]/galsim.degrees)

    np.testing.assert_array_almost_equal(cat1, [ 1.2, 0.1, -0.9, 1.2, 0.1 ])
    np.testing.assert_array_almost_equal(cat2, [ 23, 15, 82, 23, 15 ])

    # Test values generated from a uniform deviate
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        ran1 = galsim.config.ParseValue(config,'ran1',config, galsim.Angle)[0]
        theta = rng() * 2 * math.pi
        np.testing.assert_almost_equal(ran1.rad(), theta)

    # Test values generated from a Sequence
    seq1 = []
    seq2 = []
    for k in range(6):
        config['seq_index'] = k
        seq1.append(galsim.config.ParseValue(config,'seq1',config, galsim.Angle)[0].rad())
        seq2.append(galsim.config.ParseValue(config,'seq2',config, galsim.Angle)[0]/galsim.degrees)

    np.testing.assert_array_almost_equal(seq1, [ 0, 1, 2, 3, 4, 5 ])
    np.testing.assert_array_almost_equal(seq2, [ 45, 125, 205, 285, 365, 445 ])

    # Test values taken from a List
    list1 = []
    for k in range(5):
        config['seq_index'] = k
        list1.append(galsim.config.ParseValue(config,'list1',config, galsim.Angle)[0]/galsim.arcmin)

    np.testing.assert_array_almost_equal(list1, [ 73, 8.9, 3.14, 73, 8.9 ])

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
    for k in range(5):
        config['seq_index'] = k
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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_pos_value():
    """Test various ways to generate a Postion value
    """
    import time
    t1 = time.time()

    config = {
        'val1' : galsim.PositionD(0.1,0.2),
        'xy1' : { 'type' : 'XY', 'x' : 1.3, 'y' : 2.4 },
        'ran1' : { 'type' : 'RandomCircle', 'radius' : 3 },
        'list1' : { 'type' : 'List', 
                    'items' : [ galsim.PositionD(0.2, -0.3),
                                galsim.PositionD(-0.5,0.2),
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
    for k in range(6):
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
    for k in range(5):
        config['seq_index'] = k
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


