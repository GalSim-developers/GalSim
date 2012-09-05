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
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -2 } }
    }

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
    config['catalog'] = input_cat
    cat1 = [ galsim.config.ParseValue(config,'cat1',config, float)[0] for k in range(3) ]
    np.testing.assert_array_almost_equal(cat1, [ 1.234, 2.345, 3.456 ])

    cat2 = [ galsim.config.ParseValue(config,'cat2',config, float)[0] for k in range(3) ]
    np.testing.assert_array_almost_equal(cat2, [ 4.131, -900, 8000 ])

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
    seq1 = [ galsim.config.ParseValue(config,'seq1',config, float)[0] for k in range(6) ]
    np.testing.assert_array_almost_equal(seq1, [ 0, 1, 2, 3, 4, 5 ])

    seq2 = [ galsim.config.ParseValue(config,'seq2',config, float)[0] for k in range(6) ]
    np.testing.assert_array_almost_equal(seq2, [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ])

    seq3 = [ galsim.config.ParseValue(config,'seq3',config, float)[0] for k in range(6) ]
    np.testing.assert_array_almost_equal(seq3, [ 1.5, 2, 2.5, 3, 3.5, 4 ])

    seq4 = [ galsim.config.ParseValue(config,'seq4',config, float)[0] for k in range(6) ]
    np.testing.assert_array_almost_equal(seq4, [ 10, 8, 6, 4, 2, 0 ])

    seq5 = [ galsim.config.ParseValue(config,'seq5',config, float)[0] for k in range(6) ]
    np.testing.assert_array_almost_equal(seq5, [ 1, 1, 2, 2, 1, 1 ])

    # Test values taken from a List
    list1 = [ galsim.config.ParseValue(config,'list1',config, float)[0] for k in range(5) ]
    np.testing.assert_array_almost_equal(list1, [ 73, 8.9, 3.14, 73, 8.9 ])

    list2 = [ galsim.config.ParseValue(config,'list2',config, float)[0] for k in range(5) ]
    np.testing.assert_array_almost_equal(list2, [ 10.8, 8.6, 6.1, 4.3, 2.1 ])

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



def test_int_value():
    """Test various ways to generate an int value
    """
    import time
    t1 = time.time()

    config = {
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
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -2 } }
    }

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
    config['catalog'] = input_cat
    cat1 = [ galsim.config.ParseValue(config,'cat1',config, int)[0] for k in range(3) ]
    np.testing.assert_array_equal(cat1, [ 9, 0, -4 ])

    cat2 = [ galsim.config.ParseValue(config,'cat2',config, int)[0] for k in range(3) ]
    np.testing.assert_array_equal(cat2, [ -3, 8, 17 ])

    # Test values generated from a uniform deviate
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        ran1 = galsim.config.ParseValue(config,'ran1',config, int)[0]
        np.testing.assert_equal(ran1, int(math.floor(rng() * 4)))

        ran2 = galsim.config.ParseValue(config,'ran2',config, int)[0]
        np.testing.assert_equal(ran2, int(math.floor(rng() * 16))-5)

    # Test values generated from a Sequence
    seq1 = [ galsim.config.ParseValue(config,'seq1',config, int)[0] for k in range(6) ]
    np.testing.assert_array_equal(seq1, [ 0, 1, 2, 3, 4, 5 ])

    seq2 = [ galsim.config.ParseValue(config,'seq2',config, int)[0] for k in range(6) ]
    np.testing.assert_array_equal(seq2, [ 0, 3, 6, 9, 12, 15 ])

    seq3 = [ galsim.config.ParseValue(config,'seq3',config, int)[0] for k in range(6) ]
    np.testing.assert_array_equal(seq3, [ 1, 6, 11, 16, 21, 26 ])

    seq4 = [ galsim.config.ParseValue(config,'seq4',config, int)[0] for k in range(6) ]
    np.testing.assert_array_equal(seq4, [ 10, 8, 6, 4, 2, 0 ])

    seq5 = [ galsim.config.ParseValue(config,'seq5',config, int)[0] for k in range(6) ]
    np.testing.assert_array_equal(seq5, [ 1, 1, 2, 2, 1, 1 ])

    # Test values taken from a List
    list1 = [ galsim.config.ParseValue(config,'list1',config, int)[0] for k in range(5) ]
    np.testing.assert_array_equal(list1, [ 73, 8, 3, 73, 8 ])

    list2 = [ galsim.config.ParseValue(config,'list2',config, int)[0] for k in range(5) ]
    np.testing.assert_array_equal(list2, [ 8, 6, 1, 3, 1 ])

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



def test_bool_value():
    """Test various ways to generate a bool value
    """
    import time
    t1 = time.time()

    config = {
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
                    'index' : { 'type' : 'Sequence', 'first' : 10, 'step' : -2 } }
    }

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
    config['catalog'] = input_cat
    cat1 = [ galsim.config.ParseValue(config,'cat1',config, bool)[0] for k in range(3) ]
    np.testing.assert_array_equal(cat1, [ 1, 0, 1 ])

    cat2 = [ galsim.config.ParseValue(config,'cat2',config, bool)[0] for k in range(3) ]
    np.testing.assert_array_equal(cat2, [ 1, 0, 0 ])

    # Test values generated from a uniform deviate
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.
    for k in range(6):
        ran1 = galsim.config.ParseValue(config,'ran1',config, bool)[0]
        np.testing.assert_equal(ran1, bool(int(math.floor(rng() * 2))))

    # Test values generated from a Sequence
    seq1 = [ galsim.config.ParseValue(config,'seq1',config, bool)[0] for k in range(6) ]
    np.testing.assert_array_equal(seq1, [ 0, 1, 0, 1, 0, 1 ])

    seq2 = [ galsim.config.ParseValue(config,'seq2',config, bool)[0] for k in range(6) ]
    np.testing.assert_array_equal(seq2, [ 1, 1, 0, 0, 1, 1 ])

    # Test values taken from a List
    list1 = [ galsim.config.ParseValue(config,'list1',config, bool)[0] for k in range(5) ]
    np.testing.assert_array_equal(list1, [ 1, 0, 0, 1, 0 ])

    list2 = [ galsim.config.ParseValue(config,'list2',config, bool)[0] for k in range(5) ]
    np.testing.assert_array_equal(list2, [ 0, 0, 0, 1, 1 ])

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_float_value()
    test_int_value()
    test_bool_value()


