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

def test_basic_catalog():
    """Test basic operations on Catalog."""
    import time
    t1 = time.time()

    # First the ASCII version
    cat = galsim.Catalog(dir='config_input', file_name='catalog.txt')
    np.testing.assert_equal(cat.ncols, 12)
    np.testing.assert_equal(cat.nobjects, 3)
    np.testing.assert_equal(cat.isFits(), False)
    np.testing.assert_equal(cat.get(1,11), '15')
    np.testing.assert_equal(cat.getInt(1,11), 15)
    np.testing.assert_almost_equal(cat.getFloat(2,1), 8000)

    do_pickle(cat)

    # Next the FITS version
    cat = galsim.Catalog(dir='config_input', file_name='catalog.fits')
    np.testing.assert_equal(cat.ncols, 12)
    np.testing.assert_equal(cat.nobjects, 3)
    np.testing.assert_equal(cat.isFits(), True)
    np.testing.assert_equal(cat.get(1,'angle2'), 15)
    np.testing.assert_equal(cat.getInt(1,'angle2'), 15)
    np.testing.assert_almost_equal(cat.getFloat(2,'float2'), 8000)

    do_pickle(cat)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_basic_dict():
    """Test basic operations on Dict."""
    import time
    t1 = time.time()

    # Pickle
    d = galsim.Dict(dir='config_input', file_name='dict.p')
    np.testing.assert_equal(len(d), 4)
    np.testing.assert_equal(d.file_type, 'PICKLE')
    np.testing.assert_equal(d['i'], 17)
    np.testing.assert_equal(d.get('s'), 'Life')
    np.testing.assert_equal(d.get('s2', 'Grail'), 'Grail')  # Not in dict.  Use default.
    np.testing.assert_almost_equal(d.get('f', 999.), 23.17) # In dict.  Ignore default.
    d2 = galsim.Dict(dir='config_input', file_name='dict.p', file_type='pickle')
    assert d == d2
    do_pickle(d)

    # JSON
    d = galsim.Dict(dir='config_input', file_name='dict.json')
    np.testing.assert_equal(len(d), 4)
    np.testing.assert_equal(d.file_type, 'JSON')
    np.testing.assert_equal(d['i'], -23)
    np.testing.assert_equal(d.get('s'), 'of')
    np.testing.assert_equal(d.get('s2', 'Grail'), 'Grail')  # Not in dict.  Use default.
    np.testing.assert_almost_equal(d.get('f', 999.), -17.23) # In dict.  Ignore default.
    d2 = galsim.Dict(dir='config_input', file_name='dict.json', file_type='json')
    assert d == d2
    do_pickle(d)

    # YAML
    d = galsim.Dict(dir='config_input', file_name='dict.yaml')
    np.testing.assert_equal(len(d), 5)
    np.testing.assert_equal(d.file_type, 'YAML')
    np.testing.assert_equal(d['i'], 1)
    np.testing.assert_equal(d.get('s'), 'Brian')
    np.testing.assert_equal(d.get('s2', 'Grail'), 'Grail')  # Not in dict.  Use default.
    np.testing.assert_almost_equal(d.get('f', 999.), 0.1) # In dict.  Ignore default.
    d2 = galsim.Dict(dir='config_input', file_name='dict.yaml', file_type='yaml')
    assert d == d2
    do_pickle(d)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_single_row():
    """Test that we can read catalogs with just one row (#394)
    """
    import time
    t1 = time.time()

    filename = "output/test394.txt"
    with open(filename, 'w') as f:
        f.write("3 4 5\n")
    cat = galsim.Catalog(filename, file_type='ascii')
    np.testing.assert_array_equal(
        cat.data, np.array([["3","4","5"]]),
        err_msg="galsim.Catalog.__init__ failed to read 1-row file")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_output_catalog():
    """Test basic operations on Catalog."""
    import time
    t1 = time.time()

    names = [ 'float1', 'float2', 'int1', 'int2', 'bool1', 'bool2',
              'str1', 'str2', 'str3', 'str4', 'angle1', 'angle2' ]
    types = [ float, float, int, int, bool, bool, str, str, str, str, float, float ]
    out_cat = galsim.OutputCatalog(names, types)
    out_cat.add_row( [1.234, 4.131, 9, -3, 1, True, "He's", '"ceased', 'to', 'be"', 1.2, 23.0] )
    out_cat.add_row( (2.345, -900, 0.0, 8, False, 0, "bleedin'", '"bereft', 'of', 'life"',
                      0.1, 15.0) )
    out_cat.add_row( [3.4560001, 8.e3, -4, 17.0, 1, 0, 'demised!', '"kicked', 'the', 'bucket"', 
                      -0.9, 82.0] )

    # First the ASCII version
    out_cat.write(dir='output', file_name='catalog.dat')
    cat = galsim.Catalog(dir='output', file_name='catalog.dat')
    np.testing.assert_equal(cat.ncols, 12)
    np.testing.assert_equal(cat.nobjects, 3)
    np.testing.assert_equal(cat.isFits(), False)
    np.testing.assert_almost_equal(cat.getFloat(1,0), 2.345)
    np.testing.assert_almost_equal(cat.getFloat(2,1), 8000.)
    np.testing.assert_equal(cat.getInt(0,2), 9)
    np.testing.assert_equal(cat.getInt(2,3), 17)
    np.testing.assert_equal(cat.getInt(2,4), 1)
    np.testing.assert_equal(cat.getInt(0,5), 1)
    np.testing.assert_equal(cat.get(2,6), 'demised!')
    np.testing.assert_equal(cat.get(1,7), '"bereft')
    np.testing.assert_equal(cat.get(0,8), 'to')
    np.testing.assert_equal(cat.get(2,9), 'bucket"')
    np.testing.assert_almost_equal(cat.getFloat(1,10), 0.1)
    np.testing.assert_almost_equal(cat.getFloat(0,11), 23)

    # Next the FITS version
    out_cat.write(dir='output', file_name='catalog.fits')
    cat = galsim.Catalog(dir='output', file_name='catalog.fits')
    np.testing.assert_equal(cat.ncols, 12)
    np.testing.assert_equal(cat.nobjects, 3)
    np.testing.assert_equal(cat.isFits(), True)
    np.testing.assert_almost_equal(cat.getFloat(1,'float1'), 2.345)
    np.testing.assert_almost_equal(cat.getFloat(2,'float2'), 8000.)
    np.testing.assert_equal(cat.getInt(0,'int1'), 9)
    np.testing.assert_equal(cat.getInt(2,'int2'), 17)
    np.testing.assert_equal(cat.getInt(2,'bool1'), 1)
    np.testing.assert_equal(cat.getInt(0,'bool2'), 1)
    np.testing.assert_equal(cat.get(2,'str1'), 'demised!')
    np.testing.assert_equal(cat.get(1,'str2'), '"bereft')
    np.testing.assert_equal(cat.get(0,'str3'), 'to')
    np.testing.assert_equal(cat.get(2,'str4'), 'bucket"')
    np.testing.assert_almost_equal(cat.getFloat(1,'angle1'), 0.1)
    np.testing.assert_almost_equal(cat.getFloat(0,'angle2'), 23)


    # Check pickling
    do_pickle(out_cat)
    out_cat2 = galsim.OutputCatalog(names, types)  # No data.
    do_pickle(out_cat2)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_basic_catalog()
    test_basic_dict()
    test_single_row()
    test_output_catalog()
