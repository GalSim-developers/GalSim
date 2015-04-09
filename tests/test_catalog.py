# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
import galsim.catalog

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

    filename = "test394.txt"
    with open(filename, 'w') as f:
        f.write("3 4 5\n")
    cat = galsim.catalog.Catalog(filename, file_type='ascii')
    np.testing.assert_array_equal(
        cat.data, np.array([["3","4","5"]]),
        err_msg="galsim.catalog.Catalog.__init__ failed to read 1-row file")
    os.remove(filename)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_basic_catalog()
    test_basic_dict()
    test_single_row()
