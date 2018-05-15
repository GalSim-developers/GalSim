# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

import galsim
from galsim_test_helpers import *


@timer
def test_basic_catalog():
    """Test basic operations on Catalog."""
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


@timer
def test_basic_dict():
    """Test basic operations on Dict."""
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
    try:
        import yaml
    except ImportError as e:
        # Raise a warning so this message shows up when doing pytest (or scons tests).
        import warnings
        warnings.warn("Unable to import yaml.  Skipping yaml tests")
        print("Caught ",e)
    else:
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


@timer
def test_single_row():
    """Test that we can read catalogs with just one row (#394)
    """
    filename = "output/test394.txt"
    with open(filename, 'w') as f:
        f.write("3 4 5\n")
    cat = galsim.Catalog(filename, file_type='ascii')
    np.testing.assert_array_equal(
        cat.data, np.array([["3","4","5"]]),
        err_msg="galsim.Catalog.__init__ failed to read 1-row file")


@timer
def test_output_catalog():
    """Test basic operations on Catalog."""
    names = [ 'float1', 'float2', 'int1', 'int2', 'bool1', 'bool2',
              'str1', 'str2', 'str3', 'str4', 'angle', 'posi', 'posd', 'shear' ]
    types = [ float, float, int, int, bool, bool, str, str, str, str,
              galsim.Angle, galsim.PositionI, galsim.PositionD, galsim.Shear ]
    out_cat = galsim.OutputCatalog(names, types)

    out_cat.addRow( [1.234, 4.131, 9, -3, 1, True, "He's", '"ceased', 'to', 'be"',
                      1.2 * galsim.degrees, galsim.PositionI(5,6),
                      galsim.PositionD(0.3,-0.4), galsim.Shear(g1=0.2, g2=0.1) ])
    out_cat.addRow( (2.345, -900, 0.0, 8, False, 0, "bleedin'", '"bereft', 'of', 'life"',
                      11 * galsim.arcsec, galsim.PositionI(-35,106),
                      galsim.PositionD(23.5,55.1), galsim.Shear(e1=-0.1, e2=0.15) ))
    out_cat.addRow( [3.4560001, 8.e3, -4, 17.0, 1, 0, 'demised!', '"kicked', 'the', 'bucket"',
                      0.4 * galsim.radians, galsim.PositionI(88,99),
                      galsim.PositionD(-0.99,-0.88), galsim.Shear() ])

    # First the ASCII version
    out_cat.write(dir='output', file_name='catalog.dat')
    cat = galsim.Catalog(dir='output', file_name='catalog.dat')
    np.testing.assert_equal(cat.ncols, 17)
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
    np.testing.assert_almost_equal(cat.getFloat(0,10), 1.2 * galsim.degrees / galsim.radians)
    np.testing.assert_almost_equal(cat.getInt(1,11), -35)
    np.testing.assert_almost_equal(cat.getInt(1,12), 106)
    np.testing.assert_almost_equal(cat.getFloat(2,13), -0.99)
    np.testing.assert_almost_equal(cat.getFloat(2,14), -0.88)
    np.testing.assert_almost_equal(cat.getFloat(0,15), 0.2)
    np.testing.assert_almost_equal(cat.getFloat(0,16), 0.1)

    # Next the FITS version
    if os.path.isfile('output/catalog.fits'):
        os.remove('output/catalog.fits')
    out_cat.write(dir='output', file_name='catalog.fits')
    cat = galsim.Catalog(dir='output', file_name='catalog.fits')
    np.testing.assert_equal(cat.ncols, 17)
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
    np.testing.assert_almost_equal(cat.getFloat(0,'angle.rad'),
                                   1.2 * galsim.degrees / galsim.radians)
    np.testing.assert_equal(cat.getInt(1,'posi.x'), -35)
    np.testing.assert_equal(cat.getInt(1,'posi.y'), 106)
    np.testing.assert_almost_equal(cat.getFloat(2,'posd.x'), -0.99)
    np.testing.assert_almost_equal(cat.getFloat(2,'posd.y'), -0.88)
    np.testing.assert_almost_equal(cat.getFloat(0,'shear.g1'), 0.2)
    np.testing.assert_almost_equal(cat.getFloat(0,'shear.g2'), 0.1)

    # Check that it properly overwrites an existing output file.
    out_cat.addRow( [1.234, 4.131, 9, -3, 1, True, "He's", '"ceased', 'to', 'be"',
                     1.2 * galsim.degrees, galsim.PositionI(5,6),
                     galsim.PositionD(0.3,-0.4), galsim.Shear(g1=0.2, g2=0.1) ])
    assert out_cat.rows[3] == out_cat.rows[0]
    out_cat.write(dir='output', file_name='catalog.fits')  # Same name as above.
    cat2 = galsim.Catalog(dir='output', file_name='catalog.fits')
    np.testing.assert_equal(cat2.ncols, 17)
    np.testing.assert_equal(cat2.nobjects, 4)
    for key in names[:10]:
        assert cat2.data[key][3] == cat2.data[key][0]

    # Check pickling
    do_pickle(out_cat)
    out_cat2 = galsim.OutputCatalog(names, types)  # No data.
    do_pickle(out_cat2)


if __name__ == "__main__":
    test_basic_catalog()
    test_basic_dict()
    test_single_row()
    test_output_catalog()
