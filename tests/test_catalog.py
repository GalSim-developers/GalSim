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

import galsim
from galsim_test_helpers import *


@timer
def test_ascii_catalog():
    """Test basic operations on an ASCII Catalog."""

    cat = galsim.Catalog(dir='config_input', file_name='catalog.txt')
    np.testing.assert_equal(cat.ncols, 12)
    np.testing.assert_equal(cat.nobjects, 3)
    np.testing.assert_equal(cat.isFits(), False)
    np.testing.assert_equal(cat.get(1,11), '15')
    np.testing.assert_equal(cat.getInt(1,11), 15)
    np.testing.assert_almost_equal(cat.getFloat(2,1), 8000)

    check_pickle(cat)

    cat2 = galsim.Catalog('catalog.txt', 'config_input', comments='#', file_type='ASCII')
    assert cat2 == cat
    assert len(cat2) == cat2.nobjects == cat2.getNObjects() == cat.nobjects
    assert cat2.ncols == cat.ncols

    cat2 = galsim.Catalog('catalog2.txt', 'config_input', comments='%')
    assert cat2.nobjects == cat.nobjects
    np.testing.assert_array_equal(cat2.data, cat.data)
    assert cat2 != cat
    check_pickle(cat2)

    cat3 = galsim.Catalog('catalog3.txt', 'config_input', comments='')
    assert len(cat3) == cat3.nobjects == cat.nobjects
    np.testing.assert_array_equal(cat3.data, cat.data)
    assert cat3 != cat
    check_pickle(cat3)

    # Check construction errors
    assert_raises(galsim.GalSimValueError, galsim.Catalog, 'catalog.txt', file_type='invalid')
    assert_raises(ValueError, galsim.Catalog, 'catalog3.txt', 'config_input', comments="#%")
    assert_raises(OSError, galsim.Catalog, 'catalog.txt')  # Wrong dir
    assert_raises(OSError, galsim.Catalog, 'invalid.txt', 'config_input')

    # Check indexing errors
    assert_raises(IndexError, cat.get, -1, 11)
    assert_raises(IndexError, cat.get, 3, 11)
    assert_raises(IndexError, cat.get, 1, -1)
    assert_raises(IndexError, cat.get, 1, 50)
    assert_raises(IndexError, cat.get, 'val', 11)
    assert_raises(IndexError, cat.get, 3, 'val')


@timer
def test_fits_catalog():
    """Test basic operations on a FITS Catalog."""

    cat = galsim.Catalog(dir='config_input', file_name='catalog.fits')
    np.testing.assert_equal(cat.ncols, 12)
    np.testing.assert_equal(cat.nobjects, 3)
    np.testing.assert_equal(cat.isFits(), True)
    np.testing.assert_equal(cat.get(1,'angle2'), 15)
    np.testing.assert_equal(cat.getInt(1,'angle2'), 15)
    np.testing.assert_almost_equal(cat.getFloat(2,'float2'), 8000)

    check_pickle(cat)

    cat2 = galsim.Catalog('catalog.fits', 'config_input', hdu=1, file_type='FITS')
    assert cat2 == cat
    assert len(cat2) == cat2.nobjects == cat2.getNObjects() == cat.nobjects
    assert cat2.ncols == cat.ncols

    # Check construction errors
    assert_raises(galsim.GalSimValueError, galsim.Catalog, 'catalog.fits', file_type='invalid')
    assert_raises(OSError, galsim.Catalog, 'catalog.fits')  # Wrong dir
    assert_raises(OSError, galsim.Catalog, 'invalid.fits', 'config_input')

    # Check indexing errors
    assert_raises(IndexError, cat.get, -1, 'angle2')
    assert_raises(IndexError, cat.get, 3, 'angle2')
    assert_raises(KeyError, cat.get, 1, 'invalid')
    assert_raises(KeyError, cat.get, 1, 3)
    assert_raises(IndexError, cat.get, 'val', 'angle2')

    # Check non-default hdu
    cat2 = galsim.Catalog('catalog2.fits', 'config_input', hdu=2)
    assert len(cat2) == cat2.nobjects == cat.nobjects
    np.testing.assert_array_equal(cat2.data, cat.data)
    assert cat2 != cat
    check_pickle(cat2)

    cat3 = galsim.Catalog('catalog2.fits', 'config_input', hdu='data')
    assert cat3.nobjects == cat.nobjects
    np.testing.assert_array_equal(cat3.data, cat.data)
    assert cat3 != cat
    assert cat3 != cat2  # Even though these are the same, it doesn't know 'data' is hdu 2.
    check_pickle(cat3)



@timer
def test_basic_dict():
    """Test basic operations on Dict."""
    import yaml

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
    check_pickle(d)

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
    check_pickle(d)

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
    check_pickle(d)

    # We also have longer chained keys in dict.yaml
    np.testing.assert_equal(d.get('noise.models.0.variance'), 0.12)
    np.testing.assert_equal(d.get('noise.models.1.gain'), 1.9)
    with assert_raises(KeyError):
        d.get('invalid')
    with assert_raises(KeyError):
        d.get('noise.models.invalid')
    with assert_raises(KeyError):
        d.get('noise.models.1.invalid')
    with assert_raises(IndexError):
        d.get('noise.models.2.invalid')
    with assert_raises(TypeError):
        d.get('noise.models.1.gain.invalid')

    # It's really hard to get to this error.  I think this is the only (contrived) way.
    d3 = galsim.Dict('dict.yaml', 'config_input', key_split=None)
    with assert_raises(KeyError):
        d3.get('')
    check_pickle(d3)

    with assert_raises(galsim.GalSimValueError):
        galsim.Dict(dir='config_input', file_name='dict.yaml', file_type='invalid')
    with assert_raises(galsim.GalSimValueError):
        galsim.Dict(dir='config_input', file_name='dict.txt')
    with assert_raises(OSError):
        galsim.Catalog('invalid.yaml', 'config_input')

    # Check some dict equivalences.
    assert 'noise' in d
    assert len(d) == 5
    assert sorted(d.keys()) == ['b', 'f', 'i', 'noise', 's']
    assert all( d[k] == v for k,v in d.items() )
    assert all( d[k] == v for k,v in zip(d.keys(), d.values()) )
    assert all( d[k] == v for k,v in d.iteritems() )
    assert all( d[k] == v for k,v in zip(d.iterkeys(), d.itervalues()) )
    assert all( k in d for k in d )


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
    types = [ float, 'f8', int, 'i4', bool, 'bool', str, 'str', 'S', 'S0',
              galsim.Angle, galsim.PositionI, galsim.PositionD, galsim.Shear ]
    out_cat = galsim.OutputCatalog(names, types)

    row1 = (1.234, 4.131, 9, -3, 1, True, "He's", '"ceased', 'to', 'be"',
             1.2 * galsim.degrees, galsim.PositionI(5,6),
             galsim.PositionD(0.3,-0.4), galsim.Shear(g1=0.2, g2=0.1))
    row2 = (2.345, -900, 0.0, 8, False, 0, "bleedin'", '"bereft', 'of', 'life"',
            11 * galsim.arcsec, galsim.PositionI(-35,106),
            galsim.PositionD(23.5,55.1), galsim.Shear(e1=-0.1, e2=0.15))
    row3 = (3.4560001, 8.e3, -4, 17.0, 1, 0, 'demised!', '"kicked', 'the', 'bucket"',
            0.4 * galsim.radians, galsim.PositionI(88,99),
            galsim.PositionD(-0.99,-0.88), galsim.Shear())

    out_cat.addRow(row1)
    out_cat.addRow(row2)
    out_cat.addRow(row3)

    assert out_cat.names == out_cat.getNames() == names
    assert out_cat.types == out_cat.getTypes() == types
    assert len(out_cat) == out_cat.getNObjects() == out_cat.nobjects == 3
    assert out_cat.getNCols() == out_cat.ncols == len(names)

    # Can also set the types after the fact.
    # MJ: I think this used to be used by the "truth" catalog extra output.
    #     But it doesn't seem to be used there anymore.  Probably not by anything then.
    #     I'm not sure how useful it is, I guess it doesn't hurt to leave it in.
    out_cat2 = galsim.OutputCatalog(names)
    assert out_cat2.types == [float] * len(names)
    out_cat2.setTypes(types)
    assert out_cat2.types == out_cat2.getTypes() == types

    # Another feature that doesn't seem to be used anymore is you can add the rows out of order
    # and just give a key to use for sorting at the end.
    out_cat2.addRow(row3, 3)
    out_cat2.addRow(row1, 1)
    out_cat2.addRow(row2, 2)

    # Check ASCII round trip
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

    # Check FITS round trip
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

    # The one that was made out of order should write the same file.
    out_cat2.write(dir='output', file_name='catalog2.fits')
    cat2 = galsim.Catalog(dir='output', file_name='catalog2.fits')
    np.testing.assert_array_equal(cat2.data, cat.data)
    assert cat2 != cat  # Because file_name is different.

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
    check_pickle(out_cat)
    out_cat2 = galsim.OutputCatalog(names, types)  # No data.
    check_pickle(out_cat2)

    # Check errors
    with assert_raises(galsim.GalSimValueError):
        out_cat.addRow((1,2,3))  # Wrong length
    with assert_raises(galsim.GalSimValueError):
        out_cat.write(dir='output', file_name='catalog.txt', file_type='invalid')


if __name__ == "__main__":
    runtests(__file__)
