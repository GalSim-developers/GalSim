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


# Get whatever version of pyfits or astropy we are using
from galsim._pyfits import pyfits


@timer
def test_read():
    """Test reading a FitsHeader from an existing FITS file
    """
    tpv_len = 215

    def check_tpv(header):
        """Check that the header object has correct values from the tpv.fits file
        """
        # Check using a few different access methods.
        assert header['TIME-OBS'] == '04:28:14.105'
        assert header.get('FILTER') == 'I'
        assert header['AIRMASS'] == 1.185
        assert len(header) == tpv_len
        assert 'ADC' in header
        assert ('FILPOS',6) in header.items()
        assert ('FILPOS',6) in header.iteritems()
        assert 'OBSERVAT' in header.keys()
        assert 'OBSERVAT' in header.iterkeys()
        assert 54384.18627436 in header.values() # MJD-OBS
        assert 54384.18627436 in header.itervalues()

    file_name = 'tpv.fits'
    dir = 'fits_files'
    # First option: give a file_name
    header = galsim.FitsHeader(file_name=os.path.join(dir,file_name))
    check_tpv(header)
    do_pickle(header)
    # Let the FitsHeader init handle the dir
    header = galsim.FitsHeader(file_name=file_name, dir=dir)
    check_tpv(header)
    do_pickle(header)
    # If the first arg is a str, then it should be interpreted as a file name
    header = galsim.FitsHeader(file_name, dir=dir)
    check_tpv(header)
    # If you pass in a pyfits hdulist, that should also work
    with pyfits.open(os.path.join(dir,file_name)) as hdu_list:
        header = galsim.FitsHeader(hdu_list=hdu_list)
    check_tpv(header)
    do_pickle(header)
    # Can explicitly give an hdu number to use.  In this case, there is only 1, so need to use 0.
    with pyfits.open(os.path.join(dir,file_name)) as hdu_list:
        header = galsim.FitsHeader(hdu_list=hdu_list, hdu=0)
    check_tpv(header)
    do_pickle(header)
    # Can explicitly give an hdu number to use.  In this case, there is only 1, so need to use 0.
    header = galsim.FitsHeader(file_name=file_name, dir=dir, hdu=0)
    check_tpv(header)
    do_pickle(header)
    # If you pass in a pyfits Header object, that should also work
    with pyfits.open(os.path.join(dir,file_name)) as hdu_list:
        header = galsim.FitsHeader(header=hdu_list[0].header)
    check_tpv(header)
    do_pickle(header)
    # The header is the first parameter, so don't need to name it.
    with pyfits.open(os.path.join(dir,file_name)) as hdu_list:
        header = galsim.FitsHeader(hdu_list[0].header)
    check_tpv(header)
    # FitsHeader can read from a compressed file too
    header = galsim.FitsHeader(file_name=file_name + '.gz', dir=dir, compression='auto')
    check_tpv(header)
    do_pickle(header)
    header = galsim.FitsHeader(file_name=file_name + '.gz', dir=dir, compression='gzip')
    check_tpv(header)
    do_pickle(header)

    assert_raises(TypeError, galsim.FitsHeader, file_name=file_name, header=header)
    with pyfits.open(os.path.join(dir,file_name)) as hdu_list:
        assert_raises(TypeError, galsim.FitsHeader, file_name=file_name, hdu_list=hdu_list)
        assert_raises(TypeError, galsim.FitsHeader, header=header, hdu_list=hdu_list)

    # Remove an item from the header
    # Start with file_name constructor, to test that the repr is changed by the edit.
    orig_header = header
    header = galsim.FitsHeader(file_name=os.path.join(dir,file_name))
    assert header == orig_header
    del header['AIRMASS']
    assert 'AIRMASS' not in header
    assert len(header) == tpv_len-1
    assert header != orig_header
    do_pickle(header)

    # Should be able to get with a default value if the key is not present
    assert header.get('AIRMASS', 2.0) == 2.0
    # key should still not be in the header
    assert 'AIRMASS' not in header
    assert len(header) == tpv_len-1
    assert header != orig_header

    # Add items to a header
    header['AIRMASS'] = 2
    assert header.get('AIRMASS') == 2
    assert header != orig_header

    # Pop does a similar thing:
    assert header.pop('AIRMASS') == 2.0
    assert 'AIRMASS' not in header

    # Works if not preset, given default
    assert header.pop('AIRMASS', 2.0) == 2.0
    assert 'AIRMASS' not in header
    header['AIRMASS'] = 2
    assert header['AIRMASS'] == 2

    # Get real value if preset and given default value
    assert header.pop('AIRMASS', 1.9) == 2.0
    assert 'AIRMASS' not in header
    header['AIRMASS'] = 2
    assert header['AIRMASS'] == 2

    # Overwrite an existing value
    header['AIRMASS'] = 1.7
    assert header.get('AIRMASS') == 1.7
    assert header != orig_header

    # Set with a comment field
    header['AIRMASS'] = (1.9, 'The airmass of the observation')
    assert header.get('AIRMASS') == 1.9
    assert header != orig_header

    # Update with a dict
    d = { 'AIRMASS' : 1.185 }
    header.update(d)
    assert header.get('AIRMASS') == 1.185
    # We are essentially back to where we started, except the len won't be right.
    # Deleting a key removed an item each time, but setting it overwrote a blank item.
    # But if we add back another few of these, we should be back to the original values.
    header.append('','', useblanks=False)
    header.append('','', useblanks=False)
    header.append('','', useblanks=False)
    check_tpv(header)
    do_pickle(header)
    assert header != orig_header  # It's still not equal, because the AIRMASS item is in a
                                  # different location in the list, which is relevant for equality.

    # Clear all values
    header.clear()
    assert 'AIRMASS' not in header
    assert 'FILTER' not in header
    assert len(header) == 0
    do_pickle(header)
    assert header != orig_header


@timer
def test_scamp():
    """Test that we can read in a SCamp .head file correctly
    """
    dir = 'fits_files'
    file_name = 'scamp.head'

    header = galsim.FitsHeader(file_name=file_name, dir=dir, text_file=True)
    # Just check a few values.  The full test of this file as a wcs is in test_wcs.py
    assert header['RADECSYS'] == 'FK5'
    assert header['MAGZEROP'] == 30.
    assert header['ASTINST'] == 39
    do_pickle(header)


def check_dict(d):
    def check_dict(header):
        """Check that the header object has correct values from the given dict
        """
        assert header['TIME-OBS'] == '04:28:14.105'
        assert header['FILTER'] == 'I'
        assert header['AIRMASS'] == 1.185
        assert len(header) == 3

    # Construct from a given dict
    header = galsim.FitsHeader(header = d)
    check_dict(header)
    do_pickle(header)

    # Start with a blank dict and add elements individually
    header = galsim.FitsHeader(header = {})
    do_pickle(header)
    for k in d:
        header[k] = d[k]
    check_dict(header)
    do_pickle(header)

    # Set with a comment field
    header = galsim.FitsHeader(header = {})
    for k in d:
        header[k] = (d[k], 'The value of ' + k)
    check_dict(header)
    do_pickle(header)

    # Use update
    header = galsim.FitsHeader({})
    header.update(d)
    check_dict(header)
    do_pickle(header)

    # Use default constructor
    header = galsim.FitsHeader()
    do_pickle(header)
    assert len(header) == 0
    header.update(d)
    check_dict(header)
    do_pickle(header)

@timer
def test_dict():
    """Test that we can create a FitsHeader from a dict
    """
    d = { 'TIME-OBS' : '04:28:14.105' ,
          'FILTER'   : 'I',
          'AIRMASS'  : 1.185 }
    check_dict(d)

@timer
def test_lowercase():
    """Test that lowercase keys are turned into uppercase.
    """
    d = { 'Time-Obs' : '04:28:14.105' ,
          'filter'   : 'I',
          'AirMAsS'  : 1.185 }
    check_dict(d)


if __name__ == "__main__":
    test_read()
    test_scamp()
    test_dict()
    test_lowercase()
