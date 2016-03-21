from __future__ import print_function
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

# Get whatever version of pyfits or astropy we are using
from galsim._pyfits import pyfits, pyfits_version

def test_read():
    """Test reading a FitsHeader from an existing FITS file
    """
    import time
    t1 = time.time()

    # Older pyfits versions treat the blank rows differently, so it comes out as 213.
    # I don't know exactly when it switched, but for < 3.1, I'll just update this to 
    # whatever the initial value is.
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
    if pyfits_version < '3.1':
        tpv_len = len(header)
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
    hdu_list = pyfits.open(os.path.join(dir,file_name))
    header = galsim.FitsHeader(hdu_list=hdu_list)
    check_tpv(header)
    do_pickle(header)
    # Can explicitly give an hdu number to use.  In this case, there is only 1, so need to use 0.
    header = galsim.FitsHeader(hdu_list=hdu_list, hdu=0)
    check_tpv(header)
    do_pickle(header)
    # If you pass in a pyfits Header object, that should also work
    header = galsim.FitsHeader(header=hdu_list[0].header)
    check_tpv(header)
    do_pickle(header)
    # The header is the first parameter, so don't need to name it.
    header = galsim.FitsHeader(hdu_list[0].header)
    check_tpv(header)


    # Remove an item from the header
    # Start with file_name constructor, to test that the repr is changed by the edit.
    header = galsim.FitsHeader(file_name=os.path.join(dir,file_name))
    del header['AIRMASS']
    assert 'AIRMASS' not in header
    if pyfits_version >= '3.1':
        assert len(header) == tpv_len-1
    do_pickle(header)

    # Should be able to get with a default value if the key is not present
    assert header.get('AIRMASS', 2.0) == 2.0
    # key should still not be in the header
    assert 'AIRMASS' not in header
    if pyfits_version >= '3.1':
        assert len(header) == tpv_len-1

    # Add items to a header
    header['AIRMASS'] = 2
    assert header.get('AIRMASS') == 2

    # Overwrite an existing value
    header['AIRMASS'] = 1.7
    assert header.get('AIRMASS') == 1.7

    # Set with a comment field
    header['AIRMASS'] = (1.9, 'The airmass of the observation')
    assert header.get('AIRMASS') == 1.9

    # Update with a dict
    d = { 'AIRMASS' : 1.185 }
    header.update(d)
    assert header.get('AIRMASS') == 1.185
    # We are essentially back to where we started, except the len won't be right.
    # Deleting a key removed an item, but setting it overwrote a blank item.
    # But if we add back another one of these, we should be back to the original values.
    header.append('','', useblanks=False)
    check_tpv(header)
    do_pickle(header)

    # Clear all values
    header.clear()
    assert 'AIRMASS' not in header
    assert 'FILTER' not in header
    assert len(header) == 0
    do_pickle(header)

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


def test_scamp():
    """Test that we can read in a SCamp .head file correctly
    """
    import time
    t1 = time.time()

    dir = 'fits_files'
    file_name = 'scamp.head'

    header = galsim.FitsHeader(file_name=file_name, dir=dir, text_file=True)
    # Just check a few values.  The full test of this file as a wcs is in test_wcs.py
    assert header['RADECSYS'] == 'FK5'
    assert header['MAGZEROP'] == 30.
    assert header['ASTINST'] == 39
    do_pickle(header)

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


def test_dict():
    """Test that we can create a FitsHeader from a dict
    """
    import time
    t1 = time.time()

    d = { 'TIME-OBS' : '04:28:14.105' ,
          'FILTER'   : 'I',
          'AIRMASS'  : 1.185 }

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

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


if __name__ == "__main__":
    test_read()
    test_scamp()
    test_dict()
