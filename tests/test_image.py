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

"""Unit tests for the Image class.

These tests use six externally generated (IDL + astrolib FITS writing tools) reference images for
the Image unit tests.  These are in tests/data/.

Each image is 5x7 pixels^2 and if each pixel is labelled (x, y) then each pixel value is 10*x + y.
The array thus has values:

15 25 35 45 55 65 75
14 24 34 44 54 64 74
13 23 33 43 53 63 73  ^
12 22 32 42 52 62 72  |
11 21 31 41 51 61 71  y

x ->

With array directions as indicated. This hopefully will make it easy enough to perform sub-image
checks, etc.

Images are in US, UI, S, I, F, D, CF, and CD flavors.

There are also four FITS cubes, and four FITS multi-extension files for testing.  Each is 12
images deep, with the first image being the reference above and each subsequent being the same
incremented by one.

"""

from __future__ import print_function
import os
import sys
import numpy as np
from distutils.version import LooseVersion

import galsim
from galsim_test_helpers import *
from galsim._pyfits import pyfits

# Setup info for tests, not likely to change
ntypes = 8  # Note: Most tests below only run through the first 8 types.
            # test_Image_basic tests all 11 types including the aliases.
types = [np.int16, np.int32, np.uint16, np.uint32, np.float32, np.float64,
         np.complex64, np.complex128, int, float, complex]
simple_types = [int, int, int, int, float, float, complex, complex, int, float, complex]
np_types = [np.int16, np.int32, np.uint16, np.uint32, np.float32, np.float64,
            np.complex64, np.complex128, np.int32, np.float64, np.complex128]
tchar = ['S', 'I', 'US', 'UI', 'F', 'D', 'CF', 'CD', 'I', 'D', 'CD']
int_ntypes = 4  # The first four are the integer types for which we need to test &, |, ^.

ncol = 7
nrow = 5
test_shape = (ncol, nrow)  # shape of image arrays for all tests
ref_array = np.array([
    [11, 21, 31, 41, 51, 61, 71],
    [12, 22, 32, 42, 52, 62, 72],
    [13, 23, 33, 43, 53, 63, 73],
    [14, 24, 34, 44, 54, 64, 74],
    [15, 25, 35, 45, 55, 65, 75] ]).astype(np.int16)
large_array = np.zeros((ref_array.shape[0]*3, ref_array.shape[1]*2), dtype=np.int16)
large_array[::3,::2] = ref_array

# Depth of FITS datacubes and multi-extension FITS files
if __name__ == "__main__":
    nimages = 12
else:
    # There really are 12, but testing the first 3 should be plenty as a unit test, and
    # it helps speed things up.
    nimages = 3

datadir = os.path.join(".", "Image_comparison_images")


@timer
def test_Image_basic():
    """Test that all supported types perform basic Image operations correctly
    """
    # Do all 10 types here, rather than just the 7 numpy types.  i.e. Test the aliases.
    for i in range(len(types)):

        # Check basic constructor from ncol, nrow
        array_type = types[i]
        np_array_type = np_types[i]
        print('array_type = ',array_type,np_array_type)

        # Check basic constructor from ncol, nrow
        im1 = galsim.Image(ncol,nrow,dtype=array_type)

        # Check basic features of array built by Image
        np.testing.assert_array_equal(im1.array, 0.)
        assert im1.array.shape == (nrow,ncol)
        assert im1.array.dtype.type == np_array_type
        assert im1.array.flags.writeable == True
        assert im1.array.flags.c_contiguous == True
        assert im1.dtype == np_array_type

        im1.fill(23)
        np.testing.assert_array_equal(im1.array, 23.)

        bounds = galsim.BoundsI(1,ncol,1,nrow)
        assert im1.xmin == 1
        assert im1.xmax == ncol
        assert im1.ymin == 1
        assert im1.ymax == nrow
        assert im1.bounds == bounds
        assert im1.outer_bounds == galsim.BoundsD(0.5, ncol+0.5, 0.5, nrow+0.5)

        # Same thing if ncol,nrow are kwargs.  Also can give init_value
        im1b = galsim.Image(ncol=ncol, nrow=nrow, dtype=array_type, init_value=23)
        np.testing.assert_array_equal(im1b.array, 23.)
        assert im1 == im1b

        # Adding on xmin, ymin allows you to set an origin other than (1,1)
        im1a = galsim.Image(ncol, nrow, dtype=array_type, xmin=4, ymin=7)
        im1b = galsim.Image(ncol=ncol, nrow=nrow, dtype=array_type, xmin=0, ymin=0)
        assert im1a.xmin == 4
        assert im1a.xmax == ncol+3
        assert im1a.ymin == 7
        assert im1a.ymax == nrow+6
        assert im1a.bounds == galsim.BoundsI(4,ncol+3,7,nrow+6)
        assert im1a.outer_bounds == galsim.BoundsD(3.5, ncol+3.5, 6.5, nrow+6.5)
        assert im1b.xmin == 0
        assert im1b.xmax == ncol-1
        assert im1b.ymin == 0
        assert im1b.ymax == nrow-1
        assert im1b.bounds == galsim.BoundsI(0,ncol-1,0,nrow-1)
        assert im1b.outer_bounds == galsim.BoundsD(-0.5, ncol-0.5, -0.5, nrow-0.5)

        # Also test alternate name of image type: ImageD, ImageF, etc.
        image_type = eval("galsim.Image"+tchar[i]) # Use handy eval() mimics use of ImageSIFD
        im2 = image_type(bounds, init_value=23)
        im2_view = im2.view()
        im2_cview = im2.view(make_const=True)
        im2_conj = im2.conjugate

        assert im2_view.xmin == 1
        assert im2_view.xmax == ncol
        assert im2_view.ymin == 1
        assert im2_view.ymax == nrow
        assert im2_view.bounds == bounds
        assert im2_view.array.dtype.type == np_array_type
        assert im2_view.dtype == np_array_type

        assert im2_cview.xmin == 1
        assert im2_cview.xmax == ncol
        assert im2_cview.ymin == 1
        assert im2_cview.ymax == nrow
        assert im2_cview.bounds == bounds
        assert im2_cview.array.dtype.type == np_array_type
        assert im2_cview.dtype == np_array_type

        assert im1.real.bounds == bounds
        assert im1.imag.bounds == bounds
        assert im2.real.bounds == bounds
        assert im2.imag.bounds == bounds
        assert im2_view.real.bounds == bounds
        assert im2_view.imag.bounds == bounds
        assert im2_cview.real.bounds == bounds
        assert im2_cview.imag.bounds == bounds
        if tchar[i] == 'CF':
            assert im1.real.dtype == np.float32
            assert im1.imag.dtype == np.float32
        elif tchar[i] == 'CD':
            assert im1.real.dtype == np.float64
            assert im1.imag.dtype == np.float64
        else:
            assert im1.real.dtype == np_array_type
            assert im1.imag.dtype == np_array_type

        # Check various ways to set and get values
        for y in range(1,nrow+1):
            for x in range(1,ncol+1):
                im1.setValue(x, y, 100 + 10*x + y)
                im1a.setValue(x+3, y+6, 100 + 10*x + y)
                im1b.setValue(x=x-1, y=y-1, value=100 + 10*x + y)
                im2_view._setValue(x, y, 100 + 10*x)
                im2_view._addValue(x, y, y)

        for y in range(1,nrow+1):
            for x in range(1,ncol+1):
                value = 100 + 10*x + y
                assert im1(x,y) == value
                assert im1(galsim.PositionI(x,y)) == value
                assert im1a(x+3,y+6) == value
                assert im1b(x-1,y-1) == value
                assert im1.view()(x,y) == value
                assert im1.view()(galsim.PositionI(x,y)) == value
                assert im1.view(make_const=True)(x,y) == value
                assert im2(x,y) == value
                assert im2_view(x,y) == value
                assert im2_cview(x,y) == value
                assert im1.conjugate(x,y) == value
                if tchar[i][0] == 'C':
                    # complex conjugate is not a view into the original.
                    assert im2_conj(x,y) == 23
                    assert im2.conjugate(x,y) == value
                else:
                    assert im2_conj(x,y) == value

                value2 = 53 + 12*x - 19*y
                if tchar[i] in ['US', 'UI']:
                    value2 = abs(value2)
                im1[x,y] = value2
                im2_view[galsim.PositionI(x,y)] = value2
                assert im1.getValue(x,y) == value2
                assert im1.view().getValue(x=x, y=y) == value2
                assert im1.view(make_const=True).getValue(x,y) == value2
                assert im2.getValue(x=x, y=y) == value2
                assert im2_view.getValue(x,y) == value2
                assert im2_cview._getValue(x,y) == value2

                assert im1.real(x,y) == value2
                assert im1.view().real(x,y) == value2
                assert im1.view(make_const=True).real(x,y) == value2.real
                assert im2.real(x,y) == value2.real
                assert im2_view.real(x,y) == value2.real
                assert im2_cview.real(x,y) == value2.real
                assert im1.imag(x,y) == 0
                assert im1.view().imag(x,y) == 0
                assert im1.view(make_const=True).imag(x,y) == 0
                assert im2.imag(x,y) == 0
                assert im2_view.imag(x,y) == 0
                assert im2_cview.imag(x,y) == 0

                value3 = 10*x + y
                im1.addValue(x,y, value3-value2)
                im2_view[x,y] += value3-value2
                assert im1[galsim.PositionI(x,y)] == value3
                assert im1.view()[x,y] == value3
                assert im1.view(make_const=True)[galsim.PositionI(x,y)] == value3
                assert im2[x,y] == value3
                assert im2_view[galsim.PositionI(x,y)] == value3
                assert im2_cview[x,y] == value3

        # Setting or getting the value outside the bounds should throw an exception.
        assert_raises(galsim.GalSimBoundsError,im1.setValue,0,0,1)
        assert_raises(galsim.GalSimBoundsError,im1.addValue,0,0,1)
        assert_raises(galsim.GalSimBoundsError,im1.__call__,0,0)
        assert_raises(galsim.GalSimBoundsError,im1.__getitem__,0,0)
        assert_raises(galsim.GalSimBoundsError,im1.__setitem__,0,0,1)
        assert_raises(galsim.GalSimBoundsError,im1.view().setValue,0,0,1)
        assert_raises(galsim.GalSimBoundsError,im1.view().__call__,0,0)
        assert_raises(galsim.GalSimBoundsError,im1.view().__getitem__,0,0)
        assert_raises(galsim.GalSimBoundsError,im1.view().__setitem__,0,0,1)

        assert_raises(galsim.GalSimBoundsError,im1.setValue,ncol+1,0,1)
        assert_raises(galsim.GalSimBoundsError,im1.addValue,ncol+1,0,1)
        assert_raises(galsim.GalSimBoundsError,im1.__call__,ncol+1,0)
        assert_raises(galsim.GalSimBoundsError,im1.view().setValue,ncol+1,0,1)
        assert_raises(galsim.GalSimBoundsError,im1.view().__call__,ncol+1,0)

        assert_raises(galsim.GalSimBoundsError,im1.setValue,0,nrow+1,1)
        assert_raises(galsim.GalSimBoundsError,im1.addValue,0,nrow+1,1)
        assert_raises(galsim.GalSimBoundsError,im1.__call__,0,nrow+1)
        assert_raises(galsim.GalSimBoundsError,im1.view().setValue,0,nrow+1,1)
        assert_raises(galsim.GalSimBoundsError,im1.view().__call__,0,nrow+1)

        assert_raises(galsim.GalSimBoundsError,im1.setValue,ncol+1,nrow+1,1)
        assert_raises(galsim.GalSimBoundsError,im1.addValue,ncol+1,nrow+1,1)
        assert_raises(galsim.GalSimBoundsError,im1.__call__,ncol+1,nrow+1)
        assert_raises(galsim.GalSimBoundsError,im1.view().setValue,ncol+1,nrow+1,1)
        assert_raises(galsim.GalSimBoundsError,im1.view().__call__,ncol+1,nrow+1)

        assert_raises(galsim.GalSimBoundsError,im1.__getitem__,galsim.BoundsI(0,ncol,1,nrow))
        assert_raises(galsim.GalSimBoundsError,im1.__getitem__,galsim.BoundsI(1,ncol,0,nrow))
        assert_raises(galsim.GalSimBoundsError,im1.__getitem__,galsim.BoundsI(1,ncol+1,1,nrow))
        assert_raises(galsim.GalSimBoundsError,im1.__getitem__,galsim.BoundsI(1,ncol,1,nrow+1))
        assert_raises(galsim.GalSimBoundsError,im1.__getitem__,galsim.BoundsI(0,ncol+1,0,nrow+1))
        assert_raises(galsim.GalSimBoundsError,im1.subImage,galsim.BoundsI(0,ncol,1,nrow))
        assert_raises(galsim.GalSimBoundsError,im1.subImage,galsim.BoundsI(1,ncol,0,nrow))
        assert_raises(galsim.GalSimBoundsError,im1.subImage,galsim.BoundsI(1,ncol+1,1,nrow))
        assert_raises(galsim.GalSimBoundsError,im1.subImage,galsim.BoundsI(1,ncol,1,nrow+1))
        assert_raises(galsim.GalSimBoundsError,im1.subImage,galsim.BoundsI(0,ncol+1,0,nrow+1))

        assert_raises(galsim.GalSimBoundsError,im1.setSubImage,galsim.BoundsI(0,ncol,1,nrow),
                      galsim.Image(ncol+1,nrow, init_value=10))
        assert_raises(galsim.GalSimBoundsError,im1.setSubImage,galsim.BoundsI(1,ncol,0,nrow),
                      galsim.Image(ncol+1,nrow, init_value=10))
        assert_raises(galsim.GalSimBoundsError,im1.setSubImage,galsim.BoundsI(1,ncol+1,1,nrow),
                      galsim.Image(ncol+1,nrow, init_value=10))
        assert_raises(galsim.GalSimBoundsError,im1.setSubImage,galsim.BoundsI(1,ncol,1,nrow+1),
                      galsim.Image(ncol+1,nrow, init_value=10))
        assert_raises(galsim.GalSimBoundsError,im1.setSubImage,galsim.BoundsI(0,ncol+1,0,nrow+1),
                      galsim.Image(ncol+2,nrow+2, init_value=10))
        assert_raises(galsim.GalSimBoundsError,im1.__setitem__,galsim.BoundsI(0,ncol,1,nrow),
                      galsim.Image(ncol+1,nrow, init_value=10))
        assert_raises(galsim.GalSimBoundsError,im1.__setitem__,galsim.BoundsI(1,ncol,0,nrow),
                      galsim.Image(ncol+1,nrow, init_value=10))
        assert_raises(galsim.GalSimBoundsError,im1.__setitem__,galsim.BoundsI(1,ncol+1,1,nrow),
                      galsim.Image(ncol+1,nrow, init_value=10))
        assert_raises(galsim.GalSimBoundsError,im1.__setitem__,galsim.BoundsI(1,ncol,1,nrow+1),
                      galsim.Image(ncol+1,nrow, init_value=10))
        assert_raises(galsim.GalSimBoundsError,im1.__setitem__,galsim.BoundsI(0,ncol+1,0,nrow+1),
                      galsim.Image(ncol+2,nrow+2, init_value=10))

        # Also, setting values in something that should be const
        assert_raises(galsim.GalSimImmutableError,im1.view(make_const=True).setValue,1,1,1)
        assert_raises(galsim.GalSimImmutableError,im1.view(make_const=True).real.setValue,1,1,1)
        assert_raises(galsim.GalSimImmutableError,im1.view(make_const=True).imag.setValue,1,1,1)
        if tchar[i][0] != 'C':
            assert_raises(galsim.GalSimImmutableError,im1.imag.setValue,1,1,1)

        # Finally check for the wrong number of arguments in get/setitem
        assert_raises(TypeError,im1.__getitem__,1)
        assert_raises(TypeError,im1.__setitem__,1,1)
        assert_raises(TypeError,im1.__getitem__,1,2,3)
        assert_raises(TypeError,im1.__setitem__,1,2,3,4)

        # Check view of given data
        im3_view = galsim.Image(ref_array.astype(np_array_type))
        slice_array = large_array.astype(np_array_type)[::3,::2]
        im4_view = galsim.Image(slice_array)
        im5_view = galsim.Image(ref_array.astype(np_array_type).tolist(), dtype=array_type)
        im6_view = galsim.Image(ref_array.astype(np_array_type), xmin=4, ymin=7)
        im7_view = galsim.Image(ref_array.astype(np_array_type), xmin=0, ymin=0)
        for y in range(1,nrow+1):
            for x in range(1,ncol+1):
                value3 = 10*x+y
                assert im3_view(x,y) == value3
                assert im4_view(x,y) == value3
                assert im5_view(x,y) == value3
                assert im6_view(x+3,y+6) == value3
                assert im7_view(x-1,y-1) == value3

        # Check shift ops
        im1_view = im1.view() # View with old bounds
        dx = 31
        dy = 16
        im1.shift(dx,dy)
        im2_view.setOrigin(1+dx , 1+dy)
        im3_view.setCenter((ncol+1)/2+dx , (nrow+1)/2+dy)
        shifted_bounds = galsim.BoundsI(1+dx, ncol+dx, 1+dy, nrow+dy)

        assert im1.bounds == shifted_bounds
        assert im2_view.bounds == shifted_bounds
        assert im3_view.bounds == shifted_bounds
        # Others shouldn't have changed
        assert im1_view.bounds == bounds
        assert im2.bounds == bounds
        for y in range(1,nrow+1):
            for x in range(1,ncol+1):
                value3 = 10*x+y
                assert im1(x+dx,y+dy) == value3
                assert im1_view(x,y) == value3
                assert im2(x,y) == value3
                assert im2_view(x+dx,y+dy) == value3
                assert im3_view(x+dx,y+dy) == value3

        assert_raises(TypeError, im1.shift, dx)
        assert_raises(TypeError, im1.shift, dx=dx)
        assert_raises(TypeError, im1.shift, x=dx, y=dy)
        assert_raises(TypeError, im1.shift, dx, dy=dy)
        assert_raises(TypeError, im1.shift, dx, dy, dy)
        assert_raises(TypeError, im1.shift, dx, dy, invalid=True)

        # Check picklability
        do_pickle(im1)
        do_pickle(im1_view)
        do_pickle(im2)
        do_pickle(im2_view)
        do_pickle(im2_cview)
        do_pickle(im3_view)
        do_pickle(im4_view)

    # Also check picklability of Bounds, Position here.
    do_pickle(galsim.PositionI(2,3))
    do_pickle(galsim.PositionD(2.2,3.3))
    do_pickle(galsim.BoundsI(2,3,7,8))
    do_pickle(galsim.BoundsD(2.1, 4.3, 6.5, 9.1))

@timer
def test_undefined_image():
    """Test various ways to construct an image with undefined bounds
    """
    for i in range(len(types)):
        im1 = galsim.Image(dtype=types[i])
        assert not im1.bounds.isDefined()
        assert im1.array.shape == (1,1)
        assert im1 == im1

        im2 = galsim.Image()
        assert not im2.bounds.isDefined()
        assert im2.array.shape == (1,1)
        assert im2 == im2
        if types[i] == np.float32:
            assert im2 == im1

        im3 = galsim.Image(array=np.array([[]],dtype=types[i]))
        assert not im3.bounds.isDefined()
        assert im3.array.shape == (1,1)
        assert im3 == im1

        im4 = galsim.Image(array=np.array([[]]), dtype=types[i])
        assert not im4.bounds.isDefined()
        assert im4.array.shape == (1,1)
        assert im4 == im1

        im5 = galsim.Image(array=np.array([[1]]), dtype=types[i], bounds=galsim.BoundsI())
        assert not im5.bounds.isDefined()
        assert im5.array.shape == (1,1)
        assert im5 == im1

        im6 = galsim.Image(array=np.array([[1]], dtype=types[i]), bounds=galsim.BoundsI())
        assert not im6.bounds.isDefined()
        assert im6.array.shape == (1,1)
        assert im6 == im1

        im7 = 1.0 * im1
        assert not im7.bounds.isDefined()
        assert im7.array.shape == (1,1)
        if types[i] == np.float64:
            assert im7 == im1

        im8 = im1 + 1j * im3
        assert not im8.bounds.isDefined()
        assert im8.array.shape == (1,1)
        if types[i] == np.complex128:
            assert im8 == im1

        im9 = galsim.Image(0, 0)
        assert not im9.bounds.isDefined()
        assert im9.array.shape == (1,1)
        assert im9 == im1

        im10 = galsim.Image(10, 0)
        assert not im10.bounds.isDefined()
        assert im10.array.shape == (1,1)
        assert im10 == im1

        im11 = galsim.Image(0, 19)
        assert not im11.bounds.isDefined()
        assert im11.array.shape == (1,1)
        assert im11 == im1

        assert_raises(galsim.GalSimUndefinedBoundsError,im1.setValue, 0, 0, 1)
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.__call__, 0, 0)
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.view().setValue, 0, 0, 1)
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.view().__call__, 0, 0)
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.view().addValue, 0, 0, 1)
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.fill, 3)
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.view().fill, 3)
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.invertSelf)

        assert_raises(galsim.GalSimUndefinedBoundsError,im1.__getitem__,galsim.BoundsI(1,2,1,2))
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.subImage,galsim.BoundsI(1,2,1,2))

        assert_raises(galsim.GalSimUndefinedBoundsError,im1.setSubImage,galsim.BoundsI(1,2,1,2),
                      galsim.Image(2,2, init_value=10))
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.__setitem__,galsim.BoundsI(1,2,1,2),
                      galsim.Image(2,2, init_value=10))

        im1.scale = 1.
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.calculate_fft)
        assert_raises(galsim.GalSimUndefinedBoundsError,im1.calculate_inverse_fft)

        do_pickle(im1.bounds)
        do_pickle(im1)
        do_pickle(im1.view())
        do_pickle(im1.view(make_const=True))

@timer
def test_Image_FITS_IO():
    """Test that all six FITS reference images are correctly read in by both PyFITS and our Image
    wrappers.
    """
    for i in range(ntypes):
        array_type = types[i]

        if tchar[i][0] == 'C':
            # Cannot write complex Images to fits.  Check for an exception and continue.
            ref_image = galsim.Image(ref_array.astype(array_type))
            test_file = os.path.join(datadir, "test"+tchar[i]+".fits")
            with assert_raises(ValueError):
                ref_image.write(test_file)
            continue

        #
        # Test input from a single external FITS image
        #

        # Read the reference image to from an externally-generated fits file
        test_file = os.path.join(datadir, "test"+tchar[i]+".fits")
        # Check pyfits read for sanity
        with pyfits.open(test_file) as fits:
            test_array = fits[0].data
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_array,
                err_msg="PyFITS failing to read reference image.")

        # Then use galsim fits.read function
        # First version: use pyfits HDUList
        with pyfits.open(test_file) as hdu:
            test_image = galsim.fits.read(hdu_list=hdu)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Failed reading from PyFITS PrimaryHDU input.")

        # Second version: use file name
        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" read failed reading from filename input.")

        #
        # Test full I/O on a single internally-generated FITS image
        #

        # Write the reference image to a fits file
        ref_image = galsim.Image(ref_array.astype(array_type))
        test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits")
        ref_image.write(test_file)

        # Check pyfits read for sanity
        with pyfits.open(test_file) as fits:
            test_array = fits[0].data
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_array,
                err_msg="Image"+tchar[i]+" write failed.")

        # Then use galsim fits.read function
        # First version: use pyfits HDUList
        with pyfits.open(test_file) as hdu:
            test_image = galsim.fits.read(hdu_list=hdu)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Failed reading from PyFITS PrimaryHDU input.")

        # Second version: use file name
        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" read failed reading from filename input.")

        assert_raises(ValueError, galsim.fits.read, test_file, compression='invalid')
        assert_raises(ValueError, ref_image.write, test_file, compression='invalid')
        assert_raises(OSError, galsim.fits.read, test_file, compression='rice')
        assert_raises(OSError, galsim.fits.read, 'invalid.fits')
        assert_raises(OSError, galsim.fits.read, 'config_input/catalog.fits', hdu=1)

        assert_raises(TypeError, galsim.fits.read)
        assert_raises(TypeError, galsim.fits.read, test_file, hdu_list=hdu)
        assert_raises(TypeError, ref_image.write)
        assert_raises(TypeError, ref_image.write, file_name=test_file, hdu_list=hdu)

        # If clobbert = False, then trying to overwrite will raise an OSError
        assert_raises(OSError, ref_image.write, test_file, clobber=False)

        #
        # Test various compression schemes
        #

        # These tests are a bit slow, so we only bother to run them for the first dtype
        # when doing the regular unit tests.  When running python test_image.py, all of them
        # will run, so when working on the code, it is a good idea to run the tests that way.
        if i > 0 and __name__ != "__main__":
            continue

        test_file0 = test_file  # Save the name of the uncompressed file.

        # Test full-file gzip
        test_file = os.path.join(datadir, "test"+tchar[i]+".fits.gz")
        test_image = galsim.fits.read(test_file, compression='gzip')
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" read failed for explicit full-file gzip")

        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" read failed for auto full-file gzip")

        test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits.gz")
        ref_image.write(test_file, compression='gzip')
        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" write failed for explicit full-file gzip")

        ref_image.write(test_file)
        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" write failed for auto full-file gzip")

        # With compression = None or 'none', astropy automatically figures it out anyway.
        test_image = galsim.fits.read(test_file, compression=None)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" write failed for auto full-file gzip")

        assert_raises(OSError, galsim.fits.read, test_file0, compression='gzip')

        # Test full-file bzip2
        test_file = os.path.join(datadir, "test"+tchar[i]+".fits.bz2")
        test_image = galsim.fits.read(test_file, compression='bzip2')
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" read failed for explicit full-file bzip2")

        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" read failed for auto full-file bzip2")

        test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits.bz2")
        ref_image.write(test_file, compression='bzip2')
        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" write failed for explicit full-file bzip2")

        ref_image.write(test_file)
        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" write failed for auto full-file bzip2")

        # With compression = None or 'none', astropy automatically figures it out anyway.
        test_image = galsim.fits.read(test_file, compression=None)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" write failed for auto full-file gzip")

        assert_raises(OSError, galsim.fits.read, test_file0, compression='bzip2')

        # Test rice
        test_file = os.path.join(datadir, "test"+tchar[i]+".fits.fz")
        test_image = galsim.fits.read(test_file, compression='rice')
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" read failed for explicit rice")

        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" read failed for auto rice")

        test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits.fz")
        ref_image.write(test_file, compression='rice')
        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" write failed for explicit rice")

        ref_image.write(test_file)
        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" write failed for auto rice")

        assert_raises(OSError, galsim.fits.read, test_file0, compression='rice')
        assert_raises(OSError, galsim.fits.read, test_file, compression='none')

        # Test gzip_tile
        test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits.gzt")
        ref_image.write(test_file, compression='gzip_tile')
        test_image = galsim.fits.read(test_file, compression='gzip_tile')
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                err_msg="Image"+tchar[i]+" write failed for gzip_tile")

        assert_raises(OSError, galsim.fits.read, test_file0, compression='gzip_tile')
        assert_raises(OSError, galsim.fits.read, test_file, compression='none')

        # Test hcompress
        # Note: hcompress is a lossy algorithm, and starting with astropy 2.0.5,
        # the fidelity of the reconstruction is really quite poor, so only test with
        # rtol=0.1.  I'm not sure if this is a bug in astropy, or just the nature
        # of the hcompress algorithm.  But I'm ignoring it for now, since I don't
        # think too many people use hcompress anyway.
        test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits.hc")
        ref_image.write(test_file, compression='hcompress')
        test_image = galsim.fits.read(test_file, compression='hcompress')
        np.testing.assert_allclose(ref_array.astype(types[i]), test_image.array, rtol=0.1,
                err_msg="Image"+tchar[i]+" write failed for hcompress")

        assert_raises(OSError, galsim.fits.read, test_file0, compression='hcompress')
        assert_raises(OSError, galsim.fits.read, test_file, compression='none')

        # Test plio (only valid on positive integer values)
        if tchar[i] in ['S', 'I']:
            test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits.plio")
            ref_image.write(test_file, compression='plio')
            test_image = galsim.fits.read(test_file, compression='plio')
            np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array,
                    err_msg="Image"+tchar[i]+" write failed for plio")

        assert_raises(OSError, galsim.fits.read, test_file0, compression='plio')
        assert_raises(OSError, galsim.fits.read, test_file, compression='none')


@timer
def test_Image_MultiFITS_IO():
    """Test that all six FITS reference images are correctly read in by both PyFITS and our Image
    wrappers.
    """
    for i in range(ntypes):
        array_type = types[i]

        if tchar[i][0] == 'C':
            # Cannot write complex Images to fits.  Check for an exception and continue.
            ref_image = galsim.Image(ref_array.astype(array_type))
            image_list = []
            for k in range(nimages):
                image_list.append(ref_image + k)
            test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+".fits")
            with assert_raises(ValueError):
                galsim.fits.writeMulti(image_list, test_multi_file)
            continue

        #
        # Test input from an external multi-extension fits file
        #

        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+".fits")
        # Check pyfits read for sanity
        with pyfits.open(test_multi_file) as fits:
            test_array = fits[0].data
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_array,
                err_msg="PyFITS failing to read multi file.")

        # Then use galsim fits.readMulti function
        # First version: use pyfits HDUList
        with pyfits.open(test_multi_file) as hdu:
            test_image_list = galsim.fits.readMulti(hdu_list=hdu)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Failed reading from PyFITS PrimaryHDU input.")

        # Second version: use file name
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed reading from filename input.")

        #
        # Test full I/O for an internally-generated multi-extension fits file
        #

        # Build a list of images with different values
        ref_image = galsim.Image(ref_array.astype(array_type))
        image_list = []
        for k in range(nimages):
            image_list.append(ref_image + k)

        # Write the list to a multi-extension fits file
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits")
        galsim.fits.writeMulti(image_list,test_multi_file)

        # Check pyfits read for sanity
        with pyfits.open(test_multi_file) as fits:
            test_array = fits[0].data
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_array,
                err_msg="PyFITS failing to read multi file.")

        # Then use galsim fits.readMulti function
        # First version: use pyfits HDUList
        with pyfits.open(test_multi_file) as hdu:
            test_image_list = galsim.fits.readMulti(hdu_list=hdu)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Failed reading from PyFITS PrimaryHDU input.")

        # Second version: use file name
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed reading from filename input.")


        #
        # Test writing to hdu_list directly and then writing to file.
        #

        # Start with empty hdu_list
        hdu_list = pyfits.HDUList()

        # Add each image one at a time
        for k in range(nimages):
            image = ref_image + k
            galsim.fits.write(image, hdu_list=hdu_list)

        # Write it out with writeFile
        galsim.fits.writeFile(test_multi_file, hdu_list)

        # Check that reading it back in gives the same values
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed after using writeFile")

        # Can also use writeMulti to write directly to an hdu list
        hdu_list = pyfits.HDUList()
        galsim.fits.writeMulti(image_list, hdu_list=hdu_list)
        galsim.fits.writeFile(test_multi_file, hdu_list)
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed after using writeFile")

        assert_raises(ValueError, galsim.fits.readMulti, test_multi_file, compression='invalid')
        assert_raises(ValueError, galsim.fits.writeMulti, image_list, test_multi_file,
                      compression='invalid')
        assert_raises(ValueError, galsim.fits.writeFile, image_list, test_multi_file,
                      compression='invalid')
        assert_raises(OSError, galsim.fits.readMulti, test_multi_file, compression='rice')
        assert_raises(OSError, galsim.fits.readFile, test_multi_file, compression='rice')
        assert_raises(OSError, galsim.fits.readMulti, hdu_list=pyfits.HDUList())
        assert_raises(OSError, galsim.fits.readMulti, hdu_list=pyfits.HDUList(), compression='rice')
        assert_raises(OSError, galsim.fits.readMulti, 'invalid.fits')
        assert_raises(OSError, galsim.fits.readFile, 'invalid.fits')

        assert_raises(TypeError, galsim.fits.readMulti)
        assert_raises(TypeError, galsim.fits.readMulti, test_multi_file, hdu_list=hdu)
        assert_raises(TypeError, galsim.fits.readMulti, hdu_list=test_multi_file)
        assert_raises(TypeError, galsim.fits.writeMulti)
        assert_raises(TypeError, galsim.fits.writeMulti, image_list)
        assert_raises(TypeError, galsim.fits.writeMulti, image_list,
                      file_name=test_multi_file, hdu_list=hdu)

        assert_raises(OSError, galsim.fits.writeMulti, image_list, test_multi_file, clobber=False)

        assert_raises(TypeError, galsim.fits.writeFile)
        assert_raises(TypeError, galsim.fits.writeFile, image_list)
        assert_raises(ValueError, galsim.fits.writeFile, test_multi_file, image_list,
                      compression='invalid')
        assert_raises(ValueError, galsim.fits.writeFile, test_multi_file, image_list,
                      compression='rice')
        assert_raises(ValueError, galsim.fits.writeFile, test_multi_file, image_list,
                      compression='gzip_tile')
        assert_raises(ValueError, galsim.fits.writeFile, test_multi_file, image_list,
                      compression='hcompress')
        assert_raises(ValueError, galsim.fits.writeFile, test_multi_file, image_list,
                      compression='plio')

        galsim.fits.writeFile(test_multi_file, hdu_list)
        assert_raises(OSError, galsim.fits.writeFile, test_multi_file, image_list, clobber=False)


        #
        # Test various compression schemes
        #

        # These tests are a bit slow, so we only bother to run them for the first dtype
        # when doing the regular unit tests.  When running python test_image.py, all of them
        # will run, so when working on the code, it is a good idea to run the tests that way.
        if i > 0 and __name__ != "__main__":
            continue

        test_multi_file0 = test_multi_file

        # Test full-file gzip
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+".fits.gz")
        test_image_list = galsim.fits.readMulti(test_multi_file, compression='gzip')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed for explicit full-file gzip")

        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed for auto full-file gzip")

        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits.gz")
        galsim.fits.writeMulti(image_list,test_multi_file, compression='gzip')
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeMulti failed for explicit full-file gzip")

        galsim.fits.writeMulti(image_list,test_multi_file)
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeMulti failed for auto full-file gzip")

        # With compression = None or 'none', astropy automatically figures it out anyway.
        test_image_list = galsim.fits.readMulti(test_multi_file, compression=None)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeMulti failed for auto full-file gzip")

        assert_raises(OSError, galsim.fits.readMulti, test_multi_file0, compression='gzip')

        # Test full-file bzip2
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+".fits.bz2")
        test_image_list = galsim.fits.readMulti(test_multi_file, compression='bzip2')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed for explicit full-file bzip2")

        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed for auto full-file bzip2")

        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits.bz2")
        galsim.fits.writeMulti(image_list,test_multi_file, compression='bzip2')
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeMulti failed for explicit full-file bzip2")

        galsim.fits.writeMulti(image_list,test_multi_file)
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeMulti failed for auto full-file bzip2")

        # With compression = None or 'none', astropy automatically figures it out anyway.
        test_image_list = galsim.fits.readMulti(test_multi_file, compression=None)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeMulti failed for auto full-file gzip")

        assert_raises(OSError, galsim.fits.readMulti, test_multi_file0, compression='bzip2')

        # Test rice
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+".fits.fz")
        test_image_list = galsim.fits.readMulti(test_multi_file, compression='rice')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed for explicit rice")

        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readMulti failed for auto rice")

        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits.fz")
        galsim.fits.writeMulti(image_list,test_multi_file, compression='rice')
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeMulti failed for explicit rice")

        galsim.fits.writeMulti(image_list,test_multi_file)
        test_image_list = galsim.fits.readMulti(test_multi_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeMulti failed for auto rice")

        assert_raises(OSError, galsim.fits.readMulti, test_multi_file0, compression='rice')
        assert_raises(OSError, galsim.fits.readMulti, test_multi_file, compression='none')

        # Test gzip_tile
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits.gzt")
        galsim.fits.writeMulti(image_list,test_multi_file, compression='gzip_tile')
        test_image_list = galsim.fits.readMulti(test_multi_file, compression='gzip_tile')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeMulti failed for gzip_tile")

        assert_raises(OSError, galsim.fits.readMulti, test_multi_file0, compression='gzip_tile')
        assert_raises(OSError, galsim.fits.readMulti, test_multi_file, compression='none')

        # Test hcompress
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits.hc")
        galsim.fits.writeMulti(image_list,test_multi_file, compression='hcompress')
        test_image_list = galsim.fits.readMulti(test_multi_file, compression='hcompress')
        for k in range(nimages):
            np.testing.assert_allclose((ref_array+k).astype(types[i]),
                    test_image_list[k].array, rtol=0.1,
                    err_msg="Image"+tchar[i]+" writeMulti failed for hcompress")

        assert_raises(OSError, galsim.fits.readMulti, test_multi_file0, compression='hcompress')
        assert_raises(OSError, galsim.fits.readMulti, test_multi_file, compression='none')

        # Test plio (only valid on positive integer values)
        if tchar[i] in ['S', 'I']:
            test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits.plio")
            galsim.fits.writeMulti(image_list,test_multi_file, compression='plio')
            test_image_list = galsim.fits.readMulti(test_multi_file, compression='plio')
            for k in range(nimages):
                np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                        test_image_list[k].array,
                        err_msg="Image"+tchar[i]+" writeMulti failed for plio")

        assert_raises(OSError, galsim.fits.readMulti, test_multi_file0, compression='plio')
        assert_raises(OSError, galsim.fits.readMulti, test_multi_file, compression='none')


@timer
def test_Image_CubeFITS_IO():
    """Test that all six FITS reference images are correctly read in by both PyFITS and our Image
    wrappers.
    """
    for i in range(ntypes):
        array_type = types[i]

        if tchar[i][0] == 'C':
            # Cannot write complex Images to fits.  Check for an exception and continue.
            ref_image = galsim.Image(ref_array.astype(array_type))
            image_list = []
            for k in range(nimages):
                image_list.append(ref_image + k)
            test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+".fits")
            with assert_raises(ValueError):
                galsim.fits.writeCube(image_list, test_cube_file)
            array_list = [im.array for im in image_list]
            with assert_raises(ValueError):
                galsim.fits.writeCube(array_list, test_cube_file)
            one_array = np.asarray(array_list)
            with assert_raises(ValueError):
                galsim.fits.writeCube(one_array, test_cube_file)
            continue

        #
        # Test input from an external fits data cube
        #

        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+".fits")
        # Check pyfits read for sanity
        with pyfits.open(test_cube_file) as fits:
            test_array = fits[0].data
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]), test_array[k,:,:],
                    err_msg="PyFITS failing to read cube file.")

        # Then use galsim fits.readCube function
        # First version: use pyfits HDUList
        with pyfits.open(test_cube_file) as hdu:
            test_image_list = galsim.fits.readCube(hdu_list=hdu)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Failed reading from PyFITS PrimaryHDU input.")

        # Second version: use file name
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readCube failed reading from filename input.")

        #
        # Test full I/O for an internally-generated fits data cube
        #

        # Build a list of images with different values
        ref_image = galsim.Image(ref_array.astype(array_type))
        image_list = []
        for k in range(nimages):
            image_list.append(ref_image + k)

        # Write the list to a fits data cube
        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits")
        galsim.fits.writeCube(image_list,test_cube_file)

        # Check pyfits read for sanity
        with pyfits.open(test_cube_file) as fits:
            test_array = fits[0].data

        wrong_type_error_msg = "%s != %s" % (test_array.dtype.type, types[i])
        if types[i] == np.uint16 or types[i] == np.uint32:
            # If astropy version < 1.1.0, uint fits files will be read wrongly, so skip this test
            # note that all other tests will pass since they will be read as float32s instead
            import astropy
            if LooseVersion(astropy.__version__) >= LooseVersion('1.1.0'):
                assert test_array.dtype.type == types[i], wrong_type_error_msg
        else:
            assert test_array.dtype.type == types[i], wrong_type_error_msg

        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]), test_array[k,:,:],
                    err_msg="PyFITS failing to read cube file.")

        # Then use galsim fits.readCube function
        # First version: use pyfits HDUList
        with pyfits.open(test_cube_file) as hdu:
            test_image_list = galsim.fits.readCube(hdu_list=hdu)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Failed reading from PyFITS PrimaryHDU input.")

        # Second version: use file name
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readCube failed reading from filename input.")

        #
        # Test writeCube with arrays, rather than images.
        #

        array_list = [ im.array for im in image_list ]
        galsim.fits.writeCube(array_list, test_cube_file)
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" write/readCube failed with list of numpy arrays.")

        one_array = np.asarray(array_list)
        galsim.fits.writeCube(one_array, test_cube_file)
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" write/readCube failed with single 3D numpy array.")

        #
        # Test writing to hdu_list directly and then writing to file.
        #

        # Start with empty hdu_list
        hdu_list = pyfits.HDUList()

        # Write the images to the hdu_list
        galsim.fits.writeCube(image_list, hdu_list=hdu_list)

        # Write it out with writeFile
        galsim.fits.writeFile(test_cube_file, hdu_list)

        # Check that reading it back in gives the same values
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readCube failed after using writeFile")

        assert_raises(ValueError, galsim.fits.readCube, test_cube_file, compression='invalid')
        assert_raises(ValueError, galsim.fits.writeCube, image_list, test_cube_file,
                      compression='invalid')
        assert_raises(ValueError, galsim.fits.writeFile, image_list, test_cube_file,
                      compression='invalid')
        assert_raises(OSError, galsim.fits.readCube, test_cube_file, compression='rice')
        assert_raises(OSError, galsim.fits.readCube, 'invalid.fits')

        assert_raises(TypeError, galsim.fits.readCube)
        assert_raises(TypeError, galsim.fits.readCube, test_cube_file, hdu_list=hdu)
        assert_raises(TypeError, galsim.fits.readCube, hdu_list=test_cube_file)
        assert_raises(TypeError, galsim.fits.writeCube)
        assert_raises(TypeError, galsim.fits.writeCube, image_list)
        assert_raises(TypeError, galsim.fits.writeCube, image_list,
                      file_name=test_cube_file, hdu_list=hdu_list)

        assert_raises(OSError, galsim.fits.writeCube, image_list, test_cube_file, clobber=False)

        assert_raises(ValueError, galsim.fits.writeCube, image_list[:0], test_cube_file)
        assert_raises(ValueError, galsim.fits.writeCube,
                      [image_list[0], image_list[1].subImage(galsim.BoundsI(1,4,1,4))],
                      test_cube_file)

        #
        # Test various compression schemes
        #

        # These tests are a bit slow, so we only bother to run them for the first dtype
        # when doing the regular unit tests.  When running python test_image.py, all of them
        # will run, so when working on the code, it is a good idea to run the tests that way.
        if i > 0 and __name__ != "__main__":
            continue

        test_cube_file0 = test_cube_file

        # Test full-file gzip
        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+".fits.gz")
        test_image_list = galsim.fits.readCube(test_cube_file, compression='gzip')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readCube failed for explicit full-file gzip")

        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readCube failed for auto full-file gzip")

        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits.gz")
        galsim.fits.writeCube(image_list,test_cube_file, compression='gzip')
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeCube failed for explicit full-file gzip")

        galsim.fits.writeCube(image_list,test_cube_file)
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeCube failed for auto full-file gzip")

        # With compression = None or 'none', astropy automatically figures it out anyway.
        test_image_list = galsim.fits.readCube(test_cube_file, compression=None)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeCube failed for auto full-file gzip")

        assert_raises(OSError, galsim.fits.readCube, test_cube_file0, compression='gzip')

        # Test full-file bzip2
        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+".fits.bz2")
        test_image_list = galsim.fits.readCube(test_cube_file, compression='bzip2')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readCube failed for explicit full-file bzip2")

        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readCube failed for auto full-file bzip2")

        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits.bz2")
        galsim.fits.writeCube(image_list,test_cube_file, compression='bzip2')
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeCube failed for explicit full-file bzip2")

        galsim.fits.writeCube(image_list,test_cube_file)
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeCube failed for auto full-file bzip2")

        # With compression = None or 'none', astropy automatically figures it out anyway.
        test_image_list = galsim.fits.readCube(test_cube_file, compression=None)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeCube failed for auto full-file gzip")

        assert_raises(OSError, galsim.fits.readCube, test_cube_file0, compression='bzip2')

        # Test rice
        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+".fits.fz")
        test_image_list = galsim.fits.readCube(test_cube_file, compression='rice')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readCube failed for explicit rice")

        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" readCube failed for auto rice")

        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits.fz")
        galsim.fits.writeCube(image_list,test_cube_file, compression='rice')
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeCube failed for explicit rice")

        galsim.fits.writeCube(image_list,test_cube_file)
        test_image_list = galsim.fits.readCube(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeCube failed for auto rice")

        assert_raises(OSError, galsim.fits.readCube, test_cube_file0, compression='rice')
        assert_raises(OSError, galsim.fits.readCube, test_cube_file, compression='none')

        # Test gzip_tile
        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits.gzt")
        galsim.fits.writeCube(image_list,test_cube_file, compression='gzip_tile')
        test_image_list = galsim.fits.readCube(test_cube_file, compression='gzip_tile')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                    test_image_list[k].array,
                    err_msg="Image"+tchar[i]+" writeCube failed for gzip_tile")

        assert_raises(OSError, galsim.fits.readCube, test_cube_file0, compression='gzip_tile')
        assert_raises(OSError, galsim.fits.readCube, test_cube_file, compression='none')

        # Test hcompress
        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits.hc")
        galsim.fits.writeCube(image_list,test_cube_file, compression='hcompress')
        test_image_list = galsim.fits.readCube(test_cube_file, compression='hcompress')
        for k in range(nimages):
            np.testing.assert_allclose((ref_array+k).astype(types[i]),
                    test_image_list[k].array, rtol=0.1,
                    err_msg="Image"+tchar[i]+" writeCube failed for hcompress")

        assert_raises(OSError, galsim.fits.readCube, test_cube_file0, compression='hcompress')
        assert_raises(OSError, galsim.fits.readCube, test_cube_file, compression='none')

        # Test plio (only valid on positive integer values)
        if tchar[i] in ['S', 'I']:
            test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits.plio")
            galsim.fits.writeCube(image_list,test_cube_file, compression='plio')
            test_image_list = galsim.fits.readCube(test_cube_file, compression='plio')
            for k in range(nimages):
                np.testing.assert_array_equal((ref_array+k).astype(types[i]),
                        test_image_list[k].array,
                        err_msg="Image"+tchar[i]+" writeCube failed for plio")

        assert_raises(OSError, galsim.fits.readCube, test_cube_file0, compression='plio')
        assert_raises(OSError, galsim.fits.readCube, test_cube_file, compression='none')


@timer
def test_Image_array_view():
    """Test that all six types of supported Images correctly provide a view on an input array.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image = galsim.Image(ref_array.astype(types[i]))
        np.testing.assert_array_equal(ref_array.astype(types[i]), image.array,
                err_msg="Array look into Image class does not match input for dtype = "+
                str(types[i]))

        #Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        image = image_init_func(ref_array.astype(types[i]))
        np.testing.assert_array_equal(ref_array.astype(types[i]), image.array,
                err_msg="Array look into Image class does not match input for dtype = "+
                str(types[i]))


@timer
def test_Image_binary_add():
    """Test that all six types of supported Images add correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = galsim.Image((2 * ref_array).astype(types[i]))
        image3 = image1 + image2
        np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image3.array,
                err_msg="Binary add in Image class does not match reference for dtype = "+
                str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image3 = image1 + image2
        np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image3.array,
                err_msg="Binary add in Image class does not match reference for dtype = "
                +str(types[i]))

        for j in range(ntypes):
            image2_init_func = eval("galsim.Image"+tchar[j])
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image3 = image1 + image2
            type3 = image3.array.dtype.type
            np.testing.assert_array_equal((3 * ref_array).astype(type3), image3.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

        # Check for exceptions if we try to do this operation for images without matching
        # shape.  Note that this test is only included here (not in the unit tests for all
        # other operations) because all operations have the same error-checking code, so it should
        # only be necessary to check once.
        with assert_raises(ValueError):
            image1 + image1.subImage(galsim.BoundsI(0,4,0,4))

@timer
def test_Image_binary_subtract():
    """Test that all six types of supported Images subtract correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = galsim.Image((2 * ref_array).astype(types[i]))
        image3 = image2 - image1
        np.testing.assert_array_equal(ref_array.astype(types[i]), image3.array,
                err_msg="Binary subtract in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image3 = image2 - image1
        np.testing.assert_array_equal(ref_array.astype(types[i]), image3.array,
                err_msg="Binary subtract in Image class does not match reference for dtype = "
                +str(types[i]))

        for j in range(ntypes):
            image2_init_func = eval("galsim.Image"+tchar[j])
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image3 = image2 - image1
            type3 = image3.array.dtype.type
            np.testing.assert_array_equal(ref_array.astype(type3), image3.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

        with assert_raises(ValueError):
            image1 - image1.subImage(galsim.BoundsI(0,4,0,4))


@timer
def test_Image_binary_multiply():
    """Test that all six types of supported Images multiply correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = galsim.Image((2 * ref_array).astype(types[i]))
        image3 = image1 * image2
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image3.array,
                err_msg="Binary multiply in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image3 = image1 * image2
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image3.array,
                err_msg="Binary multiply in Image class does not match reference for dtype = "
                +str(types[i]))

        # Check unary -
        image1 = galsim.Image(ref_array.astype(types[i]))
        image3 = -image1
        np.testing.assert_array_equal(image3.array, (-1 * ref_array).astype(types[i]),
                err_msg="Unary - in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))

        for j in range(ntypes):
            image2_init_func = eval("galsim.Image"+tchar[j])
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image3 = image2 * image1
            type3 = image3.array.dtype.type
            np.testing.assert_array_equal((2*ref_array**2).astype(type3), image3.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

        with assert_raises(ValueError):
            image1 * image1.subImage(galsim.BoundsI(0,4,0,4))


@timer
def test_Image_binary_divide():
    """Test that all six types of supported Images divide correctly.
    """
    # Note: tests here are not precisely equal, since division can have rounding errors for
    # some elements.  In particular when dividing by complex, where there is a bit more to the
    # generic calculation (even though the imaginary parts are zero here).
    # So check that they are *almost* equal to 12 digits of precision (or 4 for complex64).
    for i in range(ntypes):
        decimal = 4 if types[i] == np.complex64 else 12
        # First try using the dictionary-type Image init
        # Note that I am using refarray + 1 to avoid divide-by-zero.
        image1 = galsim.Image((ref_array + 1).astype(types[i]))
        image2 = galsim.Image((3 * (ref_array + 1)**2).astype(types[i]))
        image3 = image2 / image1
        np.testing.assert_almost_equal((3 * (ref_array + 1)).astype(types[i]), image3.array,
                decimal=decimal,
                err_msg="Binary divide in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = (large_array+1).astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func((3 * (ref_array + 1)**2).astype(types[i]))
        image3 = image2 / image1
        np.testing.assert_almost_equal((3 * (ref_array + 1)).astype(types[i]), image3.array,
                decimal=decimal,
                err_msg="Binary divide in Image class does not match reference for dtype = "
                +str(types[i]))

        for j in range(ntypes):
            decimal = 4 if (types[i] == np.complex64 or types[j] == np.complex64) else 12
            image2_init_func = eval("galsim.Image"+tchar[j])
            image2 = image2_init_func((3 * (ref_array+1)**2).astype(types[j]))
            image3 = image2 / image1
            type3 = image3.array.dtype.type
            np.testing.assert_almost_equal((3*(ref_array+1)).astype(type3), image3.array,
                    decimal=decimal,
                    err_msg="Inplace divide in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

        with assert_raises(ValueError):
            image1 / image1.subImage(galsim.BoundsI(0,4,0,4))


@timer
def test_Image_binary_scalar_add():
    """Test that all six types of supported Images add scalars correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = image1 + 3
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        image2 = 3 + image1
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary radd scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image1 + 3
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class does not match reference for dtype = "
                +str(types[i]))
        image2 = 3 + image1
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary radd scalar in Image class does not match reference for dtype = "
                +str(types[i]))


@timer
def test_Image_binary_scalar_subtract():
    """Test that all six types of supported Images binary scalar subtract correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = image1 - 3
        np.testing.assert_array_equal((ref_array - 3).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        image2 = 3 - image1
        np.testing.assert_array_equal((3 - ref_array).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image1 - 3
        np.testing.assert_array_equal((ref_array - 3).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class does not match reference for dtype = "
                +str(types[i]))
        image2 = 3 - image1
        np.testing.assert_array_equal((3 - ref_array).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class does not match reference for dtype = "
                +str(types[i]))


@timer
def test_Image_binary_scalar_multiply():
    """Test that all six types of supported Images binary scalar multiply correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = image1 * 3
        np.testing.assert_array_equal((ref_array * 3).astype(types[i]), image2.array,
                err_msg="Binary multiply scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        image2 = 3 * image1
        np.testing.assert_array_equal((ref_array * 3).astype(types[i]), image2.array,
                err_msg="Binary rmultiply scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image1 * 3
        np.testing.assert_array_equal((ref_array * 3).astype(types[i]), image2.array,
                err_msg="Binary multiply scalar in Image class does"
                +" not match reference for dtype = "+str(types[i]))
        image2 = 3 * image1
        np.testing.assert_array_equal((ref_array * 3).astype(types[i]), image2.array,
                err_msg="Binary rmultiply scalar in Image class does"
                +" not match reference for dtype = "+str(types[i]))


@timer
def test_Image_binary_scalar_divide():
    """Test that all six types of supported Images binary scalar divide correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image((3 * ref_array).astype(types[i]))
        image2 = image1 / 3
        np.testing.assert_array_equal(ref_array.astype(types[i]), image2.array,
                err_msg="Binary divide scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = (3*large_array).astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image1 / 3
        np.testing.assert_array_equal(ref_array.astype(types[i]), image2.array,
                err_msg="Binary divide scalar in Image class does"
                +" not match reference for dtype = "+str(types[i]))


@timer
def test_Image_binary_scalar_pow():
    """Test that all six types of supported Images can be raised to a power (scalar) correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = galsim.Image((ref_array**2).astype(types[i]))
        image3 = image1**2
        # Note: unlike for the tests above with multiplication, the test fails if I use
        # assert_array_equal.  I have to use assert_array_almost_equal to avoid failure due to
        # small numerical issues.
        np.testing.assert_array_almost_equal(image3.array, image2.array,
            decimal=4,
            err_msg="Binary pow scalar in Image class (dictionary call) does"
            +" not match reference for dtype = "+str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func(ref_array.astype(types[i]))
        image2 **= 2
        image3 = image1**2
        np.testing.assert_array_equal(image3.array, image2.array,
            err_msg="Binary pow scalar in Image class does"
            +" not match reference for dtype = "+str(types[i]))

        # float types can also be taken to a float power
        if types[i] in [np.float32, np.float64]:
            image2 = image_init_func((ref_array**(1/1.3)).astype(types[i]))
            image3 = image2**1.3
            # Note: unlike for the tests above with multiplication/division, the test fails if I use
            # assert_array_equal.  I have to use assert_array_almost_equal to avoid failure due to
            # small numerical issues.
            np.testing.assert_array_almost_equal(ref_array.astype(types[i]), image3.array,
                decimal=4,
                err_msg="Binary pow scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))

        with assert_raises(TypeError):
            image1 ** image2


@timer
def test_Image_inplace_add():
    """Test that all six types of supported Images inplace add correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = galsim.Image((2 * ref_array).astype(types[i]))
        image1 += image2
        np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image1.array,
                err_msg="Inplace add in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.copy().astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image1 += image2
        np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image1.array,
                err_msg="Inplace add in Image class does not match reference for dtype = "
                +str(types[i]))

        for j in range(i): # Only add simpler types to this one.
            image2_init_func = eval("galsim.Image"+tchar[j])
            slice_array = large_array.copy().astype(types[i])[::3,::2]
            image1 = image_init_func(slice_array)
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image1 += image2
            np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image1.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

        with assert_raises(ValueError):
            image1 += image1.subImage(galsim.BoundsI(0,4,0,4))


@timer
def test_Image_inplace_subtract():
    """Test that all six types of supported Images inplace subtract correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image((2 * ref_array).astype(types[i]))
        image2 = galsim.Image(ref_array.astype(types[i]))
        image1 -= image2
        np.testing.assert_array_equal(ref_array.astype(types[i]), image1.array,
                err_msg="Inplace subtract in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = (2*large_array).astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func(ref_array.astype(types[i]))
        image1 -= image2
        np.testing.assert_array_equal(ref_array.astype(types[i]), image1.array,
                err_msg="Inplace subtract in Image class does"
                +" not match reference for dtype = "+str(types[i]))

        for j in range(i): # Only subtract simpler types from this one.
            image2_init_func = eval("galsim.Image"+tchar[j])
            slice_array = (2*large_array).astype(types[i])[::3,::2]
            image1 = image_init_func(slice_array)
            image2 = image2_init_func(ref_array.astype(types[j]))
            image1 -= image2
            np.testing.assert_array_equal(ref_array.astype(types[i]), image1.array,
                    err_msg="Inplace subtract in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

        with assert_raises(ValueError):
            image1 -= image1.subImage(galsim.BoundsI(0,4,0,4))


@timer
def test_Image_inplace_multiply():
    """Test that all six types of supported Images inplace multiply correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = galsim.Image((2 * ref_array).astype(types[i]))
        image1 *= image2
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image1.array,
                err_msg="Inplace multiply in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.copy().astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image1 *= image2
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image1.array,
                err_msg="Inplace multiply in Image class does not match reference for dtype = "
                +str(types[i]))

        for j in range(i): # Only multiply simpler types to this one.
            image2_init_func = eval("galsim.Image"+tchar[j])
            slice_array = large_array.copy().astype(types[i])[::3,::2]
            image1 = image_init_func(slice_array)
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image1 *= image2
            np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image1.array,
                    err_msg="Inplace multiply in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

        with assert_raises(ValueError):
            image1 *= image1.subImage(galsim.BoundsI(0,4,0,4))


@timer
def test_Image_inplace_divide():
    """Test that all six types of supported Images inplace divide correctly.
    """
    for i in range(ntypes):
        decimal = 4 if types[i] == np.complex64 else 12
        # First try using the dictionary-type Image init
        image1 = galsim.Image((2 * (ref_array + 1)**2).astype(types[i]))
        image2 = galsim.Image((ref_array + 1).astype(types[i]))
        image1 /= image2
        np.testing.assert_almost_equal((2 * (ref_array + 1)).astype(types[i]), image1.array,
                decimal=decimal,
                err_msg="Inplace divide in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = (2*(large_array+1)**2).astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func((ref_array + 1).astype(types[i]))
        image1 /= image2
        np.testing.assert_almost_equal((2 * (ref_array + 1)).astype(types[i]), image1.array,
                decimal=decimal,
                err_msg="Inplace divide in Image class does not match reference for dtype = "
                +str(types[i]))

        # Test image.invertSelf()
        # Intentionally make some elements zero, so we test that 1/0 -> 0.
        image1 = galsim.Image((ref_array // 11 - 3).astype(types[i]))
        image2 = image1.copy()
        mask1 = image1.array == 0
        mask2 = image1.array != 0
        image2.invertSelf()
        np.testing.assert_array_equal(image2.array[mask1], 0,
                err_msg="invertSelf did not do 1/0 -> 0.")
        np.testing.assert_array_equal(image2.array[mask2],
                (1./image1.array[mask2]).astype(types[i]),
                err_msg="invertSelf gave wrong answer for non-zero elements")

        for j in range(i): # Only divide simpler types into this one.
            decimal = 4 if (types[i] == np.complex64 or types[j] == np.complex64) else 12
            image2_init_func = eval("galsim.Image"+tchar[j])
            slice_array = (2*(large_array+1)**2).astype(types[i])[::3,::2]
            image1 = image_init_func(slice_array)
            image2 = image2_init_func((ref_array+1).astype(types[j]))
            image1 /= image2
            np.testing.assert_almost_equal((2 * (ref_array+1)).astype(types[i]), image1.array,
                    decimal=decimal,
                    err_msg="Inplace divide in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

        with assert_raises(ValueError):
            image1 /= image1.subImage(galsim.BoundsI(0,4,0,4))


@timer
def test_Image_inplace_scalar_add():
    """Test that all six types of supported Images inplace scalar add correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image1 += 1
        np.testing.assert_array_equal((ref_array + 1).astype(types[i]), image1.array,
                err_msg="Inplace scalar add in Image class (dictionary "
                +"call) does not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.copy().astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image1 += 1
        np.testing.assert_array_equal((ref_array + 1).astype(types[i]), image1.array,
                err_msg="Inplace scalar add in Image class does not match reference for dtype = "
                +str(types[i]))


@timer
def test_Image_inplace_scalar_subtract():
    """Test that all six types of supported Images inplace scalar subtract correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image1 -= 1
        np.testing.assert_array_equal((ref_array - 1).astype(types[i]), image1.array,
                err_msg="Inplace scalar subtract in Image class (dictionary "
                +"call) does not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.copy().astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image1 -= 1
        np.testing.assert_array_equal((ref_array - 1).astype(types[i]), image1.array,
                err_msg="Inplace scalar subtract in Image class does"
                +" not match reference for dtype = "+str(types[i]))


@timer
def test_Image_inplace_scalar_multiply():
    """Test that all six types of supported Images inplace scalar multiply correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = galsim.Image((2 * ref_array).astype(types[i]))
        image1 *= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                err_msg="Inplace scalar multiply in Image class (dictionary "
                +"call) does not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.copy().astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image1 *= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                err_msg="Inplace scalar multiply in Image class does"
                +" not match reference for dtype = "+str(types[i]))


@timer
def test_Image_inplace_scalar_divide():
    """Test that all six types of supported Images inplace scalar divide correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image(ref_array.astype(types[i]))
        image2 = galsim.Image((2 * ref_array).astype(types[i]))
        image2 /= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                err_msg="Inplace scalar divide in Image class (dictionary "
                +"call) does not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = (2*large_array).astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image1 /= 2
        np.testing.assert_array_equal(ref_array.astype(types[i]), image1.array,
                err_msg="Inplace scalar divide in Image class does"
                +" not match reference for dtype = "+str(types[i]))


@timer
def test_Image_inplace_scalar_pow():
    """Test that all six types of supported Images can be raised (in-place) to a scalar correctly.
    """
    for i in range(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.Image((ref_array**2).astype(types[i]))
        image2 = galsim.Image(ref_array.astype(types[i]))
        image2 **= 2
        np.testing.assert_array_almost_equal(image1.array, image2.array, decimal=4,
            err_msg="Inplace scalar pow in Image class (dictionary "
            +"call) does not match reference for dtype = "+str(types[i]))

        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        slice_array = large_array.copy().astype(types[i])[::3,::2]
        image1 = image_init_func(slice_array)
        image2 = image_init_func((ref_array.astype(types[i]))**2)
        image1 **= 2
        np.testing.assert_array_equal(image1.array, image2.array,
            err_msg="Inplace scalar pow in Image class does"
            +" not match reference for dtype = "+str(types[i]))

        # float types can also be taken to a float power
        if types[i] in [np.float32, np.float64]:
            # First try using the dictionary-type Image init
            image1 = galsim.Image(ref_array.astype(types[i]))
            image2 = galsim.Image((ref_array**(1./1.3)).astype(types[i]))
            image2 **= 1.3
            np.testing.assert_array_almost_equal(image1.array, image2.array, decimal=4,
                err_msg="Inplace scalar pow in Image class (dictionary "
                +"call) does not match reference for dtype = "+str(types[i]))

        with assert_raises(TypeError):
            image1 **= image2

@timer
def test_Image_subImage():
    """Test that subImages are accessed and written correctly.
    """
    for i in range(ntypes):
        image = galsim.Image(ref_array.astype(types[i]))
        bounds = galsim.BoundsI(3,4,2,3)
        sub_array = np.array([[32, 42], [33, 43]]).astype(types[i])
        np.testing.assert_array_equal(image.subImage(bounds).array, sub_array,
            err_msg="image.subImage(bounds) does not match reference for dtype = "+str(types[i]))
        np.testing.assert_array_equal(image[bounds].array, sub_array,
            err_msg="image[bounds] does not match reference for dtype = "+str(types[i]))
        image[bounds] = galsim.Image(sub_array+100)
        np.testing.assert_array_equal(image[bounds].array, (sub_array+100),
            err_msg="image[bounds] = im2 does not set correctly for dtype = "+str(types[i]))
        for xpos in range(1,test_shape[0]+1):
            for ypos in range(1,test_shape[1]+1):
                if (xpos >= bounds.xmin and xpos <= bounds.xmax and
                    ypos >= bounds.ymin and ypos <= bounds.ymax):
                    value = ref_array[ypos-1,xpos-1] + 100
                else:
                    value = ref_array[ypos-1,xpos-1]
                assert image(xpos,ypos) == value,  \
                    "image[bounds] = im2 set wrong locations for dtype = "+str(types[i])

        image = galsim.Image(ref_array.astype(types[i]))
        image[bounds] += 100
        np.testing.assert_array_equal(image[bounds].array, (sub_array+100),
            err_msg="image[bounds] += 100 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.Image(sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] += 100 set wrong locations for dtype = "+str(types[i]))

        image = galsim.Image(ref_array.astype(types[i]))
        image[bounds] -= 100
        np.testing.assert_array_equal(image[bounds].array, (sub_array-100),
            err_msg="image[bounds] -= 100 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.Image(sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] -= 100 set wrong locations for dtype = "+str(types[i]))

        image = galsim.Image(ref_array.astype(types[i]))
        image[bounds] *= 100
        np.testing.assert_array_equal(image[bounds].array, (sub_array*100),
            err_msg="image[bounds] *= 100 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.Image(sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] *= 100 set wrong locations for dtype = "+str(types[i]))

        image = galsim.Image((100*ref_array).astype(types[i]))
        image[bounds] /= 100
        np.testing.assert_array_equal(image[bounds].array, (sub_array),
            err_msg="image[bounds] /= 100 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.Image((100*sub_array).astype(types[i]))
        np.testing.assert_array_equal(image.array, (100*ref_array),
            err_msg="image[bounds] /= 100 set wrong locations for dtype = "+str(types[i]))

        im2 = galsim.Image(sub_array)
        image = galsim.Image(ref_array.astype(types[i]))
        image[bounds] += im2
        np.testing.assert_array_equal(image[bounds].array, (2*sub_array),
            err_msg="image[bounds] += im2 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.Image(sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] += im2 set wrong locations for dtype = "+str(types[i]))

        image = galsim.Image(2*ref_array.astype(types[i]))
        image[bounds] -= im2
        np.testing.assert_array_equal(image[bounds].array, sub_array,
            err_msg="image[bounds] -= im2 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.Image((2*sub_array).astype(types[i]))
        np.testing.assert_array_equal(image.array, (2*ref_array),
            err_msg="image[bounds] -= im2 set wrong locations for dtype = "+str(types[i]))

        image = galsim.Image(ref_array.astype(types[i]))
        image[bounds] *= im2
        np.testing.assert_array_equal(image[bounds].array, (sub_array**2),
            err_msg="image[bounds] *= im2 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.Image(sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] *= im2 set wrong locations for dtype = "+str(types[i]))

        image = galsim.Image((2 * ref_array**2).astype(types[i]))
        image[bounds] /= im2
        np.testing.assert_array_equal(image[bounds].array, (2*sub_array),
            err_msg="image[bounds] /= im2 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.Image((2*sub_array**2).astype(types[i]))
        np.testing.assert_array_equal(image.array, (2*ref_array**2),
            err_msg="image[bounds] /= im2 set wrong locations for dtype = "+str(types[i]))

        do_pickle(image)

    assert_raises(TypeError, image.subImage, bounds=None)
    assert_raises(TypeError, image.subImage, bounds=galsim.BoundsD(0,4,0,4))


def make_subImage(file_name, bounds):
    """Helper function for test_subImage_persistence
    """
    full_im = galsim.fits.read(file_name)
    stamp = full_im.subImage(bounds)
    return stamp

@timer
def test_subImage_persistence():
    """Test that a subimage is properly accessible even if the original image has gone out
    of scope.
    """
    file_name = os.path.join('fits_files','tpv.fits')
    bounds = galsim.BoundsI(123, 133, 45, 55)  # Something random

    # In this case, the original image has gone out of scope.  At least on some systems,
    # this used to caus a seg fault when accessing stamp1.array.  (BAD!)
    stamp1 = make_subImage(file_name, bounds)
    print('stamp1 = ',stamp1.array)

    full_im = galsim.fits.read(file_name)
    stamp2 = full_im.subImage(bounds)
    print('stamp2 = ',stamp2.array)

    np.testing.assert_array_equal(stamp1.array, stamp2.array)

@timer
def test_Image_resize():
    """Test that the Image resize function works correctly.
    """
    # Use a random number generator for some values here.
    ud = galsim.UniformDeviate(515324)

    for i in range(ntypes):

        # Resize to a bunch of different shapes (larger and smaller) to test reallocations
        for shape in [ (10,10), (3,20), (21,8), (1,3), (13,30) ]:

            # im1 starts with basic constructor with a given size
            array_type = types[i]
            im1 = galsim.Image(5,5, dtype=array_type, scale=0.1)

            # im2 stars with null constructor
            im2 = galsim.Image(dtype=array_type, scale=0.2)

            # im3 is a view into a larger image
            im3_full = galsim.Image(10,10, dtype=array_type, init_value=23, scale=0.3)
            im3 = im3_full.subImage(galsim.BoundsI(1,6,1,6))

            # Make sure at least one of the _arrays is instantiated.  This isn't required,
            # but we used to have bugs if the array was instantiated before resizing.
            # So test im1 and im3 being instantiated and im2 not instantiated.
            np.testing.assert_array_equal(im1.array, 0, "im1 is not initially all 0.")
            np.testing.assert_array_equal(im3.array, 23, "im3 is not initially all 23.")

            # Have the xmin, ymin value be random numbers from -100..100:
            xmin = int(ud() * 200) - 100
            ymin = int(ud() * 200) - 100
            xmax = xmin + shape[1] - 1
            ymax = ymin + shape[0] - 1
            b = galsim.BoundsI(xmin, xmax, ymin, ymax)
            im1.resize(b)
            im2.resize(b)
            im3.resize(b, wcs=galsim.PixelScale(0.33))

            np.testing.assert_equal(
                b, im1.bounds, err_msg="im1 has wrong bounds after resize to b = %s"%b)
            np.testing.assert_equal(
                b, im2.bounds, err_msg="im2 has wrong bounds after resize to b = %s"%b)
            np.testing.assert_equal(
                b, im3.bounds, err_msg="im3 has wrong bounds after resize to b = %s"%b)
            np.testing.assert_array_equal(
                im1.array.shape, shape, err_msg="im1.array.shape wrong after resize")
            np.testing.assert_array_equal(
                im2.array.shape, shape, err_msg="im2.array.shape wrong after resize")
            np.testing.assert_array_equal(
                im3.array.shape, shape, err_msg="im3.array.shape wrong after resize")
            np.testing.assert_equal(
                im1.scale, 0.1, err_msg="im1 has wrong scale after resize to b = %s"%b)
            np.testing.assert_equal(
                im2.scale, 0.2, err_msg="im2 has wrong scale after resize to b = %s"%b)
            np.testing.assert_equal(
                im3.scale, 0.33, err_msg="im3 has wrong scale after resize to b = %s"%b)

            # Fill the images with random numbers
            for x in range(xmin,xmax+1):
                for y in range(ymin,ymax+1):
                    val = simple_types[i](ud()*500)
                    im1.setValue(x,y,val)
                    im2._setValue(x,y,val)
                    im3.setValue(x,y,val)

            # They should be equal now.  This doesn't completely guarantee that nothing is
            # wrong, but hopefully if we are misallocating memory here, something will be
            # clobbered or we will get a seg fault.
            np.testing.assert_array_equal(
                im1.array, im2.array, err_msg="im1 != im2 after resize to b = %s"%b)
            np.testing.assert_array_equal(
                im1.array, im3.array, err_msg="im1 != im3 after resize to b = %s"%b)
            np.testing.assert_array_equal(
                im2.array, im3.array, err_msg="im2 != im3 after resize to b = %s"%b)

            # Also, since the view was resized, it should no longer be coupled to the original.
            np.testing.assert_array_equal(
                im3_full.array, 23, err_msg="im3_full changed")

            do_pickle(im1)
            do_pickle(im2)
            do_pickle(im3)

    assert_raises(TypeError, im1.resize, bounds=None)
    assert_raises(TypeError, im1.resize, bounds=galsim.BoundsD(0,5,0,5))


@timer
def test_ConstImage_array_constness():
    """Test that Image instances with make_const=True cannot be modified via their .array
    attributes, and that if this is attempted a GalSimImmutableError is raised.
    """
    for i in range(ntypes):
        image = galsim.Image(ref_array.astype(types[i]), make_const=True)
        # Apparently older numpy versions might raise a RuntimeError, a ValueError, or a TypeError
        # when trying to write to arrays that have writeable=False.
        # From the numpy 1.7.0 release notes:
        #     Attempting to write to a read-only array (one with
        #     ``arr.flags.writeable`` set to ``False``) used to raise either a
        #     RuntimeError, ValueError, or TypeError inconsistently, depending on
        #     which code path was taken. It now consistently raises a ValueError.
        with assert_raises((RuntimeError, ValueError, TypeError)):
            image.array[1, 2] = 666

        # Native image operations that are invalid just raise GalSimImmutableError
        with assert_raises(galsim.GalSimImmutableError):
            image[1, 2] = 666

        with assert_raises(galsim.GalSimImmutableError):
            image.setValue(1,2,666)

        with assert_raises(galsim.GalSimImmutableError):
            image[image.bounds] = image

        # The rest are functions, so just use assert_raises.
        assert_raises(galsim.GalSimImmutableError, image.setValue, 1, 2, 666)
        assert_raises(galsim.GalSimImmutableError, image.setSubImage, image.bounds, image)
        assert_raises(galsim.GalSimImmutableError, image.addValue, 1, 2, 666)
        assert_raises(galsim.GalSimImmutableError, image.copyFrom, image)
        assert_raises(galsim.GalSimImmutableError, image.resize, image.bounds)
        assert_raises(galsim.GalSimImmutableError, image.fill, 666)
        assert_raises(galsim.GalSimImmutableError, image.setZero)
        assert_raises(galsim.GalSimImmutableError, image.invertSelf)

        do_pickle(image)


@timer
def test_BoundsI_init_with_non_pure_ints():
    """Test that BoundsI converts its input args to int variables on input.
    """
    ref_bound_vals = (5, 35, 1, 48)
    xmin_test, xmax_test, ymin_test, ymax_test = ref_bound_vals
    ref_bounds = galsim.BoundsI(xmin_test, xmax_test, ymin_test, ymax_test)
    bound_arr_int = np.asarray(ref_bound_vals, dtype=int)
    bound_arr_flt = np.asarray(ref_bound_vals, dtype=float)
    bound_arr_flt_nonint = bound_arr_flt + 0.3

    # Check kwarg initialization:
    assert ref_bounds == galsim.BoundsI(
        xmin=bound_arr_int[0], xmax=bound_arr_int[1],
        ymin=bound_arr_int[2], ymax=bound_arr_int[3]), \
        "Cannot initialize a BoundI with int array elements"
    assert ref_bounds == galsim.BoundsI(
        xmin=bound_arr_flt[0], xmax=bound_arr_flt[1],
        ymin=bound_arr_flt[2], ymax=bound_arr_flt[3]), \
        "Cannot initialize a BoundI with float array elements"

    # Check arg initialization:
    assert ref_bounds == galsim.BoundsI(*bound_arr_int), \
        "Cannot initialize a BoundI with int array elements"
    assert ref_bounds == galsim.BoundsI(*bound_arr_flt), \
        "Cannot initialize a BoundI with float array elements"

    # Using non-integers should raise a TypeError
    assert_raises(TypeError, galsim.BoundsI, *bound_arr_flt_nonint)
    assert_raises(TypeError, galsim.BoundsI,
                  xmin=bound_arr_flt_nonint[0], xmax=bound_arr_flt_nonint[1],
                  ymin=bound_arr_flt_nonint[2], ymax=bound_arr_flt_nonint[3])


@timer
def test_Image_constructor():
    """Check that the Image constructor that takes NumPy array does not mangle input.
    """
    # Loop over types.
    for i in range(ntypes):

        array_dtype = np.dtype(types[i])

        # Make a NumPy array directly, with non-trivially interesting values.
        test_arr = np.ones((3,4), dtype=types[i])
        test_arr[1,3] = -5
        test_arr[2,2] = 7
        # Initialize the Image from it.
        test_im = galsim.Image(test_arr)
        # Check that the image.array attribute matches the original.
        np.testing.assert_array_equal(
            test_arr, test_im.array,
            err_msg="Image constructor mangled input NumPy array.")

        # Now make an opposite-endian Numpy array, to initialize the Image.
        new_type = array_dtype.newbyteorder('S')
        test_arr = np.ones((3,4), dtype=new_type)
        test_arr[1,3] = -5
        test_arr[2,2] = 7
        # Initialize the Image from it.
        test_im = galsim.Image(test_arr)
        # Check that the image.array attribute matches the original.
        np.testing.assert_array_equal(
            test_arr, test_im.array,
            err_msg="Image constructor mangled input NumPy array (endian issues).")

        do_pickle(test_im)

        # Check that some invalid sets of construction args raise the appropriate errors
        # Invalid args
        assert_raises(TypeError, galsim.Image, 1, 2, 3)
        assert_raises(TypeError, galsim.Image, 128)
        assert_raises(TypeError, galsim.Image, 1.8)
        assert_raises(TypeError, galsim.Image, 1.3, 2.7)
        # Invalid array kwarg
        assert_raises(TypeError, galsim.Image, array=5)
        assert_raises(TypeError, galsim.Image, array=test_im)
        # Invalid image kwarg
        assert_raises(TypeError, galsim.Image, image=5)
        assert_raises(TypeError, galsim.Image, image=test_arr)
        # Invalid bounds
        assert_raises(TypeError, galsim.Image, bounds=(1,4,1,3))
        assert_raises(TypeError, galsim.Image, bounds=galsim.BoundsD(1,4,1,3))
        assert_raises(TypeError, galsim.Image, array=test_arr, bounds=(1,4,1,3))
        assert_raises(ValueError, galsim.Image, array=test_arr, bounds=galsim.BoundsI(1,3,1,4))
        assert_raises(ValueError, galsim.Image, array=test_arr, bounds=galsim.BoundsI(1,4,1,1))
        # Invalid ncol, nrow
        assert_raises(TypeError, galsim.Image, ncol=1.2, nrow=3)
        assert_raises(TypeError, galsim.Image, ncol=2, nrow=3.4)
        assert_raises(ValueError, galsim.Image, ncol='four', nrow='three')
        # Invalid dtype
        assert_raises(ValueError, galsim.Image, array=test_arr, dtype=bool)
        assert_raises(ValueError, galsim.Image, array=test_arr.astype(bool))
        # Invalid scale
        assert_raises(ValueError, galsim.Image, 4,3, scale='invalid')
        # Invalid wcs
        assert_raises(TypeError, galsim.Image, 4,3, wcs='invalid')
        # Disallowed combinations
        assert_raises(TypeError, galsim.Image, ncol=4, nrow=3, bounds=galsim.BoundsI(1,4,1,3))
        assert_raises(TypeError, galsim.Image, ncol=4, nrow=3, array=test_arr)
        assert_raises(TypeError, galsim.Image, ncol=4, nrow=3, image=test_im)
        assert_raises(TypeError, galsim.Image, ncol=4)
        assert_raises(TypeError, galsim.Image, nrow=3)
        assert_raises(ValueError, galsim.Image, test_arr, bounds=galsim.BoundsI(1,2,1,3))
        assert_raises(ValueError, galsim.Image, array=test_arr, bounds=galsim.BoundsI(1,2,1,3))
        assert_raises(ValueError, galsim.Image, [[1,2]], bounds=galsim.BoundsI(1,2,1,3))
        assert_raises(TypeError, galsim.Image, test_arr, init_value=3)
        assert_raises(TypeError, galsim.Image, array=test_arr, init_value=3)
        assert_raises(TypeError, galsim.Image, test_im, init_value=3)
        assert_raises(TypeError, galsim.Image, image=test_im, init_value=3)
        assert_raises(TypeError, galsim.Image, dtype=float, init_value=3)
        assert_raises(TypeError, galsim.Image, test_im, scale=3, wcs=galsim.PixelScale(3))
        # Extra kwargs
        assert_raises(TypeError, galsim.Image, image=test_im, name='invalid')


@timer
def test_Image_view():
    """Test the functionality of image.view(...)
    """
    im = galsim.ImageI(25,25, wcs=galsim.AffineTransform(0.23,0.01,-0.02,0.22,
                       galsim.PositionI(13,13)))
    im._fill(17)
    assert im.wcs == galsim.AffineTransform(0.23,0.01,-0.02,0.22, galsim.PositionI(13,13))
    assert im.bounds == galsim.BoundsI(1,25,1,25)
    assert im(11,19) == 17  # I'll keep editing this pixel to new values.
    do_pickle(im)

    # Test view with no arguments
    imv = im.view()
    assert imv.wcs == im.wcs
    assert imv.bounds == im.bounds
    imv.setValue(11,19, 20)
    assert imv(11,19) == 20
    assert im(11,19) == 20
    do_pickle(im)
    do_pickle(imv)

    # Test view with new origin
    imv = im.view(origin=(0,0))
    assert im.wcs == galsim.AffineTransform(0.23,0.01,-0.02,0.22, galsim.PositionI(13,13))
    assert imv.wcs == galsim.AffineTransform(0.23,0.01,-0.02,0.22, galsim.PositionI(12,12))
    assert im.bounds == galsim.BoundsI(1,25,1,25)
    assert imv.bounds == galsim.BoundsI(0,24,0,24)
    imv.setValue(10,18, 30)
    assert imv(10,18) == 30
    assert im(11,19) == 30
    imv2 = im.view()
    imv2.setOrigin(0,0)
    assert imv.bounds == imv2.bounds
    assert imv.wcs == imv2.wcs
    do_pickle(imv)
    do_pickle(imv2)

    # Test view with new center
    imv = im.view(center=(0,0))
    assert im.wcs == galsim.AffineTransform(0.23,0.01,-0.02,0.22, galsim.PositionI(13,13))
    assert imv.wcs == galsim.AffineTransform(0.23,0.01,-0.02,0.22, galsim.PositionI(0,0))
    assert im.bounds == galsim.BoundsI(1,25,1,25)
    assert imv.bounds == galsim.BoundsI(-12,12,-12,12)
    imv.setValue(-2,6, 40)
    assert imv(-2,6) == 40
    assert im(11,19) == 40
    imv2 = im.view()
    imv2.setCenter(0,0)
    assert imv.bounds == imv2.bounds
    assert imv.wcs == imv2.wcs
    with assert_raises(galsim.GalSimError):
        imv.scale   # scale is invalid if wcs is not a PixelScale
    do_pickle(imv)
    do_pickle(imv2)

    # Test view with new scale
    imv = im.view(scale=0.17)
    assert im.wcs == galsim.AffineTransform(0.23,0.01,-0.02,0.22, galsim.PositionI(13,13))
    assert imv.wcs == galsim.PixelScale(0.17)
    assert imv.bounds == im.bounds
    imv.setValue(11,19, 50)
    assert imv(11,19) == 50
    assert im(11,19) == 50
    imv2 = im.view()
    with assert_raises(galsim.GalSimError):
        imv2.scale = 0.17   # Invalid if wcs is not PixelScale
    imv2.wcs = None
    imv2.scale = 0.17
    assert imv.bounds == imv2.bounds
    assert imv.wcs == imv2.wcs
    do_pickle(imv)
    do_pickle(imv2)

    # Test view with new wcs
    imv = im.view(wcs=galsim.JacobianWCS(0., 0.23, -0.23, 0.))
    assert im.wcs == galsim.AffineTransform(0.23,0.01,-0.02,0.22, galsim.PositionI(13,13))
    assert imv.wcs == galsim.JacobianWCS(0., 0.23, -0.23, 0.)
    assert imv.bounds == im.bounds
    imv.setValue(11,19, 60)
    assert imv(11,19) == 60
    assert im(11,19) == 60
    imv2 = im.view()
    imv2.wcs = galsim.JacobianWCS(0.,0.23,-0.23,0.)
    assert imv.bounds == imv2.bounds
    assert imv.wcs == imv2.wcs
    do_pickle(imv)
    do_pickle(imv2)

    # Go back to original value for that pixel and make sure all are still equal to 17
    im.setValue(11,19, 17)
    assert im.array.min() == 17
    assert im.array.max() == 17

    assert_raises(TypeError, im.view, origin=(0,0), center=(0,0))
    assert_raises(TypeError, im.view, scale=0.3, wcs=galsim.JacobianWCS(1.1, 0.1, 0.1, 1.))
    assert_raises(TypeError, im.view, scale=galsim.PixelScale(0.3))
    assert_raises(TypeError, im.view, wcs=0.3)


@timer
def test_Image_writeheader():
    """Test the functionality of image.write(...) for images that have header attributes
    """
    # First check: if we have an image.header attribute, it gets written to file.
    im_test = galsim.Image(10, 10)
    key_name = 'test_key'
    im_test.header = galsim.FitsHeader(header={key_name : 'test_val'})
    test_file = os.path.join(datadir, "test_header.fits")
    im_test.write(test_file)
    new_header = galsim.FitsHeader(test_file)
    assert key_name.upper() in new_header.keys()

    # Second check: if we have an image.header attribute that modifies some keywords used by the
    # WCS, then make sure it doesn't overwrite the WCS.
    im_test.wcs = galsim.JacobianWCS(0., 0.23, -0.23, 0.)
    im_test.header = galsim.FitsHeader(header={'CD1_1' : 10., key_name : 'test_val'})
    im_test.write(test_file)
    new_header = galsim.FitsHeader(test_file)
    assert key_name.upper() in new_header.keys()
    assert new_header['CD1_1'] == 0.0

    # If clobbert = False, then trying to overwrite will raise an OSError
    assert_raises(OSError, im_test.write, test_file, clobber=False)


@timer
def test_ne():
    """ Check that inequality works as expected."""
    array1 = np.arange(32*32).reshape(32, 32).astype(float)
    array2 = array1.copy()
    array2[15, 15] += 2

    objs = [galsim.ImageD(array1),
            galsim.ImageD(array2),
            galsim.ImageD(array2, make_const=True),
            galsim.ImageD(array1, wcs=galsim.PixelScale(0.2)),
            galsim.ImageD(array1, xmin=2)]
    all_obj_diff(objs)


@timer
def test_copy():
    """Test different ways to copy an Image.
    """
    wcs=galsim.AffineTransform(0.23,0.01,-0.02,0.22, galsim.PositionI(13,13))
    im = galsim.Image(25,25, wcs=wcs)
    gn = galsim.GaussianNoise(sigma=1.7)
    im.addNoise(gn)

    assert im.wcs == galsim.AffineTransform(0.23,0.01,-0.02,0.22, galsim.PositionI(13,13))
    assert im.bounds == galsim.BoundsI(1,25,1,25)

    # Simplest way to copy is copy()
    im2 = im.copy()
    assert im2.wcs == im.wcs
    assert im2.bounds == im.bounds
    np.testing.assert_array_equal(im2.array, im.array)

    # Make sure it actually copied the array, not just made a view of it.
    im2.setValue(3,8,11.)
    assert im(3,8) != 11.

    # Can also use constructor to "copy"
    im3 = galsim.Image(im)
    assert im3.wcs == im.wcs
    assert im3.bounds == im.bounds
    np.testing.assert_array_equal(im3.array, im.array)
    im3.setValue(3,8,11.)
    assert im(3,8) != 11.

    # If copy=False is specified, then it shares the same array
    im3b = galsim.Image(im, copy=False)
    assert im3b.wcs == im.wcs
    assert im3b.bounds == im.bounds
    np.testing.assert_array_equal(im3b.array, im.array)
    im3b.setValue(2,3,2.)
    assert im3b(2,3) == 2.
    assert im(2,3) == 2.

    # Constructor can change the wcs
    im4 = galsim.Image(im, scale=0.6)
    assert im4.wcs != im.wcs            # wcs is not equal this time.
    assert im4.bounds == im.bounds
    np.testing.assert_array_equal(im4.array, im.array)
    im4.setValue(3,8,11.)
    assert im(3,8) != 11.

    im5 = galsim.Image(im, wcs=galsim.PixelScale(1.4))
    assert im5.wcs != im.wcs            # wcs is not equal this time.
    assert im5.bounds == im.bounds
    np.testing.assert_array_equal(im5.array, im.array)
    im5.setValue(3,8,11.)
    assert im(3,8) != 11.

    im6 = galsim.Image(im, wcs=wcs)
    assert im6.wcs == im.wcs            # This is the same wcs now.
    assert im6.bounds == im.bounds
    np.testing.assert_array_equal(im6.array, im.array)
    im6.setValue(3,8,11.)
    assert im(3,8) != 11.

    # Can also change the dtype
    im7 = galsim.Image(im, dtype=float)
    assert im7.wcs == im.wcs
    assert im7.bounds == im.bounds
    np.testing.assert_array_equal(im7.array, im.array)
    im7.setValue(3,8,11.)
    assert im(3,8) != 11.

    im8 = galsim.Image(im, wcs=wcs, dtype=float)
    assert im8.wcs == im.wcs            # This is the same wcs now.
    assert im8.bounds == im.bounds
    np.testing.assert_array_equal(im8.array, im.array)
    im8.setValue(3,8,11.)
    assert im(3,8) != 11.

    # Check that a slice image copies correctly
    slice_array = large_array.astype(complex)[::3,::2]
    im_slice = galsim.Image(slice_array, wcs=wcs)
    im9 = im_slice.copy()
    assert im9.wcs == im_slice.wcs
    assert im9.bounds == im_slice.bounds
    np.testing.assert_array_equal(im9.array, im_slice.array)
    im9.setValue(2,3,11.)
    assert im9(2,3) == 11.
    assert im_slice(2,3) != 11.

    # Can also copy by giving the array and specify copy=True
    im10 = galsim.Image(im.array, bounds=im.bounds, wcs=im.wcs, copy=False)
    assert im10.wcs == im.wcs
    assert im10.bounds == im.bounds
    np.testing.assert_array_equal(im10.array, im.array)
    im10[2,3] = 17
    assert im10(2,3) == 17.
    assert im(2,3) == 17.

    im10b = galsim.Image(im.array, bounds=im.bounds, wcs=im.wcs, copy=True)
    assert im10b.wcs == im.wcs
    assert im10b.bounds == im.bounds
    np.testing.assert_array_equal(im10b.array, im.array)
    im10b[2,3] = 27
    assert im10b(2,3) == 27.
    assert im(2,3) != 27.

    # copyFrom copies the data only.
    im5.copyFrom(im8)
    assert im5.wcs != im.wcs  # im5 had a different wcs.  Should still have it.
    assert im5.bounds == im8.bounds
    np.testing.assert_array_equal(im5.array, im8.array)
    assert im5(3,8) == 11.
    im8[3,8] = 15
    assert im5(3,8) == 11.

    assert_raises(TypeError, im5.copyFrom, im8.array)
    im9 = galsim.Image(5,5,init_value=3)
    assert_raises(ValueError, im5.copyFrom, im9)


@timer
def test_complex_image():
    """Additional tests that are relevant for complex Image types
    """

    for dtype in [np.complex64, np.complex128]:
        # Some complex modifications to tests in test_Image_basic
        im1 = galsim.Image(ncol, nrow, dtype=dtype)
        im1_view = im1.view()
        im1_cview = im1.view(make_const=True)
        im2 = galsim.Image(ncol, nrow, init_value=23, dtype=dtype)
        im2_view = im2.view()
        im2_cview = im2.view(make_const=True)
        im2_conj = im2.conjugate

        # Check various ways to set and get values
        for y in range(1,nrow+1):
            for x in range(1,ncol+1):
                im1.setValue(x,y, 100 + 10*x + y + 13j*x + 23j*y)
                im2_view.setValue(x,y, 100 + 10*x + y + 13j*x + 23j*y)

        for y in range(1,nrow+1):
            for x in range(1,ncol+1):
                value = 100 + 10*x + y + 13j*x + 23j*y
                assert im1(x,y) == value
                assert im1.view()(x,y) == value
                assert im1.view(make_const=True)(x,y) == value
                assert im2(x,y) == value
                assert im2_view(x,y) == value
                assert im2_cview(x,y) == value
                assert im1.conjugate(x,y) == np.conjugate(value)

                # complex conjugate is not a view into the original.
                assert im2_conj(x,y) == 23
                assert im2.conjugate(x,y) == np.conjugate(value)

                value2 = 10*x + y + 20j*x + 2j*y
                im1.setValue(x,y, value2)
                im2_view.setValue(x=x, y=y, value=value2)
                assert im1(x,y) == value2
                assert im1.view()(x,y) == value2
                assert im1.view(make_const=True)(x,y) == value2
                assert im2(x,y) == value2
                assert im2_view(x,y) == value2
                assert im2_cview(x,y) == value2

                assert im1.real(x,y) == value2.real
                assert im1.view().real(x,y) == value2.real
                assert im1.view(make_const=True).real(x,y) == value2.real
                assert im2.real(x,y) == value2.real
                assert im2_view.real(x,y) == value2.real
                assert im2_cview.real(x,y) == value2.real
                assert im1.imag(x,y) == value2.imag
                assert im1.view().imag(x,y) == value2.imag
                assert im1.view(make_const=True).imag(x,y) == value2.imag
                assert im2.imag(x,y) == value2.imag
                assert im2_view.imag(x,y) == value2.imag
                assert im2_cview.imag(x,y) == value2.imag
                assert im1.conjugate(x,y) == np.conjugate(value2)
                assert im2.conjugate(x,y) == np.conjugate(value2)

                rvalue3 = 12*x + y
                ivalue3 = x + 21*y
                value3 = rvalue3 + 1j * ivalue3
                im1.real.setValue(x,y, rvalue3)
                im1.imag.setValue(x,y, ivalue3)
                im2_view.real.setValue(x,y, rvalue3)
                im2_view.imag.setValue(x,y, ivalue3)
                assert im1(x,y) == value3
                assert im1.view()(x,y) == value3
                assert im1.view(make_const=True)(x,y) == value3
                assert im2(x,y) == value3
                assert im2_view(x,y) == value3
                assert im2_cview(x,y) == value3
                assert im1.conjugate(x,y) == np.conjugate(value3)
                assert im2.conjugate(x,y) == np.conjugate(value3)

        # Check view of given data
        im3_view = galsim.Image((1+2j)*ref_array.astype(complex))
        slice_array = (large_array * (1+2j)).astype(complex)[::3,::2]
        im4_view = galsim.Image(slice_array)
        for y in range(1,nrow+1):
            for x in range(1,ncol+1):
                assert im3_view(x,y) == 10*x + y + 20j*x + 2j*y
                assert im4_view(x,y) == 10*x + y + 20j*x + 2j*y

        # Check picklability
        do_pickle(im1)
        do_pickle(im1_view)
        do_pickle(im1_cview)
        do_pickle(im2)
        do_pickle(im2_view)
        do_pickle(im3_view)
        do_pickle(im4_view)

@timer
def test_complex_image_arith():
    """Additional arithmetic tests that are relevant for complex Image types
    """
    image1 = galsim.ImageD(ref_array)

    # Binary ImageD op complex scalar
    image2 = image1 + (2+5j)
    np.testing.assert_array_equal(image2.array, ref_array + (2+5j),
            err_msg="ImageD + complex is not correct")
    image2 = image1 - (2+5j)
    np.testing.assert_array_equal(image2.array, ref_array - (2+5j),
            err_msg="ImageD - complex is not correct")
    image2 = image1 * (2+5j)
    np.testing.assert_array_equal(image2.array, ref_array * (2+5j),
            err_msg="ImageD * complex is not correct")
    image2 = image1 / (2+5j)
    np.testing.assert_array_equal(image2.array, ref_array / (2+5j),
            err_msg="ImageD / complex is not correct")

    # Binary complex scalar op ImageD
    image2 = (2+5j) + image1
    np.testing.assert_array_equal(image2.array, ref_array + (2+5j),
            err_msg="complex + ImageD is not correct")
    image2 = (2+5j) - image1
    np.testing.assert_array_equal(image2.array, -ref_array + (2+5j),
            err_msg="complex - ImageD is not correct")
    image2 = (2+5j) * image1
    np.testing.assert_array_equal(image2.array, ref_array * (2+5j),
            err_msg="complex * ImageD is not correct")
    image2 = (2+5j) / image1
    np.testing.assert_array_equal(image2.array, (2+5j) / ref_array.astype(float),
            err_msg="complex / ImageD is not correct")

    image2 = image1 * (3+1j)

    # Binary ImageCD op complex scalar
    image3 = image2 + (2+5j)
    np.testing.assert_array_equal(image3.array, (3+1j)*ref_array + (2+5j),
            err_msg="ImageCD + complex is not correct")
    image3 = image2 - (2+5j)
    np.testing.assert_array_equal(image3.array, (3+1j)*ref_array - (2+5j),
            err_msg="ImageCD - complex is not correct")
    image3 = image2 * (2+5j)
    np.testing.assert_array_equal(image3.array, (3+1j)*ref_array * (2+5j),
            err_msg="ImageCD * complex is not correct")
    image3 = image2 / (2+5j)
    np.testing.assert_array_equal(image3.array, (3+1j)*ref_array / (2+5j),
            err_msg="ImageCD / complex is not correct")

    # Binary complex scalar op ImageCD
    image3 = (2+5j) + image2
    np.testing.assert_array_equal(image3.array, (3+1j)*ref_array + (2+5j),
            err_msg="complex + ImageCD is not correct")
    image3 = (2+5j) - image2
    np.testing.assert_array_equal(image3.array, (-3-1j)*ref_array + (2+5j),
            err_msg="complex - ImageCD is not correct")
    image3 = (2+5j) * image2
    np.testing.assert_array_equal(image3.array, (3+1j)*ref_array * (2+5j),
            err_msg="complex * ImageCD is not correct")
    image3 = (2+5j) / image2
    np.testing.assert_array_equal(image3.array, (2+5j) / ((3+1j)*ref_array),
            err_msg="complex / ImageCD is not correct")

    # Binary ImageD op ImageCD
    image3 = image1 + image2
    np.testing.assert_array_equal(image3.array, (4+1j)*ref_array,
            err_msg="ImageD + ImageCD is not correct")
    image3 = image1 - image2
    np.testing.assert_array_equal(image3.array, (-2-1j)*ref_array,
            err_msg="ImageD - ImageCD is not correct")
    image3 = image1 * image2
    np.testing.assert_array_equal(image3.array, (3+1j)*ref_array**2,
            err_msg="ImageD * ImageCD is not correct")
    image3 = image1 / image2
    np.testing.assert_almost_equal(image3.array, 1./(3+1j), decimal=12,
            err_msg="ImageD / ImageCD is not correct")

    # Binary ImageCD op ImageD
    image3 = image2 + image1
    np.testing.assert_array_equal(image3.array, (4+1j)*ref_array,
            err_msg="ImageD + ImageCD is not correct")
    image3 = image2 - image1
    np.testing.assert_array_equal(image3.array, (2+1j)*ref_array,
            err_msg="ImageD - ImageCD is not correct")
    image3 = image2 * image1
    np.testing.assert_array_equal(image3.array, (3+1j)*ref_array**2,
            err_msg="ImageD * ImageCD is not correct")
    image3 = image2 / image1
    np.testing.assert_almost_equal(image3.array, (3+1j), decimal=12,
            err_msg="ImageD / ImageCD is not correct")

    # Binary ImageCD op ImageCD
    image3 = (4-3j) * image1
    image4 = image2 + image3
    np.testing.assert_array_equal(image4.array, (7-2j)*ref_array,
            err_msg="ImageCD + ImageCD is not correct")
    image4 = image2 - image3
    np.testing.assert_array_equal(image4.array, (-1+4j)*ref_array,
            err_msg="ImageCD - ImageCD is not correct")
    image4 = image2 * image3
    np.testing.assert_array_equal(image4.array, (15-5j)*ref_array**2,
            err_msg="ImageCD * ImageCD is not correct")
    image4 = image2 / image3
    np.testing.assert_almost_equal(image4.array, (9+13j)/25., decimal=12,
            err_msg="ImageCD / ImageCD is not correct")

    # In place ImageCD op complex scalar
    image4 = image2.copy()
    image4 += (2+5j)
    np.testing.assert_array_equal(image4.array, (3+1j)*ref_array + (2+5j),
            err_msg="ImageCD + complex is not correct")
    image4 = image2.copy()
    image4 -= (2+5j)
    np.testing.assert_array_equal(image4.array, (3+1j)*ref_array - (2+5j),
            err_msg="ImageCD - complex is not correct")
    image4 = image2.copy()
    image4 *= (2+5j)
    np.testing.assert_array_equal(image4.array, (3+1j)*ref_array * (2+5j),
            err_msg="ImageCD * complex is not correct")
    image4 = image2.copy()
    image4 /= (2+5j)
    np.testing.assert_array_equal(image4.array, (3+1j)*ref_array / (2+5j),
            err_msg="ImageCD / complex is not correct")

    # In place ImageCD op ImageD
    image4 = image2.copy()
    image4 += image1
    np.testing.assert_array_equal(image4.array, (4+1j)*ref_array,
            err_msg="ImageD + ImageCD is not correct")
    image4 = image2.copy()
    image4 -= image1
    np.testing.assert_array_equal(image4.array, (2+1j)*ref_array,
            err_msg="ImageD - ImageCD is not correct")
    image4 = image2.copy()
    image4 *= image1
    np.testing.assert_array_equal(image4.array, (3+1j)*ref_array**2,
            err_msg="ImageD * ImageCD is not correct")
    image4 = image2.copy()
    image4 /= image1
    np.testing.assert_almost_equal(image4.array, (3+1j), decimal=12,
            err_msg="ImageD / ImageCD is not correct")

    # In place ImageCD op ImageCD
    image4 = image2.copy()
    image4 += image3
    np.testing.assert_array_equal(image4.array, (7-2j)*ref_array,
            err_msg="ImageCD + ImageCD is not correct")
    image4 = image2.copy()
    image4 -= image3
    np.testing.assert_array_equal(image4.array, (-1+4j)*ref_array,
            err_msg="ImageCD - ImageCD is not correct")
    image4 = image2.copy()
    image4 *= image3
    np.testing.assert_array_equal(image4.array, (15-5j)*ref_array**2,
            err_msg="ImageCD * ImageCD is not correct")
    image4 = image2.copy()
    image4 /= image3
    np.testing.assert_almost_equal(image4.array, (9+13j)/25., decimal=12,
            err_msg="ImageCD / ImageCD is not correct")

@timer
def test_int_image_arith():
    """Additional arithmetic tests that are relevant for integer Image types
    """
    for i in range(int_ntypes):
        full = galsim.Image(ref_array.astype(types[i]))
        hi = (full // 8) * 8
        lo = (full % 8)

        #
        # Tests of __and__ and __iand__ operators:
        #

        # lo & hi = 0
        test = lo & hi
        np.testing.assert_array_equal(test.array, 0,
                err_msg="& failed for Images with dtype = %s."%types[i])

        # full & lo = lo
        test = full & lo
        np.testing.assert_array_equal(test.array, lo.array,
                err_msg="& failed for Images with dtype = %s."%types[i])

        # fullo & 0 = 0
        test = full & 0
        np.testing.assert_array_equal(test.array, 0,
                err_msg="& failed for Images with dtype = %s."%types[i])

        # lo & 24 = 0
        test = lo & 24
        np.testing.assert_array_equal(test.array, 0,
                err_msg="& failed for Images with dtype = %s."%types[i])

        # 7 & hi = 0
        test = 7 & hi
        np.testing.assert_array_equal(test.array, 0,
                err_msg="& failed for Images with dtype = %s."%types[i])

        # full & hi = hi
        test = full & hi
        np.testing.assert_array_equal(test.array, hi.array,
                err_msg="& failed for Images with dtype = %s."%types[i])

        # hi &= full => hi
        test &= full
        np.testing.assert_array_equal(test.array, hi.array,
                err_msg="&= failed for Images with dtype = %s."%types[i])

        # hi &= 8 => (hi & 8)
        test &= 8
        np.testing.assert_array_equal(test.array, (hi.array & 8),
                err_msg="&= failed for Images with dtype = %s."%types[i])

        # (hi & 8) &= hi => (hi & 8)
        test &= hi
        np.testing.assert_array_equal(test.array, (hi.array & 8),
                err_msg="&= failed for Images with dtype = %s."%types[i])


        #
        # Tests of __or__ and __ior__ operators:
        #

        # lo | hi = full
        test = lo | hi
        np.testing.assert_array_equal(test.array, full.array,
                err_msg="| failed for Images with dtype = %s."%types[i])

        # lo | lo = lo
        test = lo | lo
        np.testing.assert_array_equal(test.array, lo.array,
                err_msg="| failed for Images with dtype = %s."%types[i])

        # lo | 8 = lo + 8
        test = lo | 8
        np.testing.assert_array_equal(test.array, lo.array + 8,
                err_msg="| failed for Images with dtype = %s."%types[i])

        # 7 | hi = hi + 7
        test = 7 | hi
        np.testing.assert_array_equal(test.array, hi.array + 7,
                err_msg="| failed for Images with dtype = %s."%types[i])

        # hi | 0 = hi
        test = hi | 0
        np.testing.assert_array_equal(test.array, hi.array,
                err_msg="| failed for Images with dtype = %s."%types[i])

        # hi |= hi => hi
        test |= hi
        np.testing.assert_array_equal(test.array, hi.array,
                err_msg="|= failed for Images with dtype = %s."%types[i])

        # hi |= 3 => hi + 3
        test |= 3
        np.testing.assert_array_equal(test.array, hi.array + 3,
                err_msg="|= failed for Images with dtype = %s."%types[i])


        #
        # Tests of __xor__ and __ixor__ operators:
        #

        # lo ^ hi = full
        test = lo ^ hi
        np.testing.assert_array_equal(test.array, full.array,
                err_msg="^ failed for Images with dtype = %s."%types[i])

        # lo ^ full = hi
        test = lo ^ full
        np.testing.assert_array_equal(test.array, hi.array,
                err_msg="^ failed for Images with dtype = %s."%types[i])

        # lo ^ 40 = lo + 40
        test = lo ^ 40
        np.testing.assert_array_equal(test.array, lo.array + 40,
                err_msg="^ failed for Images with dtype = %s."%types[i])

        # 5 ^ hi = hi + 5
        test = 5 ^ hi
        np.testing.assert_array_equal(test.array, hi.array + 5,
                err_msg="^ failed for Images with dtype = %s."%types[i])

        # full ^ hi = lo
        test = full ^ hi
        np.testing.assert_array_equal(test.array, lo.array,
                err_msg="^ failed for Images with dtype = %s."%types[i])

        # lo ^= hi => full
        test ^= hi
        np.testing.assert_array_equal(test.array, full.array,
                err_msg="^= failed for Images with dtype = %s."%types[i])

        # full ^= 111 (x2) => full
        test ^= 111
        test ^= 111
        np.testing.assert_array_equal(test.array, full.array,
                err_msg="^= failed for Images with dtype = %s."%types[i])

        # full ^= lo => hi
        test ^= lo
        np.testing.assert_array_equal(test.array, hi.array,
                err_msg="^= failed for Images with dtype = %s."%types[i])


        #
        # Tests of __mod__ and __floordiv__ operators:
        #

        # lo // hi = 0
        test = lo // hi
        np.testing.assert_array_equal(test.array, 0,
                err_msg="// failed for Images with dtype = %s."%types[i])

        # lo // 8 = 0
        test = lo // 8
        np.testing.assert_array_equal(test.array, 0,
                err_msg="// failed for Images with dtype = %s."%types[i])

        # lo % 8 = lo
        test = lo % 8
        np.testing.assert_array_equal(test.array, lo.array,
                err_msg="%% failed for Images with dtype = %s."%types[i])

        # hi % 2 = hi & 1
        test = hi % 2
        np.testing.assert_array_equal(test.array, (hi & 1).array,
                err_msg="%% failed for Images with dtype = %s."%types[i])

        # lo % hi = lo
        test = lo % hi
        np.testing.assert_array_equal(test.array, lo.array,
                err_msg="%% failed for Images with dtype = %s."%types[i])

        # lo %= hi => lo
        test %= hi
        np.testing.assert_array_equal(test.array, lo.array,
                err_msg="%%= failed for Images with dtype = %s."%types[i])

        # lo %= 8 => lo
        test %= 8
        np.testing.assert_array_equal(test.array, lo.array,
                err_msg="%%= failed for Images with dtype = %s."%types[i])

        # lo //= hi => 0
        test //= hi
        np.testing.assert_array_equal(test.array, 0,
                err_msg="//= failed for Images with dtype = %s."%types[i])

        # 7 // hi = 0
        test = 7 // hi
        np.testing.assert_array_equal(test.array, 0,
                err_msg="// failed for Images with dtype = %s."%types[i])

        # 7 % hi = 7
        test = 7 % hi
        np.testing.assert_array_equal(test.array, 7,
                err_msg="%% failed for Images with dtype = %s."%types[i])

        # 7 //= 2 => 3
        test //= 2
        np.testing.assert_array_equal(test.array, 3,
                err_msg="%%= failed for Images with dtype = %s."%types[i])

        # 3 //= hi => 0
        test //= hi
        np.testing.assert_array_equal(test.array, 0,
                err_msg="//= failed for Images with dtype = %s."%types[i])

        # A subset of the above for cross-type checks.
        for j in range(i):
            full2 = galsim.Image(ref_array.astype(types[j]))
            hi2 = (full2 // 8) * 8
            lo2 = (full2 % 8)

            # full & hi = hi
            test = full & hi2
            np.testing.assert_array_equal(test.array, hi.array,
                    err_msg="& failed for Images with dtypes = %s, %s."%(types[i],types[j]))
            # hi &= full => hi
            test &= full2
            np.testing.assert_array_equal(test.array, hi.array,
                    err_msg="&= failed for Images with dtypes = %s, %s."%(types[i],types[j]))

            # lo | lo = lo
            test = lo | lo2
            np.testing.assert_array_equal(test.array, lo.array,
                    err_msg="| failed for Images with dtypes = %s, %s."%(types[i],types[j]))

            # lo |= hi => full
            test |= hi2
            np.testing.assert_array_equal(test.array, full.array,
                    err_msg="|= failed for Images with dtypes = %s, %s."%(types[i],types[j]))

            # lo ^ hi = full
            test = lo ^ hi2
            np.testing.assert_array_equal(test.array, full.array,
                    err_msg="^ failed for Images with dtypes = %s, %s."%(types[i],types[j]))

            # full ^= lo => hi
            test ^= lo2
            np.testing.assert_array_equal(test.array, hi.array,
                    err_msg="^= failed for Images with dtypes = %s, %s."%(types[i],types[j]))

            # lo // hi = 0
            test = lo // hi2
            np.testing.assert_array_equal(test.array, 0,
                    err_msg="// failed for Images with dtype = %s."%types[i])

            # lo % hi = lo
            test = lo % hi2
            np.testing.assert_array_equal(test.array, lo.array,
                    err_msg="%% failed for Images with dtype = %s."%types[i])

            # lo %= hi => lo
            test %= hi2
            np.testing.assert_array_equal(test.array, lo.array,
                    err_msg="%%= failed for Images with dtype = %s."%types[i])

            # lo //= hi => 0
            test //= hi2
            np.testing.assert_array_equal(test.array, 0,
                    err_msg="//= failed for Images with dtype = %s."%types[i])

    with assert_raises(ValueError):
        full & full.subImage(galsim.BoundsI(0,4,0,4))
    with assert_raises(ValueError):
        full | full.subImage(galsim.BoundsI(0,4,0,4))
    with assert_raises(ValueError):
        full ^ full.subImage(galsim.BoundsI(0,4,0,4))
    with assert_raises(ValueError):
        full // full.subImage(galsim.BoundsI(0,4,0,4))
    with assert_raises(ValueError):
        full % full.subImage(galsim.BoundsI(0,4,0,4))
    with assert_raises(ValueError):
        full &= full.subImage(galsim.BoundsI(0,4,0,4))
    with assert_raises(ValueError):
        full |= full.subImage(galsim.BoundsI(0,4,0,4))
    with assert_raises(ValueError):
        full ^= full.subImage(galsim.BoundsI(0,4,0,4))
    with assert_raises(ValueError):
        full //= full.subImage(galsim.BoundsI(0,4,0,4))
    with assert_raises(ValueError):
        full %= full.subImage(galsim.BoundsI(0,4,0,4))

    imd = galsim.ImageD(ref_array)
    with assert_raises(ValueError):
        imd & full
    with assert_raises(ValueError):
        imd | full
    with assert_raises(ValueError):
        imd ^ full
    with assert_raises(ValueError):
        imd // full
    with assert_raises(ValueError):
        imd % full
    with assert_raises(ValueError):
        imd &= full
    with assert_raises(ValueError):
        imd |= full
    with assert_raises(ValueError):
        imd ^= full
    with assert_raises(ValueError):
        imd //= full
    with assert_raises(ValueError):
        imd %= full

    with assert_raises(ValueError):
        full & imd
    with assert_raises(ValueError):
        full | imd
    with assert_raises(ValueError):
        full ^ imd
    with assert_raises(ValueError):
        full // imd
    with assert_raises(ValueError):
        full % imd
    with assert_raises(ValueError):
        full &= imd
    with assert_raises(ValueError):
        full |= imd
    with assert_raises(ValueError):
        full ^= imd
    with assert_raises(ValueError):
        full //= imd
    with assert_raises(ValueError):
        full %= imd


@timer
def test_wrap():
    """Test the image.wrap() function.
    """
    # Start with a fairly simple test where the image is 4 copies of the same data:
    im_orig = galsim.Image([[ 11., 12., 13., 14., 11., 12., 13., 14. ],
                            [ 21., 22., 23., 24., 21., 22., 23., 24. ],
                            [ 31., 32., 33., 34., 31., 32., 33., 34. ],
                            [ 41., 42., 43., 44., 41., 42., 43., 44. ],
                            [ 11., 12., 13., 14., 11., 12., 13., 14. ],
                            [ 21., 22., 23., 24., 21., 22., 23., 24. ],
                            [ 31., 32., 33., 34., 31., 32., 33., 34. ],
                            [ 41., 42., 43., 44., 41., 42., 43., 44. ]])
    im = im_orig.copy()
    b = galsim.BoundsI(1,4,1,4)
    im_quad = im_orig[b]
    im_wrap = im.wrap(b)
    np.testing.assert_almost_equal(im_wrap.array, 4.*im_quad.array, 12,
                                   "image.wrap() into first quadrant did not match expectation")

    # The same thing should work no matter where the lower left corner is:
    for xmin, ymin in ( (1,5), (5,1), (5,5), (2,3), (4,1) ):
        b = galsim.BoundsI(xmin, xmin+3, ymin, ymin+3)
        im_quad = im_orig[b]
        im = im_orig.copy()
        im_wrap = im.wrap(b)
        np.testing.assert_almost_equal(im_wrap.array, 4.*im_quad.array, 12,
                                       "image.wrap(%s) did not match expectation"%b)
        np.testing.assert_array_equal(im_wrap.array, im[b].array,
                                      "image.wrap(%s) did not return the right subimage")
        im[b].fill(0)
        np.testing.assert_array_equal(im_wrap.array, im[b].array,
                                      "image.wrap(%s) did not return a view of the original")

    # Now test where the subimage is not a simple fraction of the original, and all the
    # sizes are different.
    im = galsim.ImageD(17, 23, xmin=0, ymin=0)
    b = galsim.BoundsI(7,9,11,18)
    im_test = galsim.ImageD(b, init_value=0)
    for i in range(17):
        for j in range(23):
            val = np.exp(i/7.3) + (j/12.9)**3  # Something randomly complicated...
            im[i,j] = val
            # Find the location in the sub-image for this point.
            ii = (i-b.xmin) % (b.xmax-b.xmin+1) + b.xmin
            jj = (j-b.ymin) % (b.ymax-b.ymin+1) + b.ymin
            im_test.addValue(ii,jj,val)
    im_wrap = im.wrap(b)
    np.testing.assert_almost_equal(im_wrap.array, im_test.array, 12,
                                   "image.wrap(%s) did not match expectation"%b)
    np.testing.assert_array_equal(im_wrap.array, im[b].array,
                                  "image.wrap(%s) did not return the right subimage")
    np.testing.assert_equal(im_wrap.bounds, b,
                            "image.wrap(%s) does not have the correct bounds")

    # For complex images (in particular k-space images), we often want the image to be implicitly
    # Hermitian, so we only need to keep around half of it.
    M = 38
    N = 25
    K = 8
    L = 5
    im = galsim.ImageCD(2*M+1, 2*N+1, xmin=-M, ymin=-N)  # Explicitly Hermitian
    im2 = galsim.ImageCD(2*M+1, N+1, xmin=-M, ymin=0)   # Implicitly Hermitian across y axis
    im3 = galsim.ImageCD(M+1, 2*N+1, xmin=0, ymin=-N)   # Implicitly Hermitian across x axis
    #print('im = ',im)
    #print('im2 = ',im2)
    #print('im3 = ',im3)
    b = galsim.BoundsI(-K+1,K,-L+1,L)
    b2 = galsim.BoundsI(-K+1,K,0,L)
    b3 = galsim.BoundsI(0,K,-L+1,L)
    im_test = galsim.ImageCD(b, init_value=0)
    for i in range(-M,M+1):
        for j in range(-N,N+1):
            # An arbitrary, complicated Hermitian function.
            val = np.exp((i/(2.3*M))**2 + 1j*(2.8*i-1.3*j)) + ((2 + 3j*j)/(1.9*N))**3
            #val = 2*(i-j)**2 + 3j*(i+j)

            im[i,j] = val
            if j >= 0:
                im2[i,j] = val
            if i >= 0:
                im3[i,j] = val

            ii = (i-b.xmin) % (b.xmax-b.xmin+1) + b.xmin
            jj = (j-b.ymin) % (b.ymax-b.ymin+1) + b.ymin
            im_test.addValue(ii,jj,val)
    #print("im = ",im.array)

    # Confirm that the image is Hermitian.
    for i in range(-M,M+1):
        for j in range(-N,N+1):
            assert im(i,j) == im(-i,-j).conjugate()

    im_wrap = im.wrap(b)
    #print("im_wrap = ",im_wrap.array)
    np.testing.assert_almost_equal(im_wrap.array, im_test.array, 12,
                                   "image.wrap(%s) did not match expectation"%b)
    np.testing.assert_array_equal(im_wrap.array, im[b].array,
                                  "image.wrap(%s) did not return the right subimage")
    np.testing.assert_equal(im_wrap.bounds, b,
                            "image.wrap(%s) does not have the correct bounds")

    im2_wrap = im2.wrap(b2, hermitian='y')
    #print('im_test = ',im_test[b2].array)
    #print('im2_wrap = ',im2_wrap.array)
    #print('diff = ',im2_wrap.array-im_test[b2].array)
    np.testing.assert_almost_equal(im2_wrap.array, im_test[b2].array, 12,
                                   "image.wrap(%s) did not match expectation"%b)
    np.testing.assert_array_equal(im2_wrap.array, im2[b2].array,
                                  "image.wrap(%s) did not return the right subimage")
    np.testing.assert_equal(im2_wrap.bounds, b2,
                            "image.wrap(%s) does not have the correct bounds")

    im3_wrap = im3.wrap(b3, hermitian='x')
    #print('im_test = ',im_test[b3].array)
    #print('im3_wrap = ',im3_wrap.array)
    #print('diff = ',im3_wrap.array-im_test[b3].array)
    np.testing.assert_almost_equal(im3_wrap.array, im_test[b3].array, 12,
                                   "image.wrap(%s) did not match expectation"%b)
    np.testing.assert_array_equal(im3_wrap.array, im3[b3].array,
                                  "image.wrap(%s) did not return the right subimage")
    np.testing.assert_equal(im3_wrap.bounds, b3,
                            "image.wrap(%s) does not have the correct bounds")

    b = galsim.BoundsI(-K+1,K,-L+1,L)
    b2 = galsim.BoundsI(-K+1,K,0,L)
    b3 = galsim.BoundsI(0,K,-L+1,L)
    assert_raises(TypeError, im.wrap, bounds=None)
    assert_raises(ValueError, im3.wrap, b, hermitian='x')
    assert_raises(ValueError, im3.wrap, b2, hermitian='x')
    assert_raises(ValueError, im.wrap, b3, hermitian='x')
    assert_raises(ValueError, im2.wrap, b, hermitian='y')
    assert_raises(ValueError, im2.wrap, b3, hermitian='y')
    assert_raises(ValueError, im.wrap, b2, hermitian='y')
    assert_raises(ValueError, im.wrap, b, hermitian='invalid')
    assert_raises(ValueError, im2.wrap, b2, hermitian='invalid')
    assert_raises(ValueError, im3.wrap, b3, hermitian='invalid')


@timer
def test_FITS_bad_type():
    """Test that reading FITS files with an invalid data type succeeds by converting the
    type to float64.
    """
    # We check this by monkey patching the Image.valid_types list to not include int16
    # and see if it reads properly and raises the appropriate warning.
    orig_dtypes = galsim.Image.valid_dtypes
    setattr(galsim.Image,'valid_dtypes',(np.int32, np.float32, np.float64))

    testS_file = os.path.join(datadir, "testS.fits")
    testMultiS_file = os.path.join(datadir, "test_multiS.fits")
    testCubeS_file = os.path.join(datadir, "test_cubeS.fits")
    with assert_warns(galsim.GalSimWarning):
        testS_image = galsim.fits.read(testS_file)
    with assert_warns(galsim.GalSimWarning):
        testMultiS_image_list = galsim.fits.readMulti(testMultiS_file)
    with assert_warns(galsim.GalSimWarning):
        testCubeS_image_list = galsim.fits.readCube(testCubeS_file)

    np.testing.assert_equal(np.float64, testS_image.array.dtype.type)
    np.testing.assert_array_equal(ref_array.astype(float), testS_image.array,
            err_msg="ImageS read failed reading when int16 not a valid image type")
    for k in range(nimages):
        np.testing.assert_equal(np.float64, testMultiS_image_list[k].array.dtype.type)
        np.testing.assert_equal(np.float64, testCubeS_image_list[k].array.dtype.type)
        np.testing.assert_array_equal((ref_array+k).astype(float),
                testMultiS_image_list[k].array,
                err_msg="ImageS readMulti failed reading when int16 not a valid image type")
        np.testing.assert_array_equal((ref_array+k).astype(float),
                testCubeS_image_list[k].array,
                err_msg="ImageS readCube failed reading when int16 not a valid image type")

    # Don't forget to set it back to the original.
    setattr(galsim.Image,'valid_dtypes',orig_dtypes)

@timer
def test_bin():
    """Test the bin and subsample methods"""

    # Start with a relatively simple case of 2x2 binning with no partial bins to worry about.
    obj = galsim.Gaussian(sigma=2, flux=17).shear(g1=0.1, g2=0.3)
    im1 = obj.drawImage(nx=10, ny=14, scale=0.6, dtype=float)

    im2 = obj.drawImage(nx=20, ny=28, scale=0.3, dtype=float)
    im3 = im2.bin(2,2)
    ar2 = im2.array
    ar3b = ar2[0::2,0::2] + ar2[0::2,1::2] + ar2[1::2,0::2] + ar2[1::2,1::2]

    np.testing.assert_almost_equal(ar3b.sum(), im2.array.sum(), 6,
                                   "direct binning didn't perserve total flux")
    np.testing.assert_almost_equal(ar3b, im3.array, 6,
                                   "direct binning didn't match bin function.")
    np.testing.assert_almost_equal(im3.array.sum(), im2.array.sum(), 6,
                                   "bin didn't preserve the total flux")
    np.testing.assert_almost_equal(im3.array, im1.array, 6,
                                   "2x2 binned image doesn't match image with 2x2 larger pixels")
    np.testing.assert_almost_equal(im3.scale, im1.scale, 6, "bin resulted in wrong scale")

    im4 = im2.subsample(2,2)
    np.testing.assert_almost_equal(im4.array.sum(), im2.array.sum(), 6,
                                   "subsample didn't preserve the total flux")
    np.testing.assert_almost_equal(im4.scale, im2.scale/2., 6, "subsample resulted in wrong scale")
    im5 = im4.bin(2,2)
    np.testing.assert_almost_equal(im5.array, im2.array, 6,
                                   "Round trip subsample then bin 2x2 doesn't match original")
    np.testing.assert_almost_equal(im5.scale, im2.scale, 6, "round trip resulted in wrong scale")

    # Next do nx != ny.  And wcs = JacobianWCS
    wcs1 = galsim.JacobianWCS(0.6, 0.14, 0.15, 0.7)
    im1 = obj.drawImage(nx=11, ny=15, wcs=wcs1, dtype=float)
    im1.wcs = im1.wcs.withOrigin(im1.true_center, galsim.PositionD(200,300))
    center1 = im1.wcs.toWorld(im1.true_center)
    print('center1 = ',center1)

    wcs2 = galsim.JacobianWCS(0.2, 0.07, 0.05, 0.35)
    im2 = obj.drawImage(nx=33, ny=30, wcs=wcs2, dtype=float)
    im2.wcs = im2.wcs.withOrigin(im2.true_center, galsim.PositionD(200,300))
    center2 = im2.wcs.toWorld(im2.true_center)
    print('center2 = ',center2)
    im3 = im2.bin(3,2)
    center3 = im2.wcs.toWorld(im2.true_center)
    print('center3 = ',center3)
    ar2 = im2.array
    ar3b = (ar2[0::2,0::3] + ar2[0::2,1::3] + ar2[0::2,2::3] +
            ar2[1::2,0::3] + ar2[1::2,1::3] + ar2[1::2,2::3])

    np.testing.assert_almost_equal(ar3b.sum(), im2.array.sum(), 6,
                                   "direct binning didn't perserve total flux")
    np.testing.assert_almost_equal(ar3b, im3.array, 6,
                                   "direct binning didn't match bin function.")
    np.testing.assert_almost_equal(im3.array.sum(), im2.array.sum(), 6,
                                   "bin didn't preserve the total flux")
    np.testing.assert_almost_equal(im3.array, im1.array, 6,
                                   "3x2 binned image doesn't match image with 3x2 larger pixels")
    np.testing.assert_almost_equal((center3.x, center3.y), (center1.x, center1.y), 6,
                                   "3x2 binned image wcs is wrong")

    im4 = im2.subsample(3,2)
    np.testing.assert_almost_equal(im4.array.sum(), im2.array.sum(), 6,
                                   "subsample didn't preserve the total flux")
    center4 = im4.wcs.toWorld(im4.true_center)
    print('center4 = ',center4)
    np.testing.assert_almost_equal((center4.x, center4.y), (center1.x, center1.y), 6,
                                   "3x2 subsampled image wcs is wrong")

    im5 = im4.bin(3,2)
    np.testing.assert_almost_equal(im5.array, im2.array, 6,
                                   "Round trip subsample then bin 3x2 doesn't match original")
    center5 = im5.wcs.toWorld(im5.true_center)
    print('center5 = ',center5)
    np.testing.assert_almost_equal((center5.x, center5.y), (center1.x, center1.y), 6,
                                   "Round trip 3x2 image wcs is wrong")

    # If the initial wcs is None or not uniform, then the resulting wcs will be None.
    im2.wcs = galsim.UVFunction('0.6 + np.sin(x*y)', '0.6 + np.cos(x+y)')
    im3b = im2.bin(3,2)
    assert im3b.wcs is None
    np.testing.assert_array_equal(im3b.array, im3.array,
                                  "The wcs shouldn't affect what bin does to the array.")
    im4b = im2.subsample(3,2)
    assert im4b.wcs is None
    np.testing.assert_array_equal(im4b.array, im4.array,
                                  "The wcs shouldn't affect what subsample does to the array.")

    im2.wcs = None
    im3c = im2.bin(3,2)
    assert im3c.wcs is None
    np.testing.assert_array_equal(im3c.array, im3.array,
                                  "The wcs shouldn't affect what bin does to the array.")
    im4c = im2.subsample(3,2)
    assert im4c.wcs is None
    np.testing.assert_array_equal(im4c.array, im4.array,
                                  "The wcs shouldn't affect what subsample does to the array.")

    # Finally, binning should still work, even if the number of pixels doesn't go evenly into
    # the number of pixels/block specified.
    im6 = im1.bin(8,8)
    ar6 = np.array([[ im1.array[0:8,0:8].sum(), im1.array[0:8,8:].sum() ],
                    [ im1.array[8:,0:8].sum(),  im1.array[8:,8:].sum()  ]])
    np.testing.assert_almost_equal(im6.array, ar6, 6,
                                   "Binning past the edge of the image didn't work properly")
    # The center of this image doesn't correspond to the center of the original.
    # But the lower left edge does.
    origin1 = im1.wcs.toWorld(galsim.PositionD(im1.xmin-0.5, im1.ymin-0.5))
    origin6 = im6.wcs.toWorld(galsim.PositionD(im1.xmin-0.5, im6.ymin-0.5))
    print('origin1 = ',origin1)
    print('origin6 = ',origin6)
    np.testing.assert_almost_equal((origin6.x, origin6.y), (origin1.x, origin1.y), 6,
                                   "Binning past the edge resulted in wrong wcs")

if __name__ == "__main__":
    test_Image_basic()
    test_undefined_image()
    test_Image_FITS_IO()
    test_Image_MultiFITS_IO()
    test_Image_CubeFITS_IO()
    test_Image_array_view()
    test_Image_binary_add()
    test_Image_binary_subtract()
    test_Image_binary_multiply()
    test_Image_binary_divide()
    test_Image_binary_scalar_add()
    test_Image_binary_scalar_subtract()
    test_Image_binary_scalar_multiply()
    test_Image_binary_scalar_divide()
    test_Image_binary_scalar_pow()
    test_Image_inplace_add()
    test_Image_inplace_subtract()
    test_Image_inplace_multiply()
    test_Image_inplace_divide()
    test_Image_inplace_scalar_add()
    test_Image_inplace_scalar_subtract()
    test_Image_inplace_scalar_multiply()
    test_Image_inplace_scalar_divide()
    test_Image_inplace_scalar_pow()
    test_Image_subImage()
    test_subImage_persistence()
    test_Image_resize()
    test_ConstImage_array_constness()
    test_BoundsI_init_with_non_pure_ints()
    test_Image_constructor()
    test_Image_view()
    test_Image_writeheader()
    test_ne()
    test_copy()
    test_complex_image()
    test_complex_image_arith()
    test_int_image_arith()
    test_wrap()
    test_FITS_bad_type()
    test_bin()
