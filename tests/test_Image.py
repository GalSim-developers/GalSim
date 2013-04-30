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
import os
import sys
import pyfits
import numpy as np

"""Unit tests for the Image and ImageView classes.

These tests use four externally generated (IDL + astrolib FITS writing tools) reference images for
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

Images are in S, I, F & D flavours.

There are also four FITS cubes, and four FITS multi-extension files for testing.  Each is 12
images deep, with the first image being the reference above and each subsequent being the same
incremented by one.

"""

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# Setup info for tests, not likely to change
ntypes = 4
types = [np.int16, np.int32, np.float32, np.float64]
tchar = ['S', 'I', 'F', 'D']

ncol = 7
nrow = 5
test_shape = (ncol, nrow)  # shape of image arrays for all tests
ref_array = np.array([
    [11, 21, 31, 41, 51, 61, 71], 
    [12, 22, 32, 42, 52, 62, 72], 
    [13, 23, 33, 43, 53, 63, 73], 
    [14, 24, 34, 44, 54, 64, 74], 
    [15, 25, 35, 45, 55, 65, 75] ]).astype(np.int16)

nimages = 12  # Depth of FITS datacubes and multi-extension FITS files

datadir = os.path.join(".", "Image_comparison_images")

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_Image_basic():
    """Test that all supported types perform basic Image operations correctly
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):

        # Check basic constructor from ncol, nrow
        array_type = types[i]
        im1 = galsim.Image[array_type](ncol,nrow)
        bounds = galsim.BoundsI(1,ncol,1,nrow)

        assert im1.getXMin() == 1
        assert im1.getXMax() == ncol
        assert im1.getYMin() == 1
        assert im1.getYMax() == nrow
        assert im1.getBounds() == bounds
        assert im1.bounds == bounds

        # Check basic constructor from ncol, nrow
        # Also test alternate name of image type: ImageD, ImageF, etc.
        image_type = eval("galsim.Image"+tchar[i]) # Use handy eval() mimics use of ImageSIFD
        im2 = image_type(bounds)
        im2_view = im2.view()

        assert im2_view.getXMin() == 1
        assert im2_view.getXMax() == ncol
        assert im2_view.getYMin() == 1
        assert im2_view.getYMax() == nrow
        assert im2_view.bounds == bounds

        # Check various ways to set and get values
        for y in range(1,nrow):
            for x in range(1,ncol):
                im1.setValue(x,y, 100 + 10*x + y)
                im2_view.setValue(x,y, 100 + 10*x + y)

        for y in range(1,nrow):
            for x in range(1,ncol):
                assert im1.at(x,y) == 100+10*x+y
                assert im1.view().at(x,y) == 100+10*x+y
                assert im2.at(x,y) == 100+10*x+y
                assert im2_view.at(x,y) == 100+10*x+y
                im1.setValue(x,y, 10*x + y)
                im2_view.setValue(x,y, 10*x + y)
                assert im1(x,y) == 10*x+y
                assert im1.view()(x,y) == 10*x+y
                assert im2(x,y) == 10*x+y
                assert im2_view(x,y) == 10*x+y

        # Setting or getting the value outside the bounds should throw an exception.
        try:
            np.testing.assert_raises(RuntimeError,im1.setValue,0,0,1)
            np.testing.assert_raises(RuntimeError,im1.at,0,0)
            np.testing.assert_raises(RuntimeError,im1.view().setValue,0,0,1)
            np.testing.assert_raises(RuntimeError,im1.view().at,0,0)

            np.testing.assert_raises(RuntimeError,im1.setValue,ncol+1,0,1)
            np.testing.assert_raises(RuntimeError,im1.at,ncol+1,0)
            np.testing.assert_raises(RuntimeError,im1.view().setValue,ncol+1,0,1)
            np.testing.assert_raises(RuntimeError,im1.view().at,ncol+1,0)

            np.testing.assert_raises(RuntimeError,im1.setValue,0,nrow+1,1)
            np.testing.assert_raises(RuntimeError,im1.at,0,nrow+1)
            np.testing.assert_raises(RuntimeError,im1.view().setValue,0,nrow+1,1)
            np.testing.assert_raises(RuntimeError,im1.view().at,0,nrow+1)

            np.testing.assert_raises(RuntimeError,im1.setValue,ncol+1,nrow+1,1)
            np.testing.assert_raises(RuntimeError,im1.at,ncol+1,nrow+1)
            np.testing.assert_raises(RuntimeError,im1.view().setValue,ncol+1,nrow+1,1)
            np.testing.assert_raises(RuntimeError,im1.view().at,ncol+1,nrow+1)
        except ImportError:
            print 'The assert_raises tests require nose'

        # Check view of given data
        im3_view = galsim.ImageView[array_type](ref_array.astype(array_type))
        for y in range(1,nrow):
            for x in range(1,ncol):
                assert im3_view(x,y) == 10*x+y

        # Check shift ops
        im1_view = im1.view() # View with old bounds
        dx = 31
        dy = 16
        im1.shift(dx,dy)
        im2_view.setOrigin( 1+dx , 1+dy )
        im3_view.setCenter( (ncol+1)/2+dx , (nrow+1)/2+dy )
        shifted_bounds = galsim.BoundsI(1+dx, ncol+dx, 1+dy, nrow+dy)

        assert im1.bounds == shifted_bounds
        assert im2_view.bounds == shifted_bounds
        assert im3_view.bounds == shifted_bounds
        # Others shouldn't have changed
        assert im1_view.bounds == bounds
        assert im2.bounds == bounds
        for y in range(1,nrow):
            for x in range(1,ncol):
                assert im1(x+dx,y+dy) == 10*x+y
                assert im1_view(x,y) == 10*x+y
                assert im2(x,y) == 10*x+y
                assert im2_view(x+dx,y+dy) == 10*x+y
                assert im3_view(x+dx,y+dy) == 10*x+y
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_FITS_IO():
    """Test that all four FITS reference images are correctly read in by both PyFITS and our Image 
    wrappers.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        array_type = types[i]

        #
        # Test input from a single external FITS image
        #
        
        # Read the reference image to from an externally-generated fits file
        test_file = os.path.join(datadir, "test"+tchar[i]+".fits")
        # Check pyfits read for sanity
        test_array = pyfits.getdata(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_array,
                err_msg="PyFITS failing to read reference image.")

        # Then use galsim fits.read function
        # First version: use pyfits HDUList
        hdu = pyfits.open(test_file)
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
        ref_image = galsim.ImageView[array_type](ref_array.astype(array_type))
        test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits")
        ref_image.write(test_file)

        # Check pyfits read for sanity
        test_array = pyfits.getdata(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_array,
                err_msg="Image"+tchar[i]+" write failed.")

        # Then use galsim fits.read function
        # First version: use pyfits HDUList
        hdu = pyfits.open(test_file)
        test_image = galsim.fits.read(hdu_list=hdu)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array, 
                err_msg="Failed reading from PyFITS PrimaryHDU input.")

        # Second version: use file name
        test_image = galsim.fits.read(test_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array, 
                err_msg="Image"+tchar[i]+" read failed reading from filename input.")

        #
        # Test various compression schemes
        #

        # These tests are a bit slow, so we only bother to run them for the first dtype.
        # When working on the code, it is a good idea to comment out this line.
        if i > 0: continue

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

        # Test gzip_tile
        test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits.gzt")
        ref_image.write(test_file, compression='gzip_tile')
        test_image = galsim.fits.read(test_file, compression='gzip_tile')
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array, 
                err_msg="Image"+tchar[i]+" write failed for gzip_tile")

        # Test hcompress
        test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits.hc")
        ref_image.write(test_file, compression='hcompress')
        test_image = galsim.fits.read(test_file, compression='hcompress')
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array, 
                err_msg="Image"+tchar[i]+" write failed for hcompress")

        # Test plio (only valid on positive integer values)
        if tchar[i] in ['S', 'I']:
            test_file = os.path.join(datadir, "test"+tchar[i]+"_internal.fits.plio")
            ref_image.write(test_file, compression='plio')
            test_image = galsim.fits.read(test_file, compression='plio')
            np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array, 
                    err_msg="Image"+tchar[i]+" write failed for plio")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_Image_MultiFITS_IO():
    """Test that all four FITS reference images are correctly read in by both PyFITS and our Image 
    wrappers.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        array_type = types[i]

        # 
        # Test input from an external multi-extension fits file 
        #
        
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+".fits")
        # Check pyfits read for sanity
        test_array = pyfits.getdata(test_multi_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_array,
                err_msg="PyFITS failing to read multi file.")

        # Then use galsim fits.readMulti function
        # First version: use pyfits HDUList
        hdu = pyfits.open(test_multi_file)
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
        ref_image = galsim.ImageView[array_type](ref_array.astype(array_type))
        image_list = []
        for k in range(nimages):
            image_list.append(ref_image + k)

        # Write the list to a multi-extension fits file
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits")
        galsim.fits.writeMulti(image_list,test_multi_file)

        # Check pyfits read for sanity
        test_array = pyfits.getdata(test_multi_file)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_array,
                err_msg="PyFITS failing to read multi file.")

        # Then use galsim fits.readMulti function
        # First version: use pyfits HDUList
        hdu = pyfits.open(test_multi_file)
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
        # Test various compression schemes
        #

        # These tests are a bit slow, so we only bother to run them for the first dtype.
        # When working on the code, it is a good idea to comment out this line.
        if i > 0: continue

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

        # Test gzip_tile
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits.gzt")
        galsim.fits.writeMulti(image_list,test_multi_file, compression='gzip_tile')
        test_image_list = galsim.fits.readMulti(test_multi_file, compression='gzip_tile')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]), 
                    test_image_list[k].array, 
                    err_msg="Image"+tchar[i]+" writeMulti failed for gzip_tile")

        # Test hcompress
        test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits.hc")
        galsim.fits.writeMulti(image_list,test_multi_file, compression='hcompress')
        test_image_list = galsim.fits.readMulti(test_multi_file, compression='hcompress')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]), 
                    test_image_list[k].array, 
                    err_msg="Image"+tchar[i]+" writeMulti failed for hcompress")

        # Test plio (only valid on positive integer values)
        if tchar[i] in ['S', 'I']:
            test_multi_file = os.path.join(datadir, "test_multi"+tchar[i]+"_internal.fits.plio")
            galsim.fits.writeMulti(image_list,test_multi_file, compression='plio')
            test_image_list = galsim.fits.readMulti(test_multi_file, compression='plio')
            for k in range(nimages):
                np.testing.assert_array_equal((ref_array+k).astype(types[i]), 
                        test_image_list[k].array, 
                        err_msg="Image"+tchar[i]+" writeMulti failed for plio")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_Image_CubeFITS_IO():
    """Test that all four FITS reference images are correctly read in by both PyFITS and our Image 
    wrappers.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        array_type = types[i]

        # 
        # Test input from an external fits data cube
        #
        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+".fits")
        # Check pyfits read for sanity
        test_array = pyfits.getdata(test_cube_file)
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]), test_array[k,:,:],
                    err_msg="PyFITS failing to read cube file.")

        # Then use galsim fits.readCube function
        # First version: use pyfits HDUList
        hdu = pyfits.open(test_cube_file)
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
        ref_image = galsim.ImageView[array_type](ref_array.astype(array_type))
        image_list = []
        for k in range(nimages):
            image_list.append(ref_image + k)

        # Write the list to a fits data cube
        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits")
        galsim.fits.writeCube(image_list,test_cube_file)

        # Check pyfits read for sanity
        test_array = pyfits.getdata(test_cube_file)
        assert test_array.dtype.type == types[i], "%s != %s" % (test_array.dtype.type, types[i])
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]), test_array[k,:,:],
                    err_msg="PyFITS failing to read cube file.")

        # Then use galsim fits.readCube function
        # First version: use pyfits HDUList
        hdu = pyfits.open(test_cube_file)
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
        # Test various compression schemes
        #

        # These tests are a bit slow, so we only bother to run them for the first dtype.
        # When working on the code, it is a good idea to comment out this line.
        if i > 0: continue

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

        # Test gzip_tile
        test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits.gzt")
        galsim.fits.writeCube(image_list,test_cube_file, compression='gzip_tile')
        test_image_list = galsim.fits.readCube(test_cube_file, compression='gzip_tile')
        for k in range(nimages):
            np.testing.assert_array_equal((ref_array+k).astype(types[i]), 
                    test_image_list[k].array, 
                    err_msg="Image"+tchar[i]+" writeCube failed for gzip_tile")

        # Note: hcompress is invalid for data cubes

        # Test plio (only valid on positive integer values)
        if tchar[i] in ['S', 'I']:
            test_cube_file = os.path.join(datadir, "test_cube"+tchar[i]+"_internal.fits.plio")
            galsim.fits.writeCube(image_list,test_cube_file, compression='plio')
            test_image_list = galsim.fits.readCube(test_cube_file, compression='plio')
            for k in range(nimages):
                np.testing.assert_array_equal((ref_array+k).astype(types[i]), 
                        test_image_list[k].array, 
                        err_msg="Image"+tchar[i]+" writeCube failed for plio")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_Image_array_view():
    """Test that all four types of supported Images correctly provide a view on an input array.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        np.testing.assert_array_equal(ref_array.astype(types[i]), image.array,
                err_msg="Array look into Image class (dictionary call) does"
                +" not match input for dtype = "+str(types[i]))
        #Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image = image_init_func(ref_array.astype(types[i]))
        np.testing.assert_array_equal(ref_array.astype(types[i]), image.array,
                err_msg="Array look into ImageView class does not match input"
                +" for dtype = "+str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_binary_add():
    """Test that all four types of supported Images add correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image3 = image1 + image2
        np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image3.array,
                err_msg="Binary add in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image3 = image1 + image2
        np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image3.array,
                err_msg="Binary add in Image class does not match reference for dtype = "
                +str(types[i]))

        for j in xrange(ntypes):
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func(ref_array.astype(types[i]))
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image3 = image1 + image2
            type3 = image3.array.dtype.type
            np.testing.assert_array_equal((3 * ref_array).astype(type3), image3.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_binary_subtract():
    """Test that all four types of supported Images subtract correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image3 = image2 - image1
        np.testing.assert_array_equal(ref_array.astype(types[i]), image3.array,
                err_msg="Binary subtract in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image3 = image2 - image1
        np.testing.assert_array_equal(ref_array.astype(types[i]), image3.array,
                err_msg="Binary subtract in Image class does not match reference for dtype = "
                +str(types[i]))
        for j in xrange(ntypes):
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func(ref_array.astype(types[i]))
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image3 = image2 - image1
            type3 = image3.array.dtype.type
            np.testing.assert_array_equal(ref_array.astype(type3), image3.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_binary_multiply():
    """Test that all four types of supported Images multiply correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image3 = image1 * image2
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image3.array,
                err_msg="Binary multiply in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image3 = image1 * image2
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image3.array,
                err_msg="Binary multiply in Image class does not match reference for dtype = "
                +str(types[i]))
        for j in xrange(ntypes):
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func(ref_array.astype(types[i]))
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image3 = image2 * image1
            type3 = image3.array.dtype.type
            np.testing.assert_array_equal((2*ref_array**2).astype(type3), image3.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_binary_divide():
    """Test that all four types of supported Images divide correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        # Note that I am using refarray + 1 to avoid divide-by-zero. 
        image1 = galsim.ImageView[types[i]]((ref_array + 1).astype(types[i]))
        image2 = galsim.ImageView[types[i]]((3 * (ref_array + 1)**2).astype(types[i]))
        image3 = image2 / image1
        np.testing.assert_array_equal((3 * (ref_array + 1)).astype(types[i]), image3.array,
                err_msg="Binary divide in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func((ref_array + 1).astype(types[i]))
        image2 = image_init_func((3 * (ref_array + 1)**2).astype(types[i]))
        image3 = image2 / image1
        np.testing.assert_array_equal((3 * (ref_array + 1)).astype(types[i]), image3.array,
                err_msg="Binary divide in Image class does not match reference for dtype = "
                +str(types[i]))
        for j in xrange(ntypes):
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func((ref_array + 1).astype(types[i]))
            image2 = image2_init_func((3 * (ref_array+1)**2).astype(types[j]))
            image3 = image2 / image1
            type3 = image3.array.dtype.type
            np.testing.assert_array_equal((3*(ref_array+1)).astype(type3), image3.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_binary_scalar_add():
    """Test that all four types of supported Images add scalars correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = image1 + 3
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        image2 = 3 + image1
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary radd scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image1 + 3
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class does not match reference for dtype = "
                +str(types[i]))
        image2 = 3 + image1
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary radd scalar in Image class does not match reference for dtype = "
                +str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_binary_scalar_subtract():
    """Test that all four types of supported Images binary scalar subtract correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = image1 - 3
        np.testing.assert_array_equal((ref_array - 3).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image1 - 3
        np.testing.assert_array_equal((ref_array - 3).astype(types[i]), image2.array,
                err_msg="Binary add scalar in Image class does not match reference for dtype = "
                +str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_binary_scalar_multiply():
    """Test that all four types of supported Images binary scalar multiply correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = image1 * 3
        np.testing.assert_array_equal((ref_array * 3).astype(types[i]), image2.array,
                err_msg="Binary multiply scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        image2 = 3 * image1
        np.testing.assert_array_equal((ref_array * 3).astype(types[i]), image2.array,
                err_msg="Binary rmultiply scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image1 * 3
        np.testing.assert_array_equal((ref_array * 3).astype(types[i]), image2.array,
                err_msg="Binary multiply scalar in Image class does"
                +" not match reference for dtype = "+str(types[i]))
        image2 = 3 * image1
        np.testing.assert_array_equal((ref_array * 3).astype(types[i]), image2.array,
                err_msg="Binary rmultiply scalar in Image class does"
                +" not match reference for dtype = "+str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_binary_scalar_divide():
    """Test that all four types of supported Images binary scalar divide correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]]((3 * ref_array).astype(types[i]))
        image2 = image1 / 3
        np.testing.assert_array_equal(ref_array.astype(types[i]), image2.array,
                err_msg="Binary divide scalar in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func((3 * ref_array).astype(types[i]))
        image2 = image1 / 3
        np.testing.assert_array_equal(ref_array.astype(types[i]), image2.array,
                err_msg="Binary divide scalar in Image class does"
                +" not match reference for dtype = "+str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
        

def test_Image_inplace_add():
    """Test that all four types of supported Images inplace add correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image1 += image2
        np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image1.array,
                err_msg="Inplace add in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image1 += image2
        np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image1.array,
                err_msg="Inplace add in Image class does not match reference for dtype = "
                +str(types[i]))
        for j in xrange(i): # Only add simpler types to this one.
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func(ref_array.astype(types[i]))
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image1 += image2
            np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image1.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_inplace_subtract():
    """Test that all four types of supported Images inplace subtract correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image2 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image1 -= image2
        np.testing.assert_array_equal(ref_array.astype(types[i]), image1.array,
                err_msg="Inplace subtract in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func((2 * ref_array).astype(types[i]))
        image2 = image_init_func(ref_array.astype(types[i]))
        image1 -= image2
        np.testing.assert_array_equal(ref_array.astype(types[i]), image1.array,
                err_msg="Inplace subtract in Image class does"
                +" not match reference for dtype = "+str(types[i]))
        for j in xrange(i): # Only subtract simpler types from this one.
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func((2 * ref_array).astype(types[i]))
            image2 = image2_init_func(ref_array.astype(types[j]))
            image1 -= image2
            np.testing.assert_array_equal(ref_array.astype(types[i]), image1.array,
                    err_msg="Inplace subtract in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_inplace_multiply():
    """Test that all four types of supported Images inplace multiply correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image1 *= image2
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image1.array,
                err_msg="Inplace multiply in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image1 *= image2
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image1.array,
                err_msg="Inplace multiply in Image class does not match reference for dtype = "
                +str(types[i]))
        for j in xrange(i): # Only multiply simpler types to this one.
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func(ref_array.astype(types[i]))
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image1 *= image2
            np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image1.array,
                    err_msg="Inplace multiply in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_inplace_divide():
    """Test that all four types of supported Images inplace divide correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]]((2 * (ref_array + 1)**2).astype(types[i]))
        image2 = galsim.ImageView[types[i]]((ref_array + 1).astype(types[i]))
        image1 /= image2
        np.testing.assert_array_equal((2 * (ref_array + 1)).astype(types[i]), image1.array,
                err_msg="Inplace divide in Image class (dictionary call) does"
                +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func((2 * (ref_array + 1)**2).astype(types[i]))
        image2 = image_init_func((ref_array + 1).astype(types[i]))
        image1 /= image2
        np.testing.assert_array_equal((2 * (ref_array + 1)).astype(types[i]), image1.array,
                err_msg="Inplace divide in Image class does not match reference for dtype = "
                +str(types[i]))
        for j in xrange(i): # Only divide simpler types into this one.
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func((2 * (ref_array+1)**2).astype(types[i]))
            image2 = image2_init_func((ref_array+1).astype(types[j]))
            image1 /= image2
            np.testing.assert_array_equal((2 * (ref_array+1)).astype(types[i]), image1.array,
                    err_msg="Inplace divide in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_Image_inplace_scalar_add():
    """Test that all four types of supported Images inplace scalar add correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image1 += 1
        np.testing.assert_array_equal((ref_array + 1).astype(types[i]), image1.array,
                err_msg="Inplace scalar add in Image class (dictionary "
                +"call) does not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image1 += 1
        np.testing.assert_array_equal((ref_array + 1).astype(types[i]), image1.array,
                err_msg="Inplace scalar add in Image class does not match reference for dtype = "
                +str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_inplace_scalar_subtract():
    """Test that all four types of supported Images inplace scalar subtract correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image1 -= 1
        np.testing.assert_array_equal((ref_array - 1).astype(types[i]), image1.array,
                err_msg="Inplace scalar subtract in Image class (dictionary "
                +"call) does not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image1 -= 1
        np.testing.assert_array_equal((ref_array - 1).astype(types[i]), image1.array,
                err_msg="Inplace scalar subtract in Image class does"
                +" not match reference for dtype = "+str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_inplace_scalar_multiply():
    """Test that all four types of supported Images inplace scalar multiply correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image1 *= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                err_msg="Inplace scalar multiply in Image class (dictionary "
                +"call) does not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image1 *= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                err_msg="Inplace scalar multiply in Image class does"
                +" not match reference for dtype = "+str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Image_inplace_scalar_divide():
    """Test that all four types of supported Images inplace scalar divide correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image2 /= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                err_msg="Inplace scalar divide in Image class (dictionary "
                +"call) does not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image2 /= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                err_msg="Inplace scalar divide in Image class does"
                +" not match reference for dtype = "+str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
        
def test_Image_subImage():
    """Test that subImages are accessed and written correctly.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        image = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        bounds = galsim.BoundsI(3,4,2,3)
        sub_array = np.array([[32, 42], [33, 43]]).astype(types[i])
        np.testing.assert_array_equal(image.subImage(bounds).array, sub_array,
            err_msg="image.subImage(bounds) does not match reference for dtype = "+str(types[i]))
        np.testing.assert_array_equal(image[bounds].array, sub_array,
            err_msg="image[bounds] does not match reference for dtype = "+str(types[i]))
        image[bounds] = galsim.ImageView[types[i]](sub_array+100)
        np.testing.assert_array_equal(image[bounds].array, (sub_array+100),
            err_msg="image[bounds] = im2 does not set correctly for dtype = "+str(types[i]))
        for xpos in range(1,test_shape[0]):
            for ypos in range(1,test_shape[1]):
                if (xpos >= bounds.getXMin() and xpos <= bounds.getXMax() and 
                    ypos >= bounds.getYMin() and ypos <= bounds.getYMax()):
                    value = ref_array[ypos-1,xpos-1] + 100
                else:
                    value = ref_array[ypos-1,xpos-1]
                assert image(xpos,ypos) == value,  \
                    "image[bounds] = im2 set wrong locations for dtype = "+str(types[i])

        image = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image[bounds] += 100
        np.testing.assert_array_equal(image[bounds].array, (sub_array+100),
            err_msg="image[bounds] += 100 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.ImageView[types[i]](sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] += 100 set wrong locations for dtype = "+str(types[i]))

        image = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image[bounds] -= 100
        np.testing.assert_array_equal(image[bounds].array, (sub_array-100),
            err_msg="image[bounds] -= 100 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.ImageView[types[i]](sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] -= 100 set wrong locations for dtype = "+str(types[i]))

        image = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image[bounds] *= 100
        np.testing.assert_array_equal(image[bounds].array, (sub_array*100),
            err_msg="image[bounds] *= 100 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.ImageView[types[i]](sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] *= 100 set wrong locations for dtype = "+str(types[i]))

        image = galsim.ImageView[types[i]]((100*ref_array).astype(types[i]))
        image[bounds] /= 100
        np.testing.assert_array_equal(image[bounds].array, (sub_array),
            err_msg="image[bounds] /= 100 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.ImageView[types[i]]((100*sub_array).astype(types[i]))
        np.testing.assert_array_equal(image.array, (100*ref_array),
            err_msg="image[bounds] /= 100 set wrong locations for dtype = "+str(types[i]))

        im2 = galsim.ImageView[types[i]](sub_array)
        image = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image[bounds] += im2
        np.testing.assert_array_equal(image[bounds].array, (2*sub_array),
            err_msg="image[bounds] += im2 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.ImageView[types[i]](sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] += im2 set wrong locations for dtype = "+str(types[i]))

        image = galsim.ImageView[types[i]](2*ref_array.astype(types[i]))
        image[bounds] -= im2
        np.testing.assert_array_equal(image[bounds].array, sub_array,
            err_msg="image[bounds] -= im2 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.ImageView[types[i]]((2*sub_array).astype(types[i]))
        np.testing.assert_array_equal(image.array, (2*ref_array),
            err_msg="image[bounds] -= im2 set wrong locations for dtype = "+str(types[i]))

        image = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image[bounds] *= im2
        np.testing.assert_array_equal(image[bounds].array, (sub_array**2),
            err_msg="image[bounds] *= im2 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.ImageView[types[i]](sub_array)
        np.testing.assert_array_equal(image.array, ref_array,
            err_msg="image[bounds] *= im2 set wrong locations for dtype = "+str(types[i]))

        image = galsim.ImageView[types[i]]((2 * ref_array**2).astype(types[i]))
        image[bounds] /= im2
        np.testing.assert_array_equal(image[bounds].array, (2*sub_array),
            err_msg="image[bounds] /= im2 does not set correctly for dtype = "+str(types[i]))
        image[bounds] = galsim.ImageView[types[i]]((2*sub_array**2).astype(types[i]))
        np.testing.assert_array_equal(image.array, (2*ref_array**2),
            err_msg="image[bounds] /= im2 set wrong locations for dtype = "+str(types[i]))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_ConstImageView_array_constness():
    """Test that ConstImageView instances cannot be modified via their .array attributes, and that
    if this is attempted a RuntimeError is raised.
    """
    import time
    t1 = time.time()
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image = galsim.ConstImageView[types[i]](ref_array.astype(types[i]))
        try:
            image.array[1, 2] = 666
        # Apparently older numpy versions might raise a RuntimeError, a ValueError, or a TypeError
        # when trying to write to arrays that have writeable=False. 
        # From the numpy 1.7.0 release notes:
        #     Attempting to write to a read-only array (one with
        #     ``arr.flags.writeable`` set to ``False``) used to raise either a
        #     RuntimeError, ValueError, or TypeError inconsistently, depending on
        #     which code path was taken. It now consistently raises a ValueError.
        except (RuntimeError, ValueError, TypeError):
            pass
        except:
            assert False, "Unexpected error: "+str(sys.exc_info()[0])
        # Then try using the eval command to mimic use via ConstImageViewD, etc.
        image_init_func = eval("galsim.ConstImageView"+tchar[i])
        image = image_init_func(ref_array.astype(types[i]))
        try:
            image.array[1, 2] = 666
        except (RuntimeError, ValueError, TypeError):
            pass
        except:
            assert False, "Unexpected error: "+str(sys.exc_info()[0])
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
            

if __name__ == "__main__":
    test_Image_basic()
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
    test_Image_inplace_add()
    test_Image_inplace_subtract()
    test_Image_inplace_multiply()
    test_Image_inplace_divide()
    test_Image_inplace_scalar_add()
    test_Image_inplace_scalar_subtract()
    test_Image_inplace_scalar_multiply()
    test_Image_inplace_scalar_divide()
    test_Image_subImage()
    test_ConstImageView_array_constness()
