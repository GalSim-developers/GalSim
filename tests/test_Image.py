import os
import pyfits
import numpy as np
import galsim

"""Unit tests for the Image class.

These tests use four externally generated (IDL + astrolib FITS writing tools) reference images for
the Image unit tests.  These are in tests/data/.

Each image is 4x4 pixels^2 and if each pixel is labelled (x, y) then each pixel value is 10*x + y.
The array thus has values:

03 13 23 33
02 12 22 32  ^
01 11 21 31  |
00 10 20 30  y 

x ->

With array directions as indicated. This hopefully will make it easy enough to perform sub-image
checks, etc.

Images are in S, I, F & D flavours.
"""

# Setup info for tests, not likely to change
testshape = (4, 4)  # shape of image arrays for all tests
ntypes = 4
types = [np.int16, np.int32, np.float32, np.float64]
tchar = ['S', 'I', 'F', 'D']

ref_array = np.array([[00, 10, 20, 30], [01, 11, 21, 31], [02, 12, 22, 32],
                      [03, 13, 23, 33]]).astype(types[0])

datadir = os.path.join(".", "data")

def test_Image_XYmin_XYMax():
    """Test that all four types of supported arrays correctly set bounds based on an input array.
    """
    for array_type in types:
        test_array = np.zeros(testshape, dtype=array_type)
        image = galsim.Image[array_type](test_array)
        assert image.getXMin() == 1
        assert image.getYMin() == 1
        assert image.getXMax() == testshape[0]
        assert image.getYMax() == testshape[1]

def test_FITS_IO():
    """Test that all four FITS reference images are correctly read in by both PyFITS and our Image 
    wrappers.
    """
    for i in xrange(ntypes):
        # First try PyFITS for sanity
        testfile = os.path.join(datadir, "test"+tchar[i]+".fits")
        test_array = pyfits.getdata(testfile)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_array,
                                      err_msg="PyFITS failing to read reference image.")
        # Then use the Image methods... Note this also relies on the array look working too
        image_init_func = eval("galsim.Image"+tchar[i]) # Use handy eval() mimics use of ImageSIFD
        # First give ImageSIFD.read() a PyFITS PrimaryHDU
        hdu = pyfits.open(testfile)
        test_image = image_init_func.read(hdu)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array, 
                                      err_msg="Image"+tchar[i]+".read() failed reading from PyFITS"
                                              +"PrimaryHDU input.")
        # Then try an ImageSIFD.read() with the filename itself as input
        test_image = image_init_func.read(testfile)
        np.testing.assert_array_equal(ref_array.astype(types[i]), test_image.array, 
                                      err_msg="Image"+tchar[i]+".read() failed reading from string"
                                              +"filename input.")
        # TODO: test reading from an HDU list (e.g. for multi-extension FITS).

def test_Image_array_view():
    """Test that all four types of supported arrays correctly provide a view on an input array.
    """
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image = galsim.Image[types[i]](ref_array.astype(types[i]))
        np.testing.assert_array_equal(ref_array.astype(types[i]), image.array,
                                      err_msg="Array look into Image class (dictionary call) does"
                                              +" not match input for dtype = "+str(types[i]))
        #Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.Image"+tchar[i])
        image = image_init_func(ref_array.astype(types[i]))
        np.testing.assert_array_equal(ref_array.astype(types[i]), image.array,
                                      err_msg="Array look into Image class does not match input"
                                              +" for dtype = "+str(types[i]))
 


