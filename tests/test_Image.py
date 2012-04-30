import os
import sys
import pyfits
import numpy as np

"""Unit tests for the Image and ImageView classes.

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

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# Setup info for tests, not likely to change
testshape = (4, 4)  # shape of image arrays for all tests
ntypes = 4
types = [np.int16, np.int32, np.float32, np.float64]
tchar = ['S', 'I', 'F', 'D']

ref_array = np.array([[00, 10, 20, 30], [01, 11, 21, 31], [02, 12, 22, 32],
                      [03, 13, 23, 33]]).astype(types[0])

datadir = os.path.join(".", "Image_comparison_images")

def test_Image_XYmin_XYMax():
    """Test that all four types of supported arrays correctly set bounds based on an input array.
    """
    for array_type in types:
        test_array = np.zeros(testshape, dtype=array_type)
        image = galsim.ImageView[array_type](test_array)
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
    """Test that all four types of supported Images correctly provide a view on an input array.
    """
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

def test_Image_binary_add():
    """Test that all four types of supported Images add correctly.
    """
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
                                      err_msg="Binary add in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

def test_Image_binary_subtract():
    """Test that all four types of supported Images subtract correctly.
    """
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
                                      err_msg="Binary subtract in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

def test_Image_binary_multiply():
    """Test that all four types of supported Images multiply correctly.
    """
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
                                      err_msg="Binary multiply in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

def test_Image_binary_divide():
    """Test that all four types of supported Images divide correctly.
    """
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
                                      err_msg="Binary divide in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

        # Test that the ZeroDivisionError is correctly thrown if some pixel = 0.
        image1.setValue(1,3,0)
        def div_helper(image1, image2, image3):
            image3 = image2 / image1
        # MJ: It turns out that numpy division doesn't throw a ZeroDivisionError.
        #     Instead it just silently calculates Inf or Nan for floats.
        #     For integers x / 0 -> 0.  Weird, but there you go.
        #     So if we do want an exception thrown, it looks like we'll have to 
        #     implement it ourselves.
        #np.testing.assert_raises(ZeroDivisionError, div_helper, image1, image2, image3)

def test_Image_binary_scalar_add():
    """Test that all four types of supported Images add scalars correctly.
    """
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
                err_msg="Binary add scalar in Image class does"
                +" not match reference for dtype = "+str(types[i]))
        image2 = 3 + image1
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary radd scalar in Image class does"
                +" not match reference for dtype = "+str(types[i]))

def test_Image_binary_scalar_subtract():
    """Test that all four types of supported Images binary scalar subtract correctly.
    """
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
                err_msg="Binary add scalar in Image class does"
                +" not match reference for dtype = "+str(types[i]))

def test_Image_binary_scalar_multiply():
    """Test that all four types of supported Images binary scalar multiply correctly.
    """
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

def test_Image_binary_scalar_divide():
    """Test that all four types of supported Images binary scalar divide correctly.
    """
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
        
        def div_helper(image1, image2, val):
            image2 = image1 / val
        #np.testing.assert_raises(ZeroDivisionError, div_helper, image1, image2, 0)


def test_Image_inplace_add():
    """Test that all four types of supported Images inplace add correctly.
    """
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
                                      err_msg="Inplace add in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

def test_Image_inplace_subtract():
    """Test that all four types of supported Images inplace subtract correctly.
    """
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image2 -= image1
        np.testing.assert_array_equal(ref_array.astype(types[i]), image2.array,
                                    err_msg="Inplace subtract in Image class (dictionary call) does"
                                             +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image2 -= image1
        np.testing.assert_array_equal(ref_array.astype(types[i]), image2.array,
                                      err_msg="Inplace subtract in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

def test_Image_inplace_multiply():
    """Test that all four types of supported Images inplace multiply correctly.
    """
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image2 *= image1
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image2.array,
                                    err_msg="Inplace multiply in Image class (dictionary call) does"
                                             +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image2 *= image1
        np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image2.array,
                                      err_msg="Inplace multiply in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

def test_Image_inplace_divide():
    """Test that all four types of supported Images inplace divide correctly.
    """
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]]((ref_array + 1).astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * (ref_array + 1)**2).astype(types[i]))
        image2 /= image1
        np.testing.assert_array_equal((2 * (ref_array + 1)).astype(types[i]), image2.array,
                                    err_msg="Inplace divide in Image class (dictionary call) does"
                                             +" not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func((ref_array + 1).astype(types[i]))
        image2 = image_init_func((2 * (ref_array + 1)**2).astype(types[i]))
        image2 /= image1
        np.testing.assert_array_equal((2 * (ref_array + 1)).astype(types[i]), image2.array,
                                      err_msg="Inplace divide in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

        image1.setValue(1,3,0)
        def idiv_helper(image1, image2):
            image2 /= image1
        #np.testing.assert_raises(ZeroDivisionError, idiv_helper, image1, image2)


def test_Image_inplace_scalar_add():
    """Test that all four types of supported Images inplace scalar add correctly.
    """
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image1 += 1
        np.testing.assert_array_equal((ref_array + 1).astype(types[i]), image1.array,
                                      err_msg="Inplace scalar add in Image class (dictionary "
                                             +"call) does not match reference for dtype = "
                                             +str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image1 += 1
        np.testing.assert_array_equal((ref_array + 1).astype(types[i]), image1.array,
                                      err_msg="Inplace scalar add in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

def test_Image_inplace_scalar_subtract():
    """Test that all four types of supported Images inplace scalar subtract correctly.
    """
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image1 -= 1
        np.testing.assert_array_equal((ref_array - 1).astype(types[i]), image1.array,
                                      err_msg="Inplace scalar subtract in Image class (dictionary "
                                             +"call) does not match reference for dtype = "
                                             +str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image1 -= 1
        np.testing.assert_array_equal((ref_array - 1).astype(types[i]), image1.array,
                                      err_msg="Inplace scalar subtract in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

def test_Image_inplace_scalar_multiply():
    """Test that all four types of supported Images inplace scalar multiply correctly.
    """
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image1 *= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                                      err_msg="Inplace scalar multiply in Image class (dictionary "
                                             +"call) does not match reference for dtype = "
                                             +str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image1 *= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                                      err_msg="Inplace scalar multiply in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))

def test_Image_inplace_scalar_divide():
    """Test that all four types of supported Images inplace scalar divide correctly.
    """
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image1 = galsim.ImageView[types[i]](ref_array.astype(types[i]))
        image2 = galsim.ImageView[types[i]]((2 * ref_array).astype(types[i]))
        image2 /= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                                      err_msg="Inplace scalar divide in Image class (dictionary "
                                             +"call) does not match reference for dtype = "
                                             +str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image2 /= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                                      err_msg="Inplace scalar divide in Image class does"
                                             +" not match reference for dtype = "+str(types[i]))
        
        def idiv_helper(image1, val):
            image2 /= val
        #np.testing.assert_raises(ZeroDivisionError, idivhelper, image2, 0)

if __name__ == "__main__":
    test_Image_XYmin_XYMax()
    test_FITS_IO()
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
