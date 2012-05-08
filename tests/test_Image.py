import os
import sys
import pyfits
import numpy as np

"""Unit tests for the Image and ImageView classes.

These tests use four externally generated (IDL + astrolib FITS writing tools) reference images for
the Image unit tests.  These are in tests/data/.

Each image is 5x7 pixels^2 and if each pixel is labelled (x, y) then each pixel value is 10*x + y.
The array thus has values:

15 25 35 45 55
14 24 34 44 54
13 23 33 43 53  ^
12 22 32 42 52  |
11 21 31 41 51  y 

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

datadir = os.path.join(".", "Image_comparison_images")

def test_Image_basic():
    """Test that all supported types perform basic Image operations correctly
    """
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


def test_Image_FITS_IO():
    """Test that all four FITS reference images are correctly read in by both PyFITS and our Image 
    wrappers.
    """
    # The test fits file was made with the following array.  I didn't remake the file 
    # when I changed the ref_array to have ncol != nrow.
    fits_ref_array = np.array([
        [00, 10, 20, 30],
        [01, 11, 21, 31],
        [02, 12, 22, 32],
        [03, 13, 23, 33]]).astype(np.int16)
    for i in xrange(ntypes):
        # First try PyFITS for sanity
        testfile = os.path.join(datadir, "test"+tchar[i]+".fits")
        test_array = pyfits.getdata(testfile)
        np.testing.assert_array_equal(fits_ref_array.astype(types[i]), test_array,
                err_msg="PyFITS failing to read reference image.")
        # Then use the Image methods... Note this also relies on the array look working too
        image_init_func = eval("galsim.Image"+tchar[i]) # Use handy eval() mimics use of ImageSIFD
        # First give ImageSIFD.read() a PyFITS PrimaryHDU
        hdu = pyfits.open(testfile)
        # NB: The returned test_image is not necessarily of type image_init_func!
        test_image = image_init_func.read(hdu)
        np.testing.assert_array_equal(fits_ref_array.astype(types[i]), test_image.array, 
                err_msg="Image"+tchar[i]+".read() failed reading from PyFITS PrimaryHDU input.")
        # Then try an ImageSIFD.read() with the filename itself as input
        test_image = image_init_func.read(testfile)
        np.testing.assert_array_equal(fits_ref_array.astype(types[i]), test_image.array, 
                err_msg="Image"+tchar[i]+".read() failed reading from string filename input.")
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
                err_msg="Binary add scalar in Image class does not match reference for dtype = "
                +str(types[i]))
        image2 = 3 + image1
        np.testing.assert_array_equal((ref_array + 3).astype(types[i]), image2.array,
                err_msg="Binary radd scalar in Image class does not match reference for dtype = "
                +str(types[i]))

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
                err_msg="Binary add scalar in Image class does not match reference for dtype = "
                +str(types[i]))

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
                err_msg="Inplace add in Image class does not match reference for dtype = "
                +str(types[i]))
        for j in xrange(ntypes):
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func(ref_array.astype(types[i]))
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image1 += image2
            np.testing.assert_array_equal((3 * ref_array).astype(types[i]), image1.array,
                    err_msg="Inplace add in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

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
        for j in xrange(ntypes):
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func(ref_array.astype(types[i]))
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image2 -= image1
            np.testing.assert_array_equal(ref_array.astype(types[i]), image1.array,
                    err_msg="Inplace subtract in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

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
                err_msg="Inplace multiply in Image class does not match reference for dtype = "
                +str(types[i]))
        for j in xrange(ntypes):
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func(ref_array.astype(types[i]))
            image2 = image2_init_func((2 * ref_array).astype(types[j]))
            image2 *= image1
            np.testing.assert_array_equal((2 * ref_array**2).astype(types[i]), image2.array,
                    err_msg="Inplace multiply in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))

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
                err_msg="Inplace divide in Image class does not match reference for dtype = "
                +str(types[i]))
        for j in xrange(ntypes):
            image2_init_func = eval("galsim.ImageView"+tchar[j])
            image1 = image_init_func((ref_array+1).astype(types[i]))
            image2 = image2_init_func((2 * (ref_array+1)**2).astype(types[j]))
            image2 /= image1
            np.testing.assert_array_equal((2 * (ref_array+1)).astype(types[i]), image2.array,
                    err_msg="Inplace divide in Image class does not match reference for dtypes = "
                    +str(types[i])+" and "+str(types[j]))


def test_Image_inplace_scalar_add():
    """Test that all four types of supported Images inplace scalar add correctly.
    """
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

def test_Image_inplace_scalar_subtract():
    """Test that all four types of supported Images inplace scalar subtract correctly.
    """
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
                +"call) does not match reference for dtype = "+str(types[i]))
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
                +"call) does not match reference for dtype = "+str(types[i]))
        # Then try using the eval command to mimic use via ImageD, ImageF etc.
        image_init_func = eval("galsim.ImageView"+tchar[i])
        image1 = image_init_func(ref_array.astype(types[i]))
        image2 = image_init_func((2 * ref_array).astype(types[i]))
        image2 /= 2
        np.testing.assert_array_equal(image1.array, image2.array,
                err_msg="Inplace scalar divide in Image class does"
                +" not match reference for dtype = "+str(types[i]))
        
def test_Image_subImage():
    """Test that subImages are accessed and written correctly.
    """
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

def test_ConstImageView_array_constness():
    """Test that ConstImageView instances cannot be modified via their .array attributes, and that
    if this is attempted a RuntimeError is raised.
    """
    for i in xrange(ntypes):
        # First try using the dictionary-type Image init
        image = galsim.ConstImageView[types[i]](ref_array.astype(types[i]))
        try:
            image.array[1, 2] = 666
        except RuntimeError:
            pass
        except:
            assert False, "Unexpected error: "+str(sys.exc_info()[0])
        # Then try using the eval command to mimic use via ConstImageViewD, etc.
        image_init_func = eval("galsim.ConstImageView"+tchar[i])
        image = image_init_func(ref_array.astype(types[i]))
        try:
            image.array[1, 2] = 666
        except RuntimeError:
            pass
        except:
            assert False, "Unexpected error: "+str(sys.exc_info()[0])
            

if __name__ == "__main__":
    test_Image_basic()
    test_Image_FITS_IO()
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
