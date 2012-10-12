"""
@file fits.py
Support for reading and writing galsim.Image* objects to FITS.

This file includes routines for reading and writing individual Images to/from FITS files, and also
routines for handling multiple Images.
"""
import os
from sys import byteorder
from . import _galsim

# Convert sys.byteorder into the notation numpy dtypes use
native_byteorder = {'big': '>', 'little': '<'}[byteorder]

def write(image, fits, add_wcs=True, clobber=True):
    """Write a single image to a FITS file.

    Write the image to a FITS file, with details depending on the arguments.  This function can be
    called directly as `galsim.fits.write(image, ...)`, with the image as the first argument, or as
    an image method: `image.write(...)`.

    @param image     The image to write to file.  Per the description of this method, it may be
                     given explicitly via `galsim.fits.write(image, ...)` or the method may be 
                     called directly as an image method, `image.write(...)`.
    @param fits      If `fits` is a pyfits.HDUList, the image will be appended as a new HDU.  In
                     that case, the user is responsible for calling fits.writeto(...) afterwards.
                     If `fits` is a string, it will be interpreted as a filename for a new FITS
                     file.
    @param add_wcs   If `add_wcs` evaluates to `True`, a 'LINEAR' WCS will be added using the 
                     Image's bounding box.  This is not necessary to ensure an Image can be 
                     round-tripped through FITS, as the bounding box (and scale) are always saved in
                     custom header keys.  If `add_wcs` is a string, this will be used as the WCS 
                     name. (Default `add_wcs = True`.)
    @param clobber   Setting `clobber=True` when `fits` is a string will silently overwrite existing
                     files. (Default `clobber = True`.)
    """
    import pyfits    # put this at function scope to keep pyfits optional

    if isinstance(fits, pyfits.HDUList):
        hdus = fits
    else:
        hdus = pyfits.HDUList()

    if len(hdus) == 0:
        hdu = pyfits.PrimaryHDU(image.array)
    else:
        hdu = pyfits.ImageHDU(image.array)
    hdus.append(hdu)

    # In PyFITS 3.1, the update method was deprecated in favor of subscript assignment.
    # When we no longer care about supporting versions before 3.1, we can switch these
    # to e.g. hdu.header['GS_SCALE'] = (image.scale , "GalSim Image scale")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        hdu.header.update("GS_SCALE", image.scale, "GalSim Image scale")
        hdu.header.update("GS_XMIN", image.xmin, "GalSim Image minimum X coordinate")
        hdu.header.update("GS_YMIN", image.xmin, "GalSim Image minimum Y coordinate")

        if add_wcs:
            if isinstance(add_wcs, basestring):
                wcsname = add_wcs
            else:
                wcsname = ""
            hdu.header.update("CTYPE1" + wcsname, "LINEAR", "name of the coordinate axis")
            hdu.header.update("CTYPE2" + wcsname, "LINEAR", "name of the coordinate axis")
            hdu.header.update("CRVAL1" + wcsname, 0, 
                            "coordinate system value at reference pixel")
            hdu.header.update("CRVAL2" + wcsname, 0, 
                            "coordinate system value at reference pixel")
            hdu.header.update("CRPIX1" + wcsname, 1-image.xmin, "coordinate system reference pixel")
            hdu.header.update("CRPIX2" + wcsname, 1-image.ymin, "coordinate system reference pixel")
            hdu.header.update("CD1_1" + wcsname, image.scale, "CD1_1 = pixel_scale")
            hdu.header.update("CD2_2" + wcsname, image.scale, "CD2_2 = pixel_scale")
            hdu.header.update("CD1_2" + wcsname, 0, "CD1_2 = 0")
            hdu.header.update("CD2_1" + wcsname, 0, "CD2_1 = 0")
    
    if isinstance(fits, basestring):
        if clobber and os.path.isfile(fits):
            os.remove(fits)
        hdus.writeto(fits)

def writeMulti(image_list, fits, add_wcs=True, clobber=True):
    """Write a Python list of images to a multi-extension FITS file.

    The details of how the images are written to file depends on the arguments.

    @param image_list A Python list of Images.
    @param fits       If `fits` is a `pyfits.HDUList`, the images will be appended as new HDUs.  The
                      user is responsible for calling `fits.writeto(...)` afterwards. If `fits` is a
                      string, it will be interpreted as a filename for a new multi-extension FITS
                      file.
    @param add_wcs    See documentation for this parameter on the galsim.fits.write method.
    @param clobber    See documentation for this parameter on the galsim.fits.write method.
    """
    import pyfits    # put this at function scope to keep pyfits optional

    if isinstance(fits, pyfits.HDUList):
        hdus = fits
    else:
        hdus = pyfits.HDUList()

    for image in image_list:
        write(image, hdus, add_wcs=add_wcs, clobber=clobber)

    if isinstance(fits, basestring):
        if clobber and os.path.isfile(fits):
            os.remove(fits)
        hdus.writeto(fits)


def writeCube(image_list, fits, add_wcs=True, clobber=True):
    """Write a Python list of images to a FITS file as a data cube.

    The details of how the images are written to file depends on the arguments.  Unlike for 
    writeMulti, when writing a data cube it is necessary that each Image in `image_list` has the 
    same size `(nx, ny)`.  No check is made to confirm that all images have the same origin and 
    pixel scale.

    @param image_list The `image_list` can also be either an array of NumPy arrays or a 3d NumPy
                      array, in which case this is written to the fits file directly.  In the former
                      case, no explicit check is made that the numpy arrays are all the same shape,
                      but a numpy exception will be raised which we let pass upstream unmolested.
    @param fits       If `fits` is a `pyfits.HDUList`, the cube will be appended as new HDUs.  The
                      user is responsible for calling `fits.writeto(...)` afterwards.  If `fits` is
                      a string, it will be interpreted as a filename for a new FITS file.
    @param add_wcs    See documentation for this parameter on the galsim.fits.write method.
    @param clobber    See documentation for this parameter on the galsim.fits.write method.
    """
    import numpy
    import pyfits    # put this at function scope to keep pyfits optional

    if isinstance(fits, pyfits.HDUList):
        hdus = fits
    else:
        hdus = pyfits.HDUList()

    try:
        cube = numpy.asarray(image_list)
        nimages = cube.shape[0]
        nx = cube.shape[1]
        ny = cube.shape[2]
        # Use default values for xmin, ymin, scale
        scale = 1
        xmin = 1
        ymin = 1
    except:
        nimages = len(image_list)
        if (nimages == 0):
            raise IndexError("In writeCube: image_list has no images")
        im = image_list[0]
        nx = im.xmax - im.xmin + 1
        ny = im.ymax - im.ymin + 1
        scale = im.scale
        xmin = im.xmin
        ymin = im.ymin
        # Note: numpy shape is y,x
        array_shape = (nimages, ny, nx)
        cube = numpy.array([[[]]])
        cube.resize(array_shape)
        for k in range(nimages):
            im = image_list[k]
            nx_k = im.xmax-im.xmin+1
            ny_k = im.ymax-im.ymin+1
            if nx_k != nx or ny_k != ny:
                raise IndexError("In writeCube: image %d has the wrong shape"%k +
                    "Shape is (%d,%d).  Should be (%d,%d)"%(nx_k,ny_k,nx,ny))
            cube[k,:,:] = image_list[k].array

    if len(hdus) == 0:
        hdu = pyfits.PrimaryHDU(cube)
    else:
        hdu = pyfits.ImageHDU(cube)
    hdus.append(hdu)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        hdu.header.update("GS_SCALE", scale, "GalSim Image scale")
        hdu.header.update("GS_XMIN", xmin, "GalSim Image minimum X coordinate")
        hdu.header.update("GS_YMIN", xmin, "GalSim Image minimum Y coordinate")

        if add_wcs:
            if isinstance(add_wcs, basestring):
                wcsname = add_wcs
            else:
                wcsname = ""
            hdu.header.update("CTYPE1" + wcsname, "LINEAR", "name of the coordinate axis")
            hdu.header.update("CTYPE2" + wcsname, "LINEAR", "name of the coordinate axis")
            hdu.header.update("CRVAL1" + wcsname, xmin, 
                            "coordinate system value at reference pixel")
            hdu.header.update("CRVAL2" + wcsname, ymin, 
                            "coordinate system value at reference pixel")
            hdu.header.update("CRPIX1" + wcsname, 1, "coordinate system reference pixel")
            hdu.header.update("CRPIX2" + wcsname, 1, "coordinate system reference pixel")
            hdu.header.update("CD1_1" + wcsname, scale, "CD1_1 = pixel_scale")
            hdu.header.update("CD2_2" + wcsname, scale, "CD2_2 = pixel_scale")
            hdu.header.update("CD1_2" + wcsname, 0, "CD1_2 = 0")
            hdu.header.update("CD2_1" + wcsname, 0, "CD2_1 = 0")
    
    if isinstance(fits, basestring):
        if clobber and os.path.isfile(fits):
            os.remove(fits)
        pyfits.writeto(fits,cube)


def read(fits):
    """Construct a new ImageView from a FITS representation.

    Not all FITS pixel types are supported (only those with C++ Image template instantiations are:
    `short`, `int`, `float`, and `double`).  If the FITS header has GS_* keywords, these will be 
    used to initialize the bounding box and scale.  If not, the bounding box will have `(xmin,ymin)`
    at `(1,1)` and the scale will be set to 1.0.

    This function is called as `im = galsim.fits.read(...)`

    @param fits    If `fits` is a `pyfits.HDUList`, the Primary HDU will be used.  If `fits` is a
                   `pyfits.PrimaryHDU` or `pyfits.ImageHDU`, that HDU will be used. If `fits` is a
                   string, it will be interpreted as a filename to open; the Primary HDU of that
                   file will be used.
    """
    import pyfits     # put this at function scope to keep pyfits optional
    
    if isinstance(fits, basestring):
        fits = pyfits.open(fits)
    if isinstance(fits, pyfits.HDUList):
        fits = fits[0]
    xmin = fits.header.get("GS_XMIN", 1)
    ymin = fits.header.get("GS_YMIN", 1)
    scale = fits.header.get("GS_SCALE", 1.0)
    pixel = fits.data.dtype.type
    try:
        Class = _galsim.ImageView[pixel]
    except KeyError:
        raise TypeError("No C++ Image template instantiation for pixel type %s" % pixel)
    # Check through byteorder possibilities, compare to native (used for numpy and our default) and
    # swap if necessary so that C++ gets the correct view.
    if fits.data.dtype.byteorder == '!':
        if native_byteorder == '>':
            pass
        else:
            fits.data.byteswap(True)
    elif fits.data.dtype.byteorder in (native_byteorder, '=', '@'):
        pass
    else:
        fits.data.byteswap(True)   # Note inplace is just an arg, not a kwarg, inplace=True throws
                                   # a TypeError exception in EPD Python 2.7.2
    image = Class(array=fits.data, xmin=xmin, ymin=ymin)
    image.scale = scale
    return image

def readMulti(fits):
    """Construct a Python list of ImageViews from a Multi-extension FITS file.

    Not all FITS pixel types are supported (only those with C++ Image template instantiations are:
    `short`, `int`, `float`, and `double`).  If the FITS header has GS_* keywords, these will be 
    used to initialize the bounding box and scale.  If not, the bounding box will have `(xmin,ymin)`
    at `(1,1)` and the scale will be set to 1.0.

    This function is called as `im = galsim.fits.readMulti(...)`

    @param   fits  If `fits` is a `pyfits.HDUList`, readMulti will read images from these.  If 
                   `fits` is a string, it will be interpreted as a filename to open and read.
    @returns A Python list of ImageView instances.
    """

    import pyfits     # put this at function scope to keep pyfits optional
    
    if isinstance(fits, basestring):
        hdu_list = pyfits.open(fits)
    elif isinstance(fits, pyfits.HDUList):
        hdu_list = fits
    else:
        raise TypeError("In readMulti, fits is not a string or HDUList")

    image_list = []
    for hdu in hdu_list:
        image_list.append(read(hdu))
    return image_list

def readCube(fits):
    """Construct a Python list of ImageViews from a FITS data cube.

    Not all FITS pixel types are supported (only those with C++ Image template instantiations are:
    `short`, `int`, `float`, and `double`).  If the FITS header has GS_* keywords, these will be  
    used to initialize the bounding boxes and scales.  If not, the bounding boxes will have 
    `(xmin,ymin)` at `(1,1)` and the scale will be set to 1.0.

    This function is called as `image_list = galsim.fits.readCube(...)`

    @param fits  If `fits` is a `pyfits.HDUList`, the Primary HDU will be used.  If `fits` is a
                 `pyfits.PrimaryHDU` or `pyfits.ImageHDU`, that HDU will be used.  If `fits` is a
                 string, it will be interpreted as a filename to open; the Primary HDU of that file
                 will be used.
    @returns     A Python list of ImageView instances.
    """
    import pyfits     # put this at function scope to keep pyfits optional
    
    if isinstance(fits, basestring):
        fits = pyfits.open(fits)
    if isinstance(fits, pyfits.HDUList):
        fits = fits[0]

    xmin = fits.header.get("GS_XMIN", 1)
    ymin = fits.header.get("GS_YMIN", 1)
    scale = fits.header.get("GS_SCALE", 1.0)
    pixel = fits.data.dtype.type

    try:
        Class = _galsim.ImageView[pixel]
    except KeyError:
        raise TypeError("No C++ Image template instantiation for pixel type %s" % pixel)
    # Check through byteorder possibilities, compare to native (used for numpy and our default) and
    # swap if necessary so that C++ gets the correct view.
    if fits.data.dtype.byteorder == '!':
        if native_byteorder == '>':
            pass
        else:
            fits.data.byteswap(True)
    elif fits.data.dtype.byteorder in (native_byteorder, '=', '@'):
        pass
    else:
        fits.data.byteswap(True)   # Note inplace is just an arg, not a kwarg, inplace=True throws
                                   # a TypeError exception in EPD Python 2.7.2

    nimages = fits.data.shape[0]
    image_list = []
    for k in range(nimages):
        image = Class(array=fits.data[k,:,:], xmin=xmin, ymin=ymin)
        image.scale = scale
        image_list.append(image)
    return image_list


# inject write as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.write = write

for Class in _galsim.ImageView.itervalues():
    Class.write = write

for Class in _galsim.ConstImageView.itervalues():
    Class.write = write

del Class    # cleanup public namespace
