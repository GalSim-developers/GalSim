"""
Support for reading and writing galsim.Image* objects to FITS, via new
Python-only methods injected into the Image classes.
"""
import os
from sys import byteorder
from . import _galsim

# Convert sys.byteorder into the notation numpy dtypes use
native_byteorder = {'big': '>', 'little': '<'}[byteorder]

def write(image, fits, add_wcs=True, clobber=True):
    """
    Write the image to a FITS file, with details depending on the type of
    the 'fits' argument:
    - If 'fits' is a pyfits.HDUList, the image will be appended as a new HDU.
      The user is responsible for calling fits.writeto(...) afterwards.
    - If 'fits' is a string, it will be interpreted as a filename for a new
      FITS file.
   
    If add_wcs evaluates to True, a 'LINEAR' WCS will be added using the Image's
    bounding box.  This is not necessary to ensure an Image can be round-tripped
    through FITS, as the bounding box (and scale) are always saved in custom header
    keys.  If add_wcs is a string, this will be used as the WCS name.

    This function can be called directly as "galsim.fits.write(image, ...)",
    with the image as the first argument, or as an image method: "image.write(...)".

    Setting clobber=True when 'fits' is a string will silently overwrite existing files.
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
    hdu.header.update("GS_SCALE", image.scale, "GalSim Image scale")
    hdu.header.update("GS_XMIN", image.xMin, "GalSim Image minimum X coordinate")
    hdu.header.update("GS_YMIN", image.xMin, "GalSim Image minimum Y coordinate")

    if add_wcs:
        if isinstance(add_wcs, basestring):
            wcsname = add_wcs
        else:
            wcsname = ""
        hdu.header.update("CTYPE1" + wcsname, "LINEAR", "name of the coordinate axis")
        hdu.header.update("CTYPE2" + wcsname, "LINEAR", "name of the coordinate axis")
        hdu.header.update("CRVAL1" + wcsname, image.xMin, 
                          "coordinate system value at reference pixel")
        hdu.header.update("CRVAL2" + wcsname, image.yMin, 
                          "coordinate system value at reference pixel")
        hdu.header.update("CRPIX1" + wcsname, 1, "coordinate system reference pixel")
        hdu.header.update("CRPIX2" + wcsname, 1, "coordinate system reference pixel")
    
    if isinstance(fits, basestring):
        if clobber and os.path.isfile(fits):
            os.remove(fits)
        hdus.writeto(fits)

def writeCube(image_list, fits, add_wcs=True, clobber=True):
    """
    Write the image to a FITS file as a data cube:
    - If 'fits' is a pyfits.HDUList, the images will be appended as new HDUs.
      The user is responsible for calling fits.writeto(...) afterwards.
    - If 'fits' is a string, it will be interpreted as a filename for a new
      FITS file.
   
    If add_wcs evaluates to True, a 'LINEAR' WCS will be added using each Image's
    bounding box.  This is not necessary to ensure an Image can be round-tripped
    through FITS, as the bounding box (and scale) are always saved in custom header
    keys.  If add_wcs is a string, this will be used as the WCS name.

    Setting clobber=True when 'fits' is a string will silently overwrite existing files.
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


def read(fits):
    """
    Construct a new ImageView from a FITS representation.
     - If 'fits' is a pyfits.HDUList, the Primary HDU will be used.
     - If 'fits' is a pyfits.PrimaryHDU or pyfits.ImageHDU, that HDU will be used.
     - If 'fits' is a string, it will be interpreted as a filename to open;
       the Primary HDU of that file will be used.

    If the FITS header has GS_* keywords, these will be used to initialize the
    bounding box and scale.  If not, the bounding box will have (xMin,yMin) at
    (1,1) and the scale will be set to 1.0.

    Not all FITS pixel types are supported (only those with C++ Image template
    instantiations are: short, int, float, and double).

    This function can be called directly as "galsim.fits.read(...)", or as a static
    method of an image class: "ImageD.read(...)".  Note, however, that in the
    latter case the image type returned is determined by the type of the FITS file,
    not the image class (in other words, "ImageD.read(...)" might return an ImageF).
    """
    # MJ: I find this last syntax: ImageD.read(...) a bit confusing, since as you 
    # point out the return value isn't necessarily the class you call it from.
    # Also, the return value is now an ImageView, not an Image, but that should
    # be transparent to the user.
    # So I'd recommend removing the ImageD.read(...) syntax and just having
    # the galsim.fits.read(...) syntax.

    import pyfits     # put this at function scope to keep pyfits optional
    
    if isinstance(fits, basestring):
        fits = pyfits.open(fits)
    if isinstance(fits, pyfits.HDUList):
        fits = fits[0]
    xMin = fits.header.get("GS_XMIN", 1)
    yMin = fits.header.get("GS_YMIN", 1)
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
    image = Class(array=fits.data, xMin=xMin, yMin=yMin)
    image.scale = scale
    return image

def readCube(fits):
    """
    Construct an array of ImageViews from a FITS data cube.
     - If 'fits' is a pyfits.HDUList, it will read images from these
     - If 'fits' is a string, it will be interpreted as a filename to open and read

    If the FITS header has GS_* keywords, these will be used to initialize the
    bounding box and scale.  If not, the bounding box will have (xMin,yMin) at
    (1,1) and the scale will be set to 1.0.

    Not all FITS pixel types are supported (only those with C++ Image template
    instantiations are: short, int, float, and double).
    """

    import pyfits     # put this at function scope to keep pyfits optional
    
    if isinstance(fits, basestring):
        hdu_list = pyfits.open(fits)
    elif isinstance(fits, pyfits.HDUList):
        hdu_list = fits
    else:
        raise TypeError("In readCube, fits is not a string or HDUList")

    images = []
    for hdu in hdu_list:
        images.append(read(hdu))
    return images


# inject read/write as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.write = write
    Class.read = staticmethod(read)

for Class in _galsim.ImageView.itervalues():
    Class.write = write
    Class.read = staticmethod(read)

for Class in _galsim.ConstImageView.itervalues():
    Class.write = write

del Class    # cleanup public namespace
