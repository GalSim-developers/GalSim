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
 
def parse_compression(compression, fits):
    file_compress = None
    pyfits_compress = None
    if compression == 'rice': pyfits_compress = 'RICE_1'
    elif compression == 'gzip_tile': pyfits_compress = 'GZIP_1'
    elif compression == 'hcompress': pyfits_compress = 'HCOMPRESS_1'
    elif compression == 'plio': pyfits_compress = 'PLIO_1'
    elif compression == 'gzip': file_compress = 'gzip'
    elif compression == 'bzip2': file_compress = 'bzip2'
    elif compression == 'none' or compression == None: pass
    elif compression == 'auto':
        if isinstance(fits, basestring):
            if fits.endswith('.fz'): pyfits_compress = 'RICE_1'
            elif fits.endswith('.gz'): file_compress = 'gzip'
            elif fits.endswith('.bz2'): file_compress = 'bzip2'
            else: pass
        else:
            # Default is None if fits is not a file name.
            pass
    else:
        raise TypeError("Invalid compression")
    return file_compress, pyfits_compress

# This is a class rather than a def, since we want to store a variable, and 
# python doesn't really have static variables.  But this will be used as though
# it were a normal function: read_file(file, file_compress)
class _ReadFile:
    def __init__(self):
        # Store whether we have a bad interaction between gzip and pyfits, so we 
        # don't need to keep trying code that doesn't work after the first time 
        # we discover it fails.
        self.gzip_in_mem = True
        self.bz2_in_mem = True

    def __call__(self, file, file_compress):
        import pyfits
        if not file_compress:
            hdus = pyfits.open(file, 'readonly')
            return hdus, None
        elif file_compress == 'gzip':
            import gzip
            if self.gzip_in_mem:
                try:
                    fin = gzip.GzipFile(file, 'rb')
                    hdus = pyfits.open(fin, 'readonly')
                    # Sometimes this doesn't work.  The symptoms may be that this raises an
                    # exception, or possibly the hdus list comes back empty, in which case the 
                    # next line will raise an exception.
                    hdu = hdus[0]
                    # pyfits doesn't actually read the file yet, so we can't close fin here.
                    # Need to pass it back to the caller and let them close it when they are 
                    # done with hdus.
                    return hdus, fin
                except:
                    # Mark that we can't do this the efficient way so next time (and afterward)
                    # it will use the below code instead.
                    self.gzip_in_mem = False
                    return self(file,file_compress)
            else:
                try:
                    # This usually works, although pyfits internally uses a temporary file,
                    # which is why we prefer the above code if it works.
                    hdus = pyfits.open(file, 'readonly')
                    return hdus, None
                except:
                    # But just in case, here is an implementation that should always work.
                    fin = gzip.GzipFile(file, 'rb')
                    data = fin.read()
                    tmp = file + '.tmp'
                    # It would be pretty odd for this filename to already exist, but just in case...
                    while os.path.isfile(tmp):
                        tmp = tmp + '.tmp'
                    tmpout = open(tmp,"w")
                    tmpout.write(data)
                    tmpout.close()
                    hdus = pyfits.open(tmp)
                    return hdus, tmp
        elif file_compress == 'bzip2':
            import bz2
            if self.bz2_in_mem:
                try:
                    # This normally works.  But it might not on old versions of pyfits.
                    fin = bz2.BZ2File(file, 'rb')
                    hdus = pyfits.open(fin, 'readonly')
                    hdu = hdus[0]
                    return hdus, fin
                except:
                    # Mark that we can't do this the efficient way so next time (and afterward)
                    # it will use the below code instead.
                    self.bz2_in_mem = False
                    return self(file,file_compress)
            else:
                fin = bz2.BZ2File(file, 'rb')
                data = fin.read()
                tmp = file + '.tmp'
                # It would be pretty odd for this filename to already exist, but just in case...
                while os.path.isfile(tmp):
                    tmp = tmp + '.tmp'
                tmpout = open(tmp,"w")
                tmpout.write(data)
                tmpout.close()
                hdus = pyfits.open(tmp)
                return hdus, tmp
        else:
            raise ValueError("Unknown file_compression")
read_file = _ReadFile()

# Do the same trick for write_file(file,hdus,clobber,file_compress):
class _WriteFile:
    def __init__(self):
        # Store whether it is ok to use the in-memory version.
        self.in_mem = True

    def __call__(self, file, hdus, clobber, file_compress):
        import os
        if os.path.isfile(file):
            if clobber:
                os.remove(file)
            else:
                raise IOError('File %r already exists'%file)
    
        if not file_compress:
            hdus.writeto(file)
        else:
            if self.in_mem:
                try:
                    # The compression routines work better if we first write to an internal buffer
                    # and then output that to a file.
                    import io
                    buf = io.BytesIO()
                    hdus.writeto(buf)
                    data = buf.getvalue()
                except:
                    self.in_mem = False
                    return self(file,hdus,clobber,file_compress)
            else:
                # However, pyfits versions before 2.3 do not support writing to a buffer, so the
                # abover code with fail.  We need to use a temporary in that case.
                tmp = file + '.tmp'
                # It would be pretty odd for this filename to already exist, but just in case...
                while os.path.isfile(tmp):
                    tmp = tmp + '.tmp'
                hdus.writeto(tmp)
                buf = open(tmp,"r")
                data = buf.read()
                buf.close()
                os.remove(tmp)

            if file_compress == 'gzip':
                import gzip
                # There is a compresslevel option (for both gzip and bz2), but we just use the 
                # default.
                fout = gzip.GzipFile(file, 'wb')  
            elif file_compress == 'bzip2':
                import bz2
                fout = bz2.BZ2File(file, 'wb')
            else:
                raise ValueError("Unknown file_compression")
    
            fout.write(data)
            fout.close()
write_file = _WriteFile()

def write_header(hdu, add_wcs, scale, xmin, ymin):
    # In PyFITS 3.1, the update method was deprecated in favor of subscript assignment.
    # When we no longer care about supporting versions before 3.1, we can switch these
    # to e.g. hdu.header['GS_SCALE'] = (image.scale , "GalSim Image scale")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        hdu.header.update("GS_SCALE", scale, "GalSim Image scale")
        hdu.header.update("GS_XMIN", xmin, "GalSim Image minimum X coordinate")
        hdu.header.update("GS_YMIN", ymin, "GalSim Image minimum Y coordinate")

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


def add_hdu(hdus, data, pyfits_compress):
    import pyfits
    if len(hdus) == 0:
        if pyfits_compress:
            hdus.append(pyfits.PrimaryHDU())  # Need a blank PrimaryHDU
            hdu = pyfits.CompImageHDU(data, compressionType=pyfits_compress)
        else:
            hdu = pyfits.PrimaryHDU(data)
    else:
        if pyfits_compress:
            hdu = pyfits.CompImageHDU(data, compressionType=pyfits_compress)
        else:
            hdu = pyfits.ImageHDU(data)
    hdus.append(hdu)
    return hdu


def write(image, fits, add_wcs=True, clobber=True, compression='auto'):
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
    @param compression  Which compression scheme to use (if any).  Options are:
                        None or 'none' = no compression
                        'rice' = use rice compression in tiles (preserves header readability)
                        'gzip' = use gzip to compress the full file
                        'bzip2' = use bzip2 to compress the full file
                        'gzip_tile' = use gzip in tiles (preserves header readability)
                        'hcompress' = use hcompress in tiles (only valid for 2-d images)
                        'plio' = use plio compression in tiles (only valid for pos integer data)
                        'auto' = determine the compression from the extension of the file name
                            (requires fits to be a string).  
                            '*.fz' => 'rice'
                            '*.gz' => 'gzip'
                            '*.bz2' => 'bzip2'
                            otherwise None
    """
    import pyfits    # put this at function scope to keep pyfits optional

    file_compress, pyfits_compress = parse_compression(compression,fits)

    if isinstance(fits, pyfits.HDUList):
        hdus = fits
    else:
        hdus = pyfits.HDUList()

    hdu = add_hdu(hdus, image.array, pyfits_compress)
    write_header(hdu, add_wcs, image.scale, image.xmin, image.ymin)
   
    if isinstance(fits, basestring):
        write_file(fits,hdus,clobber,file_compress)


def writeMulti(image_list, fits, add_wcs=True, clobber=True, compression='auto'):
    """Write a Python list of images to a multi-extension FITS file.

    The details of how the images are written to file depends on the arguments.

    @param image_list A Python list of Images.
    @param fits       If `fits` is a `pyfits.HDUList`, the images will be appended as new HDUs.  The
                      user is responsible for calling `fits.writeto(...)` afterwards. If `fits` is a
                      string, it will be interpreted as a filename for a new multi-extension FITS
                      file.
    @param add_wcs    See documentation for this parameter on the galsim.fits.write method.
    @param clobber    See documentation for this parameter on the galsim.fits.write method.
    @param compression See documentation for this parameter on the galsim.fits.write method.
    """
    import pyfits    # put this at function scope to keep pyfits optional

    file_compress, pyfits_compress = parse_compression(compression,fits)

    if isinstance(fits, pyfits.HDUList):
        hdus = fits
    else:
        hdus = pyfits.HDUList()

    for image in image_list:
        hdu = add_hdu(hdus, image.array, pyfits_compress)
        write_header(hdu, add_wcs, image.scale, image.xmin, image.ymin)
   
    if isinstance(fits, basestring):
        write_file(fits,hdus,clobber,file_compress)


def writeCube(image_list, fits, add_wcs=True, clobber=True, compression='auto'):
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
    @param compression See documentation for this parameter on the galsim.fits.write method.
    """
    import numpy
    import pyfits    # put this at function scope to keep pyfits optional

    file_compress, pyfits_compress = parse_compression(compression,fits)

    if isinstance(fits, pyfits.HDUList):
        hdus = fits
    else:
        hdus = pyfits.HDUList()

    is_all_numpy = (isinstance(image_list, numpy.ndarray) or
                    all(isinstance(item, numpy.ndarray) for item in image_list))
    if is_all_numpy:
        cube = numpy.asarray(image_list)
        nimages = cube.shape[0]
        nx = cube.shape[1]
        ny = cube.shape[2]
        # Use default values for xmin, ymin, scale
        scale = 1
        xmin = 1
        ymin = 1
    else:
        nimages = len(image_list)
        if (nimages == 0):
            raise IndexError("In writeCube: image_list has no images")
        im = image_list[0]
        dtype = im.array.dtype
        nx = im.xmax - im.xmin + 1
        ny = im.ymax - im.ymin + 1
        scale = im.scale
        xmin = im.xmin
        ymin = im.ymin
        # Note: numpy shape is y,x
        array_shape = (nimages, ny, nx)
        cube = numpy.zeros(array_shape, dtype=dtype)
        for k in range(nimages):
            im = image_list[k]
            nx_k = im.xmax-im.xmin+1
            ny_k = im.ymax-im.ymin+1
            if nx_k != nx or ny_k != ny:
                raise IndexError("In writeCube: image %d has the wrong shape"%k +
                    "Shape is (%d,%d).  Should be (%d,%d)"%(nx_k,ny_k,nx,ny))
            cube[k,:,:] = image_list[k].array

    hdu = add_hdu(hdus, cube, pyfits_compress)
    write_header(hdu, add_wcs, scale, xmin, ymin)

    if isinstance(fits, basestring):
        write_file(fits,hdus,clobber,file_compress)


def read(fits, compression='auto'):
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
    @param compression  Which decompression scheme to use (if any).  Options are:
                        None or 'none' = no decompression
                        'rice' = use rice decompression in tiles
                        'gzip' = use gzip to decompress the full file
                        'bzip2' = use bzip2 to decompress the full file
                        'gzip_tile' = use gzip decompression in tiles
                        'hcompress' = use hcompress decompression in tiles
                        'plio' = use plio decompression in tiles
                        'auto' = determine the decompression from the extension of the file name
                            (requires fits to be a string).  
                            '*.fz' => 'rice'
                            '*.gz' => 'gzip'
                            '*.bz2' => 'bzip2'
                            otherwise None
    """
    import pyfits     # put this at function scope to keep pyfits optional
    
    file_compress, pyfits_compress = parse_compression(compression,fits)

    fin = None
    if isinstance(fits, basestring):
        hdus, fin = read_file(fits, file_compress)
        fits = hdus

    if isinstance(fits, pyfits.HDUList):
        # Note: Nothing special needs to be done when reading a compressed hdu.
        # However, such compressed hdu's may not be the PrimaryHDU, so if we think we are
        # reading a compressed file, skip to hdu 1.
        if pyfits_compress:
            if len(fits) < 2:
                raise IOError('Expecting at least one extension HDU in galsim.read')
            fits = fits[1]
        else:
            if len(fits) < 1:
                raise IOError('Expecting at least one HDU in galsim.read')
            fits = fits[0]

    xmin = fits.header.get("GS_XMIN", 1)
    ymin = fits.header.get("GS_YMIN", 1)
    scale = fits.header.get("GS_SCALE", 1.0)
    pixel = fits.data.dtype.type
    if pixel in _galsim.ImageView.keys():
        Class = _galsim.ImageView[pixel]
        data = fits.data
    else:
        import warnings
        warnings.warn("No C++ Image template instantiation for pixel type %s" % pixel)
        warnings.warn("Using float")
        Class = _galsim.ImageViewD
        import numpy
        data = fits.data.astype(numpy.float64)

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

    image = Class(array=data, xmin=xmin, ymin=ymin)
    image.scale = scale

    # If we opened a file, don't forget to close it.
    if fin: 
        hdus.close()
        if isinstance(fin, basestring):
            # In this case, it is a file name that we need to delete.
            import os
            os.remove(fin)
        else:
            fin.close()

    return image

def readMulti(fits, compression='auto'):
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
    
    file_compress, pyfits_compress = parse_compression(compression,fits)

    fin = None
    if isinstance(fits, basestring):
        hdus, fin = read_file(fits, file_compress)
        fits = hdus
    elif not isinstance(fits, pyfits.HDUList):
        raise TypeError("In readMulti, fits is not a string or HDUList")

    image_list = []
    if pyfits_compress:
        first = 1
        if len(fits) < 2:
            raise IOError('Expecting at least one extension HDU in galsim.readMulti')
    else:
        first = 0
        if len(fits) < 1:
            raise IOError('Expecting at least one HDU in galsim.readMulti')
    for hdu in fits[first:]:
        image_list.append(read(hdu))

    # If we opened a file, don't forget to close it.
    if fin:
        hdus.close()
        if isinstance(fin, basestring):
            # In this case, it is a file name that we need to delete.
            import os
            os.remove(fin)
        else:
            fin.close()

    return image_list

def readCube(fits, compression='auto'):
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
    
    file_compress, pyfits_compress = parse_compression(compression,fits)

    fin = None
    if isinstance(fits, basestring):
        hdus, fin = read_file(fits, file_compress)
        fits = hdus
    
    if isinstance(fits, pyfits.HDUList):
        # Note: Nothing special needs to be done when reading a compressed hdu.
        # However, such compressed hdu's may not be the PrimaryHDU, so if we think we are
        # reading a compressed file, skip to hdu 1.
        if pyfits_compress:
            if len(fits) < 2:
                raise IOError('Expecting at least one extension HDU in galsim.readCube')
            fits = fits[1]
        else:
            if len(fits) < 1:
                raise IOError('Expecting at least one HDU in galsim.readCube')
            fits = fits[0]

    xmin = fits.header.get("GS_XMIN", 1)
    ymin = fits.header.get("GS_YMIN", 1)
    scale = fits.header.get("GS_SCALE", 1.0)
    pixel = fits.data.dtype.type
    if pixel in _galsim.ImageView.keys():
        Class = _galsim.ImageView[pixel]
        data = fits.data
    else:
        import warnings
        warnings.warn("No C++ Image template instantiation for pixel type %s" % pixel)
        warnings.warn("Using float")
        Class = _galsim.ImageViewD
        import numpy

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

    # If we opened a file, don't forget to close it.
    if fin: 
        hdus.close()
        if isinstance(fin, basestring):
            # In this case, it is a file name that we need to delete.
            import os
            os.remove(fin)
        else:
            fin.close()


    return image_list


# inject write as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.write = write

for Class in _galsim.ImageView.itervalues():
    Class.write = write

for Class in _galsim.ConstImageView.itervalues():
    Class.write = write

del Class    # cleanup public namespace
