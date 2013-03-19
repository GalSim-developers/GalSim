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
"""@file fits.py
Support for reading and writing galsim.Image* objects to FITS.

This file includes routines for reading and writing individual Images to/from FITS files, and also
routines for handling multiple Images.
"""


import os
from sys import byteorder
from . import _galsim

# Convert sys.byteorder into the notation numpy dtypes use
native_byteorder = {'big': '>', 'little': '<'}[byteorder]

 
def parse_compression(compression, file_name):
    file_compress = None
    pyfits_compress = None
    if compression == 'rice' or compression == 'RICE_1': pyfits_compress = 'RICE_1'
    elif compression == 'gzip_tile' or compression == 'GZIP_1': pyfits_compress = 'GZIP_1'
    elif compression == 'hcompress' or compression == 'HCOMPRESS_1': pyfits_compress = 'HCOMPRESS_1'
    elif compression == 'plio' or compression == 'PLIO_1': pyfits_compress = 'PLIO_1'
    elif compression == 'gzip': file_compress = 'gzip'
    elif compression == 'bzip2': file_compress = 'bzip2'
    elif compression == 'none' or compression == None: pass
    elif compression == 'auto':
        if file_name:
            if file_name.lower().endswith('.fz'): pyfits_compress = 'RICE_1'
            elif file_name.lower().endswith('.gz'): file_compress = 'gzip'
            elif file_name.lower().endswith('.bz2'): file_compress = 'bzip2'
            else: pass
    else:
        raise TypeError("Invalid compression")
    if pyfits_compress:
        import pyfits
        if 'CompImageHDU' not in pyfits.__dict__:
            raise NotImplementedError(
                'Compressed Images not supported before pyfits version 2.0. You have version %s.'%(
                    pyfits.__version__))
            
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

# Do the same trick for write_file(file,hdus,clobber,file_compress,pyfits_compress):
class _WriteFile:
    def __init__(self):
        # Store whether it is ok to use the in-memory version.
        self.in_mem = True

    def __call__(self, file, hdus, clobber, file_compress, pyfits_compress):
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

        # There is a bug in pyfits where they don't add the size of the variable length array
        # to the TFORMx header keywords.  They should have size at the end of them.
        # This bug has been fixed in version 3.1.2.
        # (See http://trac.assembla.com/pyfits/ticket/199)
        import pyfits
        if pyfits_compress and pyfits.__version__ < '3.1.2':
            hdus = pyfits.open(file,'update',disable_image_compression=True)
            for hdu in hdus[1:]: # Skip PrimaryHDU
                # Find the maximum variable array length  
                max_ar_len = max([ len(ar[0]) for ar in hdu.data ])
                # Add '(N)' to the TFORMx keywords for the variable array items
                s = '(%d)'%max_ar_len
                for key in hdu.header.keys():
                    if key.startswith('TFORM'):
                        tform = hdu.header[key]
                        # Only update if the form is a P (= variable length data)
                        # and the (*) is not there already.
                        if 'P' in tform and '(' not in tform:
                            hdu.header[key] = tform + s
            hdus.close()

            # Workaround for a bug in some pyfits 3.0.x versions
            # It was fixed in 3.0.8.  I'm not sure when the bug was 
            # introduced, but I believe it was 3.0.3.  
            if (pyfits.__version__ > '3.0' and pyfits.__version__ < '3.0.8' and
                'COMPRESSION_ENABLED' in pyfits.hdu.compressed.__dict__):
                pyfits.hdu.compressed.COMPRESSION_ENABLED = True
                
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
    if pyfits_compress:
        if len(hdus) == 0:
            hdus.append(pyfits.PrimaryHDU())  # Need a blank PrimaryHDU
        hdu = pyfits.CompImageHDU(data, compressionType=pyfits_compress)
    else:
        if len(hdus) == 0:
            hdu = pyfits.PrimaryHDU(data)
        else:
            hdu = pyfits.ImageHDU(data)
    hdus.append(hdu)
    return hdu

def check_hdu(hdu, pyfits_compress):
    """Check that an input hdu is valid
    """
    import pyfits
    if pyfits_compress:
        if not isinstance(hdu, pyfits.CompImageHDU):
            #print 'pyfits_compress = ',pyfits_compress
            #print 'hdu = ',hdu
            if isinstance(hdu, pyfits.BinTableHDU):
                raise IOError('Expecting a CompImageHDU, but got a BinTableHDU\n' +
                    'Probably your pyfits installation does not have the pyfitsComp module '+
                    'installed.')
            elif isinstance(hdu, pyfits.ImageHDU):
                import warnings
                warnings.warn("Expecting a CompImageHDU, but found an uncompressed ImageHDU")
            else:
                raise IOError('Found invalid HDU reading FITS file (expected an ImageHDU)')
    else:
        if not isinstance(hdu, pyfits.ImageHDU) and not isinstance(hdu, pyfits.PrimaryHDU):
            #print 'pyfits_compress = ',pyfits_compress
            #print 'hdu = ',hdu
            raise IOError('Found invalid HDU reading FITS file (expected an ImageHDU)')



def write(image, file_name=None, dir=None, hdu_list=None, add_wcs=True, clobber=True,
          compression='auto'):
    """Write a single image to a FITS file.

    Write the image to a FITS file, with details depending on the arguments.  This function can be
    called directly as `galsim.fits.write(image, ...)`, with the image as the first argument, or as
    an image method: `image.write(...)`.

    @param image        The image to write to file.  Per the description of this method, it may be
                        given explicitly via `galsim.fits.write(image, ...)` or the method may be 
                        called directly as an image method, `image.write(...)`.
    @param file_name    The name of the file to write to.  Either `file_name` or `hdu_list` is 
                        required.
    @param dir          Optionally a directory name can be provided if the file_name does not 
                        already include it.
    @param hdu_list     A pyfits HDUList.  If this is provided instead of file_name, then the 
                        image is appended to the end of the HDUList as a new HDU. In that case, 
                        the user is responsible for calling fits.writeto(...) afterwards.  Either
                        `file_name` or `hdu_list` is required.
    @param add_wcs      If `add_wcs` evaluates to `True`, a 'LINEAR' WCS will be added using the 
                        Image's bounding box.  This is not necessary to ensure an Image can be 
                        round-tripped through FITS, as the bounding box (and scale) are always 
                        saved in custom header keys.  If `add_wcs` is a string, this will be used 
                        as the WCS name. (Default `add_wcs = True`.)
    @param clobber      Setting `clobber=True` when `fits` is a string will silently overwrite 
                        existing files. (Default `clobber = True`.)
    @param compression  Which compression scheme to use (if any).  Options are:
                        - None or 'none' = no compression
                        - 'rice' = use rice compression in tiles (preserves header readability)
                        - 'gzip' = use gzip to compress the full file
                        - 'bzip2' = use bzip2 to compress the full file
                        - 'gzip_tile' = use gzip in tiles (preserves header readability)
                        - 'hcompress' = use hcompress in tiles (only valid for 2-d images)
                        - 'plio' = use plio compression in tiles (only valid for pos integer data)
                        - 'auto' = determine the compression from the extension of the file name
                                   (requires fits to be a string).  
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
    """
    import pyfits    # put this at function scope to keep pyfits optional
  
    file_compress, pyfits_compress = parse_compression(compression,file_name)

    if file_name and hdu_list:
        raise TypeError("Cannot provide both file_name and hdu_list to write()")
    if not (file_name or hdu_list):
        raise TypeError("Must provide either file_name or hdu_list to write()")

    if not hdu_list:
        hdu_list = pyfits.HDUList()

    hdu = add_hdu(hdu_list, image.array, pyfits_compress)
    write_header(hdu, add_wcs, image.scale, image.xmin, image.ymin)

    if file_name:
        if dir:
            import os
            file_name = os.path.join(dir,file_name)
        write_file(file_name, hdu_list, clobber, file_compress, pyfits_compress)


def writeMulti(image_list, file_name=None, dir=None, hdu_list=None, add_wcs=True, clobber=True,
               compression='auto'):
    """Write a Python list of images to a multi-extension FITS file.

    The details of how the images are written to file depends on the arguments.

    @param image_list   A Python list of Images.
    @param file_name    The name of the file to write to.  Either `file_name` or `hdu_list` is 
                        required.
    @param dir          Optionally a directory name can be provided if the file_name does not 
                        already include it.
    @param hdu_list     A pyfits HDUList.  If this is provided instead of file_name, then the 
                        images are appended to the end of the HDUList as a new HDU. In that case, 
                        the user is responsible for calling fits.writeto(...) afterwards.  Either
                        `file_name` or `hdu_list` is required.
    @param add_wcs      See documentation for this parameter on the galsim.fits.write method.
    @param clobber      See documentation for this parameter on the galsim.fits.write method.
    @param compression  See documentation for this parameter on the galsim.fits.write method.
    """
    import pyfits    # put this at function scope to keep pyfits optional

    file_compress, pyfits_compress = parse_compression(compression,file_name)

    if file_name and hdu_list:
        raise TypeError("Cannot provide both file_name and hdu_list to write()")
    if not (file_name or hdu_list):
        raise TypeError("Must provide either file_name or hdu_list to write()")

    if not hdu_list:
        hdu_list = pyfits.HDUList()

    for image in image_list:
        hdu = add_hdu(hdu_list, image.array, pyfits_compress)
        write_header(hdu, add_wcs, image.scale, image.xmin, image.ymin)

    if file_name:
        if dir:
            import os
            file_name = os.path.join(dir,file_name)
        write_file(file_name, hdu_list, clobber, file_compress, pyfits_compress)



def writeCube(image_list, file_name=None, dir=None, hdu_list=None, add_wcs=True, clobber=True,
              compression='auto'):
    """Write a Python list of images to a FITS file as a data cube.

    The details of how the images are written to file depends on the arguments.  Unlike for 
    writeMulti, when writing a data cube it is necessary that each Image in `image_list` has the 
    same size `(nx, ny)`.  No check is made to confirm that all images have the same origin and 
    pixel scale.

    @param image_list   The `image_list` can also be either an array of NumPy arrays or a 3d NumPy
                        array, in which case this is written to the fits file directly.  In the 
                        former case, no explicit check is made that the numpy arrays are all the 
                        same shape, but a numpy exception will be raised which we let pass upstream
                        unmolested.
    @param file_name    The name of the file to write to.  Either `file_name` or `hdu_list` is 
                        required.
    @param dir          Optionally a directory name can be provided if the file_name does not 
                        already include it.
    @param hdu_list     A pyfits HDUList.  If this is provided instead of file_name, then the 
                        cube is appended to the end of the HDUList as a new HDU. In that case, 
                        the user is responsible for calling fits.writeto(...) afterwards.  Either
                        `file_name` or `hdu_list` is required.
    @param add_wcs      See documentation for this parameter on the galsim.fits.write method.
    @param clobber      See documentation for this parameter on the galsim.fits.write method.
    @param compression  See documentation for this parameter on the galsim.fits.write method.
    """
    import numpy
    import pyfits    # put this at function scope to keep pyfits optional

    file_compress, pyfits_compress = parse_compression(compression,file_name)

    if file_name and hdu_list:
        raise TypeError("Cannot provide both file_name and hdu_list to write()")
    if not (file_name or hdu_list):
        raise TypeError("Must provide either file_name or hdu_list to write()")

    if not hdu_list:
        hdu_list = pyfits.HDUList()

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

    hdu = add_hdu(hdu_list, cube, pyfits_compress)
    write_header(hdu, add_wcs, scale, xmin, ymin)

    if file_name:
        if dir:
            import os
            file_name = os.path.join(dir,file_name)
        write_file(file_name, hdu_list, clobber, file_compress, pyfits_compress)



def read(file_name=None, dir=None, hdu_list=None, hdu=0, compression='auto'):
    """Construct an Image from a FITS file or pyfits HDUList.

    The normal usage for this function is to read a fits file and return the image contained
    therein, automatically decompressing it if necessary.  However, you may also pass it 
    an HDUList, in which case it will select the indicated hdu (with the hdu parameter) 
    from that.

    Not all FITS pixel types are supported (only those with C++ Image template instantiations:
    `short`, `int`, `float`, and `double`).  If the FITS header has GS_* keywords, these will be 
    used to initialize the bounding box and scale.  If not, the bounding box will have `(xmin,ymin)`
    at `(1,1)` and the scale will be set to 1.0.

    This function is called as `im = galsim.fits.read(...)`

    @param file_name    The name of the file to read in.  Either `file_name` or `hdu_list` is 
                        required.
    @param dir          Optionally a directory name can be provided if the file_name does not 
                        already include it.
    @param hdu_list     Either a `pyfits.HDUList`, a `pyfits.PrimaryHDU`, or `pyfits.ImageHDU`.
                        In the former case, the `hdu` in the list will be selected.  In the latter
                        two cases, the `hdu` parameter is ignored.  Either `file_name` or 
                        `hdu_list` is required.
    @param hdu          The number of the HDU to use.  The default is to use the primary HDU,
                        which is numbered 0.
    @param compression  Which decompression scheme to use (if any).  Options are:
                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                                   (requires fits to be a string).  
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
    @returns An ImageView instance
    """
    import pyfits     # put this at function scope to keep pyfits optional
    
    file_compress, pyfits_compress = parse_compression(compression,file_name)

    if file_name and hdu_list:
        raise TypeError("Cannot provide both file_name and hdu_list to read()")
    if not (file_name or hdu_list):
        raise TypeError("Must provide either file_name or hdu_list to read()")

    fin = None
    if file_name:
        if dir:
            import os
            file_name = os.path.join(dir,file_name)
        hdu_list, fin = read_file(file_name, file_compress)

    if isinstance(hdu_list, pyfits.HDUList):
        # Note: Nothing special needs to be done when reading a compressed hdu.
        # However, such compressed hdu's may not be the PrimaryHDU, so if we think we are
        # reading a compressed file, skip to hdu 1.
        if pyfits_compress and hdu==0:
            if len(hdu_list) <= 1:
                raise IOError('Expecting at least one extension HDU in galsim.read')
            hdu = 1
        elif len(hdu_list) <= hdu:
            raise IOError('Expecting at least %d HDUs in galsim.read'%(hdu+1))
        fits = hdu_list[hdu]
    else:
        fits = hdu_list
    check_hdu(fits, pyfits_compress)

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
        warnings.warn("   Using float64 instead.")
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
        hdu_list.close()
        if isinstance(fin, basestring):
            # In this case, it is a file name that we need to delete.
            import os
            os.remove(fin)
        else:
            fin.close()

    return image

def readMulti(file_name=None, dir=None, hdu_list=None, compression='auto'):
    """Construct a list of Images from a FITS file or pyfits HDUList.

    The normal usage for this function is to read a fits file and return a list of all the images 
    contained therein, automatically decompressing them if necessary.  However, you may also pass 
    it an HDUList, in which case it will build the images from these directly.

    Not all FITS pixel types are supported (only those with C++ Image template instantiations:
    `short`, `int`, `float`, and `double`).  If the FITS header has GS_* keywords, these will be 
    used to initialize the bounding box and scale.  If not, the bounding box will have `(xmin,ymin)`
    at `(1,1)` and the scale will be set to 1.0.

    This function is called as `im = galsim.fits.readMulti(...)`


    @param file_name    The name of the file to read in.  Either `file_name` or `hdu_list` is 
                        required.
    @param dir          Optionally a directory name can be provided if the file_name does not 
                        already include it.
    @param hdu_list     A `pyfits.HDUList` from which to read the images.  Either `file_name` or
                        `hdu_list` is required.
    @param compression  Which decompression scheme to use (if any).  Options are:
                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                                   (requires fits to be a string).  
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
    @returns A Python list of ImageView instances.
    """

    import pyfits     # put this at function scope to keep pyfits optional
     
    file_compress, pyfits_compress = parse_compression(compression,file_name)

    if file_name and hdu_list:
        raise TypeError("Cannot provide both file_name and hdu_list to readMulti()")
    if not (file_name or hdu_list):
        raise TypeError("Must provide either file_name or hdu_list to readMulti()")

    fin = None
    if file_name:
        if dir:
            import os
            file_name = os.path.join(dir,file_name)
        hdu_list, fin = read_file(file_name, file_compress)
    elif not isinstance(hdu_list, pyfits.HDUList):
        raise TypeError("In readMulti, hdu_list is not an HDUList")

    image_list = []
    if pyfits_compress:
        first = 1
        if len(hdu_list) <= 1:
            raise IOError('Expecting at least one extension HDU in galsim.read')
    else:
        first = 0
        if len(hdu_list) < 1:
            raise IOError('Expecting at least one HDU in galsim.readMulti')
    for hdu in range(first,len(hdu_list)):
        image_list.append(read(hdu_list=hdu_list, hdu=hdu, compression=pyfits_compress))

    # If we opened a file, don't forget to close it.
    if fin:
        hdu_list.close()
        if isinstance(fin, basestring):
            # In this case, it is a file name that we need to delete.
            import os
            os.remove(fin)
        else:
            fin.close()

    return image_list

def readCube(file_name=None, dir=None, hdu_list=None, hdu=0, compression='auto'):
    """Construct a Python list of ImageViews from a FITS data cube.

    Not all FITS pixel types are supported (only those with C++ Image template instantiations are:
    `short`, `int`, `float`, and `double`).  If the FITS header has GS_* keywords, these will be  
    used to initialize the bounding boxes and scales.  If not, the bounding boxes will have 
    `(xmin,ymin)` at `(1,1)` and the scale will be set to 1.0.

    This function is called as `image_list = galsim.fits.readCube(...)`

    @param file_name    The name of the file to read in.  Either `file_name` or `hdu_list` is 
                        required.
    @param dir          Optionally a directory name can be provided if the file_name does not 
                        already include it.
    @param hdu_list     Either a `pyfits.HDUList`, a `pyfits.PrimaryHDU`, or `pyfits.ImageHDU`.
                        In the former case, the `hdu` in the list will be selected.  In the latter
                        two cases, the `hdu` parameter is ignored.  Either `file_name` or 
                        `hdu_list` is required.
    @param hdu          The number of the HDU to use.  The default is to use the primary HDU,
                        which is numbered 0.
    @param compression  Which decompression scheme to use (if any).  Options are:
                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                                   (requires fits to be a string).  
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
    @returns A Python list of ImageView instances.
    """
    import pyfits     # put this at function scope to keep pyfits optional
  
    file_compress, pyfits_compress = parse_compression(compression,file_name)

    if file_name and hdu_list:
        raise TypeError("Cannot provide both file_name and hdu_list to read()")
    if not (file_name or hdu_list):
        raise TypeError("Must provide either file_name or hdu_list to read()")

    fin = None
    if file_name:
        if dir:
            import os
            file_name = os.path.join(dir,file_name)
        hdu_list, fin = read_file(file_name, file_compress)

    if isinstance(hdu_list, pyfits.HDUList):
        # Note: Nothing special needs to be done when reading a compressed hdu.
        # However, such compressed hdu's may not be the PrimaryHDU, so if we think we are
        # reading a compressed file, skip to hdu 1.
        if pyfits_compress and hdu==0:
            if len(hdu_list) <= 1:
                raise IOError('Expecting at least one extension HDU in galsim.read')
            hdu = 1
        elif len(hdu_list) <= hdu:
            raise IOError('Expecting at least %d HDUs in galsim.read'%(hdu+1))
        fits = hdu_list[hdu]
    else:
        fits = hdu_list
    check_hdu(fits, pyfits_compress)

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

    nimages = fits.data.shape[0]
    image_list = []
    for k in range(nimages):
        image = Class(array=fits.data[k,:,:], xmin=xmin, ymin=ymin)
        image.scale = scale
        image_list.append(image)

    # If we opened a file, don't forget to close it.
    if fin: 
        hdu_list.close()
        if isinstance(fin, basestring):
            # In this case, it is a file name that we need to delete.
            import os
            os.remove(fin)
        else:
            fin.close()


    return image_list

class FitsHeader(object):
    """A class storing key/value pairs from a FITS Header

    This class works a lot like the regular read() function, but rather than returning
    the image part of the FITS file, it stores the header from which you can access the
    various key values. 

    After construction, you can access a header value by

        value = fits_header[key]

    In fact, all the normal functions available for an immutable dict are available:
    
        keys = fits_header.keys()
        items = fits_header.items()
        for key in fits_header:
            value = fits_header[key]
        value = fits_header.get(key, default)
        etc.

    Constructor parameters:

    @param file_name    The name of the file to read in.  Either `file_name` or `hdu_list` is 
                        required.
    @param dir          Optionally a directory name can be provided if the file_name does not 
                        already include it.
    @param hdu_list     Either a `pyfits.HDUList`, a `pyfits.PrimaryHDU`, or `pyfits.ImageHDU`.
                        In the former case, the `hdu` in the list will be selected.  In the latter
                        two cases, the `hdu` parameter is ignored.  Either `file_name` or 
                        `hdu_list` is required.
    @param hdu          The number of the HDU to use.  The default is to use the primary HDU,
                        which is numbered 0.
    @param compression  Which decompression scheme to use (if any).  Options are:
                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                                   (requires fits to be a string).  
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'hdu' : int , 'compression' : str }
    _single_params = []
    _takes_rng = False
    def __init__(self, file_name=None, dir=None, hdu_list=None, hdu=0, compression='auto'):
        import pyfits     # put this at function scope to keep pyfits optional
    
        file_compress, pyfits_compress = parse_compression(compression,file_name)

        if file_name and hdu_list:
            raise TypeError("Cannot provide both file_name and hdu_list to read()")
        if not (file_name or hdu_list):
            raise TypeError("Must provide either file_name or hdu_list to read()")

        fin = None
        if file_name:
            if dir:
                import os
                file_name = os.path.join(dir,file_name)
            hdu_list, fin = read_file(file_name, file_compress)

        if isinstance(hdu_list, pyfits.HDUList):
            # Note: Nothing special needs to be done when reading a compressed hdu.
            # However, such compressed hdu's may not be the PrimaryHDU, so if we think we are
            # reading a compressed file, skip to hdu 1.
            if pyfits_compress and hdu==0:
                if len(hdu_list) <= 1:
                    raise IOError('Expecting at least one extension HDU in galsim.read')
                hdu = 1
            elif len(hdu_list) <= hdu:
                raise IOError('Expecting at least %d HDUs in galsim.read'%(hdu+1))
            fits = hdu_list[hdu]
        else:
            fits = hdu_list
        check_hdu(fits, pyfits_compress)
        import copy
        self.header = copy.copy(fits.header)

        # If we opened a file, don't forget to close it.
        if fin: 
            hdu_list.close()
            if isinstance(fin, basestring):
                # In this case, it is a file name that we need to delete.
                import os
                os.remove(fin)
            else:
                fin.close()

    # The rest of the functions are typical non-mutating functions for a dict, for which we just
    # pass the request along to self.header.
    def __len__(self):
        return len(self.header)

    def __getitem__(self, key):
        return self.header[key]

    def __contains__(self, key):
        return key in self.header

    def __iter__(self):
        return self.header.__iter__

    def get(self, key, default=None):
        return self.header.get(key, default)

    def keys(self):
        return self.header.keys()

    def values(self):
        return self.header.values()

    def items(self):
        return self.header.iteritems()

    def iterkeys(self):
        return self.header.iterkeys()

    def itervalues(self):
        return self.header.itervalues()

    def iteritems(self):
        return self.header.iteritems()


# inject write as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.write = write

for Class in _galsim.ImageView.itervalues():
    Class.write = write

for Class in _galsim.ConstImageView.itervalues():
    Class.write = write

del Class    # cleanup public namespace
