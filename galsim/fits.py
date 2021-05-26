# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

import os
import numpy as np

from .image import Image
from .errors import GalSimError, GalSimValueError, GalSimIncompatibleValuesError, galsim_warn
from .utilities import basestring


##############################################################################################
#
# We start off with some helper functions for some common operations that will be used in
# more than one of our primary read and write functions.
#
##############################################################################################

def _parse_compression(compression, file_name):
    file_compress = None
    pyfits_compress = None
    if compression == 'rice' or compression == 'RICE_1': pyfits_compress = 'RICE_1'
    elif compression == 'gzip_tile' or compression == 'GZIP_1': pyfits_compress = 'GZIP_1'
    elif compression == 'hcompress' or compression == 'HCOMPRESS_1': pyfits_compress = 'HCOMPRESS_1'
    elif compression == 'plio' or compression == 'PLIO_1': pyfits_compress = 'PLIO_1'
    elif compression == 'gzip': file_compress = 'gzip'
    elif compression == 'bzip2': file_compress = 'bzip2'
    elif compression == 'none' or compression is None: pass
    elif compression == 'auto':
        if file_name:
            if file_name.lower().endswith('.fz'): pyfits_compress = 'RICE_1'
            elif file_name.lower().endswith('.gz'): file_compress = 'gzip'
            elif file_name.lower().endswith('.bz2'): file_compress = 'bzip2'
            else:  # pragma: no cover  (Not sure why Travis thinks this isn't covered.)
                pass
    else:
        raise GalSimValueError("Invalid compression", compression,
                               ('rice', 'gzip_tile', 'hcompress', 'plio', 'gzip', 'bzip2',
                                'none', 'auto'))
    return file_compress, pyfits_compress

# This is a class rather than a def, since we want to store some variable, and that's easier
# to do with a class than a function.  But this will be used as though it were a normal
# function: _read_file(file, file_compress)
class _ReadFile:

    # There are several methods available for each of gzip and bzip2.  Each is its own function.
    def gunzip_call(self, file):
        # cf. http://bugs.python.org/issue7471
        import subprocess
        from io import BytesIO
        from ._pyfits import pyfits
        # We use gunzip -c rather than zcat, since the latter is sometimes called gzcat
        # (with zcat being a symlink to uncompress instead).
        # Also, I'd rather all these use `with subprocess.Popen(...) as p:`, but that's not
        # supported in 2.7.  So need to keep things this way for now.
        try:
            p = subprocess.Popen(["gunzip", "-c", file], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, close_fds=True)
        except OSError:
            # This OSError should mean that the gunzip call itself was invalid on this system.
            # Convert to a NotImplementedError, so we can try a different method.
            raise NotImplementedError()
        ret = p.communicate()
        if ret[0] == b'':  # pragma: no cover
            raise OSError("Error running gunzip. stderr output = %s"%ret[1])
        if p.returncode != 0:  # pragma: no cover
            raise OSError("Error running gunzip. Return code = %s"%p.returncode)
        fin = BytesIO(ret[0])
        p.wait()
        try:
            hdu_list = pyfits.open(fin, 'readonly')
        except (OSError, AttributeError, TypeError, ValueError): # pragma: no cover
            # In case astropy fails.
            raise NotImplementedError()
        return hdu_list, fin

    # Note: the above gzip_call function succeeds on travis, so the rest don't get run.
    # Omit them from the coverage test.
    def gzip_in_mem(self, file): # pragma: no cover
        import gzip
        from ._pyfits import pyfits
        fin = gzip.open(file, 'rb')
        hdu_list = pyfits.open(fin, 'readonly')
        # pyfits doesn't actually read the file yet, so we can't close fin here.
        # Need to pass it back to the caller and let them close it when they are
        # done with hdu_list.
        return hdu_list, fin

    def bunzip2_call(self, file):
        import subprocess
        from io import BytesIO
        from ._pyfits import pyfits
        try:
            p = subprocess.Popen(["bunzip2", "-c", file], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, close_fds=True)
        except OSError:
            # This OSError should mean that the gunzip call itself was invalid on this system.
            # Convert to a NotImplementedError, so we can try a different method.
            raise NotImplementedError()
        ret = p.communicate()
        if ret[0] == b'':  # pragma: no cover
            raise OSError("Error running bunzip2. stderr output = %s"%ret[1])
        if p.returncode != 0:  # pragma: no cover
            raise OSError("Error running bunzip2. Return code = %s"%p.returncode)
        fin = BytesIO(ret[0])
        p.wait()
        try:
            hdu_list = pyfits.open(fin, 'readonly')
        except (OSError, AttributeError, TypeError, ValueError): # pragma: no cover
            # In case astropy fails.
            raise NotImplementedError()
        return hdu_list, fin

    def bz2_in_mem(self, file): # pragma: no cover
        import bz2
        from ._pyfits import pyfits
        fin = bz2.BZ2File(file, 'rb')
        hdu_list = pyfits.open(fin, 'readonly')
        return hdu_list, fin

    def __init__(self):
        # We used to have multiple options for gzip and bzip2.  However, with recent versions of
        # astropy for the fits I/O, the in memory version should always work.  So we first
        # try the command line method, which is usually faster.  Then if that fails, we let
        # astropy do the compression.
        self.gz_index = 0
        self.bz2_index = 0
        self.gz_methods = [self.gunzip_call, self.gzip_in_mem]
        self.bz2_methods = [self.bunzip2_call, self.bz2_in_mem]
        self.gz = self.gz_methods[0]
        self.bz2 = self.bz2_methods[0]

    def __call__(self, file, dir, file_compress):
        from ._pyfits import pyfits
        if dir:
            file = os.path.join(dir,file)

        if not os.path.isfile(file):
            raise OSError("File %s not found"%file)

        if not file_compress:
            hdu_list = pyfits.open(file, 'readonly')
            return hdu_list, None
        elif file_compress == 'gzip':
            # Before trying all the gzip options, first make sure the file exists and is readable.
            # The easiest way to do this is to try to open it.  Just let the open command return
            # its normal error message if the file doesn't exist or cannot be opened.
            with open(file) as fid: pass
            while self.gz_index < len(self.gz_methods):
                try:
                    return self.gz(file)
                except (ImportError, NotImplementedError): # pragma: no cover
                    if self.gz_index == len(self.gz_methods)-1:
                        raise
                    else:
                        self.gz_index += 1
                        self.gz = self.gz_methods[self.gz_index]
            else:  # pragma: no cover
                raise GalSimError("None of the options for gunzipping were successful.")
        elif file_compress == 'bzip2':
            with open(file) as fid: pass
            while self.bz2_index < len(self.bz2_methods):
                try:
                    return self.bz2(file)
                except (ImportError, NotImplementedError): # pragma: no cover
                    if self.bz2_index == len(self.bz2_methods)-1:
                        raise
                    else:
                        self.bz2_index += 1
                        self.bz2 = self.bz2_methods[self.bz2_index]
            else:  # pragma: no cover
                raise GalSimError("None of the options for bunzipping were successful.")
        else:  # pragma: no cover  (can't get here from public API)
            raise GalSimValueError("Unknown file_compression", file_compress, ('gzip', 'bzip2'))
_read_file = _ReadFile()

# Do the same trick for _write_file(file,hdu_list,clobber,file_compress,pyfits_compress):
class _WriteFile:
    # kwargs to use for the writeto function
    kw = {'output_verify' : 'silentfix+ignore'}

    # There are several methods available for each of gzip and bzip2.  Each is its own function.
    def gzip_call(self, hdu_list, file):
        import subprocess
        with open(file, 'wb') as fout:
            try:
                p = subprocess.Popen(["gzip", "-"], stdin=subprocess.PIPE, stdout=fout,
                                     close_fds=True)
                hdu_list.writeto(p.stdin, **self.kw)
            except (OSError, AttributeError, TypeError, ValueError): # pragma: no cover
                # This OSError should mean that the gunzip call itself was invalid on this system.
                # Convert to a NotImplementedError, so we can try a different method.
                # The others are in case astropy fails.
                raise NotImplementedError()
            p.communicate()
            if p.returncode != 0:  # pragma: no cover
                raise OSError("Error running gzip. Return code = %s"%p.returncode)
            p.wait()

    def gzip_in_mem(self, hdu_list, file):  # pragma: no cover
        import gzip
        import io
        # The compression routines work better if we first write to an internal buffer
        # and then output that to a file.
        buf = io.BytesIO()
        hdu_list.writeto(buf, **self.kw)
        data = buf.getvalue()
        # There is a compresslevel option (for both gzip and bz2), but we just use the
        # default.
        with gzip.open(file, 'wb') as fout:
            fout.write(data)

    def bzip2_call(self, hdu_list, file):
        import subprocess
        with open(file, 'wb') as fout:
            try:
                p = subprocess.Popen(["bzip2"], stdin=subprocess.PIPE, stdout=fout, close_fds=True)
                hdu_list.writeto(p.stdin, **self.kw)
            except (OSError, AttributeError, TypeError, ValueError): # pragma: no cover
                # This OSError should mean that the gunzip call itself was invalid on this system.
                # Convert to a NotImplementedError, so we can try a different method.
                # The others are in case astropy fails.
                raise NotImplementedError()
            p.communicate()
            if p.returncode != 0:  # pragma: no cover
                raise OSError("Error running bzip2. Return code = %s"%p.returncode)
            p.wait()

    def bz2_in_mem(self, hdu_list, file):  # pragma: no cover
        import bz2
        import io
        buf = io.BytesIO()
        hdu_list.writeto(buf, **self.kw)
        data = buf.getvalue()
        with bz2.BZ2File(file, 'wb') as fout:
            fout.write(data)

    def __init__(self):
        # Again, we used to have a number of methods here for gzip and bzip2, but now only two.
        # We first try using a command-line call to either gzip or bzip2.  But if that doesn't
        # work, we use either the gzip or bz2 module in memory, which is usually not quite as
        # fast, but should always work.
        self.gz_index = 0
        self.bz2_index = 0
        self.gz_methods = [self.gzip_call, self.gzip_in_mem]
        self.bz2_methods = [self.bzip2_call, self.bz2_in_mem]
        self.gz = self.gz_methods[0]
        self.bz2 = self.bz2_methods[0]

    def __call__(self, file, dir, hdu_list, clobber, file_compress, pyfits_compress):
        from .utilities import ensure_dir
        if dir:
            file = os.path.join(dir,file)

        ensure_dir(file)
        if os.path.isfile(file):
            if clobber:
                os.remove(file)
            else:
                raise OSError('File %r already exists'%file)

        if not file_compress:
            hdu_list.writeto(file, **self.kw)
        elif file_compress == 'gzip':
            while self.gz_index < len(self.gz_methods):
                try:
                    return self.gz(hdu_list, file)
                except (ImportError, NotImplementedError):  # pragma: no cover
                    if self.gz_index == len(self.gz_methods)-1:
                        raise
                    else:
                        self.gz_index += 1
                        self.gz = self.gz_methods[self.gz_index]
            else:  # pragma: no cover
                raise GalSimError("None of the options for gzipping were successful.")
        elif file_compress == 'bzip2':
            while self.bz2_index < len(self.bz2_methods):
                try:
                    return self.bz2(hdu_list, file)
                except (ImportError, NotImplementedError):  # pragma: no cover
                    if self.bz2_index == len(self.bz2_methods)-1:
                        raise
                    else:
                        self.bz2_index += 1
                        self.bz2 = self.bz2_methods[self.bz2_index]
            else:  # pragma: no cover
                raise GalSimError("None of the options for bzipping were successful.")
        else:  # pragma: no cover  (can't get here from public API)
            raise GalSimValueError("Unknown file_compression", file_compress, ('gzip', 'bzip2'))

_write_file = _WriteFile()

def _add_hdu(hdu_list, data, pyfits_compress):
    from ._pyfits import pyfits
    if pyfits_compress:
        if len(hdu_list) == 0:
            hdu_list.append(pyfits.PrimaryHDU())  # Need a blank PrimaryHDU
        hdu = pyfits.CompImageHDU(data, compression_type=pyfits_compress)
    else:
        if len(hdu_list) == 0:
            hdu = pyfits.PrimaryHDU(data)
        else:
            hdu = pyfits.ImageHDU(data)
    hdu_list.append(hdu)
    return hdu

def _add_to_header(hdu, image):
    h = FitsHeader(hdu.header)
    if hasattr(image, 'header'):
        h.extend(image.header)
    if image.wcs:
        image.wcs.writeToFitsHeader(h, image.bounds)


def _check_hdu(hdu, pyfits_compress, header_only=False):
    """Check that an input ``hdu`` is valid
    """
    from ._pyfits import pyfits
    # Check for fixable verify errors
    try:
        hdu.header
        if not header_only:
            hdu.data
    except pyfits.VerifyError:
        hdu.verify('fix')

    # Check that the specified compression is right for the given hdu type.
    if pyfits_compress:
        if not isinstance(hdu, pyfits.CompImageHDU):
            raise OSError('Found invalid HDU type reading FITS file (expected a CompImageHDU)')
    else:
        if not isinstance(hdu, (pyfits.CompImageHDU, pyfits.ImageHDU, pyfits.PrimaryHDU)):
            raise OSError('Found invalid HDU type reading FITS file (expected an ImageHDU)')


def _get_hdu(hdu_list, hdu, pyfits_compress, header_only=False):
    from ._pyfits import pyfits
    if isinstance(hdu_list, pyfits.HDUList):
        # Note: Nothing special needs to be done when reading a compressed hdu.
        # However, such compressed hdu's may not be the PrimaryHDU, so if we think we are
        # reading a compressed file, skip to hdu 1.
        if hdu is None:
            if pyfits_compress:
                if len(hdu_list) <= 1:
                    raise OSError('Expecting at least one extension HDU in galsim.read')
                hdu = 1
            else:
                hdu = 0
        if len(hdu_list) <= hdu:
            raise OSError('Expecting at least %d HDUs in galsim.read'%(hdu+1))
        hdu = hdu_list[hdu]
    elif isinstance(hdu_list, (pyfits.ImageHDU, pyfits.PrimaryHDU, pyfits.CompImageHDU)):
        hdu = hdu_list
    else:
        raise TypeError("Invalid hdu_list: %s",hdu_list)
    _check_hdu(hdu, pyfits_compress, header_only)
    return hdu


# Unlike the other helpers, this one doesn't start with an underscore, since we make it
# available to people who use the function ReadFile.
def closeHDUList(hdu_list, fin):
    """If necessary, close the file handle that was opened to read in the ``hdu_list``"""
    hdu_list.close()
    if fin:
        fin.close()

##############################################################################################
#
# Now the primary write functions.  We have:
#    write(image, ...)
#    writeMulti(image_list, ...)
#    writeCube(image_list, ...)
#    writeFile(hdu_list, ...)
#
##############################################################################################


def write(image, file_name=None, dir=None, hdu_list=None, clobber=True, compression='auto'):
    """Write a single image to a FITS file.

    Write the `Image` instance ``image`` to a FITS file, with details depending on the arguments.
    This function can be called directly as ``galsim.fits.write(image, ...)``, with the image as the
    first argument, or as an `Image` method: ``image.write(...)``.

    Parameters:
        image:          The `Image` to write to file.  Per the description of this method, it may be
                        given explicitly via ``galsim.fits.write(image, ...)`` or the method may be
                        called directly as an image method, ``image.write(...)``.  Note that if the
                        image has a 'header' attribute containing a `FitsHeader`, then the
                        `FitsHeader` is written to the header in the PrimaryHDU, followed by the
                        WCS as usual.
        file_name:      The name of the file to write to.  [Either ``file_name`` or ``hdu_list`` is
                        required.]
        dir:            Optionally a directory name can be provided if ``file_name`` does not
                        already include it. [default: None]
        hdu_list:       An astropy.io.fits.HDUList.  If this is provided instead of ``file_name``,
                        then the `Image` is appended to the end of the HDUList as a new HDU. In
                        that case, the user is responsible for calling either
                        ``hdu_list.writeto(...)`` or ``galsim.fits.writeFile(...)`` afterwards.
                        [Either ``file_name`` or ``hdu_list`` is required.]
        clobber:        Setting ``clobber=True`` will silently overwrite existing files.
                        [default: True]
        compression:    Which compression scheme to use (if any).  Options are:

                        - None or 'none' = no compression
                        - 'rice' = use rice compression in tiles (preserves header readability)
                        - 'gzip' = use gzip to compress the full file
                        - 'bzip2' = use bzip2 to compress the full file
                        - 'gzip_tile' = use gzip in tiles (preserves header readability)
                        - 'hcompress' = use hcompress in tiles (only valid for 2-d images)
                        - 'plio' = use plio compression in tiles (only valid for pos integer data)
                        - 'auto' = determine the compression from the extension of the file name
                          (requires ``file_name`` to be given):

                          - '.fz' => 'rice'
                          - '.gz' => 'gzip'
                          - '.bz2' => 'bzip2'
                          - otherwise None

                        [default: 'auto']
    """
    from ._pyfits import pyfits

    if image.iscomplex:
        raise GalSimValueError("Cannot write complex Images to a fits file. "
                               "Write image.real and image.imag separately.", image)

    file_compress, pyfits_compress = _parse_compression(compression,file_name)

    if file_name and hdu_list is not None:
        raise GalSimIncompatibleValuesError(
            "Cannot provide both file_name and hdu_list", file_name=file_name, hdu_list=hdu_list)
    if not (file_name or hdu_list is not None):
        raise GalSimIncompatibleValuesError(
            "Must provide either file_name or hdu_list", file_name=file_name, hdu_list=hdu_list)

    if hdu_list is None:
        hdu_list = pyfits.HDUList()

    hdu = _add_hdu(hdu_list, image.array, pyfits_compress)
    _add_to_header(hdu, image)

    if file_name:
        _write_file(file_name, dir, hdu_list, clobber, file_compress, pyfits_compress)


def writeMulti(image_list, file_name=None, dir=None, hdu_list=None, clobber=True,
               compression='auto'):
    """Write a Python list of images to a multi-extension FITS file.

    The details of how the images are written to file depends on the arguments.

    .. note::

        This function along with `readMulti` can be used to effect the equivalent of a simple
        version of fpack or funpack. To Rice compress a fits file, you can call::

            fname = 'some_image_file.fits'
            galsim.fits.writeMulti(galsim.fits.readMulti(fname, read_headers=True), fname+'.fz')

        To uncompress::

            fname = 'some_image_file.fits.fz'
            galsim.fits.writeMulti(galsim.fits.readMulti(fname, read_headers=True), fname[:-3])

    Parameters:
        image_list:     A Python list of `Image` instances.  (For convenience, some items in this
                        list may be HDUs already.  Any `Image` will be converted into an
                        astropy.io.fits.HDU.)
        file_name:      The name of the file to write to.  [Either ``file_name`` or ``hdu_list`` is
                        required.]
        dir:            Optionally a directory name can be provided if ``file_name`` does not
                        already include it. [default: None]
        hdu_list:       An astropy.io.fits.HDUList.  If this is provided instead of ``file_name``,
                        then the `Image` is appended to the end of the HDUList as a new HDU. In
                        that case, the user is responsible for calling either
                        ``hdu_list.writeto(...)`` or ``galsim.fits.writeFile(...)`` afterwards.
                        [Either ``file_name`` or ``hdu_list`` is required.]
        clobber:        Setting ``clobber=True`` will silently overwrite existing files.
                        [default: True]
        compression:    Which compression scheme to use (if any).  Options are:

                        - None or 'none' = no compression
                        - 'rice' = use rice compression in tiles (preserves header readability)
                        - 'gzip' = use gzip to compress the full file
                        - 'bzip2' = use bzip2 to compress the full file
                        - 'gzip_tile' = use gzip in tiles (preserves header readability)
                        - 'hcompress' = use hcompress in tiles (only valid for 2-d images)
                        - 'plio' = use plio compression in tiles (only valid for pos integer data)
                        - 'auto' = determine the compression from the extension of the file name
                          (requires ``file_name`` to be given):

                          - '.fz' => 'rice'
                          - '.gz' => 'gzip'
                          - '.bz2' => 'bzip2'
                          - otherwise None

                        [default: 'auto']
    """
    from ._pyfits import pyfits

    if any(image.iscomplex for image in image_list if isinstance(image, Image)):
        raise GalSimValueError("Cannot write complex Images to a fits file. "
                               "Write image.real and image.imag separately.", image_list)

    file_compress, pyfits_compress = _parse_compression(compression,file_name)

    if file_name and hdu_list is not None:
        raise GalSimIncompatibleValuesError(
            "Cannot provide both file_name and hdu_list", file_name=file_name, hdu_list=hdu_list)
    if not (file_name or hdu_list is not None):
        raise GalSimIncompatibleValuesError(
            "Must provide either file_name or hdu_list", file_name=file_name, hdu_list=hdu_list)

    if hdu_list is None:
        hdu_list = pyfits.HDUList()

    for image in image_list:
        if isinstance(image, Image):
            hdu = _add_hdu(hdu_list, image.array, pyfits_compress)
            _add_to_header(hdu, image)
        else:
            # Assume that image is really an HDU.  If not, this should give a reasonable error
            # message.  (The base type of HDUs vary among versions of pyfits, so it's hard to
            # check explicitly with an isinstance call.  For newer pyfits versions, it is
            # pyfits.hdu.base.ExtensionHDU, but not in older versions.)
            hdu_list.append(image)

    if file_name:
        _write_file(file_name, dir, hdu_list, clobber, file_compress, pyfits_compress)


def writeCube(image_list, file_name=None, dir=None, hdu_list=None, clobber=True,
              compression='auto'):
    """Write a Python list of images to a FITS file as a data cube.

    The details of how the images are written to file depends on the arguments.  Unlike for
    writeMulti, when writing a data cube it is necessary that each `Image` in ``image_list`` has
    the same size ``(nx, ny)``.  No check is made to confirm that all images have the same origin
    and pixel scale (or WCS).

    In fact, the WCS of the first image is the one that gets put into the FITS header (since only
    one WCS can be put into a FITS header).  Thus, if the images have different WCS functions,
    only the first one will be rendered correctly by plotting programs such as ds9.  The FITS
    standard does not support any way to have the various images in a data cube to have different
    WCS solutions.

    Parameters:
        image_list:     The ``image_list`` can also be either an array of NumPy arrays or a 3d NumPy
                        array, in which case this is written to the fits file directly.  In the
                        former case, no explicit check is made that the NumPy arrays are all the
                        same shape, but a NumPy exception will be raised which we let pass upstream
                        unmolested.
        file_name:      The name of the file to write to.  [Either ``file_name`` or ``hdu_list`` is
                        required.]
        dir:            Optionally a directory name can be provided if ``file_name`` does not
                        already include it. [default: None]
        hdu_list:       An astropy.io.fits.HDUList.  If this is provided instead of ``file_name``,
                        then the cube is appended to the end of the HDUList as a new HDU. In that
                        case, the user is responsible for calling either ``hdu_list.writeto(...)``
                        or ``galsim.fits.writeFile(...)`` afterwards.  [Either ``file_name`` or
                        ``hdu_list`` is required.]
        clobber:        Setting ``clobber=True`` will silently overwrite existing files.
                        [default: True]
        compression:    Which compression scheme to use (if any).  Options are:

                        - None or 'none' = no compression
                        - 'rice' = use rice compression in tiles (preserves header readability)
                        - 'gzip' = use gzip to compress the full file
                        - 'bzip2' = use bzip2 to compress the full file
                        - 'gzip_tile' = use gzip in tiles (preserves header readability)
                        - 'hcompress' = use hcompress in tiles (only valid for 2-d images)
                        - 'plio' = use plio compression in tiles (only valid for pos integer data)
                        - 'auto' = determine the compression from the extension of the file name
                          (requires ``file_name`` to be given):

                          - '.fz' => 'rice'
                          - '.gz' => 'gzip'
                          - '.bz2' => 'bzip2'
                          - otherwise None

                        [default: 'auto']
    """
    from ._pyfits import pyfits
    from .bounds import BoundsI

    if isinstance(image_list, np.ndarray):
        is_all_numpy = True
        if image_list.dtype.kind == 'c':
            raise GalSimValueError("Cannot write complex numpy arrays to a fits file. "
                                   "Write array.real and array.imag separately.", image_list)
    elif len(image_list) == 0:
        raise GalSimValueError("In writeCube: image_list has no images", image_list)
    elif all(isinstance(item, np.ndarray) for item in image_list):
        is_all_numpy = True
        if any(a.dtype.kind == 'c' for a in image_list):
            raise GalSimValueError("Cannot write complex numpy arrays to a fits file. "
                                   "Write array.real and array.imag separately.", image_list)
    else:
        is_all_numpy = False
        if any(im.iscomplex for im in image_list):
            raise GalSimValueError("Cannot write complex images to a fits file. "
                                   "Write image.real and image.imag separately.", image_list)

    file_compress, pyfits_compress = _parse_compression(compression,file_name)

    if file_name and hdu_list is not None:
        raise GalSimIncompatibleValuesError(
            "Cannot provide both file_name and hdu_list", file_name=file_name, hdu_list=hdu_list)
    if not (file_name or hdu_list is not None):
        raise GalSimIncompatibleValuesError(
            "Must provide either file_name or hdu_list", file_name=file_name, hdu_list=hdu_list)

    if hdu_list is None:
        hdu_list = pyfits.HDUList()

    if is_all_numpy:
        cube = np.asarray(image_list)
        nimages = cube.shape[0]
        nx = cube.shape[1]
        ny = cube.shape[2]
        # Use default values for bounds
        bounds = BoundsI(1,nx,1,ny)
        wcs = None
    else:
        nimages = len(image_list)
        im = image_list[0]
        dtype = im.array.dtype
        nx = im.xmax - im.xmin + 1
        ny = im.ymax - im.ymin + 1
        # Use the first image's wcs and bounds
        wcs = im.wcs
        bounds = im.bounds
        # Note: numpy shape is y,x
        array_shape = (nimages, ny, nx)
        cube = np.zeros(array_shape, dtype=dtype)
        for k in range(nimages):
            im = image_list[k]
            nx_k = im.xmax-im.xmin+1
            ny_k = im.ymax-im.ymin+1
            if nx_k != nx or ny_k != ny:
                raise GalSimValueError("In writeCube: image %d has the wrong shape. "
                                       "Shape is (%d,%d) should be (%d,%d)"%(k,nx_k,ny_k,nx,ny),
                                       im)
            cube[k,:,:] = image_list[k].array


    hdu = _add_hdu(hdu_list, cube, pyfits_compress)
    if not is_all_numpy:
        # Use any header/wcs in the first image
        _add_to_header(hdu, image_list[0])

    if file_name:
        _write_file(file_name, dir, hdu_list, clobber, file_compress, pyfits_compress)


def writeFile(file_name, hdu_list, dir=None, clobber=True, compression='auto'):
    """Write a Pyfits hdu_list to a FITS file, taking care of the GalSim compression options.

    If you have used the write(), writeMulti() or writeCube() functions with the ``hdu_list``
    option rather than writing directly to a file, you may subsequently use the command
    ``hdu_list.writeto(...)``.  However, it may be more convenient to use this function, writeFile()
    instead, since it treats the compression option consistently with how that option is handled in
    the above functions.

    Parameters:
        file_name:      The name of the file to write to.
        hdu_list:       An astropy.io.fits.HDUList.
        dir:            Optionally a directory name can be provided if ``file_name`` does not
                        already include it. [default: None]
        clobber:        Setting ``clobber=True`` will silently overwrite existing files.
                        [default: True]
        compression:    Which compression scheme to use (if any).  Options are:

                        - None or 'none' = no compression
                        - 'gzip' = use gzip to compress the full file
                        - 'bzip2' = use bzip2 to compress the full file
                        - 'auto' = determine the compression from the extension of the file name
                          (requires ``file_name`` to be given):

                          - '.gz' => 'gzip'
                          - '.bz2' => 'bzip2'
                          - otherwise None

                        Note that the other options, such as 'rice', that operate on the image
                        directly are not available at this point.  If you want to use one of them,
                        it must be applied when writing each hdu.
                        [default: 'auto']
    """
    file_compress, pyfits_compress = _parse_compression(compression,file_name)
    if pyfits_compress and compression != 'auto':
        # If compression is auto and it determined that it should use rice, then we
        # should presume that the hdus were already rice compressed, so we can ignore it here.
        # Otherwise, any pyfits_compression options are invalid.
        raise GalSimValueError("Compression %s is invalid for writeFile",compression)
    _write_file(file_name, dir, hdu_list, clobber, file_compress, pyfits_compress)


##############################################################################################
#
# Now the primary read functions.  We have:
#    image = read(...)
#    image_list = readMulti(...)
#    image_list = readCube(...)
#    hdu, hdu_list, fin = readFile(...)
#
##############################################################################################


def read(file_name=None, dir=None, hdu_list=None, hdu=None, compression='auto',
         read_header=False, suppress_warning=True):
    """Construct an `Image` from a FITS file or HDUList.

    The normal usage for this function is to read a fits file and return the image contained
    therein, automatically decompressing it if necessary.  However, you may also pass it
    an HDUList, in which case it will select the indicated hdu (with the ``hdu`` parameter)
    from that.

    Not all FITS pixel types are supported -- only ``short``, ``int``, ``unsigned short``,
    ``unsigned int``, ``float``, and ``double``.

    If the FITS header has keywords that start with ``GS_``, these will be used to initialize the
    bounding box and WCS.  If these are absent, the code will try to read whatever WCS is given
    in the FITS header.  cf. `galsim.wcs.readFromFitsHeader`.  The default bounding box will have
    ``(xmin,ymin)`` at ``(1,1)``.  The default WCS, if there is no WCS information in the FITS
    file, will be PixelScale(1.0).

    This function is called as ``im = galsim.fits.read(...)``

    Parameters:
        file_name:      The name of the file to read in.  [Either ``file_name`` or ``hdu_list`` is
                        required.]
        dir:            Optionally a directory name can be provided if ``file_name`` does not
                        already include it. [default: None]
        hdu_list:       Either an astropy.io.fits.HDUList, astropy.io.fits.PrimaryHDU, or
                        astropy.io.fits..ImageHDU.  In the former case, the ``hdu`` in the list
                        will be selected.  In the latter two cases, the ``hdu`` parameter is
                        ignored.  [Either ``file_name`` or ``hdu_list`` is required.]
        hdu:            The number of the HDU to return.  [default: None, which means to return
                        either the primary or first extension as appropriate for the given
                        compression.  (e.g. for 'rice', the first extension is the one you normally
                        want.)]
        compression:    Which decompression scheme to use (if any).  Options are:

                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                          (requires ``file_name`` to be given).

                          - '.fz' => 'rice'
                          - '.gz' => 'gzip'
                          - '.bz2' => 'bzip2'
                          - otherwise None

                        [default: 'auto']
        read_header:    Whether to read the header and store it as image.header.[default: False]
        suppress_warning: Whether to suppress a warning that the WCS could not be read from the
                          FITS header, so the WCS defaulted to either a `PixelScale` or
                          `AffineTransform`. [default: True]

    Returns:
        the image as an `Image` instance.
    """
    from . import wcs
    file_compress, pyfits_compress = _parse_compression(compression,file_name)

    if file_name and hdu_list is not None:
        raise GalSimIncompatibleValuesError(
            "Cannot provide both file_name and hdu_list", file_name=file_name, hdu_list=hdu_list)
    if not (file_name or hdu_list is not None):
        raise GalSimIncompatibleValuesError(
            "Must provide either file_name or hdu_list", file_name=file_name, hdu_list=hdu_list)

    if file_name:
        hdu_list, fin = _read_file(file_name, dir, file_compress)

    try:
        hdu = _get_hdu(hdu_list, hdu, pyfits_compress)

        if hdu.data is None:
            raise OSError("HDU is empty.  (data is None)")

        wcs, origin = wcs.readFromFitsHeader(hdu.header, suppress_warning)
        dt = hdu.data.dtype.type
        if dt in Image.valid_dtypes:
            data = hdu.data
        else:
            galsim_warn("No C++ Image template instantiation for data type %s. "
                        "Using numpy.float64 instead."%(dt))
            data = hdu.data.astype(np.float64)

        image = Image(array=data)
        image.setOrigin(origin)
        image.wcs = wcs

        if read_header:
            image.header = FitsHeader(hdu.header)

    finally:
        # If we opened a file, don't forget to close it.
        if file_name:
            closeHDUList(hdu_list, fin)

    return image

def readMulti(file_name=None, dir=None, hdu_list=None, compression='auto',
              read_headers=False, suppress_warning=True):
    """Construct a list of `Image` instances from a FITS file or HDUList.

    The normal usage for this function is to read a fits file and return a list of all the images
    contained therein, automatically decompressing them if necessary.  However, you may also pass
    it an HDUList, in which case it will build the images from these directly.

    Not all FITS pixel types are supported -- only ``short``, ``int``, ``unsigned short``,
    ``unsigned int``, ``float``, and ``double``.

    If the FITS header has keywords that start with ``GS_``, these will be used to initialize the
    bounding box and WCS.  If these are absent, the code will try to read whatever WCS is given
    in the FITS header.  cf. `galsim.wcs.readFromFitsHeader`.  The default bounding box will have
    ``(xmin,ymin)`` at ``(1,1)``.  The default WCS, if there is no WCS information in the FITS
    file, will be PixelScale(1.0).

    This function is called as ``im = galsim.fits.readMulti(...)``

    .. note::

        This function along with `writeMulti` can be used to effect the equivalent of a simple
        version of fpack or funpack. To Rice compress a fits file, you can call::

            fname = 'some_image_file.fits'
            galsim.fits.writeMulti(galsim.fits.readMulti(fname, read_headers=True), fname+'.fz')

        To uncompress::

            fname = 'some_image_file.fits.fz'
            galsim.fits.writeMulti(galsim.fits.readMulti(fname, read_headers=True), fname[:-3])

    Parameters:
        file_name:      The name of the file to read in.  [Either ``file_name`` or ``hdu_list`` is
                        required.]
        dir:            Optionally a directory name can be provided if ``file_name`` does not
                        already include it. [default: None]
        hdu_list:       An astropy.io.fits.HDUList from which to read the images.  [Either
                        ``file_name`` or ``hdu_list`` is required.]
        compression:    Which decompression scheme to use (if any).  Options are:

                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                          (requires ``file_name`` to be given).

                          - '.fz' => 'rice'
                          - '.gz' => 'gzip'
                          - '.bz2' => 'bzip2'
                          - otherwise None

                        [default: 'auto']
        read_headers:   Whether to read the headers and store them as image.header.[default: False]
        suppress_warning: Whether to suppress a warning that the WCS could not be read from the
                          FITS header, so the WCS defaulted to either a `PixelScale` or
                          `AffineTransform`. [default: True]

    Returns:
        a Python list of `Image` instances.
    """
    from ._pyfits import pyfits

    file_compress, pyfits_compress = _parse_compression(compression,file_name)

    if file_name and hdu_list is not None:
        raise GalSimIncompatibleValuesError(
            "Cannot provide both file_name and hdu_list", file_name=file_name, hdu_list=hdu_list)
    if not (file_name or hdu_list is not None):
        raise GalSimIncompatibleValuesError(
            "Must provide either file_name or hdu_list", file_name=file_name, hdu_list=hdu_list)

    if file_name:
        hdu_list, fin = _read_file(file_name, dir, file_compress)
    elif not isinstance(hdu_list, pyfits.HDUList):
        raise TypeError("In readMulti, hdu_list is not an HDUList")

    try:
        image_list = []
        if pyfits_compress:
            first = 1
            if len(hdu_list) <= 1:
                raise OSError('Expecting at least one extension HDU in galsim.read')
        else:
            first = 0
            if len(hdu_list) < 1:
                raise OSError('Expecting at least one HDU in galsim.readMulti')
        for hdu in range(first,len(hdu_list)):
            image_list.append(read(hdu_list=hdu_list, hdu=hdu, compression=pyfits_compress,
                                   read_header=read_headers, suppress_warning=suppress_warning))

    finally:
        # If we opened a file, don't forget to close it.
        if file_name:
            closeHDUList(hdu_list, fin)

    return image_list

def readCube(file_name=None, dir=None, hdu_list=None, hdu=None, compression='auto',
             suppress_warning=True):
    """Construct a Python list of `Image` instances from a FITS data cube.

    Not all FITS pixel types are supported -- only ``short``, ``int``, ``unsigned short``,
    ``unsigned int``, ``float``, and ``double``.

    If the FITS header has keywords that start with ``GS_``, these will be used to initialize the
    bounding box and WCS.  If these are absent, the code will try to read whatever WCS is given
    in the FITS header.  cf. `galsim.wcs.readFromFitsHeader`.  The default bounding box will have
    ``(xmin,ymin)`` at ``(1,1)``.  The default WCS, if there is no WCS information in the FITS
    file, will be PixelScale(1.0).

    This function is called as ``image_list = galsim.fits.readCube(...)``

    Parameters:
        file_name:      The name of the file to read in.  [Either ``file_name`` or ``hdu_list`` is
                        required.]
        dir:            Optionally a directory name can be provided if ``file_name`` does not
                        already include it. [default: None]
        hdu_list:       Either an astropy.io.fits.HDUList, an astropy.io.fits.PrimaryHDU, or
                        astropy.io.fits.ImageHDU.  In the former case, the ``hdu`` in the list will
                        be selected.  In the latter two cases, the ``hdu`` parameter is ignored.
                        [Either ``file_name`` or ``hdu_list`` is required.]
        hdu:            The number of the HDU to return.  [default: None, which means to return
                        either the primary or first extension as appropriate for the given
                        compression.  (e.g. for rice, the first extension is the one you normally
                        want.)]
        compression:    Which decompression scheme to use (if any).  Options are:

                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                          (requires ``file_name`` to be given).

                          - '.fz' => 'rice'
                          - '.gz' => 'gzip'
                          - '.bz2' => 'bzip2'
                          - otherwise None

                        [default: 'auto']
        suppress_warning: Whether to suppress a warning that the WCS could not be read from the
                          FITS header, so the WCS defaulted to either a `PixelScale` or
                          `AffineTransform`. [default: True]

    Returns:
        a Python list of `Image` instances.
    """
    from . import wcs
    file_compress, pyfits_compress = _parse_compression(compression,file_name)

    if file_name and hdu_list is not None:
        raise GalSimIncompatibleValuesError(
            "Cannot provide both file_name and hdu_list", file_name=file_name, hdu_list=hdu_list)
    if not (file_name or hdu_list is not None):
        raise GalSimIncompatibleValuesError(
            "Must provide either file_name or hdu_list", file_name=file_name, hdu_list=hdu_list)

    if file_name:
        hdu_list, fin = _read_file(file_name, dir, file_compress)

    try:
        hdu = _get_hdu(hdu_list, hdu, pyfits_compress)

        if hdu.data is None:
            raise OSError("HDU is empty.  (data is None)")

        wcs, origin = wcs.readFromFitsHeader(hdu.header, suppress_warning)
        dt = hdu.data.dtype.type
        if dt in Image.valid_dtypes:
            data = hdu.data
        else:
            galsim_warn("No C++ Image template instantiation for data type %s. "
                        "Using numpy.float64 instead."%(dt))
            data = hdu.data.astype(np.float64)
        data = np.atleast_3d(data)

        nimages = data.shape[0]
        image_list = []
        for k in range(nimages):
            image = Image(array=data[k,:,:])
            image.setOrigin(origin)
            image.wcs = wcs
            image_list.append(image)

    finally:
        # If we opened a file, don't forget to close it.
        if file_name:
            closeHDUList(hdu_list, fin)

    return image_list

def readFile(file_name, dir=None, hdu=None, compression='auto'):
    """Read in a Pyfits hdu_list from a FITS file, taking care of the GalSim compression options.

    If you want to do something different with an hdu or hdu_list than one of our other read
    functions, you can use this function.  It handles the compression options in the standard
    GalSim way and just returns the hdu (and hdu_list) for you to use as you see fit.

    This function is called as::

        >>> hdu, hdu_list, fin = galsim.fits.readFile(...)

    The first item in the returned tuple is the specified hdu (or the primary if none was
    specifically requested).  The other two are returned so you can properly close them.
    They are the full HDUList and possibly a file handle.  The appropriate cleanup can be
    done with::

        >>> galsim.fits.closeHDUList(hdu_list, fin)

    Parameters:
        file_name:      The name of the file to read in.
        dir:            Optionally a directory name can be provided if ``file_name`` does not
                        already include it. [default: None]
        hdu:            The number of the HDU to return.  [default: None, which means to return
                        either the primary or first extension as appropriate for the given
                        compression.  (e.g. for rice, the first extension is the one you normally
                        want.)]
        compression:    Which decompression scheme to use (if any).  Options are:

                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                          (requires ``file_name`` to be given).

                          - '.fz' => 'rice'
                          - '.gz' => 'gzip'
                          - '.bz2' => 'bzip2'
                          - otherwise None

                        [default: 'auto']

    Returns:
        a tuple with three items: ``(hdu, hdu_list, fin)``.
    """
    file_compress, pyfits_compress = _parse_compression(compression,file_name)
    hdu_list, fin = _read_file(file_name, dir, file_compress)
    hdu = _get_hdu(hdu_list, hdu, pyfits_compress)
    return hdu, hdu_list, fin


##############################################################################################
#
# Finally, we have a class for handling FITS headers called FitsHeader.
#
##############################################################################################


class FitsHeader(object):
    """A class storing key/value pairs from a FITS Header

    This class works a lot like the regular read() function, but rather than returning
    the image part of the FITS file, it gives you access to the header information.

    After construction, you can access a header value by::

        >>> value = fits_header[key]

    or write to it with::

        >>> fits_header[key] = value                # If you just want to set a value.
        >>> fits_header[key] = (value, comment)     # If you want to include a comment field.

    In fact, most of the normal functions available for a dict are available:::

        >>> keys = fits_header.keys()
        >>> items = fits_header.items()
        >>> for key in fits_header:
        >>>     value = fits_header[key]
        >>> value = fits_header.get(key, default)
        >>> del fits_header[key]
        >>> etc.

    .. note::
        This used to be a particularly useful abstraction, since PyFITS and then AstroPy used to
        keep changing their syntax for how to write to a fits header rather often, so this class
        had numerous checks for which version of PPyFITS or AstroPy was installed and call things
        the right way depending on the version.  Thus, it was able to maintain a stable, intuitive
        API that would work with any version on the backend.  We no longer support PyFITS or older
        versions of AstroPy, so now much of the syntax of this class is very similar in interface
        to the current version of astropy.io.fits.Header.  Indeed it is now a rather light wrapper
        around their Header class with just a few convenience features to make it easier to work
        with GalSim objects.

    The underlying Header object is available as a ``.header`` attribute::

        >>> apy_header = fits_header.header

    A FitsHeader may be constructed from a file name, an open PyFits (or astropy.io.fits) HDUList
    object, or a PyFits (or astropy.io.fits) Header object.  It can also be constructed with
    no parameters, in which case a blank Header will be constructed with no keywords yet if
    you want to add the keywords you want by hand.::

        >>> h1 = galsim.FitsHeader(file_name = file_name)
        >>> h2 = galsim.FitsHeader(header = header)
        >>> h3 = galsim.FitsHeader(hdu_list = hdu_list)
        >>> h4 = galsim.FitsHeader()

    For convenience, the first parameter may be unnamed as either a header or a file_name::

        >>> h1 = galsim.FitsHeader(file_name)
        >>> h2 = galsim.FitsHeader(header)

    Parameters:
        header:         An astropy.io.fits.Header object or in fact any dict-like object or list of
                        (key,value) pairs.  [default: None]
        file_name:      The name of the file to read in.  [default: None]
        dir:            Optionally a directory name can be provided if ``file_name`` does not
                        already include it. [default: None]
        hdu_list:       Either an astropy.io.fits.HDUList, an astropy.io.fits.PrimaryHDU, or
                        astropy.io.fits.ImageHDU.  In the former case, the ``hdu`` in the list will
                        be selected.  In the latter two cases, the ``hdu`` parameter is ignored.
                        [default: None]
        hdu:            The number of the HDU to return.  [default: None, which means to return
                        either the primary or first extension as appropriate for the given
                        compression.  (e.g. for rice, the first extension is the one you normally
                        want.)]
        compression:    Which decompression scheme to use (if any).  Options are:

                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                          (requires ``file_name`` to be given).

                          - '.fz' => 'rice'
                          - '.gz' => 'gzip'
                          - '.bz2' => 'bzip2'
                          - otherwise None

                        [default: 'auto']
        text_file:      Normally a file is taken to be a fits file, but you can also give it a
                        text file with the header information (like the .head file output from
                        SCamp).  In this case you should set ``text_file = True`` to tell GalSim
                        to parse the file this way.  [default: False]
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'hdu' : int , 'compression' : str , 'text_file' : bool }

    def __init__(self, header=None, file_name=None, dir=None, hdu_list=None, hdu=None,
                 compression='auto', text_file=False):
        from ._pyfits import pyfits

        if header and file_name:
           raise GalSimIncompatibleValuesError(
               "Cannot provide both file_name and header", file_name=file_name, header=header)
        if header and hdu_list:
           raise GalSimIncompatibleValuesError(
               "Cannot provide both hdu_list and header", hdu_list=hdu_list, header=header)
        if file_name and hdu_list:
           raise GalSimIncompatibleValuesError(
               "Cannot provide both file_name and hdu_list", file_name=file_name, hdu_list=hdu_list)

        # Interpret a string header as though it were passed as file_name.
        if isinstance(header, basestring):
            file_name = header
            header = None

        file_compress, pyfits_compress = _parse_compression(compression,file_name)
        self._tag = None # Used for the repr

        if file_name is not None:
            if dir is not None:
                self._tag = 'file_name='+repr(os.path.join(dir,file_name))
            else:
                self._tag = 'file_name='+repr(file_name)
            if hdu is not None:
                self._tag += ', hdu=%r'%hdu
            if compression != 'auto':
                self._tag += ', compression=%r'%compression

            if text_file:
                self._tag += ', text_file=True'
                if dir is not None:
                    file_name = os.path.join(dir,file_name)
                with open(file_name,"r") as fin:
                    lines = [ line.strip() for line in fin ]
                # Don't include END (or later lines)
                end = lines.index('END') if 'END' in lines else len(lines)
                lines = lines[:end]
                # Later pyfits versions changed this to a class method, so you can write
                # pyfits.Card.fromstring(text).  But in older pyfits versions, it was
                # a regular method.  This syntax should work in both cases.
                cards = [ pyfits.Card().fromstring(line) for line in lines ]
                header = pyfits.Header(cards)
            else:
                hdu_list, fin = _read_file(file_name, dir, file_compress)

        if hdu_list:
            hdu = _get_hdu(hdu_list, hdu, pyfits_compress, header_only=True)
            header = hdu.header

        if file_name and not text_file:
            # If we opened a file, don't forget to close it.
            # Also need to make a copy of the header to keep it available.
            # If we construct a FitsHeader from an hdu_list, then we don't want to do this,
            # since we want the header to remain attached to the original hdu.
            import copy
            self.header = copy.copy(header)
            closeHDUList(hdu_list, fin)
        elif isinstance(header, pyfits.Header):
            # If header is a pyfits.Header, then we just use it.
            self.header = header
        else:
            # Otherwise, header may be any kind of dict-like object or list of (key,value) pairs.
            self.header = pyfits.Header()
            if header is not None:
                if hasattr(header, 'items'):
                    # update() should handle anything that acts like a dict.
                    self.update(header)
                else:
                    for card in header:
                        self.header.append(card, end=True)

    # The rest of the functions are typical non-mutating functions for a dict, for which we
    # generally just pass the request along to self.header.
    def __len__(self):
        return len(self.header)

    def __contains__(self, key):
        return key in self.header

    def __delitem__(self, key):
        self._tag = None
        del self.header[key]

    def __getitem__(self, key):
        return self.header[key]

    def __iter__(self):
        return self.header.__iter__()

    def __setitem__(self, key, value):
        self._tag = None
        self.header[key.upper()] = value

    def comment(self, key):
        """Get the comment field for the given key.

        Parameter:
            key:    The header key for which to get the comment field.

        Returns:
            the comment field.
        """
        return self.header.comments[key.upper()]

    def clear(self):
        """Clear all values in the header.  Works like dict.clear.
        """
        self._tag = None
        self.header.clear()

    def get(self, key, default=None):
        """Get the value of a given key.  Works like dict.get.

        Parameters:
            key:        The header key for which to get the value
            default:    Optionally, A value to use if the key is not present. [default: None]

        Returns:
            the value of the given key
        """
        return self.header.get(key, default)

    def pop(self, key, default=None):
        """Pop off a value from the header.  Works like dict.pop.

        Parameters:
            key:        The header key for which to get the value
            default:    Optionally, A value to use if the key is not present. [default: None]

        Returns:
            the value of the given key
        """
        return self.header.pop(key, default)

    def items(self):
        """Get all header items.  Works like dict.items.

        Returns:
            A list of (key, value) tuples.
        """
        return self.header.items()

    def iteritems(self):
        """Synonym for self.items()
        """
        return self.items()

    def iterkeys(self):
        """Synonym for self.keys()
        """
        return self.keys()

    def itervalues(self):
        """Synonym for self.values()
        """
        return self.values()

    def keys(self):
        """Get all header keys.  Works like dict.keys

        Returns:
            A list of keys.
        """
        return self.header.keys()

    def update(self, dict2):
        """Update the header with a dict-like object.  Works like dict.update.

        If there are any items in ``dict2`` that are duplicates of items already in the header,
        the current items will ber overwritten.

        Parameters:
            dict2:      Another header or dict-like object with keys and values to update.
        """
        self._tag = None
        for key, item in dict2.items():
            self.header[key.upper()] = item

    def values(self):
        """Get all header values.  Works like dict.values.

        Returns:
            A list of values.
        """
        return self.header.values()

    def append(self, key, value='', comment=None, useblanks=True):
        """Append an item to the end of the header.

        This breaks convention a bit by treating the header more like a list than a dict,
        but sometimes that is necessary to get the header structured the way you want it.

        Parameters:
            key:            The key of the entry to append
            value:          The value of the entry to append [default: '']
            comment:        A comment field if desired [default: None]
            useblanks:      If there are blank entries currently at the end, should they be
                            overwritten with the new entry? [default: True]
        """
        self._tag = None
        self.header.append((key.upper(), value, comment), useblanks=useblanks)

    def extend(self, other, replace=False, useblanks=True):
        """Extend this `FitsHeader` with items from another `FitsHeader`.

        Equivalent to appending all the other's items to the end of this one with the
        exception that it ignores items that are already in self.
        If you want to replace existing values rather than ignore duplicates, use
        ``replace=True``.

        Parameters:
            other:          Another FitsHeader object.
            replace:        Replace duplicate entries rather than ignore them. [default: False]
            useblanks:      If there are blank entries currently at the end, should they be
                            overwritten with the new entry? [default: True]
        """
        self._tag = None
        self.header.extend(other.header, unique=not replace, update=replace, useblanks=useblanks)

    def __repr__(self):
        if self._tag is None:
            return "galsim.FitsHeader(header=%r)"%list(self.items())
        else:
            return "galsim.FitsHeader(%s)"%self._tag

    def __str__(self):
        if self._tag is None:
            return "galsim.FitsHeader(header=<Header object at %s>)"%id(self.header)
        else:
            return "galsim.FitsHeader(%s)"%self._tag

    def __eq__(self, other):
        return (self is other or
                (isinstance(other,FitsHeader) and
                 list(self.header.items()) == list(other.header.items())))

    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

# inject write as method of Image class
Image.write = write
