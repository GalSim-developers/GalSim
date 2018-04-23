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
"""@file fits.py
Support for reading and writing Image objects to FITS.

This file includes routines for reading and writing individual Images to/from FITS files, and also
routines for handling multiple Images.
"""

from future.utils import iteritems, iterkeys, itervalues
from past.builtins import basestring
import os
import numpy as np

from .image import Image
from .errors import GalSimError, GalSimValueError, GalSimWarning, GalSimIncompatibleValuesError


##############################################################################################
#
# We start off with some helper functions for some common operations that will be used in
# more than one of our primary read and write functions.
#
##############################################################################################

def _parse_compression(compression, file_name):
    from ._pyfits import pyfits, pyfits_version
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
            else: pass
    else:
        raise GalSimValueError("Invalid compression", compression,
                               ('rice', 'gzip_tile', 'hcompress', 'plio', 'gzip', 'bzip2',
                                'none', 'auto'))
    if pyfits_compress:
        if 'CompImageHDU' not in pyfits.__dict__:
            raise NotImplementedError(
                'Compressed Images not supported before pyfits version 2.0. You have version %s.'%(
                    pyfits_version))

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
        p = subprocess.Popen(["gunzip", "-c", file], stdout=subprocess.PIPE, close_fds=True)
        fin = BytesIO(p.communicate()[0])
        if p.returncode != 0:
            raise IOError("Error running gunzip. Return code = %s"%p.returncode)
        p.wait()
        hdu_list = pyfits.open(fin, 'readonly')
        return hdu_list, fin

    # Note: the above gzip_call function succeeds on travis, so the rest don't get run.
    # Omit them from the coverage test.
    def gzip_in_mem(self, file): # pragma: no cover
        import gzip
        from ._pyfits import pyfits
        fin = gzip.open(file, 'rb')
        hdu_list = pyfits.open(fin, 'readonly')
        # Sometimes this doesn't work.  The symptoms may be that this raises an
        # exception, or possibly the hdu_list comes back empty, in which case the
        # next line will raise an exception.
        hdu = hdu_list[0]
        # pyfits doesn't actually read the file yet, so we can't close fin here.
        # Need to pass it back to the caller and let them close it when they are
        # done with hdu_list.
        return hdu_list, fin

    def pyfits_open(self, file):  # pragma: no cover
        from ._pyfits import pyfits
        # This usually works, although pyfits internally may (depending on the version)
        # use a temporary file, which is why we prefer the above in-memory code if it works.
        # For some versions of pyfits, this is actually the same as the in_mem version.
        hdu_list = pyfits.open(file, 'readonly')
        return hdu_list, None

    def gzip_tmp(self, file):  # pragma: no cover
        import gzip
        from ._pyfits import pyfits
        # Finally, just in case, if everything else failed, here is an implementation that
        # should always work.
        fin = gzip.open(file, 'rb')
        data = fin.read()
        tmp = file + '.tmp'
        # It would be pretty odd for this filename to already exist, but just in case...
        while os.path.isfile(tmp):
            tmp = tmp + '.tmp'
        with open(tmp,"w") as tmpout:
            tmpout.write(data)
        hdu_list = pyfits.open(tmp)
        return hdu_list, tmp

    def bunzip2_call(self, file):
        import subprocess
        from io import BytesIO
        from ._pyfits import pyfits
        p = subprocess.Popen(["bunzip2", "-c", file], stdout=subprocess.PIPE, close_fds=True)
        fin = BytesIO(p.communicate()[0])
        if p.returncode != 0:
            raise IOError("Error running bunzip2. Return code = %s"%p.returncode)
        p.wait()
        hdu_list = pyfits.open(fin, 'readonly')
        return hdu_list, fin

    def bz2_in_mem(self, file): # pragma: no cover
        import bz2
        from ._pyfits import pyfits
        # This normally works.  But it might not on old versions of pyfits.
        fin = bz2.BZ2File(file, 'rb')
        hdu_list = pyfits.open(fin, 'readonly')
        # Sometimes this doesn't work.  The symptoms may be that this raises an
        # exception, or possibly the hdu_list comes back empty, in which case the
        # next line will raise an exception.
        hdu = hdu_list[0]
        return hdu_list, fin

    def bz2_tmp(self, file):  # pragma: no cover
        import bz2
        from ._pyfits import pyfits
        fin = bz2.BZ2File(file, 'rb')
        data = fin.read()
        tmp = file + '.tmp'
        # It would be pretty odd for this filename to already exist, but just in case...
        while os.path.isfile(tmp):
            tmp = tmp + '.tmp'
        with open(tmp,"w") as tmpout:
            tmpout.write(data)
        hdu_list = pyfits.open(tmp)
        return hdu_list, tmp

    def __init__(self):
        # For each compression type, we try them in rough order of efficiency and keep track of
        # which method worked for next time.  Whenever one doesn't work, we increment the
        # method number and try the next one.  The *_call methods are usually the fastest,
        # sometimes much, much faster than the *_in_mem version.  At least for largish files,
        # which are precisely the ones that people would most likely want to compress.
        # However, we can't require the user to have the system executables installed.  So if
        # that fails, we move on to the other options.  It varies which of the other options
        # is fastest, but they all usually succeed, which is the most important thing for a
        # backup method, so it probably doesn't matter much what order we do the rest.
        self.gz_index = 0
        self.bz2_index = 0
        self.gz_methods = [self.gunzip_call, self.gzip_in_mem, self.pyfits_open, self.gzip_tmp]
        self.bz2_methods = [self.bunzip2_call, self.bz2_in_mem, self.bz2_tmp]
        self.gz = self.gz_methods[0]
        self.bz2 = self.bz2_methods[0]

    def __call__(self, file, dir, file_compress):
        from ._pyfits import pyfits, pyfits_version
        if dir:
            import os
            file = os.path.join(dir,file)

        if not file_compress:
            if pyfits_version < '3.1': # pragma: no cover
                # Sometimes early versions of pyfits do weird things with the final hdu when
                # writing fits files with rice compression.  It seems to add a bunch of '\0'
                # characters after the end of what should be the last hdu.  When reading this
                # back in, it gets interpreted as the start of another hdu, which is then found
                # to be missing its END card in the header.  The easiest workaround is to just
                # tell it to ignore any missing END problems on the read command.  Also ignore
                # the warnings it emits along the way.
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hdu_list = pyfits.open(file, 'readonly', ignore_missing_end=True)
            else:
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
                except KeyboardInterrupt:
                    raise
                except: # pragma: no cover
                    self.gz_index += 1
                    self.gz = self.gz_methods[self.gz_index]
            else:  # pragma: no cover
                raise GalSimError("None of the options for gunzipping were successful.")
        elif file_compress == 'bzip2':
            with open(file) as fid: pass
            while self.bz2_index < len(self.bz2_methods):
                try:
                    return self.bz2(file)
                except KeyboardInterrupt:
                    raise
                except: # pragma: no cover
                    self.bz2_index += 1
                    self.bz2 = self.bz2_methods[self.bz2_index]
            else:  # pragma: no cover
                raise GalSimError("None of the options for bunzipping were successful.")
        else:
            raise GalSimValueError("Unknown file_compression", file_compress, ('gzip', 'bzip2'))
_read_file = _ReadFile()

# Do the same trick for _write_file(file,hdu_list,clobber,file_compress,pyfits_compress):
class _WriteFile:

    # There are several methods available for each of gzip and bzip2.  Each is its own function.
    def gzip_call2(self, hdu_list, file):  # pragma: no cover
        root, ext = os.path.splitext(file)
        import subprocess
        if os.path.isfile(root):
            tmp = root + '.tmp'
            # It would be pretty odd for this filename to already exist, but just in case...
            while os.path.isfile(tmp):
                tmp = tmp + '.tmp'
            hdu_list.writeto(tmp)
            p = subprocess.Popen(["gzip", tmp], close_fds=True)
            p.communicate()
            if p.returncode != 0:
                raise IOError("Error running gzip. Return code = %s"%p.returncode)
            p.wait()
            os.rename(tmp+".gz",file)
        else:
            hdu_list.writeto(root)
            p = subprocess.Popen(["gzip", "-S", ext, "-f", root], close_fds=True)
            p.communicate()
            if p.returncode != 0:
                raise IOError("Error running gzip. Return code = %s"%p.returncode)
            p.wait()

    def gzip_call(self, hdu_list, file):
        import subprocess
        with open(file, 'wb') as fout:
            p = subprocess.Popen(["gzip", "-"], stdin=subprocess.PIPE, stdout=fout, close_fds=True)
            hdu_list.writeto(p.stdin)
            p.communicate()
            if p.returncode != 0:
                raise IOError("Error running gzip. Return code = %s"%p.returncode)
            p.wait()

    def gzip_in_mem(self, hdu_list, file):  # pragma: no cover
        import gzip
        import io
        # The compression routines work better if we first write to an internal buffer
        # and then output that to a file.
        buf = io.BytesIO()
        hdu_list.writeto(buf)
        data = buf.getvalue()
        # There is a compresslevel option (for both gzip and bz2), but we just use the
        # default.
        with gzip.open(file, 'wb') as fout:
            fout.write(data)

    def gzip_tmp(self, hdu_list, file):  # pragma: no cover
        import gzip
        # However, pyfits versions before 2.3 do not support writing to a buffer, so the
        # above code will fail.  We need to use a temporary in that case.
        tmp = file + '.tmp'
        # It would be pretty odd for this filename to already exist, but just in case...
        while os.path.isfile(tmp):
            tmp = tmp + '.tmp'
        hdu_list.writeto(tmp)
        with open(tmp,"r") as buf:
            data = buf.read()
        os.remove(tmp)
        with gzip.open(file, 'wb') as fout:
            fout.write(data)

    def bzip2_call2(self, hdu_list, file):  # pragma: no cover
        root, ext = os.path.splitext(file)
        import subprocess
        if os.path.isfile(root) or ext != '.bz2':
            tmp = root + '.tmp'
            # It would be pretty odd for this filename to already exist, but just in case...
            while os.path.isfile(tmp):
                tmp = tmp + '.tmp'
            hdu_list.writeto(tmp)
            p = subprocess.Popen(["bzip2", tmp], close_fds=True)
            p.communicate()
            if p.returncode != 0:
                raise IOError("Error running bzip2. Return code = %s"%p.returncode)
            p.wait()
            os.rename(tmp+".bz2",file)
        else:
            hdu_list.writeto(root)
            p = subprocess.Popen(["bzip2", root], close_fds=True)
            p.communicate()
            if p.returncode != 0:
                raise IOError("Error running bzip2. Return code = %s"%p.returncode)
            p.wait()

    def bzip2_call(self, hdu_list, file):
        import subprocess
        with open(file, 'wb') as fout:
            p = subprocess.Popen(["bzip2"], stdin=subprocess.PIPE, stdout=fout, close_fds=True)
            hdu_list.writeto(p.stdin)
            p.communicate()
            if p.returncode != 0:
                raise IOError("Error running bzip2. Return code = %s"%p.returncode)
            p.wait()

    def bz2_in_mem(self, hdu_list, file):  # pragma: no cover
        import bz2
        import io
        buf = io.BytesIO()
        hdu_list.writeto(buf)
        data = buf.getvalue()
        with bz2.BZ2File(file, 'wb') as fout:
            fout.write(data)

    def bz2_tmp(self, hdu_list, file):  # pragma: no cover
        import bz2
        tmp = file + '.tmp'
        while os.path.isfile(tmp):
            tmp = tmp + '.tmp'
        hdu_list.writeto(tmp)
        with open(tmp,"r") as buf:
            data = buf.read()
        os.remove(tmp)
        with bz2.BZ2File(file, 'wb') as fout:
            fout.write(data)

    def __init__(self):
        # For each compression type, we try them in rough order of efficiency and keep track of
        # which method worked for next time.  Whenever one doesn't work, we increment the
        # method number and try the next one.  The *_call methods seem to be usually the fastest,
        # and we expect that they will usually work.  However, we can't require the user
        # to have the system executables.  Also, some versions of pyfits can't handle writing
        # to the stdin pipe of a subprocess.  So if that fails, the next one, *_call2 is often
        # fastest if the failure was due to pyfits.  If the user does not have gzip or bzip2 (then
        # why are they requesting this compression?), we switch to *_in_mem, which is often
        # almost as good.  (Sometimes it is faster than the call2 option, but when it is slower it
        # can be much slower.)  And finally, if this fails, which I think may happen for very old
        # versions of pyfits, *_tmp is the fallback option.
        self.gz_index = 0
        self.bz2_index = 0
        self.gz_methods = [self.gzip_call, self.gzip_call2, self.gzip_in_mem, self.gzip_tmp]
        self.bz2_methods = [self.bzip2_call, self.bzip2_call2,  self.bz2_in_mem, self.bz2_tmp]
        self.gz = self.gz_methods[0]
        self.bz2 = self.bz2_methods[0]

    def __call__(self, file, dir, hdu_list, clobber, file_compress, pyfits_compress):
        import os
        from ._pyfits import pyfits, pyfits_version
        if dir:
            file = os.path.join(dir,file)

        if os.path.isfile(file):
            if clobber:
                os.remove(file)
            else:
                raise IOError('File %r already exists'%file)

        if not file_compress:
            hdu_list.writeto(file)
        elif file_compress == 'gzip':
            while self.gz_index < len(self.gz_methods):
                try:
                    return self.gz(hdu_list, file)
                except KeyboardInterrupt:
                    raise
                except:  # pragma: no cover
                    self.gz_index += 1
                    self.gz = self.gz_methods[self.gz_index]
            else:  # pragma: no cover
                raise GalSimError("None of the options for gunzipping were successful.")
        elif file_compress == 'bzip2':
            while self.bz2_index < len(self.bz2_methods):
                try:
                    return self.bz2(hdu_list, file)
                except KeyboardInterrupt:
                    raise
                except:  # pragma: no cover
                    self.bz2_index += 1
                    self.bz2 = self.bz2_methods[self.bz2_index]
            else:  # pragma: no cover
                raise GalSimError("None of the options for bunzipping were successful.")
        else:
            raise GalSimValueError("Unknown file_compression", file_compress, ('gzip', 'bzip2'))

        # There is a bug in pyfits where they don't add the size of the variable length array
        # to the TFORMx header keywords.  They should have size at the end of them.
        # This bug has been fixed in version 3.1.2.
        # (See http://trac.assembla.com/pyfits/ticket/199)
        if pyfits_compress and pyfits_version < '3.1.2':
            with pyfits.open(file,'update',disable_image_compression=True) as hdu_list:
                for hdu in hdu_list[1:]: # Skip PrimaryHDU
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

            # Workaround for a bug in some pyfits 3.0.x versions
            # It was fixed in 3.0.8.  I'm not sure when the bug was
            # introduced, but I believe it was 3.0.3.
            if (pyfits_version > '3.0' and pyfits_version < '3.0.8' and
                'COMPRESSION_ENABLED' in pyfits.hdu.compressed.__dict__):
                pyfits.hdu.compressed.COMPRESSION_ENABLED = True
_write_file = _WriteFile()

def _add_hdu(hdu_list, data, pyfits_compress):
    from ._pyfits import pyfits, pyfits_version
    if pyfits_compress:
        if len(hdu_list) == 0:
            hdu_list.append(pyfits.PrimaryHDU())  # Need a blank PrimaryHDU
        if pyfits_version < '4.3':
            hdu = pyfits.CompImageHDU(data, compressionType=pyfits_compress)
        else:
            hdu = pyfits.CompImageHDU(data, compression_type=pyfits_compress)
    else:
        if len(hdu_list) == 0:
            hdu = pyfits.PrimaryHDU(data)
        else:
            hdu = pyfits.ImageHDU(data)
    hdu_list.append(hdu)
    return hdu


def _check_hdu(hdu, pyfits_compress):
    """Check that an input `hdu` is valid
    """
    from ._pyfits import pyfits
    # Check for fixable verify errors
    try:
        hdu.header
        hdu.data
    except pyfits.VerifyError:
        hdu.verify('fix')

    # Check that the specified compression is right for the given hdu type.
    if pyfits_compress:
        if not isinstance(hdu, pyfits.CompImageHDU):  # pragma: no cover
            if isinstance(hdu, pyfits.BinTableHDU):
                raise IOError('Expecting a CompImageHDU, but got a BinTableHDU. Probably your '
                              'pyfits installation does not have the pyfitsComp module installed.')
            elif isinstance(hdu, pyfits.ImageHDU):
                import warnings
                warnings.warn("Expecting a CompImageHDU, but found an uncompressed ImageHDU",
                              GalSimWarning)
            else:
                raise IOError('Found invalid HDU reading FITS file (expected an ImageHDU)')
    else:
        if not isinstance(hdu, pyfits.ImageHDU) and not isinstance(hdu, pyfits.PrimaryHDU):
            raise IOError('Found invalid HDU reading FITS file (expected an ImageHDU)')


def _get_hdu(hdu_list, hdu, pyfits_compress):
    from ._pyfits import pyfits
    if isinstance(hdu_list, pyfits.HDUList):
        # Note: Nothing special needs to be done when reading a compressed hdu.
        # However, such compressed hdu's may not be the PrimaryHDU, so if we think we are
        # reading a compressed file, skip to hdu 1.
        if hdu is None:
            if pyfits_compress:
                if len(hdu_list) <= 1:
                    raise IOError('Expecting at least one extension HDU in galsim.read')
                hdu = 1
            else:
                hdu = 0
        if len(hdu_list) <= hdu:
            raise IOError('Expecting at least %d HDUs in galsim.read'%(hdu+1))
        hdu = hdu_list[hdu]
    else:
        hdu = hdu_list
    _check_hdu(hdu, pyfits_compress)
    return hdu


# Unlike the other helpers, this one doesn't start with an underscore, since we make it
# available to people who use the function ReadFile.
def closeHDUList(hdu_list, fin):
    """If necessary, close the file handle that was opened to read in the `hdu_list`"""
    hdu_list.close()
    if fin:
        if isinstance(fin, basestring): # pragma: no cover
            # In this case, it is a file name that we need to delete.
            # Note: This is relevant for the _tmp versions that are not run on Travis, so
            # don't include this bit in the coverage report.
            import os
            os.remove(fin)
        else:
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

    Write the Image instance `image` to a FITS file, with details depending on the arguments.  This
    function can be called directly as `galsim.fits.write(image, ...)`, with the image as the first
    argument, or as an image method: `image.write(...)`.

    @param image        The image to write to file.  Per the description of this method, it may be
                        given explicitly via `galsim.fits.write(image, ...)` or the method may be
                        called directly as an image method, `image.write(...)`.  Note that if the
                        image has a 'header' attribute containing a FitsHeader, then the FitsHeader
                        is written to the header in the PrimaryHDU, followed by the WCS as usual.
    @param file_name    The name of the file to write to.  [Either `file_name` or `hdu_list` is
                        required.]
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param hdu_list     A pyfits HDUList.  If this is provided instead of `file_name`, then the
                        image is appended to the end of the HDUList as a new HDU. In that case,
                        the user is responsible for calling either `hdu_list.writeto(...)` or
                        `galsim.fits.writeFile(...)` afterwards.  [Either `file_name` or `hdu_list`
                        is required.]
    @param clobber      Setting `clobber=True` when `file_name` is given will silently overwrite
                        existing files. [default: True]
    @param compression  Which compression scheme to use (if any).  Options are:
                        - None or 'none' = no compression
                        - 'rice' = use rice compression in tiles (preserves header readability)
                        - 'gzip' = use gzip to compress the full file
                        - 'bzip2' = use bzip2 to compress the full file
                        - 'gzip_tile' = use gzip in tiles (preserves header readability)
                        - 'hcompress' = use hcompress in tiles (only valid for 2-d images)
                        - 'plio' = use plio compression in tiles (only valid for pos integer data)
                        - 'auto' = determine the compression from the extension of the file name
                                   (requires `file_name` to be given):
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
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
    if hasattr(image, 'header'):
        # Automatically handle old pyfits versions correctly...
        hdu_header = FitsHeader(hdu.header)
        for key in image.header.keys():
            hdu_header[key] = image.header[key]
    if image.wcs:
        image.wcs.writeToFitsHeader(hdu.header, image.bounds)

    if file_name:
        _write_file(file_name, dir, hdu_list, clobber, file_compress, pyfits_compress)


def writeMulti(image_list, file_name=None, dir=None, hdu_list=None, clobber=True,
               compression='auto'):
    """Write a Python list of images to a multi-extension FITS file.

    The details of how the images are written to file depends on the arguments.

    @param image_list   A Python list of Images.  (For convenience, some items in this list
                        may be HDUs already.  Any Images will be converted into pyfits HDUs.)
    @param file_name    The name of the file to write to.  [Either `file_name` or `hdu_list` is
                        required.]
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param hdu_list     A pyfits HDUList.  If this is provided instead of `file_name`, then the
                        image is appended to the end of the HDUList as a new HDU. In that case,
                        the user is responsible for calling either `hdu_list.writeto(...)` or
                        `galsim.fits.writeFile(...)` afterwards.  [Either `file_name` or `hdu_list`
                        is required.]
    @param clobber      See documentation for this parameter on the galsim.fits.write() method.
    @param compression  See documentation for this parameter on the galsim.fits.write() method.
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
            if image.wcs:
                image.wcs.writeToFitsHeader(hdu.header, image.bounds)
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
    writeMulti, when writing a data cube it is necessary that each Image in `image_list` has the
    same size `(nx, ny)`.  No check is made to confirm that all images have the same origin and
    pixel scale (or WCS).

    In fact, the WCS of the first image is the one that gets put into the FITS header (since only
    one WCS can be put into a FITS header).  Thus, if the images have different WCS functions,
    only the first one will be rendered correctly by plotting programs such as ds9.  The FITS
    standard does not support any way to have the various images in a data cube to have different
    WCS solutions.

    @param image_list   The `image_list` can also be either an array of NumPy arrays or a 3d NumPy
                        array, in which case this is written to the fits file directly.  In the
                        former case, no explicit check is made that the NumPy arrays are all the
                        same shape, but a NumPy exception will be raised which we let pass upstream
                        unmolested.
    @param file_name    The name of the file to write to.  [Either `file_name` or `hdu_list` is
                        required.]
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param hdu_list     A pyfits HDUList.  If this is provided instead of `file_name`, then the
                        cube is appended to the end of the HDUList as a new HDU. In that case,
                        the user is responsible for calling either `hdu_list.writeto(...)` or
                        `galsim.fits.writeFile(...)` afterwards.  [Either `file_name` or `hdu_list`
                        is required.]
    @param clobber      See documentation for this parameter on the galsim.fits.write() method.
    @param compression  See documentation for this parameter on the galsim.fits.write() method.
    """
    from ._pyfits import pyfits
    from .bounds import BoundsI

    if isinstance(image_list, np.ndarray):
        is_all_numpy = True
        if image_list.dtype.kind == 'c':
            raise GalSimValueError("Cannot write complex numpy arrays to a fits file. "
                                   "Write array.real and array.imag separately.", image_list)
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
        if (nimages == 0):
            raise GalSimValueError("In writeCube: image_list has no images", image_list)
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
    if wcs:
        wcs.writeToFitsHeader(hdu.header, bounds)

    if file_name:
        _write_file(file_name, dir, hdu_list, clobber, file_compress, pyfits_compress)


def writeFile(file_name, hdu_list, dir=None, clobber=True, compression='auto'):
    """Write a Pyfits hdu_list to a FITS file, taking care of the GalSim compression options.

    If you have used the write(), writeMulti() or writeCube() functions with the `hdu_list` option
    rather than writing directly to a file, you may subsequently use the pyfits command
    `hdu_list.writeto(...)`.  However, it may be more convenient to use this function, writeFile()
    instead, since it treats the compression option consistently with how that option is handled in
    the above functions.

    @param file_name    The name of the file to write to.
    @param hdu_list     A pyfits HDUList.
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param clobber      Setting `clobber=True` will silently overwrite existing files.
                        [default: True]
    @param compression  Which compression scheme to use (if any).  Options are:
                        - None or 'none' = no compression
                        - 'gzip' = use gzip to compress the full file
                        - 'bzip2' = use bzip2 to compress the full file
                        - 'auto' = determine the compression from the extension of the file name
                                   (requires `file_name` to be given):
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
                        Note that the other options, such as 'rice', that operate on the image
                        directly are not available at this point.  If you want to use one of them,
                        it must be applied when writing each hdu.
                        [default: 'auto']
    """
    file_compress, pyfits_compress = _parse_compression(compression,file_name)
    if pyfits_compress and compression != 'auto':
        # If compression is auto and it determined that it should use rice, then we
        # should presume that the hdus were already rice compressed, so we can ignore it here.
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


def read(file_name=None, dir=None, hdu_list=None, hdu=None, compression='auto'):
    """Construct an Image from a FITS file or pyfits HDUList.

    The normal usage for this function is to read a fits file and return the image contained
    therein, automatically decompressing it if necessary.  However, you may also pass it
    an HDUList, in which case it will select the indicated hdu (with the `hdu` parameter)
    from that.

    Not all FITS pixel types are supported (only those with C++ Image template instantiations:
    `short`, `int`, `float`, and `double`).  If the FITS header has GS_* keywords, these will be
    used to initialize the bounding box and WCS.  If not, the bounding box will have `(xmin,ymin)`
    at `(1,1)` and the scale will be set to 1.0.

    This function is called as `im = galsim.fits.read(...)`

    @param file_name    The name of the file to read in.  [Either `file_name` or `hdu_list` is
                        required.]
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param hdu_list     Either a `pyfits.HDUList`, a `pyfits.PrimaryHDU`, or `pyfits.ImageHDU`.
                        In the former case, the `hdu` in the list will be selected.  In the latter
                        two cases, the `hdu` parameter is ignored.  [Either `file_name` or
                        `hdu_list` is required.]
    @param hdu          The number of the HDU to return.  [default: None, which means to return
                        either the primary or first extension as appropriate for the given
                        compression.  (e.g. for 'rice', the first extension is the one you normally
                        want.)]
    @param compression  Which decompression scheme to use (if any).  Options are:
                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                                   (requires `file_name` to be given).
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
                        [default: 'auto']

    @returns the image as an Image instance.
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

        wcs, origin = wcs.readFromFitsHeader(hdu.header)
        dt = hdu.data.dtype.type
        if dt in Image.valid_dtypes:
            data = hdu.data
        else:
            import warnings
            warnings.warn("No C++ Image template instantiation for data type %s. "
                          "Using numpy.float64 instead."%(dt), GalSimWarning)
            data = hdu.data.astype(np.float64)

        image = Image(array=data)
        image.setOrigin(origin)
        image.wcs = wcs

    finally:
        # If we opened a file, don't forget to close it.
        if file_name:
            closeHDUList(hdu_list, fin)

    return image

def readMulti(file_name=None, dir=None, hdu_list=None, compression='auto'):
    """Construct a list of Images from a FITS file or pyfits HDUList.

    The normal usage for this function is to read a fits file and return a list of all the images
    contained therein, automatically decompressing them if necessary.  However, you may also pass
    it an HDUList, in which case it will build the images from these directly.

    Not all FITS pixel types are supported (only those with C++ Image template instantiations:
    `short`, `int`, `float`, and `double`).  If the FITS header has GS_* keywords, these will be
    used to initialize the bounding box and WCS.  If not, the bounding box will have `(xmin,ymin)`
    at `(1,1)` and the scale will be set to 1.0.

    This function is called as `im = galsim.fits.readMulti(...)`


    @param file_name    The name of the file to read in.  [Either `file_name` or `hdu_list` is
                        required.]
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param hdu_list     A `pyfits.HDUList` from which to read the images.  [Either `file_name` or
                        `hdu_list` is required.]
    @param compression  Which decompression scheme to use (if any).  Options are:
                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                                   (requires `file_name` to be given).
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
                        [default: 'auto']

    @returns a Python list of Images
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
                raise IOError('Expecting at least one extension HDU in galsim.read')
        else:
            first = 0
            if len(hdu_list) < 1:
                raise IOError('Expecting at least one HDU in galsim.readMulti')
        for hdu in range(first,len(hdu_list)):
            image_list.append(read(hdu_list=hdu_list, hdu=hdu, compression=pyfits_compress))

    finally:
        # If we opened a file, don't forget to close it.
        if file_name:
            closeHDUList(hdu_list, fin)

    return image_list

def readCube(file_name=None, dir=None, hdu_list=None, hdu=None, compression='auto'):
    """Construct a Python list of Images from a FITS data cube.

    Not all FITS pixel types are supported (only those with C++ Image template instantiations are:
    `short`, `int`, `float`, and `double`).  If the FITS header has GS_* keywords, these will be
    used to initialize the bounding boxes and WCS's.  If not, the bounding boxes will have
    `(xmin,ymin)` at `(1,1)` and the scale will be set to 1.0.

    This function is called as `image_list = galsim.fits.readCube(...)`

    @param file_name    The name of the file to read in.  [Either `file_name` or `hdu_list` is
                        required.]
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param hdu_list     Either a `pyfits.HDUList`, a `pyfits.PrimaryHDU`, or `pyfits.ImageHDU`.
                        In the former case, the `hdu` in the list will be selected.  In the latter
                        two cases, the `hdu` parameter is ignored.  [Either `file_name` or
                        `hdu_list` is required.]
    @param hdu          The number of the HDU to return.  [default: None, which means to return
                        either the primary or first extension as appropriate for the given
                        compression.  (e.g. for rice, the first extension is the one you normally
                        want.)]
    @param compression  Which decompression scheme to use (if any).  Options are:
                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                                   (requires `file_name` to be given).
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
                        [default: 'auto']

    @returns a Python list of Images.
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

    hdu = _get_hdu(hdu_list, hdu, pyfits_compress)

    try:
        wcs, origin = wcs.readFromFitsHeader(hdu.header)
        dt = hdu.data.dtype.type
        if dt in Image.valid_dtypes:
            data = hdu.data
        else:
            import warnings
            warnings.warn("No C++ Image template instantiation for data type %s. "
                          "Using numpy.float64 instead."%(dt), GalSimWarning)
            data = hdu.data.astype(np.float64)

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

    This function is called as:

        >>> hdu, hdu_list, fin = galsim.fits.readFile(...)

    The first item in the returned tuple is the specified hdu (or the primary if none was
    specifically requested).  The other two are returned so you can properly close them.
    They are the full HDUList and possibly a file handle.  The appropriate cleanup can be
    done with:

        >>> galsim.fits.closeHDUList(hdu_list, fin)

    @param file_name    The name of the file to read in.
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param hdu          The number of the HDU to return.  [default: None, which means to return
                        either the primary or first extension as appropriate for the given
                        compression.  (e.g. for rice, the first extension is the one you normally
                        want.)]
    @param compression  Which decompression scheme to use (if any).  Options are:
                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                                   (requires `file_name` to be given).
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
                        [default: 'auto']

    @returns a tuple with three items: `(hdu, hdu_list, fin)`.
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

    After construction, you can access a header value by

        >>> value = fits_header[key]

    or write to it with

        >>> fits_header[key] = value                # If you just want to set a value.
        >>> fits_header[key] = (value, comment)     # If you want to include a comment field.

    In fact, most of the normal functions available for a dict are available:

        >>> keys = fits_header.keys()
        >>> items = fits_header.items()
        >>> for key in fits_header:
        >>>     value = fits_header[key]
        >>> value = fits_header.get(key, default)
        >>> del fits_header[key]
        >>> etc.

    This is a particularly useful abstraction, since pyfits has changed its syntax for how
    to write to a fits header, so this class will work regardless of which version of pyfits
    (or astropy.io.fits) is installed.

    The underlying pyfits.Header object is available as a `.header` attribute:

        >>> pyf_header = fits_header.header

    A FitsHeader may be constructed from a file name, an open PyFits (or astropy.io.fits) HDUList
    object, or a PyFits (or astropy.io.fits) Header object.  It can also be constructed with
    no parameters, in which case a blank Header will be constructed with no keywords yet if
    you want to add the keywords you want by hand.

        >>> h1 = galsim.FitsHeader(file_name = file_name)
        >>> h2 = galsim.FitsHeader(header = header)
        >>> h3 = galsim.FitsHeader(hdu_list = hdu_list)
        >>> h4 = galsim.FitsHeader()

    For convenience, the first parameter may be unnamed as either a header or a file_name:

        >>> h1 = galsim.FitsHeader(file_name)
        >>> h2 = galsim.FitsHeader(header)

    Constructor parameters:

    @param header       A pyfits Header object or in fact any dict-like object or list of
                        (key,value) pairs.  [default: None]
    @param file_name    The name of the file to read in.  [default: None]
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param hdu_list     Either a `pyfits.HDUList`, a `pyfits.PrimaryHDU`, or `pyfits.ImageHDU`.
                        In the former case, the `hdu` in the list will be selected.  In the latter
                        two cases, the `hdu` parameter is ignored.  [default: None]
    @param hdu          The number of the HDU to return.  [default: None, which means to return
                        either the primary or first extension as appropriate for the given
                        compression.  (e.g. for rice, the first extension is the one you normally
                        want.)]
    @param compression  Which decompression scheme to use (if any).  Options are:
                        - None or 'none' = no decompression
                        - 'rice' = use rice decompression in tiles
                        - 'gzip' = use gzip to decompress the full file
                        - 'bzip2' = use bzip2 to decompress the full file
                        - 'gzip_tile' = use gzip decompression in tiles
                        - 'hcompress' = use hcompress decompression in tiles
                        - 'plio' = use plio decompression in tiles
                        - 'auto' = determine the decompression from the extension of the file name
                                   (requires `file_name` to be given).
                                   '*.fz' => 'rice'
                                   '*.gz' => 'gzip'
                                   '*.bz2' => 'bzip2'
                                   otherwise None
                        [default: 'auto']
    @param text_file    Normally a file is taken to be a fits file, but you can also give it a
                        text file with the header information (like the .head file output from
                        SCamp).  In this case you should set `text_file = True` to tell GalSim
                        to parse the file this way.  [default: False]
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'hdu' : int , 'compression' : str , 'text_file' : bool }
    _single_params = []
    _takes_rng = False

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
                import os
                self._tag = 'file_name='+repr(os.path.join(dir,file_name))
            else:
                self._tag = 'file_name='+repr(file_name)
            if hdu is not None:
                self._tag += ', hdu=%r'%hdu
            if compression is not 'auto':
                self._tag += ', compression=%r'%compression

            if text_file:
                self._tag += ', text_file=True'
                if dir is not None:
                    import os
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
            hdu = _get_hdu(hdu_list, hdu, pyfits_compress)
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
                    # for a list, just add each item one at a time.
                    for k,v in header:
                        self.append(k,v,useblanks=False)

    # The rest of the functions are typical non-mutating functions for a dict, for which we
    # generally just pass the request along to self.header.
    def __len__(self):
        from ._pyfits import pyfits_version
        if pyfits_version < '3.1':
            return len(self.header.ascard)
        else:
            return len(self.header)

    def __contains__(self, key):
        return key in self.header

    def __delitem__(self, key):
        self._tag = None
        # This is equivalent to the newer pyfits implementation, but older versions silently
        # did nothing if the key was not in the header.
        if key in self.header:
            del self.header[key]
        else:
            raise KeyError("key %r not in FitsHeader"%(key))

    def __getitem__(self, key):
        return self.header[key]

    def __iter__(self):
        return self.header.__iter__()

    def __setitem__(self, key, value):
        # pyfits doesn't like getting bytes in python 3, so decode if appropriate
        try:
            key = str(key.decode())
        except AttributeError:
            pass
        try:
            value = str(value.decode())
        except AttributeError:
            pass
        from ._pyfits import pyfits_version
        self._tag = None
        if pyfits_version < '3.1':
            if isinstance(value, tuple):
                # header[key] = (value, comment) syntax
                if not (0 < len(value) <= 2):
                    raise GalSimValueError(
                        'A Header item may be set with either a scalar value, '
                        'a 1-tuple containing a scalar value, or a 2-tuple '
                        'containing a scalar value and comment string.', value)
                elif len(value) == 1:
                    self.header.update(key, value[0])
                else:
                    self.header.update(key, value[0], value[1])
            else:
                # header[key] = value syntax
                self.header.update(key, value)
        else:
            # Recent versions implement the above logic with the regular setitem method.
            self.header[key] = value

    def clear(self):
        from ._pyfits import pyfits_version
        self._tag = None
        if pyfits_version < '3.1':
            # Not sure when clear() was added, but not present in 2.4, and present in 3.1.
            del self.header.ascardlist()[:]
        else:
            self.header.clear()

    def get(self, key, default=None):
        return self.header.get(key, default)

    def items(self):
        return self.header.items()

    def iteritems(self):
        from ._pyfits import pyfits_version
        if pyfits_version < '3.1':
            return self.header.items()
        else:
            return iteritems(self.header)

    def iterkeys(self):
        from ._pyfits import pyfits_version
        if pyfits_version < '3.1':
            return self.header.keys()
        else:
            return iterkeys(self.header)

    def itervalues(self):
        from ._pyfits import pyfits_version
        if pyfits_version < '3.1':
            return self.header.ascard.values()
        else:
            return itervalues(self.header)

    def keys(self):
        return self.header.keys()

    def update(self, dict2):
        from ._pyfits import pyfits_version
        self._tag = None
        # dict2 may be a dict or another FitsHeader (or anything that acts like a dict).
        # Note: Don't use self.header.update, since that sometimes has problems (in astropy)
        # with COMMENT lines.  The __setitem__ syntax seems to work properly though.
        for k, v in iteritems(dict2):
            self[k] = v

    def values(self):
        from ._pyfits import pyfits_version
        if pyfits_version < '3.1':
            return self.header.ascard.values()
        else:
            return self.header.values()

    def append(self, key, value='', useblanks=True):
        """Append an item to the end of the header.

        This breaks convention a bit by treating the header more like a list than a dict,
        but sometimes that is necessary to get the header structured the way you want it.

        @param key          The key of the entry to append
        @param value        The value of the entry to append
        @param useblanks    If there are blank entries currently at the end, should they be
                            overwritten with the new entry? [default: True]
        """
        from ._pyfits import pyfits, pyfits_version
        self._tag = None
        if pyfits_version < '3.1':
            # NB. append doesn't quite do what it claims when useblanks=False.
            # If there are blanks, it doesn't put the new item after the blanks.
            # Inserting before the end does do what we want.
            self.header.ascardlist().insert(len(self), pyfits.Card(key, value),
                                            useblanks=useblanks)
        else:
            self.header.insert(len(self), (key, value), useblanks=useblanks)

    def __repr__(self):
        from ._pyfits import pyfits_str
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
        return (isinstance(other,FitsHeader) and
                list(self.header.items()) == list(other.header.items()))

    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def __deepcopy__(self, memo):
        # Need this because pyfits.Header deepcopy was broken before 3.0.6.
        # cf. https://aeon.stsci.edu/ssb/trac/pyfits/ticket/115
        from ._pyfits import pyfits, pyfits_version
        import copy
        # Boilerplate deepcopy implementation.
        # cf. http://stackoverflow.com/questions/1500718/what-is-the-right-way-to-override-the-copy-deepcopy-operations-on-an-object-in-p
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        d1 = self.__dict__
        # This is the special bit for this case.
        if pyfits_version < '3.0.6':
            # Not technically a deepcopy apparently, but good enough in most cases.
            result.header = self.header.copy()
            d1 = d1.copy()
            del d1['header']
        for k, v in d1.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

# inject write as method of Image class
Image.write = write
