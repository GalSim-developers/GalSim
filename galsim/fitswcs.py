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

import warnings
import numpy as np

from .wcs import CelestialWCS
from .position import PositionD, _PositionD
from .angle import radians, arcsec, degrees, AngleUnit
from . import _galsim
from . import fits
from .errors import GalSimError, GalSimValueError, GalSimIncompatibleValuesError
from .errors import GalSimNotImplementedError, convert_cpp_errors, galsim_warn
from .utilities import horner2d

#########################################################################################
#
# We have the following WCS classes that know how to read the WCS from a FITS file:
#
#     AstropyWCS
#     PyAstWCS
#     WcsToolsWCS
#     GSFitsWCS
#
# As for all CelestialWCS classes, they must define the following:
#
#     _radec            function returning (ra, dec) in _radians_ at position (x,y)
#     _xy               function returning (x, y) given (ra, dec) in _radians_.
#     _writeHeader      function that writes the WCS to a fits header.
#     _readHeader       static function that reads the WCS from a fits header.
#     copy              return a copy
#     __eq__            check if this equals another WCS
#
#########################################################################################


class AstropyWCS(CelestialWCS):
    """This WCS uses astropy.wcs to read WCS information from a FITS file.
    It requires the astropy.wcs python module to be installed.

    Astropy may be installed using pip, fink, or port::

        >>> pip install astropy
        >>> fink install astropy-py27
        >>> port install py27-astropy

    It also comes by default with Enthought and Anaconda. For more information, see their website:

        http://www.astropy.org/

    An AstropyWCS is initialized with one of the following commands::

        >>> wcs = galsim.AstropyWCS(file_name=file_name)  # Open a file on disk
        >>> wcs = galsim.AstropyWCS(header=header)        # Use an existing pyfits header
        >>> wcs = galsim.AstropyWCS(wcs=wcs)              # Use an existing astropy.wcs.WCS instance

    Exactly one of the parameters ``file_name``, ``header`` or ``wcs`` is required.  Also, since
    the most common usage will probably be the first, you can also give a ``file_name`` without it
    being named::

        >>> wcs = galsim.AstropyWCS(file_name)

    Parameters:
        file_name:        The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
        dir:              Optional directory to prepend to ``file_name``. [default: None]
        hdu:              Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as
                          appropriate for the given compression.  (e.g. for rice, the first
                          extension is the one you normally want.) [default: None]
        header:           The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a FitsHeader object.  [default: None]
        compression:      Which decompression scheme to use (if any). See galsim.fits.read()
                          for the available options.  [default: 'auto']
        wcs:              An existing astropy.wcs.WCS instance [default: None]
        origin:           Optional origin position for the image coordinate system.
                          If provided, it should be a PositionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : PositionD,
                    "compression" : str }

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 wcs=None, origin=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            import astropy.wcs
            import scipy  # AstropyWCS constructor will do this, so check now.
            import scipy.optimize # Check this too, since it's actually what we need from scipy.

        self._color = None
        self._tag = None # Write something useful here (see below). This is just used for the repr.
        self._set_origin(origin)

        # Read the file if given.
        if file_name is not None:
            if dir is not None:
                import os
                self._tag = repr(os.path.join(dir,file_name))
            else:
                self._tag = repr(file_name)
            if hdu is not None:
                self._tag += ', hdu=%r'%hdu
            if compression != 'auto':
                self._tag += ', compression=%r'%compression
            if header is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both file_name and pyfits header",
                    file_name=file_name, header=header)
            if wcs is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both file_name and wcs", file_name=file_name, wcs=wcs)
            hdu, hdu_list, fin = fits.readFile(file_name, dir, hdu, compression)

        try:
            if file_name is not None:
                header = hdu.header

            # Load the wcs from the header.
            if header is not None:
                if wcs is not None:
                    raise GalSimIncompatibleValuesError(
                        "Cannot provide both pyfits header and wcs", header=header, wcs=wcs)

                # These can mess things up later if they stick around.
                header.pop('BZERO', None)
                header.pop('BSCALE', None)

                self.header = fits.FitsHeader(header)
                try:
                    wcs = self._load_from_header(self.header)
                except Exception as e:  # pragma: no cover
                    # Not sure if this can still trigger.  There used to be input files that
                    # caused various errors in astropy, but that no longer seems to be true
                    # with astropy 4.x.  Leave this check here though, so the user can potentially
                    # get a more comprehensible error message if astropy fails.
                    raise OSError("Astropy failed to read WCS from %s. Original error: %s"%(
                                  file_name, e))
                else:
                    # New kind of error starting in astropy 2.0.5 (I think).  Sometimes, it
                    # gets through the above, but doesn't actually load the right WCS.
                    # E.g. ZPX gets marked as just a ZPN.
                    # As of version 4.0, ZPX is now the only one known to not work.  TPV has
                    # a similar behavior, but we can make it work by an adjustment to the header
                    # in _load_from_header.
                    if 'ZPX' in header.get('CTYPE1','') and 'ZPX' not in wcs.wcs.ctype[0]:
                        raise OSError(
                            "Cannot read WCS in %s with astropy. "%(file_name) +
                            "As of astropy version 4.0.1, ZPX WCS's were still not being " +
                            "correctly read by astropy.wcs. If you believe this has been " +
                            "fixed, please open a GalSim issue to remove this check.")
            else:
                self.header = None

            if wcs is None:
                raise GalSimIncompatibleValuesError(
                    "Must provide one of file_name, header, or wcs",
                    file_name=file_name, header=header, wcs=wcs)

        finally:
            if file_name is not None:
                fits.closeHDUList(hdu_list, fin)

        if not wcs.is_celestial:
            raise GalSimError("The WCS read in does not define a pair of celestial axes" )
        self._wcs = wcs

    def _load_from_header(self, header):
        import astropy.wcs
        if 'TAN' in header.get('CTYPE1','') and 'PV1_1' in header:
            header['CTYPE1'] = header['CTYPE1'].replace('TAN','TPV')
            header['CTYPE2'] = header['CTYPE2'].replace('TAN','TPV')
        with warnings.catch_warnings():
            # The constructor might emit warnings if it wants to fix the header
            # information (e.g. RADECSYS -> RADESYSa).  We'd rather ignore these
            # warnings, since we don't much care if the input file is non-standard
            # so long as we can make it work.
            warnings.simplefilter("ignore")
            wcs = astropy.wcs.WCS(header.header)
        return wcs

    @property
    def wcs(self):
        """The underlying ``astropy.wcs.WCS`` object.
        """
        return self._wcs

    @property
    def origin(self):
        """The origin in image coordinates of the WCS function.
        """
        return self._origin

    def _radec(self, x, y, color=None):
        x1 = np.atleast_1d(x)
        y1 = np.atleast_1d(y)

        ra, dec = self.wcs.all_pix2world(x1, y1, 1, ra_dec_order=True)

        # astropy outputs ra, dec in degrees.  Need to convert to radians.
        factor = degrees / radians

        if np.ndim(x) == np.ndim(y) == 0:
            return ra[0] * factor, dec[0] * factor
        else:
            # Sanity checks that the inputs are the same shape.
            assert np.ndim(x) == np.ndim(y)
            assert x.shape == y.shape
            ra *= factor
            dec *= factor
            return ra, dec

    def _xy(self, ra, dec, color=None):
        factor = radians / degrees

        r1 = np.atleast_1d(ra) * factor
        d1 = np.atleast_1d(dec) * factor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, y = self.wcs.all_world2pix(r1, d1, 1, ra_dec_order=True)

        if np.ndim(ra) == np.ndim(dec) == 0:
            return x[0], y[0]
        else:
            # Sanity checks that the inputs are the same shape.
            assert np.ndim(ra) == np.ndim(dec)
            assert ra.shape == dec.shape
            return x, y

    def _newOrigin(self, origin):
        ret = self.copy()
        ret._origin = origin
        return ret

    def _writeHeader(self, header, bounds):
        # Make a new header with the contents of this WCS.
        # Note: relax = True means to write out non-standard FITS types.
        # Weirdly, this is the default when reading the header, but not when writing.
        header.update(self.wcs.to_header(relax=True))

        # And write the name as a special GalSim key
        header["GS_WCS"] = ("AstropyWCS", "GalSim WCS name")
        # And the image origin.
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")
        return header

    @staticmethod
    def _readHeader(header):
        x0 = header.get("GS_X0",0.)
        y0 = header.get("GS_Y0",0.)
        return AstropyWCS(header=header, origin=_PositionD(x0,y0))

    def copy(self):
        ret = AstropyWCS.__new__(AstropyWCS)
        ret.__dict__.update(self.__dict__)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, AstropyWCS) and
                 self.wcs.to_header(relax=True) == other.wcs.to_header(relax=True) and
                 self.origin == other.origin))

    def __repr__(self):
        if self._tag is not None:
            tag = self._tag
        elif self.header is not None:
            tag = 'header=%r'%self.header
        else:
            tag = 'wcs=%r'%self.wcs
        return "galsim.AstropyWCS(%s, origin=%r)"%(tag, self.origin)

    def __hash__(self): return hash(repr(self))

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_wcs']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._wcs = self._load_from_header(self.header)


class PyAstWCS(CelestialWCS):
    """This WCS uses PyAst (the python front end for the Starlink AST code) to read WCS
    information from a FITS file.  It requires the starlink.Ast python module to be installed.

    Starlink may be installed using pip::

        >>> pip install starlink-pyast

    For more information, see their website:

    https://pypi.python.org/pypi/starlink-pyast/

    A PyAstWCS is initialized with one of the following commands::

        >>> wcs = galsim.PyAstWCS(file_name=file_name)  # Open a file on disk
        >>> wcs = galsim.PyAstWCS(header=header)        # Use an existing pyfits header
        >>> wcs = galsim.PyAstWCS(wcsinfo=wcsinfo)      # Use an existing starlink.Ast.FrameSet

    Exactly one of the parameters ``file_name``, ``header`` or ``wcsinfo`` is required.  Also,
    since the most common usage will probably be the first, you can also give a file name without
    it being named::

        >>> wcs = galsim.PyAstWCS(file_name)

    Parameters:
        file_name:        The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
        dir:              Optional directory to prepend to ``file_name``. [default: None]
        hdu:              Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as
                          appropriate for the given compression.  (e.g. for rice, the first
                          extension is the one you normally want.) [default: None]
        header:           The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a FitsHeader object.  [default: None]
        compression:      Which decompression scheme to use (if any). See galsim.fits.read()
                          for the available options.  [default:'auto']
        wcsinfo:          An existing starlink.Ast.FrameSet [default: None]
        origin:           Optional origin position for the image coordinate system.
                          If provided, it should be a PositionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : PositionD,
                    "compression" : str }

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 wcsinfo=None, origin=None):
        self._color = None
        self._tag = None # Write something useful here (see below). This is just used for the repr.
        self._set_origin(origin)

        # Read the file if given.
        if file_name is not None:
            if dir is not None:
                import os
                self._tag = repr(os.path.join(dir,file_name))
            else:
                self._tag = repr(file_name)
            if hdu is not None:
                self._tag += ', hdu=%r'%hdu
            if compression != 'auto':
                self._tag += ', compression=%r'%compression
            if header is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both file_name and pyfits header",
                    file_name=file_name, header=header)
            if wcsinfo is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both file_name and wcsinfo",
                    file_name=file_name, wcsinfo=wcsinfo)
            hdu, hdu_list, fin = fits.readFile(file_name, dir, hdu, compression)

        try:
            if file_name is not None:
                header = hdu.header

            # Load the wcs from the header.
            if header is not None:
                if wcsinfo is not None:
                    raise GalSimIncompatibleValuesError(
                        "Cannot provide both pyfits header and wcsinfo",
                        header=header, wcsinfo=wcsinfo)

                # These can mess things up later if they stick around.
                header.pop('BZERO', None)
                header.pop('BSCALE', None)

                self.header = fits.FitsHeader(header)
                wcsinfo = self._load_from_header(self.header)
            else:
                self.header = None

            if wcsinfo is None:
                raise GalSimIncompatibleValuesError(
                    "Must provide one of file_name, header, or wcsinfo",
                    file_name=file_name, header=header, wcsinfo=wcsinfo)

            #  We can only handle WCS with 2 pixel axes (given by Nin) and 2 WCS axes
            # (given by Nout).
            if wcsinfo.Nin != 2 or wcsinfo.Nout != 2:  # pragma: no cover
                raise GalSimError("The world coordinate system is not 2-dimensional")

        finally:
            if file_name is not None:
                fits.closeHDUList(hdu_list, fin)

        self._wcsinfo = wcsinfo

    def _load_from_header(self, header):
        import starlink.Atl
        # Note: For much of this class implementation, I've followed the example provided here:
        #       http://dsberry.github.io/starlink/node4.html
        self._fix_header(header)

        # PyFITSAdapter requires an hdu, not a header, so just put it in a pyfits header object.
        # It turns out there are subtle differences between this and using the original FITS
        # file hdu that we read in above.  So there is a slight inefficiency here in creating
        # a new blank PrimaryHDU for this.  But in return we gain more reliable serializability.
        from ._pyfits import pyfits
        hdu = pyfits.PrimaryHDU()
        fits.FitsHeader(hdu_list=hdu).update(header)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # They aren't so good at keeping up with the latest pyfits and numpy syntax, so
            # this next line can emit deprecation warnings.
            # We can safely ignore them (for now...)
            fc = starlink.Ast.FitsChan(starlink.Atl.PyFITSAdapter(hdu))
            #  Read a FrameSet from the FITS header.
            wcsinfo = fc.read()

        if wcsinfo is None:
            raise OSError("Failed to read WCS information from fits file")

        # The PyAst WCS might not have (RA,Dec) axes, which we want.  It might for instance have
        # (Dec, RA) instead.  If it's possible to convert to an (RA,Dec) system, this next line
        # will do so.  And if not, the result will be None.
        # cf. https://github.com/timj/starlink-pyast/issues/8
        wcsinfo = wcsinfo.findframe(starlink.Ast.SkyFrame())
        if wcsinfo is None:
            raise GalSimError("The WCS read in does not define a pair of celestial axes" )

        return wcsinfo

    @property
    def wcsinfo(self):
        """The underlying ``starlink.Ast.FrameSet`` for this object.
        """
        return self._wcsinfo

    @property
    def origin(self):
        """The origin in image coordinates of the WCS function.
        """
        return self._origin

    def _fix_header(self, header):
        # We allow for the option to fix up the header information when a modification can
        # make it readable by PyAst.

        # There was an older proposed standard that used TAN with PV values, which is used by
        # SCamp, so we want to support it if possible.  The standard is now called TPV, which
        # PyAst understands.  All we need to do is change the names of the CTYPE values.
        if ( 'CTYPE1' in header and header['CTYPE1'].endswith('TAN') and
             'CTYPE2' in header and header['CTYPE2'].endswith('TAN') and
             'PV1_1' in header ):
            header['CTYPE1'] = header['CTYPE1'].replace('TAN','TPV')
            header['CTYPE2'] = header['CTYPE2'].replace('TAN','TPV')

    def _radec(self, x, y, color=None):
        # Need this to look like
        #    [ [ x1, x2, x3... ], [ y1, y2, y3... ] ]
        # if input is either scalar x,y or two arrays.
        xy = np.array([np.atleast_1d(x), np.atleast_1d(y)], dtype=float)

        ra, dec = self.wcsinfo.tran( xy )
        # PyAst returns ra, dec in radians, so we're good.

        if np.ndim(x) == np.ndim(y) == 0:
            return ra[0], dec[0]
        else:
            # Sanity checks that the inputs are the same shape.
            assert np.ndim(x) == np.ndim(y)
            assert x.shape == y.shape
            return ra, dec

    def _xy(self, ra, dec, color=None):
        rd = np.array([np.atleast_1d(ra), np.atleast_1d(dec)], dtype=float)
        x, y = self.wcsinfo.tran( rd, False )

        if np.ndim(ra) == np.ndim(dec) == 0:
            return x[0], y[0]
        else:
            # Sanity checks that the inputs are the same shape.
            assert np.ndim(ra) == np.ndim(dec)
            assert ra.shape == dec.shape
            return x, y

    def _newOrigin(self, origin):
        ret = self.copy()
        ret._origin = origin
        return ret

    def _writeHeader(self, header, bounds):
        # See https://github.com/Starlink/starlink/issues/24 for helpful information from
        # David Berry, who assisted me in getting this working.

        from ._pyfits import pyfits
        import starlink.Atl

        hdu = pyfits.PrimaryHDU()
        with warnings.catch_warnings():
            # Again, we can get deprecation warnings here.  Safe to ignore.
            warnings.simplefilter("ignore")
            fc = starlink.Ast.FitsChan(None, starlink.Atl.PyFITSAdapter(hdu) , "Encoding=FITS-WCS")
            # Let Ast know how big the image is that we'll be writing.
            for key in ('NAXIS', 'NAXIS1', 'NAXIS2'):
                if key in header:  # pragma: no branch
                    fc[key] = header[key]
            success = fc.write(self.wcsinfo)
            # PyAst doesn't write out TPV or ZPX correctly.  It writes them as TAN and ZPN
            # respectively.  However, if the maximum error is less than 0.1 pixel, it claims
            # success nonetheless.  This doesn't seem accurate enough for many purposes,
            # so we need to countermand that.
            # The easiest way I found to check for them is that the string TPN is in the string
            # version of wcsinfo.  So check for that and set success = False in that case.
            if 'TPN' in str(self.wcsinfo): success = False
            # Likewise for SIP.  MPF seems to be an appropriate string to look for.
            if 'MPF' in str(self.wcsinfo): success = False
            if not success:
                # This should always work, since it uses starlinks own proprietary encoding, but
                # it won't necessarily be readable by ds9.
                fc = starlink.Ast.FitsChan(None, starlink.Atl.PyFITSAdapter(hdu))
                fc.write(self.wcsinfo)
            fc.writefits()
        header.update(hdu.header)

        # And write the name as a special GalSim key
        header["GS_WCS"] = ("PyAstWCS", "GalSim WCS name")
        # And the image origin.
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")
        return header

    @staticmethod
    def _readHeader(header):
        x0 = header.get("GS_X0",0.)
        y0 = header.get("GS_Y0",0.)
        return PyAstWCS(header=header, origin=_PositionD(x0,y0))

    def copy(self):
        ret = PyAstWCS.__new__(PyAstWCS)
        ret.__dict__.update(self.__dict__)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, PyAstWCS) and
                 repr(self.wcsinfo) == repr(other.wcsinfo) and
                 self.origin == other.origin))

    def __repr__(self):
        if self._tag is not None:
            tag = self._tag
        elif self.header is not None:
            tag = 'header=%r'%self.header
        else:
            # Ast doesn't have a good repr for a FrameSet, so do it ourselves.
            tag = 'wcsinfo=<starlink.Ast.FrameSet at %s>'%id(self.wcsinfo)
        return "galsim.PyAstWCS(%s, origin=%r)"%(tag, self.origin)

    def __hash__(self): return hash(repr(self))

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_wcsinfo']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._wcsinfo = self._load_from_header(self.header)


# I can't figure out how to get wcstools installed in the travis environment (cf. .travis.yml).
# So until that gets resolved, we omit this class from the coverage report.
# This class was mostly useful as a refernce implementation anyway. It's much too slow for most
# users to ever want to use it.
class WcsToolsWCS(CelestialWCS): # pragma: no cover
    """This WCS uses wcstools executables to perform the appropriate WCS transformations
    for a given FITS file.  It requires wcstools command line functions to be installed.

    Note: It uses the wcstools executables xy2sky and sky2xy, so it can be quite a bit less
    efficient than other options that keep the WCS in memory.

    See their website for information on downloading and installing wcstools:

        http://tdc-www.harvard.edu/software/wcstools/

    A WcsToolsWCS is initialized with the following command::

        >>> wcs = galsim.WcsToolsWCS(file_name)

    Parameters:
        file_name:        The FITS file from which to read the WCS information.
        dir:              Optional directory to prepend to ``file_name``. [default: None]
        origin:           Optional origin position for the image coordinate system.
                          If provided, it should be a PositionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "origin" : PositionD }

    def __init__(self, file_name, dir=None, origin=None):
        self._color = None
        self._set_origin(origin)

        import os
        if dir:
            file_name = os.path.join(dir, file_name)
        if not os.path.isfile(file_name):
            raise OSError('Cannot find file '+file_name)
        self._file_name = file_name

        # Check wcstools is installed and that it can read the file.
        import subprocess
        # If xy2sky is not installed, this will raise an OSError
        p = subprocess.Popen(['xy2sky', '-d', '-n', '10', file_name, '0', '0'],
                             stdout=subprocess.PIPE)
        results = p.communicate()[0].decode()
        p.stdout.close()
        if len(results) == 0 or 'cannot' in results:
            raise OSError('wcstools (specifically xy2sky) was unable to read '+file_name)

        # wcstools supports LINEAR WCS's, but we don't want to allow them, since then
        # the CelestialWCS base class is inappropriate.  The clue to detect this is that
        # the results only have 4 values, rather than use usual 5 (missing epoch).
        if len(results.split()) == 4:
            raise GalSimError("The WCS read in does not define a pair of celestial axes" )

    @property
    def file_name(self):
        """The file name of the FITS file with the WCS information.
        """
        return self._file_name

    @property
    def origin(self):
        """The origin in image coordinates of the WCS function.
        """
        return self._origin

    def _radec(self, x, y, color=None):
        import subprocess
        import os

        # Need this to look like
        #    [ x1, y1, x2, y2, ... ]
        # if input is either scalar x,y or two arrays.
        xy = np.array([x, y], dtype=float).transpose().ravel()

        # The OS cannot handle arbitrarily long command lines, so we may need to split up
        # the list into smaller chunks.
        if 'SC_ARG_MAX' in os.sysconf_names:
            arg_max = os.sysconf('SC_ARG_MAX')
        else:
            # A conservative guess. My machines have 131072, 262144, and 2621440
            arg_max = 32768

        # Sometimes SC_ARG_MAX is listed as -1.  Apparently that means "the configuration name
        # is known, but the value is not defined." So, just go with the above conservative value.
        if arg_max <= 0:
            arg_max = 32768

        # Just in case something weird happened.  This should be _very_ conservative.
        # It's the smallest value in this list of values for a bunch of systems:
        # http://www.in-ulm.de/~mascheck/various/argmax/
        if arg_max < 4096:
            arg_max = 4096

        # This corresponds to the total number of characters in the line.
        # But we really need to know how many arguments we are allowed to use in each call.
        # Lets be conservative again and assume each argument is at most 20 characters.
        # (We ignore the few characters at the start for the command name and such.)
        nargs = int(arg_max / 40) * 2  # Make sure it is even!

        xy_strs = [ str(z) for z in xy ]
        ra = []
        dec = []

        for i in range(0,len(xy_strs),nargs):
            xy1 = xy_strs[i:i+nargs]
            # We'd like to get the output to 10 digits of accuracy.  This corresponds to
            # an accuracy of about 1.e-6 arcsec.  But sometimes xy2sky cannot handle it,
            # in which case the output will start with *************.  If this happens, just
            # decrease digits and try again.
            for digits in range(10,5,-1):
                # If xy2sky is not installed, this will raise an OSError
                p = subprocess.Popen(['xy2sky', '-d', '-n', str(digits), self._file_name] + xy1,
                                    stdout=subprocess.PIPE)
                results = p.communicate()[0].decode()
                p.stdout.close()
                if len(results) == 0:
                    raise OSError('wcstools command xy2sky was unable to read '+ self._file_name)
                if results[0] != '*': break
            if results[0] == '*':
                raise OSError('wcstools command xy2sky was unable to read '+self._file_name)
            lines = results.splitlines()

            # Each line of output should looke like:
            #    x y J2000 ra dec
            # But if there was an error, the J200 might be missing or the output might look like
            #    Off map x y
            for line in lines:
                vals = line.split()
                if len(vals) != 5:
                    raise GalSimError('wcstools xy2sky returned invalid result near %s'%(xy1))
                ra.append(float(vals[0]))
                dec.append(float(vals[1]))

        # wcstools reports ra, dec in degrees, so convert to radians
        factor = degrees / radians

        if np.ndim(x) == np.ndim(y) == 0:
            return ra[0]*factor, dec[0]*factor
        else:
            # Sanity checks that the inputs are the same shape.
            assert np.ndim(x) == np.ndim(y)
            assert x.shape == y.shape
            return np.array(ra)*factor, np.array(dec)*factor

    def _xy(self, ra, dec, color=None):
        import subprocess
        import os

        rd = np.array([ra, dec], dtype=float).transpose().ravel()
        rd *= radians / degrees

        # The boilerplate here is exactly the same as in _radec.  See that function for an
        # explanation of how this works.
        if 'SC_ARG_MAX' in os.sysconf_names:
            arg_max = os.sysconf('SC_ARG_MAX')
        else:
            arg_max = 32768
        if arg_max <= 0: arg_max = 32768
        if arg_max < 4096: arg_max = 4096
        nargs = int(arg_max / 40) * 2

        rd_strs = [ str(z) for z in rd ]
        x = []
        y = []

        for i in range(0,len(rd_strs),nargs):
            rd1 = rd_strs[i:i+nargs]
            for digits in range(10,5,-1):
                p = subprocess.Popen(['sky2xy', '-n', str(digits), self._file_name] + rd1,
                                    stdout=subprocess.PIPE)
                results = p.communicate()[0].decode()
                p.stdout.close()
                if len(results) == 0:
                    raise OSError('wcstools command sky2xy was unable to read '+self._file_name)
                if results[0] != '*': break
            if results[0] == '*':
                raise OSError('wcstools command sky2xy was unable to read '+self._file_name)

            lines = results.splitlines()

            # Each line of output should looke like:
            #    ra dec J2000 -> x y
            # However, if there was an error, the J200 might be missing.
            for line in lines:
                vals = line.split()
                if len(vals) < 6:
                    raise GalSimError('wcstools sky2xy returned invalid result for %f,%f'%(ra,dec))
                if len(vals) > 6:
                    galsim_warn("wcstools sky2xy indicates that %f,%f is off the image. "
                                "output is %r"%(ra,dec,results))
                x.append(float(vals[4]))
                y.append(float(vals[5]))

        if np.ndim(ra) == np.ndim(dec) == 0:
            return x[0], y[0]
        else:
            # Sanity checks that the inputs are the same shape.
            assert np.ndim(ra) == np.ndim(dec)
            assert ra.shape == dec.shape
            return np.array(x), np.array(y)

    def _newOrigin(self, origin):
        ret = self.copy()
        ret._origin = origin
        return ret

    def _writeHeader(self, header, bounds):
        # These are all we need to load it back.  Just use the original file.
        header["GS_WCS"]  = ("WcsToolsWCS", "GalSim WCS name")
        header["GS_FILE"] = (self._file_name, "GalSim original file with WCS data")
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")

        # We also copy over some of the fields we need.  wcstools doesn't seem to have something
        # that lists _all_ the keys that define the WCS.  This just gets the approximate WCS.
        import subprocess
        p = subprocess.Popen(['wcshead', self._file_name], stdout=subprocess.PIPE)
        results = p.communicate()[0].decode()
        p.stdout.close()
        v = results.split()
        header["CTYPE1"] = v[3]
        header["CTYPE2"] = v[4]
        header["CRVAL1"] = v[5]
        header["CRVAL2"] = v[6]
        header["CRPIX1"] = v[8]
        header["CRPIX2"] = v[9]
        header["CDELT1"] = v[10]
        header["CDELT2"] = v[11]
        header["CROTA2"] = v[12]
        return header

    @staticmethod
    def _readHeader(header):
        file = header["GS_FILE"]
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        return WcsToolsWCS(file, origin=_PositionD(x0,y0))

    def copy(self):
        # The copy module version of copying the dict works fine here.
        import copy
        return copy.copy(self)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, WcsToolsWCS) and
                 self._file_name == other.file_name and
                 self.origin == other.origin))

    def __repr__(self):
        return "galsim.WcsToolsWCS(%r, origin=%r)"%(self._file_name, self.origin)

    def __hash__(self): return hash(repr(self))


class GSFitsWCS(CelestialWCS):
    """This WCS uses a GalSim implementation to read a WCS from a FITS file.

    It doesn't do nearly as many WCS types as the other options, and it does not try to be
    as rigorous about supporting all possible valid variations in the FITS parameters.
    However, it does several popular WCS types properly, and it doesn't require any additional
    python modules to be installed, which can be helpful.

    Currrently, it is able to parse the following WCS types: TAN, STG, ZEA, ARC, TPV, TNX

    A GSFitsWCS is initialized with one of the following commands::

        >>> wcs = galsim.GSFitsWCS(file_name=file_name)  # Open a file on disk
        >>> wcs = galsim.GSFitsWCS(header=header)        # Use an existing pyfits header

    Also, since the most common usage will probably be the first, you can also give a file name
    without it being named::

        >>> wcs = galsim.GSFitsWCS(file_name)

    In addition to reading from a FITS file, there is also a factory function that builds
    a GSFitsWCS object implementing a TAN projection.  See the docstring of `TanWCS` for
    more details.

    Parameters:
        file_name:        The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
        dir:              Optional directory to prepend to ``file_name``. [default: None]
        hdu:              Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as
                          appropriate for the given compression.  (e.g. for rice, the first
                          extension is the one you normally want.) [default: None]
        header:           The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a FitsHeader object.  [default: None]
        compression:      Which decompression scheme to use (if any). See galsim.fits.read()
                          for the available options.  [default: 'auto']
        origin:           Optional origin position for the image coordinate system.
                          If provided, it should be a PositionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : PositionD,
                    "compression" : str }

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 origin=None, _data=None, _doiter=True):
        # Note: _data is not intended for end-user use.  It enables the equivalent of a
        #       private constructor of GSFitsWCS by the function TanWCS.  The details of its
        #       use are intentionally not documented above.

        self._color = None
        self._tag = None # Write something useful here (see below). This is just used for the str.
        self._doiter = _doiter

        # If _data is given, copy the data and we're done.
        if _data is not None:
            self.wcs_type = _data[0]
            self.crpix = _data[1]
            self.cd = _data[2]
            self.center = _data[3]
            self.pv = _data[4]
            self.ab = _data[5]
            self.abp = _data[6]
            if self.wcs_type in ('TAN', 'TPV', 'TNX', 'TAN-SIP'):
                self.projection = 'gnomonic'
            elif self.wcs_type in ('STG', 'STG-SIP'):
                self.projection = 'stereographic'
            elif self.wcs_type in ('ZEA', 'ZEA-SIP'):
                self.projection = 'lambert'
            elif self.wcs_type in ('ARC', 'ARC-SIP'):
                self.projection = 'postel'
            else:
                raise ValueError("Invalid wcs_type in _data")
            return

        # Read the file if given.
        if file_name is not None:
            if dir is not None:
                import os
                self._tag = repr(os.path.join(dir,file_name))
            else:
                self._tag = repr(file_name)
            if hdu is not None:
                self._tag += ', hdu=%r'%hdu
            if compression != 'auto':
                self._tag += ', compression=%r'%compression
            if header is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both file_name and pyfits header",
                    file_name=file_name, header=header)
            hdu, hdu_list, fin = fits.readFile(file_name, dir, hdu, compression)

        try:
            if file_name is not None:
                header = hdu.header

            if header is None:
                raise GalSimIncompatibleValuesError(
                    "Must provide either file_name or header", file_name=file_name, header=header)

            # Read the wcs information from the header.
            self._read_header(header)

        finally:
            if file_name is not None:
                fits.closeHDUList(hdu_list, fin)

        if origin is not None:
            self.crpix += [ origin.x, origin.y ]

    # The origin is a required attribute/property, since it is used by some functions like
    # shiftOrigin to get the current origin value.  We don't use it in this class, though, so
    # just make origin a dummy property that returns 0,0.
    @property
    def origin(self):
        """The origin in image coordinates of the WCS function.
        """
        return _PositionD(0.,0.)

    def _read_header(self, header):
        from .angle import AngleUnit
        from .celestial import CelestialCoord
        # Start by reading the basic WCS stuff that most types have.
        ctype1 = header.get('CTYPE1','')
        ctype2 = header.get('CTYPE2','')
        if ctype1.startswith('DEC--') and ctype2.startswith('RA---'):
            flip = True
        elif ctype1.startswith('RA---') and ctype2.startswith('DEC--'):
            flip = False
        else:
            raise GalSimError(
                "GSFitsWCS only supports celestial coordinate systems."
                "Expecting CTYPE1,2 to start with RA--- and DEC--.  Got %s, %s"%(ctype1, ctype2))
        if ctype1[5:] != ctype2[5:]:  # pragma: no cover
            raise OSError("ctype1, ctype2 do not seem to agree on the WCS type")
        self.wcs_type = ctype1[5:]
        if self.wcs_type in ('TAN', 'TPV', 'TNX', 'TAN-SIP'):
            self.projection = 'gnomonic'
        elif self.wcs_type in ('STG', 'STG-SIP'):
            self.projection = 'stereographic'
        elif self.wcs_type in ('ZEA', 'ZEA-SIP'):
            self.projection = 'lambert'
        elif self.wcs_type in ('ARC', 'ARC-SIP'):
            self.projection = 'postel'
        else:
            raise GalSimValueError("GSFitsWCS cannot read files using given wcs_type.",
                                   self.wcs_type,
                                   ('TAN', 'TPV', 'TNX', 'TAN-SIP', 'STG', 'STG-SIP', 'ZEA',
                                    'ZEA-SIP', 'ARC', 'ARC-SIP'))
        crval1 = float(header['CRVAL1'])
        crval2 = float(header['CRVAL2'])
        crpix1 = float(header['CRPIX1'])
        crpix2 = float(header['CRPIX2'])
        if 'CD1_1' in header:
            cd11 = float(header['CD1_1'])
            cd12 = float(header['CD1_2'])
            cd21 = float(header['CD2_1'])
            cd22 = float(header['CD2_2'])
        elif 'CDELT1' in header:
            if 'PC1_1' in header:
                cd11 = float(header['PC1_1']) * float(header['CDELT1'])
                cd12 = float(header['PC1_2']) * float(header['CDELT1'])
                cd21 = float(header['PC2_1']) * float(header['CDELT2'])
                cd22 = float(header['PC2_2']) * float(header['CDELT2'])
            else:
                cd11 = float(header['CDELT1'])
                cd12 = 0.
                cd21 = 0.
                cd22 = float(header['CDELT2'])
        else:  # pragma: no cover  (all our test files have either CD or CDELT)
            cd11 = 1.
            cd12 = 0.
            cd21 = 0.
            cd22 = 1.

        # Usually the units are degrees, but make sure
        if 'CUNIT1' in header:
            cunit1 = header['CUNIT1']
            cunit2 = header['CUNIT2']
            ra_units = AngleUnit.from_name(cunit1)
            dec_units = AngleUnit.from_name(cunit2)
        else:
            ra_units = degrees
            dec_units = degrees

        if flip:
            crval1, crval2 = crval2, crval1
            ra_units, dec_units = dec_units, ra_units
            cd11, cd21 = cd21, cd11
            cd12, cd22 = cd22, cd12

        self.crpix = np.array( [ crpix1, crpix2 ] )
        self.cd = np.array( [ [ cd11, cd12 ],
                              [ cd21, cd22 ] ] )

        self.center = CelestialCoord(crval1 * ra_units, crval2 * dec_units)

        # There was an older proposed standard that used TAN with PV values, which is used by
        # SCamp, so we want to support it if possible.  The standard is now called TPV, so
        # use that for our wcs_type if we see the PV values with TAN.
        if self.wcs_type == 'TAN' and 'PV1_1' in header:
            self.wcs_type = 'TPV'

        self.pv = None
        self.ab = None
        self.abp = None
        if self.wcs_type == 'TPV':
            self._read_tpv(header)
        elif self.wcs_type == 'TNX':
            self._read_tnx(header)
        elif self.wcs_type in ('TAN-SIP', 'STG-SIP', 'ZEA-SIP', 'ARC-SIP'):
            self._read_sip(header)

        # I think the CUNIT specification applies to the CD matrix as well, but I couldn't actually
        # find good documentation for this.  Plus all the examples I saw used degrees anyway, so
        # it's hard to tell.  Hopefully this will never matter, but if CUNIT is not deg, this
        # next bit might be wrong.
        # I did see documentation that the PV matrices always use degrees, so at least we shouldn't
        # have to worry about that.
        if ra_units != degrees:  # pragma: no cover
            self.cd[0,:] *= 1. * ra_units / degrees
        if dec_units != degrees:  # pragma: no cover
            self.cd[1,:] *= 1. * dec_units / degrees

    def _read_tpv(self, header):
        # See http://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html for details about how
        # the TPV standard is defined.

        # The standard includes an option to have odd powers of r, which kind of screws
        # up the numbering of these coefficients.  We don't implement these terms, so
        # before going further, check to make sure none are present.
        odd_indices = [3, 11, 23, 39]
        if any((header.get('PV%s_%s'%(i,j), 0.) != 0. for i in [1,2] for j in odd_indices)):
            raise GalSimNotImplementedError("TPV not implemented for odd powers of r")

        pv1 = [ float(header.get('PV1_%s'%k, 0.)) for k in range(40) if k not in odd_indices ]
        pv2 = [ float(header.get('PV2_%s'%k, 0.)) for k in range(40) if k not in odd_indices ]

        maxk = max(np.nonzero(pv1)[0][-1], np.nonzero(pv2)[0][-1])
        # maxk = (order+1) * (order+2) / 2 - 1
        order = int(np.floor(np.sqrt(2*(maxk+1)))) - 1
        self.pv = np.zeros((2,order+1,order+1))

        # Another strange thing is that the two matrices are defined in the opposite order
        # with respect to their element ordering.  But at least now, without the odd terms,
        # we can just proceed in order in the k indices.  So what we call k=3..9 here were
        # originally PVi_4..10.
        # For reference, here is what it would look like for order = 3:
        # self.pv = np.array( [ [ [ pv1[0], pv1[2], pv1[5], pv1[9] ],
        #                         [ pv1[1], pv1[4], pv1[8],   0.   ],
        #                         [ pv1[3], pv1[7],   0.  ,   0.   ],
        #                         [ pv1[6],   0.  ,   0.  ,   0.   ] ],
        #                       [ [ pv2[0], pv2[1], pv2[3], pv2[6] ],
        #                         [ pv2[2], pv2[4], pv2[7],   0.   ],
        #                         [ pv2[5], pv2[8],   0.  ,   0.   ],
        #                         [ pv2[9],   0.  ,   0.  ,   0.   ] ] ] )
        k = 0
        for N in range(order+1):
            for j in range(N+1):
                i = N-j
                self.pv[0,i,j] = pv1[k]
                self.pv[1,j,i] = pv2[k]
                k = k+1

    def _read_sip(self, header):
        a_order = int(header['A_ORDER'])
        b_order = int(header['B_ORDER'])
        order = max(a_order,b_order)  # Use the same order for both
        a = [ float(header.get('A_'+str(i)+'_'+str(j),0.))
                for i in range(order+1) for j in range(order+1) ]
        a = np.array(a).reshape((order+1,order+1))
        b = [ float(header.get('B_'+str(i)+'_'+str(j),0.))
                for i in range(order+1) for j in range(order+1) ]
        b = np.array(b).reshape((order+1,order+1))
        a[1,0] += 1  # Standard A,B are a differential calculation.  It's more convenient to
        b[0,1] += 1  # keep this as an absolute calculation like PV does.
        self.ab = np.array([a, b])

        # The reverse transformation is not required to be there.
        if 'AP_ORDER' in header:
            ap_order = int(header['AP_ORDER'])
            bp_order = int(header['BP_ORDER'])
            order = max(ap_order,bp_order)  # Use the same order for both
            ap = [ float(header.get('AP_'+str(i)+'_'+str(j),0.))
                    for i in range(order+1) for j in range(order+1) ]
            ap = np.array(ap).reshape((order+1,order+1))
            bp = [ float(header.get('BP_'+str(i)+'_'+str(j),0.))
                    for i in range(order+1) for j in range(order+1) ]
            bp = np.array(bp).reshape((order+1,order+1))
            ap[1,0] += 1
            bp[0,1] += 1
            self.abp = np.array([ap, bp])

    def _read_tnx(self, header):

        # TNX has a few different options.  Rather than keep things in the native format,
        # we actually convert to the equivalent of TPV to make the actual operations faster.
        # See http://iraf.noao.edu/projects/ccdmosaic/tnx.html for details.

        # First, parse the input values, which are stored in WAT keywords:
        k = 1
        wat1 = ""
        key = 'WAT1_%03d'%k
        while key in header:
            wat1 += header[key]
            k = k+1
            key = 'WAT1_%03d'%k
        wat1 = wat1.split()

        k = 1
        wat2 = ""
        key = 'WAT2_%03d'%k
        while key in header:
            wat2 += header[key]
            k = k+1
            key = 'WAT2_%03d'%k
        wat2 = wat2.split()

        if ( len(wat1) < 12 or
             wat1[0] != 'wtype=tnx' or
             wat1[1] != 'axtype=ra' or
             wat1[2] != 'lngcor' or
             wat1[3] != '=' or
             not wat1[4].startswith('"') or
             not wat1[-1].endswith('"') ):  # pragma: no cover
            raise GalSimError("TNX WAT1 was not as expected")
        if ( len(wat2) < 12 or
             wat2[0] != 'wtype=tnx' or
             wat2[1] != 'axtype=dec' or
             wat2[2] != 'latcor' or
             wat2[3] != '=' or
             not wat2[4].startswith('"') or
             not wat2[-1].endswith('"') ):  # pragma: no cover
            raise GalSimError("TNX WAT2 was not as expected")

        # Break the next bit out into another function, since it is the same for x and y.
        pv1 = self._parse_tnx_data(wat1[4:])
        pv2 = self._parse_tnx_data(wat2[4:])

        # Those just give the adjustments to the position, not the matrix that gives the final
        # position.  i.e. the TNX standard uses u = u + [1 u u^2 u^3] PV [1 v v^2 v^3]T.
        # So we need to add 1 to the correct term in each matrix to get what we really want.
        pv1[1,0] += 1.
        pv2[0,1] += 1.

        # Finally, store these as our pv 3-d array.
        self.pv = np.array([pv1, pv2])

        # We've now converted this to TPV, so call it that when we output to a fits header.
        self.wcs_type = 'TPV'

    def _parse_tnx_data(self, data):

        # I'm not sure if there is any requirement on there being a space before the final " and
        # not before the initial ".  But both the example in the description of the standard and
        # the one we have in our test directory are this way.  Here, if the " is by itself, I
        # remove the item, and if it is part of a longer string, I just strip it off.  Seems the
        # most sensible thing to do.
        if data[0] == '"':  # pragma: no cover
            data = data[1:]
        else:
            data[0] = data[0][1:]
        if data[-1] == '"':
            data = data[:-1]
        else:  # pragma: no cover
            data[-1] = data[-1][:-1]

        code = int(data[0].strip('.'))  # Weirdly, these integers are given with decimal points.
        xorder = int(data[1].strip('.'))
        yorder = int(data[2].strip('.'))
        cross = int(data[3].strip('.'))
        if cross != 2:  # pragma: no cover
            raise GalSimNotImplementedError("TNX only implemented for half-cross option.")
        if xorder != 4 or yorder != 4:  # pragma: no cover
            raise GalSimNotImplementedError("TNX only implemented for order = 4")
        # Note: order = 4 really means cubic.  order is how large the pv matrix is, i.e. 4x4.

        xmin = float(data[4])
        xmax = float(data[5])
        ymin = float(data[6])
        ymax = float(data[7])

        pv1 = [ float(x) for x in data[8:] ]
        if len(pv1) != 10:  # pragma: no cover
            raise GalSimError("Wrong number of items found in WAT data")

        # Put these into our matrix formulation.
        pv = np.array( [ [ pv1[0], pv1[4], pv1[7], pv1[9] ],
                         [ pv1[1], pv1[5], pv1[8],   0.   ],
                         [ pv1[2], pv1[6],   0.  ,   0.   ],
                         [ pv1[3],   0.  ,   0.  ,   0.   ] ] )

        # Convert from Legendre or Chebyshev polynomials into regular polynomials.
        if code < 3: # pragma: no branch (The only test file I can find has code = 1)
            # Instead of 1, x, x^2, x^3, Chebyshev uses: 1, x', 2x'^2 - 1, 4x'^3 - 3x
            # where x' = (2x - xmin - xmax) / (xmax-xmin).
            # Similarly, with y' = (2y - ymin - ymin) / (ymax-ymin)
            # We'd like to convert the pv matrix from being in terms of x' and y' to being
            # in terms of just x, y.  To see how this works, look at what pv[1,1] means:
            #
            # First, let's say we can write x as (a + bx), and we can write y' as (c + dy).
            # Then the term for pv[1,1] is:
            #
            # term = x' * pv[1,1] * y'
            #      = (a + bx) * pv[1,1] * (d + ey)
            #      =       a * pv[1,1] * c  +      a * pv[1,1] * d * y
            #        + x * b * pv[1,1] * c  +  x * b * pv[1,1] * d * y
            #
            # So the single term initially will contribute to 4 different terms in the final
            # matrix.  And the contributions will just be pv[1,1] times the outer product
            # [a b]T [d e].  So if we can determine the matrix that converts from
            # [1, x, x^2, x^3] to the Chebyshev vector, the the matrix we want is simply
            # xmT pv ym.
            a = -(xmax+xmin)/(xmax-xmin)
            b = 2./(xmax-xmin)
            c = -(ymax+ymin)/(ymax-ymin)
            d = 2./(ymax-ymin)
            xm = np.zeros((4,4))
            ym = np.zeros((4,4))
            xm[0,0] = 1.
            xm[1,0] = a
            xm[1,1] = b
            ym[0,0] = 1.
            ym[1,0] = c
            ym[1,1] = d
            if code == 1:
                for m in range(2,4):
                    # The recursion rule is Pm = 2 x' Pm-1 - Pm-2
                    # Pm = 2 a Pm-1 - Pm-2 + x * 2 b Pm-1
                    xm[m] = 2. * a * xm[m-1] - xm[m-2]
                    xm[m,1:] += 2. * b * xm[m-1,:-1]
                    ym[m] = 2. * c * ym[m-1] - ym[m-2]
                    ym[m,1:] += 2. * d * ym[m-1,:-1]
            else:  # pragma: no cover
                # code == 2 means Legendre.  The same argument applies, but we have a
                # different recursion rule.
                # WARNING: This branch has not been tested!  I don't have any TNX files
                # with Legendre functions to test it on.  I think it's right, but beware!
                for m in range(2,4):
                    # The recursion rule is Pm = ((2m-1) x' Pm-1 - (m-1) Pm-2) / m
                    # Pm = ((2m-1) a Pm-1 - (m-1) Pm-2) / m
                    #      + x * ((2m-1) b Pm-1) / m
                    xm[m] = ((2.*m-1.) * a * xm[m-1] - (m-1.) * xm[m-2]) / m
                    xm[m,1:] += ((2.*m-1.) * b * xm[m-1,:-1]) / m
                    ym[m] = ((2.*m-1.) * c * ym[m-1] - (m-1.) * ym[m-2]) / m
                    ym[m,1:] += ((2.*m-1.) * d * ym[m-1,:-1]) / m

            pv2 = np.dot(xm.T , np.dot(pv, ym))
            return pv2

    def _apply_ab(self, x, y, ab):
        # Note: this is used for both pv and ab, since the action is the same.
        # They just occur at two different places in the calculation.
        x1 = horner2d(x, y, ab[0], triangle=True)
        y1 = horner2d(x, y, ab[1], triangle=True)
        return x1, y1

    def _apply_cd(self, x, y):
        # Do this in C++ layer for speed.
        nx = len(x.ravel())
        _x = x.__array_interface__['data'][0]
        _y = y.__array_interface__['data'][0]
        _cd = self.cd.__array_interface__['data'][0]
        _galsim.ApplyCD(nx, _x, _y, _cd)
        return x, y

    def _uv(self, x, y):
        # Most of the work for _radec.  But stop at (u,v).

        # Start with (u,v) = the image position
        x = np.ascontiguousarray(x, dtype=float)
        y = np.ascontiguousarray(y, dtype=float)

        x -= self.crpix[0]
        y -= self.crpix[1]

        if self.ab is not None:
            x, y = self._apply_ab(x, y, self.ab)

        # This converts to (u,v) in the tangent plane
        # Expanding this out is a bit faster than using np.dot for 2x2 matrix.
        u, v = self._apply_cd(x, y)

        if self.pv is not None:
            u, v = self._apply_ab(u, v, self.pv)

        # Convert (u,v) from degrees to radians
        # Also, the FITS standard defines u,v backwards relative to our standard.
        # They have +u increasing to the east, not west.  Hence the - for u.
        factor = 1. * degrees / radians
        u *= -factor
        v *= factor
        return u, v

    def _radec(self, x, y, color=None):
        # Get the position in the tangent plane
        u, v = self._uv(x, y)
        # Then convert from (u,v) to (ra, dec) using the appropriate projection.
        ra, dec = self.center.deproject_rad(u, v, projection=self.projection)

        if np.ndim(x) == np.ndim(y) == 0:
            return ra[0], dec[0]
        else:
            # Sanity checks that the inputs are the same shape.
            assert np.ndim(x) == np.ndim(y)
            assert x.shape == y.shape
            return ra, dec

    def _invert_ab(self, u, v, ab, abp=None):
        # This is used both for inverting (u,v) = PV (u',v')
        # and for inverting (x,y) = AB (x',y')
        # Here (and in C++) the notation is (u,v) = AB(x,y), even though both (u,v) and (x,y)
        # in this context are either in CCD coordinates (normally called x,y) or tangent plane
        # coordinates (normally called u,v).
        # abp is an optional set of coefficients to make a good guess for x,y

        uu = np.ascontiguousarray(u)  # Don't overwrite the given u,v, since we need it at the end
        vv = np.ascontiguousarray(v)  # to check it we were provided scalars or arrays.

        x = np.atleast_1d(u.copy())    # Start with x,y = u,v.
        y = np.atleast_1d(v.copy())    # This may be updated below if abp is provided.

        nab = ab.shape[1]
        nabp = abp.shape[1] if abp is not None else 0
        nx = len(x.ravel())
        _uu = uu.__array_interface__['data'][0]
        _vv = vv.__array_interface__['data'][0]
        _x = x.__array_interface__['data'][0]
        _y = y.__array_interface__['data'][0]
        _ab = ab.__array_interface__['data'][0]
        _abp = 0 if abp is None else abp.__array_interface__['data'][0]
        with convert_cpp_errors():
            _galsim.InvertAB(nx, nab, _uu, _vv, _ab, _x, _y, self._doiter, nabp, _abp)

        # Return the right type for u,v
        try:
            len(u)
        except TypeError:
            return x[0], y[0]
        else:
            return x, y


    def _xy(self, ra, dec, color=None):
        u, v = self.center.project_rad(ra, dec, projection=self.projection)

        # Again, FITS has +u increasing to the east, not west.  Hence the - for u.
        factor = radians / degrees
        u *= -factor
        v *= factor

        if self.pv is not None:
            u, v = self._invert_ab(u, v, self.pv)

        if not hasattr(self, 'cdinv'):
            self.cdinv = np.linalg.inv(self.cd)
        # This is a bit faster than using np.dot for 2x2 matrix.
        x = self.cdinv[0,0] * u + self.cdinv[0,1] * v
        y = self.cdinv[1,0] * u + self.cdinv[1,1] * v

        if self.ab is not None:
            x, y = self._invert_ab(x, y, self.ab, abp=self.abp)

        x += self.crpix[0]
        y += self.crpix[1]

        return x, y

    # Override the version in CelestialWCS, since we can do this more efficiently.
    def _local(self, image_pos, color=None):
        from .wcs import JacobianWCS

        if image_pos is None:
            raise TypeError("origin must be a PositionD or PositionI argument")

        # The key lemma here is that chain rule for jacobians is just matrix multiplication.
        # i.e. if s = s(u,v), t = t(u,v) and u = u(x,y), v = v(x,y), then
        # ( dsdx  dsdy ) = ( dsdu dudx + dsdv dvdx   dsdu dudy + dsdv dvdy )
        # ( dtdx  dtdy ) = ( dtdu dudx + dtdv dvdx   dtdu dudy + dtdv dvdy )
        #                = ( dsdu  dsdv )  ( dudx  dudy )
        #                  ( dtdu  dtdv )  ( dvdx  dvdy )
        #
        # So if we can find the jacobian for each step of the process, we just multiply the
        # jacobians.
        #
        # We also need to keep track of the position along the way, so we have to repeat many
        # of the steps in _radec.

        p1 = np.array([image_pos.x, image_pos.y], dtype=float)

        # Start with unit jacobian
        jac = np.diag([1,1])

        # No effect on the jacobian from this step.
        p1 -= self.crpix

        if self.ab is not None:
            x = p1[0]
            y = p1[1]
            order = len(self.ab[0])-1
            xpow = x ** np.arange(order+1)
            ypow = y ** np.arange(order+1)
            p1 = np.dot(np.dot(self.ab, ypow), xpow)

            dxpow = np.zeros(order+1)
            dypow = np.zeros(order+1)
            dxpow[1:] = (np.arange(order)+1.) * xpow[:-1]
            dypow[1:] = (np.arange(order)+1.) * ypow[:-1]
            j1 = np.transpose([ np.dot(np.dot(self.ab, ypow), dxpow) ,
                                np.dot(np.dot(self.ab, dypow), xpow) ])
            jac = np.dot(j1,jac)

        # The jacobian here is just the cd matrix.
        p2 = np.dot(self.cd, p1)
        jac = np.dot(self.cd, jac)

        if self.pv is not None:
            # Now we apply the distortion terms
            u = p2[0]
            v = p2[1]
            order = len(self.pv[0])-1

            upow = u ** np.arange(order+1)
            vpow = v ** np.arange(order+1)

            p2 = np.dot(np.dot(self.pv, vpow), upow)

            # The columns of the jacobian for this step are the same function with dupow
            # or dvpow.
            dupow = np.zeros(order+1)
            dvpow = np.zeros(order+1)
            dupow[1:] = (np.arange(order)+1.) * upow[:-1]
            dvpow[1:] = (np.arange(order)+1.) * vpow[:-1]
            j1 = np.transpose([ np.dot(np.dot(self.pv, vpow), dupow) ,
                                np.dot(np.dot(self.pv, dvpow), upow) ])
            jac = np.dot(j1,jac)

        unit_convert = [ -1 * degrees / radians, 1 * degrees / radians ]
        p2 *= unit_convert
        # Subtle point: Don't use jac *= ..., because jac might currently be self.cd, and
        #               that would change self.cd!
        jac = jac * np.transpose( [ unit_convert ] )

        # Finally convert from (u,v) to (ra, dec).  We have a special function that computes
        # the jacobian of this step in the CelestialCoord class.
        j2 = self.center.jac_deproject_rad(p2[0], p2[1], projection=self.projection)
        jac = np.dot(j2,jac)

        # This now has units of radians/pixel.  We want instead arcsec/pixel.
        jac *= radians / arcsec

        return JacobianWCS(jac[0,0], jac[0,1], jac[1,0], jac[1,1])


    def _newOrigin(self, origin):
        ret = self.copy()
        ret.crpix = ret.crpix + [ origin.x, origin.y ]
        return ret

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("GSFitsWCS", "GalSim WCS name")
        header["CTYPE1"] = 'RA---' + self.wcs_type
        header["CTYPE2"] = 'DEC--' + self.wcs_type
        header["CRPIX1"] = self.crpix[0]
        header["CRPIX2"] = self.crpix[1]
        header["CD1_1"] = self.cd[0][0]
        header["CD1_2"] = self.cd[0][1]
        header["CD2_1"] = self.cd[1][0]
        header["CD2_2"] = self.cd[1][1]
        header["CUNIT1"] = 'deg'
        header["CUNIT2"] = 'deg'
        header["CRVAL1"] = self.center.ra / degrees
        header["CRVAL2"] = self.center.dec / degrees
        if self.pv is not None:
            order = len(self.pv[0])-1
            k = 0
            odd_indices = [3, 11, 23, 39]
            for n in range(order+1):
                for j in range(n+1):
                    i = n-j
                    header["PV1_" + str(k)] = self.pv[0, i, j]
                    header["PV2_" + str(k)] = self.pv[1, j, i]
                    k = k + 1
                    if k in odd_indices: k = k + 1
        if self.ab is not None:
            order = len(self.ab[0])-1
            header["A_ORDER"] = order
            for i in range(order+1):
                for j in range(order+1):
                    aij = self.ab[0,i,j]
                    if i==1 and j==0: aij -= 1  # Turn back into standard form.
                    if aij != 0.:
                        header["A_"+str(i)+"_"+str(j)] = aij
            header["B_ORDER"] = order
            for i in range(order+1):
                for j in range(order+1):
                    bij = self.ab[1,i,j]
                    if i==0 and j==1: bij -= 1
                    if bij != 0.:
                        header["B_"+str(i)+"_"+str(j)] = bij
        if self.abp is not None:
            order = len(self.abp[0])-1
            header["AP_ORDER"] = order
            for i in range(order+1):
                for j in range(order+1):
                    apij = self.abp[0,i,j]
                    if i==1 and j==0: apij -= 1
                    if apij != 0.:
                        header["AP_"+str(i)+"_"+str(j)] = apij
            header["BP_ORDER"] = order
            for i in range(order+1):
                for j in range(order+1):
                    bpij = self.abp[1,i,j]
                    if i==0 and j==1: bpij -= 1
                    if bpij != 0.:
                        header["BP_"+str(i)+"_"+str(j)] = bpij
        return header

    @staticmethod
    def _readHeader(header):
        return GSFitsWCS(header=header)

    def copy(self):
        # The copy module version of copying the dict works fine here.
        import copy
        return copy.copy(self)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, GSFitsWCS) and
                 self.wcs_type == other.wcs_type and
                 np.array_equal(self.crpix,other.crpix) and
                 np.array_equal(self.cd,other.cd) and
                 np.array_equal(self.cd,other.cd) and
                 self.center == other.center and
                 np.array_equal(self.pv,other.pv) and
                 np.array_equal(self.ab,other.ab) and
                 np.array_equal(self.abp,other.abp)))

    def __repr__(self):
        if self.pv is None:
            pv_repr = repr(self.pv)
        else:
            pv_repr = 'array(%r)'%self.pv.tolist()
        if self.ab is None:
            ab_repr = repr(self.ab)
        else:
            ab_repr = 'array(%r)'%self.ab.tolist()
        if self.abp is None:
            abp_repr = repr(self.abp)
        else:
            abp_repr = 'array(%r)'%self.abp.tolist()
        return "galsim.GSFitsWCS(_data = [%r, array(%r), array(%r), %r, %s, %s, %s])"%(
                self.wcs_type, self.crpix.tolist(), self.cd.tolist(), self.center,
                pv_repr, ab_repr, abp_repr)

    def __str__(self):
        if self._tag is None:
            return self.__repr__()
        else:
            return "galsim.GSFitsWCS(%s)"%(self._tag)

    def __hash__(self): return hash(repr(self))


def TanWCS(affine, world_origin, units=arcsec):
    """This is a function that returns a `GSFitsWCS` object for a TAN WCS projection.

    The TAN projection is essentially an affine transformation from image coordinates to
    Euclidean (u,v) coordinates on a tangent plane, and then a "deprojection" of this plane
    onto the sphere given a particular RA, Dec for the location of the tangent point.
    The tangent point will correspond to the location of (u,v) = (0,0) in the intermediate
    coordinate system.

    Parameters:
        affine:          An `AffineTransform` defining the transformation from image coordinates
                         to the coordinates on the tangent plane.
        world_origin:    A `CelestialCoord` defining the location on the sphere where the
                         tangent plane is centered.
        units:           The angular units of the (u,v) intermediate coordinate system.
                         [default: galsim.arcsec]

    Returns:
        a `GSFitsWCS` describing this WCS.
    """
    # These will raise the appropriate errors if affine is not the right type.
    dudx = affine.dudx * units / degrees
    dudy = affine.dudy * units / degrees
    dvdx = affine.dvdx * units / degrees
    dvdy = affine.dvdy * units / degrees
    origin = affine.origin
    # The - signs are because the Fits standard is in terms of +u going east, rather than west
    # as we have defined.  So just switch the sign in the CD matrix.
    cd = np.array([[ -dudx, -dudy ], [ dvdx, dvdy ]], dtype=float)
    crpix = np.array([ origin.x, origin.y ], dtype=float)

    # We also need to absorb the affine world_origin back into crpix, since GSFits is expecting
    # crpix to be the location of the tangent point in image coordinates. i.e. where (u,v) = (0,0)
    # (u,v) = CD * (x-x0,y-y0) + (u0,v0)
    # (0,0) = CD * (x0',y0') - CD * (x0,y0) + (u0,v0)
    # CD (x0',y0') = CD (x0,y0) - (u0,v0)
    # (x0',y0') = (x0,y0) - CD^-1 (u0,v0)
    uv = np.array( [ affine.world_origin.x * units / degrees,
                     affine.world_origin.y * units / degrees ] )
    crpix -= np.dot(np.linalg.inv(cd) , uv)

    # Invoke the private constructor of GSFits using the _data kwarg.
    data = ('TAN', crpix, cd, world_origin, None, None, None)
    return GSFitsWCS(_data=data)


# This is a list of all the WCS types that can potentially read a WCS from a FITS file.
# The function FitsWCS will try each of these in order and return the first one that
# succeeds.  AffineTransform should be last, since it will always succeed.
# The list is defined here at global scope so that external modules can add extra
# WCS types to the list if desired.

fits_wcs_types = [

    GSFitsWCS,      # This doesn't work for very many WCS types, but it works for the very common
                    # TAN projection, and also TPV, which is used by SCamp.  If it does work, it
                    # is a good choice, since it is easily the fastest of any of these.

    PyAstWCS,       # This requires ``import starlink.Ast`` to succeed.  This handles the largest
                    # number of WCS types of any of these.  In fact, it worked for every one
                    # we tried in our unit tests (which was not exhaustive).

    AstropyWCS,     # This requires ``import astropy.wcs`` to succeed.  It doesn't support quite as
                    # many WCS types as PyAst.  It's also usually a little slower, so we prefer
                    # PyAstWCS when it is available.

    WcsToolsWCS,    # This requires the wcstool command line functions to be installed.
                    # It is very slow, so it should only be used as a last resort.

]


def FitsWCS(file_name=None, dir=None, hdu=None, header=None, compression='auto',
            text_file=False, suppress_warning=False):
    """This factory function will try to read the WCS from a FITS file and return a WCS that will
    work.  It tries a number of different WCS classes until it finds one that succeeds in reading
    the file.

    If none of them work, then the last class it tries, `AffineTransform`, is guaranteed to succeed,
    but it will only model the linear portion of the WCS (the CD matrix, CRPIX, and CRVAL), using
    reasonable defaults if even these are missing.  If you think that you have the right software
    for one of the WCS types, but FitsWCS still defaults to `AffineTransform`, it may be helpful to
    update your installation of astropy and/or starlink (if you don't already have the latest
    version).

    Note: The list of classes this function will try may be edited, e.g. by an external module
    that wants to add an additional WCS type.  The list is ``galsim.fitswcs.fits_wcs_types``.

    Parameters:
        file_name:        The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
        dir:              Optional directory to prepend to ``file_name``. [default: None]
        hdu:              Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as
                          appropriate for the given compression.  (e.g. for rice, the first
                          extension is the one you normally want.) [default: None]
        header:           The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a FitsHeader object.  [default: None]
        compression:      Which decompression scheme to use (if any). See galsim.fits.read()
                          for the available options.  [default: 'auto']
        text_file:        Normally a file is taken to be a fits file, but you can also give it a
                          text file with the header information (like the .head file output from
                          SCamp).  In this case you should set ``text_file = True`` to tell GalSim
                          to parse the file this way.  [default: False]
        suppress_warning: Whether to suppress a warning that the WCS could not be read from the
                          FITS header, so the WCS defaulted to either a `PixelScale` or
                          `AffineTransform`. [default: False]
                          (Note: this is (by default) set to True when this function is implicitly
                          called from one of the galsim.fits.read* functions.)
    """
    from .wcs import AffineTransform, PixelScale, OffsetWCS

    if file_name is not None:
        if header is not None:
            raise GalSimIncompatibleValuesError(
                "Cannot provide both file_name and pyfits header",
                file_name=file_name, header=header)
        header = fits.FitsHeader(file_name=file_name, dir=dir, hdu=hdu, compression=compression,
                                 text_file=text_file)
    else:
        file_name = 'header' # For sensible error messages below.
    if header is None:
        raise GalSimIncompatibleValuesError(
            "Must provide either file_name or header", file_name=file_name, header=header)
    if not isinstance(header, fits.FitsHeader):
        header = fits.FitsHeader(header)

    if 'CTYPE1' not in header and 'CDELT1' not in header:
        if not suppress_warning:
            galsim_warn("No WCS information found in %r. Defaulting to PixelScale(1.0)"%(file_name))
        return PixelScale(1.0)

    # For linear WCS specifications, AffineTransformation should work.
    # Note: Most files will have CTYPE1,2, but old style with only CDELT1,2 sometimes omits it.
    if header.get('CTYPE1', 'LINEAR') == 'LINEAR':
        wcs = AffineTransform._readHeader(header)
        # Convert to PixelScale if possible.
        if (wcs.dudx == wcs.dvdy and wcs.dudy == wcs.dvdx == 0):
            if wcs.x0 == wcs.y0 == wcs.u0 == wcs.v0 == 0:
                wcs = PixelScale(wcs.dudx)
            else:
                wcs = OffsetWCS(wcs.dudx, wcs.origin, wcs.world_origin)
        return wcs

    # Otherwise (and typically), try the various wcs types that can read celestial coordinates.
    for wcs_type in fits_wcs_types:
        try:
            wcs = wcs_type._readHeader(header)
            # Give it a better tag for the repr if appropriate.
            if hasattr(wcs,'_tag') and file_name != 'header':
                if dir is not None:
                    import os
                    wcs._tag = repr(os.path.join(dir,file_name))
                else:
                    wcs._tag = repr(file_name)
                if hdu is not None:
                    wcs._tag += ', hdu=%r'%hdu
                if compression != 'auto':
                    wcs._tag += ', compression=%r'%compression
            return wcs
        except KeyboardInterrupt:
            raise
        except Exception as err:
            pass
    else:
        # Finally, this one is really the last resort, since it only reads in the linear part of the
        # WCS.  It defaults to the equivalent of a pixel scale of 1.0 if even these are not present.
        if not suppress_warning:
            galsim_warn("All the fits WCS types failed to read %r. Using AffineTransform "
                        "instead, which will not really be correct."%(file_name))
        return AffineTransform._readHeader(header)

# Let this function work like a class in config.
FitsWCS._req_params = { "file_name" : str }
FitsWCS._opt_params = { "dir" : str, "hdu" : int, "compression" : str, 'text_file' : bool }

def FittedSIPWCS(x, y, ra, dec, wcs_type='TAN', order=3, center=None):
    """A WCS constructed from a list of reference celestial and image
    coordinates.

    Parameters:
        x:         Image x-coordinates of reference stars in pixels
        y:         Image y-coordinates of reference stars in pixels
        ra:        Right ascension of reference stars in radians
        dec:       Declination of reference stars in radians
        wcs_type:  The type of the tangent plane projection to use.  Should be
                   one of ['TAN', 'STG', 'ZEA', or 'ARC'].  [default: 'TAN']
        order:     The order of the Simple Imaging Polynomial (SIP) used to
                   describe the WCS distortion.  SIP coefficients kick in when
                   order >= 2.  If you supply order=1, then just fit a WCS
                   without any SIP coefficients.  [default: 3]
        center:    A `CelestialCoord` defining the location on the sphere where
                   the tangent plane is centered.  [default: None, which means
                   use the average position of the list of reference stars]
    """
    from scipy.optimize import least_squares
    from .celestial import CelestialCoord

    if order < 1:
        raise GalSimValueError("Illegal SIP order", order)

    nstar = len(x)
    # Make sure we have enough stars.
    # We need 1 star for crpix, 2 more for cd, and then
    # (order+2)*(order+1)/2 - 3 for ab.  The total is then (order+1)*(order+2)/2
    nrequire = (order+1)*(order+2)/2
    if nstar < nrequire:
        raise GalSimError(
            "Require at least {:0} stars for SIP order {:1}"
            .format(nrequire, order)
        )

    if center is None:
        # Use deprojected 3D mean of ra/dec unit sphere points as center
        wx = np.mean(np.cos(dec)*np.cos(ra))
        wy = np.mean(np.cos(dec)*np.sin(ra))
        wz = np.mean(np.sin(dec))
        center = CelestialCoord.from_xyz(wx, wy, wz)

    # Project radec onto uv so we can linearly fit the CRPIX and CD matrix
    # initial guesses
    u, v = center.project_rad(ra, dec)
    a = np.array(np.broadcast_arrays(1., -u, v)).T
    b = np.array([x, y]).T
    r, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    crpix_guess = r[0]
    cd_guess = np.linalg.inv(np.deg2rad(r[1:]))

    # SIP coefficient initial guesses are just 0.0
    ab_guess = []
    for i in range(order+1):
        for j in range(order+1):
            if (i+j > 1) and (i+j <= order):
                ab_guess.extend([0.0, 0.0])
    ab_guess = np.array(ab_guess)
    guess = np.hstack([crpix_guess, cd_guess.ravel(), ab_guess.ravel()])

    def _getWCS(wcs_type, center, crpix, cd, ab=None, abp=None, doiter=False):
        _data = [
            wcs_type if order == 1 else wcs_type+'-SIP',
            crpix,
            cd,
            center,
            None, # pv, unused
            ab,
            abp
        ]
        return GSFitsWCS(_data=_data, _doiter=doiter)

    def _decodeSIP(order, params, min=2):
        if order == 1:
            return None
        k = 0
        a = []
        b = []
        for i in range(order+1):
            for j in range(order+1):
                if (i+j < min) or (i+j > order):
                    a.append(0.0)
                    b.append(0.0)
                else:
                    a.append(params[k])
                    b.append(params[k+1])
                    k += 2
        a = np.array(a).reshape((order+1, order+1))
        b = np.array(b).reshape((order+1, order+1))
        a[1,0] += 1  # GSFitsWCS wants these with the identity included.
        b[0,1] += 1
        ab = np.array([a, b])
        return ab

    def _abLoss(params, order, wcs_type, center, x, y, u, v):
        crpix = params[:2]
        cd = params[2:6].reshape(2, 2)
        ab = _decodeSIP(order, params[6:])
        wcs = _getWCS(wcs_type, center, crpix, cd, ab)
        ra_p, dec_p = wcs.xyToradec(x, y, units='rad')
        u_p, v_p = center.project_rad(ra_p, dec_p)
        resid = np.hstack([u-u_p, v-v_p])
        resid = np.rad2deg(resid)*3600*1e6  # Work in microarcseconds
        return resid

    result = least_squares(
        _abLoss,
        guess,
        args=(order, wcs_type, center, x, y, u, v)
    )
    # rmse = np.sqrt(2*result.cost/len(u))
    # print(f"rmse: {rmse:.2f} microarcsec")
    crpix = result.x[0:2]
    cd = result.x[2:6].reshape(2, 2)
    ab = _decodeSIP(order, result.x[6:])

    if order == 1:
        return _getWCS(wcs_type, center, crpix, cd)

    # Now go back holding crpix, cd, and ab constant, and solve for inverse
    # coefficients abp
    # ABP SIP coefficient initial guesses are just 0.0
    abp_guess = []
    for i in range(order+1):
        for j in range(order+1):
            if (i+j > 0) and (i+j <= order):
                abp_guess.extend([0.0, 0.0])
    abp_guess = np.array(abp_guess)

    def _abpLoss(params, order, wcs_type, center, crpix, cd, ab, ra, dec, x, y):
        abp = _decodeSIP(order, params, min=1)
        wcs = _getWCS(wcs_type, center, crpix, cd, ab, abp)
        x_p, y_p = wcs.radecToxy(ra, dec, units='rad')
        resid = np.hstack([x-x_p, y-y_p])*1e6  # work in micropixels
        return resid
    result = least_squares(
        _abpLoss, abp_guess,
        args=(order, wcs_type, center, crpix, cd, ab, ra, dec, x, y)
    )
    # rmse = np.sqrt(2*result.cost/len(u))
    # print(f"rmse: {rmse:.2f} micropixels")
    abp = _decodeSIP(order, result.x, min=1)
    return _getWCS(wcs_type, center, crpix, cd, ab, abp, doiter=True)
