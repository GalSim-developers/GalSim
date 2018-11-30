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
"""@file fitswcs.py
The function FitsWCS() acts like a BaseWCS class, but it really calls one of several other
classes depending on what python modules are available and what kind of FITS file you are
trying to read.
"""

import warnings
import numpy as np

from .wcs import CelestialWCS
from .position import PositionD, PositionI
from .angle import radians, arcsec, degrees, AngleUnit
from . import _galsim
from . import fits
from .errors import GalSimError, GalSimValueError, GalSimIncompatibleValuesError
from .errors import GalSimNotImplementedError, convert_cpp_errors, galsim_warn

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

    Astropy may be installed using pip, fink, or port:

        >>> pip install astropy
        >>> fink install astropy-py27
        >>> port install py27-astropy

    It also comes by default with Enthought and Anaconda. For more information, see their website:

        http://www.astropy.org/

    Initialization
    --------------
    An AstropyWCS is initialized with one of the following commands:

        >>> wcs = galsim.AstropyWCS(file_name=file_name)  # Open a file on disk
        >>> wcs = galsim.AstropyWCS(header=header)        # Use an existing pyfits header
        >>> wcs = galsim.AstropyWCS(wcs=wcs)              # Use an existing astropy.wcs.WCS instance

    Exactly one of the parameters `file_name`, `header` or `wcs` is required.  Also, since the most
    common usage will probably be the first, you can also give a `file_name` without it being named:

        >>> wcs = galsim.AstropyWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
    @param dir            Optional directory to prepend to `file_name`. [default: None]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as
                          appropriate for the given compression.  (e.g. for rice, the first
                          extension is the one you normally want.) [default: None]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a FitsHeader object.  [default: None]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read()
                          for the available options.  [default: 'auto']
    @param wcs            An existing astropy.wcs.WCS instance [default: None]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PositionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False

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
            if compression is not 'auto':
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
                except (TypeError, AttributeError, ValueError, RuntimeError) as e:
                    # When parsing ZPX files, astropy raises a very unhelpful error message.
                    # Ignore that (ValueError in that case, but ignore any similarly mundane error)
                    # and turn it into a more appropriate OSError.
                    raise OSError("Astropy failed to read WCS from %s. Original error: %s"%(
                                  file_name, e))
                else:
                    # New kind of error starting in astropy 2.0.5 (I think).  Sometimes, it
                    # gets through the above, but doesn't actually load the right WCS.
                    # E.g. ZPX gets marked as just a ZPN.
                    if 'CTYPE1' in header and 'CTYPE2' in header:
                        if (header['CTYPE1'] != wcs.wcs.ctype[0] or
                            header['CTYPE2'] != wcs.wcs.ctype[1]):
                            raise OSError("Astropy failed to read WCS from %s. Converted %s->%s"%(
                                          file_name, (header['CTYPE1'], header['CTYPE2']),
                                          wcs.wcs.ctype))
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
        from . import fits
        with warnings.catch_warnings():
            # The constructor might emit warnings if it wants to fix the header
            # information (e.g. RADECSYS -> RADESYSa).  We'd rather ignore these
            # warnings, since we don't much care if the input file is non-standard
            # so long as we can make it work.
            warnings.simplefilter("ignore")
            wcs = astropy.wcs.WCS(header.header)
        return wcs

    @property
    def wcs(self): return self._wcs

    @property
    def origin(self): return self._origin

    def _radec(self, x, y, color=None):
        x1 = np.atleast_1d(x)
        y1 = np.atleast_1d(y)

        ra, dec = self.wcs.all_pix2world(x1, y1, 1, ra_dec_order=True)

        # astropy outputs ra, dec in degrees.  Need to convert to radians.
        factor = degrees / radians
        ra *= factor
        dec *= factor

        try:
            # If the inputs were numpy arrays, return the same
            len(x)
        except TypeError:
            # Otherwise, return scalars
            #assert len(ra) == 1
            #assert len(dec) == 1
            ra = ra[0]
            dec = dec[0]
        return ra, dec

    def _xy(self, ra, dec, color=None):
        import astropy
        factor = radians / degrees
        rd = np.atleast_2d([ra, dec]) * factor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xy = self.wcs.all_world2pix(rd, 1, ra_dec_order=True)[0]
        x, y = xy
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
        return AstropyWCS(header=header, origin=PositionD(x0,y0))

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
        import galsim
        self.__dict__ = d
        self._wcs = self._load_from_header(self.header)


class PyAstWCS(CelestialWCS):
    """This WCS uses PyAst (the python front end for the Starlink AST code) to read WCS
    information from a FITS file.  It requires the starlink.Ast python module to be installed.

    Starlink may be installed using pip:

        >>> pip install starlink-pyast

    For more information, see their website:

        https://pypi.python.org/pypi/starlink-pyast/

    Initialization
    --------------
    A PyAstWCS is initialized with one of the following commands:

        >>> wcs = galsim.PyAstWCS(file_name=file_name)  # Open a file on disk
        >>> wcs = galsim.PyAstWCS(header=header)        # Use an existing pyfits header
        >>> wcs = galsim.PyAstWCS(wcsinfo=wcsinfo)      # Use an existing starlink.Ast.FrameSet

    Exactly one of the parameters `file_name`, `header` or `wcsinfo` is required.  Also, since the
    most common usage will probably be the first, you can also give a file name without it being
    named:

        >>> wcs = galsim.PyAstWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
    @param dir            Optional directory to prepend to `file_name`. [default: None]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as
                          appropriate for the given compression.  (e.g. for rice, the first
                          extension is the one you normally want.) [default: None]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a FitsHeader object.  [default: None]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read()
                          for the available options.  [default:'auto']
    @param wcsinfo        An existing starlink.Ast.FrameSet [default: None]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PositionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False

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
            if compression is not 'auto':
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
    def wcsinfo(self): return self._wcsinfo

    @property
    def origin(self): return self._origin

    def _fix_header(self, header):
        # We allow for the option to fix up the header information when a modification can
        # make it readable by PyAst.

        # There was an older proposed standard that used TAN with PV values, which is used by
        # SCamp, so we want to support it if possible.  The standard is now called TPV, which
        # PyAst understands.  All we need to do is change the names of the CTYPE values.
        if ( 'CTYPE1' in header and header['CTYPE1'].endswith('TAN') and
             'CTYPE2' in header and header['CTYPE2'].endswith('TAN') and
             'PV1_10' in header ):
            header['CTYPE1'] = header['CTYPE1'].replace('TAN','TPV')
            header['CTYPE2'] = header['CTYPE2'].replace('TAN','TPV')

    def _radec(self, x, y, color=None):
        # Need this to look like
        #    [ [ x1, x2, x3... ], [ y1, y2, y3... ] ]
        # if input is either scalar x,y or two arrays.
        xy = np.array([np.atleast_1d(x), np.atleast_1d(y)])

        ra, dec = self.wcsinfo.tran( xy )
        # PyAst returns ra, dec in radians, so we're good.

        try:
            len(x)
        except TypeError:
            # If the inputs weren't numpy arrays, return scalars
            #assert len(ra) == 1
            #assert len(dec) == 1
            ra = ra[0]
            dec = dec[0]
        return ra, dec

    def _xy(self, ra, dec, color=None):
        rd = np.array([np.atleast_1d(ra), np.atleast_1d(dec)])
        x, y = self.wcsinfo.tran( rd, False )
        return x[0], y[0]

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
        return PyAstWCS(header=header, origin=PositionD(x0,y0))

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
        import galsim
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

    Initialization
    --------------
    A WcsToolsWCS is initialized with the following command:

        >>> wcs = galsim.WcsToolsWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.
    @param dir            Optional directory to prepend to `file_name`. [default: None]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PositionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "origin" : PositionD }
    _single_params = []
    _takes_rng = False

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
    def file_name(self): return self._file_name

    @property
    def origin(self): return self._origin

    def _radec(self, x, y, color=None):
        # Need this to look like
        #    [ x1, y1, x2, y2, ... ]
        # if input is either scalar x,y or two arrays.
        xy = np.array([x, y]).transpose().ravel()

        # The OS cannot handle arbitrarily long command lines, so we may need to split up
        # the list into smaller chunks.
        import os
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
            import subprocess
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

        try:
            len(x)
            # If the inputs were numpy arrays, return the same
            return np.array(ra)*factor, np.array(dec)*factor
        except TypeError:
            # Otherwise return scalars
            #assert len(ra) == 1
            #assert len(dec) == 1
            return ra[0]*factor, dec[0]*factor

    def _xy(self, ra, dec, color=None):
        import subprocess
        rd = np.array([ra, dec])
        rd *= radians / degrees
        for digits in range(10,5,-1):
            rd_strs = [ str(z) for z in rd ]
            p = subprocess.Popen(['sky2xy', '-n', str(digits), self._file_name] + rd_strs,
                                 stdout=subprocess.PIPE)
            results = p.communicate()[0].decode()
            p.stdout.close()
            if len(results) == 0:
                raise OSError('wcstools (specifically sky2xy) was unable to read '+self._file_name)
            if results[0] != '*': break
        if results[0] == '*':
            raise OSError('wcstools (specifically sky2xy) was unable to read '+self._file_name)

        # The output should looke like:
        #    ra dec J2000 -> x y
        # However, if there was an error, the J200 might be missing.
        vals = results.split()
        if len(vals) < 6:
            raise GalSimError('wcstools sky2xy returned invalid result for %f,%f'%(ra,dec))
        if len(vals) > 6:
            galsim_warn("wcstools sky2xy indicates that %f,%f is off the image. "
                        "output is %r"%(ra,dec,results))
        x = float(vals[4])
        y = float(vals[5])

        return x, y

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
        return WcsToolsWCS(file, origin=PositionD(x0,y0))

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

    Initialization
    --------------
    A GSFitsWCS is initialized with one of the following commands:

        >>> wcs = galsim.GSFitsWCS(file_name=file_name)  # Open a file on disk
        >>> wcs = galsim.GSFitsWCS(header=header)        # Use an existing pyfits header

    Also, since the most common usage will probably be the first, you can also give a file name
    without it being named:

        >>> wcs = galsim.GSFitsWCS(file_name)

    In addition to reading from a FITS file, there is also a factory function that builds
    a GSFitsWCS object implementing a TAN projection.  See the docstring of TanWCS() for
    more details.

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
    @param dir            Optional directory to prepend to `file_name`. [default: None]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as
                          appropriate for the given compression.  (e.g. for rice, the first
                          extension is the one you normally want.) [default: None]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a FitsHeader object.  [default: None]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read()
                          for the available options.  [default: 'auto']
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PositionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 origin=None, _data=None):
        # Note: _data is not intended for end-user use.  It enables the equivalent of a
        #       private constructor of GSFitsWCS by the function TanWCS.  The details of its
        #       use are intentionally not documented above.

        self._color = None
        self._tag = None # Write something useful here (see below). This is just used for the str.

        # If _data is given, copy the data and we're done.
        if _data is not None:
            self.wcs_type = _data[0]
            self.crpix = _data[1]
            self.cd = _data[2]
            self.center = _data[3]
            self.pv = _data[4]
            self.ab = _data[5]
            self.abp = _data[6]
            if self.wcs_type in ('TAN', 'TPV'):
                self.projection = 'gnomonic'
            elif self.wcs_type == 'STG':
                self.projection = 'stereographic'
            elif self.wcs_type == 'ZEA':
                self.projection = 'lambert'
            elif self.wcs_type == 'ARC':
                self.projection = 'postel'
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
            if compression is not 'auto':
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
    # withOrigin to get the current origin value.  We don't use it in this class, though, so
    # just make origin a dummy property that returns 0,0.
    @property
    def origin(self): return PositionD(0.,0.)

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
        elif self.wcs_type == 'STG':
            self.projection = 'stereographic'
        elif self.wcs_type == 'ZEA':
            self.projection = 'lambert'
        elif self.wcs_type == 'ARC':
            self.projection = 'postel'
        else:
            raise GalSimValueError("GSFitsWCS cannot read files using given wcs_type.",
                                   self.wcs_type,
                                   ('TAN', 'TPV', 'TNX', 'TAN-SIP', 'STG', 'ZEA', 'ARC'))
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
        if self.wcs_type == 'TAN' and 'PV1_10' in header:
            self.wcs_type = 'TPV'

        self.pv = None
        self.ab = None
        self.abp = None
        if self.wcs_type == 'TPV':
            self._read_tpv(header)
        elif self.wcs_type == 'TNX':
            self._read_tnx(header)
        elif self.wcs_type == 'TAN-SIP':
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

        # Strangely, the PV values skip k==3.
        # Well, the reason is that it is for coefficients of r = sqrt(u^2+v^2).
        # But no one seems to use it, so it is almost always skipped.
        pv1 = [ float(header['PV1_'+str(k)]) for k in range(11) if k != 3 ]
        pv2 = [ float(header['PV2_'+str(k)]) for k in range(11) if k != 3 ]

        # In fact, the standard allows up to PVi_39, which is for r^7.  And all
        # unlisted values have defaults of 0 (except PVi_1, which defaults to 1).
        # A better implementation would check how high up the numbers go and build
        # the appropriate matrix, but since it is usually up to PVi_10, so far
        # we just implement that.
        # See http://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html for details.
        if ( 'PV1_3' in header and header['PV1_3'] != 0.0 or
             'PV1_11' in header and header['PV1_11'] != 0.0 or
             'PV2_3' in header and header['PV1_3'] != 0.0 or
             'PV2_11' in header and header['PV1_11'] != 0.0 ): # pragma: no cover
            raise GalSimNotImplementedError("TPV not implemented for odd powers of r")
        if 'PV1_12' in header: # pragma: no cover
            raise GalSimNotImplementedError("TPV not implemented past 3rd order terms")

        # Another strange thing is that the two matrices are defined in the opposite order
        # with respect to their element ordering.  And remember that we skipped k=3 in the
        # original reading, so indices 3..9 here were originally called PVi_4..10
        self.pv = np.array( [ [ [ pv1[0], pv1[2], pv1[5], pv1[9] ],
                                [ pv1[1], pv1[4], pv1[8],   0.   ],
                                [ pv1[3], pv1[7],   0.  ,   0.   ],
                                [ pv1[6],   0.  ,   0.  ,   0.   ] ],
                              [ [ pv2[0], pv2[1], pv2[3], pv2[6] ],
                                [ pv2[2], pv2[4], pv2[7],   0.   ],
                                [ pv2[5], pv2[8],   0.  ,   0.   ],
                                [ pv2[9],   0.  ,   0.  ,   0.   ] ] ] )

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

    def _apply_pv(self, u, v):
        # Do this in C++ layer for speed.
        with convert_cpp_errors():
            _galsim.ApplyPV(len(u), 4, u.ctypes.data, v.ctypes.data, self.pv.ctypes.data)
        return u, v

    def _apply_ab(self, x, y):
        # Do this in C++ layer for speed.
        dx = x.copy()
        dy = y.copy()
        with convert_cpp_errors():
            _galsim.ApplyPV(len(x), len(self.ab[0]), dx.ctypes.data, dy.ctypes.data,
                            self.ab.ctypes.data)
        return x+dx, y+dy

    def _apply_cd(self, x, y):
        # Do this in C++ layer for speed.
        with convert_cpp_errors():
            _galsim.ApplyCD(len(x), x.ctypes.data, y.ctypes.data, self.cd.ctypes.data)
        return x, y

    def _uv(self, x, y):
        # Most of the work for _radec.  But stop at (u,v).

        # Start with (u,v) = the image position
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        x -= self.crpix[0]
        y -= self.crpix[1]

        if self.ab is not None:
            x, y = self._apply_ab(x, y)

        # This converts to (u,v) in the tangent plane
        # Expanding this out is a bit faster than using np.dot for 2x2 matrix.
        u, v = self._apply_cd(x, y)

        if self.pv is not None:
            u, v = self._apply_pv(u, v)

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
        try:
            len(x)
            # If the inputs were numpy arrays, return the same
            return ra, dec
        except TypeError:
            # Otherwise return scalars
            #assert len(ra) == 1
            #assert len(dec) == 1
            return ra[0], dec[0]

    def _invert_pv(self, u, v):
        # Do this in C++ layer for speed.
        with convert_cpp_errors():
            return _galsim.InvertPV(u, v, self.pv.ctypes.data)

    def _invert_ab(self, x, y):
        # Do this in C++ layer for speed.
        abp_data = 0 if self.abp is None else self.abp.ctypes.data
        with convert_cpp_errors():
            return _galsim.InvertAB(len(self.ab[0]), x, y, self.ab.ctypes.data, abp_data)

    def _xy(self, ra, dec, color=None):
        u, v = self.center.project_rad(ra, dec, projection=self.projection)

        # Again, FITS has +u increasing to the east, not west.  Hence the - for u.
        factor = radians / degrees
        u *= -factor
        v *= factor

        if self.pv is not None:
            u, v = self._invert_pv(u, v)

        if not hasattr(self, 'cdinv'):
            self.cdinv = np.linalg.inv(self.cd)
        # This is a bit faster than using np.dot for 2x2 matrix.
        x = self.cdinv[0,0] * u + self.cdinv[0,1] * v
        y = self.cdinv[1,0] * u + self.cdinv[1,1] * v

        if self.ab is not None:
            x, y = self._invert_ab(x, y)

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

        p1 = np.array( [ image_pos.x, image_pos.y ] )

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
            p1 += np.dot(np.dot(self.ab, ypow), xpow)

            dxpow = np.zeros(order+1)
            dypow = np.zeros(order+1)
            dxpow[1:] = (np.arange(order)+1.) * xpow[:-1]
            dypow[1:] = (np.arange(order)+1.) * ypow[:-1]
            j1 = np.transpose([ np.dot(np.dot(self.ab, ypow), dxpow) ,
                                np.dot(np.dot(self.ab, dypow), xpow) ])
            j1 += np.diag([1,1])
            jac = np.dot(j1,jac)

        # The jacobian here is just the cd matrix.
        p2 = np.dot(self.cd, p1)
        jac = np.dot(self.cd, jac)

        if self.pv is not None:
            # Now we apply the distortion terms
            u = p2[0]
            v = p2[1]
            usq = u*u
            vsq = v*v

            upow = np.array([ 1., u, usq, usq*u ])
            vpow = np.array([ 1., v, vsq, vsq*v ])

            p2 = np.dot(np.dot(self.pv, vpow), upow)

            # The columns of the jacobian for this step are the same function with dupow
            # or dvpow.
            dupow = np.array([ 0., 1., 2.*u, 3.*usq ])
            dvpow = np.array([ 0., 1., 2.*v, 3.*vsq ])
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
            k = 0
            for n in range(4):
                for j in range(n+1):
                    i = n-j
                    header["PV1_" + str(k)] = self.pv[0, i, j]
                    header["PV2_" + str(k)] = self.pv[1, j, i]
                    k = k + 1
                    if k == 3: k = k + 1
        if self.ab is not None:
            order = len(self.ab[0])-1
            header["A_ORDER"] = order
            for i in range(order+1):
                for j in range(order+1):
                    if self.ab[0,i,j] != 0.:
                        header["A_"+str(i)+"_"+str(j)] = self.ab[0, i, j]
            header["B_ORDER"] = order
            for i in range(order+1):
                for j in range(order+1):
                    if self.ab[1,i,j] != 0.:
                        header["B_"+str(i)+"_"+str(j)] = self.ab[1, i, j]
        if self.abp is not None:
            order = len(self.abp[0])-1
            header["AP_ORDER"] = order
            for i in range(order+1):
                for j in range(order+1):
                    if self.abp[0,i,j] != 0.:
                        header["AP_"+str(i)+"_"+str(j)] = self.abp[0, i, j]
            header["BP_ORDER"] = order
            for i in range(order+1):
                for j in range(order+1):
                    if self.abp[1,i,j] != 0.:
                        header["BP_"+str(i)+"_"+str(j)] = self.abp[1, i, j]
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
    """This is a function that returns a GSFitsWCS object for a TAN WCS projection.

    The TAN projection is essentially an affine transformation from image coordinates to
    Euclidean (u,v) coordinates on a tangent plane, and then a "deprojection" of this plane
    onto the sphere given a particular RA, Dec for the location of the tangent point.
    The tangent point will correspond to the location of (u,v) = (0,0) in the intermediate
    coordinate system.

    @param affine        An AffineTransform defining the transformation from image coordinates
                         to the coordinates on the tangent plane.
    @param world_origin  A CelestialCoord defining the location on the sphere where the
                         tangent plane is centered.
    @param units         The angular units of the (u,v) intermediate coordinate system.
                         [default: galsim.arcsec]

    @returns a GSFitsWCS describing this WCS.
    """
    # These will raise the appropriate errors if affine is not the right type.
    dudx = affine.dudx * units / degrees
    dudy = affine.dudy * units / degrees
    dvdx = affine.dvdx * units / degrees
    dvdy = affine.dvdy * units / degrees
    origin = affine.origin
    # The - signs are because the Fits standard is in terms of +u going east, rather than west
    # as we have defined.  So just switch the sign in the CD matrix.
    cd = np.array( [ [ -dudx, -dudy ], [ dvdx, dvdy ] ] )
    crpix = np.array( [ origin.x, origin.y ] )

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

    PyAstWCS,       # This requires `import starlink.Ast` to succeed.  This handles the largest
                    # number of WCS types of any of these.  In fact, it worked for every one
                    # we tried in our unit tests (which was not exhaustive).

    AstropyWCS,     # This requires `import astropy.wcs` to succeed.  It doesn't support quite as
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

    If none of them work, then the last class it tries, AffineTransform, is guaranteed to succeed,
    but it will only model the linear portion of the WCS (the CD matrix, CRPIX, and CRVAL), using
    reasonable defaults if even these are missing.  If you think that you have the right software
    for one of the WCS types, but FitsWCS still defaults to AffineTransform, it may be helpful to
    update your installation of PyFITS/astropy and the relevant WCS software (if you don't already
    have the latest version).

    Note: The list of classes this function will try may be edited, e.g. by an external module
    that wants to add an additional WCS type.  The list is `galsim.fitswcs.fits_wcs_types`.

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
    @param dir            Optional directory to prepend to `file_name`. [default: None]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as
                          appropriate for the given compression.  (e.g. for rice, the first
                          extension is the one you normally want.) [default: None]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a FitsHeader object.  [default: None]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read()
                          for the available options.  [default: 'auto']
    @param text_file      Normally a file is taken to be a fits file, but you can also give it a
                          text file with the header information (like the .head file output from
                          SCamp).  In this case you should set `text_file = True` to tell GalSim
                          to parse the file this way.  [default: False]
    @param suppress_warning Should a warning be emitted if none of the real FITS WCS classes
                          are able to successfully read the file, and we have to reset to
                          an AffineTransform instead?  [default: False]
                          (Note: this is set to True when this function is implicitly called from
                          one of the galsim.fits.read* functions.)
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

    # For linear WCS specifications, AffineTransformation should work.
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
            if hasattr(wcs,'_tag') and file_name is not 'header':
                if dir is not None:
                    import os
                    wcs._tag = repr(os.path.join(dir,file_name))
                else:
                    wcs._tag = repr(file_name)
                if hdu is not None:
                    wcs._tag += ', hdu=%r'%hdu
                if compression is not 'auto':
                    wcs._tag += ', compression=%r'%compression
            return wcs
        except KeyboardInterrupt:
            raise
        except Exception as err:
            pass
    else:  # pragma: no cover
        # Finally, this one is really the last resort, since it only reads in the linear part of the
        # WCS.  It defaults to the equivalent of a pixel scale of 1.0 if even these are not present.
        if not suppress_warning:
            galsim_warn("All the fits WCS types failed to read %r. Using AffineTransform "
                        "instead, which will not really be correct."%(file_name))
        return AffineTransform._readHeader(header)

# Let this function work like a class in config.
FitsWCS._req_params = { "file_name" : str }
FitsWCS._opt_params = { "dir" : str, "hdu" : int, "compression" : str, 'text_file' : bool }
FitsWCS._single_params = []
FitsWCS._takes_rng = False


