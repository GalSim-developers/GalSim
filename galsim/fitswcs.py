# Copyright 2012-2014 The GalSim developers:
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
"""@file fitswcs.py
The function FitsWCS.py acts like a BaseWCS class, but it really calls one of several other
classes depending on what python modules are available and what kind of FITS file you are 
trying to read.
"""

import galsim

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


class AstropyWCS(galsim.wcs.CelestialWCS):
    """This WCS uses astropy.wcs to read WCS information from a FITS file.
    It requires the astropy.wcs python module to be installed.

    Astropy may be installed using pip, fink, or port:

            pip install astropy
            fink install astropy-py27
            port install py27-astropy

    It also comes by default with Enthought and Anaconda. For more information, see their website:

            http://www.astropy.org/

    Initialization
    --------------
    An AstropyWCS is initialized with one of the following commands:

        wcs = galsim.AstropyWCS(file_name=file_name)  # Open a file on disk
        wcs = galsim.AstropyWCS(header=header)        # Use an existing pyfits header
        wcs = galsim.AstropyWCS(wcs=wcs)              # Use an existing astropy.wcs.WCS instance

    Exactly one of the parameters file_name, header or wcs is required.  Also, since the
    most common usage will probably be the first, you can also give a file_name without it
    being named:

        wcs = galsim.AstropyWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
    @param dir            Optional directory to prepend to the file name. [default: None]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [default: None]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a galsim.FitsHeader object.  [default: None]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [default: 'auto']
    @param wcs            An existing astropy.wcs.WCS instance [default: None]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 wcs=None, origin=None):
        import astropy.wcs
        import scipy # We don't need this yet, but we want it to fail now if it's not available.

        self._tag = None # Write something useful here (see below). This is just used for the repr.

        # Read the file if given.
        if file_name is not None:
            self._tag = file_name
            if header is not None:
                raise TypeError("Cannot provide both file_name and pyfits header")
            if wcs is not None:
                raise TypeError("Cannot provide both file_name and wcs")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)
            header = hdu.header

        # Load the wcs from the header.
        if header is not None:
            if self._tag is None: self._tag = 'header'
            if wcs is not None:
                raise TypeError("Cannot provide both pyfits header and wcs")
            self._fix_header(header)
            import warnings
            with warnings.catch_warnings():
                # The constructor might emit warnings if it wants to fix the header
                # information (e.g. RADECSYS -> RADESYSa).  We'd rather ignore these
                # warnings, since we don't much care if the input file is non-standard
                # so long as we can make it work.
                warnings.simplefilter("ignore")
                # Some versions of astropy don't like to accept a galsim.FitsHeader object
                # as the header attribute here, even though they claim that dict-like objects
                # are ok.  So pull out the astropy.io.header object in this case.
                if isinstance(header,galsim.fits.FitsHeader):
                    header = header.header
                wcs = astropy.wcs.WCS(header)
        if wcs is None:
            raise TypeError("Must provide one of file_name, header, or wcs")
        if self._tag is None: self._tag = 'wcs'
        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

        # If astropy.wcs cannot parse the header, it won't notice from just doing the 
        # WCS(header) command.  It will silently move on, thinking things are fine until
        # later when if will fail (with `RuntimeError: NULL error object in wcslib`).
        # We'd rather get that to happen now rather than later.
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ra, dec = wcs.all_pix2world( [ [0, 0] ], 1)[0]
        except Exception as err:
            raise RuntimeError("AstropyWCS was unable to read the WCS specification in the header.")

        self._wcs = wcs
        if origin == None:
            self._origin = galsim.PositionD(0,0)
        else:
            self._origin = origin

    @property
    def wcs(self): return self._wcs

    @property
    def origin(self): return self._origin

    def _fix_header(self, header):
        # We allow for the option to fix up the header information when a modification can
        # make it readable by astropy.wcs.

        # So far, we don't have any, but something could be added in the future.
        pass

    def _radec(self, x, y):
        import numpy
        x1 = numpy.atleast_1d(x)
        y1 = numpy.atleast_1d(y)

        try:
            # Apparently, the returned values aren't _necessarily_ (ra, dec).  They could be
            # (dec, ra) instead!  But if you add ra_dec_order=True, then it will be (ra, dec).
            # I can't imagine why that isn't the default, but there you go.
            # This currently fails with an AttributeError about astropy.wcs.Wcsprm.lattype
            # cf. https://github.com/astropy/astropy/pull/1463
            # Once they fix it, this is what we want.
            ra, dec = self._wcs.all_pix2world(x1, y1, 1, ra_dec_order=True)
        except AttributeError:
            # Until then, just assume that the returned values really are ra, dec.
            ra, dec = self._wcs.all_pix2world(x1, y1, 1)

        # astropy outputs ra, dec in degrees.  Need to convert to radians.
        factor = 1. * galsim.degrees / galsim.radians
        ra *= factor
        dec *= factor

        try:
            # If the inputs were numpy arrays, return the same
            len(x)
        except:
            # Otherwise, return scalars
            assert len(ra) == 1
            assert len(dec) == 1
            ra = ra[0]
            dec = dec[0]
        return ra, dec

    def _xy(self, ra, dec):
        import numpy
        factor = 1. * galsim.radians / galsim.degrees
        rd = numpy.atleast_2d([ra, dec]) * factor
        # Here we have to work around another astropy.wcs bug.  The way they use scipy's
        # Broyden's method doesn't work.  So I implement a fix here.
        if False:
            # This is what I would like to have done, but it doesn't work.  I've reported
            # the issue at:
            # https://github.com/astropy/astropy/issues/1977

            # Try their version first (with and without ra_dec_order) in case they fix this.
            import warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xy = self._wcs.all_world2pix(rd, 1, ra_dec_order=True)[0]
            except AttributeError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xy = self._wcs.all_world2pix(rd, 1)[0]
        else:
            # This section is basically a copy of astropy.wcs's _all_world2pix function, but
            # simplified a bit to remove some features we don't need, and with corrections
            # to make it work correctly.
            import astropy.wcs
            import scipy.optimize
            import numpy
            import warnings

            origin = 1
            tolerance = 1.e-6

            # This call emits a RuntimeWarning about:
            #     [...]/site-packages/scipy/optimize/nonlin.py:943: RuntimeWarning: invalid value encountered in divide
            #       d = v / vdot(df, v)
            # It seems to be harmless, so we explicitly ignore it here:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xy0 = self._wcs.wcs_world2pix(rd, origin)

            # Note that the fmod bit accounts for the possibility that ra and the ra returned
            # from all_pix2world have a different wrapping around 360.  We fmod dec too even
            # though it won't do anything, since that's how the numpy array fmod2 has to work.
            func = lambda pix: (
                    (numpy.fmod(self._wcs.all_pix2world(numpy.atleast_2d(pix),origin) - 
                                rd + 180,360) - 180).flatten() )

            # This is the main bit that the astropy function is missing.
            # The scipy.optimize.broyden1 function can't handle starting at exactly the right
            # solution.  It iterates to its limit and then ends with
            #     Traceback (most recent call last):
            #       [... snip ...]
            #       File "[...]/site-packages/scipy/optimize/nonlin.py", line 331, in nonlin_solve
            #         raise NoConvergence(_array_like(x, x0))
            #     scipy.optimize.nonlin.NoConvergence: [ 113.74961526  179.99982209]
            #
            # Providing a good estimate of the scale size gets rid of this.  And even if we aren't
            # starting at exactly the right value, it is hugely more efficient to give it an
            # estimate of alpha, since it is not typically near unity in this case, so it is much
            # faster to start with something closer to the right value.
            alpha = numpy.mean(numpy.abs(self._wcs.wcs.get_cdelt()))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xy = [ scipy.optimize.broyden1(func, xy_init, x_tol=tolerance, alpha=alpha)
                            for xy_init in xy0 ]

        try:
            # If the inputs were numpy arrays, return the same
            len(ra)
            x, y = numpy.array(xy).transpose()
        except:
            # Otherwise, return scalars
            assert len(xy) == 1
            x, y = xy[0]
        return x, y

    def _newOrigin(self, origin):
        return AstropyWCS(wcs=self._wcs, origin=origin)

    def _writeHeader(self, header, bounds):
        # Make a new header with the contents of this WCS.
        # Note: relax = True means to write out non-standard FITS types.
        # Weirdly, this is the default when reading the header, but not when writing.
        header.update(self._wcs.to_header(relax=True))

        # And write the name as a special GalSim key
        header["GS_WCS"] = ("AstropyWCS", "GalSim WCS name")
        # Finally, update the CRPIX items if necessary.
        if self.origin.x != 0:
            header["CRPIX1"] = header["CRPIX1"] + self.origin.x
        if self.origin.y != 0:
            header["CRPIX2"] = header["CRPIX2"] + self.origin.y
        return header

    @staticmethod
    def _readHeader(header):
        return AstropyWCS(header=header)

    def copy(self):
        # The copy module version of copying the dict works fine here.
        import copy
        return copy.copy(self)

    def __eq__(self, other):
        return ( isinstance(other, AstropyWCS) and
                 self._wcs == other._wcs and
                 self.origin == other.origin )

    def __repr__(self):
        return "AstropyWCS(%r,%r)"%(self._tag, self.origin)


class PyAstWCS(galsim.wcs.CelestialWCS):
    """This WCS uses PyAst (the python front end for the Starlink AST code) to read WCS
    information from a FITS file.  It requires the starlink.Ast python module to be installed.

    Starlink may be installed using pip:

            pip install starlink-pyast

    For more information, see their website:

            https://pypi.python.org/pypi/starlink-pyast/

    Note: There were bugs in starlink.Ast prior to version 2.6, so if you have an earlier version,
    you should upgrate to at least 2.6.

    Initialization
    --------------
    A PyAstWCS is initialized with one of the following commands:

        wcs = galsim.PyAstWCS(file_name=file_name)  # Open a file on disk
        wcs = galsim.PyAstWCS(header=header)        # Use an existing pyfits header
        wcs = galsim.PyAstWCS(wcsinfo=wcsinfo)      # Use an existing starlink.Ast.FrameSet

    Exactly one of the parameters file_name, header or wcsinfo is required.  Also, since the
    most common usage will probably be the first, you can also give a file_name without it
    being named:

        wcs = galsim.PyAstWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
    @param dir            Optional directory to prepend to the file name. [default: None]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [default: None]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a galsim.FitsHeader object.  [default: None]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [default:'auto']
    @param wcsinfo        An existing starlink.Ast.WcsMap [default: None]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 wcsinfo=None, origin=None):
        import starlink.Ast, starlink.Atl
        # Note: For much of this class implementation, I've followed the example provided here:
        #       http://dsberry.github.io/starlink/node4.html
        self._tag = None # Write something useful here (see below). This is just used for the repr.
        hdu = None

        # Read the file if given.
        if file_name is not None:
            self._tag = file_name
            if header is not None:
                raise TypeError("Cannot provide both file_name and pyfits header")
            if wcsinfo is not None:
                raise TypeError("Cannot provide both file_name and wcsinfo")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)
            header = hdu.header

        # Load the wcs from the header.
        if header is not None:
            if self._tag is None: self._tag = 'header'
            if wcsinfo is not None:
                raise TypeError("Cannot provide both pyfits header and wcsinfo")
            self._fix_header(header)
            # PyFITSAdapter requires an hdu, not a header, so if we were given a header directly,
            # then we need to mock it up.
            if hdu is None:
                from galsim import pyfits
                hdu = pyfits.PrimaryHDU()
                galsim.fits.FitsHeader(hdu_list=hdu).update(header)
            fc = starlink.Ast.FitsChan( starlink.Atl.PyFITSAdapter(hdu) )
            wcsinfo = fc.read()
            if wcsinfo == None:
                raise RuntimeError("Failed to read WCS information from fits file")

        if wcsinfo is None:
            raise TypeError("Must provide one of file_name, header, or wcsinfo")
        if self._tag is None: self._tag = 'wcsinfo'

        #  We can only handle WCS with 2 pixel axes (given by Nin) and 2 WCS axes
        # (given by Nout).
        if wcsinfo.Nin != 2 or wcsinfo.Nout != 2:
            raise RuntimeError("The world coordinate system is not 2-dimensional")

        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

        self._wcsinfo = wcsinfo
        if origin == None:
            self._origin = galsim.PositionD(0,0)
        else:
            self._origin = origin

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

    def _radec(self, x, y):
        import numpy
        # Need this to look like 
        #    [ [ x1, x2, x3... ], [ y1, y2, y3... ] ] 
        # if input is either scalar x,y or two arrays.
        xy = numpy.array([numpy.atleast_1d(x), numpy.atleast_1d(y)])

        ra, dec = self._wcsinfo.tran( xy )
        # PyAst returns ra, dec in radians, so we're good.

        try:
            len(x)
        except:
            # If the inputs weren't numpy arrays, return scalars
            assert len(ra) == 1
            assert len(dec) == 1
            ra = ra[0]
            dec = dec[0]
        return ra, dec

    def _xy(self, ra, dec):
        import numpy
        rd = numpy.array([numpy.atleast_1d(ra), numpy.atleast_1d(dec)])

        x, y = self._wcsinfo.tran( rd, False )

        try:
            len(ra)
        except:
            assert len(x) == 1
            assert len(y) == 1
            x = x[0]
            y = y[0]
        return x, y

    def _newOrigin(self, origin):
        return PyAstWCS(wcsinfo=self._wcsinfo, origin=origin)

    def _writeHeader(self, header, bounds):
        # See https://github.com/Starlink/starlink/issues/24 for helpful information from 
        # David Berry, who assisted me in getting this working.

        from galsim import pyfits
        import starlink.Atl

        hdu = pyfits.PrimaryHDU()
        fc = starlink.Ast.FitsChan( None, starlink.Atl.PyFITSAdapter(hdu) , "Encoding=FITS-WCS")
        success = fc.write(self._wcsinfo)
        # PyAst doesn't write out TPV or ZPX correctly.  It writes them as TAN and ZPN 
        # respectively.  However, it claims success nonetheless, so we need to countermand that.  
        # The easiest way I found to check for them is that the string TPN is in the string 
        # version of wcsinfo.  So check for that and set success = False in that case.
        if 'TPN' in str(self._wcsinfo): success = False
        if not success:
            # This should always work, since it uses starlinks own proprietary encoding, but 
            # it won't necessarily be readable by ds9.
            fc = starlink.Ast.FitsChan( None, starlink.Atl.PyFITSAdapter(hdu))
            fc.write(self._wcsinfo)
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
        return PyAstWCS(header=header, origin=galsim.PositionD(x0,y0))
 
    def copy(self):
        # The copy module version of copying the dict works fine here.
        import copy
        return copy.copy(self)

    def __eq__(self, other):
        return ( isinstance(other, PyAstWCS) and
                 self._wcsinfo == other._wcsinfo and
                 self.origin == other.origin)

    def __repr__(self):
        return "PyAstWCS(%r,%r)"%(self._tag, self.origin)


class WcsToolsWCS(galsim.wcs.CelestialWCS):
    """This WCS uses wcstools executables to perform the appropriate WCS transformations
    for a given FITS file.  It requires wcstools command line functions to be installed.

    Note: It uses the wcstools executables xy2sky and sky2xy, so it can be quite a bit less
          efficient than other options that keep the WCS in memory.

    See their website for information on downloading and installing wcstools:

            http://tdc-www.harvard.edu/software/wcstools/

    Initialization
    --------------
    A WcsToolsWCS is initialized with the following command:

        wcs = galsim.WcsToolsWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.
    @param dir            Optional directory to prepend to the file name. [default: None]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "origin" : galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name, dir=None, origin=None):
        import os
        if dir:
            file_name = os.path.join(dir, file_name)
        if not os.path.isfile(file_name):
            raise IOError('Cannot find file '+file_name)

        # Check wcstools is installed and that it can read the file.
        import subprocess
        # If xy2sky is not installed, this will raise an OSError
        p = subprocess.Popen(['xy2sky', '-d', '-n', '10', file_name, '0', '0'],
                             stdout=subprocess.PIPE)
        results = p.communicate()[0]
        p.stdout.close()
        if len(results) == 0:
            raise IOError('wcstools (specifically xy2sky) was unable to read '+file_name)

        self._file_name = file_name
        if origin == None:
            self._origin = galsim.PositionD(0,0)
        else:
            self._origin = origin

    @property
    def file_name(self): return self._file_name

    @property
    def origin(self): return self._origin

    def _radec(self, x, y):
        #print 'start wcstools _radec'
        #print 'x = ',x
        #print 'y = ',y

        import numpy
        # Need this to look like 
        #    [ x1, y1, x2, y2, ... ] 
        # if input is either scalar x,y or two arrays.
        xy = numpy.array([x, y]).transpose().flatten()
        #print 'xy = ',xy
        
        # The OS cannot handle arbitrarily long command lines, so we may need to split up
        # the list into smaller chunks.
        import os
        if 'SC_ARG_MAX' in os.sysconf_names:
            arg_max = os.sysconf('SC_ARG_MAX') 
        else:
            # A conservative guess. My machines have 131072, 262144, and 2621440
            arg_max = 32768  
        #print 'arg_max = ',arg_max

        # Sometimes SC_ARG_MAX is listed as -1.  Apparently that means "the configuration name
        # is known, but the value is not defined." So, just go with the above conservative value.
        if arg_max <= 0:
            arg_max = 32768
            #print 'arg_max => ',arg_max

        # Just in case something weird happened.  This should be _very_ conservative.
        # It's the smallest value in this list of values for a bunch of systems:
        # http://www.in-ulm.de/~mascheck/various/argmax/
        if arg_max < 4096:
            arg_max = 4096
            #print 'arg_max => ',arg_max

        # This corresponds to the total number of characters in the line.  
        # But we really need to know how many arguments we are allowed to use in each call.
        # Lets be conservative again and assume each argument is at most 20 characters.
        # (We ignore the few characters at the start for the command name and such.)
        nargs = int(arg_max / 40) * 2  # Make sure it is even!
        #print 'nargs = ',nargs

        xy_strs = [ str(z) for z in xy ]
        #print 'xy_strs = ',xy_strs
        ra = []
        dec = []

        for i in range(0,len(xy_strs),nargs):
            #print 'i = ',i
            xy1 = xy_strs[i:i+nargs]
            #print 'xy1 = ',xy1
            import subprocess
            # We'd like to get the output to 10 digits of accuracy.  This corresponds to
            # an accuracy of about 1.e-6 arcsec.  But sometimes xy2sky cannot handle it,
            # in which case the output will start with *************.  If this happens, just
            # decrease digits and try again.
            for digits in range(10,5,-1):
                # If xy2sky is not installed, this will raise an OSError
                p = subprocess.Popen(['xy2sky', '-d', '-n', str(digits), self._file_name] + xy1,
                                    stdout=subprocess.PIPE)
                results = p.communicate()[0]
                #print 'results for digits = ',digits,' = ',results
                p.stdout.close()
                if len(results) == 0:
                    raise IOError('wcstools command xy2sky was unable to read '+ self._file_name)
                if results[0] != '*': break
            if results[0] == '*':
                raise IOError('wcstools command xy2sky was unable to read '+self._file_name)
            lines = results.splitlines()
            #print 'lines = ',lines

            # Each line of output should looke like:
            #    x y J2000 ra dec
            # But if there was an error, the J200 might be missing or the output might look like
            #    Off map x y
            for line in lines:
                vals = line.split()
                #print 'vals = ',vals
                if len(vals) != 5:
                    raise RuntimeError('wcstools xy2sky returned invalid result near %f,%f'%(x0,y0))
                ra.append(float(vals[0]))
                dec.append(float(vals[1]))
            #print 'ra => ',ra
            #print 'dec => ',dec

        # wcstools reports ra, dec in degrees, so convert to radians
        factor = 1. * galsim.degrees / galsim.radians

        try:
            len(x)
            # If the inputs were numpy arrays, return the same
            return numpy.array(ra)*factor, numpy.array(dec)*factor
        except:
            # Otherwise return scalars
            assert len(ra) == 1
            assert len(dec) == 1
            return ra[0]*factor, dec[0]*factor

    def _xy(self, ra, dec):
        import subprocess
        import numpy
        rd = numpy.array([ra, dec]).transpose().flatten()
        rd *= 1. * galsim.radians / galsim.degrees
        for digits in range(10,5,-1):
            rd_strs = [ str(z) for z in rd ]
            p = subprocess.Popen(['sky2xy', '-n', str(digits), self._file_name] + rd_strs,
                                 stdout=subprocess.PIPE)
            results = p.communicate()[0]
            p.stdout.close()
            if len(results) == 0:
                raise IOError('wcstools (specifically sky2xy) was unable to read '+self._file_name)
            if results[0] != '*': break
        if results[0] == '*':
            raise IOError('wcstools (specifically sky2xy) was unable to read '+self._file_name)
        lines = results.splitlines()

        # The output should looke like:
        #    ra dec J2000 -> x y
        # However, if there was an error, the J200 might be missing.
        x = []
        y = []
        for line in lines:
            vals = results.split()
            if len(vals) < 6:
                raise RuntimeError('wcstools sky2xy returned invalid result for %f,%f'%(ra,dec))
            if len(vals) > 6:
                import warnings
                warnings.warn('wcstools sky2xy indicates that %f,%f is off the image\n'%(ra,dec) +
                              'output is %r'%results)
            x.append(float(vals[4]))
            y.append(float(vals[5]))

        try:
            len(ra)
            # If the inputs were numpy arrays, return the same
            return numpy.array(x), numpy.array(y)
        except:
            # Otherwise return scalars
            assert len(x) == 1
            assert len(y) == 1
            return x[0], y[0]
 
    def _newOrigin(self, origin):
        return WcsToolsWCS(self._file_name, origin=origin)

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
        results = p.communicate()[0]
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
        return WcsToolsWCS(file, origin=galsim.PositionD(x0,y0))

    def copy(self):
        # The copy module version of copying the dict works fine here.
        import copy
        return copy.copy(self)

    def __eq__(self, other):
        return ( isinstance(other, WcsToolsWCS) and
                 self._file_name == other._file_name and
                 self.origin == other.origin )

    def __repr__(self):
        return "WcsToolsWCS(%r,%r)"%(self._file_name, self.origin)


class GSFitsWCS(galsim.wcs.CelestialWCS):
    """This WCS uses a GalSim implementation to read a WCS from a FITS file.

    It doesn't do nearly as many WCS types as the other options, and it does not try to be
    as rigorous about supporting all possible valid variations in the FITS parameters.
    However, it does several popular WCS types properly, and it doesn't require any additional 
    python modules to be installed, which can be helpful.

    Currrently, it is able to parse the following WCS types: TAN, STG, ZEA, ARC, TPV, TNX

    Initialization
    --------------
    A GSFitsWCS is initialized with one of the following commands:

        wcs = galsim.GSFitsWCS(file_name=file_name)  # Open a file on disk
        wcs = galsim.GSFitsWCS(header=header)        # Use an existing pyfits header

    Also, since the most common usage will probably be the first, you can also give a file_name 
    without it being named:

        wcs = galsim.GSFitsWCS(file_name)

    In addition to reading from a FITS file, there is also a factory function that builds
    a GSFitsWCS object implementing a TAN projection.  See the docstring of TanWCS for
    more details.

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
    @param dir            Optional directory to prepend to the file name. [default: None]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [default: None]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a galsim.FitsHeader object.  [default: None]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [default: 'auto']
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [default: None]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 origin=None, _data=None):
        # Note: _data is not intended for end-user use.  It enables the equivalent of a 
        #       private constructor of GSFitsWCS by the function TanWCS.  The details of its
        #       use are intentionally not documented above.


        # If _data is given, copy the data and we're done.
        if _data is not None:
            self.wcs_type = _data[0]
            self.crpix = _data[1]
            self.cd = _data[2]
            self.center = _data[3]
            self.pv = _data[4]
            if self.wcs_type in [ 'TAN', 'TPV' ]:
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
            if header is not None:
                raise TypeError("Cannot provide both file_name and pyfits header")
            hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)
            header = hdu.header

        if header is None:
            raise TypeError("Must provide either file_name or header")

        # Read the wcs information from the header.
        self._read_header(header)

        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

        if origin is not None:
            self.crpix += [ origin.x, origin.y ]

    # The origin is a required attribute/property, since it is used by some functions like
    # withOrigin to get the current origin value.  We don't use it in this class, though, so
    # just make origin a dummy property that returns 0,0.
    @property
    def origin(self): return galsim.PositionD(0.,0.)

    def _read_header(self, header):
        # Start by reading the basic WCS stuff that most types have.
        ctype1 = header['CTYPE1']
        ctype2 = header['CTYPE2']
        if not (ctype1.startswith('RA---') and ctype2.startswith('DEC--')):
            raise NotImplementedError("GSFitsWCS can only handle cases where CTYPE1 is RA " +
                                      "and CTYPE2 is DEC")
        if ctype1[5:] != ctype2[5:]:
            raise RuntimeError("ctype1, ctype2 do not seem to agree on the WCS type")
        self.wcs_type = ctype1[5:]
        if self.wcs_type in [ 'TAN', 'TPV', 'TNX' ]:
            self.projection = 'gnomonic'
        elif self.wcs_type == 'STG':
            self.projection = 'stereographic'
        elif self.wcs_type == 'ZEA':
            self.projection = 'lambert'
        elif self.wcs_type == 'ARC':
            self.projection = 'postel'
        else:
            raise RuntimeError("GSFitsWCS cannot read files using WCS type "+self.wcs_type)
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
            cd11 = float(header['CDELT1'])
            cd12 = 0.
            cd21 = 0.
            cd22 = float(header['CDELT2'])
        else:
            cd11 = 1.
            cd12 = 0.
            cd21 = 0.
            cd22 = 1.

        import numpy
        self.crpix = numpy.array( [ crpix1, crpix2 ] )
        self.cd = numpy.array( [ [ cd11, cd12 ], 
                                 [ cd21, cd22 ] ] )

        # Usually the units are degrees, but make sure
        if 'CUNIT1' in header:
            cunit1 = header['CUNIT1']
            cunit2 = header['CUNIT2']
            ra_units = galsim.angle.get_angle_unit(cunit1)
            dec_units = galsim.angle.get_angle_unit(cunit2)
        else:
            ra_units = galsim.degrees
            dec_units = galsim.degrees

        self.center = galsim.CelestialCoord(crval1 * ra_units, crval2 * dec_units)

        # There was an older proposed standard that used TAN with PV values, which is used by
        # SCamp, so we want to support it if possible.  The standard is now called TPV, so
        # use that for our wcs_type if we see the PV values with TAN.
        if self.wcs_type == 'TAN' and 'PV1_10' in header:
            self.wcs_type = 'TPV'

        self.pv = None
        if self.wcs_type == 'TPV':
            self._read_tpv(header)
        elif self.wcs_type == 'TNX':
            self._read_tnx(header)

        # I think the CUNIT specification applies to the CD matrix as well, but I couldn't actually
        # find good documentation for this.  Plus all the examples I saw used degrees anyway, so 
        # it's hard to tell.  Hopefully this will never matter, but if CUNIT is not deg, this 
        # next bit might be wrong.
        # I did see documentation that the PV matrices always use degrees, so at least we shouldn't
        # have to worry about that.
        if ra_units != galsim.degrees:
            self.cd[0,:] *= 1. * ra_units / galsim.degrees
        if dec_units != galsim.degrees:
            self.cd[1,:] *= 1. * dec_units / galsim.degrees

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
             'PV2_11' in header and header['PV1_11'] != 0.0 ):
            raise NotImplementedError("We don't implement odd powers of r for TPV")
        if 'PV1_12' in header:
            raise NotImplementedError("We don't implement past 3rd order terms for TPV")

        import numpy
        # Another strange thing is that the two matrices are defined in the opposite order
        # with respect to their element ordering.  And remember that we skipped k=3 in the
        # original reading, so indices 3..9 here were originally called PVi_4..10
        self.pv = numpy.array( [ [ [ pv1[0], pv1[2], pv1[5], pv1[9] ],
                                   [ pv1[1], pv1[4], pv1[8],   0.   ],
                                   [ pv1[3], pv1[7],   0.  ,   0.   ],
                                   [ pv1[6],   0.  ,   0.  ,   0.   ] ],
                                 [ [ pv2[0], pv2[1], pv2[3], pv2[6] ],
                                   [ pv2[2], pv2[4], pv2[7],   0.   ],
                                   [ pv2[5], pv2[8],   0.  ,   0.   ],
                                   [ pv2[9],   0.  ,   0.  ,   0.   ] ] ] )

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
             not wat1[-1].endswith('"') ):
            raise RuntimeError("TNX WAT1 was not as expected")
        if ( len(wat2) < 12 or
             wat2[0] != 'wtype=tnx' or 
             wat2[1] != 'axtype=dec' or
             wat2[2] != 'latcor' or
             wat2[3] != '=' or 
             not wat2[4].startswith('"') or
             not wat2[-1].endswith('"') ):
            raise RuntimeError("TNX WAT2 was not as expected")

        # Break the next bit out into another function, since it is the same for x and y.
        pv1 = self._parse_tnx_data(wat1[4:])
        pv2 = self._parse_tnx_data(wat2[4:])

        # Those just give the adjustments to the position, not the matrix that gives the final
        # position.  i.e. the TNX standard uses u = u + [1 u u^2 u^3] PV [1 v v^2 v^3]T.
        # So we need to add 1 to the correct term in each matrix to get what we really want.
        pv1[1,0] += 1.
        pv2[0,1] += 1.

        # Finally, store these as our pv 3-d array.
        import numpy
        self.pv = numpy.array([pv1, pv2])

        # We've now converted this to TPV, so call it that when we output to a fits header.
        self.wcs_type = 'TPV'

    def _parse_tnx_data(self, data):

        # I'm not sure if there is any requirement on there being a space before the final " and
        # not before the initial ".  But both the example in the description of the standard and
        # the one we have in our test directory are this way.  Here, if the " is by itself, I
        # remove the item, and if it is part of a longer string, I just strip it off.  Seems the 
        # most sensible thing to do.
        if data[0] == '"':
            data = data[1:]
        else:
            data[0] = data[0][1:]
        if data[-1] == '"':
            data = data[:-1]
        else:
            data[-1] = data[-1][:-1]

        code = int(data[0].strip('.'))  # Weirdly, these integers are given with decimal points.
        xorder = int(data[1].strip('.'))
        yorder = int(data[2].strip('.'))
        cross = int(data[3].strip('.'))
        if cross != 2:
            raise NotImplementedError("TNX only implemented for half-cross option.")
        if xorder != 4 or yorder != 4:
            raise NotImplementedError("TNX only implemented for order = 4")
        # Note: order = 4 really means cubic.  order is how large the pv matrix is, i.e. 4x4.

        xmin = float(data[4])
        xmax = float(data[5])
        ymin = float(data[6])
        ymax = float(data[7])

        pv1 = [ float(x) for x in data[8:] ]
        if len(pv1) != 10:
            raise RuntimeError("Wrong number of items found in WAT data")

        # Put these into our matrix formulation.
        import numpy
        pv = numpy.array( [ [ pv1[0], pv1[4], pv1[7], pv1[9] ],
                            [ pv1[1], pv1[5], pv1[8],   0.   ],
                            [ pv1[2], pv1[6],   0.  ,   0.   ],
                            [ pv1[3],   0.  ,   0.  ,   0.   ] ] )

        # Convert from Legendre or Chebyshev polynomials into regular polynomials.
        if code < 3:
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
            xm = numpy.zeros((4,4))
            ym = numpy.zeros((4,4))
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
            else:
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

            pv2 = numpy.dot(xm.T , numpy.dot(pv, ym))
            return pv2

    def _radec(self, x, y):
        import numpy

        # Start with (x,y) = the image position
        p1 = numpy.array( [ numpy.atleast_1d(x), numpy.atleast_1d(y) ] )

        # This converts to (u,v) in the tangent plane
        p2 = numpy.dot(self.cd, p1 - self.crpix[:,numpy.newaxis]) 

        if self.pv is not None:
            # Now we apply the distortion terms
            u = p2[0]
            v = p2[1]
            usq = u*u
            vsq = v*v
            ones = numpy.ones(u.shape)
            upow = numpy.array([ ones, u, usq, usq*u ])
            vpow = numpy.array([ ones, v, vsq, vsq*v ])
            # If we only have one input position, then p2 is 
            #     p2[0] = upowT . pv[0] . vpow
            #     p2[1] = upowT . pv[1] . vpow
            # using matrix products, which are effected with the numpy.dot function.
            # When there are multiple inputs, then upow and vpow are each 4xN matrices.
            # The values we want are the diagonal of the matrix you would get from the 
            # above formulae.  So we use the fact that 
            #     diag(AT . B) = sum_rows(A * B)
            temp = numpy.dot(self.pv, vpow)
            p2 = numpy.sum(upow * temp, axis=1)

        # Convert (u,v) from degrees to arcsec
        # Also, the FITS standard defines u,v backwards relative to our standard.
        # They have +u increasing to the east, not west.  Hence the - for u.
        factor = 1. * galsim.degrees / galsim.arcsec
        u = -p2[0] * factor
        v = p2[1] * factor

        # Finally convert from (u,v) to (ra, dec) using the appropriate projection.
        ra, dec = self.center.deproject_rad(u, v, projection=self.projection)

        try:
            len(x)
            # If the inputs were numpy arrays, return the same
            return ra, dec
        except:
            # Otherwise return scalars
            assert len(ra) == 1
            assert len(dec) == 1
            return ra[0], dec[0]
 
    def _xy(self, ra, dec):
        import numpy, numpy.linalg

        u, v = self.center.project_rad(ra, dec, projection=self.projection)

        # Again, FITS has +u increasing to the east, not west.  Hence the - for u.
        factor = 1. * galsim.arcsec / galsim.degrees
        u *= -factor
        v *= factor

        p2 = numpy.array( [ u, v ] )

        if self.pv is not None:
            # Let (s,t) be the current value of (u,v).  Then we want to find a new (u,v) such that
            #
            #       [ s t ] = [ 1 u u^2 u^3 ] pv [ 1 v v^2 v^3 ]^T
            #
            # Start with (u,v) = (s,t)
            #
            # Then use Newton-Raphson iteration to improve (u,v).  This is extremely fast
            # for typical PV distortions, since the distortions are generally very small.
            # Newton-Raphson doubles the number of significant digits in each iteration.


            MAX_ITER = 10
            TOL = 1.e-8 * galsim.arcsec / galsim.degrees   # pv always uses degrees units
            prev_err = None
            u = p2[0]
            v = p2[1]
            for iter in range(MAX_ITER):
                usq = u*u
                vsq = v*v
                upow = numpy.array([ 1., u, usq, usq*u ])
                vpow = numpy.array([ 1., v, vsq, vsq*v ])

                diff = numpy.dot(numpy.dot(self.pv, vpow), upow) - p2

                # Check that things are improving...
                err = numpy.max(numpy.abs(diff))
                if prev_err:
                    if err > prev_err:
                        raise RuntimeError("Unable to solve for image_pos (not improving)")
                prev_err = err

                # If we are below tolerance, break out of the loop
                if err < TOL: 
                    # Update p2 to the new value.
                    p2 = numpy.array( [ u, v ] )
                    break
                else:
                    dupow = numpy.array([ 0., 1., 2.*u, 3.*usq ])
                    dvpow = numpy.array([ 0., 1., 2.*v, 3.*vsq ])
                    j1 = numpy.transpose([ numpy.dot(numpy.dot(self.pv, vpow), dupow) ,
                                           numpy.dot(numpy.dot(self.pv, dvpow), upow) ])
                    dp = numpy.linalg.solve(j1, diff)
                    u -= dp[0]
                    v -= dp[1]
            if not err < TOL:
                raise RuntimeError("Unable to solve for image_pos (max iter reached)")

        p1 = numpy.dot(numpy.linalg.inv(self.cd), p2) + self.crpix
        x, y = p1
        return x, y

    # Override the version in CelestialWCS, since we can do this more efficiently.
    def _local(self, image_pos, world_pos):
        if image_pos is None:
            if world_pos is None:
                raise TypeError("Either image_pos or world_pos must be provided")
            image_pos = self._posToImage(world_pos)

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

        import numpy
        p1 = numpy.array( [ image_pos.x, image_pos.y ] )

        p2 = numpy.dot(self.cd, p1 - self.crpix) 
        # The jacobian here is just the cd matrix.
        jac = self.cd

        if self.pv is not None:
            # Now we apply the distortion terms
            u = p2[0]
            v = p2[1]
            usq = u*u
            vsq = v*v

            upow = numpy.array([ 1., u, usq, usq*u ])
            vpow = numpy.array([ 1., v, vsq, vsq*v ])

            p2 = numpy.dot(numpy.dot(self.pv, vpow), upow)

            # The columns of the jacobian for this step are the same function with dupow 
            # or dvpow.
            dupow = numpy.array([ 0., 1., 2.*u, 3.*usq ])
            dvpow = numpy.array([ 0., 1., 2.*v, 3.*vsq ])
            j1 = numpy.transpose([ numpy.dot(numpy.dot(self.pv, vpow), dupow) ,
                                   numpy.dot(numpy.dot(self.pv, dvpow), upow) ])
            jac = numpy.dot(j1,jac)

        unit_convert = [ -1 * galsim.degrees / galsim.arcsec, 1 * galsim.degrees / galsim.arcsec ]
        p2 *= unit_convert
        # Subtle point: Don't use jac *= ..., because jac might currently be self.cd, and 
        #               that would change self.cd!
        jac = jac * numpy.transpose( [ unit_convert ] )

        # Finally convert from (u,v) to (ra, dec).  We have a special function that computes
        # the jacobian of this step in the CelestialCoord class.
        drdu, drdv, dddu, dddv = self.center.deproject_jac(p2[0], p2[1], projection=self.projection)
        j2 = numpy.array([ [ drdu, drdv ],
                           [ dddu, dddv ] ])
        jac = numpy.dot(j2,jac)

        return galsim.JacobianWCS(jac[0,0], jac[0,1], jac[1,0], jac[1,1])


    def _newOrigin(self, origin):
        ret = self.copy()
        if origin is not None:
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
        header["CRVAL1"] = self.center.ra / galsim.degrees
        header["CRVAL2"] = self.center.dec / galsim.degrees
        if self.pv is not None:
            k = 0
            for n in range(4):
                for j in range(n+1):
                    i = n-j
                    header["PV1_" + str(k)] = self.pv[0, i, j]
                    header["PV2_" + str(k)] = self.pv[1, j, i]
                    k = k + 1
                    if k == 3: k = k + 1
        return header

    @staticmethod
    def _readHeader(header):
        return GSFitsWCS(header=header)

    def copy(self):
        # The copy module version of copying the dict works fine here.
        import copy
        return copy.copy(self)

    def __eq__(self, other):
        return ( isinstance(other, GSFitsWCS) and
                 self.wcs_type == other.wcs_type and
                 (self.crpix == other.crpix).all() and
                 (self.cd == other.cd).all() and
                 self.center == other.center and
                 ( (self.pv == None and other.pv == None) or
                   (self.pv == other.pv).all() ) )

    def __repr__(self):
        return "GSFitsWCS(%r,%r,%r,%r,%r)"%(self.wcs_type, repr(self.crpix), repr(self.cd), 
                                            self.center, repr(self.pv))


def TanWCS(affine, world_origin, units=galsim.arcsec):
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
    import numpy, numpy.linalg
    # These will raise the appropriate errors if affine is not the right type.
    dudx = affine.dudx * units / galsim.degrees
    dudy = affine.dudy * units / galsim.degrees
    dvdx = affine.dvdx * units / galsim.degrees
    dvdy = affine.dvdy * units / galsim.degrees
    origin = affine.origin
    # The - signs are because the Fits standard is in terms of +u going east, rather than west
    # as we have defined.  So just switch the sign in the CD matrix.
    cd = numpy.array( [ [ -dudx, -dudy ], [ dvdx, dvdy ] ] )
    crpix = numpy.array( [ origin.x, origin.y ] )

    if affine.world_origin is not None:
        # Then we need to absorb this back into crpix, since GSFits is expecting crpix to 
        # be the location of the tangent point in image coordinates.  i.e. where (u,v) = (0,0)
        # (u,v) = CD * (x-x0,y-y0) + (u0,v0)
        # (0,0) = CD * (x0',y0') - CD * (x0,y0) + (u0,v0)
        # CD (x0',y0') = CD (x0,y0) - (u0,v0)
        # (x0',y0') = (x0,y0) - CD^-1 (u0,v0)
        uv = numpy.array( [ affine.world_origin.x * units / galsim.degrees,
                            affine.world_origin.y * units / galsim.degrees ] )
        crpix -= numpy.dot(numpy.linalg.inv(cd) , uv)

    # Invoke the private constructor of GSFits using the _data kwarg.
    data = ('TAN', crpix, cd, world_origin, None)
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

    AstropyWCS,     # This requires `import astropy.wcs` to succeed.  So far, they only handle
                    # the standard official WCS types.  So not TPV, for instance.  Also, it is
                    # a little faster than PyAst, so we prefer PyAst when it is available.
                    # (But only because of our fix in the _xy function to not use the astropy
                    # version of all_world2pix function!)

    PyAstWCS,       # This requires `import starlink.Ast` to succeed.  This handles the largest
                    # number of WCS types of any of these.  In fact, it worked for every one
                    # we tried in our unit tests (which was not exhaustive).  This is a bit 
                    # slower than Astropy, but I think mostly due to their initial reading of 
                    # the fits header -- that seems to take a lot of time for some reason.
                    # Once it is loaded, the actual usage seems to be quite fast.

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
    reasonable defaults if even these are missing.

    Note: The list of classes this function will try may be edited, e.g. by an external module 
    that wants to add an additional WCS type.  The list is `galsim.wcs.fits_wcs_types`.

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [default: None]
    @param dir            Optional directory to prepend to the file name. [default: None]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [default: None]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a galsim.FitsHeader object.  [default: None]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
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
    if file_name is not None:
        if header is not None:
            raise TypeError("Cannot provide both file_name and pyfits header")
        header = galsim.FitsHeader(file_name=file_name, dir=dir, hdu=hdu, compression=compression,
                                   text_file=text_file)
    else:
        file_name = 'header' # For sensible error messages below.
    if header is None:
        raise TypeError("Must provide either file_name or header")

    for wcs_type in fits_wcs_types:
        try:
            wcs = wcs_type._readHeader(header)
            return wcs
        except Exception as err:
            #print 'caught ',err
            pass
    # Finally, this one is really the last resort, since it only reads in the linear part of the 
    # WCS.  It defaults to the equivalent of a pixel scale of 1.0 if even these are not present.
    if not suppress_warning:
        import warnings
        warnings.warn("All the fits WCS types failed to read "+file_name+".  " +
                      "Using AffineTransform instead, which will not really be correct.")
    wcs = galsim.wcs.AffineTransform._readHeader(header)
    return wcs

# Let this function work like a class in config.
FitsWCS._req_params = { "file_name" : str }
FitsWCS._opt_params = { "dir" : str, "hdu" : int, "compression" : str, 'text_file' : bool }
FitsWCS._single_params = []
FitsWCS._takes_rng = False
FitsWCS._takes_logger = False


