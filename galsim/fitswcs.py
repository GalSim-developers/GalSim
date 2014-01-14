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
#
# As for all global WCS classes, they must define the following:
#
#     _is_local         boolean variable declaring whether the WCS is local, linear
#     _is_uniform       boolean variable declaring whether the pixels are uniform
#     _is_celestial     boolean variable declaring whether the world coords are celestial
#     _posToWorld       function converting image_pos to world_pos
#     _posToImage       function converting world_pos to image_pos
#     copy              return a copy
#     __eq__            check if this equals another WCS
#     __ne__            check if this is not equal to another WCS
#     _writeHeader      function that writes the WCS to a fits header.
#     _readHeader       static function that reads the WCS from a fits header.
#
#########################################################################################


class AstropyWCS(galsim.wcs.BaseWCS):
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
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [ Default `hdu = None` ]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a galsim.FitsHeader object.  [ Default `header = None` ]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    @param wcs            An existing astropy.wcs.WCS instance [ Default: `wcs = None` ]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 wcs=None, origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        import astropy.wcs
        self._tag = None # Write something useful here.  (It is just used for the repr.)

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
                wcs = astropy.wcs.WCS(header)
        if wcs is None:
            raise TypeError("Must provide one of file_name, header, or wcs")
        if self._tag is None: self._tag = 'wcs'
        if file_name is not None:
            galsim.fits.closeHDUList(hdu_list, fin)

        # If astropy.wcs cannot parse the header, it won't notice from just doing the 
        # WCS(header) command.  It will silently move on, thinking things are fine until
        # later when if will fail (with `RuntimeError: NULL error object in wcslib`).
        # We're rather get that to happen now rather than later.
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ra, dec = wcs.all_pix2world( [ [0, 0] ], 1)[0]
        except Exception as err:
            raise RuntimeError("AstropyWCS was unable to read the WCS specification in the header.")

        self._wcs = wcs
        if origin == None:
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = origin.x
            self._y0 = origin.y


    @property
    def wcs(self): return self._wcs

    @property
    def origin(self): return galsim.PositionD(self._x0, self._y0)

    def _fix_header(self, header):
        # We allow for the option to fix up the header information when a modification can
        # make it readable by astropy.wcs.

        # So far, we don't have any, but something could be added in the future.
        pass

    def _radec(self, x, y):
        import numpy
        # Need this to look like 
        #    [ [ x1, y1 ], [ x2, y2 ], ... ] 
        # if input is either scalar x,y or two arrays.
        xy = numpy.atleast_2d([x, y])
        # Apparently, the returned values aren't _necessarily_ (ra, dec).  They could be
        # (dec, ra) instead!  But if you add ra_dec_order=True, then it will be (ra, dec).
        # I can't imagnie why that isn't the default, but there you go.
        try:
            # This currently fails with an AttributeError about astropy.wcs.Wcsprm.lattype
            # c.f. https://github.com/astropy/astropy/pull/1463
            # Once they fix it, this is what we want.
            world = self._wcs.all_pix2world( xy, 1, ra_dec_order=True)
        except:
            # Until then, just assume that the returned values really are ra, dec.
            world = self._wcs.all_pix2world( xy, 1)

        # astropy outputs ra, dec in degrees.  Need to convert to radians.
        world *= 1. * galsim.degrees / galsim.radians

        try:
            # If the inputs were numpy arrays, return the same
            len(x)
            ra, dec = world.transpose()
        except:
            # Otherwise, return scalars
            assert len(world) == 1
            ra, dec = world[0]
        return ra, dec

    def _posToWorld(self, image_pos):
        x = image_pos.x - self._x0
        y = image_pos.y - self._y0
        ra, dec = self._radec(x,y)
        return galsim.CelestialCoord(ra * galsim.radians, dec * galsim.radians)

    def _posToImage(self, world_pos):
        ra = world_pos.ra / galsim.degrees
        dec = world_pos.dec / galsim.degrees
        # Here we have to work around another astropy.wcs bug.  The way they use scipy's
        # Broyden's method doesn't work.  So I implement a fix here.
        try:
            # Try their version first (with and without ra_dec_order) in case they fix this.
            import warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, y = self._wcs.all_world2pix( [ [ra, dec] ], 1, ra_dec_order=True)[0]
            except AttributeError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, y = self._wcs.all_world2pix( [ [ra, dec] ], 1)[0]
        except:
            # This section is basically a copy of astropy.wcs's _all_world2pix function, but
            # simplified a bit to remove some features we don't need, and with corrections
            # to make it work correctly.
            import astropy.wcs
            import scipy.optimize
            import numpy
            import warnings

            world = [ra,dec]
            origin = 1
            tolerance = 1.e-6

            # This call emits a RuntimeWarning about:
            #     /sw/lib/python2.7/site-packages/scipy/optimize/nonlin.py:943: RuntimeWarning: invalid value encountered in divide
            #       d = v / vdot(df, v)
            # It seems to be harmless, so we explicitly ignore it here:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x0 = self._wcs.wcs_world2pix(numpy.atleast_2d(world), origin).flatten()

            # Note that fmod bit accounts for the possibility that ra and the ra returned from 
            # all_pix2world have a different wrapping around 360.  We fmod dec too even though 
            # it won't do anything, since that's how the numpy array fmod2 has to work.
            func = lambda pix: (
                    (numpy.fmod(self._wcs.all_pix2world(numpy.atleast_2d(pix),origin) - 
                                world + 180,360) - 180).flatten() )

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
                soln = scipy.optimize.broyden1(func, x0, x_tol=tolerance, alpha=alpha)
            x,y = soln

        return galsim.PositionD(x + self._x0, y + self._y0)

    def _setOrigin(self, origin):
        return AstropyWCS(wcs=self._wcs, origin=origin)

    def _writeHeader(self, header, bounds):
        # Make a new header with the contents of this WCS.
        # Note: relax = True means to write out non-standard FITS types.
        # Weirdly, this is the default when reading the header, but not when writing.
        header = self._wcs.to_header(relax=True)

        # And write the name as a special GalSim key
        header["GS_WCS"] = ("AstropyWCS", "GalSim WCS name")
        # Finally, update the CRPIX items if necessary.
        header["CRPIX1"] = header["CRPIX1"] + self.origin.x
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
        if not isinstance(other, AstropyWCS):
            return False
        else:
            return (
                self._wcs == other._wcs and
                self._x0 == other._x0 and
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "AstropyWCS(%r,%r)"%(self._tag, self.origin)


class PyAstWCS(galsim.wcs.BaseWCS):
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
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [ Default `hdu = None` ]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a galsim.FitsHeader object.  [ Default `header = None` ]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    @param wcsinfo        An existing starlink.Ast.WcsMap [ Default: `wcsinfo = None` ]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "hdu" : int, "origin" : galsim.PositionD,
                    "compression" : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name=None, dir=None, hdu=None, header=None, compression='auto',
                 wcsinfo=None, origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
        import starlink.Ast, starlink.Atl
        # Note: For much of this class implementation, I've followed the example provided here:
        #       http://dsberry.github.io/starlink/node4.html
        self._tag = None # Write something useful here for the repr.
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
                hdu.header = header
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
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = origin.x
            self._y0 = origin.y

    @property
    def wcsinfo(self): return self._wcsinfo

    @property
    def origin(self): return galsim.PositionD(self._x0, self._y0)

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

    def _posToWorld(self, image_pos):
        x = image_pos.x - self._x0
        y = image_pos.y - self._y0
        ra, dec = self._radec(x,y)
        return galsim.CelestialCoord(ra * galsim.radians, dec * galsim.radians)

    def _posToImage(self, world_pos):
        ra = world_pos.ra / galsim.radians
        dec = world_pos.dec / galsim.radians
        x,y = self._wcsinfo.tran( [ [ra], [dec] ], False)
        return galsim.PositionD(x[0] + self._x0, y[0] + self._y0)

    def _setOrigin(self, origin):
        return PyAstWCS(wcsinfo=self._wcsinfo, origin=origin)

    def _writeHeader(self, header, bounds):
        # See https://github.com/Starlink/starlink/issues/24 for helpful information from 
        # David Berry, who assisted me in getting this working.

        # Note: As David described on that page, starlink knows how to write using a 
        # FITS-WCS encoding that things like ds9 can read.  However, it doesn't do so at 
        # very high precision.  So the WCS after a round trip through the FITS-WCS encoding
        # is only accurate to about 1.e-2 arcsec.  The NATIVE encoding (which is the default
        # used here) usually writes things with enough digits to remain accurate.  But even 
        # then, there are a couple of WCS types where the round trip is only accurate to 
        # about 1.e-2 arcsec.
        
        from galsim import pyfits
        import starlink.Atl

        hdu = pyfits.PrimaryHDU()
        fc2 = starlink.Ast.FitsChan( None, starlink.Atl.PyFITSAdapter(hdu) )
        fc2.write(self._wcsinfo)
        fc2.writefits()
        header = hdu.header

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
        if not isinstance(other, PyAstWCS):
            return False
        else:
            return (
                self._wcsinfo == other._wcsinfo and
                self._x0 == other._x0 and
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "PyAstWCS(%r,%r)"%(self._tag, self.origin)


class WcsToolsWCS(galsim.wcs.BaseWCS):
    """This WCS uses wcstools executables to perform the appropriate WCS transformations
    for a given FITS file.  It requires wcstools command line functions to be installed.

    Note: It uses the wcstools executalbes xy2sky and sky2xy, so it can be quite a bit less
          efficient than other options that keep the WCS in memory.

    See their website for information on downloading and installing wcstools:

            http://tdc-www.harvard.edu/software/wcstools/

    Initialization
    --------------
    A WcsToolsWCS is initialized with the following command:

        wcs = galsim.WcsToolsWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
    """
    _req_params = { "file_name" : str }
    _opt_params = { "dir" : str, "origin" : galsim.PositionD }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, file_name, dir=None, origin=None):
        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True
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
            self._x0 = 0
            self._y0 = 0
        else:
            self._x0 = origin.x
            self._y0 = origin.y

    @property
    def file_name(self): return self._file_name

    @property
    def origin(self): return galsim.PositionD(self._x0, self._y0)

    def _radec(self, x, y):
        import numpy
        # Need this to look like 
        #    [ x1, y1, x2, y2, ... ] 
        # if input is either scalar x,y or two arrays.
        xy = numpy.array([x, y]).transpose().flatten()
        
        import subprocess
        # We'd like to get the output to 10 digits of accuracy.  This corresponds to
        # an accuracy of about 1.e-6 arcsec.  But sometimes xy2sky cannot handle it,
        # in which case the output will start with *************.  If this happens, just
        # decrease digits and try again.
        for digits in range(10,5,-1):
            xy_strs = [ str(z) for z in xy ]
            # If xy2sky is not installed, this will raise an OSError
            p = subprocess.Popen(['xy2sky', '-d', '-n', str(digits), self._file_name] + xy_strs,
                                 stdout=subprocess.PIPE)
            results = p.communicate()[0]
            p.stdout.close()
            if len(results) == 0:
                raise IOError('wcstools (specifically xy2sky) was unable to read '+self._file_name)
            if results[0] != '*': break
        if results[0] == '*':
            raise IOError('wcstools (specifically xy2sky) was unable to read '+self._file_name)
        lines = results.splitlines()

        # Each line of output should looke like:
        #    x y J2000 ra dec
        # However, if there was an error, the J200 might be missing or the output might look like
        #    Off map x y
        ra = []
        dec = []
        for line in lines:
            vals = line.split()
            if len(vals) != 5:
                raise RuntimeError('wcstools xy2sky returned invalid result near %f,%f'%(x0,y0))
            ra.append(float(vals[0]))
            dec.append(float(vals[1]))

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

    def _posToWorld(self, image_pos):
        x = image_pos.x - self._x0
        y = image_pos.y - self._y0
        ra, dec = self._radec(x,y)
        return galsim.CelestialCoord(ra * galsim.radians, dec * galsim.radians)

    def _posToImage(self, world_pos):
        ra = world_pos.ra / galsim.degrees
        dec = world_pos.dec / galsim.degrees

        import subprocess
        for digits in range(10,5,-1):
            p = subprocess.Popen(['sky2xy', '-n', str(digits), self._file_name,
                                  str(ra), str(dec)], stdout=subprocess.PIPE)
            results = p.communicate()[0]
            p.stdout.close()
            if len(results) == 0:
                raise IOError('wcstools (specifically sky2xy) was unable to read '+self._file_name)
            if results[0] != '*': break
        if results[0] == '*':
            raise IOError('wcstools (specifically sky2xy) was unable to read '+self._file_name)

        # The output should looke like:
        #    ra dec J2000 -> x y
        # However, if there was an error, the J200 might be missing.
        vals = results.split()
        if len(vals) < 6:
            raise RuntimeError('wcstools sky2xy returned invalid result for %f,%f'%(ra,dec))
        if len(vals) > 6:
            import warnings
            warnings.warn('wcstools sky2xy indicates that %f,%f is off the image\n'%(ra,dec) +
                          'output is %r'%results)
        x = float(vals[4])
        y = float(vals[5])
        return galsim.PositionD(x + self._x0, y + self._y0)

    def _setOrigin(self, origin):
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
        if not isinstance(other, WcsToolsWCS):
            return False
        else:
            return (
                self._file_name == other._file_name and
                self._x0 == other._x0 and
                self._y0 == other._y0)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "WcsToolsWCS(%r,%r)"%(self._file_name, self.origin)


class GSFitsWCS(galsim.wcs.BaseWCS):
    """This WCS uses a GalSim implementation to read a WCS from a FITS file.

    It doesn't do nearly as many WCS types as the other options, and it does not try to be
    as rigorous about supporting all possible valid variations in the FITS parameters.
    However, it does a few popular WCS types properly, and it doesn't require any additional 
    python modules to be installed, which can be helpful.

    Currrently, it is able to parse the following WCS types: TAN, TPV

    Initialization
    --------------
    A GSFitsWCS is initialized with one of the following commands:

        wcs = galsim.GSFitsWCS(file_name=file_name)  # Open a file on disk
        wcs = galsim.GSFitsWCS(header=header)        # Use an existing pyfits header

    Also, since the most common usage will probably be the first, you can also give a file_name 
    without it being named:

        wcs = galsim.GSFitsWCS(file_name)

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [ Default `hdu = None` ]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a galsim.FitsHeader object.  [ Default `header = None` ]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    @param origin         Optional origin position for the image coordinate system.
                          If provided, it should be a PostionD or PositionI.
                          [ Default: `origin = None` ]
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

        self._is_local = False
        self._is_uniform = False
        self._is_celestial = True

        # If _data is given, copy the data and we're done.
        if _data is not None:
            self.wcs_type = _data[0]
            self.crpix = _data[1]
            self.cd = _data[2]
            self.center = _data[3]
            self.pv = _data[4]
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

    # Some function assume these exist.  In this class, we keep crpix up to date rather than
    # use these.  So just make dummy properties that return 0.
    @property
    def origin(self): return galsim.PositionD(0.,0.)
    @property
    def _x0(self): return 0.
    @property
    def _y0(self): return 0.

    def _read_header(self, header):
        # Start by reading the basic WCS stuff that most types have.
        ctype1 = header['CTYPE1']
        ctype2 = header['CTYPE2']
        if ctype1 in [ 'RA---TAN', 'RA---TPV' ]:
            self.wcs_type = ctype1[-3:]
            if ctype2 != 'DEC--' + self.wcs_type:
                raise RuntimeError("ctype1, ctype2 are not as expected")
        else:
            raise RuntimeError("GSFitsWCS cannot read this type of FITS WCS")
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
        # Another strange thing is that the two matrices are define in the opposite order
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

    def _radec(self, x, y):
        import numpy

        # Start with (x,y) = the image position
        p1 = numpy.array( [ x, y ] )

        # This converts to (u,v) in the tangent plane
        p2 = numpy.dot(self.cd, p1 - self.crpix) 

        if self.wcs_type == 'TPV':
            # Now we apply the distortion terms
            u = p2[0]
            v = p2[1]
            usq = u*u
            vsq = v*v
            upow = numpy.array([ 1., u, usq, usq*u ])
            vpow = numpy.array([ 1., v, vsq, vsq*v ])
            p2 = numpy.dot(numpy.dot(self.pv, vpow), upow)

        # Convert (u,v) from degrees to arcsec
        # Also, the FITS standard defines u,v backwards relative to our standard.
        # They have +u increasing to the east, not west.  Hence the -1 for u.
        p2 *= [ -1. * galsim.degrees / galsim.arcsec , 1. * galsim.degrees / galsim.arcsec ]

        # Finally convert from (u,v) to (ra, dec)
        # The TAN projection is also known as a gnomonic projection, which is what
        # we call it in the CelestialCoord class.
        world_pos = self.center.deproject( galsim.PositionD(p2[0],p2[1]) , projection='gnomonic' )
        return world_pos.ra.rad(), world_pos.dec.rad()

    def _posToWorld(self, image_pos):
        x = image_pos.x 
        y = image_pos.y
        ra, dec = self._radec(x,y)
        return galsim.CelestialCoord(ra * galsim.radians, dec * galsim.radians)

    def _posToImage(self, world_pos):
        import numpy, numpy.linalg

        uv = self.center.project( world_pos, projection='gnomonic' )
        # Again, FITS has +u increasing to the east, not west.  Hence the -1.
        u = uv.x * (-1. * galsim.arcsec / galsim.degrees)
        v = uv.y * (1. * galsim.arcsec / galsim.degrees)
        p2 = numpy.array( [ u, v ] )

        if self.wcs_type == 'TPV':
            # Let (s,t) be the current value of (u,v).  Then we want to find a new (u,v) such that
            #
            #       [ s t ] = [ 1 u u^2 u^3 ] pv [ 1 v v^2 v^3 ]^T
            #
            # Start with (u,v) = (s,t)
            #
            # Then use Newton-Raphson iteration to improve (u,v).


            MAX_ITER = 10
            TOL = 1.e-8 * galsim.arcsec / galsim.degrees   # pv always uses degrees units
            prev_err = None
            for iter in range(MAX_ITER):
                usq = u*u
                vsq = v*v
                upow = numpy.array([ 1., u, usq, usq*u ])
                vpow = numpy.array([ 1., v, vsq, vsq*v ])

                diff = numpy.dot(numpy.dot(self.pv, vpow), upow) - p2

                # Check that things are improving...
                err = numpy.max(diff)
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

        return galsim.PositionD(p1[0], p1[1])

    # Override the version in the base class, since we can do this more efficiently.
    def _localFromCelestial(self, image_pos):
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
        # We also need to keep track of the position along the way, so we have to repeat the 
        # steps in _posToWorld.

        import numpy
        p1 = numpy.array( [ image_pos.x, image_pos.y ] )

        p2 = numpy.dot(self.cd, p1 - self.crpix) 
        # The jacobian here is just the cd matrix.
        jac = self.cd

        if self.wcs_type == 'TPV':
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
        # the jacobian of this set in the CelestialCoord class.
        drdu, drdv, dddu, dddv = self.center.deproject_jac( galsim.PositionD(p2[0],p2[1]) ,
                                                            projection='gnomonic' )
        j2 = numpy.array([ [ drdu, drdv ],
                           [ dddu, dddv ] ])
        jac = numpy.dot(j2,jac)

        return galsim.JacobianWCS(jac[0,0], jac[0,1], jac[1,0], jac[1,1])


    def _setOrigin(self, origin):
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
        if self.wcs_type == 'TPV':
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
        if not isinstance(other, GSFitsWCS):
            return False
        else:
            return (
                self.wcs_type == other.wcs_type and
                (self.crpix == other.crpix).all() and
                (self.cd == other.cd).all() and
                self.center == other.center and
                self.pv == other.pv )

    def __ne__(self, other):
        return not self.__eq__(other)

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
                         [ Default `units = galsim.arcsec` ]

    @returns A GSFitsWCS describing this WCS.
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
                    # the standard official WCS types.  So not TPV, for instance.

    PyAstWCS,       # This requires `import starlink.Ast` to succeed.  This handles the largest
                    # number of WCS types of any of these.  In fact, it worked for every one
                    # we tried in our unit tests (which was not exhaustive).

    WcsToolsWCS,    # This requires the wcstool command line functions to be installed.
                    # It is very slow, so it should only be used as a last resort.

    galsim.AffineTransform 
                    # Finally, this one is really the last resort, since it only reads in
                    # the linear part of the WCS.  It defaults to the equivalent of a 
                    # pixel scale of 1.0 if even these are not present.
]

def FitsWCS(file_name=None, dir=None, hdu=None, header=None, compression='auto'):
    """This factory function will try to read the WCS from a FITS file and return a WCS that will 
    work.  It tries a number of different WCS classes until it finds one that succeeds in reading 
    the file.
    
    If none of them work, then the last class it tries, AffineTransform, is guaranteed to succeed, 
    but it will only model the linear portion of the WCS (the CD matrix, CRPIX, and CRVAL), using 
    reasonable defaults if even these are missing.

    Note: The list of classes this function will try may be edited, e.g. by an external module 
    that wants to add an additional WCS type.  The list is `galsim.wcs.fits_wcs_types`.

    @param file_name      The FITS file from which to read the WCS information.  This is probably
                          the usual parameter to provide.  [ Default: `file_name = None` ]
    @param dir            Optional directory to prepend to the file name. [ Default `dir = None` ]
    @param hdu            Optionally, the number of the HDU to use if reading from a file.
                          The default is to use either the primary or first extension as 
                          appropriate for the given compression.  (e.g. for rice, the first 
                          extension is the one you normally want.) [ Default `hdu = None` ]
    @param header         The header of an open pyfits (or astropy.io) hdu.  Or, it can be
                          a galsim.FitsHeader object.  [ Default `header = None` ]
    @param compression    Which decompression scheme to use (if any). See galsim.fits.read
                          for the available options.  [ Default `compression = 'auto'` ]
    """
    if file_name is not None:
        if header is not None:
            raise TypeError("Cannot provide both file_name and pyfits header")
        hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir, hdu, compression)
        header = hdu.header
    else:
        file_name = 'header' # For sensible error messages below.
    if header is None:
        raise TypeError("Must provide either file_name or header")

    for type in fits_wcs_types:
        try:
            wcs = type._readHeader(header)
            return wcs
        except Exception as err:
            #print 'caught ',err
            pass
    raise RuntimeError("All possible fits WCS types failed to read "+file_name)

# Let this function work like a class in config.
FitsWCS._req_params = { "file_name" : str }
FitsWCS._opt_params = { "dir" : str, "hdu" : int, "compression" : str }
FitsWCS._single_params = []
FitsWCS._takes_rng = False
FitsWCS._takes_logger = False


