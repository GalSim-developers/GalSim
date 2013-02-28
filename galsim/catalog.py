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
"""@file catalog.py
Routines for controlling catalog Input/Output with GalSim. 
"""

import galsim

class InputCatalog(object):
    """A class storing the data from an input catalog.

    Each row corresponds to a different object to be built, and each column stores some item of
    information about that object (e.g. flux or half_light_radius).

    After construction, the following fields are available:

        self.nobjects   The number of objects in the catalog
        self.ncols      The number of columns in the catalog
        self.isfits     Whether the catalog is a fits catalog
        self.names      For a fits catalog, the valid column names


    @param file_name     Filename of the input catalog. (Required)
    @param dir           Directory catalog is in.  (default `dir = None`)
    @param file_type     Either 'ASCII' or 'FITS'.  If None, infer from the file name ending.
                         (default `file_type = None`)
    @param comments      The character used to indicate the start of a comment in an
                         ASCII catalog.  (default `comments='#'`)
    @param hdu           Which hdu to use for FITS files.  (default `hdu = 1`)
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'file_type' : str , 'comments' : str , 'hdu' : int }
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, dir=None, file_type=None, comments='#', hdu=1):

        # First build full file_name
        self.file_name = file_name.strip()
        if dir:
            import os
            self.file_name = os.path.join(dir,self.file_name)
    
        if not file_type:
            if self.file_name.lower().endswith('.fits'):
                file_type = 'FITS'
            else:
                file_type = 'ASCII'
        file_type = file_type.upper()
        if file_type not in ['FITS', 'ASCII']:
            raise ValueError("file_type must be either FITS or ASCII if specified.")
        self.file_type = file_type

        try:
            if file_type == 'FITS':
                self.read_fits(hdu)
            else:
                self.read_ascii(comments)
        except Exception, e:
            print e
            raise RuntimeError("Unable to read %s catalog file %s."%(
                    self.file_type, self.file_name))
            
    def read_ascii(self, comments):
        """Read in an input catalog from an ASCII file.
        """
        import numpy
        # Read in the data using the numpy convenience function
        # Note: we leave the data as str, rather than convert to float, so that if
        # we have any str fields, they don't give an error here.  They'll only give an 
        # error if one tries to convert them to float at some point.
        self.data = numpy.loadtxt(self.file_name, comments=comments, dtype=str)
        self.names = None
        self.nobjects = self.data.shape[0]
        self.ncols = self.data.shape[1]
        self.isfits = False

    def read_fits(self, hdu):
        """Read in an input catalog from a FITS file.
        """
        import pyfits
        import numpy
        self.data = pyfits.getdata(self.file_name, hdu)
        self.names = self.data.columns.names
        self.ncols = len(self.names)
        self.nobjects = numpy.min([ len(self.data.field(name)) for name in self.names])
        self.isfits = True

    def get(self, index, col):
        """Return the data for the given index and col as a string

        For ASCII catalogs, col is the column number.  
        For FITS catalogs, col is a string giving the name of the column in the FITS table.
        """
        if self.isfits:
            if col not in self.names:
                raise KeyError("Column %s is invalid for catalog %s"%(col,self.file_name))

            if index < 0 or index >= len(self.data.field(col)):
                raise IndexError("Object %d is invalid for catalog %s"%(index,self.file_name))
            return self.data.field(col)[index]
        else:
            try:
                col = int(col)
            except:
                raise ValueError("For ASCII catalogs, col must be an integer")
            if col < 0 or col >= self.ncols:
                raise IndexError("Column %d is invalid for catalog %s"%(col,self.file_name))
            if index < 0 or index >= self.nobjects:
                raise IndexError("Object %d is invalid for catalog %s"%(index,self.file_name))
            return self.data[index, col]

    def getFloat(self, index, col):
        """Return the data for the given index and col as a float if possible
        """
        try:
            return float(self.get(index,col))
        except:
            raise TypeError("The data at (%d,%d) in catalog %s could not be converted to float"%(
                    index,col,self.file_name))

    def getInt(self, index, col):
        """Return the data for the given index and col as an int if possible
        """
        try:
            return int(self.get(index,col))
        except:
            raise TypeError("The data at (%d,%d) in catalog %s could not be converted to int"%(
                    index,col,self.file_name))

