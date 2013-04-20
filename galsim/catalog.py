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

        self.nobjects   The number of objects in the catalog.
        self.ncols      The number of columns in the catalog.
        self.isfits     Whether the catalog is a fits catalog.
        self.names      For a fits catalog, the valid column names.


    @param file_name     Filename of the input catalog. (Required)
    @param dir           Optionally a directory name can be provided if the file_name does not 
                         already include it.
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

    # nobjects_only is an intentionally undocumented kwarg that should be used only by
    # the config structure.  It indicates that all we care about is the nobjects parameter.
    # So skip any other calculations that might normally be necessary on construction.
    def __init__(self, file_name, dir=None, file_type=None, comments='#', hdu=1,
                 nobjects_only=False):

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

        if file_type == 'FITS':
            self.read_fits(hdu, nobjects_only)
        else:
            self.read_ascii(comments, nobjects_only)
            
    def read_ascii(self, comments, nobjects_only):
        """Read in an input catalog from an ASCII file.
        """
        # If all we care about is nobjects, this is quicker:
        if nobjects_only:
            # See the script devel/testlinecounting.py that tests several possibilities.
            # An even faster version using buffering is possible although it requires some care
            # around edge cases, so we use this one instead, which is "correct by inspection".
            f = open(self.file_name)
            if (len(comments) == 1):
                c = comments[0]
                self.nobjects = sum(1 for line in f if line[0] != c)
            else:
                self.nobjects = sum(1 for line in f if not line.startswith(comments))
            return

        import numpy
        # Read in the data using the numpy convenience function
        # Note: we leave the data as str, rather than convert to float, so that if
        # we have any str fields, they don't give an error here.  They'll only give an 
        # error if one tries to convert them to float at some point.
        self.data = numpy.loadtxt(self.file_name, comments=comments, dtype=str)
        # If only one row, then the shape comes in as one-d.
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(1, -1)
        if len(self.data.shape) != 2:
            raise IOError('Unable to parse the input catalog as a 2-d array')

        self.nobjects = self.data.shape[0]
        self.ncols = self.data.shape[1]
        self.isfits = False

    def read_fits(self, hdu, nobjects_only):
        """Read in an input catalog from a FITS file.
        """
        import pyfits
        raw_data = pyfits.getdata(self.file_name, hdu)
        if pyfits.__version__ > '3.0':
            self.names = raw_data.columns.names
        else:
            self.names = raw_data.dtype.names
        self.nobjects = len(raw_data.field(self.names[0]))
        if (nobjects_only): return
        # The pyfits raw_data is a FITS_rec object, which isn't picklable, so we need to 
        # copy the fields into a new structure to make sure our InputCatalog is picklable.
        # The simplest is probably a dict keyed by the field names, which we save as self.data.
        self.data = {}
        for name in self.names:
            self.data[name] = raw_data.field(name)
        self.ncols = len(self.names)
        self.isfits = True

    def get(self, index, col):
        """Return the data for the given index and col in its native type.

        For ASCII catalogs, col is the column number.  
        For FITS catalogs, col is a string giving the name of the column in the FITS table.

        Also, for ASCII catalogs, the "native type" is always str.  For FITS catalogs, it is 
        whatever type is specified for each field in the binary table.
        """
        if self.isfits:
            if col not in self.names:
                raise KeyError("Column %s is invalid for catalog %s"%(col,self.file_name))
            if index < 0 or index >= self.nobjects:
                raise IndexError("Object %d is invalid for catalog %s"%(index,self.file_name))
            if index >= len(self.data[col]):
                raise IndexError("Object %d is invalid for column %s"%(index,col))
            return self.data[col][index]
        else:
            icol = int(col)
            if icol < 0 or icol >= self.ncols:
                raise IndexError("Column %d is invalid for catalog %s"%(icol,self.file_name))
            if index < 0 or index >= self.nobjects:
                raise IndexError("Object %d is invalid for catalog %s"%(index,self.file_name))
            return self.data[index, icol]

    def getFloat(self, index, col):
        """Return the data for the given index and col as a float if possible
        """
        return float(self.get(index,col))

    def getInt(self, index, col):
        """Return the data for the given index and col as an int if possible
        """
        return int(self.get(index,col))

