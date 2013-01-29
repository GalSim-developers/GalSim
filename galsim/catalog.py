"""@file catalog.py
Routines for controlling catalog Input/Output with GalSim. 
"""

import galsim

class InputCatalog(object):
    """A class storing the data from an input catalog.

    Each row corresponds to a different object to be built, and each column stores some item of
    information about that object (e.g. flux or half_light_radius).

    @param file_name     Filename of the input catalog. (Required)
    @param dir           Directory catalog is in.
    @param file_type     Either 'ASCII' (currently the only, default, option) or (soon) 'FITS'.
                         (TODO: default = determine from extension or, if that fails, ASCII)
    @param comments      The character used to indicate the start of a comment in an
                         ASCII catalog.  (default='#').
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'file_type' : str , 'comments' : str }
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, dir=None, file_type='ASCII', comments='#'):
        # First build full file_name
        self.file_name = file_name
        if dir:
            import os
            self.file_name = os.path.join(dir,self.file_name)
    
        # Raise an apologetic exception for FITS input-wanting users
        self.file_type = file_type
        if self.file_type.upper() == 'FITS':
            raise NotImplementedError("FITS catalog inputs not yet implemented, sorry!")
        # Then read in from the ASCII-type catalogs
        elif self.file_type.upper() ==  'ASCII':
            self.read_ascii(comments)
        else:
            raise AttributeError("User must specify input catalog file type as either 'ASCII' "+
                                 "or 'FITS' (case-insensitive).")
        # Also store the number of objects as nobjects for easy access by other routines
        self.nobjects = self.data.shape[0]
        self.ncols = self.data.shape[1]

    def read_ascii(self, comments):
        """Read in an input catalog from an ASCII file.

        Does not check for sensible inputs, leaving this up to the wrapper function read.
        """
        from numpy import loadtxt
        # Read in the data using the numpy convenience function
        # Note: we leave the data as str, rather than convert to float, so that if
        # we have any str fields, they don't give an error here.  They'll only give an 
        # error if one tries to convert them to float at some point.
        self.data = loadtxt(self.file_name, comments=comments, dtype=str)

    def nObjects(self):
        """Return the number of objects in the catalog
        """
        return self.nobjects

    def nCols(self):
        """Return the number of columns in the catalog
        """
        return self.ncols

    def get(self, index, col):
        """Return the data for the given index and col as a string
        """
        if index < 0 or index >= self.nobjects:
            raise ValueError("Object %d is invalid for catalog %s"%(index,self.file_name))
        if col < 0 or col >= self.ncols:
            raise ValueError("Column %d is invalid for catalog %s"%(col,self.file_name))
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


