import galsim

class InputCatalog(object):
    """@brief A class storing the data from an input catalog where each row corresponds to 
    a different object to be built, and each column stores some item of information about
    that object (e.g. flux or half_light_radius).
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'file_type' : str , 'comments' : str }
    _single_params = []

    def __init__(self, file_name, dir=None, file_type='ASCII', comments='#'):
        """
        @param file_name     Filename of the input catalog. (Required)
        @param dir           Directory catalog is in. (default = .)
        @param file_type     Either 'ASCII' (currently the only, default, option) or (soon) 'FITS'.
                             (TODO: default = determine from extension or, if that fails, ASCII)
        @param comments      The character used to indicate the start of a comment in an 
                             ASCII catalog.  (default='#').
        """
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
        """@brief Read in an input catalog from an ASCII file.

        Does not check for sensible inputs, leaving this up to the wrapper function read.
        """
        from numpy import loadtxt
        # Read in the data using the numpy convenience function
        # Note: we leave the data as str, rather than convert to float, so that if
        # we have any str fields, they don't give an error here.  They'll only give an 
        # error if one tries to convert them to float at some point.
        self.data = loadtxt(self.file_name, comments=comments, dtype=str)
