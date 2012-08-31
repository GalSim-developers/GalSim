import galsim

class InputCatalog(object):
    """@brief A class storing the data from an input catalog where each row corresponds to 
    a different object to be built, and each column stores some item of information about
    that object (e.g. flux or half_light_radius).
    """
    def __init__(self, config):
        """
        @param config   A configuration dict containing the parameters required for
                        reading in the input cat. 
    
        The configuration paramters that may be set:

        dir           Directory catalog is in. (default = .)
        file_name     Filename of the input catalog. (Required)
        file_type     Either 'ASCII' (currently the only, default, option) or (soon) 'FITS'.
                    (default = determine from extension or, if that fails, ASCII)
        comments      The character used to indicate the start of a comment in an ASCII catalog.
                    (default='#').

        Does some checking for sensible inputs, unlike the functions it calls (ReadAsciiInputCat()
        and ReadFitsInputCat()).
        """
        # First check for sensible inputs
        if 'file_name' not in config:
            raise AttributeError("file_name is required for input catalog")
        self.file_name = config['file_name']
        if 'dir' in config:
            import os
            dir = config['dir']
            self.file_name = os.path.join(dir,self.file_name)
    
        # Raise an apologetic exception for FITS input-wanting users
        self.file_type = config.get('file_type','ASCII')
        if self.file_type.upper() == 'FITS':
            raise NotImplementedError("FITS catalog inputs not yet implemented, sorry!")
        # Then read in from the ASCII-type catalogs
        elif self.file_type.upper() ==  'ASCII':
            self.read_ascii(config)
        else:
            raise AttributeError("User must specify input catalog file type as either 'ASCII' "+
                                "or 'FITS' (case-insensitive).")

    def read_ascii(self, config):
        """@brief Read in an input catalog from an ASCII file.

        Does not check for sensible inputs, leaving this up to the wrapper function read.
        """
        from numpy import loadtxt
        # Read in the data using the numpy convenience function
        comments = config.get('comments','#')
        self.data = loadtxt(self.file_name, comments=comments)
        # Also store the number of objects as nobjects for easy access by other routines
        self.nobjects = self.data.shape[0]
        self.ncols = self.data.shape[1]
        # Store that this is an ASCII catalog
        self.file_type = 'ASCII'
