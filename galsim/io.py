import galsim

def ReadInputCat(config, cat_file_name=None, filetype="ASCII", comments="#"):
    """@brief Read in an input catalog for object-by-object parameter specification.

    @param config         A configuration galsim.Config instance containing all the required
                          information for running GalSim in Multi-Object mode.
    @param cat_file_name  Filename of the input catalog.
    @param filetype       Either 'ASCII' (currently the only, default, option) or (soon) 'FITS'.
    @param comments       The character used to indicate the start of a comment in an ASCII catalog
                          (default='#').

    @returns A galsim.Config instance, each attribute of which is a vector of parameter values of
             length equal to the number of valid rows in the input catalog.

    Does some checking for sensible inputs, unlike the functions it calls (ReadAsciiInputCat()
    and ReadFitsInputCat()).
    """
    # First check for sensible inputs
    if cat_file_name == None:
        raise IOError("No filename given!")

    # Raise an apologetic exception for FITS input-wanting users
    if filetype.upper() == "FITS":
        raise NotImplementedError("FITS catalog inputs not yet implemented, sorry!")
    # Then read in from the ASCII-type catalogs
    elif filetype.upper() ==  "ASCII":
        input_cat = ReadAsciiInputCat(cat_file_name=cat_file_name, comments=comments)
    else:
        raise AttributeError("User must specify input catalog file type as either 'ASCII' "+
                             "or 'FITS' (case-insensitive).")
    # Return catalog to the user
    return input_cat

def ReadAsciiInputCat(cat_file_name=None, comments="#"):
    """@brief Read in an input catalog from an ASCII file.

    @param cat_file_name  Filename of the input catalog.
    @param comments       The character used to indicate the start of a comment in an ASCII catalog
                          (default='#').

    @returns A galsim.Config instance, containing useful metadata and the ASCII table itself stored
             as a NumPy array in the .data attribute.

    Does not check for sensible inputs, leaving this up to the wrapper function ReadInputCat().
    """
    from numpy import loadtxt
    # Initialize the Config() ready for storing the field values
    input_cat = galsim.Config()
    # Store basic meta data
    input_cat.cat_file_name = cat_file_name
    # Read in the data using the numpy convenience function
    input_cat.data = loadtxt(cat_file_name, comments=comments)
    # Also store the number of objects as input_cat.nobjects for easy access by other routines
    input_cat.nobjects = input_cat.data.shape[0]
    # Store that this is an ASCII catalog
    input_cat.type = 'ASCII'
    # Keep track of which row we should read from.  Start at 0.
    input_cat.current = 0
    # Return catalog to the user
    return input_cat


