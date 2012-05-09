import galsim

def read_input_cat(filename=None, filetype="ASCII", ascii_fields=None, comments="#"):
    """@brief Read in an input catalog for object-by-object parameter specification.

    @param filename      Filename of the input catalog.
    @param filetype      Either 'ASCII' (currently the only, default, option) or (soon) 'FITS'.
    @param ascii_fields  A list containing the strings that represent the field name of each column
                         in an input ASCII catalog.  Must be provided for filetype='ASCII'.
    @param comments      The character used to indicate the start of a comment in an ASCII catalog
                         (default='#').

    @returns An AttributeDict instance, each attribute of which is a vector of parameter values of
             length equal to the number of valid rows in the input catalog.

    Does some checking for sensible inputs, unlike the functions it calls (read_ascii_input_cat()
    and read_fits_input_cat()).
    """

    # First check for sensible inputs
    if filename == None:
        raise IOError("No filename given!")

    # Raise an apologetic exception for FITS input-wanting users
    if filetype == "FITS":
        raise NotImplementedError("FITS catalog inputs not yet implemented, sorry!")
    
    # Then read in from the ASCII-type catalogs
    if filetype == "ASCII":
        # Raise an error if ASCII is given as the type and there is no ascii_fields kwarg
        if ascii_fields == None:
            raise ValueError("Must currently supply an ascii_fields list keyword if reading ASCII"
                             " files")
        else:
            input_cat = read_ascii_input_cat(filename=filename, ascii_fields=ascii_fields,
                                              comments=comments)
    # Return catalog to the user
    return input_cat

def read_ascii_input_cat(filename=None, ascii_fields=None, comments="#"):
    """@brief Read in an input catalog from an ASCII file.

    @param filename      Filename of the input catalog.
    @param ascii_fields  A list containing the strings that represent the field name of each column
                         in an input ASCII catalog.  Must be provided for filetype='ASCII'.
    @param comments      The character used to indicate the start of a comment in an ASCII catalog
                         (default='#').

    @returns An AttributeDict instance, each attribute of which is a vector of parameter values of
             length equal to the number of valid rows in the input catalog.

    Does not check for sensible inputs, leaving this up to the wrapper function read_input_cat().
    """
    
    from numpy import loadtxt
    # Read in the data using the numpy convenience function
    data = loadtxt(filename, comments=comments)
    # Test the shape is right
    nfields = len(ascii_fields)
    if data.shape[1] < nfields:
        raise IOError("Input ASCII catalog must have at least as many columns as the ascii_fields "
                      "keyword has entries.")
    
    # Initialize the AttributeDict() ready for storing the field values
    input_cat = galsim.AttributeDict()
    # Store basic data
    input_cat.filename = filename
    input_cat.ascii_fields = ascii_fields
    # Always store the number of objects as input_cat.nobjects for easy access by other routines
    input_cat.nobjects = data.shape[0]
    # Run through the fields in ascii_fields and add the column entries to the output
    for i in range(nfields):
        # Test for None elements (this means ingore that column in the input cat)
        if ascii_fields[i] == None:
            pass
        else:
            # Give the input_cat a new attribute containing these data vectors
            input_cat.__setattr__(ascii_fields[i], data[:, i])
    # Return catalog to the user
    return input_cat


