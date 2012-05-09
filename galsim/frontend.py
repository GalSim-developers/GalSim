import galsim


def build_image_output(config=None, input_cat=None, logger=None):
    """Build output galaxy and/or PSF image using an config and input_cat AttributeDict().

    @param config     A configuration AttributDict() read in using galsim.config.load().
    @param input_cat  An input catalog AttributeDict() read in using galsim.io.read_input_cat().
    @param logger     Output logging object.
    """
    # Some sanity checks
    if config == None and input_cat == None:
        raise ValueError("Cannot build objects without input config file or catalog!")
    
    # Test to see whether to build *only* PSF images
    if config.output.PSFs.only == True:
        psfimage = build_psf_image(config, input_cat)
        if logger != None:
            logger.info("Writing PSF image to "+config.output.PSFs.filename)
        psfimage.write(config.output.PSFs.filename)
    else:
        # Go ahead and build galaxy images
        galimage = build_galaxy_image(config, input_cat)
        if logger != None:
            logger.info("Writing galaxy image to "+config.output.filename)
        galimage.write(config.output.filename)
        # Test to see whether to build PSF image
        if config.output.PSFs.only == True:
            psfimage = build_psf_image(config, input_cat)
            if logger != None:
                logger.info("Writing PSF image to "+config.output.PSFs.filename)
            psfimage.write(config.output.PSFs.filename)
    return logger

def build_galaxy_image(config=None, input_cat=None, logger=None):
    """Build output galaxy image using an config and input_cat AttributeDict().

    @param config     A configuration AttributDict() read in using galsim.config.load().
    @param input_cat  An input catalog AttributeDict() read in using galsim.io.read_input_cat().
    @param logger     Output logging object.
    """
    # Get the number of objects
    if input_cat != None:
        # The input catalogue generally takes precedence over the configuration file.
        nobjects = input_cat.nobjects
    else:
        try:
            nobjects = config.nobjects
        except AttributeError:
            if logger != None:
                logger.info("Error: no config.nobjects in config and no input_cat given.")
                raise
            
    # Then set up the basic galaxy types with default parameter values (these will be changed later
    # as required):
    try:
        types = config.galaxy.type
    except AttributeError:
        if logger != None:
            logger.info("Error: no config.galaxy.type given.")
            raise
    ntypes = len(types)
    for galtype in types:
        exec "basic_"+galtype+" = galsim."+galtype+"()"

    # Having setup the default galaxy types, then loop through the nobjects, setting parameter
    # values as found in the input_cat by precedence over those in the config file.  Finally, sum
    # these objects to create the output.
    for i in xrange(nobjects):

        for galtype in types:

            exec "config_params = config.galaxy."+galtype+".__dict__"
            # Set via the config first, then overwrite using the
            for param in config_params:
                print galtype+"."+param

            exec "input_cat_params = input_cat."+galtype+".__dict__"
            for param in input_cat_params:
                print galtype+"."+param
            
    return

def build_psf_image(config, input_cat, logger):
    return
