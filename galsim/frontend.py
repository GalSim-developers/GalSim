import galsim

# USERS NOTE - THIS IS CURRENTLY IN DEVELOPMENT AND UNFINISHED!
#


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

def BuildGSObject(config, input_cat=None, logger=None):
    """Build a GSObject using a config (AttributeDict) and possibly an input_cat (AttributeDict).

    @param config     A configuration AttributDict() read in using galsim.config.load().
    @param input_cat  An input catalog AttributeDict() read in using galsim.io.read_input_cat().
    @param logger     Output logging object.
    """
    try:
        type = config.type
    except AttributeError:
        if logger != None:
            logger.info("Error: type attribute required")
            raise
    return eval('galsim.Build' + type + '(config, input_cat, logger)')

def BuildSimple(config, req=[], size_opt=[], opt=[], input_cat=None, logger=None):
    """Most of the functionality of the Build function is the same for the simple
       objects that are just a profile.  So encapsulate all that here.
       
       @param req       A list of required attributes that config must have
       @param size_opt  A list of size attributes, of which 1 (and only 1) is required
       @param opt       A list of optional attributes
       In addition to what is listed, the flux is always optional.
    """
    kwargs = {}
    try:
        for key in req:
            value = Generate(eval("config." + key),input_cat,logger)
            kwargs += { key : value }
    except AttributeError:
        if logger != None:
            logger.info("Error: %s requires the following attributes: %s",config.type,req)
            raise
    for key in opt:
        try:
            value = Generate(eval("config." + key),input_cat,logger)
            kwargs += { key : value }
        except AttributeError:
            pass
    found = False
    for key in size_opt:
        try:
            value = Generate(eval("config." + key),input_cat,logger)
            if (found):
                logger.info("Error: %s requires exactly one of the following attributes: %s",
                    config.type,size_opt)
                raise AttributeError("Too many sizes for %s"%config.type)
            kwargs += { key : value }
            found = True
        except AttributeError:
            pass
    if not found:
        logger.info("Error: %s requires one of the following attributes: %s",config.type,size_opt)

    return eval("galsim."+config.type+"(**kwargs)")

def BuildGaussian(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, [], ['sigma'], [], input_cat, logger)

def BuildMoffat(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, ['beta'], ['re'], ['trunc'], input_cat, logger)

def BuildSersic(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, ['n'], ['re'], [], input_cat, logger)

def BuildExponential(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, [], ['r0'], [], input_cat, logger)

def BuildDeVaucouleurs(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, [], ['re'], [], input_cat, logger)

def BuildAiry(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, [], ['D'], ['obs'], input_cat, logger)

