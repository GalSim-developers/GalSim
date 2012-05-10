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
    if not config.hasattr('type'):
        if logger != None:
            logger.info("Error: type attribute required")
        raise AttributeError("type attribute required")
    return eval('galsim.Build' + config.type + '(config, input_cat, logger)')

def BuildSimple(config, req=[], size_opt=[], opt=[], input_cat=None, logger=None):
    """Most of the functionality of the Build function is the same for the simple
       objects that are just a profile.  So encapsulate all that here.
       
       @param req       A list of required attributes that config must have
       @param size_opt  A list of size attributes, of which 1 (and only 1) is required
       @param opt       A list of optional attributes
       In addition to what is listed, the flux is always optional.
    """
    #print 'Start BuildSimple for ',config.type
    # All simple builders have an optional flux attribute so add that to opt
    opt += ['flux']

    # Make the argument list for the constructor
    kwargs = {}
    for key in req:
        #print 'req key = ',key
        if not config.hasattr(key):
            if logger != None:
                logger.info("Error: %s requires the following attributes: %s",config.type,req)
            raise AttributeError()
        value = Generate(eval("config." + key),input_cat,logger)
        #print 'value = ',value
        kwargs[key] = value

    for key in opt:
        if config.hasattr(key):
            #print 'opt key = ',key
            value = Generate(eval("config." + key),input_cat,logger)
            #print 'value = ',value
            kwargs[key] = value

    # Make sure one and only one size is present
    found = False
    for key in size_opt:
        #print 'size key = ',key
        if config.hasattr(key):
            value = Generate(eval("config." + key),input_cat,logger)
            #print 'value = ',value
            if (found):
                if logger != None:
                    logger.info("Error: %s requires exactly one of the following attributes: %s",
                        config.type,size_opt)
                raise AttributeError("Too many sizes for %s"%config.type)
            kwargs[key] = value
            found = True
    if not found:
        if logger != None:
            logger.info("Error: %s requires one of the following attributes: %s",
                config.type,size_opt)
        raise AttributeError("No size specified for %s"%config.type)

    # Now ready to call the constructor
    return eval("galsim."+config.type+"(**kwargs)")

def BuildGaussian(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, [], ['sigma','fwhm','half_light_radius'], 
        [], input_cat, logger)

def BuildMoffat(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, ['beta'], ['fwhm','scale_radius','half_light_radius'],
        ['trunc'], input_cat, logger)

def BuildSersic(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, ['n'], ['half_light_radius'], [], input_cat, logger)

def BuildExponential(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, [], ['half_light_radius','scale_radius'], [],
        input_cat, logger)

def BuildDeVaucouleurs(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, [], ['half_light_radius'], [], input_cat, logger)

def BuildAiry(config, input_cat=None, logger=None):
    return galsim.BuildSimple(config, [], ['D'], ['obs'], input_cat, logger)

def BuildPixel(config, input_cat=None, logger=None):
    #print 'Start BuildPixel'

    for key in ['xw','yw']:
        if not config.hasattr(key):
            if logger != None:
                logger.info('Error: Pixel requires the following attributes: %s',[xw, yw])
            raise AttributeError('Pixel requires attribute %s'%key)
    kwargs = {}
    kwargs['xw'] = Generate(config.xw,input_cat,logger)
    kwargs['yw'] = Generate(config.yw,input_cat,logger)

    if (xw != yw):
        raise Warning("xw != yw found (%f != %f) "%(xw,yw) +
            "This is supported for the pixel, but not the draw routines. " +
            "There might be weirdness....")
    return galsim.Pixel(xw=xw,yw=yw)

    if config.hasattr('flux'):
        kwargs['flux'] = Generate(config.flux,input_cat,logger)

    return galsim.Pixel(**kwargs)

def BuildSquarePixel(config, input_cat=None, logger=None):
    #print 'Start BuildSquarePixel'

    if not config.hasattr('size'):
        if logger != None:
            logger.info('Error: SquarePixel requires attribute size')
        raise AttributeError('SquarePixel requires attribute size')
    kwargs = {}
    kwargs['xw'] = Generate(config.size,input_cat,logger)
    if config.hasattr('flux'):
        kwargs['flux'] = Generate(config.flux,input_cat,logger)
    return galsim.Pixel(**kwargs)

def BuildSum(config, input_cat=None, logger=None):
    #print 'Start BuildSum'

    if not config.hasattr('items'):
        if logger != None:
            logger.info('Error: Sum requires attribute items')
        raise AttributeError('Sum requires attribute items')
    list = []
    for item in config.items:
        list += [ BuildGSObject(item, input_cat, logger=None) ]
    return galsim.Add(list)

def BuildConvolve(config, input_cat=None, logger=None):
    #print 'Start BuildConvolve'

    if not config.hasattr('items'):
        if logger != None:
            logger.info('Error: Convolve requires attribute items')
        raise AttributeError('Convolve requires attribute items')
    list = []
    for item in config.items:
        list += [ BuildGSObject(item, input_cat, logger=None) ]
    return galsim.Convolve(list)




def Generate(config, input_cat=None, logger=None):
    #print 'Start Generate with config = ',config
    try:
        if config.hasattr('type'):
            return eval('galsim.GenerateFrom' + config.type + '(config, input_cat, logger)')
    except AttributeError:
        pass
    # else assume config is really a value.
    return config

def GenerateFromInputCatalog(config, input_cat, logger=None):
    #print 'Start GenerateFromInputCatalog'
    if input_cat is None:
        raise ValueError("Use of InputCatalog requested, but no input_cat given")

    if not config.hasattr('col'):
        if logger != None:
            logger.info("Error: InputCatalog requires col attribute",config.type)
        raise AttributeError("No col specified for InputCatalog")
    col = config.col
    #print 'col = ',col

    # input_cat stores the current row to use.
    current = input_cat.current
    #print 'current = ',current

    if current >= input_cat.nobjects:
        raise ValueError("Trying to access past the end of the catalog data.")
    #TODO: Add a similar check on the column?

    if input_cat.type is 'ASCII':

        try:
            # config values are 1-based, but we access is 0-based, so use col-1
            value = input_cat.data[current,col-1]
            #print 'value = ',value
        except TypeError:
            if logger != None:
                logger.info("Error: col should be an integer, but is %s",col)
            raise

    elif input_cat.type is 'FITS':
        raise NotImplementedError("FITS catalog inputs not yet implemented, sorry!")

    else:
        raise NotImplementedError("Unknown catalog type %s"%input_cat.type)

    return value
