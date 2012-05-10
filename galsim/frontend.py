import galsim
op_dict = galsim.object_param_dict

# USERS NOTE - THIS IS CURRENTLY IN DEVELOPMENT AND UNFINISHED!
#

def BuildGSOBject(config, input_cat=None, object_id=None, logger=None):
    """Build GSObjects from input args given in config and input_cat structures.
    """
    # Check that the input config has a type to even begin with!
    try:
        type = config.type
    except AttributeError:
        if logger != None:
            logger.info("Error: type attribute required in config.")
        raise
    if config.type in op_dict:
        gsobject = BuildSingle(config, input_cat, object_id, logger)
    elif config.type in ("Sum", "Convolution"):
        gsobjects = []
        for i in config.nitems:
            gsobjects.append(BuildSingle(config.item[i], input_cat, object_id, logger))
        if config.type == "Sum":
            gsobject = galsim.Add(gsobjects)
        else config.type == "Convolution":
            gsobject = galsim.Convolve(gsobjects)
    else:
        raise NotImplementedError("Unrecognised config.type = "+str(config.type))
    return gsobject

def BuildSingle(config, input_cat=None, object_id=None, logger=None):
    """@brief Construct simple GSObjects, not Sums, Convolutions or other compunds.

    Does basic error handling.
    """
    # First do a no-brainer checks
    if not hasattr(config, "type"):
        raise AttributeError("No type attribute in config!")
    
    init_params = []
    for param_name in _op_dict[config.type]:

        # First of all try and get the correctly named attribute
        try:
            param = config.__getattr__(param_name)
        except AttributeError:
            if logger != None:
                logger.info("Error: "+param_name+" attribute required in config for "
                            "initializing "+type+" objects.")
            raise
            
        # Now we have it, see if this param has a type (if not interpret as fixed scalar constant)
        if hasattr(param, "type"):
            # If it's an InputCatalog, look it up from the catalog data
            if param.type == "InputCatalog":
                # Then check for correct input kwargs and give a useful message
                if input_cat == None or object_id == None:
                    raise ValueError("Either input_cat or object_id not given on input, and "+
                                     "the config requires an InputCatalog entry for "+type+"."+
                                     param_name)
                # Then set the param_value from the requisite [object_id, col] entry in the data
                # ...if this fails, try to work out why and give info.
                try:
                    col = param.col
                    param_value = input_cat.data[object_id, col]
                except AttributeError:
                    if logger != None:
                        logger.info("Error: "+param_name+".col attribute required in config "
                                    "for initializing with "+param_name+".type = InputCatalog.")
                    raise
                except IndexError:
                    if logger != None:
                        logger.info("Error: "+param_name+".col attribute or object_id out of "+
                                    "bounds for accessing input_cat.data [col, object_id] = "+
                                    "["+str(param.col)+", "+str(object_id)+"]")
                    raise
            else:
                raise NotImplementedError("Sorry, only InputCatalog config types are "+
                                          "currently implemented.")
        else: # Do a straight assignment from the param itself
            param_value = param
            
        # Then append this param value to the list for passing to the GSObject __init__
        init_params.append(param_value)

    init_func = eval("galsim."+config.type)
    return init_func(*init_params)


#def build_image_output(config=None, input_cat=None, logger=None):
#    """Build output galaxy and/or PSF image using an config and input_cat AttributeDict().
#
#    @param config     A configuration AttributDict() read in using galsim.config.load().
#    @param input_cat  An input catalog AttributeDict() read in using galsim.io.read_input_cat().
#    @param logger     Output logging object.
#    """
#    # Some sanity checks
#    if config == None and input_cat == None:
#        raise ValueError("Cannot build objects without input config file or catalog!")
#    
#    # Test to see whether to build *only* PSF images
#    if config.output.PSFs.only == True:
#        psfimage = build_psf_image(config, input_cat)
#        if logger != None:
#            logger.info("Writing PSF image to "+config.output.PSFs.filename)
#        psfimage.write(config.output.PSFs.filename)
#    else:
#        # Go ahead and build galaxy images
#        galimage = build_galaxy_image(config, input_cat)
#        if logger != None:
#            logger.info("Writing galaxy image to "+config.output.filename)
#        galimage.write(config.output.filename)
#        # Test to see whether to build PSF image
#        if config.output.PSFs.only == True:
#            psfimage = build_psf_image(config, input_cat)
#            if logger != None:
#                logger.info("Writing PSF image to "+config.output.PSFs.filename)
#            psfimage.write(config.output.PSFs.filename)
#    return logger
#
#def build_galaxy_image(config=None, input_cat=None, logger=None):
#    """Build output galaxy image using an config and input_cat AttributeDict().
#
#    @param config     A configuration AttributDict() read in using galsim.config.load().
#    @param input_cat  An input catalog AttributeDict() read in using galsim.io.read_input_cat().
#    @param logger     Output logging object.
#    """
#    # Get the number of objects
#    if input_cat != None:
#        # The input catalogue generally takes precedence over the configuration file.
#        nobjects = input_cat.nobjects
#    else:
#        try:
#            nobjects = config.nobjects
#        except AttributeError:
#            if logger != None:
#                logger.info("Error: no config.nobjects in config and no input_cat given.")
#            raise   # ooops this was wrongly indented earlier!
#            
#    # Then set up the basic galaxy types with default parameter values (these will be changed later
#    # as required):
#    try:
#        types = config.galaxy.type
#    except AttributeError:
#        if logger != None:
#            logger.info("Error: no config.galaxy.type given.")
#        raise
#    ntypes = len(types)
#    for galtype in types:
#        exec "basic_"+galtype+" = galsim."+galtype+"()"
#
#    # Having setup the default galaxy types, then loop through the nobjects, setting parameter
#    # values as found in the input_cat by precedence over those in the config file.  Finally, sum
#    # these objects to create the output.
#    for i in xrange(nobjects):
#
#        for galtype in types:
#
#            exec "config_params = config.galaxy."+galtype+".__dict__"
#            # Set via the config first, then overwrite using the
#            for param in config_params:
#                print galtype+"."+param
#
#            exec "input_cat_params = input_cat."+galtype+".__dict__"
#            for param in input_cat_params:
#                print galtype+"."+param
#            
#    return
#
#def build_psf_image(config, input_cat, logger):
#    return


