import galsim
op_dict = galsim.object_param_dict

# USERS NOTE - THIS IS CURRENTLY IN DEVELOPMENT AND UNFINISHED!
#

def BuildGSObject(config, input_cat=None, logger=None):
    """Build a GSObject using a config (AttributeDict) and possibly an input_cat (AttributeDict).

    @param config     A configuration AttributDict() read in using galsim.config.load().
    @param input_cat  An input catalog AttributeDict() read in using galsim.io.read_input_cat().
    @param logger     Output logging object (NOT USED IN THIS IMPLEMENTATION: RAISED ERRORS
                      AUTOMATICALLY PASSED TO LOGGER)
    """

    # Check that the input config has a type to even begin with!
    if not "type" in config.__dict__:
        raise AttributeError("type attribute required in config.")
    # Then build object depending on type
    if config.type in op_dict:  # Single object from primary keys in galsim.object_param_dict
        gsobject = _BuildSingle(config, input_cat)
    elif config.type in ("Sum", "Convolution"): # Compound object
        gsobjects = []
        # MJ: Bug here. Should check if items is in config.__dict__.  
        # Else will automatically create one and give a confusing error.
        for i in range(len(config.items)):
            gsobjects.append(_BuildSingle(config.items[i], input_cat))
        if config.type == "Sum":
            gsobject = galsim.Add(gsobjects)
        elif config.type == "Convolve":
            gsobject = galsim.Convolve(gsobjects)
    # MJ: Should pull out Pixel separately as well, since for that the sizes work differently
    # I think we want both to be required, although maybe ok if only one.  But at least
    # having both xw and yw must be allowed.  I raise a warning when both are present
    # but unequal, since I don't think this is fully supported by GalSim yet.
    elif config.type == "SquarePixel":  # Mike is treating Pixels separately
        if not "size" in config.__dict__:
            raise AttributeError("size attribute required in config for initializing SquarePixel "+
                                 "objects.")
        init_kwargs = {"xw": config.size, "yw": config.size}
        if "flux" in config.__dict__:
            init_kwargs["flux"] = config.flux
        gsobject = galsim.Pixel(**init_kwargs)
    else:
        raise NotImplementedError("Unrecognised config.type = "+str(config.type))
    return gsobject

def _BuildSingle(config, input_cat=None):
    """@brief Construct simple GSObjects (i.e. not Sums, Convolutions or other compounds).
    """
    # First do a no-brainer checks
    if not "type" in config.__dict__:
        raise AttributeError("No type attribute in config!")
    init_kwargs = {}
    init_kwargs.update(_GetRequiredKwargs(config, input_cat))
    init_kwargs.update(_GetSizeKwarg(config, input_cat))
    init_kwargs.update(_GetOptionalKwargs(config, input_cat))
    # Finally, after pulling together all the params, try making the GSObject.
    # Check for TypeErrors (sign of multiple radius definitions being passed, among other problems).
    init_func = eval("galsim."+config.type)
    try:
        gsobject = init_func(**init_kwargs)
    except Error, err_msg:
        raise RuntimeError("Problem sending init_kwargs to galsim."+config.type+" object. "+
                         "Original error message: "+err_msg)
    return gsobject


def _GetRequiredKwargs(config, input_cat=None):
    """@brief Get the required kwargs.
    """
    req_kwargs = {}
    for req_name in op_dict[config.type]["required"]:
        # Sanity check here, as far upstream as possible
        if not req_name in config.__dict__:
            raise AttributeError("No required attribute "+req_name+" within input config.")
        else:
            req_kwargs[req_name] = _GetParamValue(config, req_name, input_cat=input_cat)
    return req_kwargs

def _GetSizeKwarg(config, input_cat=None):
    """@brief Get the one, and one only, required size kwarg.
    """
    size_kwarg = {}
    counter = 0
    for size_name in op_dict[config.type]["size"]:
        if size_name in config.__dict__:
            counter += 1
            if counter == 1:
                size_kwarg[size_name] = _GetParamValue(config, size_name, input_cat=input_cat)
            elif counter > 1:
                raise ValueError("More than one size parameter specified for")
    # MJ: Check for counter == 0 here?
    return size_kwarg

def _GetOptionalKwargs(config, input_cat=None):
    """@brief Get the optional kwargs, if any present in the config.
    """
    optional_kwargs = {}
    for entry_name in config.__dict__:
        if entry_name in op_dict[config.type]["optional"]:
            optional_kwargs[entry_name] = _GetParamValue(config, entry_name, input_cat=input_cat)
    return optional_kwargs

def _GetParamValue(config, param_name, input_cat=None):
    """@brief Function to read parameter values from config AttributeDicts.
    """
   # Assume that basic sanity checking done upstream for maximum efficiency 
    param = config.__getattr__(param_name)
    # First see if we can assign by direct value
    if not hasattr(param, "__dict__"):  # This already exists for AttributeDicts, not for values
        param_value = param
    elif "type" in param.__dict__:  # Explore type to decide next steps..
        # First check if it's an InputCatalog, & look it up from the catalog data.
        if param.type == "InputCatalog":
            # Then check for correct input kwargs and give a useful message
            if input_cat == None:
                    raise ValueError("Keyword input_cat not given to GetParamValue:  the config "+
                                     "requires an InputCatalog entry for "+param_name)
            # If OK, set the param_value from the requisite [input_cat.current, col] entry in the
            # input_cat.data
            # ...if this fails, try to work out why and give info.
            if "col" in param.__dict__:
                col = param.col
            else:
                raise AttributeError(param_name+".col attribute required in config for "+
                                     "initializing with "+param_name+".type = InputCatalog.")
            # MJ: Should query input_cat.type == 'ASCII' here.  FITS will be different.
            try:
                # MJ: You want col-1 here.  I adpoted convention that first column is 1
                # in the config definition.  But numpy wants 0-based.
                param_value = input_cat.data[input_cat.current, col]
            except IndexError:
                raise IndexError(param_name+".col attribute or input_cat.current out of bounds "+
                                 " for accessing input_cat.data [col, object_id] = "+
                                 "["+str(param.col)+", "+str(object_id)+"]")
        else: # If config.type != "InputCatalog"
            raise NotImplementedError("Sorry, only InputCatalog config types are currently "+
                                      "implemented.")
    else:
        raise AttributeError("No type attribute in non-constant config entry!")
    return param_value


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


