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
        if "items" in config.__dict__:
            for i in range(len(config.items)):
                gsobjects.append(_BuildSingle(config.items[i], input_cat))
            if config.type == "Sum":
                gsobject = galsim.Add(gsobjects)
            elif config.type == "Convolve":
                gsobject = galsim.Convolve(gsobjects)
        else:
            raise AttributeError("items attribute required in for config."+type+" entry.")
    # MJ: Should pull out Pixel separately as well, since for that the sizes work differently
    # BR: This is covered by moving xw, and yw into my "required" list: the sizes work differently
    #     enough that I think they can deserve to no longer be called a size param.
    elif config.type == "SquarePixel":  # Mike is treating Pixels separately, I'll wrap SquarePixel
        if not "size" in config.__dict__:
            raise AttributeError("size attribute required in config for initializing SquarePixel "+
                                 "objects.")
        init_kwargs = {"xw": config.size, "yw": config.size}
        if "flux" in config.__dict__:
            init_kwargs["flux"] = config.flux
        print config.type, init_kwargs
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
    init_func = eval("galsim."+config.type)
    print config.type, init_kwargs
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
                raise ValueError("More than one size parameter specified in "+config.type+
                                 " object config.")
    # MJ: Check for counter == 0 here?  BR: If counter == 0 that's fine sometimes, c.f. Pixel.
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
            if input_cat.type == "ASCII":
                try:
                    param_value = input_cat.data[input_cat.current, col - 1]
                except IndexError:
                    raise IndexError(param_name+".col attribute or input_cat.current out of "+
                                     "bounds for accessing input_cat.data [col, object_id] = "+
                                     "["+str(param.col)+", "+str(object_id)+"]")
            elif input_cat.type == "FITS":
                raise NotImplementedError("Sorry, FITS input not implemented.")
            else:
                raise ValueError("input_cat.type must be either 'FITS' or 'ASCII' please.")
        else: # If config.type != "InputCatalog"
            raise NotImplementedError("Sorry, only InputCatalog config types are currently "+
                                      "implemented.")
    else:
        raise AttributeError("No type attribute in non-constant config entry!")
    return param_value
