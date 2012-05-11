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

    # Then build the object depending on type, and shift/shear etc. if supported for that type
    #
    if config.type in ("Sum", "Convolution"):   # Compound object
        gsobjects = []
        if "items" in config.__dict__:
            for i in range(len(config.items)):
                gsobjects.append(BuildGSObject(config.items[i], input_cat))
            if config.type == "Sum":
                gsobject = galsim.Add(gsobjects)
            elif config.type == "Convolve":
                gsobject = galsim.Convolve(gsobjects)
            # Allow the setting of the overall flux of the object. Individual component fluxes
            # retain the ratio of their own specified flux parameter settings.
            if "flux" in config.__dict__:
                gsobject.setFlux(_GetParamValue(config, "flux", input_cat))
            gsobject = _BuildEllipRotateShearShiftObject(gsobject, config, input_cat)
        else:
            raise AttributeError("items attribute required in for config."+type+" entry.")

    elif config.type == "Pixel": # BR: under duress ;)
        gsobject = _BuildPixel(config, input_cat)
        # Note we do not shear, shift, rotate etc. Pixels, such params in the config are ignored.

    elif config.type == "SquarePixel":
        gsobject = _BuildSquarePixel(config, input_cat)
        # Note we do not shear, shift, rotate etc. Pixels, such params in the config are ignored.

    # Else Build object from primary GSObject keys in galsim.object_param_dict
    elif config.type in op_dict: 
        gsobject = _BuildSimple(config, input_cat)
        gsobject = _BuildEllipRotateShearShiftObject(gsobject, config, input_cat)
    else:
        raise NotImplementedError("Unrecognised config.type = "+str(config.type))
    return gsobject


def _BuildPixel(config, input_cat=None):
    """@brief Build a Pixel type GSObject from user input.
    """
    for key in ["xw", "yw"]:
        if not key in config.__dict__:
            raise AttributeError("Pixel type requires attribute %s in input config."%key)
    init_kwargs = {"xw": _GetParamValue(config, "xw", input_cat),
                   "yw": _GetParamValue(config, "yw", input_cat)}
    if (init_kwargs["xw"] != init_kwargs["yw"]):
        raise Warning("xw != yw found (%f != %f) "%(init_kwargs["xw"], init_kwargs["yw"]) +
            "This is supported for the pixel, but not the draw routines. " +
            "There might be weirdness....")
    if "flux" in config.__dict__:
        kwargs["flux"] = _GetParamValue(config, "flux", input_cat)
    return galsim.Pixel(**init_kwargs)


def _BuildSquarePixel(config, input_cat=None):
    """@brief Build a SquarePixel type GSObject from user input.
    """
    if not "size" in config.__dict__:
        raise AttributeError("size attribute required in config for initializing SquarePixel "+
                             "objects.")
    init_kwargs = {"xw": _GetParamValue(config, "size", input_cat)}
    init_kwargs["yw"] = init_kwargs["xw"]
    if "flux" in config.__dict__:
        init_kwargs["flux"] = _GetParamValue(config, "flux", input_cat)
    return galsim.Pixel(**init_kwargs)

    
def _BuildSimple(config, input_cat=None):
    """@brief Build a simple GSObject (i.e. not Sums, Convolutions, Pixels or SquarePixel) from
    user input.
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
    try:
        gsobject = init_func(**init_kwargs)
    except Error, err_msg:
        raise RuntimeError("Problem sending init_kwargs to galsim."+config.type+" object. "+
                           "Original error message: "+err_msg)
    return gsobject

# --- Now we define a function for "ellipsing", rotating, shifting, shearing, in that order.
#
def _BuildEllipRotateShearShiftObject(gsobject, config, input_cat=None):
    """@brief Applies ellipticity, rotation, gravitational shearing and centroid shifting to a
    supplied GSObject, in that order, based on input config and input_cat.

    @returns transformed GSObject.
    """
    if "ellip" in config.__dict__:
        gsobject = _BuildEllipObject(gsobject, config.ellip, input_cat)
    if "rotate" in config.__dict__:
        gsobject = _BuildRotateObject(gsobject, config.rotate, input_cat)
    if "shear" in config.__dict__:
        gsobject = _BuildEllipObject(gsobject, config.shear, input_cat)
    if "shift" in config.__dict__:
        gsobject = _BuildShiftObject(gsobject, config.shift, input_cat)
    return gsobject


def _BuildEllipObject(gsobject, config, input_cat=None):
    """@brief Applies ellipticity to a supplied GSObject based on input config and input_cat, also
    used for gravitational shearing.

    @returns transformed GSObject.
    """
    @returns transformed GSObject.
    if not "type" in config.__dict__:
        raise AttributeError("No type attribute in config!")
    if config.type == "E1E2":
        e1 = _GetParamValue(config, "e1", input_cat)
        e2 = _GetParamValue(config, "e2", input_cat)
        gsobject.applyDistortion(galsim.Ellipse(e1, e2))
    elif config.type == "G1G2":
        g1 = _GetParamValue(config, "g1", input_cat)
        g2 = _GetParamValue(config, "g2", input_cat)
        gsobject.applyShear(g1, g2)
    else:
        raise NotImplementedError("Sorry only ellip.type = 'E1E2', 'G1G2' currently supported.")
    return gsobject


def _BuildRotateObject(gsobject, config, input_cat=None):
    """@brief Applies rotation to a supplied GSObject based on input config and input_cat.

    @returns transformed GSObject.

    CURRENTLY NOT IMPLEMENTED WILL RAISE AN EXCEPTION IF CALLED.
    """
    raise NotImplementedError("Sorry, rotation (with new angle class) not currently supported.")


def _BuildShiftObject(gsobject, config, input_cat=None):
    """@brief Applies centroid shift to a supplied GSObject based on input config and input_cat.

    @returns transformed GSObject.
    """
    if not "type" in config.__dict__:
        raise AttributeError("No type attribute in config!")
    if config.type == "DXDY":
        dx = _GetParamValue(config, "dx", input_cat)
        dy = _GetParamValue(config, "dy", input_cat)
        gsobject.applyShift(dx, dy)
    else:
        raise NotImplementedError("Sorry only shift.type = 'DXDY' currently supported.")
    return gsobject


# --- Below this point are the functions for getting the required parameters from the user input ---
#
def _GetRequiredKwargs(config, input_cat=None):
    """@brief Get the required kwargs.
    """
    req_kwargs = {}
    for req_name in op_dict[config.type]["required"]:
        # Sanity check here, as far upstream as possible
        if not req_name in config.__dict__:
            raise AttributeError("No required attribute "+req_name+" within input config for type "+
                                 config.type+".")
        else:
            req_kwargs[req_name] = _GetParamValue(config, req_name, input_cat=input_cat)
    return req_kwargs

def _GetSizeKwarg(config, input_cat=None):
    """@brief Get the one, and one only, required size kwarg.
    """
    size_kwarg = {}
    counter = 0  # start the counter
    for size_name in op_dict[config.type]["size"]:
        if size_name in config.__dict__:
            counter += 1
            if counter == 1:
                size_kwarg[size_name] = _GetParamValue(config, size_name, input_cat=input_cat)
            elif counter > 1:
                raise ValueError("More than one size attribute within input config for type "+
                                 config.type+".")
    if counter == 0:
        raise ValueError("No size attribute within input config for type "+config.type+".")
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
    
    # First see if we can assign by param by a direct constant value
    if not hasattr(param, "__dict__"):  # This already exists for AttributeDicts, not for values
        param_value = param
    elif not "type" in param.__dict__: 
        raise AttributeError(param_name+".type attribute required in config for non-constant "+
                             "parameter "+param_name+".")
    else: # Use type to set param value. Currently catalog input supported only.
        if param.type == "InputCatalog":
            param_value = _GetInputCatParamValue(config, param_name, input_cat)
        else:
            raise NotImplementedError("Sorry, only InputCatalog config types are currently "+
                                      "implemented.")
    return param_value


def _GetInputCatParamValue(config, param_name, input_cat=None):
    """@brief Specialized function for getting param values from an input cat.
    """
    param = config.__getattr__(param_name)
    # Assuming param_name.type == InputCatalog checking/setting done upstream to avoid excess tests.
    if input_cat == None:
        raise ValueError("Keyword input_cat not given to _GetInputCatParamValue: the config "+
                         "requires an InputCatalog entry for "+param_name)

    # Set the param_value from the requisite [input_cat.current, col] entry in the
    # input_cat.data... if this fails, try to work out why and give info.
    if "col" in param.__dict__:
        col = param.col
    else:
        raise AttributeError(param_name+".col attribute required in config for "+
                             "initializing with "+param_name+".type = InputCatalog.")
    if input_cat.type == "ASCII":
        try:    # Try setting the param value from the catalog
            param_value = input_cat.data[input_cat.current, col - 1]
        except IndexError:
            raise IndexError(param_name+".col attribute or input_cat.current out of bounds for "+
                             "accessing input_cat.data [col, object_id] = ["+str(param.col)+", "+
                             str(object_id)+"]")
    elif input_cat.type == "FITS":
        raise NotImplementedError("Sorry, FITS input not implemented.")
    else:
        raise ValueError("input_cat.type must be either 'FITS' or 'ASCII' please.")
    return param_value

