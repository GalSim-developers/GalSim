import galsim
op_dict = galsim.object_param_dict


def BuildGSObject(config, input_cat=None, logger=None):
    """Build a GSObject using a config (Config instance) and possibly an input_cat.

    @param config     A configuration galsim.Config instance read in using galsim.config.load().
    @param input_cat  An input catalog read in using galsim.io.ReadInputCat().
    @param logger     Output logging object (NOT USED IN THIS IMPLEMENTATION: RAISED ERRORS
                      AUTOMATICALLY PASSED TO LOGGER)
    """
    # Start by parsing the config object in case it is a config string
    config = _Parse(config)

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
        # Note we do not shear, shift, rotate etc. Pixels, such params raise an Exception.
        for transform in ("ellip", "rotate", "shear", "shift"):
            if transform in config.__dict__:
                raise AttributeError(transform+" operation specified in config not supported for "+
                                     "Pixel objects.")
        gsobject = _BuildPixel(config, input_cat)

    elif config.type == "SquarePixel":
        # Note we do not shear, shift, rotate etc. SquarePixels, such params raise an Exception.
        for transform in ("ellip", "rotate", "shear", "shift"):
            if transform in config.__dict__:
                raise AttributeError(transform+" operation specified in config not supported for "+
                                     "SquarePixel objects.")
        gsobject = _BuildSquarePixel(config, input_cat)

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
    except Exception, err_msg:
        raise RuntimeError("Problem sending init_kwargs to galsim."+config.type+" object. "+
                           "Original error message: %s"% err_msg)
    return gsobject

# --- Now we define a function for "ellipsing", rotating, shifting, shearing, in that order.
#
def _BuildEllipRotateShearShiftObject(gsobject, config, input_cat=None):
    """@brief Applies ellipticity, rotation, gravitational shearing and centroid shifting to a
    supplied GSObject, in that order, from user input.

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
    """@brief Applies ellipticity to a supplied GSObject from user input, also used for
    gravitational shearing.

    @returns transformed GSObject.
    """
    config = _Parse(config)
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
    """@brief Applies rotation to a supplied GSObject based on user input.

    @returns transformed GSObject.

    CURRENTLY NOT IMPLEMENTED WILL RAISE AN EXCEPTION IF CALLED.
    """
    raise NotImplementedError("Sorry, rotation (with new angle class) not currently supported.")


def _BuildShiftObject(gsobject, config, input_cat=None):
    """@brief Applies centroid shift to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    config = _Parse(config)
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
    """@brief Function to read parameter values from config.
    """
    # Assume that basic sanity checking done upstream for maximum efficiency 
    param = config.__getattr__(param_name)

    # Parse the param in case it is a configuration string
    param = _Parse(param)
    
    # First see if we can assign by param by a direct constant value
    if not hasattr(param, "__dict__"):  # This already exists for Config instances, not for values
        param_value = param
    elif not "type" in param.__dict__: 
        raise AttributeError(param_name+".type attribute required in config for non-constant "+
                             "parameter "+param_name+".")
    else: # Use type to set param value. Currently catalog input supported only.
        if param.type == "InputCatalog":
            param_value = _GetInputCatParamValue(param, param_name, input_cat)
        else:
            raise NotImplementedError("Sorry, only InputCatalog config types are currently "+
                                      "implemented.")
    return param_value


def _GetInputCatParamValue(param, param_name, input_cat=None):
    """@brief Specialized function for getting param values from an input cat.
    """
    # Assuming param.type == InputCatalog checking/setting done upstream to avoid excess tests.
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

def _MatchDelim(str,start,end):
    nest_count = 0
    for i in range(len(str)):
        if str[i] == start:
            nest_count += 1
        if str[i] == end:
            nest_count -= 1
        if nest_count == 0:
            break
    return (str[1:i],str[i+1:])
 

def _Parse(config):
    """@brief config=_Parse(config) does initial parsing of strings if necessary.
    
       If a parameter or its type is a string, this means that it should be parsed to 
       build the appropriate attributes.  For example,

       @code
       parsing gal.type = 'Exponential scale_radius=3 flux=100'
       @endcode

       would result in the equivalent of:

       @code
       gal.type = 'Exponential'
       gal.scale_radius = 3
       gal.flux = 100
       @endcode

       Furthermore, if the first (non-whitespace) character after an = is '<',
       then the contents of the <> are recursively parsed for that value.
       e.g.
       @code
       psf = 'Moffat beta=<InputCatalog col=3> fwhm=<InputCatalog col=4>'
       @endcode
       would result in the equivalent of:
       @code
       psf.type = 'Moffat'
       psf.beta.type = 'InputCatalog'
       psf.beta.col = 3
       psf.fwhm.type = 'InputCatalog'
       psf.fwhm.col = 4
       @endcode

       If the first (non-whitespace) character after an = is '[', 
       then the contents are taken to be an array of configuration strings.
       e.g. 

       @code
       gal = 'Sum items = [ <Sersic n=1.2>, <Sersic n=3.5> ]'
       @endcode

       The string can be at either the base level (e.g. psf above) or as the type
       attribute (e.g. gal.type above).  The difference is that if the user
       specifies the string as a type, then other attributes can also be set separately.
       e.g.

       @code
       gal.type = 'Sersic n=1.5 half_light_radius=4 flux=1000'
       gal.shear = 'G1G2 g1=0.3 g2=0'
       @endcode
    """
    if isinstance(config, basestring):
        orig = config
        tokens = config.split(None,1)
        if len(tokens) < 2:
            # Special case string only has one word.  So this isn't a string to
            # be parsed.  It's just a string value. 
            # e.g. config.catalog.file_name = 'in.cat'
            return config
        config = galsim.Config()
        config.type = tokens[0]
        str = tokens[1]
    elif hasattr(config, "__dict__"):  
        if hasattr(config, "type"):  
            if isinstance(config.type, basestring):
                orig = config.type
                tokens = config.type.split(None,1)
                if len(tokens) == 1:
                    # Then this config is already parsed.
                    return config
                elif len(tokens) == 0:
                    raise AttributeError('Provided type is an empty string: %s',config.type)
                config.type = tokens[0]
                str = tokens[1]
            else:
                raise AttributeError('Provided type is not a string: %s',config.type)
        else:
            raise AttributeError("type attribute required in config.")
    else:
        # This is just a value
        return config

    # Now config.type is set correctly and str holds the rest of the string to be parsed.
    try :
        while str:
            tokens = str.split('=',1)
            attrib = tokens[0].strip()
            str = tokens[1].strip()
            if str.startswith('<'):
                value, str = _MatchDelim(str,'<','>')
            elif str.startswith('['):
                value, str = _MatchDelim(str,'[',']')
                # TODO: If there are nested arrays, this next line does the wrong thing!
                value = value.split(',')
            elif str.startswith('('):
                value, str = _MatchDelim(str,'(',')')
                value = eval(value)
            else:
                tokens = str.split(None,1)
                value = eval(tokens[0])
                if (len(tokens) == 1):
                    str = ""
                else:
                    str = tokens[1]
            config.__setattr__(attrib,value)
    except:
        raise ValueError("Error parsing configuration string " + orig)
    return config
 
