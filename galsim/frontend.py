import galsim
op_dict = galsim.object_param_dict


def BuildGSObject(config, rng=None, input_cat=None):
    """Build a GSObject using a config (Config instance) and possibly an input_cat.

    @param config     A configuration galsim.Config instance read in using galsim.config.load().
    @param rng        A random number generator to use when required
    @param input_cat  An input catalog read in using galsim.io.ReadInputCat().
    """
    #print 'Start BuildGSObject: config = ',config
    # Start by parsing the config object in case it is a config string
    config = _Parse(config)
    #print 'After Parse: config = ',config

    # Check that the input config has a type to even begin with!
    if not "type" in config.__dict__:
        raise AttributeError("type attribute required in config.")

    # Then build the object depending on type, and shift/shear etc. if supported for that type
    #print 'config.type = ',config.type
    if config.type in ("Sum", "Convolution", "Add", "Convolve"):   # Compound object
        gsobjects = []
        if "items" in config.__dict__:
            for i in range(len(config.items)):
                gsobjects.append(BuildGSObject(config.items[i], rng, input_cat))
            #print 'After built component items for ',config.type
            if config.type in ("Sum", "Add"):
                gsobject = galsim.Add(gsobjects)
            else:  # config.type in ("Convolution", "Convolve"):
                gsobject = galsim.Convolve(gsobjects)
            #print 'After built gsobject = ',gsobject
            # Allow the setting of the overall flux of the object. Individual component fluxes
            # retain the ratio of their own specified flux parameter settings.
            if "flux" in config.__dict__:
                gsobject.setFlux(_GetParamValue(config, "flux", rng, input_cat))
            #print 'After set flux, gsobject = ',gsobject
            gsobject = _BuildEllipRotateShearShiftObject(gsobject, config, rng, input_cat)
            #print 'After BuildEllipRotateShearShiftObject, gsobject = ',gsobject
        else:
            raise AttributeError("items attribute required in for config."+type+" entry.")

    elif config.type == "Pixel": # BR: under duress ;)
        # Note we do not shear, shift, rotate etc. Pixels, such params raise an Exception.
        for transform in ("ellip", "rotate", "shear", "shift"):
            if transform in config.__dict__:
                raise AttributeError(transform+" operation specified in config not supported for "+
                                     "Pixel objects.")
        gsobject = _BuildPixel(config, rng, input_cat)

    elif config.type == "SquarePixel":
        # Note we do not shear, shift, rotate etc. SquarePixels, such params raise an Exception.
        for transform in ("ellip", "rotate", "shear", "shift"):
            if transform in config.__dict__:
                raise AttributeError(transform+" operation specified in config not supported for "+
                                     "SquarePixel objects.")
        gsobject = _BuildSquarePixel(config, rng, input_cat)

    # Else Build object from primary GSObject keys in galsim.object_param_dict
    elif config.type in op_dict: 
        gsobject = _BuildSimple(config, rng, input_cat)
        gsobject = _BuildEllipRotateShearShiftObject(gsobject, config, rng, input_cat)
    else:
        raise NotImplementedError("Unrecognised config.type = "+str(config.type))
    return gsobject


def _BuildPixel(config, rng, input_cat):
    """@brief Build a Pixel type GSObject from user input.
    """
    for key in ["xw", "yw"]:
        if not key in config.__dict__:
            raise AttributeError("Pixel type requires attribute %s in input config."%key)
    init_kwargs = {"xw": _GetParamValue(config, "xw", rng, input_cat),
                   "yw": _GetParamValue(config, "yw", rng, input_cat)}
    if (init_kwargs["xw"] != init_kwargs["yw"]):
        raise Warning("xw != yw found (%f != %f) "%(init_kwargs["xw"], init_kwargs["yw"]) +
            "This is supported for the pixel, but not the draw routines. " +
            "There might be weirdness....")
    if "flux" in config.__dict__:
        kwargs["flux"] = _GetParamValue(config, "flux", rng, input_cat)
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    init_kwargs = dict([(k.encode('utf-8'), v) for k,v in init_kwargs.iteritems()]) 
    return galsim.Pixel(**init_kwargs)


def _BuildSquarePixel(config, rng, input_cat):
    """@brief Build a SquarePixel type GSObject from user input.
    """
    if not "size" in config.__dict__:
        raise AttributeError("size attribute required in config for initializing SquarePixel "+
                             "objects.")
    init_kwargs = {"xw": _GetParamValue(config, "size", rng, input_cat)}
    init_kwargs["yw"] = init_kwargs["xw"]
    if "flux" in config.__dict__:
        init_kwargs["flux"] = _GetParamValue(config, "flux", rng, input_cat)
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    init_kwargs = dict([(k.encode('utf-8'), v) for k,v in init_kwargs.iteritems()]) 
    return galsim.Pixel(**init_kwargs)

    
def _BuildSimple(config, rng, input_cat):
    """@brief Build a simple GSObject (i.e. not Sums, Convolutions, Pixels or SquarePixel) from
    user input.
    """
    # First do a no-brainer checks
    if not "type" in config.__dict__:
        raise AttributeError("No type attribute in config!")
    init_kwargs = {}
    init_kwargs.update(_GetRequiredKwargs(config, rng, input_cat))
    init_kwargs.update(_GetSizeKwarg(config, rng, input_cat))
    init_kwargs.update(_GetOptionalKwargs(config, rng, input_cat))
    # Finally, after pulling together all the params, try making the GSObject.
    init_func = eval("galsim."+config.type)
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    init_kwargs = dict([(k.encode('utf-8'), v) for k,v in init_kwargs.iteritems()]) 
    try:
        #print 'Construct ',config.type,' with kwargs: ',str(init_kwargs)
        gsobject = init_func(**init_kwargs)
    except Exception, err_msg:
        raise RuntimeError("Problem sending init_kwargs to galsim."+config.type+" object. "+
                           "Original error message: %s"% err_msg)
    return gsobject

# --- Now we define a function for "ellipsing", rotating, shifting, shearing, in that order.
#
def _BuildEllipRotateShearShiftObject(gsobject, config, rng, input_cat):
    """@brief Applies ellipticity, rotation, gravitational shearing and centroid shifting to a
    supplied GSObject, in that order, from user input.

    @returns transformed GSObject.
    """
    if "ellip" in config.__dict__:
        gsobject = _BuildEllipObject(gsobject, config.ellip, rng, input_cat)
    if "rotate" in config.__dict__:
        gsobject = _BuildRotateObject(gsobject, config.rotate, rng, input_cat)
    if "shear" in config.__dict__:
        gsobject = _BuildEllipObject(gsobject, config.shear, rng, input_cat)
    if "shift" in config.__dict__:
        gsobject = _BuildShiftObject(gsobject, config.shift, rng, input_cat)
    return gsobject


def BuildShear(config, rng=None, input_cat=None):
    """@brief Build and return a Shear object from the configuration file

       Implemented types are:
          - G1G2 = reduced shear 
          - GBeta = polar reduced shear 
          - E1E2 = distortion or ellipticity
          - EBeta = polar reduced shear 
          - QBeta = axis ratio and position angle

       Note that in terms of the axis ratio q=b/a (minor-to-major axis ratio):
       |e| = (1-q^2)/(1+q^2)
       |g| = (1-q)/(1+q)
    """
    #print 'Start BuildShear'
    #print 'config = ',config
    config = _Parse(config)
    #print 'After Parse: config = ',config
    if not "type" in config.__dict__:
        raise AttributeError("No type attribute in config!")
    if config.type == "E1E2":
        e1 = _GetParamValue(config, "e1", rng, input_cat)
        e2 = _GetParamValue(config, "e2", rng, input_cat)
        #print 'e1,e2 = ',e1,e2
        return galsim.Shear(e1=e1, e2=e2)
    elif config.type == "G1G2":
        g1 = _GetParamValue(config, "g1", rng, input_cat)
        g2 = _GetParamValue(config, "g2", rng, input_cat)
        #print 'g1,g2 = ',g1,g2
        return galsim.Shear(g1=g1, g2=g2)
    elif config.type == "GBeta":
        g = _GetParamValue(config, "g", rng, input_cat)
        beta = _GetParamValue(config, "beta", rng, input_cat, type=galsim.Angle)
        #print 'g,beta = ',g,beta
        return galsim.Shear(g=g, beta=beta)
    elif config.type == "EBeta":
        e = _GetParamValue(config, "e", rng, input_cat)
        beta = _GetParamValue(config, "beta", rng, input_cat, type=galsim.Angle)
        #print 'e,beta = ',e,beta
        return galsim.Shear(e=e, beta=beta)
    elif config.type == "QBeta":
        q = _GetParamValue(config, "q", rng, input_cat)
        beta = _GetParamValue(config, "beta", rng, input_cat, type=galsim.Angle)
        #print 'q,beta = ',q,beta
        return galsim.Shear(q=q, beta=beta)
    elif config.type == "Ring":
        if not all (k in config.__dict__ for k in ('num', 'first')) :
            raise AttributeError(
                "%s.num and first attributes required in config for %s.type == Ring"
                    %(param_name,param_name))
        num = config.num
        #print 'In Ring parameter'
        # We store the current index in the ring and the last value as 
        # attributes of this function, so it's available next time we get here.
        # This is basically like a static variable in C++.
        if not hasattr(BuildShear, 'i'):
            BuildShear.i = num
            BuildShear.current = None
        #print 'i = ',BuildShear.i
        if BuildShear.i == num:
            #print 'at i = num'
            BuildShear.current = BuildShear(config.first, rng, input_cat)
            BuildShear.i = 1
        elif num == 2:  # Special easy case for only 2 in ring.
            #print 'i = ',BuildShear.i,' Simple case of n=2'
            BuildShear.current = -BuildShear.current
            BuildShear.i = BuildShear.i + 1
        else:
            import math
            #print 'i = ',BuildShear.i
            s = BuildShear.current
            BuildShear.current = galsim.Shear(g=s.g, beta=s.beta + math.pi/num * galsim.radians)
            BuildShear.i = BuildShear.i + 1
        #print 'return shear = ',BuildShear.current
        return BuildShear.current
    else:
        raise NotImplementedError("Unrecognised shear type %s."%config.type)

def _BuildEllipObject(gsobject, config, rng, input_cat):
    """@brief Applies ellipticity to a supplied GSObject from user input, also used for
    gravitational shearing.

    @returns transformed GSObject.
    """
    shear = BuildShear(config, rng, input_cat)
    #print 'shear = ',shear
    gsobject.applyShear(shear)
    #print 'After applyShear, gsobject = ',gsobject
    return gsobject


def _BuildRotateObject(gsobject, config, rng, input_cat):
    """@brief Applies rotation to a supplied GSObject based on user input.

    @returns transformed GSObject.

    CURRENTLY NOT IMPLEMENTED WILL RAISE AN EXCEPTION IF CALLED.
    """
    raise NotImplementedError("Sorry, rotation (with new angle class) not currently supported.")


def BuildShift(config, rng=None, input_cat=None):
    """@brief Construct and return the (dx,dy) tuple to be used for a shift
    """
    config = _Parse(config)
    if not "type" in config.__dict__:
        raise AttributeError("No type attribute in config!")
    if config.type == "DXDY":
        dx = _GetParamValue(config, "dx", rng, input_cat)
        dy = _GetParamValue(config, "dy", rng, input_cat)
        return (dx,dy)
    elif config.type == "RandomTopHat":
        return _GetTopHatParamValue(config, "shift", rng)
    else:
        raise NotImplementedError("Unrecognised shift type %s."%config.type)


def _BuildShiftObject(gsobject, config, rng, input_cat):
    """@brief Applies centroid shift to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    (dx,dy) = BuildShift(config, rng, input_cat)
    gsobject.applyShift(dx, dy)
    return gsobject


# --- Below this point are the functions for getting the required parameters from the user input ---
#
def _GetRequiredKwargs(config, rng, input_cat):
    """@brief Get the required kwargs.
    """
    req_kwargs = {}
    for req_name in op_dict[config.type]["required"]:
        # Sanity check here, as far upstream as possible
        if not req_name in config.__dict__:
            raise AttributeError("No required attribute "+req_name+" within input config for type "+
                                 config.type+".")
        else:
            req_kwargs[req_name] = _GetParamValue(config, req_name, rng, input_cat)
    return req_kwargs

def _GetSizeKwarg(config, rng, input_cat):
    """@brief Get the one, and one only, required size kwarg.
    """
    size_kwarg = {}
    counter = 0  # start the counter
    for size_name in op_dict[config.type]["size"]:
        if size_name in config.__dict__:
            counter += 1
            if counter == 1:
                size_kwarg[size_name] = _GetParamValue(config, size_name, rng, input_cat)
            elif counter > 1:
                raise ValueError("More than one size attribute within input config for type "+
                                 config.type+".")
    if counter == 0:
        raise ValueError("No size attribute within input config for type "+config.type+".")
    return size_kwarg

def _GetOptionalKwargs(config, rng, input_cat):
    """@brief Get the optional kwargs, if any present in the config.
    """
    optional_kwargs = {}
    for entry_name in config.__dict__:
        if entry_name in op_dict[config.type]["optional"]:
            optional_kwargs[entry_name] = _GetParamValue(config, entry_name, rng, input_cat)
    return optional_kwargs

def _GetParamValue(config, param_name, rng, input_cat, type=float):
    """@brief Function to read parameter values from config.
    """
    # Assume that basic sanity checking done upstream for maximum efficiency 
    param = config.__getattr__(param_name)

    # Parse the param in case it is a configuration string
    param = _Parse(param)
    
    # First see if we can assign by param by a direct constant value
    if not hasattr(param, "__dict__"):  # This already exists for Config instances, not for values
        if type is galsim.Angle :
            # Angle is a special case.  Angles are specified with a final string to 
            # declare what unit to use.
            try :
                (value, unit) = param.rsplit(None,1)
                value = float(value)
                unit = unit.lower()
                if unit.startswith('rad') :
                    return galsim.Angle(value, galsim.radians)
                elif unit.startswith('deg') :
                    return galsim.Angle(value, galsim.degrees)
                elif unit.startswith('hour') :
                    return galsim.Angle(value, galsim.hours)
                elif unit.startswith('arcmin') :
                    return galsim.Angle(value, galsim.arcmin)
                elif unit.startswith('arcsec') :
                    return galsim.Angle(value, galsim.arcsec)
                else :
                    print 'Unknown Angle unit:',unit
                    raise AttributeError()
            except :
                raise AttributeError("Unable to parse %s as an Angle."%param)
        else :
            # Make sure strings are converted to float (or other type) if necessary.
            # In particular things like 1.e6 aren't converted to float automatically
            # by the yaml reader. (Although I think this is a bug.)
            try : 
                param_value = type(param)
                return param_value
            except :
                raise AttributeError("Could not convert %s to %s."%(param,type))
    elif not "type" in param.__dict__: 
        raise AttributeError(
            "%s.type attribute required in config for non-constant parameter %s."
                    %(param_name,param_name))
    else: # Use type to set param value. Currently catalog input supported only.
        if param.type == "InputCatalog":
            return _GetInputCatParamValue(param, param_name, rng, input_cat)
        elif param.type == "RandomAngle":
            import math
            ud = galsim.UniformDeviate(rng)
            return ud() * 2 * math.pi * galsim.radians
        elif param.type == "Random":
            if not all (k in param.__dict__ for k in ("min","max")):
                raise AttributeError(
                    param_name+".min and max attributes required in config for "+
                    "initializing with "+param_name+".type = Random.")
            ud = galsim.UniformDeviate(rng)
            return ud() * (param.max-param.min) + param.min
        elif param.type == "RandomGaussian":
            if 'sigma' not in param.__dict__:
                raise AttributeError(
                    param_name+".sigma attribute required in config for "+
                    "initializing with "+param_name+".type = RandomGaussian.")
            sigma = float(param.sigma)
            mean = float(param.__dict__.get('mean',0.))
            min = float(param.__dict__.get('min',-float('inf')))
            max = float(param.__dict__.get('max',float('inf')))
            #print 'RandomGaussian'
            #print 'sigma = ',sigma
            #print 'mean = ',mean
            #print 'min = ',min
            #print 'max = ',max
            gd = galsim.GaussianDeviate(rng, mean=mean, sigma=sigma)

            # Clip at min/max.
            # However, special cases if min == mean or max == mean
            #  -- can use abs to double the chances of falling in the range.
            do_abs = False
            do_neg = False
            if min == mean:
                do_abs = True
                min = 2*mean - max
            elif max == mean:
                do_abs = True
                do_neg = True
                max = 2*mean - min

            # Emulate a do-while loop
            while True:
                val = gd()
                if do_abs:
                    import math
                    val = math.fabs(val)
                if val >= min and val <= max:
                    break
            if do_neg:
                val = -val
            return val
        else:
            raise NotImplementedError("Unrecognised parameter type %s."%param.type)


def _GetInputCatParamValue(param, param_name, input_cat):
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
        raise NotImplementedError("Unrecognised input_cat type %s."%param.type)
    return param_value

def _GetTopHatParamValue(param, param_name, rng):
    """@brief Return an (x,y) pair drawn from a circular top hat distribution.
    """

    if "radius" in param.__dict__:
        radius = param.radius
    else:
        raise AttributeError(param_name+".radius attribute required in config for "+
                             "initializing with "+param_name+".type = RandomTopHat.")

    ud = galsim.UniformDeviate(rng)
    max_rsq = radius*radius
    rsq = 2*max_rsq
    while (rsq > max_rsq):
        x = (2*rng()-1) * radius
        y = (2*rng()-1) * radius
        rsq = x**2 + y**2
    return (x,y)


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
    orig = config
    if isinstance(config, basestring):
        tokens = config.split(None,1)
        if len(tokens) < 2:
            # Special case string only has one word.  So this isn't a string to
            # be parsed.  It's just a string value. 
            # e.g. config.catalog.file_name = 'in.cat'
            return orig
        config = galsim.Config()
        config.type = tokens[0]
        str = tokens[1]
    elif isinstance(config, dict):
        # If we are provided a regular dict rather than a Config object, convert it.
        config = galsim.Config()
        config.__dict__.update(orig)
        return _Parse(config)
    elif hasattr(config, "__dict__"):  
        if hasattr(config, "type"):  
            if isinstance(config.type, basestring):
                tokens = config.type.split(None,1)
                if len(tokens) == 1:
                    # Then this config is already parsed.
                    return config
                elif len(tokens) == 0:
                    raise AttributeError('Provided type is an empty string: %s'%config.type)
                config.type = tokens[0]
                str = tokens[1]
            else:
                raise AttributeError('Provided type is not a string: %s'%config.type)
        else:
            raise AttributeError("type attribute required in config.")
    else:
        # This is just a value
        return orig

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
        return config
    except:
        # If this didn't parse correctly, then this is probably just a string value
        # with more than one token.  In this case, just return the original.
        return orig
 
