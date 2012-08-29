import galsim
op_dict = galsim.object_param_dict


def BuildGSObject(config, key, base=None):
    """Build a GSObject using config dict for key=key.

    @param config     A dict with the configuration information.
    @param key        A configuration galsim.Config instance read in using galsim.config.load().
    @param base       A dict which stores potentially useful things like
                      base['rng'] = random number generator
                      base['input_cat'] = input catalog for InputCat items
                      base['real_cat'] = real galaxy catalog for RealGalaxy objects
                      Typically on the initial call to BuildGSObject, this will be 
                      the same as config, hence the name base.

    @returns gsobject, safe 
        gsobject is the built object 
        safe is a bool that says whether it is safe to use this object again next time
    """
    # I'd like to be able to have base=config be the default value, but python doesn't
    # allow that.  So None is the default, and if it's None, we set it to config.
    if not base:
        base = config

    # For backwards compatibility:  If config is an AttributeDict, then use its __dict__
    if isinstance(config,galsim.AttributeDict):
        return BuildGSObject(config.__dict__,key,base)
    if isinstance(base,galsim.AttributeDict):
        return BuildGSObject(config,key,base.__dict__)

    #print 'Start BuildGSObject: config = ',config
    if isinstance(config,dict):
        if not key in config:
            raise AttributeError("key %s not found in config"%key)
    elif isinstance(config,list):
        if not key < len(config):
            raise AttributeError("Trying to build past the end of a list in config")
    else:
        raise AttributeError("BuildGSObject not given a valid dictionary")

    # Start by parsing the config object in case it is a config string
    _Parse(config,key)
    #print 'After Parse: config = ',config

    # Alias for convenience
    ck = config[key]

    # Check that the input config has a type to even begin with!
    if not 'type' in ck:
        raise AttributeError("type attribute required in config.%s"%key)
    type = ck['type']

    # If we have previously saved an object and marked it as safe, then use it.
    if 'current' in ck and ck['safe']:
        #print 'current is safe:  ',ck['current'], True
        return ck['current'], True

    # Otherwise build the object depending on type, shift/shear, etc. if supported for that type
    #print 'config.%s.type = %s'%(key,ck['type'])
    if type in ('Sum', 'Convolution', 'Add', 'Convolve'):   # Compound object
        gsobjects = []
        if 'items' not in ck:
            raise AttributeError("items attribute required in for config."+type+" entry.")
        items = ck['items']
        if not isinstance(items,list):
            raise AttributeError("items entry for config."+type+" entry is not a list.")
        safe = True
        for i in range(len(items)):
            # The way we parse the configuration items requires that we be able to 
            # update the config dictionary items with a line like config[key] = new_value
            # This is a workaround to the python deficiency of not having real assignment
            # of values.  So if we only have ck = config[key], then ck = new_value doesn't
            # actually update the entry in the original dictionary.  It just makes a 
            # new variable equal to new_values.
            # Anyway, the upshot of all this is that items[i] must have a key value
            # so we can do this.  But normally it doesn't.  So we have to add a level
            # to the dictionary here if we haven't done so yet.
            gsobject, safe1 = BuildGSObject(items, i, base)
            safe = safe and safe1
            gsobjects.append(gsobject)
        #print 'After built component items for ',type,' safe = ',safe

        if type in ('Sum', 'Add'):
            # Special: if the last item in a Sum doesn't specify a flux, we scale it
            # to bring the total flux up to 1.
            if ('flux' not in items[-1]) and all('flux' in item for item in items[0:-1]):
                sum = 0
                for item in items[0:-1]:
                    sum += _GetCurrentParamValue(item,'flux')
                #print 'sum = ',sum
                f = 1. - sum
                #print 'f = ',f
                if (f < 0):
                    import warnings
                    warnings.warn(
                        "Automatically scaling the last item in Sum to make the total flux\n" +
                        "equal 1 requires the last item to have negative flux = %f"%f)
                gsobjects[-1].setFlux(f)
            gsobject = galsim.Add(gsobjects)
        else:  # type in ('Convolution', 'Convolve'):
            gsobject = galsim.Convolve(gsobjects)
        #print 'After built gsobject = ',gsobject

        # Allow the setting of the overall flux of the object. Individual component fluxes
        # retain the ratio of their own specified flux parameter settings.
        if 'flux' in ck:
            flux, safe1 = _GetParamValue(ck, 'flux', base)
            #print 'flux = ',flux
            gsobject.setFlux(flux)
            safe = safe and safe1
        #print 'After set flux, gsobject = ',gsobject

    elif type == 'Pixel': # BR: under duress ;)
        # Note we do not shear, shift, rotate etc. Pixels, such params raise an Exception.
        for transform in ('ellip', 'rotate', 'shear', 'shift'):
            if transform in ck:
                raise AttributeError(transform+" operation specified in ck not supported for "+
                                     "Pixel objects.")
        gsobject, safe = _BuildPixel(ck, base)

    elif type == 'SquarePixel':
        # Note we do not shear, shift, rotate etc. SquarePixels, such params raise an Exception.
        for transform in ('ellip', 'rotate', 'shear', 'shift'):
            if transform in ck:
                raise AttributeError(transform+" operation specified in ck not supported for "+
                                     "SquarePixel objects.")
        gsobject, safe = _BuildSquarePixel(ck, base)

    elif type == 'RealGalaxy':
        # RealGalaxy is a bit special, since it uses base['real_cat'] and
        # it doesn't have any size values.
        gsobject, safe = _BuildRealGalaxy(ck, base)

    elif type in op_dict: 
        # Build object from primary GSObject keys in galsim.object_param_dict
        gsobject, safe = _BuildSimple(ck, base)

    else:
        raise NotImplementedError("Unrecognised type = "+str(type))

    try : 
        ck['saved_re'] = gsobject.half_light_radius
    except :
        pass

    gsobject, safe1 = _BuildEllipRotateShearShiftObject(gsobject, ck, base)
    safe = safe and safe1

    ck['current'] = gsobject
    ck['safe'] = safe
    #print 'Done BuildGSObject: ',gsobject,safe
    return gsobject, safe


def _BuildPixel(config, base):
    """@brief Build a Pixel type GSObject from user input.
    """
    for key in ['xw', 'yw']:
        if not key in config:
            raise AttributeError("Pixel type requires attribute %s in input config."%key)
    xw, safe1 = _GetParamValue(config, 'xw', base)
    yw, safe2 = _GetParamValue(config, 'yw', base)
    safe = safe1 and safe2

    init_kwargs = {'xw' : xw, 'yw' : yw }
    if (xw != yw):
        raise Warning(
            "xw != yw found (%f != %f) "%(init_kwargs['xw'], init_kwargs['yw']) +
            "This is supported for the pixel, but not the draw routines. " +
            "There might be weirdness....")
    if 'flux' in config:
        flux, safe3 = _GetParamValue(config, 'flux', base)
        init_kwargs['flux'] = flux
        safe = safe and safe3
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    init_kwargs = dict([(k.encode('utf-8'), v) for k,v in init_kwargs.iteritems()]) 
    #print 'Pixel ',init_kwargs,safe
    return galsim.Pixel(**init_kwargs), safe


def _BuildSquarePixel(config, base):
    """@brief Build a SquarePixel type GSObject from user input.
    """
    if not 'size' in config:
        raise AttributeError(
            "size attribute required in config for initializing SquarePixel objects.")
    xw, safe = _GetParamValue(config, 'size', base)
    init_kwargs = {'xw': xw, 'yw': xw}
    if 'flux' in config:
        flux, safe1 = _GetParamValue(config, 'flux', base)
        init_kwargs['flux'] = flux
        safe = safe and safe1
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    init_kwargs = dict([(k.encode('utf-8'), v) for k,v in init_kwargs.iteritems()]) 
    #print 'Pixel ',init_kwargs,safe
    return galsim.Pixel(**init_kwargs), safe

    
def _BuildSimple(config, base):
    """@brief Build a simple GSObject (i.e. not Sums, Convolutions, Pixels or SquarePixel) from
    user input.
    """
    # First do a no-brainer checks
    if not 'type' in config:
        raise AttributeError("No type attribute in config!")
    type = config['type']

    init_kwargs = {}
    req_kwargs, safe1 = _GetRequiredKwargs(config, base)
    size_kwarg, safe2 = _GetSizeKwarg(config, base)
    opt_kwargs, safe3 = _GetOptionalKwargs(config, base)
    init_kwargs.update(req_kwargs)
    init_kwargs.update(size_kwarg)
    init_kwargs.update(opt_kwargs)
    safe = safe1 and safe2 and safe3

    # Finally, after pulling together all the params, try making the GSObject.
    init_func = eval("galsim."+type)
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    init_kwargs = dict([(k.encode('utf-8'), v) for k,v in init_kwargs.iteritems()]) 
    try:
        #print 'Construct ',type,' with kwargs: ',str(init_kwargs)
        gsobject = init_func(**init_kwargs)
    except Exception, err_msg:
        raise RuntimeError("Problem sending init_kwargs to galsim."+type+" object. "+
                           "Original error message: %s"% err_msg)

    #print 'Simple ',type,init_kwargs,safe
    return gsobject, safe

def _BuildRealGalaxy(config, base):
    """@brief Build a RealGalaxy type GSObject from user input.
    """
    if not 'index' in config:
        raise AttributeError(
            "index attribute required in config for initializing RealGalaxy objects.")
    if 'real_cat' not in base:
        raise ValueError("No real galaxy catalog available for building type = RealGalaxy")

    real_cat = base['real_cat']
    if isinstance(config['index'],dict) and 'type' in config['index'] :
        index = config['index']
        type = index['type']
        if (type == 'Sequence' or type == 'RandomInt') and 'max' not in index:
            index['max'] = real_cat.n

    index, safe = _GetParamValue(config, 'index', base, type=int)
    real_gal = galsim.RealGalaxy(real_cat, index=index)

    if 'flux' in config:
        flux, safe1 = _GetParamValue(config, 'flux', base)
        safe = safe and safe1
        real_gal.setFlux(flux)

    #print 'RealGal: ',real_gal,safe
    return real_gal, safe


# --- Now we define a function for "ellipsing", rotating, shifting, shearing, in that order.
#
def _BuildEllipRotateShearShiftObject(gsobject, config, base):
    """@brief Applies ellipticity, rotation, gravitational shearing and centroid shifting to a
    supplied GSObject, in that order, from user input.

    @returns transformed GSObject.
    """
    safe = True
    if 'ellip' in config:
        gsobject, safe1 = _BuildEllipObject(gsobject, config, 'ellip', base)
        safe = safe and safe1
    if 'rotate' in config:
        gsobject, safe1 = _BuildRotateObject(gsobject, config, 'rotate', base)
        safe = safe and safe1
    if 'shear' in config:
        gsobject, safe1 = _BuildEllipObject(gsobject, config, 'shear', base)
        safe = safe and safe1
    if 'shift' in config:
        gsobject, safe1 = _BuildShiftObject(gsobject, config, 'shift', base)
        safe = safe and safe1
    #print 'Transformed: ',gsobject,safe
    return gsobject, safe


def BuildShear(config, key, base):
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
    # For backwards compatibility:  If config is an AttributeDict, then use its __dict__
    if isinstance(config,galsim.AttributeDict):
        return BuildShear(config.__dict__,key,base)

    #print 'Start BuildShear'
    #print 'config[key] = ',config[key]
    _Parse(config,key)

    # Alias for convenience
    ck = config[key]
    #print 'After Parse: ck = ',ck

    if not 'type' in ck:
        raise AttributeError("No type attribute in config.%s."%key)
    type = ck['type']

    if type == 'E1E2':
        e1, safe1 = _GetParamValue(ck, 'e1', base)
        e2, safe2 = _GetParamValue(ck, 'e2', base)
        safe = safe1 and safe2
        #print 'e1,e2 = ',e1,e2,safe
        return galsim.Shear(e1=e1, e2=e2), safe
    elif type == 'G1G2':
        g1, safe1 = _GetParamValue(ck, 'g1', base)
        g2, safe2 = _GetParamValue(ck, 'g2', base)
        safe = safe1 and safe2
        #print 'g1,g2 = ',g1,g2,safe
        return galsim.Shear(g1=g1, g2=g2), safe
    elif type == 'GBeta':
        g, safe1 = _GetParamValue(ck, 'g', base)
        beta, safe2 = _GetParamValue(ck, 'beta', base, type=galsim.Angle)
        safe = safe1 and safe2
        #print 'g,beta = ',g,beta,safe
        return galsim.Shear(g=g, beta=beta), safe
    elif type == 'EBeta':
        e, safe1 = _GetParamValue(ck, 'e', base)
        beta, safe2 = _GetParamValue(ck, 'beta', base, type=galsim.Angle)
        safe = safe1 and safe2
        #print 'e,beta = ',e,beta,safe
        return galsim.Shear(e=e, beta=beta), safe
    elif type == 'QBeta':
        q, safe1 = _GetParamValue(ck, 'q', base)
        beta, safe2 = _GetParamValue(ck, 'beta', base, type=galsim.Angle)
        safe = safe1 and safe2
        #print 'q,beta = ',q,beta,safe
        return galsim.Shear(q=q, beta=beta), safe
    elif type == 'Ring':
        if not all (k in ck for k in ('num', 'first')) :
            raise AttributeError(
                "num and first attributes required in config.%s for type == Ring"%key)
        num = ck['num']
        #print 'In Ring parameter'
        # We store the current index in the ring and the last value in the dictionary,
        # so they will be available next time we get here.

        i = ck.get('i',num)
        #print 'i = ',i
        if i == num:
            #print 'at i = num'
            current, safe1 = BuildShear(ck, 'first', base)
            i = 1
        elif num == 2:  # Special easy case for only 2 in ring.
            #print 'i = ',i,' Simple case of n=2'
            current = -ck['current']
            i = i + 1
        else:
            import math
            #print 'i = ',i
            s = ck['current']
            current = galsim.Shear(g=s.g, beta=s.beta + math.pi/num * galsim.radians)
            i = i + 1
        ck['i'] = i
        ck['current'] = current
        #print 'return shear = ',current,False
        return current, False
    else:
        raise NotImplementedError("Unrecognised shear type %s."%type)

def _BuildEllipObject(gsobject, config, key, base):
    """@brief Applies ellipticity to a supplied GSObject from user input, also used for
    gravitational shearing.

    @returns transformed GSObject.
    """
    shear, safe = BuildShear(config, key, base)
    #print 'applyShear with ',shear
    gsobject.applyShear(shear)
    #print 'After applyShear, gsobject = ',gsobject, safe
    return gsobject, safe


def _BuildRotateObject(gsobject, config, key, base):
    """@brief Applies rotation to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    theta, safe = _GetParamValue(config, key, base, type=galsim.Angle)
    gsobject.applyRotation(theta)
    #print 'After applyRotation, gsobject = ',gsobject, safe
    return gsobject, safe

def BuildShift(config, key, base):
    """@brief Construct and return the (dx,dy) tuple to be used for a shift
    """
    # For backwards compatibility:  If config is an AttributeDict, then use its __dict__
    if isinstance(config,galsim.AttributeDict):
        return BuildShift(config.__dict__,key,base)

    _Parse(config,key)

    # Alias for convenience
    ck = config[key]

    if not 'type' in config:
        raise AttributeError("No type attribute in config!")
    type = ck['type']
    if type == 'DXDY':
        dx, safe1 = _GetParamValue(ck, 'dx', base)
        dy, safe2 = _GetParamValue(ck, 'dy', base)
        safe = safe1 and safe2
        #print 'DXDY = ',(dx,dy),safe
        return (dx,dy), safe
    elif type == 'RandomTopHat':
        return _GetRandomTopHatParamValue(ck, 'shift', base)
    else:
        raise NotImplementedError("Unrecognised shift type %s."%type)


def _BuildShiftObject(gsobject, config, key, base):
    """@brief Applies centroid shift to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    (dx,dy), safe = BuildShift(config, key, base)
    gsobject.applyShift(dx, dy)
    #print 'Shifted: ',gsobject,safe
    return gsobject, safe


# --- Below this point are the functions for getting the required parameters from the user input ---
#
def _GetRequiredKwargs(config, base):
    """@brief Get the required kwargs.
    """
    type = config['type']
    req_kwargs = {}
    safe = True
    for req_name in op_dict[type]['required']:
        # Sanity check here, as far upstream as possible
        if not req_name in config:
            raise AttributeError("No required attribute "+req_name+" within input config for type "+
                                 type+".")
        else:
            req_kwargs[req_name], safe1 = _GetParamValue(config, req_name, base)
            safe = safe and safe1
    #print 'req ',req_kwargs,safe
    return req_kwargs, safe

def _GetSizeKwarg(config, base):
    """@brief Get the one, and one only, required size kwarg.
    """
    type = config['type']
    size_kwarg = {}
    safe = True
    counter = 0  # start the counter
    for size_name in op_dict[type]['size']:
        if size_name in config:
            counter += 1
            if counter == 1:
                size_kwarg[size_name], safe1 = _GetParamValue(config, size_name, base)
                safe = safe and safe1
            elif counter > 1:
                raise ValueError("More than one size attribute within input config for type "+
                                 type+".")
    if counter == 0 and len(op_dict[type]['size']) > 0:
        raise ValueError("No size attribute within input config for type "+type+".")
    #print 'size ',size_kwarg,safe
    return size_kwarg, safe

def _GetOptionalKwargs(config, base):
    """@brief Get the optional kwargs, if any present in the config.
    """
    type = config['type']
    optional_kwargs = {}
    safe = True
    for entry_name in config:
        if entry_name in op_dict[type]['optional']:
            optional_kwargs[entry_name], safe1 = _GetParamValue(config, entry_name, base)
            safe = safe and safe1
    #print 'opt ',optional_kwargs,safe
    return optional_kwargs, safe

def _GetParamValue(config, param_name, base, type=float):
    """@brief Function to read parameter values from config.
    """
    # Parse the param in case it is a configuration string
    _Parse(config, param_name)
    
    param = config[param_name]

    # First see if we can assign by param by a direct constant value
    if isinstance(param, type):
        #print 'param == type: ',param,True
        return param, True
    elif not isinstance(param, dict):
        if type is galsim.Angle :
            # Angle is a special case.  Angles are specified with a final string to 
            # declare what unit to use.
            try :
                (value, unit) = param.rsplit(None,1)
                value = float(value)
                unit = unit.lower()
                if unit.startswith('rad') :
                    val = galsim.Angle(value, galsim.radians)
                elif unit.startswith('deg') :
                    val = galsim.Angle(value, galsim.degrees)
                elif unit.startswith('hour') :
                    val = galsim.Angle(value, galsim.hours)
                elif unit.startswith('arcmin') :
                    val = galsim.Angle(value, galsim.arcmin)
                elif unit.startswith('arcsec') :
                    val = galsim.Angle(value, galsim.arcsec)
                else :
                    raise AttributeError("Unknown Angle unit: %s"%unit)
                config[param_name] = val
            except :
                raise AttributeError("Unable to parse %s as an Angle."%param)
        else :
            # Make sure strings are converted to float (or other type) if necessary.
            # In particular things like 1.e6 aren't converted to float automatically
            # by the yaml reader. (Although I think this is a bug.)
            try : 
                val = type(param)
                config[param_name] = val
            except :
                raise AttributeError("Could not convert %s to %s."%(param,type))
        #print 'param => type: ',val,True
        return val, True
    elif not 'type' in param:
        raise AttributeError(
            "%s.type attribute required in config for non-constant parameter %s."
                    %(param_name,param_name))
    else: # Use type to set param value. Currently catalog input supported only.
        type = param['type']
        if type == 'InputCatalog':
            return _GetInputCatParamValue(param, param_name, base)
        elif type == 'RandomAngle':
            return _GetRandomAngleParamValue(param, param_name, base)
        elif type == 'Random':
            return _GetRandomParamValue(param, param_name, base)
        elif type == 'RandomGaussian':
            return _GetRandomGaussianParamValue(param, param_name, base)
        elif type == 'Sequence':
            return _GetSequenceParamValue(param, param_name, base)
        else:
            raise NotImplementedError("Unrecognised parameter type %s."%type)

def _GetCurrentParamValue(config, param_name):
    """@brief Function to find the current value (either stored or a simple value)
    """
    param = config[param_name]
    if isinstance(param, dict):
        return param['current']
    else: 
        return param


def _GetInputCatParamValue(param, param_name, base):
    """@brief Specialized function for getting param values from an input cat.
    """
    if 'input_cat' not in base:
        raise ValueError("No input catalog available for %s.type = InputCatalog"%param_name)
    input_cat = base['input_cat']

    # Set the val from the requisite [input_cat.current, col] entry in the
    # input_cat.data... if this fails, try to work out why and give info.
    if 'col' not in param:
        raise AttributeError(
            "%s.col attribute required %s.type = InputCatalog"%(param_name,param_name))
    col = int(param['col'])

    if input_cat.type == 'ASCII':
        index = param.get('current_index',-1)
        index = index + 1
        if index >= input_cat.nobjects:
            raise IndexError(
                "%s index has gone past the number of entries in the catalog"%param_name)
        param['current_index'] = index

        try:    # Try setting the param value from the catalog
            val = input_cat.data[index, col - 1]
        except IndexError:
            raise IndexError("%s.col attribute (=%d) out of bounds"%(param_name,col))
    elif input_cat.type == 'FITS':
        raise NotImplementedError("Sorry, FITS input not implemented.")
    else:
        raise NotImplementedError("Unrecognised input_cat type %s."%input_cat.type)
    param['current'] = val
    #print 'InputCat: ',val,False
    return val, False

def _GetRandomParamValue(param, param_name, base):
    """@brief Specialized function for getting a random value
    """
    if 'rng' not in base:
        raise ValueError("No rng available for %s.type = Random"%param_name)
    rng = base['rng']

    if not all (k in param for k in ('min','max')):
        raise AttributeError(
            "%s.min and max attributes required %s.type = Random"%(param_name,param_name))
    min = float(param['min'])
    max = float(param['max'])

    ud = galsim.UniformDeviate(rng)
    val = ud() * (max-min) + min
    param['current'] = val
    #print 'Random: ',val,False
    return val, False

def _GetRandomAngleParamValue(param, param_name, base):
    """@brief Specialized function for getting a random angle
    """
    if 'rng' not in base:
        raise ValueError("No rng available for %s.type = RandomAngle"%param_name)
    rng = base['rng']

    import math
    ud = galsim.UniformDeviate(rng)
    val = ud() * 2 * math.pi * galsim.radians
    param['current'] = val
    #print 'RandomAngle: ',val,False
    return val, False

def _GetRandomGaussianParamValue(param, param_name, base):
    """@brief Specialized function for getting a random gaussian deviate
    """
    if 'rng' not in base:
        raise ValueError("No rng available for %s.type = RandomGaussian"%param_name)
    rng = base['rng']

    if 'sigma' not in param:
        raise AttributeError(
            "%s.sigma attribute required %s.type = RandomGaussian"%(param_name,param_name))
    sigma = float(param['sigma'])

    mean = float(param.get('mean',0.))
    min = float(param.get('min',-float('inf')))
    max = float(param.get('max',float('inf')))

    if 'gd' in param:
        gd = param['gd']
    else:
        # Minor subtlety here.  GaussianDeviate requires two random numbers to 
        # generate a single Gaussian deviate.  But then it gets a second 
        # deviate for free.  So it's more efficient to store gd here to use
        # the next time through.
        gd = galsim.GaussianDeviate(rng, sigma=sigma)
        param['gd'] = gd

    # Clip at min/max.
    # However, special cases if min == mean or max == mean
    #  -- can use fabs to double the chances of falling in the range.
    do_abs = False
    do_neg = False
    if min == mean:
        do_abs = True
        max -= mean
        min = -max
    elif max == mean:
        do_abs = True
        do_neg = True
        min -= mean
        max = -min
    else:
        min -= mean
        max -= mean

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
    val += mean
    param['current'] = val
    #print 'RandomGaussian: ',val,False
    return val, False

def _GetRandomIntParamValue(param, param_name, base):
    """@brief Specialized function for getting a random integer value
    """
    if 'rng' not in base:
        raise ValueError("No rng available for %s.type = RandomInt"%param_name)
    rng = base['rng']

    if not all (k in param for k in ('min','max')):
        raise AttributeError(
            "%s.min and max attributes required %s.type = RandomInt"%(param_name,param_name))
    min = int(param['min'])
    max = int(param['max'])

    ud = galsim.UniformDeviate(rng)
    import math
    val = int(math.floor(ud() * (max-min+1))) + min
    # In case ud() == 1
    if val > max:
        val = max
    param['current'] = val
    #print 'RandomInt: ',val,False
    return val, False

def _GetRandomTopHatParamValue(param, param_name, base):
    """@brief Return an (x,y) pair drawn from a circular top hat distribution.
    """
    if 'rng' not in base:
        raise ValueError("No rng available for %s.type = RandomTopHat"%param_name)
    rng = base['rng']

    if 'radius' not in param:
        raise AttributeError(
            "%s.radius attribute required %s.type = RandomTopHat"%(param_name,param_name))
    radius = float(param['radius'])

    ud = galsim.UniformDeviate(rng)
    max_rsq = radius*radius
    rsq = 2*max_rsq
    while (rsq > max_rsq):
        x = (2*ud()-1) * radius
        y = (2*ud()-1) * radius
        rsq = x**2 + y**2
    param['current'] = (x,y)
    #print 'RandomTopHat: ',(x,y),False
    return (x,y), False

def _GetSequenceParamValue(param, param_name, base):
    """@brief Specialized function for getting a sequence of integers
    """
    if 'max' not in param:
        raise AttributeError(
            "%s.max attribute required %s.type = Sequence"%(param_name,param_name))
    max = int(param['max'])
    min = int(param.get('min',0))
    step = int(param.get('step',1))

    index = param.get('current',min-step)
    index = index + step
    if index > max:
        index = min
    param['current'] = index
    #print 'Sequence: ',index,False
    return index, False

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
 

def _Parse(config, key):
    """@brief _Parse(config, key) does initial parsing of strings if necessary.
    
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
    ck = config[key]
    orig = ck
    if isinstance(ck, basestring):
        tokens = ck.split(None,1)
        if len(tokens) < 2:
            # Special case string only has one word.  So this isn't a string to
            # be parsed.  It's just a string value. 
            # e.g. config.catalog.file_name = 'in.cat'
            return 
        config[key] = {}
        ck = config[key]
        ck['type'] = tokens[0]
        str = tokens[1]
    elif isinstance(ck,galsim.AttributeDict):
        config[key] = ck.__dict__
        _Parse(config,key)
        return
    elif isinstance(ck,dict):
        if 'type' in ck:
            type = ck['type']
            if isinstance(type, basestring):
                tokens = type.split(None,1)
                if len(tokens) == 1:
                    # Then this config is already parsed.
                    return 
                elif len(tokens) == 0:
                    raise AttributeError('Provided type is an empty string: %s'%type)
                type = tokens[0]
                str = tokens[1]
            else:
                raise AttributeError('Provided type is not a string: %s'%type)
        else:
            raise AttributeError("type attribute required in config.")
    else:
        # This is just a value
        return 

    # Now type is set correctly and str holds the rest of the string to be parsed.
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
            ck[attrib] = value
    except:
        # If this didn't parse correctly, then this is probably just a string value
        # with more than one token.  In this case, just use the original value for config[key]
        config[key] = orig
        pass
 
