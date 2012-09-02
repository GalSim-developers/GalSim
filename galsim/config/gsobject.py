import galsim

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

    #print 'Start BuildGSObject: config = ',config
    if isinstance(config,dict):
        if not key in config:
            raise AttributeError("key %s not found in config"%key)
    elif isinstance(config,list):
        if not key < len(config):
            raise AttributeError("Trying to build past the end of a list in config")
    else:
        raise AttributeError("BuildGSObject not given a valid dictionary")

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
    #print 'config.%s.type = %s'%(key,type)
    if type in ('Sum', 'Convolution', 'Add', 'Convolve'):   # Compound object
        gsobjects = []
        if 'items' not in ck:
            raise AttributeError("items attribute required for config.%s entry."%type)
        items = ck['items']
        if not isinstance(items,list):
            raise AttributeError("items entry for config.%s entry is not a list."%type)
        safe = True
        for i in range(len(items)):
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
                    sum += galsim.config.value._GetCurrentValue(item,'flux')
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
            flux, safe1 = galsim.config.ParseValue(ck, 'flux', base, float)
            #print 'flux = ',flux
            gsobject.setFlux(flux)
            safe = safe and safe1
        #print 'After set flux, gsobject = ',gsobject

    elif type == 'List':
        if 'items' not in ck:
            raise AttributeError("items attribute required for config.%s entry."%type)
        items = ck['items']
        if not isinstance(items,list):
            raise AttributeError("items entry for config.%s entry is not a list."%type)
        if 'index' not in ck:
            ck['index'] = { 'type' : 'Sequence' , 'min' : 0 , 'max' : len(items)-1 }
        index, safe = galsim.config.ParseValue(ck, 'index', base, int)
        if index < 0 or index >= len(items):
            raise AttributeError("index %d out of bounds for config.%s"%(index,type))
        #print items[index]['type']
        #print 'index = ',index,' From ',key,' List: ',items[index]
        gsobject, safe1 = BuildGSObject(items, index, base)
        safe = safe and safe1
        if 'flux' in ck:
            flux, safe1 = galsim.config.ParseValue(ck, 'flux', base, float)
            #print 'flux = ',flux
            gsobject.setFlux(flux)
            safe = safe and safe1
 
    elif type == 'RealGalaxy':
        # RealGalaxy is a bit special, since it uses base['real_cat'] and
        # it doesn't have any size values.
        gsobject, safe = _BuildRealGalaxy(ck, base)

    elif type in galsim.__dict__:
        if issubclass(galsim.__dict__[type], galsim.GSObject):
            gsobject, safe = _BuildSimple(ck, base)
        else:
            TypeError("Input config type = %s is not a GSObject."%type)
    else:
        raise NotImplementedError("Unrecognised config type = %s"%type)

    try : 
        ck['saved_re'] = gsobject.getHalfLightRadius()
    except :
        pass

    gsobject, safe1 = _BuildTransformObject(gsobject, ck, base)
    safe = safe and safe1

    if 'no_save' not in base:
        ck['current'] = gsobject
        ck['safe'] = safe
    #print 'Done BuildGSObject: ',gsobject
    return gsobject, safe


def _BuildPixel(config, base):
    """@brief Build a Pixel type GSObject from user input.
    """
    for key in ['xw', 'yw']:
        if not key in config:
            raise AttributeError("Pixel type requires attribute %s in input config."%key)
    xw, safe1 = galsim.config.ParseValue(config, 'xw', base, float)
    yw, safe2 = galsim.config.ParseValue(config, 'yw', base, float)
    safe = safe1 and safe2

    init_kwargs = {'xw' : xw, 'yw' : yw }
    if (xw != yw):
        raise Warning(
            "xw != yw found (%f != %f) "%(init_kwargs['xw'], init_kwargs['yw']) +
            "This is supported for the pixel, but not the draw routines. " +
            "There might be weirdness....")
    if 'flux' in config:
        flux, safe3 = galsim.config.ParseValue(config, 'flux', base, float)
        init_kwargs['flux'] = flux
        safe = safe and safe3
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    init_kwargs = dict([(k.encode('utf-8'), v) for k,v in init_kwargs.iteritems()]) 
    #print 'Pixel ',init_kwargs
    return galsim.Pixel(**init_kwargs), safe


def _BuildSimple(config, base):
    """@brief Build a simple GSObject (i.e. not Sums, Convolutions, etc.) from user input.
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
        raise RuntimeError("Problem sending init_kwargs to galsim.%s object. "%type+
                           "Original error message: %s"%err_msg)

    #print 'Simple ',type,init_kwargs
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
            index['max'] = real_cat.n-1

    index, safe = galsim.config.ParseValue(config, 'index', base, int)
    real_gal = galsim.RealGalaxy(real_cat, index=index)

    if 'flux' in config:
        flux, safe1 = galsim.config.ParseValue(config, 'flux', base, float)
        safe = safe and safe1
        real_gal.setFlux(flux)

    #print 'RealGal: ',real_gal
    return real_gal, safe


# --- Now we define a function for "ellipsing", rotating, shifting, shearing, in that order.
#
def _BuildTransformObject(gsobject, config, base):
    """@brief Applies ellipticity, rotation, gravitational shearing and centroid shifting to a
    supplied GSObject, in that order, from user input.

    @returns transformed GSObject.
    """
    safe = True
    orig = True
    if 'dilate' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _BuildDilateObject(gsobject, config, 'dilate', base)
        safe = safe and safe1
    if 'dilation' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _BuildDilateObject(gsobject, config, 'dilation', base)
        safe = safe and safe1
    if 'ellip' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _BuildEllipObject(gsobject, config, 'ellip', base)
        safe = safe and safe1
    if 'rotate' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _BuildRotateObject(gsobject, config, 'rotate', base)
        safe = safe and safe1
    if 'rotation' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _BuildRotateObject(gsobject, config, 'rotation', base)
        safe = safe and safe1
    if 'magnify' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _BuildMagnifyObject(gsobject, config, 'magnify', base)
        safe = safe and safe1
    if 'magnification' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _BuildMagnifyObject(gsobject, config, 'magnification', base)
        safe = safe and safe1
    if 'shear' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _BuildEllipObject(gsobject, config, 'shear', base)
        safe = safe and safe1
    if 'shift' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _BuildShiftObject(gsobject, config, 'shift', base)
        safe = safe and safe1
    #print 'Transformed: ',gsobject
    return gsobject, safe

def _BuildEllipObject(gsobject, config, key, base):
    """@brief Applies ellipticity to a supplied GSObject from user input, also used for
    gravitational shearing.

    @returns transformed GSObject.
    """
    shear, safe = galsim.config.ParseValue(config, key, base, galsim.Shear)
    #print 'applyShear with ',shear
    gsobject.applyShear(shear)
    #print 'After applyShear, gsobject = ',gsobject, safe
    return gsobject, safe


def _BuildRotateObject(gsobject, config, key, base):
    """@brief Applies rotation to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    theta, safe = galsim.config.ParseValue(config, key, base, galsim.Angle)
    #print 'theta = ',theta
    gsobject.applyRotation(theta)
    #print 'After applyRotation, gsobject = ',gsobject, safe
    return gsobject, safe

def _BuildDilateObject(gsobject, config, key, base):
    """@brief Applies dilation to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    scale, safe = galsim.config.ParseValue(config, key, base, float)
    #print 'scale = ',scale
    gsobject.applyDilation(scale)
    #print 'After applyDilation, gsobject = ',gsobject, safe
    return gsobject, safe

def _BuildMagnifyObject(gsobject, config, key, base):
    """@brief Applies magnification to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    scale, safe = galsim.config.ParseValue(config, key, base, float)
    #print 'scale = ',scale
    gsobject.applyMagnification(scale)
    #print 'After applyMagnification, gsobject = ',gsobject, safe
    return gsobject, safe

def _BuildShiftObject(gsobject, config, key, base):
    """@brief Applies centroid shift to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    shift, safe = galsim.config.ParseValue(config, key, base, galsim.config.Shift)
    gsobject.applyShift(dx=shift.dx, dy=shift.dy)
    #print 'Shifted: ',gsobject
    return gsobject, safe


# --- Below this point are the functions for getting the required parameters from the user input ---
#
def _GetRequiredKwargs(config, base):
    """@brief Get the required kwargs.
    """
    type = config['type']
    req_kwargs = {}
    safe = True
    for name, properties in galsim.__dict__[type]._params.iteritems():
        if properties[0] is "required":
            # Sanity check here, as far upstream as possible
            if not name in config:
                raise AttributeError("Attribute %s is required for type=%s."%(name,type))
            else:
                value_type = properties[1]
                req_kwargs[name], safe1 = galsim.config.ParseValue(config, name, base, value_type)
                safe = safe and safe1
    return req_kwargs, safe

def _GetSizeKwarg(config, base):
    """@brief Get the one, and one only, required size kwarg.
    """
    type = config['type']
    size_kwarg = {}
    safe = True
    counter = 0  # start the counter
    for name, properties in galsim.__dict__[type]._params.iteritems():
        if name in config and properties[0] is "size":
            counter += 1
            if counter == 1:
                value_type = properties[1]
                size_kwarg[name], safe1 = galsim.config.ParseValue(config, name, base, value_type)
                safe = safe and safe1
            elif counter > 1:
                raise ValueError(
                    "More than one size attribute within input config for type=%s."%type)
    if counter == 0:
        raise ValueError("No size attribute within input config for type=%s."%type)
    return size_kwarg, safe

def _GetOptionalKwargs(config, base):
    """@brief Get the optional kwargs, if any present in the config.
    """
    type = config['type']
    optional_kwargs = {}
    safe = True
    for name, properties in galsim.__dict__[type]._params.iteritems():
        if name in config and properties[0] is "optional":
            value_type = properties[1]
            optional_kwargs[name], safe1 = galsim.config.ParseValue(config, name, base, value_type)
            safe = safe and safe1
    #print 'opt ',optional_kwargs
    return optional_kwargs, safe
