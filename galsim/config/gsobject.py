import galsim

def BuildGSObject(config, key, base=None):
    """Build a GSObject using config dict for key=key.

    @param config     A dict with the configuration information.
    @param key        The key name in config indicating which object to build.
    @param base       A dict which stores potentially useful things like
                      base['rng'] = random number generator
                      base['catalog'] = input catalog for InputCat items
                      base['real_catalog'] = real galaxy catalog for RealGalaxy objects
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
    if 'current_val' in ck and ck['safe']:
        #print 'current is safe:  ',ck['current'], True
        return ck['current_val'], True

    # Change the value for valid aliases:
    if type == 'Sum': type = 'Add'
    if type == 'Convolution': type = 'Convolve'

    # Set up the initial default list of attributes to ignore while building the object:
    ignore = [ 
        'dilate', 'dilation', 'dilate_mu', 'dilation_mu',
        'ellip', 'rotate', 'rotation',
        'magnify', 'magnification', 'magnify_mu', 'magnification_mu',
        'shear', 'shift', 
        'current_val', 'safe' ]
    # There are a few more that are specific to which key we have.
    if key == 'gal':
        ignore += [ 'resolution', 'signal_to_noise' ]
    elif key == 'psf':
        ignore += [ 'saved_re' ]

    # See if this type has a specialized build function:
    build_func_name  = '_Build' + type
    if build_func_name in galsim.config.gsobject.__dict__:
        build_func  = eval(build_func_name)
        gsobject, safe = build_func(ck, key, base, ignore)
    # Next, we check if this name is in the galsim dictionary.
    elif type in galsim.__dict__:
        if issubclass(galsim.__dict__[type], galsim.GSObject):
            gsobject, safe = _BuildSimple(ck, key, base, ignore)
        else:
            TypeError("Input config type = %s is not a GSObject."%type)
    # Otherwise, it's not a valid type.
    else:
        raise NotImplementedError("Unrecognised config type = %s"%type)

    # If this is a psf, try to save the half_light_radius in case gal uses resolution.
    if key == 'psf':
        try : 
            ck['saved_re'] = gsobject.getHalfLightRadius()
        except :
            pass
    
    # Apply any dilation, ellip, shear, etc. modifications.
    gsobject, safe1 = _TransformObject(gsobject, ck, base)
    safe = safe and safe1

    if 'no_save' not in base:
        ck['current_val'] = gsobject
        ck['safe'] = safe

    return gsobject, safe

def _BuildAdd(config, key, base, ignore):
    """@brief  Build an Add object
    """
    req = { 'items' : list }
    opt = { 'flux' : float }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    galsim.config.CheckAllParams(config, key, req=req, opt=opt, ignore=ignore)

    gsobjects = []
    items = config['items']
    if not isinstance(items,list):
        raise AttributeError("items entry for config.%s entry is not a list."%type)
    safe = True
    for i in range(len(items)):
        gsobject, safe1 = BuildGSObject(items, i, base)
        safe = safe and safe1
        gsobjects.append(gsobject)
    #print 'After built component items for ',type,' safe = ',safe

    # Special: if the last item in a Sum doesn't specify a flux, we scale it
    # to bring the total flux up to 1.
    if ('flux' not in items[-1]) and all('flux' in item for item in items[0:-1]):
        sum = 0
        for item in items[0:-1]:
            sum += galsim.config.value.GetCurrentValue(item,'flux')
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

    if 'flux' in config:
        flux, safe1 = galsim.config.ParseValue(config, 'flux', base, float)
        #print 'flux = ',flux
        gsobject.setFlux(flux)
        safe = safe and safe1

    return gsobject, safe

def _BuildConvolve(config, key, base, ignore):
    """@brief  Build a Convolve object
    """
    req = { 'items' : list }
    opt = { 'flux' : float }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    galsim.config.CheckAllParams(config, key, req=req, opt=opt, ignore=ignore)

    gsobjects = []
    items = config['items']
    if not isinstance(items,list):
        raise AttributeError("items entry for config.%s entry is not a list."%type)
    safe = True
    for i in range(len(items)):
        gsobject, safe1 = BuildGSObject(items, i, base)
        safe = safe and safe1
        gsobjects.append(gsobject)
    #print 'After built component items for ',type,' safe = ',safe

    gsobject = galsim.Convolve(gsobjects)

    if 'flux' in config:
        flux, safe1 = galsim.config.ParseValue(config, 'flux', base, float)
        #print 'flux = ',flux
        gsobject.setFlux(flux)
        safe = safe and safe1

    return gsobject, safe

def _BuildList(config, key, base, ignore):
    """@brief  Build a GSObject selected from a List
    """
    req = { 'items' : list }
    opt = { 'index' : float , 'flux' : float }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    galsim.config.CheckAllParams(config, key, req=req, opt=opt, ignore=ignore)

    items = config['items']
    if not isinstance(items,list):
        raise AttributeError("items entry for config.%s entry is not a list."%type)
    if 'index' not in config:
        config['index'] = { 'type' : 'Sequence' , 'first' : 0 , 'last' : len(items)-1 }
    index, safe = galsim.config.ParseValue(config, 'index', base, int)
    if index < 0 or index >= len(items):
        raise AttributeError("index %d out of bounds for config.%s"%(index,type))
    #print items[index]['type']
    #print 'index = ',index,' From ',key,' List: ',items[index]
    gsobject, safe1 = BuildGSObject(items, index, base)
    safe = safe and safe1

    if 'flux' in config:
        flux, safe1 = galsim.config.ParseValue(config, 'flux', base, float)
        #print 'flux = ',flux
        gsobject.setFlux(flux)
        safe = safe and safe1

    return gsobject, safe


def _BuildPixel(config, key, base, ignore):
    """@brief Build a Pixel type GSObject from user input.
    """
    kwargs, safe = galsim.config.GetAllParams(config, key, base, 
        req = galsim.__dict__['Pixel']._req_params,
        opt = galsim.__dict__['Pixel']._opt_params,
        single = galsim.__dict__['Pixel']._single_params,
        ignore = ignore)

    if 'yw' in kwargs.keys() and (kwargs['xw'] != kwargs['yw']):
        import warnings
        warnings.warn(
            "xw != yw found (%f != %f) "%(kwargs['xw'], kwargs['yw']) +
            "This is supported for the pixel, but not the draw routines. " +
            "There might be weirdness....")

    try:
        return galsim.Pixel(**kwargs), safe
    except Exception, err_msg:
        raise RuntimeError("Unable to construct Pixel object with kwargs=%s."%str(kwargs) +
                           "Original error message: %s"%err_msg)


def _BuildRealGalaxy(config, key, base, ignore):
    """@brief Build a RealGalaxy type GSObject from user input.
    """
    if 'real_catalog' not in base:
        raise ValueError("No real galaxy catalog available for building type = RealGalaxy")
    real_cat = base['real_catalog']

    # Special: if index is Sequence or Random, and max isn't set, set it to real_cat.nobjects-1
    galsim.config.SetDefaultIndex(config, real_cat.nobjects)

    kwargs, safe = galsim.config.GetAllParams(config, key, base, 
        req = galsim.__dict__['RealGalaxy']._req_params,
        opt = galsim.__dict__['RealGalaxy']._opt_params,
        single = galsim.__dict__['RealGalaxy']._single_params,
        ignore = ignore)

    index = kwargs['index']
    if index >= real_cat.nobjects:
        raise IndexError(
            "%s index has gone past the number of entries in the catalog"%param_name)

    try:
        return galsim.RealGalaxy(real_cat, **kwargs), safe
    except Exception, err_msg:
        raise RuntimeError("Unable to construct RealGalaxy object with kwargs=%s."%str(kwargs) +
                           "Original error message: %s"%err_msg)


def _BuildSimple(config, key, base, ignore):
    """@brief Build a simple GSObject (i.e. one without a specialized _Build function) or
    any other galsim object that defines _req_params, _opt_params and _single_params.
    """
    # Build the kwargs according to the various params objects in the class definition.
    type = config['type']
    kwargs, safe = galsim.config.GetAllParams(config, key, base, 
        req = galsim.__dict__[type]._req_params,
        opt = galsim.__dict__[type]._opt_params,
        single = galsim.__dict__[type]._single_params,
        ignore = ignore)

    # Finally, after pulling together all the params, try making the GSObject.
    try:
        init_func = eval("galsim."+type)
        return init_func(**kwargs), safe
    except Exception, err_msg:
        raise RuntimeError("Unable to construct %s object with kwargs=%s."%(type,str(kwargs)) +
                           "Original error message: %s"%err_msg)


def _TransformObject(gsobject, config, base):
    """@brief Applies ellipticity, rotation, gravitational shearing and centroid shifting to a
    supplied GSObject, in that order, from user input.

    @returns transformed GSObject.
    """
    safe = True
    orig = True
    if 'dilate' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _DilateObject(gsobject, config, 'dilate', base)
        safe = safe and safe1
    if 'dilation' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _DilateObject(gsobject, config, 'dilation', base)
        safe = safe and safe1
    if 'dilate_mu' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _DilateMuObject(gsobject, config, 'dilate_mu', base)
        safe = safe and safe1
    if 'dilation_mu' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _DilateMuObject(gsobject, config, 'dilation_mu', base)
        safe = safe and safe1
    if 'ellip' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _EllipObject(gsobject, config, 'ellip', base)
        safe = safe and safe1
    if 'rotate' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _RotateObject(gsobject, config, 'rotate', base)
        safe = safe and safe1
    if 'rotation' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _RotateObject(gsobject, config, 'rotation', base)
        safe = safe and safe1
    if 'magnify' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _MagnifyObject(gsobject, config, 'magnify', base)
        safe = safe and safe1
    if 'magnification' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _MagnifyObject(gsobject, config, 'magnification', base)
        safe = safe and safe1
    if 'magnify_mu' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _MagnifyMuObject(gsobject, config, 'magnify_mu', base)
        safe = safe and safe1
    if 'magnification_mu' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _MagnifyMuObject(gsobject, config, 'magnification_mu', base)
        safe = safe and safe1
    if 'shear' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _EllipObject(gsobject, config, 'shear', base)
        safe = safe and safe1
    if 'shift' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _ShiftObject(gsobject, config, 'shift', base)
        safe = safe and safe1
    #print 'Transformed: ',gsobject
    return gsobject, safe

def _EllipObject(gsobject, config, key, base):
    """@brief Applies ellipticity to a supplied GSObject from user input, also used for
    gravitational shearing.

    @returns transformed GSObject.
    """
    shear, safe = galsim.config.ParseValue(config, key, base, galsim.Shear)
    #print 'applyShear with ',shear
    gsobject.applyShear(shear)
    #print 'After applyShear, gsobject = ',gsobject, safe
    return gsobject, safe


def _RotateObject(gsobject, config, key, base):
    """@brief Applies rotation to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    theta, safe = galsim.config.ParseValue(config, key, base, galsim.Angle)
    #print 'theta = ',theta
    gsobject.applyRotation(theta)
    #print 'After applyRotation, gsobject = ',gsobject, safe
    return gsobject, safe

def _DilateObject(gsobject, config, key, base):
    """@brief Applies dilation to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    scale, safe = galsim.config.ParseValue(config, key, base, float)
    #print 'scale = ',scale
    gsobject.applyDilation(scale)
    #print 'After applyDilation, gsobject = ',gsobject, safe
    return gsobject, safe

def _DilateMuObject(gsobject, config, key, base):
    """@brief Applies dilation to a supplied GSObject based on user input
       according to a mu value rather than scale.  (scale = exp(mu))

    @returns transformed GSObject.
    """
    mu, safe = galsim.config.ParseValue(config, key, base, float)
    #print 'scale = ',scale
    gsobject.applyDilation(exp(mu))
    #print 'After applyDilation, gsobject = ',gsobject, safe
    return gsobject, safe

def _MagnifyObject(gsobject, config, key, base):
    """@brief Applies magnification to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    scale, safe = galsim.config.ParseValue(config, key, base, float)
    #print 'scale = ',scale
    gsobject.applyMagnification(scale)
    #print 'After applyMagnification, gsobject = ',gsobject, safe
    return gsobject, safe

def _MagnifyMuObject(gsobject, config, key, base):
    """@brief Applies magnification to a supplied GSObject based on user input
       according to a mu value rather than scale.  (scale = exp(mu))

    @returns transformed GSObject.
    """
    mu, safe = galsim.config.ParseValue(config, key, base, float)
    #print 'scale = ',scale
    gsobject.applyMagnification(exp(mu))
    #print 'After applyMagnification, gsobject = ',gsobject, safe
    return gsobject, safe

def _ShiftObject(gsobject, config, key, base):
    """@brief Applies centroid shift to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    shift, safe = galsim.config.ParseValue(config, key, base, galsim.config.Shift)
    gsobject.applyShift(dx=shift.dx, dy=shift.dy)
    #print 'Shifted: ',gsobject
    return gsobject, safe

