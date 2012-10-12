import galsim


def ParseValue(config, param_name, base, value_type):
    """@brief Read or generate a parameter value from config.

    @return value, safe
    """
    param = config[param_name]
    #print 'ParseValue for param_name = ',param_name,', value_type = ',str(value_type)
    #print 'param = ',param

    # First see if we can assign by param by a direct constant value
    if isinstance(param, value_type):
        #print param_name,' = ',param
        return param, True
    elif not isinstance(param, dict):
        if value_type is galsim.Angle:
            # Angle is a special case.  Angles are specified with a final string to 
            # declare what unit to use.
            val = _GetAngleValue(param, param_name)
        elif value_type is bool:
            # For bool, we allow a few special string conversions
            val = _GetBoolValue(param, param_name)
        else:
            # Make sure strings are converted to float (or other type) if necessary.
            # In particular things like 1.e6 aren't converted to float automatically
            # by the yaml reader. (Although I think this is a bug.)
            try: 
                val = value_type(param)
            except:
                raise AttributeError(
                    "Could not convert %s param = %s to type %s."%(param_name,param,value_type))
        # Save the converted type for next time.
        config[param_name] = val
        #print param_name,' = ',val
        return val, True
    elif 'type' not in param:
        raise AttributeError(
            "%s.type attribute required in config for non-constant parameter %s."
                    %(param_name,param_name))
    else:
        # Otherwise, we need to generate the value according to its type
        valid_types = {
            float : [ 'InputCatalog', 'Random', 'RandomGaussian', 'NFWHaloMag',
                      'Sequence', 'List', 'Eval' ],
            int : [ 'InputCatalog', 'Random', 'Sequence', 'List', 'Eval' ],
            bool : [ 'InputCatalog', 'Random', 'Sequence', 'List', 'Eval' ],
            str : [ 'InputCatalog', 'NumberedFile', 'List', 'Eval' ],
            galsim.Angle : [ 'Rad', 'Deg', 'Random', 'List', 'Eval' ],
            galsim.Shear : [ 'E1E2', 'EBeta', 'G1G2', 'GBeta', 'Eta1Eta2', 'EtaBeta', 'QBeta',
                             'NFWHaloShear', 'PowerSpectrumShear', 'List', 'Eval' ],
            galsim.PositionD : [ 'XY', 'RTheta', 'RandomCircle', 'List', 'Eval' ] 
        }

        type = param['type']
        #print 'type = ',type

        # Apply valid aliases:
        if type == 'Radians': type = 'Rad'
        if type == 'Degrees': type = 'Deg'

        # First check if the value_type is valid.
        if value_type not in valid_types.keys():
            raise AttributeError(
                "Unrecognized value_type = %s in ParseValue"%value_type)
            
        if type not in valid_types[value_type]:
            raise AttributeError(
                "Invalid type = %s specified for parameter %s with value_type = %s."%(
                        type, param_name, value_type))

        generate_func = eval('_GenerateFrom' + type)
        #print 'generate_func = ',generate_func
        val, safe = generate_func(param, param_name, base, value_type)
        #print 'returned val, safe = ',val,safe

        # Make sure we really got the right type back.  (Just in case...)
        try : 
            if not isinstance(val,value_type):
                val = value_type(val)
        except :
            raise AttributeError(
                "Could not convert %s param = %s to type %s."%(param_name,val,value_type))
        param['current_val'] = val
        #print param_name,' = ',val
        return val, safe


def _GetAngleValue(param, param_name):
    """ @brief Convert a string consisting of a value and an angle unit into an Angle.
    """
    try :
        value, unit = param.rsplit(None,1)
        value = float(value)
        unit = unit.lower()
        if unit.startswith('rad') :
            return galsim.Angle(value, galsim.radians)
        elif unit.startswith('deg') :
            return galsim.Angle(value, galsim.degrees)
        elif unit.startswith('hour') :
            return galsim.Angle(value, galsim.hours)
        elif unit.startswith('hr') :
            return galsim.Angle(value, galsim.hours)
        elif unit.startswith('arcmin') :
            return galsim.Angle(value, galsim.arcmin)
        elif unit.startswith('arcsec') :
            return galsim.Angle(value, galsim.arcsec)
        else :
            raise AttributeError("Unknown Angle unit: %s for %s param"%(unit,param_name))
    except :
        raise AttributeError("Unable to parse %s param = %s as an Angle."%(param_name,param))


def _GetPositionValue(param, param_name):
    """ @brief Convert a string that looks like "a,b" into a galsim.PositionD.
    """
    try :
        x, y = param.split(',')
        x = x.strip()
        y = y.strip()
        return galsim.PositionD(x,y)
    except :
        raise AttributeError("Unable to parse %s param = %s as a PositionD."%(param_name,param))


def _GetBoolValue(param, param_name):
    """ @brief Convert a string to a bool
    """
    #print 'GetBoolValue: param = ',param
    if isinstance(param,str):
        #print 'param.strip.upper = ',param.strip().upper()
        if param.strip().upper() in [ 'TRUE', 'YES', '1' ]:
            return True
        elif param.strip().upper() in [ 'FALSE', 'NO', '0' ]:
            return False
        else:
            raise AttributeError("Unable to parse %s param = %s as a bool."%(param_name,param))
    else:
        try:
            val = bool(param)
            return val
        except:
            raise AttributeError("Unable to parse %s param = %s as a bool."%(param_name,param))


def CheckAllParams(param, param_name, req={}, opt={}, single=[], ignore=[]):
    """@brief Check that the parameters for a particular item are all valid
    
    @return a dict, get, with get[key] = value_type for all keys to get
    """
    get = {}
    valid_keys = req.keys() + opt.keys()
    # Check required items:
    for (key, value_type) in req.items():
        if key in param:
            get[key] = value_type
        else:
            raise AttributeError(
                "Attribute %s is required for %s.type = %s"%(key,param_name,param['type']))

    # Check optional items:
    for (key, value_type) in opt.items():
        if key in param:
            get[key] = value_type

    # Check items for which exacly 1 should be defined:
    for s in single: 
        if not s: # If no items in list, don't require one of them to be present.
            break
        valid_keys += s.keys()
        count = 0
        for (key, value_type) in s.items():
            if key in param:
                count += 1
                if count > 1:
                    raise AttributeError(
                        "Only one of the attributes %s is allowed for %s.type = %s"%(
                            s.keys(),param_name,param['type']))
                get[key] = value_type
        if count == 0:
            raise AttributeError(
                "One of the attributes %s is required for %s.type = %s"%(
                    s.keys(),param_name,param['type']))

    # Check that there aren't any extra keys in param:
    valid_keys += ignore
    valid_keys += [ 'type', 'current_val' ]  # These might be there, and it's ok.
    valid_keys += [ '#' ] # When we read in json files, there represent comments
    for key in param.keys():
        if key not in valid_keys:
            raise AttributeError(
                "Unexpected attribute %s found for parameter %s"%(key,param_name))

    return get


def GetAllParams(param, param_name, base, req={}, opt={}, single=[], ignore=[]):
    """@brief Check and get all the parameters for a particular item

    @return kwargs, safe
    """
    get = CheckAllParams(param,param_name,req,opt,single,ignore)
    kwargs = {}
    safe = True
    for (key, value_type) in sorted(get.items()):
        val, safe1 = ParseValue(param, key, base, value_type)
        safe = safe and safe1
        kwargs[key] = val
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    kwargs = dict([(k.encode('utf-8'), v) for k,v in kwargs.iteritems()])
    return kwargs, safe


def GetCurrentValue(config, param_name):
    """@brief Return the current value of a parameter (either stored or a simple value)
    """
    param = config[param_name]
    if isinstance(param, dict):
        return param['current_val']
    else: 
        return param


#
# Now all the GenerateFrom functions:
#

def _GenerateFromG1G2(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (g1, g2)
    """
    req = { 'g1' : float, 'g2' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print 'Generate from G1G2: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromE1E2(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (e1, e2)
    """
    req = { 'e1' : float, 'e2' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    return galsim.Shear(**kwargs), safe

def _GenerateFromEta1Eta2(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (eta1, eta2)
    """
    req = { 'eta1' : float, 'eta2' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    return galsim.Shear(**kwargs), safe

def _GenerateFromGBeta(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (g, beta)
    """
    req = { 'g' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    return galsim.Shear(**kwargs), safe

def _GenerateFromEBeta(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (e, beta)
    """
    req = { 'e' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    return galsim.Shear(**kwargs), safe

def _GenerateFromEtaBeta(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (eta, beta)
    """
    req = { 'eta' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    return galsim.Shear(**kwargs), safe

def _GenerateFromQBeta(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (q, beta)
    """
    req = { 'q' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    return galsim.Shear(**kwargs), safe

def _GenerateFromXY(param, param_name, base, value_type):
    """@brief Return a PositionD constructed from given (x,y)
    """
    req = { 'x' : float, 'y' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    return galsim.PositionD(**kwargs), safe

def _GenerateFromRTheta(param, param_name, base, value_type):
    """@brief Return a PositionD constructed from given (r,theta)
    """
    req = { 'r' : float, 'theta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    r = kwargs['r']
    theta = kwargs['theta']
    import math
    return galsim.PositionD(r*math.cos(theta.rad()), r*math.sin(theta.rad())), safe

def _GenerateFromRad(param, param_name, base, value_type):
    """@brief Return an Angle constructed from given theta in radians
    """
    req = { 'theta' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    return kwargs['theta'] * galsim.radians, safe

def _GenerateFromDeg(param, param_name, base, value_type):
    """@brief Return an Angle constructed from given theta in degrees
    """
    req = { 'theta' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    return kwargs['theta'] * galsim.degrees, safe


def _GenerateFromInputCatalog(param, param_name, base, value_type):
    """@brief Return a value read from an input catalog
    """
    if 'catalog' not in base:
        raise ValueError("No input catalog available for %s.type = InputCatalog"%param_name)
    input_cat = base['catalog']

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with an InputCatalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    SetDefaultIndex(param, input_cat.nobjects)

    req = { 'col' : int , 'index' : int }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)

    col = kwargs['col']
    if col >= input_cat.ncols:
        raise IndexError("%s.col attribute (=%d) out of bounds"%(param_name,col))

    index = kwargs['index']
    if index >= input_cat.nobjects:
        raise IndexError(
            "%s index has gone past the number of entries in the catalog"%param_name)

    str = input_cat.data[index, col]
    # We want to parse this string with ParseValue, but we need a dict to do that:
    temp_dict = { param_name : str }
    val = ParseValue(temp_dict,param_name,base,value_type)[0]

    #print 'InputCatalog: ',str,val
    return val, False


def _GenerateFromRandom(param, param_name, base, value_type):
    """@brief Return a random value drawn from a uniform distribution
    """
    if 'rng' not in base:
        raise ValueError("No rng available for %s.type = Random"%param_name)
    rng = base['rng']
    ud = galsim.UniformDeviate(rng)

    # Each value_type works a bit differently:
    if value_type is galsim.Angle:
        import math
        CheckAllParams(param, param_name)
        val = ud() * 2 * math.pi * galsim.radians
        #print 'Random angle = ',val
        return val, False
    elif value_type is bool:
        CheckAllParams(param, param_name)
        val = ud() < 0.5
        #print 'Random bool = ',val
        return val, False
    else:
        req = { 'min' : value_type , 'max' : value_type }
        kwargs, safe = GetAllParams(param, param_name, base, req=req)

        min = kwargs['min']
        max = kwargs['max']

        if value_type is int:
            import math
            val = int(math.floor(ud() * (max-min+1))) + min
            # In case ud() == 1
            if val > max: val = max
        else:
            val = ud() * (max-min) + min

        #print 'Random = ',val
        return val, False


def _GenerateFromRandomGaussian(param, param_name, base, value_type):
    """@brief Return a random value drawn from a Gaussian distribution
    """
    if 'rng' not in base:
        raise ValueError("No rng available for %s.type = RandomGaussian"%param_name)
    rng = base['rng']

    req = { 'sigma' : float }
    opt = { 'mean' : float, 'min' : float, 'max' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req, opt=opt)

    sigma = kwargs['sigma']

    if 'gd' in base:
        # Minor subtlety here.  GaussianDeviate requires two random numbers to 
        # generate a single Gaussian deviate.  But then it gets a second 
        # deviate for free.  So it's more efficient to store gd than to make
        # a new one each time.  So check if we did that.
        gd = base['gd']
        if base['current_gdsigma'] != sigma:
            gd.setSigma(sigma)
            base['current_gdsigma'] = sigma
    else:
        # Otherwise, just go ahead and make a new one.
        gd = galsim.GaussianDeviate(rng,sigma=sigma)
        base['gd'] = gd
        base['current_gdsigma'] = sigma

    if 'min' in kwargs or 'max' in kwargs:
        # Clip at min/max.
        # However, special cases if min == mean or max == mean
        #  -- can use fabs to double the chances of falling in the range.
        mean = kwargs.get('mean',0.)
        min = kwargs.get('min',-float('inf'))
        max = kwargs.get('max',float('inf'))

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
        #print 'sigma = ',sigma
        import math
        while True:
            val = gd()
            #print 'val = ',val
            if do_abs: val = math.fabs(val)
            if val >= min and val <= max: break
        if do_neg: val = -val
        val += mean
    else:
        val = gd()
        if 'mean' in kwargs: val += kwargs['mean']

    #print 'RandomGaussian: ',val
    return val, False


def _GenerateFromRandomCircle(param, param_name, base, value_type):
    """@brief Return a PositionD drawn from a circular top hat distribution.
    """
    if 'rng' not in base:
        raise ValueError("No rng available for %s.type = RandomCircle"%param_name)
    rng = base['rng']

    req = { 'radius' : float }
    opt = { 'inner_radius' : float, 'center' : galsim.PositionD }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    radius = kwargs['radius']

    ud = galsim.UniformDeviate(rng)
    max_rsq = radius**2
    if 'inner_radius' in kwargs:
        inner_radius = kwargs['inner_radius']
        min_rsq = inner_radius**2
    else:
        min_rsq = 0.
    # Emulate a do-while loop
    while True:
        x = (2*ud()-1) * radius
        y = (2*ud()-1) * radius
        rsq = x**2 + y**2
        if rsq <= max_rsq: break
    #print 'RandomCircle: ',(x,y)
    pos = galsim.PositionD(x,y)
    if 'center' in kwargs:
        pos += params['center']
    return pos, False


def _GenerateFromSequence(param, param_name, base, value_type):
    """@brief Return next in a sequence of integers
    """
    #print 'Start Sequence for ',param_name,' -- param = ',param
    opt = { 'first' : value_type, 'last' : value_type, 'step' : value_type,
            'repeat' : int, 'nitems' : int }
    kwargs, safe = GetAllParams(param, param_name, base, opt=opt)

    step = kwargs.get('step',1)
    first = kwargs.get('first',0)
    repeat = kwargs.get('repeat',1)
    last = kwargs.get('last',None)
    nitems = kwargs.get('nitems',None)
    #print 'first, step, last, repeat, nitems = ',first,step,last,repeat,nitems
    if repeat <= 0:
        raise ValueError(
            "Invalid repeat=%d (must be > 0) for %s.type = Sequence"%(repeat,param_name))
    if last is not None and nitems is not None:
        raise AttributeError(
            "At most one of the attributes last and nitems is allowed for %s.type = Sequence"%(
                param_name))

    if value_type is bool:
        # Then there are only really two valid sequences: Either 010101... or 101010...
        # Aside from the repeat value of course.
        if first:
            first = 1
            step = -1
            nitems = 2
        else:
            first = 0
            step = 1
            nitems = 2
        #print 'bool sequence: first, step, repeat, n => ',first,step,repeat,nitems

    elif value_type is float:
        if last is not None:
            nitems = int( (last-first)/step + 0.5 ) + 1
        #print 'float sequence: first, step, repeat, n => ',first,step,repeat,nitems
    else:
        if last is not None:
            nitems = (last - first)/step + 1
        #print 'int sequence: first, step, repeat, n => ',first,step,repeat,nitems

    k = base['seq_index']
    #print 'k = ',k

    k = k / repeat
    #print 'k/repeat = ',k

    if nitems is not None and nitems > 0:
        #print 'nitems = ',nitems
        k = k % nitems
        #print 'k%nitems = ',k

    index = first + k*step
    #print 'first + k*step = ',index

    return index, False


def _GenerateFromNumberedFile(param, param_name, base, value_type):
    """@brief Return a file_name using a root, a number, and an extension
    """
    #print 'Start NumberedFile for ',param_name,' -- param = ',param
    if 'num' not in param:
        param['num'] = { 'type' : 'Sequence' }
    req = { 'root' : str , 'num' : int }
    opt = { 'ext' : str , 'digits' : int }
    kwargs, safe = GetAllParams(param, param_name, base, req=req, opt=opt)

    template = kwargs['root']
    if 'digits' in kwargs:
        template += '%%0%dd'%kwargs['digits']
    else:
        template += '%d'
    if 'ext' in kwargs:
        template += kwargs['ext']
    #print 'template = ',template
    s = eval("'%s'%%%d"%(template,kwargs['num']))
    #print 'num = ',kwargs['num']
    #print 's = ',s
    
    return s, safe


def _GenerateFromNFWHaloShear(param, param_name, base, value_type):
    """@brief Return a shear calculated from an NFWHalo object.
    """
    if 'pos' not in base:
        raise ValueError("NFWHaloShear requested, but no position defined.")
    pos = base['pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise ValueError("NFWHaloShear requested, but no gal.redshift defined.")
    redshift = GetCurrentValue(base['gal'],'redshift')

    if 'nfw_halo' not in base:
        raise ValueError("NFWHaloShear requested, but no input.nfw_halo defined.")
    
    req = {}
    # Only Check, not Get.  (There's nothing to get -- just make sure there aren't extra params.)
    CheckAllParams(param, param_name, req=req)

    #print 'NFWHaloShear: pos = ',pos,' z = ',redshift
    try:
        g1,g2 = base['nfw_halo'].getShear(pos,redshift)
        #print 'g1,g2 = ',g1,g2
        shear = galsim.Shear(g1=g1,g2=g2)
    except:
        import warnings
        warnings.warn("Warning: NFWHalo shear is invalid -- probably strong lensing!  " +
                      "Using shear = 0.")
        shear = galsim.Shear(g1=0,g2=0)
    #print 'shear = ',shear
    return shear, False


def _GenerateFromNFWHaloMag(param, param_name, base, value_type):
    """@brief Return a magnification calculated from an NFWHalo object.
    """
    if 'pos' not in base:
        raise ValueError("NFWHaloMag requested, but no position defined.")
    pos = base['pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise ValueError("NFWHaloMag requested, but no gal.redshift defined.")
    redshift = GetCurrentValue(base['gal'],'redshift')

    if 'nfw_halo' not in base:
        raise ValueError("NFWHaloMag requested, but no input.nfw_halo defined.")
    
    opt = { 'max_scale' : float }
    kwargs = GetAllParams(param, param_name, base, opt=opt)[0]

    #print 'NFWHaloMag: pos = ',pos,' z = ',redshift
    mu = base['nfw_halo'].getMag(pos,redshift)
    #print 'mu = ',mu

    max_scale = kwargs.get('max_scale', 5.)
    if not max_scale > 0.: 
        raise ValueError(
            "Invalid max_scale=%f (must be > 0) for %s.type = NFWHaloMag"%(repeat,param_name))

    if mu < 0 or mu > max_scale**2:
        import warnings
        warnings.warn("Warning: NFWHalo mu = %f means strong lensing!  Using scale=5."%mu)
        scale = max_scale
    else:
        import math
        scale = math.sqrt(mu)
    #print 'scale = ',scale
    return scale, False


def _GenerateFromPowerSpectrumShear(param, param_name, base, value_type):
    """@brief Return a shear calculated from a PowerSpectrum object.
    """
    if 'pos' not in base:
        raise ValueError("PowerSpectrumShear requested, but no position defined.")
    pos = base['pos']

    if 'power_spectrum' not in base:
        raise ValueError("PowerSpectrumShear requested, but no input.power_spectrum defined.")
    
    req = {}
    # Only Check, not Get.  (There's nothing to get -- just make sure there aren't extra params.)
    CheckAllParams(param, param_name, req=req)

    #print 'PowerSpectrumShear: pos = ',pos
    try:
        g1,g2 = base['power_spectrum'].getShear(pos)
        #print 'g1,g2 = ',g1,g2
        shear = galsim.Shear(g1=g1,g2=g2)
    except:
        import warnings
        warnings.warn("Warning: PowerSpectrum shear is invalid -- probably strong lensing!  " +
                      "Using shear = 0.")
        shear = galsim.Shear(g1=0,g2=0)
    #print 'shear = ',shear
    return shear, False


def _GenerateFromList(param, param_name, base, value_type):
    """@brief Return next item from a provided list
    """
    req = { 'items' : list }
    opt = { 'index' : int }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    CheckAllParams(param, param_name, req=req, opt=opt)
    items = param['items']
    if not isinstance(items,list):
        raise AttributeError("items entry for parameter %s is not a list."%param_name)

    # Setup the indexing sequence if it hasn't been specified using the length of items.
    SetDefaultIndex(param, len(items))
    index, safe = ParseValue(param, 'index', base, int)

    if index < 0 or index >= len(items):
        raise AttributeError("index %d out of bounds for parameter %s"%(index,param_name))
    val, safe1 = ParseValue(items, index, base, value_type)
    safe = safe and safe1
    return val, safe
 
def type_by_letter(key):
    if len(key) < 2:
        raise AttributeError("Invalid user-defined variable %r"%key)
    if key[0] == 'f':
        return float
    elif key[0] == 'i':
        return int
    elif key[0] == 'b':
        return bool
    elif key[0] == 's':
        return str
    elif key[0] == 'a':
        return galsim.Angle
    elif key[0] == 'p':
        return galsim.PositionD
    elif key[0] == 'g':
        return galsim.Shear
    else:
        raise AttributeError("Invalid Eval variable: %s (starts with an invalid letter)"%key)

def _GenerateFromEval(param, param_name, base, value_type):
    """@brief Evaluate a string as the provided type
    """
    #print 'Start Eval for ',param_name
    req = { 'str' : str }
    opt = {}
    ignore = [ 'type' , 'current_val' ]
    for key in param.keys():
        if key not in (ignore + req.keys()):
            opt[key] = type_by_letter(key)
    #print 'opt = ',opt
            
    params, safe = GetAllParams(param, param_name, base, req=req, opt=opt, ignore=ignore)
    #print 'params = ',params
    string = params['str']
    #print 'string = ',string

    # Bring the user-defined variables into scope.
    for key in opt.keys():
        exec(key[1:] + ' = params[key]')
        #print key[1:],'=',eval(key[1:])

    # Also bring in any top level eval_variables
    if 'eval_variables' in base:
        #print 'found eval_variables = ',base['eval_variables']
        if not isinstance(base['eval_variables'],dict):
            raise AttributeError("eval_variables must be a dict")
        opt = {}
        for key in base['eval_variables'].keys():
            if key not in ignore:
                opt[key] = type_by_letter(key)
        #print 'opt = ',opt
        params, safe1 = GetAllParams(base['eval_variables'], 'eval_variables', base, opt=opt,
                                     ignore=ignore)
        #print 'params = ',params
        safe = safe and safe1
        for key in opt.keys():
            exec(key[1:] + ' = params[key]')
            #print key[1:],'=',eval(key[1:])

    # Also, we allow the use of math functions
    import math
    import numpy
    import os

    # Try evaluating the string as is.
    try:
        val = value_type(eval(string))
        #print 'Simple success: val = ',val
        return val, safe
    except:
        pass

    # Then try bringing in the allowed variables to see if that works:
    if 'pos' in base:
        pos = base['pos']
        #print 'pos = ',pos
    if 'rng' in base:
        rng = base['rng']
    if 'catalog' in base:
        catalog = base['catalog']
    if 'real_catalog' in base:
        real_catalog = base['real_catalog']
    if 'nfw_halo' in base:
        nfw_halo = base['nfw_halo']
    if 'power_spectrum' in base:
        power_spectrum = base['power_spectrum']

    try:
        val = value_type(eval(string))
        #print 'Needed pos: val = ',val
        return val, False
    except:
        raise ValueError("Unable to evaluate string %r as a %s for %s"%(
                string,value_type,param_name))


def SetDefaultIndex(config, num):
    """
    When the number of items in a list is known, we allow the user to omit some of 
    the parameters of a Sequence or Random and set them automatically based on the 
    size of the list, catalog, etc.
    """
    if 'index' not in config:
        config['index'] = { 'type' : 'Sequence', 'nitems' : num }
    elif isinstance(config['index'],dict) and 'type' in config['index'] :
        index = config['index']
        type = index['type']
        if ( type == 'Sequence' and 
             ('step' not in index or (isinstance(index['step'],int) and index['step'] > 0) ) and
             'last' not in index and 'nitems' not in index ):
            index['last'] = num-1
        elif ( type == 'Sequence' and 
             ('step' in index and (isinstance(index['step'],int) and index['step'] < 0) ) ):
            if 'first' not in index:
                index['first'] = num-1
            if 'last' not in index and 'nitems' not in index:
                index['last'] = 0
        elif type == 'Random':
            if 'max' not in index:
                index['max'] = num-1
            if 'min' not in index:
                index['min'] = 0


