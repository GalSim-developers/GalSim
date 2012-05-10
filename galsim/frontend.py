import galsim

# USERS NOTE - THIS IS CURRENTLY IN DEVELOPMENT AND UNFINISHED!
#

def BuildGSObject(config, input_cat=None, logger=None):
    """Build a GSObject using a config (AttributeDict) and possibly an input_cat (AttributeDict).

    @param config     A configuration AttributDict() read in using galsim.config.load().
    @param input_cat  An input catalog AttributeDict() read in using galsim.io.read_input_cat().
    @param logger     Output logging object. Not used.  Rely on logger picking up 
                        any raised exceptions.
    """
    if not 'type' in config.__dict__:
        raise AttributeError("type attribute required")
    return eval('galsim.Build' + config.type + '(config, input_cat)')

def BuildSimple(config, input_cat, req=[], size_opt=[], opt=[]):
    """Most of the functionality of the Build function is the same for the simple
       objects that are just a profile.  So encapsulate all that here.
       
       @param req       A list of required attributes that config must have
       @param size_opt  A list of size attributes, of which 1 (and only 1) is required
       @param opt       A list of optional attributes
    """
    # Make the argument list for the constructor
    kwargs = {}
    for key in req:
        if not key in config.__dict__:
            raise AttributeError("Required item %s not found for %s."%(key,config.type))
        value = Generate(eval("config." + key),key,input_cat)
        kwargs[key] = value

    for key in opt:
        if key in config.__dict__:
            value = Generate(eval("config." + key),key,input_cat)
            kwargs[key] = value

    # Make sure one and only one size is present
    found = False
    for key in size_opt:
        if key in config.__dict__:
            value = Generate(eval("config." + key),key,input_cat)
            if (found):
                raise AttributeError("Too many sizes for %s"%config.type)
            kwargs[key] = value
            found = True
    if not found:
        raise AttributeError("No size specified for %s"%config.type)

    # Now ready to call the constructor
    return eval("galsim."+config.type+"(**kwargs)")

def BuildGaussian(config, input_cat):
    return galsim.BuildSimple(config, input_cat, [], ['sigma','fwhm','half_light_radius'], ['flux'])

def BuildMoffat(config, input_cat):
    return galsim.BuildSimple(config, input_cat, 
        ['beta'], ['fwhm','scale_radius','half_light_radius'], ['flux','trunc'])

def BuildSersic(config, input_cat):
    return galsim.BuildSimple(config, input_cat, ['n'], ['half_light_radius'], ['flux'])

def BuildExponential(config, input_cat):
    return galsim.BuildSimple(config, input_cat, [], ['half_light_radius','scale_radius'], ['flux'])

def BuildDeVaucouleurs(config, input_cat):
    return galsim.BuildSimple(config, input_cat, [], ['half_light_radius'], ['flux'])

def BuildAiry(config, input_cat):
    return galsim.BuildSimple(config, input_cat, [], ['D'], ['flux','obs'])

def BuildOpticalPSF(config, input_cat):
    return galsim.BuildSimple(config, input_cat, [], ['lam_over_D'],
        ['defocus','astig1','astig2','coma1','coma2','spher','circular_pupil','obs',
         'oversampling','pad_factor'])

def BuildPixel(config, input_cat):

    for key in ['xw','yw']:
        if not key in config.__dict__:
            raise AttributeError("Pixel requires attribute %s."%key)
    kwargs = {}
    kwargs['xw'] = Generate(config.xw,'xw',input_cat)
    kwargs['yw'] = Generate(config.yw,'xw',input_cat)

    if (xw != yw):
        raise Warning("xw != yw found (%f != %f) "%(xw,yw) +
            "This is supported for the pixel, but not the draw routines. " +
            "There might be weirdness....")

    if 'flux' in config.__dict__:
        kwargs['flux'] = Generate(config.flux,'flux',input_cat)

    return galsim.Pixel(**kwargs)

def BuildSquarePixel(config, input_cat):

    if not 'size' in config.__dict__:
        raise AttributeError("SquarePixel requires attribute size.")
    kwargs = {}
    kwargs['xw'] = Generate(config.size,'size',input_cat)
    if 'flux' in config.__dict__:
        kwargs['flux'] = Generate(config.flux,'size',input_cat)
    return galsim.Pixel(**kwargs)

def BuildSum(config, input_cat):

    if not 'items' in config.__dict__:
        raise AttributeError("Sum requires attribute items.")
    list = []
    for item in config.items:
        list.append(BuildGSObject(item, input_cat))
    return galsim.Add(list)

def BuildConvolve(config, input_cat):

    if not 'items' in config.__dict__:
        raise AttributeError("Convolve requires attribute items.")
    list = []
    for item in config.items:
        list.append(BuildGSObject(item, input_cat))
    return galsim.Convolve(list)

def Generate(config, name, input_cat):
    if not hasattr(config,'__dict__'):
        # Then config is really a value
        return config
    elif 'type' in config.__dict__:
        return eval('galsim.GenerateFrom' + config.type + '(config, name, input_cat)')
    else:
        raise AttributeError("Non-value item %s requires a type attribute."%name)

def GenerateFromInputCatalog(config, name, input_cat):
    if input_cat is None:
        raise ValueError("Use of InputCatalog requested for %s, but no input_cat given."%name)

    if not 'col' in config.__dict__:
        raise AttributeError("Use of InputCatalog requested for %s, but no col specified."%name)
    col = config.col

    # input_cat stores the current row to use.
    object_id = input_cat.current

    if object_id > input_cat.nobjects:
        raise IndexError("Trying to access past the end of input catalog for %s."%name +
            " col = %s, object_id = %d"%(col,object_id))

    if input_cat.type is 'ASCII':
        try:
            # config values are 1-based, but we access is 0-based, so use col-1
            value = input_cat.data[object_id,col-1]
        except IndexError:
            raise IndexError("col attribute out of bounds for %s."%name +
                " col = %s, object_id = %d"%(col,object_id))
                    " for accessing input_cat.data [col, object_id] = [%s,%s]"%(col,object_id))

    elif input_cat.type is 'FITS':
        raise NotImplementedError("FITS catalog inputs not yet implemented, sorry!")

    else:
        raise NotImplementedError("Unknown catalog type %s"%input_cat.type)

    return value
