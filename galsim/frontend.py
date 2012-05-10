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
    if not config.hasattr('type'):
        raise AttributeError("type attribute required")
    return eval('galsim.Build' + config.type + '(config, input_cat)')

def BuildSimple(config, input_cat, req=[], size_opt=[], opt=[]):
    """Most of the functionality of the Build function is the same for the simple
       objects that are just a profile.  So encapsulate all that here.
       
       @param req       A list of required attributes that config must have
       @param size_opt  A list of size attributes, of which 1 (and only 1) is required
       @param opt       A list of optional attributes
       In addition to what is listed, the flux is always optional.
    """
    # All simple builders have an optional flux attribute so add that to opt
    opt += ['flux']

    # Make the argument list for the constructor
    kwargs = {}
    for key in req:
        if not config.hasattr(key):
            raise AttributeError()
        value = Generate(eval("config." + key),input_cat)
        kwargs[key] = value

    for key in opt:
        if config.hasattr(key):
            value = Generate(eval("config." + key),input_cat)
            kwargs[key] = value

    # Make sure one and only one size is present
    found = False
    for key in size_opt:
        if config.hasattr(key):
            value = Generate(eval("config." + key),input_cat)
            if (found):
                raise AttributeError("Too many sizes for %s"%config.type)
            kwargs[key] = value
            found = True
    if not found:
        raise AttributeError("No size specified for %s"%config.type)

    # Now ready to call the constructor
    return eval("galsim."+config.type+"(**kwargs)")

def BuildGaussian(config, input_cat):
    return galsim.BuildSimple(config, input_cat, [], ['sigma','fwhm','half_light_radius'], [])

def BuildMoffat(config, input_cat):
    return galsim.BuildSimple(config, input_cat, 
        ['beta'], ['fwhm','scale_radius','half_light_radius'], ['trunc'])

def BuildSersic(config, input_cat):
    return galsim.BuildSimple(config, input_cat, ['n'], ['half_light_radius'], [])

def BuildExponential(config, input_cat):
    return galsim.BuildSimple(config, input_cat, [], ['half_light_radius','scale_radius'], [])

def BuildDeVaucouleurs(config, input_cat):
    return galsim.BuildSimple(config, input_cat, [], ['half_light_radius'], [])

def BuildAiry(config, input_cat):
    return galsim.BuildSimple(config, input_cat, [], ['D'], ['obs'])

def BuildPixel(config, input_cat):

    for key in ['xw','yw']:
        if not config.hasattr(key):
            raise AttributeError('Pixel requires attribute %s'%key)
    kwargs = {}
    kwargs['xw'] = Generate(config.xw,input_cat)
    kwargs['yw'] = Generate(config.yw,input_cat)

    if (xw != yw):
        raise Warning("xw != yw found (%f != %f) "%(xw,yw) +
            "This is supported for the pixel, but not the draw routines. " +
            "There might be weirdness....")
    return galsim.Pixel(xw=xw,yw=yw)

    if config.hasattr('flux'):
        kwargs['flux'] = Generate(config.flux,input_cat)

    return galsim.Pixel(**kwargs)

def BuildSquarePixel(config, input_cat):

    if not config.hasattr('size'):
        raise AttributeError('SquarePixel requires attribute size')
    kwargs = {}
    kwargs['xw'] = Generate(config.size,input_cat)
    if config.hasattr('flux'):
        kwargs['flux'] = Generate(config.flux,input_cat)
    return galsim.Pixel(**kwargs)

def BuildSum(config, input_cat):

    if not config.hasattr('items'):
        raise AttributeError('Sum requires attribute items')
    list = []
    for item in config.items:
        list += [ BuildGSObject(item, input_cat) ]
    return galsim.Add(list)

def BuildConvolve(config, input_cat):

    if not config.hasattr('items'):
        raise AttributeError('Convolve requires attribute items')
    list = []
    for item in config.items:
        list += [ BuildGSObject(item, input_cat) ]
    return galsim.Convolve(list)

def Generate(config, input_cat):
    try:
        if config.hasattr('type'):
            return eval('galsim.GenerateFrom' + config.type + '(config, input_cat)')
    except AttributeError:
        pass
    # else assume config is really a value.
    return config

def GenerateFromInputCatalog(config, input_cat):
    if input_cat is None:
        raise ValueError("Use of InputCatalog requested, but no input_cat given")

    if not config.hasattr('col'):
        raise AttributeError("No col specified for InputCatalog")
    col = config.col

    # input_cat stores the current row to use.
    current = input_cat.current

    if input_cat.type is 'ASCII':

        try:
            # config values are 1-based, but we access is 0-based, so use col-1
            value = input_cat.data[current,col-1]
        except IndexError:
            raise IndexError("col attribute or input_cat.current out of bounds "+
                    " for accessing input_cat.data [col, object_id] = [%s,%s]"%(col,current))

    elif input_cat.type is 'FITS':
        raise NotImplementedError("FITS catalog inputs not yet implemented, sorry!")

    else:
        raise NotImplementedError("Unknown catalog type %s"%input_cat.type)

    return value
