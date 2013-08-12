# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
import galsim

valid_gsobject_types = {
    # Note: these are just the types that need a special builder.  Most of GSObject sub-classes
    # in base.py (and some elsewhere) can use the default builder, called _BuildSimple, which
    # just uses the req, opt, and single class variables.
    # See the des module for examples of how to extend this from a module.
    'None' : '_BuildNone',
    'Add' : '_BuildAdd',
    'Sum' : '_BuildAdd',
    'Convolve' : '_BuildConvolve',
    'Convolution' : '_BuildConvolve',
    'List' : '_BuildList',
    'Ring' : '_BuildRing',
    'Pixel' : '_BuildPixel',
    'RealGalaxy' : '_BuildRealGalaxy',
}

class SkipThisObject(Exception):
    """
    A class that a builder can throw to indicate that nothing went wrong, but for some
    reason, this particular object should be skipped and just move onto the next object.
    The constructor takes an optional message that will be output to the logger if 
    logging is active.
    """
    def __init__(self, message=None):
        # Using self.message gives a deprecation warning.  Avoid this by using a different name.
        self.msg = message


def BuildGSObject(config, key, base=None, gsparams={}):
    """Build a GSObject using config dict for key=key.

    @param config     A dict with the configuration information.
    @param key        The key name in config indicating which object to build.
    @param base       A dict which stores potentially useful things like
                      base['rng'] = random number generator
                      base['catalog'] = input catalog for InputCat items
                      base['real_catalog'] = real galaxy catalog for RealGalaxy objects
                      Typically on the initial call to BuildGSObject, this will be 
                      the same as config, hence the name base.
    @param gsparams   Optionally, provide non-default gsparams items.  Any gsparams specified
                      at this level will be added to the list.  This should be a dict with
                      whatever kwargs should be used in constructing the GSParams object.

    @returns gsobject, safe 
        gsobject is the built object 
        safe is a bool that says whether it is safe to use this object again next time
    """
    # I'd like to be able to have base=config be the default value, but python doesn't
    # allow that.  So None is the default, and if it's None, we set it to config.
    if not base:
        base = config
 
    #print 'Start BuildGSObject: config = ',config
    #print 'gsparams = ',gsparams
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
        #print 'current is safe:  ',ck['current_val'], True
        return ck['current_val'], True

    # Ring is only allowed for top level gal (since it requires special handling in 
    # multiprocessing, and that's the only place we look for it currently).
    if type == 'Ring' and key != 'gal':
        raise AttributeError("Ring type only allowed for top level gal")

    # Check if we need to skip this object
    if 'skip' in ck:
        skip = galsim.config.ParseValue(ck, 'skip', base, bool)[0]
        if skip: 
            raise SkipThisObject('config.skip = True')

    # Set up the initial default list of attributes to ignore while building the object:
    ignore = [ 
        'dilate', 'dilation', 'ellip', 'rotate', 'rotation',
        'magnify', 'magnification', 'shear', 'shift', 
        'gsparams', 'skip', 'current_val', 'safe' 
    ]
    # There are a few more that are specific to which key we have.
    if key == 'gal':
        ignore += [ 'resolution', 'signal_to_noise', 'redshift', 're_from_res' ]
        # If redshift is present, parse it here, since it might be needed by the Build functions.
        # All we actually care about is setting the current_val, so don't assign to anything.
        if 'redshift' in ck:
            galsim.config.ParseValue(ck, 'redshift', base, float)
    elif key == 'psf':
        ignore += [ 'saved_re' ]
    elif key != 'pix':
        # As long as key isn't psf or pix, allow resolution.
        # Ideally, we'd like to check that it's something within the gal hierarchy, but
        # I don't know an easy way to do that.
        ignore += [ 'resolution' , 're_from_res' ]

    # If we are specifying the size according to a resolution, then we 
    # need to get the PSF's half_light_radius.
    if 'resolution' in ck:
        if 'psf' not in base:
            raise AttributeError(
                "Cannot use gal.resolution if no psf is set.")
        if 'saved_re' not in base['psf']:
            raise AttributeError(
                'Cannot use gal.resolution with psf.type = %s'%base['psf']['type'])
        psf_re = base['psf']['saved_re']
        resolution = galsim.config.ParseValue(ck, 'resolution', base, float)[0]
        gal_re = resolution * psf_re
        if 're_from_res' not in ck:
            # The first time, check that half_light_radius isn't also specified.
            if 'half_light_radius' in ck:
                raise AttributeError(
                    'Cannot specify both gal.resolution and gal.half_light_radius')
            ck['re_from_res'] = True
        ck['half_light_radius'] = gal_re

    # Make sure the PSF gets flux=1 unless explicitly overridden by the user.
    if key == 'psf' and 'flux' not in ck:
        ck['flux'] = 1

    if 'gsparams' in ck:
        gsparams = UpdateGSParams(gsparams, ck['gsparams'], 'gsparams', config)

    # See if this type has a specialized build function:
    if type in valid_gsobject_types:
        build_func = eval(valid_gsobject_types[type])
        gsobject, safe = build_func(ck, key, base, ignore, gsparams)
    # Next, we check if this name is in the galsim dictionary.
    elif type in galsim.__dict__:
        if issubclass(galsim.__dict__[type], galsim.GSObject):
            gsobject, safe = _BuildSimple(ck, key, base, ignore, gsparams)
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


def UpdateGSParams(gsparams, config, key, base):
    """@brief Add additional items to the gsparams dict based on config['gsparams']
    """
    opt = galsim.GSObject._gsparams
    kwargs, safe = galsim.config.GetAllParams(config, key, base, opt=opt)
    # When we update gsparams, we don't want to corrupt the original, so we need to
    # make a copy first, then update with kwargs.
    ret = {}
    ret.update(gsparams)
    ret.update(kwargs)
    return ret


def _BuildNone(config, key, base, ignore, gsparams):
    """@brief Special type=None returns None
    """
    return None, True


def _BuildAdd(config, key, base, ignore, gsparams):
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
        gsobject, safe1 = BuildGSObject(items, i, base, gsparams)
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
    if gsparams: gsparams = galsim.GSParams(**gsparams)
    else: gsparams = None
    gsobject = galsim.Add(gsobjects,gsparams=gsparams)

    if 'flux' in config:
        flux, safe1 = galsim.config.ParseValue(config, 'flux', base, float)
        #print 'flux = ',flux
        gsobject.setFlux(flux)
        safe = safe and safe1

    return gsobject, safe

def _BuildConvolve(config, key, base, ignore, gsparams):
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
        gsobject, safe1 = BuildGSObject(items, i, base, gsparams)
        safe = safe and safe1
        gsobjects.append(gsobject)
    #print 'After built component items for ',type,' safe = ',safe

    if gsparams: gsparams = galsim.GSParams(**gsparams)
    else: gsparams = None
    gsobject = galsim.Convolve(gsobjects,gsparams=gsparams)

    if 'flux' in config:
        flux, safe1 = galsim.config.ParseValue(config, 'flux', base, float)
        #print 'flux = ',flux
        gsobject.setFlux(flux)
        safe = safe and safe1

    return gsobject, safe

def _BuildList(config, key, base, ignore, gsparams):
    """@brief  Build a GSObject selected from a List
    """
    req = { 'items' : list }
    opt = { 'index' : float , 'flux' : float }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    galsim.config.CheckAllParams(config, key, req=req, opt=opt, ignore=ignore)

    items = config['items']
    if not isinstance(items,list):
        raise AttributeError("items entry for config.%s entry is not a list."%type)

    # Setup the indexing sequence if it hasn't been specified using the length of items.
    galsim.config.SetDefaultIndex(config, len(items))
    index, safe = galsim.config.ParseValue(config, 'index', base, int)
    if index < 0 or index >= len(items):
        raise AttributeError("index %d out of bounds for config.%s"%(index,type))
    #print items[index]['type']
    #print 'index = ',index,' From ',key,' List: ',items[index]

    gsobject, safe1 = BuildGSObject(items, index, base, gsparams)
    safe = safe and safe1

    if 'flux' in config:
        flux, safe1 = galsim.config.ParseValue(config, 'flux', base, float)
        #print 'flux = ',flux
        gsobject.setFlux(flux)
        safe = safe and safe1

    return gsobject, safe

def _BuildRing(config, key, base, ignore, gsparams):
    """@brief  Build a GSObject in a Ring
    """
    req = { 'num' : int, 'first' : dict }
    opt = { 'full_rotation' : galsim.Angle }
    # Only Check, not Get.  We need to handle first a bit differently, since it's a gsobject.
    galsim.config.CheckAllParams(config, key, req=req, opt=opt, ignore=ignore)

    num = galsim.config.ParseValue(config, 'num', base, int)[0]
    if num <= 0:
        raise ValueError("Attribute num for gal.type == Ring must be > 0")

    if 'full_rotation' in config:
        full_rotation = galsim.config.ParseValue(config, 'full_rotation', base, galsim.Angle)[0]
    else:
        import math
        full_rotation = math.pi * galsim.radians

    dtheta = full_rotation / num
    #print 'dtheta = ',dtheta

    k = base['seq_index']
    #print 'k = ',k
    if k % num == 0:
        #print 'first pass -- rebuilding'
        # Then this is the first in the Ring.  
        gsobject = BuildGSObject(config, 'first', base, gsparams)[0]
    else:
        #print 'not first pass rotate by ',dtheta
        if not isinstance(config['first'],dict) or 'current_val' not in config['first']:
            raise RuntimeError("Building Ring after the first item, but no current_val stored.")
        gsobject = config['first']['current_val'].createRotated(k*dtheta)

    return gsobject, False


def _BuildPixel(config, key, base, ignore, gsparams):
    """@brief Build a Pixel type GSObject from user input.
    """
    kwargs, safe = galsim.config.GetAllParams(config, key, base, 
        req = galsim.__dict__['Pixel']._req_params,
        opt = galsim.__dict__['Pixel']._opt_params,
        single = galsim.__dict__['Pixel']._single_params,
        ignore = ignore)
    if gsparams: kwargs['gsparams'] = galsim.GSParams(**gsparams)

    if 'yw' in kwargs.keys() and (kwargs['xw'] != kwargs['yw']):
        import warnings
        warnings.warn(
            "xw != yw found (%f != %f) "%(kwargs['xw'], kwargs['yw']) +
            "This is supported for the pixel, but not the draw routines. " +
            "There might be weirdness....")

    return galsim.Pixel(**kwargs), safe


def _BuildRealGalaxy(config, key, base, ignore, gsparams):
    """@brief Build a RealGalaxy type GSObject from user input.
    """
    if 'real_catalog' not in base:
        raise ValueError("No real galaxy catalog available for building type = RealGalaxy")
    real_cat = base['real_catalog']

    # Special: if index is Sequence or Random, and max isn't set, set it to real_cat.nobjects-1
    if 'id' not in config:
        galsim.config.SetDefaultIndex(config, real_cat.nobjects)

    kwargs, safe = galsim.config.GetAllParams(config, key, base, 
        req = galsim.__dict__['RealGalaxy']._req_params,
        opt = galsim.__dict__['RealGalaxy']._opt_params,
        single = galsim.__dict__['RealGalaxy']._single_params,
        ignore = ignore)
    if gsparams: kwargs['gsparams'] = galsim.GSParams(**gsparams)

    if 'rng' not in base:
        raise ValueError("No base['rng'] available for %s.type = RealGalaxy"%(key))
    kwargs['rng'] = base['rng']

    if 'index' in kwargs:
        index = kwargs['index']
        if index >= real_cat.nobjects:
            raise IndexError(
                "%s index has gone past the number of entries in the catalog"%param_name)

    return galsim.RealGalaxy(real_cat, **kwargs), safe


def _BuildSimple(config, key, base, ignore, gsparams={}):
    """@brief Build a simple GSObject (i.e. one without a specialized _Build function) or
    any other galsim object that defines _req_params, _opt_params and _single_params.
    """
    # Build the kwargs according to the various params objects in the class definition.
    type = config['type']
    if type in galsim.__dict__:
        init_func = eval("galsim."+type)
    else:
        init_func = eval(type)

    kwargs, safe = galsim.config.GetAllParams(config, key, base, 
                                              req = init_func._req_params,
                                              opt = init_func._opt_params,
                                              single = init_func._single_params,
                                              ignore = ignore)
    if gsparams: kwargs['gsparams'] = galsim.GSParams(**gsparams)

    if init_func._takes_rng:
        if 'rng' not in base:
            raise ValueError("No base['rng'] available for %s.type = %s"%(key,type))
        kwargs['rng'] = base['rng']
        safe = False

    # Finally, after pulling together all the params, try making the GSObject.
    return init_func(**kwargs), safe


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
    if 'shear' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _EllipObject(gsobject, config, 'shear', base)
        safe = safe and safe1
    if 'shift' in config:
        if orig: gsobject = gsobject.copy(); orig = False
        gsobject, safe1 = _ShiftObject(gsobject, config, 'shift', base)
        safe = safe and safe1
    return gsobject, safe

def _EllipObject(gsobject, config, key, base):
    """@brief Applies ellipticity to a supplied GSObject from user input, also used for
    gravitational shearing.

    @returns transformed GSObject.
    """
    shear, safe = galsim.config.ParseValue(config, key, base, galsim.Shear)
    gsobject = gsobject.createSheared(shear)
    return gsobject, safe

def _RotateObject(gsobject, config, key, base):
    """@brief Applies rotation to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    theta, safe = galsim.config.ParseValue(config, key, base, galsim.Angle)
    gsobject = gsobject.createRotated(theta)
    return gsobject, safe

def _DilateObject(gsobject, config, key, base):
    """@brief Applies dilation to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    scale, safe = galsim.config.ParseValue(config, key, base, float)
    gsobject = gsobject.createDilated(scale)
    return gsobject, safe

def _MagnifyObject(gsobject, config, key, base):
    """@brief Applies magnification to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    mu, safe = galsim.config.ParseValue(config, key, base, float)
    gsobject = gsobject.createMagnified(mu)
    return gsobject, safe

def _ShiftObject(gsobject, config, key, base):
    """@brief Applies centroid shift to a supplied GSObject based on user input.

    @returns transformed GSObject.
    """
    shift, safe = galsim.config.ParseValue(config, key, base, galsim.PositionD)
    gsobject = gsobject.createShifted(shift.x,shift.y)
    return gsobject, safe

