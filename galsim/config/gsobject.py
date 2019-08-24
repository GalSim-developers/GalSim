# Copyright (c) 2012-2019 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
import logging
import inspect

from .util import LoggerWrapper, GetIndex, GetRNG
from .value import ParseValue, GetCurrentValue, GetAllParams, CheckAllParams, SetDefaultIndex
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..position import PositionD
from ..sum import Add
from ..convolve import Convolve
from ..phase_psf import OpticalPSF
from ..shear import Shear
from ..angle import Angle
from ..gsobject import GSObject
from ..gsparams import GSParams

# This file handles the building of GSObjects in the config['psf'] and config['gal'] fields.
# This file includes many of the simple object types.  Additional types are defined in
# gsobject_ring.py, input_real.py, and input_cosmos.py.

# This module-level dict will store all the registered gsobject types.
# See the RegisterObjectType function at the end of this file.
# The keys will be the (string) names of the object types, and the values are the function
# to call to build an object of that type.
valid_gsobject_types = {}

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


def BuildGSObject(config, key, base=None, gsparams={}, logger=None):
    """Build a GSObject from the parameters in config[key].

    Parameters:
        config:     A dict with the configuration information.
        key:        The key name in config indicating which object to build.
        base:       The base dict of the configuration. [default: config]
        gsparams:   Optionally, provide non-default GSParams items.  Any ``gsparams`` specified
                    at this level will be added to the list.  This should be a dict with
                    whatever kwargs should be used in constructing the GSParams object.
                    [default: {}]
        logger:     Optionally, provide a logger for logging debug statements.
                    [default: None]

    Returns:
        the tuple (gsobject, safe), where ``gsobject`` is the built object, and ``safe`` is
        a bool that says whether it is safe to use this object again next time.
    """
    from .. import __dict__ as galsim_dict

    logger = LoggerWrapper(logger)
    if base is None:
        base = config

    logger.debug('obj %d: Start BuildGSObject %s',base.get('obj_num',0),key)

    # If key isn't in config, then just return None.
    try:
        param = config[key]
    except KeyError:
        return None, True

    # Check what index key we want to use for this object.
    # Note: this call will also set base['index_key'] and base['rng'] to the right values
    index, index_key = GetIndex(param, base)

    # Get the type to be parsed.
    if not 'type' in param:
        raise GalSimConfigError("type attribute required in config.%s"%key)
    type_name = param['type']

    # If we are repeating, then we get to use the current object for repeat times.
    if 'repeat' in param:
        repeat = ParseValue(param, 'repeat', base, int)[0]
    else:
        repeat = 1

    # Check if we need to skip this object
    if 'skip' in param:
        skip = ParseValue(param, 'skip', base, bool)[0]
        if skip:
            logger.debug('obj %d: Skipping because field skip=True',base.get('obj_num',0))
            raise SkipThisObject()

    # Check if we can use the current cached object
    if 'current' in param:
        # NB. "current" tuple is (obj, safe, None, index, index_type)
        cobj, csafe, cvalue_type, cindex, cindex_type = param['current']
        if csafe or cindex//repeat == index//repeat:
            # If logging, explain why we are using the current object.
            if logger:
                if csafe:
                    logger.debug('obj %d: current is safe',base.get('obj_num',0))
                elif repeat > 1:
                    logger.debug('obj %d: repeat = %d, index = %d, use current object',
                                base.get('obj_num',0),repeat,index)
                else:
                    logger.debug('obj %d: This object is already current', base.get('obj_num',0))

            return cobj, csafe

    # Set up the initial default list of attributes to ignore while building the object:
    ignore = [
        'dilate', 'dilation', 'ellip', 'rotate', 'rotation', 'scale_flux',
        'magnify', 'magnification', 'shear', 'shift',
        'gsparams', 'skip',
        'current', 'index_key', 'repeat'
    ]
    # There are a few more that are specific to which key we have.
    if key == 'gal':
        ignore += [ 'resolution', 'signal_to_noise', 'redshift', 're_from_res' ]
    elif key == 'psf':
        ignore += [ 'saved_re' ]
    else:
        # As long as key isn't psf, allow resolution.
        # Ideally, we'd like to check that it's something within the gal hierarchy, but
        # I don't know an easy way to do that.
        ignore += [ 'resolution' , 're_from_res' ]

    # Allow signal_to_noise for PSFs only if there is not also a galaxy.
    if 'gal' not in base and key == 'psf':
        ignore += [ 'signal_to_noise']

    # If we are specifying the size according to a resolution, then we
    # need to get the PSF's half_light_radius.
    if 'resolution' in param:
        if 'psf' not in base:
            raise GalSimConfigError("Cannot use gal.resolution if no psf is set.")
        if 'saved_re' not in base['psf']:
            raise GalSimConfigError(
                'Cannot use gal.resolution with psf.type = %s'%base['psf']['type'])
        psf_re = base['psf']['saved_re']
        resolution = ParseValue(param, 'resolution', base, float)[0]
        gal_re = resolution * psf_re
        if 're_from_res' not in param:
            # The first time, check that half_light_radius isn't also specified.
            if 'half_light_radius' in param:
                raise GalSimConfigError(
                    'Cannot specify both gal.resolution and gal.half_light_radius')
            param['re_from_res'] = True
        param['half_light_radius'] = gal_re

    # Make sure the PSF gets flux=1 unless explicitly overridden by the user.
    if key == 'psf' and 'flux' not in param and 'signal_to_noise' not in param:
        param['flux'] = 1

    if 'gsparams' in param:
        gsparams = UpdateGSParams(gsparams, param['gsparams'], base)

    # See if this type is registered as a valid type.
    if type_name in valid_gsobject_types:
        build_func = valid_gsobject_types[type_name]
    elif type_name in galsim_dict:
        from future.utils import exec_
        gdict = globals().copy()
        exec_('import galsim', gdict)
        build_func = eval("galsim."+type_name, gdict)
    else:
        raise GalSimConfigValueError("Unrecognised gsobject type", type_name)

    if inspect.isclass(build_func) and issubclass(build_func, GSObject):
        gsobject, safe = _BuildSimple(build_func, param, base, ignore, gsparams, logger)
    else:
        gsobject, safe = build_func(param, base, ignore, gsparams, logger)

    # If this is a psf, try to save the half_light_radius in case gal uses resolution.
    if key == 'psf':
        try:
            param['saved_re'] = gsobject.half_light_radius
        except (AttributeError, NotImplementedError, TypeError):
            pass

    # Apply any dilation, ellip, shear, etc. modifications.
    gsobject, safe1 = TransformObject(gsobject, param, base, logger)
    safe = safe and safe1

    param['current'] = gsobject, safe, None, index, index_key

    return gsobject, safe


def UpdateGSParams(gsparams, config, base):
    """Add additional items to the ``gsparams`` dict based on config['gsparams'].

    Parameters:
        gsparams:   A dict with whatever kwargs should be used in constructing the GSParams object.
        config:     A dict with the configuration information.
        base:       The base dict of the configuration.

    Returns:
        an updated gsparams dict
    """
    opt = GSObject._gsparams_opt
    kwargs, safe = GetAllParams(config, base, opt=opt)
    # When we update gsparams, we don't want to corrupt the original, so we need to
    # make a copy first, then update with kwargs.
    ret = {}
    ret.update(gsparams)
    ret.update(kwargs)
    return ret


#
# The following are private functions to implement the simpler GSObject types.
# These are not imported into galsim.config namespace.
#

def _BuildSimple(build_func, config, base, ignore, gsparams, logger):
    """Build a simple GSObject (i.e. one without a specialized _Build function) or
    any other GalSim object that defines _req_params, _opt_params and _single_params.
    """
    # Build the kwargs according to the various params objects in the class definition.
    type_name = config['type']
    logger.debug('obj %d: BuildSimple for type = %s',base.get('obj_num',0),type_name)

    req = build_func._req_params if hasattr(build_func,'_req_params') else {}
    opt = build_func._opt_params if hasattr(build_func,'_opt_params') else {}
    single = build_func._single_params if hasattr(build_func,'_single_params') else []
    kwargs, safe = GetAllParams(config, base, req=req, opt=opt, single=single, ignore=ignore)
    if gsparams: kwargs['gsparams'] = GSParams(**gsparams)

    if build_func._takes_rng:
        kwargs['rng'] = GetRNG(config, base, logger, type_name)
        safe = False

    logger.debug('obj %d: kwargs = %s',base.get('obj_num',0),kwargs)

    # Finally, after pulling together all the params, try making the GSObject.
    return build_func(**kwargs), safe


def _BuildNone(config, base, ignore, gsparams, logger):
    """Special type=None returns None.
    """
    return None, True


def _BuildAdd(config, base, ignore, gsparams, logger):
    """Build a Sum object.
    """
    req = { 'items' : list }
    opt = { 'flux' : float }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    CheckAllParams(config, req=req, opt=opt, ignore=ignore)

    gsobjects = []
    items = config['items']
    if not isinstance(items,list):
        raise GalSimConfigError("items entry for type=Add is not a list.")
    safe = True

    for i in range(len(items)):
        gsobject, safe1 = BuildGSObject(items, i, base, gsparams, logger)
        # Skip items with flux=0
        if 'flux' in items[i] and GetCurrentValue('flux',items[i],float,base) == 0.:
            logger.debug('obj %d: Not including component with flux == 0',base.get('obj_num',0))
            continue
        safe = safe and safe1
        gsobjects.append(gsobject)

    if len(gsobjects) == 0:
        raise GalSimConfigError("No valid items for type=Add")
    elif len(gsobjects) == 1:
        gsobject = gsobjects[0]
    else:
        # Special: if the last item in a Sum doesn't specify a flux, we scale it
        # to bring the total flux up to 1.
        if ('flux' not in items[-1]) and all('flux' in item for item in items[0:-1]):
            sum_flux = 0
            for item in items[0:-1]:
                sum_flux += GetCurrentValue('flux',item,float,base)
            f = 1. - sum_flux
            if (f < 0):
                logger.warning(
                    "Warning: Automatic flux for the last item in Sum (to make the total flux=1) "
                    "resulted in negative flux = %f for that item"%f)
            logger.debug('obj %d: Rescaling final object in sum to have flux = %f',
                         base.get('obj_num',0), f)
            gsobjects[-1] = gsobjects[-1].withFlux(f)
        if gsparams: gsparams = GSParams(**gsparams)
        else: gsparams = None
        gsobject = Add(gsobjects,gsparams=gsparams)

    if 'flux' in config:
        flux, safe1 = ParseValue(config, 'flux', base, float)
        logger.debug('obj %d: flux == %f',base.get('obj_num',0),flux)
        gsobject = gsobject.withFlux(flux)
        safe = safe and safe1

    return gsobject, safe

def _BuildConvolve(config, base, ignore, gsparams, logger):
    """Build a Convolution object.
    """
    req = { 'items' : list }
    opt = { 'flux' : float }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    CheckAllParams(config, req=req, opt=opt, ignore=ignore)

    gsobjects = []
    items = config['items']
    if not isinstance(items,list):
        raise GalSimConfigError("items entry for type=Convolve is not a list.")
    safe = True
    for i in range(len(items)):
        gsobject, safe1 = BuildGSObject(items, i, base, gsparams, logger)
        safe = safe and safe1
        gsobjects.append(gsobject)

    if len(gsobjects) == 0:
        raise GalSimConfigError("No valid items for type=Convolve")
    elif len(gsobjects) == 1:
        gsobject = gsobjects[0]
    else:
        if gsparams: gsparams = GSParams(**gsparams)
        else: gsparams = None
        gsobject = Convolve(gsobjects,gsparams=gsparams)

    if 'flux' in config:
        flux, safe1 = ParseValue(config, 'flux', base, float)
        logger.debug('obj %d: flux == %f',base.get('obj_num',0),flux)
        gsobject = gsobject.withFlux(flux)
        safe = safe and safe1

    return gsobject, safe

def _BuildList(config, base, ignore, gsparams, logger):
    """Build a GSObject selected from a List.
    """
    req = { 'items' : list }
    opt = { 'index' : float , 'flux' : float }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    CheckAllParams(config, req=req, opt=opt, ignore=ignore)

    items = config['items']
    if not isinstance(items,list):
        raise GalSimConfigError("items entry for type=List is not a list.")

    # Setup the indexing sequence if it hasn't been specified using the length of items.
    SetDefaultIndex(config, len(items))
    index, safe = ParseValue(config, 'index', base, int)
    if index < 0 or index >= len(items):
        raise GalSimConfigError("index %d out of bounds for List"%index)

    gsobject, safe1 = BuildGSObject(items, index, base, gsparams, logger)
    safe = safe and safe1

    if 'flux' in config:
        flux, safe1 = ParseValue(config, 'flux', base, float)
        logger.debug('obj %d: flux == %f',base.get('obj_num',0),flux)
        gsobject = gsobject.withFlux(flux)
        safe = safe and safe1

    return gsobject, safe

def _BuildOpticalPSF(config, base, ignore, gsparams, logger):
    """Build an OpticalPSF.
    """
    kwargs, safe = GetAllParams(config, base,
        req = OpticalPSF._req_params,
        opt = OpticalPSF._opt_params,
        single = OpticalPSF._single_params,
        ignore = [ 'aberrations' ] + ignore)
    if gsparams: kwargs['gsparams'] = GSParams(**gsparams)

    if 'aberrations' in config:
        aber_list = [0.0] * 4  # Initial 4 values are ignored.
        aberrations = config['aberrations']
        if not isinstance(aberrations,list):
            raise GalSimConfigError(
                "aberrations entry for config.OpticalPSF entry is not a list.")
        for i in range(len(aberrations)):
            value, safe1 = ParseValue(aberrations, i, base, float)
            aber_list.append(value)
            safe = safe and safe1
        kwargs['aberrations'] = aber_list

    return OpticalPSF(**kwargs), safe


#
# Now the functions for performing transformations
#

def TransformObject(gsobject, config, base, logger):
    """Applies ellipticity, rotation, gravitational shearing and centroid shifting to a
    supplied GSObject, in that order.

    Parameters:
        gsobject:   The GSObject to be transformed.
        config:     A dict with the tranformation information for this object.
        base:       The base dict of the configuration.
        logger:     A logger for logging debug statements.

    Returns:
        transformed GSObject.
    """
    logger = LoggerWrapper(logger)
    # The transformations are applied in the following order:
    _transformation_list = [
        ('dilate', _Dilate),
        ('dilation', _Dilate),
        ('ellip', _Shear),
        ('rotate', _Rotate),
        ('rotation', _Rotate),
        ('scale_flux', _ScaleFlux),
        ('shear', _Shear),
        ('magnify', _Magnify),
        ('magnification', _Magnify),
        ('shift', _Shift),
    ]

    safe = True
    for key, func in _transformation_list:
        if key in config:
            gsobject, safe1 = func(gsobject, config, key, base, logger)
            safe = safe and safe1
    return gsobject, safe

def _Shear(gsobject, config, key, base, logger):
    shear, safe = ParseValue(config, key, base, Shear)
    logger.debug('obj %d: shear = %f,%f',base.get('obj_num',0),shear.g1,shear.g2)
    gsobject = gsobject.shear(shear)
    return gsobject, safe

def _Rotate(gsobject, config, key, base, logger):
    theta, safe = ParseValue(config, key, base, Angle)
    logger.debug('obj %d: theta = %f rad',base.get('obj_num',0),theta.rad)
    gsobject = gsobject.rotate(theta)
    return gsobject, safe

def _ScaleFlux(gsobject, config, key, base, logger):
    flux_ratio, safe = ParseValue(config, key, base, float)
    logger.debug('obj %d: flux_ratio  = %f',base.get('obj_num',0),flux_ratio)
    gsobject = gsobject * flux_ratio
    return gsobject, safe

def _Dilate(gsobject, config, key, base, logger):
    scale, safe = ParseValue(config, key, base, float)
    logger.debug('obj %d: scale  = %f',base.get('obj_num',0),scale)
    gsobject = gsobject.dilate(scale)
    return gsobject, safe

def _Magnify(gsobject, config, key, base, logger):
    mu, safe = ParseValue(config, key, base, float)
    logger.debug('obj %d: mu  = %f',base.get('obj_num',0),mu)
    gsobject = gsobject.magnify(mu)
    return gsobject, safe

def _Shift(gsobject, config, key, base, logger):
    shift, safe = ParseValue(config, key, base, PositionD)
    logger.debug('obj %d: shift  = %f,%f',base.get('obj_num',0),shift.x,shift.y)
    gsobject = gsobject.shift(shift.x,shift.y)
    return gsobject, safe

def RegisterObjectType(type_name, build_func, input_type=None):
    """Register an object type for use by the config apparatus.

    A few notes about the signature of the build functions:

    1. The config parameter is the dict for the current object to be generated.  So it should
       be the case that config['type'] == type_name.
    2. The base parameter is the original config dict being processed.
    3. The ignore parameter  is a list of items that should be ignored in the config dict if they
       are present and not valid for the object being built.
    4. The gsparams parameter is a dict of kwargs that should be used to build a GSParams object
       to use when building this object.
    5. The logger parameter is a logging.Logger object to use for logging progress if desired.
    6. The return value of build_func should be a tuple consisting of the object and a boolean,
       safe, which indicates whether the generated object is safe to use again rather than
       regenerate for subsequent postage stamps. e.g. if a PSF has all constant values, then it
       can be used for all the galaxies in a simulation, which lets it keep any FFTs that it has
       performed internally.  OpticalPSF is a good example of where this can have a significant
       speed up.

    Parameters:
        type_name:      The name of the 'type' specification in the config dict.
        build_func:     A function to build a GSObject from the config information.
                        The call signature is::

                            obj, safe = Build(config, base, ignore, gsparams, logger)

        input_type:     If the type requires an input object, give the key name of the input
                        type here.  (If it uses more than one, this may be a list.)
                        [default: None]
    """
    valid_gsobject_types[type_name] = build_func
    if input_type is not None:
        from .input import RegisterInputConnectedType
        RegisterInputConnectedType(input_type, type_name)

RegisterObjectType('None', _BuildNone)
RegisterObjectType('Add', _BuildAdd)
RegisterObjectType('Sum', _BuildAdd)
RegisterObjectType('Convolve', _BuildConvolve)
RegisterObjectType('Convolution', _BuildConvolve)
RegisterObjectType('List', _BuildList)
RegisterObjectType('OpticalPSF', _BuildOpticalPSF)

