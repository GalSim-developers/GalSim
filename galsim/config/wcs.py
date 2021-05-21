# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

from .util import LoggerWrapper, GetIndex, GetRNG, get_cls_params
from .value import ParseValue, GetAllParams, CheckAllParams, SetDefaultIndex
from .input import RegisterInputConnectedType
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..wcs import BaseWCS, PixelScale, OffsetWCS, JacobianWCS, AffineTransform
from ..wcs import ShearWCS, OffsetShearWCS, UVFunction, RaDecFunction
from ..fitswcs import TanWCS, FitsWCS
from ..angle import Angle, AngleUnit
from ..position import PositionD
from ..celestial import CelestialCoord
from ..utilities import basestring

# This file handles the construction of wcs types in config['image']['wcs'].

# This module-level dict will store all the registered wcs types.
# See the RegisterWCSType function at the end of this file.
# The keys are the (string) names of the wcs types, and the values will be builders that know
# how to build the WCS object.
valid_wcs_types = {}


def BuildWCS(config, key, base, logger=None):
    """Read the wcs parameters from config[key] and return a constructed wcs object.

    Parameters:
        config:     A dict with the configuration information. (usually base['image'])
        key:        The key name in config indicating which object to build.
        base:       The base dict of the configuration.
        logger:     Optionally, provide a logger for logging debug statements. [default: None]

    Returns:
        a BaseWCS instance
    """
    logger = LoggerWrapper(logger)
    logger.debug('image %d: Start BuildWCS key = %s',base.get('image_num',0),key)

    try:
        param = config[key]
    except KeyError:
        # Default if no wcs is to use PixelScale
        if 'pixel_scale' in config:
            scale = ParseValue(config, 'pixel_scale', base, float)[0]
        else:
            scale = 1.0
        return PixelScale(scale)

    # Check for direct value, else get the wcs type
    if isinstance(param, BaseWCS):
        return param
    elif isinstance(param, basestring) and (param[0] == '$' or param[0] == '@'):
        return ParseValue(config, key, base, None)[0]
    elif isinstance(param, dict):
        wcs_type = param.get('type', 'PixelScale')
    else:
        raise GalSimConfigError("wcs must be either a BaseWCS or a dict")

    # For these two, just do the usual ParseValue function.
    if wcs_type in ('Eval', 'Current'):
        return ParseValue(config, key, base, None)[0]

    # Check if we can use the current cached object
    orig_index_key = base.get('index_key','image_num')
    base['index_key'] = 'image_num'
    index, index_key = GetIndex(param, base)
    if 'current' in param:
        cwcs, csafe, cvalue_type, cindex, cindex_key = param['current']
        if cindex == index:
            logger.debug('image %d: The wcs object is already current', base.get('image_num',0))
            logger.debug('image %d: index_key = %s, index = %d',base.get('image_num',0),
                         cindex_key, cindex)
            return cwcs

    # Special case: origin == center means to use image_center for the wcs origin
    if 'origin' in param and param['origin'] == 'center':
        origin = base['image_center']
        param['origin'] = origin

    if wcs_type not in valid_wcs_types:
        raise GalSimConfigValueError("Invalid image.wcs.type.", wcs_type,
                                     list(valid_wcs_types.keys()))
    logger.debug('image %d: Building wcs type %s', base.get('image_num',0), wcs_type)
    builder = valid_wcs_types[wcs_type]
    wcs = builder.buildWCS(param, base, logger)
    logger.debug('image %d: wcs = %s', base.get('image_num',0), wcs)

    param['current'] = wcs, False, None, index, index_key
    base['index_key'] = orig_index_key

    return wcs


class WCSBuilder(object):
    """A base class for building WCS objects.

    The base class defines the call signatures of the methods that any derived class should follow.
    """
    def buildWCS(self, config, base, logger):
        """Build the WCS based on the specifications in the config dict.

        Note: Sub-classes must override this function with a real implementation.

        Parameters:
            config:     The configuration dict for the wcs type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed WCS object.
        """
        raise NotImplementedError("The %s class has not overridden buildWCS"%self.__class__)


class SimpleWCSBuilder(WCSBuilder):
    """A class for building simple WCS objects.

    The initializer takes an init_func, which is the class or function to call to build the WCS.
    For the kwargs, it calls getKwargs, which does the normal parsing of the req_params and
    related class attributes.
    """
    def __init__(self, init_func):
        self.init_func = init_func

    def getKwargs(self, build_func, config, base):
        """Get the kwargs to pass to the build function based on the following attributes of
        build_func:

        _req_params
                        A dict of required parameters and their types.
        _opt_params
                        A dict of optional parameters and their types.
        _single_params
                        A list of dicts of parameters such that one and only one of
                        parameter in each dict is required.
        _takes_rng
                        A bool value saying whether an rng object is required.
                        (Which would be weird for this, but it's part of our standard set.)

        See any of the classes in wcs.py for examples of classes that set these attributes.

        Parameters:
            build_func: The class or function from which to get the
            config:     The configuration dict for the wcs type.
            base:       The base configuration dict.

        Returns:
            kwargs
        """
        # Then use the standard trick of reading the required and optional parameters
        # from the class or function attributes.
        req, opt, single, takes_rng = get_cls_params(build_func)

        # Pull in the image layer pixel_scale as a scale item if necessary.
        if ( ('scale' in req or 'scale' in opt) and 'scale' not in config and
            'pixel_scale' in base['image'] ):
            config['scale'] = base['image']['pixel_scale']

        kwargs, safe = GetAllParams(config, base, req, opt, single)

        # This would be weird, but might as well check...
        if takes_rng: # pragma: no cover
            kwargs['rng'] = GetRNG(config, base)
        return kwargs

    def buildWCS(self, config, base, logger):
        """Build the WCS based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the wcs type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed WCS object.
        """
        kwargs = self.getKwargs(self.init_func,config,base)
        return self.init_func(**kwargs)


class OriginWCSBuilder(SimpleWCSBuilder):
    """A specialization for WCS classes that use a different type depending on whether there
    is an origin or world_origin parameter in the config dict.
    """
    def __init__(self, init_func, origin_init_func):
        self.init_func = init_func
        self.origin_init_func = origin_init_func

    def buildWCS(self, config, base, logger):
        """Build the WCS based on the specifications in the config dict, using the appropriate
        type depending on whether an origin is provided.

        Parameters:
            config:     The configuration dict for the wcs type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed WCS object.
        """
        if 'origin' in config or 'world_origin' in config:
            build_func = self.origin_init_func
        else:
            build_func = self.init_func
        kwargs = self.getKwargs(build_func,config,base)
        return build_func(**kwargs)


class TanWCSBuilder(WCSBuilder):
    """The TanWCS type needs special handling to get the kwargs, since the TanWCS function
    takes an AffineTransform as one of the arguments, so we need to build that from
    dudx, dudy, etc.  We also need to construct a CelestialCoord object for the world_origin,
    which we make from ra, dec paramters.
    """
    def __init__(self): pass

    def buildWCS(self, config, base, logger):
        """Build the TanWCS based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the wcs type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed WCS object.
        """
        req = { "dudx" : float, "dudy" : float, "dvdx" : float, "dvdy" : float,
                "ra" : Angle, "dec" : Angle }
        opt = { "units" : str, "origin" : PositionD }
        params, safe = GetAllParams(config, base, req=req, opt=opt)

        dudx = params['dudx']
        dudy = params['dudy']
        dvdx = params['dvdx']
        dvdy = params['dvdy']
        ra = params['ra']
        dec = params['dec']
        units = params.get('units', 'arcsec')
        origin = params.get('origin', None)

        affine = AffineTransform(dudx, dudy, dvdx, dvdy, origin)
        world_origin = CelestialCoord(ra, dec)
        units = AngleUnit.from_name(units)

        return TanWCS(affine=affine, world_origin=world_origin, units=units)

class ListWCSBuilder(WCSBuilder):
    """Select a wcs from a list
    """
    def buildWCS(self, config, base, logger):
        req = { 'items' : list }
        opt = { 'index' : int }
        # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
        CheckAllParams(config, req=req, opt=opt)
        items = config['items']
        if not isinstance(items,list):
            raise GalSimConfigError("items entry for type=List is not a list.")

        # Setup the indexing sequence if it hasn't been specified using the length of items.
        SetDefaultIndex(config, len(items))
        index, safe = ParseValue(config, 'index', base, int)

        if index < 0 or index >= len(items):
            raise GalSimConfigError("index %d out of bounds for wcs type=List"%index)
        return BuildWCS(items, index, base)

def RegisterWCSType(wcs_type, builder, input_type=None):
    """Register a wcs type for use by the config apparatus.

    Parameters:
        wcs_type:       The name of the type in config['image']['wcs']
        builder:        A builder object to use for building the WCS object.  It should
                        be an instance of a subclass of WCSBuilder.
        input_type:     If the WCS builder utilises an input object, give the key name of the
                        input type here.  (If it uses more than one, this may be a list.)
                        [default: None]
    """
    valid_wcs_types[wcs_type] = builder
    RegisterInputConnectedType(input_type, wcs_type)


RegisterWCSType('PixelScale', OriginWCSBuilder(PixelScale, OffsetWCS))
RegisterWCSType('Shear', OriginWCSBuilder(ShearWCS, OffsetShearWCS))
RegisterWCSType('Jacobian', OriginWCSBuilder(JacobianWCS, AffineTransform))
RegisterWCSType('Affine', OriginWCSBuilder(JacobianWCS, AffineTransform))
RegisterWCSType('UVFunction', SimpleWCSBuilder(UVFunction))
RegisterWCSType('RaDecFunction', SimpleWCSBuilder(RaDecFunction))
RegisterWCSType('Fits', SimpleWCSBuilder(FitsWCS))
RegisterWCSType('Tan', TanWCSBuilder())
RegisterWCSType('List', ListWCSBuilder())
