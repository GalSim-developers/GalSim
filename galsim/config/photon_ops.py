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
from .sed import BuildSED
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..utilities import basestring
from ..photon_array import PhotonArray, PhotonOp
from ..photon_array import WavelengthSampler, FRatioAngles, PhotonDCR, Refraction, FocusDepth

# This file handles the construction of photon_ops in config['stamp']['photon_ops'].

# This module-level dict will store all the registered photon_op types.
# See the RegisterPhotonOpType function at the end of this file.
# The keys are the (string) names of the photon_ops types, and the values will be builders that
# know how to build the photon operator object.
valid_photon_op_types = {}


def BuildPhotonOps(config, key, base, logger=None):
    """Read the parameters from config[key] (which should be a list) and return a constructed
    photon_ops as a list.

    Parameters:
        config:     A dict with the configuration information for the photon ops.
                    (usually base['stamp'])
        key:        The key in the dict for the photon_ops list.
        base:       The base dict of the configuration.
        logger:     Optionally, provide a logger for logging debug statements. [default: None]

    Returns:
        the photon_ops list
    """
    logger = LoggerWrapper(logger)
    if not isinstance(config[key], list):
        raise GalSimConfigError("photon_ops must be a list")
    photon_ops = config[key]  # The list in the config dict
    ops = [] # List of the actual operators
    for i in range(len(photon_ops)):
        op = BuildPhotonOp(photon_ops, i, base, logger)
        ops.append(op)
    return ops

def BuildPhotonOp(config, key, base, logger=None):
    """Read the parameters from config[key] and return a single constructed photon_op object.

    Parameters:
        config:     A list with the configuration information for the photon ops.
                    (usually base['stamp']['photon_ops'])
        key:        The index in the list for this photon_op.  It's called key, since for most
                    things, this is a key into a dict, but here it's normally an integer index
                    into the photon_ops list.
        base:       The base dict of the configuration.
        logger:     Optionally, provide a logger for logging debug statements. [default: None]

    Returns:
        a object that would be valid in a photon_ops list
    """
    logger = LoggerWrapper(logger)
    logger.debug('obj %d: Start BuildPhotonOp key = %s',base.get('obj_num',0),key)

    param = config[key]

    # Check for direct value, else get the type
    if isinstance(param, PhotonOp):
        return param
    elif isinstance(param, basestring) and (param[0] == '$' or param[0] == '@'):
        return ParseValue(config, key, base, None)[0]
    elif isinstance(param, dict) and 'type' in param:
        op_type = param['type']
    else:
        raise GalSimConfigError("photon_op must be either a PhotonOp or a dict")

    # For these two, just do the usual ParseValue function.
    if op_type in ('Eval', 'Current'):
        return ParseValue(config, key, base, None)[0]

    if op_type not in valid_photon_op_types:
        raise GalSimConfigValueError("Invalid photon_op type.", op_type,
                                     list(valid_photon_op_types.keys()))

    # Check if we can use the current cached object
    index, index_key = GetIndex(param, base)
    if 'current' in param:
        cop, csafe, cvalue_type, cindex, cindex_key = param['current']
        if cindex == index:
            logger.debug('obj %d: The photon_op is already current', base.get('obj_num',0))
            logger.debug('obj %d: index_key = %s, index = %d',base.get('obj_num',0),
                         cindex_key, cindex)
            return cop

    # Need to use a builder.
    logger.debug('obj %d: Building photon_op type %s', base.get('obj_num',0), op_type)
    builder = valid_photon_op_types[op_type]
    op = builder.buildPhotonOp(param, base, logger)
    logger.debug('obj %d: photon_op = %s', base.get('obj_num',0), op)

    param['current'] = op, False, None, index, index_key

    return op


class PhotonOpBuilder(object):
    """A base class for building PhotonOp objects.

    The base class defines the call signatures of the methods that any derived class should follow.
    """
    def buildPhotonOp(self, config, base, logger):
        """Build the PhotonOp based on the specifications in the config dict.

        Note: Sub-classes must override this function with a real implementation.

        Parameters:
            config:     The configuration dict for the PhotonOp
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed PhotonOp object.
        """
        raise NotImplementedError("The %s class has not overridden buildPhotonOp"%self.__class__)


class SimplePhotonOpBuilder(PhotonOpBuilder):
    """A class for building simple PhotonOp objects.

    The initializer takes an init_func, which is the class or function to call to build the
    PhotonOp.  For the kwargs, it calls getKwargs, which does the normal parsing of the req_params
    and related class attributes.
    """
    def __init__(self, init_func):
        self.init_func = init_func

    def getKwargs(self, config, base, logger):
        """Get the kwargs to pass to the build function based on the following attributes of
        init_func:

        _req_params
                        A dict of required parameters and their types.
        _opt_params
                        A dict of optional parameters and their types.
        _single_params
                        A list of dicts of parameters such that one and only one of
                        parameter in each dict is required.
        _takes_rng
                        A bool value saying whether an rng object is required.

        See the classes in photon_array.py for examples of classes that set these attributes.

        Parameters:
            config:     The configuration dict for the photon_op type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            kwargs
        """
        req, opt, single, takes_rng = get_cls_params(self.init_func)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        if takes_rng:  # pragma: no cover  None of ours have this anymore.  But it's still allowed.
            kwargs['rng'] = GetRNG(config, base, logger, self.init_func.__name__)
        return kwargs

    def buildPhotonOp(self, config, base, logger):
        """Build the PhotonOp based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the photon_op type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed PhotonOp object.
        """
        kwargs = self.getKwargs(config,base,logger)
        return self.init_func(**kwargs)

class WavelengthSamplerBuilder(PhotonOpBuilder):
    """Build a WavelengthSampler
    """
    # This one needs special handling for sed and bandpass
    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(WavelengthSampler)
        kwargs, safe = GetAllParams(config, base, req, opt, single, ignore=['sed'])
        if 'sed' not in config:
            raise GalSimConfigError("sed is required for WavelengthSampler")
        sed = BuildSED(config, 'sed', base, logger)[0]
        kwargs['sed'] = sed
        if 'bandpass' not in base:
            raise GalSimConfigError("bandpass is required for WavelengthSampler")
        kwargs['bandpass'] = base['bandpass']
        return WavelengthSampler(**kwargs)

class PhotonDCRBuilder(PhotonOpBuilder):
    """Build a PhotonDCR
    """
    # This one needs special handling for obj_coord
    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(PhotonDCR)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        kwargs['obj_coord'] = base['sky_pos']
        return PhotonDCR(**kwargs)

class ListPhotonOpBuilder(PhotonOpBuilder):
    """Select a photon_op from a list
    """
    def buildPhotonOp(self, config, base, logger):
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
            raise GalSimConfigError("index %d out of bounds for photon_op type=List"%index)
        return BuildPhotonOp(items, index, base)

def RegisterPhotonOpType(photon_op_type, builder, input_type=None):
    """Register a photon_op type for use by the config apparatus.

    Parameters:
        photon_op_type: The name of the config type to register
        builder:        A builder object to use for building the PhotonOp object.  It should
                        be an instance of a subclass of PhotonOpBuilder.
        input_type:     If the PhotonOp builder utilises an input object, give the key name of the
                        input type here.  (If it uses more than one, this may be a list.)
                        [default: None]
    """
    valid_photon_op_types[photon_op_type] = builder
    RegisterInputConnectedType(input_type, photon_op_type)


RegisterPhotonOpType('WavelengthSampler', WavelengthSamplerBuilder())
RegisterPhotonOpType('FRatioAngles', SimplePhotonOpBuilder(FRatioAngles))
RegisterPhotonOpType('PhotonDCR', PhotonDCRBuilder())
RegisterPhotonOpType('Refraction', SimplePhotonOpBuilder(Refraction))
RegisterPhotonOpType('FocusDepth', SimplePhotonOpBuilder(FocusDepth))
RegisterPhotonOpType('List', ListPhotonOpBuilder())
