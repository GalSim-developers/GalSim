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

from .util import LoggerWrapper
from .value import ParseValue, GetAllParams, GetIndex
from .input import RegisterInputConnectedType
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..bandpass import Bandpass
from ..utilities import basestring

# This module-level dict will store all the registered Bandpass types.
# See the RegisterBandpassType function at the end of this file.
# The keys are the (string) names of the Bandpass types, and the values will be builders that know
# how to build the Bandpass object.
valid_bandpass_types = {}

def BuildBandpass(config, key, base, logger=None):
    """Read the Bandpass parameters from config[key] and return a constructed Bandpass object.

    Parameters:
        config:     A dict with the configuration information.
        key:        The key name in config indicating which object to build.
        base:       The base dict of the configuration.
        logger:     Optionally, provide a logger for logging debug statements. [default: None]

    Returns:
        (bandpass, safe) where bandpass is a Bandpass instance, and safe is whether it is safe to
                         reuse.
    """
    logger = LoggerWrapper(logger)
    logger.debug('obj %d: Start BuildBandpass key = %s',base.get('obj_num',0),key)

    param = config[key]

    # Check for direct value, else get the bandpass type
    if isinstance(param, Bandpass):
        return param, True
    elif isinstance(param, basestring) and (param[0] == '$' or param[0] == '@'):
        return ParseValue(config, key, base, None)
    elif isinstance(param, dict):
        bandpass_type = param.get('type', 'FileBandpass')
    else:
        raise GalSimConfigError("%s must be either a Bandpass or a dict"%key)

    # For these two, just do the usual ParseValue function.
    if bandpass_type in ('Eval', 'Current'):
        return ParseValue(config, key, base, None)

    # Check if we can use the current cached object
    index, index_key = GetIndex(param, base)
    if 'current' in param:
        cbandpass, csafe, cvalue_type, cindex, cindex_key = param['current']
        if cindex == index:
            logger.debug('obj %d: The Bandpass object is already current', base.get('obj_num',0))
            logger.debug('obj %d: index_key = %s, index = %d',base.get('obj_num',0),
                         cindex_key, cindex)
            return cbandpass, csafe

    if bandpass_type not in valid_bandpass_types:
        raise GalSimConfigValueError("Invalid bandpass.type.", bandpass_type,
                                     list(valid_bandpass_types.keys()))
    logger.debug('obj %d: Building bandpass type %s', base.get('obj_num',0), bandpass_type)
    builder = valid_bandpass_types[bandpass_type]
    bandpass, safe = builder.buildBandpass(param, base, logger)
    logger.debug('obj %d: bandpass = %s', base.get('obj_num',0), bandpass)

    param['current'] = bandpass, safe, Bandpass, index, index_key

    return bandpass, safe


class BandpassBuilder(object):
    """A base class for building Bandpass objects.

    The base class defines the call signatures of the methods that any derived class should follow.
    """
    def buildBandpass(self, config, base, logger):
        """Build the Bandpass based on the specifications in the config dict.

        Note: Sub-classes must override this function with a real implementation.

        Parameters:
            config:     The configuration dict for the bandpass type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed Bandpass object.
        """
        raise NotImplementedError("The %s class has not overridden buildBandpass"%self.__class__)


class FileBandpassBuilder(BandpassBuilder):
    """A class for loading a Bandpass from a file

    FileBandpass expected the following parameters:

        file_name (str)     The file to load (required)
        wave_type(str)      The units (nm or Ang) of the wavelengths in the file (required)
        thin (float)        A relative error to use for thinning the file (default: None)
        blue_limit (float)  A cutoff wavelength on the blue side (default: None)
        red_limit (float)   A cutoff wavelength on the red side (default: None)
        zeropoint (float)   A zeropoint to use (default: None)
    """
    def buildBandpass(self, config, base, logger):
        """Build the Bandpass based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the bandpass type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed Bandpass object.
        """
        logger = LoggerWrapper(logger)

        req = {'file_name': str, 'wave_type': str}
        opt = {'thin' : float, 'blue_limit' : float, 'red_limit' : float, 'zeropoint': float }

        kwargs, safe = GetAllParams(config, base, req=req, opt=opt)

        file_name = kwargs.pop('file_name')
        thin = kwargs.pop('thin', None)

        logger.info("Reading Bandpass file: %s",file_name)
        bandpass = Bandpass(file_name, **kwargs)
        if thin:
            bandpass = bandpass.thin(thin)

        return bandpass, safe

def RegisterBandpassType(bandpass_type, builder, input_type=None):
    """Register a bandpass type for use by the config apparatus.

    Parameters:
        bandpass_type:  The name of the type in the config dict.
        builder:        A builder object to use for building the Bandpass object.  It should
                        be an instance of a subclass of BandpassBuilder.
        input_type:     If the Bandpass builder utilises an input object, give the key name of the
                        input type here.  (If it uses more than one, this may be a list.)
                        [default: None]
    """
    valid_bandpass_types[bandpass_type] = builder
    RegisterInputConnectedType(input_type, bandpass_type)

RegisterBandpassType('FileBandpass', FileBandpassBuilder())
