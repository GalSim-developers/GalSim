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
from ..sed import SED
from ..bandpass import Bandpass
from ..utilities import basestring, LRU_Cache

# This module-level dict will store all the registered SED types.
# See the RegisterSEDType function at the end of this file.
# The keys are the (string) names of the SED types, and the values will be builders that know
# how to build the SED object.
valid_sed_types = {}

def BuildSED(config, key, base, logger=None):
    """Read the SED parameters from config[key] and return a constructed SED object.

    Parameters:
        config:     A dict with the configuration information.
        key:        The key name in config indicating which object to build.
        base:       The base dict of the configuration.
        logger:     Optionally, provide a logger for logging debug statements. [default: None]

    Returns:
        (sed, safe) where sed is an SED instance, and safe is whether it is safe to reuse.
    """
    logger = LoggerWrapper(logger)
    logger.debug('obj %d: Start BuildSED key = %s',base.get('obj_num',0),key)

    param = config[key]

    # Check for direct value, else get the SED type
    if isinstance(param, SED):
        return param, True
    elif isinstance(param, basestring) and (param[0] == '$' or param[0] == '@'):
        return ParseValue(config, key, base, None)
    elif isinstance(param, dict):
        sed_type = param.get('type','FileSED')
    else:
        raise GalSimConfigError("%s must be either an SED or a dict"%key)

    # For these two, just do the usual ParseValue function.
    if sed_type in ('Eval', 'Current'):
        return ParseValue(config, key, base, None)

    # Check if we can use the current cached object
    index, index_key = GetIndex(param, base)
    if 'current' in param:
        csed, csafe, cvalue_type, cindex, cindex_key = param['current']
        if cindex == index:
            logger.debug('obj %d: The SED object is already current', base.get('obj_num',0))
            logger.debug('obj %d: index_key = %s, index = %d',base.get('obj_num',0),
                         cindex_key, cindex)
            return csed, csafe

    if sed_type not in valid_sed_types:
        raise GalSimConfigValueError("Invalid sed.type.", sed_type, list(valid_sed_types.keys()))
    logger.debug('obj %d: Building sed type %s', base.get('obj_num',0), sed_type)
    builder = valid_sed_types[sed_type]
    sed, safe = builder.buildSED(param, base, logger)
    logger.debug('obj %d: sed = %s', base.get('obj_num',0), sed)

    param['current'] = sed, safe, SED, index, index_key

    return sed, safe


class SEDBuilder(object):
    """A base class for building SED objects.

    The base class defines the call signatures of the methods that any derived class should follow.
    """
    def buildSED(self, config, base, logger):
        """Build the SED based on the specifications in the config dict.

        Note: Sub-classes must override this function with a real implementation.

        Parameters:
            config:     The configuration dict for the SED type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed SED object.
        """
        raise NotImplementedError("The %s class has not overridden buildSED"%self.__class__)


def _read_sed_file(file_name, wave_type, flux_type):
    return SED(file_name, wave_type, flux_type)
read_sed_file = LRU_Cache(_read_sed_file)

class FileSEDBuilder(SEDBuilder):
    """A class for loading an SED from a file

    FileSED expected the following parameters:

        file_name (required)    The file to load
        wave_type(required)     The units (nm or Ang) of the wavelengths in the file
        flux_type (required)    Which kind of flux values are in the file
                                Allowed values: flambda, fnu, fphotons, 1
    """
    def buildSED(self, config, base, logger):
        """Build the SED based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the SED type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed SED object.
        """
        from .bandpass import BuildBandpass
        logger = LoggerWrapper(logger)

        req = {'file_name': str, 'wave_type': str, 'flux_type': str}
        opt = {'norm_flux_density': float, 'norm_wavelength': float,
               'norm_flux': float, 'redshift': float}
        ignore = ['norm_bandpass']

        kwargs, safe = GetAllParams(config, base, req=req, opt=opt, ignore=ignore)

        file_name = kwargs.pop('file_name')

        norm_flux_density = kwargs.pop('norm_flux_density', None)
        norm_wavelength = kwargs.pop('norm_wavelength', None)
        norm_flux = kwargs.pop('norm_flux', None)
        redshift = kwargs.pop('redshift', 0.)
        wave_type = kwargs.pop('wave_type')
        flux_type = kwargs.pop('flux_type')

        logger.info("Using SED file: %s",file_name)
        sed = read_sed_file(file_name, wave_type, flux_type)
        if norm_flux_density:
            sed = sed.withFluxDensity(norm_flux_density, wavelength=norm_wavelength)
        elif norm_flux:
            bandpass, safe1 = BuildBandpass(config, 'norm_bandpass', base, logger)
            sed = sed.withFlux(norm_flux, bandpass=bandpass)
            safe = safe and safe1
        sed = sed.atRedshift(redshift)

        return sed, safe

def RegisterSEDType(sed_type, builder, input_type=None):
    """Register a SED type for use by the config apparatus.

    Parameters:
        sed_type:       The name of the type in the config dict.
        builder:        A builder object to use for building the SED object.  It should
                        be an instance of a subclass of SEDBuilder.
        input_type:     If the SED builder utilises an input object, give the key name of the
                        input type here.  (If it uses more than one, this may be a list.)
                        [default: None]
    """
    valid_sed_types[sed_type] = builder
    RegisterInputConnectedType(input_type, sed_type)

RegisterSEDType('FileSED', FileSEDBuilder())
