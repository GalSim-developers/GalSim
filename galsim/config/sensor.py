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
from ..sensor import Sensor, SiliconSensor
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..utilities import basestring

# This file handles the construction of a Sensor in config['image']['sensor'].

# This module-level dict will store all the registered sensor types.
# See the RegisterSensorType function at the end of this file.
# The keys are the (string) names of the sensor types, and the values will be builders that
# know how to build the Sensor objects.
valid_sensor_types = {}


def BuildSensor(config, key, base, logger=None):
    """Read the parameters from config[key] and return a constructed Sensor.

    Parameters:
        config:     A dict with the configuration information for the sensor.
                    (usually base['image'])
        key:        The key in the dict for the sensor configuration.
        base:       The base dict of the configuration.
        logger:     Optionally, provide a logger for logging debug statements. [default: None]

    Returns:
        a Sensor
    """
    logger = LoggerWrapper(logger)
    logger.debug('obj %d: Start BuildSensor key = %s',base.get('obj_num',0),key)

    param = config[key]

    # Check for direct value, else get the type
    if isinstance(param, Sensor):
        return param
    elif isinstance(param, basestring) and (param[0] == '$' or param[0] == '@'):
        return ParseValue(config, key, base, None)[0]
    elif isinstance(param, dict):
        sensor_type = param.get('type', 'Simple')
    else:
        raise GalSimConfigError("sensor must be either a Sensor or a dict")

    # For these two, just do the usual ParseValue function.
    if sensor_type in ('Eval', 'Current'):
        return ParseValue(config, key, base, None)[0]

    if sensor_type not in valid_sensor_types:
        raise GalSimConfigValueError("Invalid sensor type.", sensor_type,
                                     list(valid_sensor_types.keys()))

    # Check if we can use the current cached object
    index, index_key = GetIndex(param, base)
    if 'current' in param:
        csensor, csafe, cvalue_type, cindex, cindex_key = param['current']
        if cindex == index:
            logger.debug('obj %d: The sensor is already current', base.get('obj_num',0))
            logger.debug('obj %d: index_key = %s, index = %d',base.get('obj_num',0),
                         cindex_key, cindex)
            return csensor

    # Need to use a builder.
    logger.debug('obj %d: Building sensor type %s', base.get('obj_num',0), sensor_type)
    builder = valid_sensor_types[sensor_type]
    sensor = builder.buildSensor(param, base, logger)
    logger.debug('obj %d: sensor = %s', base.get('obj_num',0), sensor)

    param['current'] = sensor, False, None, index, index_key

    return sensor


class SensorBuilder(object):
    """A base class for building Sensor objects.

    The base class defines the call signatures of the methods that any derived class should follow.
    """
    def buildSensor(self, config, base, logger):
        """Build the Sensor based on the specifications in the config dict.

        Note: Sub-classes must override this function with a real implementation.

        Parameters:
            config:     The configuration dict for the Sensor
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed Sensor object.
        """
        raise NotImplementedError("The %s class has not overridden buildSensor"%self.__class__)


class SimpleSensorBuilder(SensorBuilder):
    """A class for building simple Sensor objects.

    The initializer takes an init_func, which is the class or function to call to build the
    Sensor.  For the kwargs, it calls getKwargs, which does the normal parsing of the req_params
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

        See the classes in sensor.py for examples of classes that set these attributes.

        Parameters:
            config:     The configuration dict for the sensor type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            kwargs
        """
        req, opt, single, takes_rng = get_cls_params(self.init_func)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        if takes_rng:
            kwargs['rng'] = GetRNG(config, base, logger, self.init_func.__name__)
        return kwargs

    def buildSensor(self, config, base, logger):
        """Build the Sensor based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the sensor type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed Sensor object.
        """
        kwargs = self.getKwargs(config,base,logger)
        return self.init_func(**kwargs)

class ListSensorBuilder(SensorBuilder):
    """Select a sensor from a list
    """
    def buildSensor(self, config, base, logger):
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
            raise GalSimConfigError("index %d out of bounds for sensor type=List"%index)
        return BuildSensor(items, index, base)

def RegisterSensorType(sensor_type, builder, input_type=None):
    """Register a sensor type for use by the config apparatus.

    Parameters:
        sensor_type:    The name of the config type to register
        builder:        A builder object to use for building the Sensor object.  It should
                        be an instance of a subclass of SensorBuilder.
        input_type:     If the Sensor builder utilises an input object, give the key name of the
                        input type here.  (If it uses more than one, this may be a list.)
                        [default: None]
    """
    valid_sensor_types[sensor_type] = builder
    RegisterInputConnectedType(input_type, sensor_type)


RegisterSensorType('Simple', SimpleSensorBuilder(Sensor))
RegisterSensorType('Silicon', SimpleSensorBuilder(SiliconSensor))
RegisterSensorType('List', ListSensorBuilder())
