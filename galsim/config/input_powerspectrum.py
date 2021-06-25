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

from __future__ import print_function

import math
import warnings
import logging

from .input import InputLoader, GetInputObj, RegisterInputType
from .util import LoggerWrapper, SetupConfigRNG, GetRNG
from .value import ParseValue, GetAllParams, CheckAllParams, RegisterValueType
from .stamp import ParseWorldPos
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..position import PositionD
from ..shear import Shear
from ..lensing_ps import PowerSpectrum

# This file adds input type nfw_halo and value types PowerSpectrumShear and
# PowerSpectrumMagnification.

# A PowerSpectrum input type requires a special initialization at the start of each image
# to build the shear grid.  This is done in SetupPowerSpecrum.  There are also a couple of
# parameters that are specific to that step, which we want to ignore when getting the
# initialization kwargs, so we define a special GetPowerSpectrumKwargs function here.

class PowerSpectrumLoader(InputLoader):

    def getKwargs(self, config, base, logger):
        """Parse the config dict and return the kwargs needed to build the PowerSpectrum object.

        Parameters:
            config:     The configuration dict for 'power_spectrum'
            base:       The base configuration dict
            logger:     If given, a logger object to log progress.

        Returns:
            kwargs, safe
        """
        logger = LoggerWrapper(logger)

        # If we are going to use a different rebuilding cadence than the normal once per image,
        # then in order for this feature to work properly in a multiprocessing context,
        # we need to have it use an rng that also updates at the same cadence as this
        # index value.  So if we don't have an rng_num yet, make a new random_seed value
        # that tracks this index.
        if ('index' in config and 'rng_num' not in config and 'image' in base and
            'random_seed' in base['image']):
            obj_num = base.get('obj_num',None)
            image_num = base.get('image_num',None)
            file_num = base.get('file_num',None)
            base['obj_num'] = base['image_num'] = base['file_num'] = 0
            if isinstance(base['image']['random_seed'],list):
                first_seed = ParseValue(base['image']['random_seed'], 0, base, int)[0]
            else:
                first_seed = ParseValue(base['image'], 'random_seed', base, int)[0]
            rs = base['image']['random_seed']
            if not isinstance(rs, list): rs = [rs]
            first_seed += 31415  # An arbitrary offset to avoid the chance of unwanted correlations
            rs.append({ 'type' : 'Eval',
                        'str' : 'first + ps_index',
                        'ifirst' : first_seed,
                        'ips_index' : config['index'] })
            config['rng_num'] = len(rs) - 1
            base['image']['random_seed'] = rs
            base['index_key'] = 'file_num'
            SetupConfigRNG(base, logger=logger)
            if image_num is not None:
                base['index_key'] = 'image_num'
                SetupConfigRNG(base, logger=logger)
            base['index_key'] = 'file_num'  # This is what we want to leave it as.
            base['obj_num'] = obj_num
            base['image_num'] = image_num
            base['file_num'] = file_num

        # Ignore these parameters here, since they are for the buildGrid step, not the
        # initialization of the PowerSpectrum object.
        ignore = ['grid_spacing', 'ngrid', 'interpolant', 'variance', 'center', 'index']
        opt = PowerSpectrum._opt_params
        return GetAllParams(config, base, opt=opt, ignore=ignore)

    def setupImage(self, input_obj, config, base, logger=None):
        """Set up the PowerSpectrum input object's gridded values based on the
        size of the image and the grid spacing.

        Parameters:
            input_obj:  The PowerSpectrum object to use
            config:     The configuration dict for 'power_spectrum'
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress.
        """
        logger = LoggerWrapper(logger)
        # Attach the logger to the input_obj so we can use it when evaluating values.
        input_obj.logger = logger

        if 'grid_spacing' in config:
            grid_spacing = ParseValue(config, 'grid_spacing', base, float)[0]
        elif 'grid_xsize' in base and 'grid_ysize' in base:
            # Then we have a tiled image.  Can use the tile spacing as the grid spacing.
            grid_size = min(base['grid_xsize'], base['grid_ysize'])
            # This size is in pixels, so we need to convert to arcsec using the pixel scale.
            # Note: we use the (max) pixel scale at the image center.  This isn't
            # necessarily optimal, but it seems like the best choice for a non-trivial WCS.
            scale = base['wcs'].maxLinearScale(base['image_center'])
            grid_spacing = grid_size * scale
        else:
            raise GalSimConfigError(
                "power_spectrum.grid_spacing required for non-tiled images")

        if 'ngrid' in config:
            ngrid = ParseValue(config, 'ngrid', base, float)[0]
        elif 'grid_xsize' in base and base['grid_xsize'] == base['grid_ysize']:
            # PowerSpectrum can only do a square FFT, so make it the larger of the two n's.
            nx_grid = int(math.ceil(base['image_xsize']/base['grid_xsize']))
            ny_grid = int(math.ceil(base['image_ysize']/base['grid_ysize']))
            ngrid = max(nx_grid, ny_grid) + 1
            # Normally that's good, but if tiles aren't square, need to drop through to the
            # second option.
        else:
            image_size = max(base['image_xsize'], base['image_ysize'])
            scale = base['wcs'].maxLinearScale(base['image_center'])
            ngrid = int(math.ceil(image_size * scale / grid_spacing)) + 1

        if 'interpolant' in config:
            interpolant = ParseValue(config, 'interpolant', base, str)[0]
        else:
            interpolant = None

        if 'variance' in config:
            variance = ParseValue(config, 'variance', base, float)[0]
        else:
            variance = None

        if 'center' in config:
            center = ParseWorldPos(config, 'center', base, logger)
        elif base['wcs']._isCelestial:
            center = PositionD(0,0)
        else:
            center = base['wcs'].toWorld(base['image_center'])

        if 'index' in config:
            index = ParseValue(config, 'index', base, int)[0]
            current_index = config.get('current_setup_index',None)
            if index == current_index:
                logger.info('image %d: power spectrum grid is already current',
                            base.get('image_num',0))
                return
            config['current_setup_index'] = index

        rng = GetRNG(config, base, logger, 'PowerSpectrum')

        # We don't care about the output here.  This just builds the grid, which we'll
        # access for each object using its position.
        logger.debug('image %d: PowerSpectrum buildGrid(grid_spacing=%s, ngrid=%s, center=%s, '
                     'interpolant=%s, variance=%s)',
                     base.get('image_num',0), grid_spacing, ngrid, center, interpolant, variance)
        input_obj.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid, center=center,
                            rng=rng, interpolant=interpolant, variance=variance)

        # Make sure this process gives consistent results regardless of the number of processes
        # being used.
        if not isinstance(input_obj, PowerSpectrum) and rng is not None:
            # Then input_obj is really a proxy, which means the rng was pickled, so we need to
            # discard the same number of random calls from the one in the config dict.
            rng.discard(input_obj.nRandCallsForBuildGrid())

# Register this as a valid input type
RegisterInputType('power_spectrum', PowerSpectrumLoader(PowerSpectrum))


# There are two value types associated with this: PowerSpectrumShear and
# PowerSpectrumMagnification.

def _GenerateFromPowerSpectrumShear(config, base, value_type):
    """Return a shear calculated from a PowerSpectrum object.
    """
    power_spectrum = GetInputObj('power_spectrum', config, base, 'PowerSpectrumShear')
    logger = power_spectrum.logger

    if 'uv_pos' not in base:
        raise GalSimConfigError("PowerSpectrumShear requested, but no position defined.")
    pos = base['uv_pos']

    # There aren't any parameters for this, so just make sure num is the only (optional)
    # one present.
    CheckAllParams(config, opt={ 'num' : int })

    with warnings.catch_warnings(record=True) as w:
        g1,g2 = power_spectrum.getShear(pos)
    if len(w) > 0:
        # Send the warning to the logger, rather than raising a normal warning.
        # The warning here would typically be that the position is out of range of where the
        # power spectrum is defined.  So if we do get this and the position is not on the image,
        # we probably don't care.  In that case, just log it as debug, not warn.
        log_level = (logging.WARNING if 'current_image' in base and
                                        base['current_image'].outer_bounds.includes(pos)
                     else logging.DEBUG)
        for ww in w:
            logger.log(log_level, 'obj %d: %s',base['obj_num'], ww.message)

    try:
        shear = Shear(g1=g1,g2=g2)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.warning('obj %d: Warning: PowerSpectrum shear (g1=%f, g2=%f) is invalid. '%(
                       base['obj_num'],g1,g2) + 'Using shear = 0.')
        shear = Shear(g1=0,g2=0)

    logger.debug('obj %d: PowerSpectrum shear = %s',base['obj_num'],shear)
    return shear, False

def _GenerateFromPowerSpectrumMagnification(config, base, value_type):
    """Return a magnification calculated from a PowerSpectrum object.
    """
    power_spectrum = GetInputObj('power_spectrum', config, base, 'PowerSpectrumMagnification')
    logger = power_spectrum.logger

    if 'uv_pos' not in base:
        raise GalSimConfigError(
            "PowerSpectrumMagnification requested, but no position defined.")
    pos = base['uv_pos']

    opt = { 'max_mu' : float, 'num' : int }
    kwargs = GetAllParams(config, base, opt=opt)[0]

    with warnings.catch_warnings(record=True) as w:
        mu = power_spectrum.getMagnification(pos)
    if len(w) > 0:
        log_level = (logging.WARNING if 'current_image' in base and
                                        base['current_image'].outer_bounds.includes(pos)
                     else logging.DEBUG)
        for ww in w:
            logger.log(log_level, 'obj %d: %s',base['obj_num'], ww.message)

    max_mu = kwargs.get('max_mu', 25.)
    if not max_mu > 0.:
        raise GalSimConfigValueError(
            "Invalid max_mu for type = PowerSpectrumMagnification (must be > 0)", max_mu)

    if mu < 0 or mu > max_mu:
        logger.warning('obj %d: Warning: PowerSpectrum mu = %f means strong lensing. '%(
                       base['obj_num'],mu) + 'Using mu=%f'%max_mu)
        mu = max_mu

    logger.debug('obj %d: PowerSpectrum mu = %s',base['obj_num'],mu)
    return mu, False

# Register these as valid value types
RegisterValueType('PowerSpectrumShear', _GenerateFromPowerSpectrumShear, [Shear],
                  input_type='power_spectrum')
RegisterValueType('PowerSpectrumMagnification', _GenerateFromPowerSpectrumMagnification, [float],
                  input_type='power_spectrum')
