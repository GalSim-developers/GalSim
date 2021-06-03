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

from .input import InputLoader, GetInputObj, RegisterInputType
from .value import GetCurrentValue, CheckAllParams, GetAllParams, RegisterValueType
from .util import LoggerWrapper
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..shear import Shear
from ..nfw_halo import NFWHalo

# This file adds input type nfw_halo and value types NFWHaloShear and NFWHaloMagnification.

class NFWLoader(InputLoader):
    def setupImage(self, input_obj, config, base, logger=None):
        # Just attach the logger to the input_obj so we can use it when evaluating values.
        input_obj.logger = LoggerWrapper(logger)

# Register this as a valid input type
RegisterInputType('nfw_halo', NFWLoader(NFWHalo))

def _GenerateFromNFWHaloShear(config, base, value_type):
    """Return a shear calculated from an NFWHalo object.
    """
    nfw_halo = GetInputObj('nfw_halo', config, base, 'NFWHaloShear')
    logger = nfw_halo.logger

    if 'uv_pos' not in base:
        raise GalSimConfigError("NFWHaloShear requested, but no position defined.")
    pos = base['uv_pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise GalSimConfigError("NFWHaloShear requested, but no gal.redshift defined.")
    redshift = GetCurrentValue('redshift', base['gal'], float, base)

    # There aren't any parameters for this, so just make sure num is the only (optional)
    # one present.
    CheckAllParams(config, opt={ 'num' : int })

    g1,g2 = nfw_halo.getShear(pos,redshift)

    try:
        shear = Shear(g1=g1,g2=g2)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.warning('obj %d: Warning: NFWHalo shear (g1=%f, g2=%f) is invalid. '%(
                       base['obj_num'],g1,g2) + 'Using shear = 0.')
        shear = Shear(g1=0,g2=0)

    logger.debug('obj %d: NFWHalo shear = %s',base['obj_num'],shear)
    return shear, False


def _GenerateFromNFWHaloMagnification(config, base, value_type):
    """Return a magnification calculated from an NFWHalo object.
    """
    nfw_halo = GetInputObj('nfw_halo', config, base, 'NFWHaloMagnification')
    logger = nfw_halo.logger

    if 'uv_pos' not in base:
        raise GalSimConfigError("NFWHaloMagnification requested, but no position defined.")
    pos = base['uv_pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise GalSimConfigError("NFWHaloMagnification requested, but no gal.redshift defined.")
    redshift = GetCurrentValue('redshift', base['gal'], float, base)

    opt = { 'max_mu' : float, 'num' : int }
    kwargs = GetAllParams(config, base, opt=opt)[0]

    max_mu = kwargs.get('max_mu', 25.)
    if not max_mu > 0.:
        raise GalSimConfigValueError(
            "Invalid max_mu for type = NFWHaloMagnification (must be > 0)", max_mu)

    mu = nfw_halo.getMagnification(pos,redshift)
    if mu < 0 or mu > max_mu:
        logger.warning('obj %d: Warning: NFWHalo mu = %f means strong lensing. '%(
                       base['obj_num'],mu) + 'Using mu = %f'%max_mu)
        mu = max_mu

    logger.debug('obj %d: NFWHalo mu = %s',base['obj_num'],mu)
    return mu, False


# Register these as valid value types
RegisterValueType('NFWHaloShear', _GenerateFromNFWHaloShear, [ Shear ],
                  input_type='nfw_halo')
RegisterValueType('NFWHaloMagnification', _GenerateFromNFWHaloMagnification, [ float ],
                  input_type='nfw_halo')
