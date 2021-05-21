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

from .input import InputLoader, GetInputObj, RegisterInputType, GetNumInputObj
from .util import LoggerWrapper, GetIndex, GetRNG, get_cls_params
from .value import GetAllParams, SetDefaultIndex
from .gsobject import RegisterObjectType
from ..gsparams import GSParams
from ..errors import GalSimConfigError
from ..real import RealGalaxyCatalog, RealGalaxy, ChromaticRealGalaxy

# This file adds input type real_catalog and gsobject types RealGalaxy and RealGalaxyOriginal.

# The RealGalaxyCatalog doesn't need anything special other than registration as a valid
# input type.
RegisterInputType('real_catalog', InputLoader(RealGalaxyCatalog, takes_logger=True))

# There are two gsobject types that are coupled to this: RealGalaxy and RealGalaxyOriginal.

def _BuildRealGalaxy(config, base, ignore, gsparams, logger, param_name='RealGalaxy'):
    """Build a RealGalaxy from the real_catalog input item.
    """
    real_cat = GetInputObj('real_catalog', config, base, param_name)

    # Special: if index is Sequence or Random, and max isn't set, set it to nobjects-1.
    # But not if they specify 'id' or have 'random=True', which overrides that.
    if 'id' not in config:
        if 'random' not in config:
            SetDefaultIndex(config, real_cat.getNObjects())
        else:
            if not config['random']:
                SetDefaultIndex(config, real_cat.getNObjects())
                # Need to do this to avoid being caught by the GetAllParams() call, which will flag
                # it if it has 'index' and 'random' set (but 'random' is False, so really it's OK).
                del config['random']

    req, opt, single, _ = get_cls_params(RealGalaxy)
    kwargs, safe = GetAllParams(config, base, req, opt, single, ignore = ignore + ['num'])
    if gsparams: kwargs['gsparams'] = GSParams(**gsparams)

    kwargs['rng'] = GetRNG(config, base, logger, param_name)

    if 'index' in kwargs:
        index = kwargs['index']
        if index >= real_cat.getNObjects() or index < 0:
            raise GalSimConfigError(
                "index=%s has gone past the number of entries in the RealGalaxyCatalog"%index)

    kwargs['real_galaxy_catalog'] = real_cat
    logger.debug('obj %d: %s kwargs = %s',base.get('obj_num',0),param_name,kwargs)

    gal = RealGalaxy(**kwargs)

    return gal, safe


def _BuildRealGalaxyOriginal(config, base, ignore, gsparams, logger):
    """Return the original image from a RealGalaxy using the real_catalog input item.
    """
    gal, safe = _BuildRealGalaxy(config, base, ignore, gsparams, logger,
                                   param_name='RealGalaxyOriginal')
    return gal.original_gal, safe

def _BuildChromaticRealGalaxy(config, base, ignore, gsparams, logger):
    """Build a ChromaticRealGalaxy from several real_catalog input items.
    """
    param_name = 'ChromaticRealGalaxy'
    ncats = GetNumInputObj('real_catalog', base)
    real_cats = [ GetInputObj('real_catalog', config, base, param_name, n)
                  for n in range(ncats)]

    # Special: if index is Sequence or Random, and max isn't set, set it to nobjects-1.
    # But not if they specify 'id' or have 'random=True', which overrides that.
    if 'id' not in config:
        if 'random' not in config:
            SetDefaultIndex(config, real_cats[0].getNObjects())
        else:
            if not config['random']:
                SetDefaultIndex(config, real_cats[0].getNObjects())
                # Need to do this to avoid being caught by the GetAllParams() call, which will flag
                # it if it has 'index' and 'random' set (but 'random' is False, so really it's OK).
                del config['random']

    req, opt, single, _ = get_cls_params(ChromaticRealGalaxy)

    kwargs, safe = GetAllParams(config, base, req, opt, single, ignore)
    if gsparams: kwargs['gsparams'] = GSParams(**gsparams)

    kwargs['rng'] = GetRNG(config, base, logger, param_name)

    if 'index' in kwargs:
        index = kwargs['index']
        if index >= real_cats[0].getNObjects() or index < 0:
            raise GalSimConfigError(
                "index=%s has gone past the number of entries in the RealGalaxyCatalog"%index)

    kwargs['real_galaxy_catalogs'] = real_cats
    logger.debug('obj %d: %s kwargs = %s',base.get('obj_num',0),param_name,kwargs)

    gal = ChromaticRealGalaxy(**kwargs)

    return gal, safe


# Register these as valid gsobject types
RegisterObjectType('RealGalaxy', _BuildRealGalaxy, input_type='real_catalog')
RegisterObjectType('RealGalaxyOriginal', _BuildRealGalaxyOriginal, input_type='real_catalog')
RegisterObjectType('ChromaticRealGalaxy', _BuildChromaticRealGalaxy, input_type='real_catalog')
