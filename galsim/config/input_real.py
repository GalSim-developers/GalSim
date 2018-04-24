# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
import galsim

# This file adds input type real_catalog and gsobject types RealGalaxy and RealGalaxyOriginal.

# The RealGalaxyCatalog doesn't need anything special other than registration as a valid
# input type.
from .input import RegisterInputType, InputLoader
RegisterInputType('real_catalog', InputLoader(galsim.RealGalaxyCatalog))

# There are two gsobject types that are coupled to this: RealGalaxy and RealGalaxyOriginal.

def _BuildRealGalaxy(config, base, ignore, gsparams, logger, param_name='RealGalaxy'):
    """@brief Build a RealGalaxy from the real_catalog input item.
    """
    real_cat = galsim.config.GetInputObj('real_catalog', config, base, param_name)

    # Special: if index is Sequence or Random, and max isn't set, set it to nobjects-1.
    # But not if they specify 'id' or have 'random=True', which overrides that.
    if 'id' not in config:
        if 'random' not in config:
            galsim.config.SetDefaultIndex(config, real_cat.getNObjects())
        else:
            if not config['random']:
                galsim.config.SetDefaultIndex(config, real_cat.getNObjects())
                # Need to do this to avoid being caught by the GetAllParams() call, which will flag
                # it if it has 'index' and 'random' set (but 'random' is False, so really it's OK).
                del config['random']

    kwargs, safe = galsim.config.GetAllParams(config, base,
        req = galsim.__dict__['RealGalaxy']._req_params,
        opt = galsim.__dict__['RealGalaxy']._opt_params,
        single = galsim.__dict__['RealGalaxy']._single_params,
        ignore = ignore + ['num'])
    if gsparams: kwargs['gsparams'] = galsim.GSParams(**gsparams)

    kwargs['rng'] = galsim.config.GetRNG(config, base, logger, param_name)

    if 'index' in kwargs:
        index = kwargs['index']
        if index >= real_cat.getNObjects() or index < 0:
            raise galsim.GalSimConfigError(
                "index=%s has gone past the number of entries in the RealGalaxyCatalog"%index)

    kwargs['real_galaxy_catalog'] = real_cat
    logger.debug('obj %d: %s kwargs = %s',base.get('obj_num',0),param_name,kwargs)

    gal = galsim.RealGalaxy(**kwargs)

    return gal, safe


def _BuildRealGalaxyOriginal(config, base, ignore, gsparams, logger):
    """@brief Return the original image from a RealGalaxy using the real_catalog input item.
    """
    gal, safe = _BuildRealGalaxy(config, base, ignore, gsparams, logger,
                                   param_name='RealGalaxyOriginal')
    return gal.original_gal, safe


# Register these as valid gsobject types
from .gsobject import RegisterObjectType
RegisterObjectType('RealGalaxy', _BuildRealGalaxy, input_type='real_catalog')
RegisterObjectType('RealGalaxyOriginal', _BuildRealGalaxyOriginal, input_type='real_catalog')
