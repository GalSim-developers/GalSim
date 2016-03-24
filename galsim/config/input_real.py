# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
    # But not if they specify 'id' which overrides that.
    if 'id' not in config:
        galsim.config.SetDefaultIndex(config, real_cat.getNObjects())

    kwargs, safe = galsim.config.GetAllParams(config, base,
        req = galsim.__dict__['RealGalaxy']._req_params,
        opt = galsim.__dict__['RealGalaxy']._opt_params,
        single = galsim.__dict__['RealGalaxy']._single_params,
        ignore = ignore + ['num'])
    if gsparams: kwargs['gsparams'] = galsim.GSParams(**gsparams)

    if 'rng' not in base:
        raise ValueError("No base['rng'] available for type = %s"%param_name)
    kwargs['rng'] = base['rng']

    if 'index' in kwargs:
        index = kwargs['index']
        if index >= real_cat.getNObjects():
            raise IndexError(
                "%s index has gone past the number of entries in the catalog"%index)

    kwargs['real_galaxy_catalog'] = real_cat
    if logger:
        logger.debug('obj %d: %s kwargs = %s',base['obj_num'],param_name,kwargs)

    gal = galsim.RealGalaxy(**kwargs)

    return gal, safe


def _BuildRealGalaxyOriginal(config, base, ignore, gsparams, logger):
    """@brief Return the original image from a RealGalaxy using the real_catalog input item.
    """
    image, safe = _BuildRealGalaxy(config, base, ignore, gsparams, logger,
                                   param_name='RealGalaxyOriginal')
    return image.original_image, safe


# Register these as valid gsobject types
from .gsobject import RegisterObjectType
RegisterObjectType('RealGalaxy', _BuildRealGalaxy, input_type='real_catalog')
RegisterObjectType('RealGalaxyOriginal', _BuildRealGalaxyOriginal, input_type='real_catalog')
