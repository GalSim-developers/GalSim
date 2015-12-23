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
import logging


def _BuildRealGalaxy(config, base, ignore, gsparams, logger):
    """@brief Build a RealGalaxy from the real_catalog input item.
    """
    if 'real_catalog' not in base['input_objs']:
        raise ValueError("No real galaxy catalog available for building type = RealGalaxy")

    if 'num' in config:
        num, safe = ParseValue(config, 'num', base, int)
    else:
        num, safe = (0, True)
    ignore.append('num')

    if num < 0:
        raise ValueError("Invalid num < 0 supplied for RealGalaxy: num = %d"%num)
    if num >= len(base['input_objs']['real_catalog']):
        raise ValueError("Invalid num supplied for RealGalaxy (too large): num = %d"%num)

    real_cat = base['input_objs']['real_catalog'][num]

    # Special: if index is Sequence or Random, and max isn't set, set it to nobjects-1.
    # But not if they specify 'id' which overrides that.
    if 'id' not in config:
        galsim.config.SetDefaultIndex(config, real_cat.getNObjects())

    kwargs, safe1 = galsim.config.GetAllParams(config, base, 
        req = galsim.__dict__['RealGalaxy']._req_params,
        opt = galsim.__dict__['RealGalaxy']._opt_params,
        single = galsim.__dict__['RealGalaxy']._single_params,
        ignore = ignore)
    safe = safe and safe1
    if gsparams: kwargs['gsparams'] = galsim.GSParams(**gsparams)

    if 'rng' not in base:
        raise ValueError("No base['rng'] available for RealGalaxy")
    kwargs['rng'] = base['rng']

    if 'index' in kwargs:
        index = kwargs['index']
        if index >= real_cat.getNObjects():
            raise IndexError(
                "%s index has gone past the number of entries in the catalog"%index)

    kwargs['real_galaxy_catalog'] = real_cat
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('obj %d: RealGalaxy kwargs = %s',base['obj_num'],str(kwargs))

    gal = galsim.RealGalaxy(**kwargs)

    return gal, safe


def _BuildRealGalaxyOriginal(config, base, ignore, gsparams, logger):
    """@brief Return the original image from a RealGalaxy using the real_catalog input item.
    """
    image, safe = _BuildRealGalaxy(config, base, ignore, gsparams, logger)
    return image.original_image, safe    


