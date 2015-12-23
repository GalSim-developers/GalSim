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


def _BuildCOSMOSGalaxy(config, key, base, ignore, gsparams, logger):
    """@brief Build a COSMOS galaxy using the cosmos_catalog input item.
    """
    if 'cosmos_catalog' not in base['input_objs']:
        raise ValueError("No COSMOS galaxy catalog available for building type = COSMOSGalaxy")

    if 'num' in config:
        num, safe = ParseValue(config, 'num', base, int)
    else:
        num, safe = (0, True)
    ignore.append('num')

    if num < 0:
        raise ValueError("Invalid num < 0 supplied for COSMOSGalaxy: num = %d"%num)
    if num >= len(base['input_objs']['cosmos_catalog']):
        raise ValueError("Invalid num supplied for COSMOSGalaxy (too large): num = %d"%num)

    cosmos_cat = base['input_objs']['cosmos_catalog'][num]

    # Special: if index is Sequence or Random, and max isn't set, set it to nobjects-1.
    galsim.config.SetDefaultIndex(config, cosmos_cat.getNObjects())

    kwargs, safe1 = galsim.config.GetAllParams(config, key, base,
        req = galsim.COSMOSCatalog.makeGalaxy._req_params,
        opt = galsim.COSMOSCatalog.makeGalaxy._opt_params,
        single = galsim.COSMOSCatalog.makeGalaxy._single_params,
        ignore = ignore)
    safe = safe and safe1
    if gsparams: kwargs['gsparams'] = galsim.GSParams(**gsparams)

    if 'gal_type' in kwargs and kwargs['gal_type'] == 'real':
        if 'rng' not in base:
            raise ValueError("No base['rng'] available for %s.type = COSMOSGalaxy"%(key))
        kwargs['rng'] = base['rng']

    if 'index' in kwargs:
        index = kwargs['index']
        if index >= cosmos_cat.getNObjects():
            raise IndexError(
                "%s index has gone past the number of entries in the catalog"%index)

    if logger:
        logger.debug('obj %d: COSMOSGalaxy kwargs = %s',base['obj_num'],str(kwargs))

    kwargs['cosmos_catalog'] = cosmos_cat

    # Use a staticmethod of COSMOSCatalog to avoid pickling the result of makeGalaxy()
    # The RealGalaxy in particular has a large serialization, so it is more efficient to
    # make it in this process, which is what happens here.
    gal = galsim.COSMOSCatalog._makeSingleGalaxy(**kwargs)

    return gal, safe


