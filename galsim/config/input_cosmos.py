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

# This file adds input type cosmos_catalog and gsobject typs COSMOSGalaxy.

# The COSMOSCatalog doesn't need anything special other than registration as a valid input type.
from .input import valid_input_types
valid_input_types['cosmos_catalog'] = (
    galsim.COSMOSCatalog, None, True, False, None, ['COSMOSGalaxy']
)

# The gsobject type coupled to this is COSMOSGalaxy.

def _BuildCOSMOSGalaxy(config, base, ignore, gsparams, logger):
    """@brief Build a COSMOS galaxy using the cosmos_catalog input item.
    """
    cosmos_cat = galsim.config.GetInputObj('cosmos_catalog', config, base, 'COSMOSGalaxy')

    ignore = ignore + ['num']

    # Special: if index is Sequence or Random, and max isn't set, set it to nobjects-1.
    galsim.config.SetDefaultIndex(config, cosmos_cat.getNObjects())

    kwargs, safe = galsim.config.GetAllParams(config, base,
        req = galsim.COSMOSCatalog.makeGalaxy._req_params,
        opt = galsim.COSMOSCatalog.makeGalaxy._opt_params,
        single = galsim.COSMOSCatalog.makeGalaxy._single_params,
        ignore = ignore)
    if gsparams: kwargs['gsparams'] = galsim.GSParams(**gsparams)

    if 'gal_type' in kwargs and kwargs['gal_type'] == 'real':
        if 'rng' not in base:
            raise ValueError("No base['rng'] available for type = COSMOSGalaxy")
        kwargs['rng'] = base['rng']

    if 'index' in kwargs:
        index = kwargs['index']
        if index >= cosmos_cat.getNObjects():
            raise IndexError(
                "%s index has gone past the number of entries in the catalog"%index)

    if False:
        logger.debug('obj %d: COSMOSGalaxy kwargs = %s',base['obj_num'],kwargs)

    kwargs['cosmos_catalog'] = cosmos_cat

    # Use a staticmethod of COSMOSCatalog to avoid pickling the result of makeGalaxy()
    # The RealGalaxy in particular has a large serialization, so it is more efficient to
    # make it in this process, which is what happens here.
    gal = galsim.COSMOSCatalog._makeSingleGalaxy(**kwargs)

    return gal, safe

# Register this as a valid gsobject type
from .gsobject import valid_gsobject_types
valid_gsobject_types['COSMOSGalaxy'] = _BuildCOSMOSGalaxy
