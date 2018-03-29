# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
import galsim.config
from galsim.deprecated import depr

# This file adds gsobject type Ring which builds an object once every n times, and then
# rotates it in a ring for the other n-1 times per per group.

def _BuildRing(config, base, ignore, gsparams, logger):
    """@brief  Build a GSObject in a Ring.  Now deprecated.
    """
    depr('gal.type = Ring', 1.4, 'stamp.type = Ring',
         'The galaxy Ring type may not work properly in conjunction with image.nproc != 1. '+
         'See demo5 and demo10 for examples of the new stamp type=Ring syntax.')

    req = { 'num' : int, 'first' : dict }
    opt = { 'full_rotation' : galsim.Angle , 'index' : int }
    # Only Check, not Get.  We need to handle first a bit differently, since it's a gsobject.
    galsim.config.CheckAllParams(config, req=req, opt=opt, ignore=ignore)

    num = galsim.config.ParseValue(config, 'num', base, int)[0]
    if num <= 0:
        raise ValueError("Attribute num for gal.type == Ring must be > 0")

    # Setup the indexing sequence if it hasn't been specified using the number of items.
    galsim.config.SetDefaultIndex(config, num)
    index, safe = galsim.config.ParseValue(config, 'index', base, int)
    if index < 0 or index >= num:
        raise AttributeError("index %d out of bounds for config.%s"%(index,type))

    if 'full_rotation' in config:
        full_rotation = galsim.config.ParseValue(config, 'full_rotation', base, galsim.Angle)[0]
    else:
        import math
        full_rotation = math.pi * galsim.radians

    dtheta = full_rotation / num
    if logger:
        logger.debug('obj %d: Ring dtheta = %f',base['obj_num'],dtheta.rad)

    if index % num == 0:
        # Then this is the first in the Ring.
        gsobject = galsim.config.BuildGSObject(config, 'first', base, gsparams, logger)[0]
    else:
        if not isinstance(config['first'],dict) or 'current' not in config['first']:
            raise RuntimeError("Building Ring after the first item, but no current val stored.")
        gsobject = config['first']['current'][0].rotate(index*dtheta)

    return gsobject, False

# Register this as a valid gsobject type
galsim.config.RegisterObjectType('Ring', _BuildRing, _is_block=True)
