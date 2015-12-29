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


def _BuildRing(config, base, ignore, gsparams, logger):
    """@brief  Build a GSObject in a Ring.
    """
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
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('obj %d: Ring dtheta = %f',base['obj_num'],dtheta.rad())

    if index % num == 0:
        # Then this is the first in the Ring.  
        gsobject = galsim.config.BuildGSObject(config, 'first', base, gsparams, logger)[0]
    else:
        if not isinstance(config['first'],dict) or 'current_val' not in config['first']:
            raise RuntimeError("Building Ring after the first item, but no current_val stored.")
        gsobject = config['first']['current_val'].rotate(index*dtheta)

    return gsobject, False

