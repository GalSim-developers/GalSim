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
import galsim.hsm

def HSM_Shape_Measure(config, base, value_type):
    """@brief Measure a specified profile's shape with hsm
    """
    req = { 'obj' : str }
    params, safe = galsim.config.GetAllParams(config, base, req=req)

    obj_key = config['obj']
    obj = galsim.config.GetCurrentValue(obj_key, base)

    im = obj.drawImage()
    try:
        shape = im.FindAdaptiveMom().observed_shape
    except Exception as e:
        shape = galsim.Shear(-99 - 99j)
    return shape, False

galsim.config.RegisterValueType('HSM_Shape_Measure', HSM_Shape_Measure, [ galsim.Shear ])
