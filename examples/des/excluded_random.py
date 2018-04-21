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

# This file defines a custom value type called ExcludedRandom that selects random integers
# from a range, but excluding particular values.

import galsim
import math

#
# Define the ExcludedRandom type for integer values.
#

def GenExcludedRandom(config, base, value_type):
    """Generate a random integer within some range, but excluding some given values.
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for type = ExcludedRandom")

    rng = base['rng']
    ud = galsim.UniformDeviate(rng)

    req = { 'min' : int, 'max' : int }
    ignore = [ 'exclude' ]  # We handle this separately.  Just tell GetAllParams not to
                            # raise an exception when it sees this.
    params, safe = galsim.config.GetAllParams(config, base, req=req, ignore=ignore)

    min = params['min']
    max = params['max']

    if 'exclude' in config:
        exclude = config['exclude']
    else:
        exclude = []

    # Also exclude max+1 just in case ud() == 1.
    exclude.append(max+1)

    # Draw values until we get one not in the exclude list.
    while True:
        val = int(math.floor(ud() * (max-min+1))) + min
        if val not in exclude:
            break

    return val, False

galsim.config.RegisterValueType('ExcludedRandom', GenExcludedRandom, [ int ])


