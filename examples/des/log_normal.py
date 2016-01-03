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
import numpy

# This uses the ngmix code, downloadable from here:
# https://github.com/esheldon/ngmix
import ngmix

# Specifically, we use the LogNormal class here:
# https://github.com/esheldon/ngmix/blob/master/ngmix/priors.py#L2832

# The only thing we need to be careful about is the random number generator.  
# Erin's code uses numpy.random, not a GalSim rng.  So we use the GalSim rng to 
# seed numpy.random.  This should produce deterministic results.

def GenLogNormal(config, base, value_type):
    """Generate a random number from a log-normal distribution.
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for type = LogNormal")

    rng = base['rng']

    req = { 'mean' : float, 'sigma' : float }
    params, safe = galsim.config.GetAllParams(config, base, req=req)

    mean = params['mean']
    sigma = params['sigma']

    lgn = ngmix.priors.LogNormal(mean, sigma)
    seed = rng.raw()
    numpy.random.seed(seed)
    value = lgn.sample()

    return value, False

galsim.config.RegisterValueType('LogNormal', GenLogNormal, [ float ])
