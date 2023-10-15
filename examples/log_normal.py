# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

def GenLogNormal(config, base, value_type):
    """
    Generate a random number from a log-normal distribution.
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for type = LogNormal")

    rng = galsim.config.GetRNG(config, base)

    req = { 'mean' : float, 'sigma' : float }
    params, safe = galsim.config.GetAllParams(config, base, req=req)

    mean = params['mean']
    sigma = params['sigma']

    logmean  = numpy.log(mean) - 0.5*numpy.log( 1 + sigma**2/mean**2 )
    logvar   = numpy.log(1 + sigma**2/mean**2 )
    logsigma = numpy.sqrt(logvar)
    gd = galsim.GaussianDeviate(rng, mean=logmean, sigma=logsigma)
    value = numpy.exp(gd())

    return value, False

galsim.config.RegisterValueType('LogNormal', GenLogNormal, [ float ])
