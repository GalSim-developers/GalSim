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
import logging
import numpy as np

from .generative_model import GenerativeGalaxyModel

class GenerativeModelLoader(galsim.config.InputLoader):

    def getKwargs(self, config, base, logger):
        """Parse the config dict and return the kwargs needed to build the input object.

        The default implementation looks for special class attributes called:

            _req_params     A dict of required parameters and their types.
            _opt_params     A dict of optional parameters and their types.
            _single_params  A list of dicts of parameters such that one and only one of
                            parameter in each dict is required.
            _takes_rng      A bool value saying whether an rng object is required.

        See galsim.Catalog for an example of a class that sets these attributes.

        In addition to the kwargs, we also return a bool value, safe, that indicates whether
        the constructed object will be safe to keep around for multiple files (True) of if
        it will need to be rebuilt for each output file (False).

        @param config       The config dict for this input item
        @param base         The base config dict
        @param logger       If given, a logger object to log progress. [default: None]

        @returns kwargs, safe
        """
        req = GenerativeGalaxyModel._req_params
        opt = GenerativeGalaxyModel._opt_params
        single = GenerativeGalaxyModel._single_params
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, single=single)
        return kwargs, safe

galsim.config.RegisterInputType('generative_model', GenerativeModelLoader(GenerativeGalaxyModel))


def _SampleGalaxy(config, base, ignore, gsparams, logger, param_name='GenerativeModelGalaxy'):
    """
    Samples a galaxy from a generative model
    """
    model = galsim.config.GetInputObj('generative_model', config, base, param_name)

    kwargs, safe = galsim.config.GetAllParams(config, base,
        req = model.sample_req_params,
        opt = model.sample_opt_params,
        single = model.sample_single_params,
        ignore = ignore)

    # Creates a recarray with just one entry
    t = []
    for q in model.quantities:
        t.append(kwargs[q])
    cat_entry = np.array([tuple(t)], dtype=[(q, float) for q in model.quantities])

    # Sample from model
    gal = model.sample(cat_entry)

    return gal, safe

# Register this as a valid  gsobject type
galsim.config.RegisterObjectType('GenerativeModelGalaxy', _SampleGalaxy, input_type='generative_model')
