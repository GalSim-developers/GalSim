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

from . import _galsim
from .gsobject import GSObject


class DeltaFunction(GSObject):
    """A class describing a DeltaFunction surface brightness profile.

    The DeltaFunction surface brightness profile is characterized by a single property,
    its `flux'.

    Initialization
    --------------

    A DeltaFunction can be initialized with a specified flux.

    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    DeltaFunction simply has the usual GSObject properties.
    """
    # Initialization parameters of the object, with type information, to indicate
    # which attributes are allowed / required in a config file for this object.
    # _req_params are required
    # _opt_params are optional
    # _single_params are a list of sets for which exactly one in the list is required.
    # _takes_rng indicates whether the constructor should be given the current rng.
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False

    def __init__(self, flux=1., gsparams=None):
        self._gsparams = galsim.GSParams.check(gsparams)
        self._sbp = _galsim.SBDeltaFunction(flux, self.gsparams._gsp)

    def __eq__(self, other):
        return (isinstance(other, galsim.DeltaFunction) and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.DeltaFunction", self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.DeltaFunction(flux=%r, gsparams=%r)'%(self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.DeltaFunction('
        if self.flux != 1.0:
            s += 'flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBDeltaFunction.__getinitargs__ = lambda self: (
        self.getFlux(), self.getGSParams())
