# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

import numpy as np
import math

from . import _galsim
from .gsobject import GSObject
from .gsparams import GSParams
from .utilities import doc_inherit


class DeltaFunction(GSObject):
    """A class describing a DeltaFunction surface brightness profile.

    The DeltaFunction surface brightness profile is characterized by a single property,
    its ``flux``.

    A DeltaFunction can be initialized with a specified flux.

    Parameters:
        flux:       The flux (in photons/cm^2/s) of the profile. [default: 1]
        gsparams:   An optional `GSParams` argument. [default: None]
    """
    _opt_params = { "flux" : float }

    _mock_inf = 1.e300  # Some arbitrary very large number to use when we need infinity.

    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, flux=1., gsparams=None):
        self._gsparams = GSParams.check(gsparams)
        self._flux = float(flux)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, DeltaFunction) and
                 self.flux == other.flux and
                 self.gsparams == other.gsparams))

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

    @property
    def _maxk(self):
        return DeltaFunction._mock_inf

    @property
    def _stepk(self):
        return DeltaFunction._mock_inf

    @property
    def _max_sb(self):
        return DeltaFunction._mock_inf

    def _xValue(self, pos):
        if pos.x == 0. and pos.y == 0.:
            return DeltaFunction._mock_inf
        else:
            return 0.

    def _kValue(self, kpos):
        return self.flux

    def _shoot(self, photons, rng):
        flux_per_photon = self.flux / len(photons)
        photons.x = 0.
        photons.y = 0.
        photons.flux = flux_per_photon

    def _drawKImage(self, image, jac=None):
        image.array[:,:] = self.flux

    @doc_inherit
    def withFlux(self, flux):
        return DeltaFunction(flux=flux, gsparams=self.gsparams)
