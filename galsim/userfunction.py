# Copyright (c) 2012-2026 by the GalSim developers team on GitHub
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
from .utilities import doc_inherit, math_eval


class UserFunction(GSObject):
    """A class describing a surface brightness profile given by an arbitrary user-provided
    function.

    The function may be provided either as a string or as a real python function.

    If the function is a string, it is evaluated as ``lambda u,v: func``.  The eval may use
    numpy or math module commands if needed, but it cannot be more complicated than what is
    possible in a lambda function.  One advantage of this approach is that the resulting
    instance will be picklable and hashable, which may be desirable is some situations.

    Ideally, the function should be written in such a way that the u,v arguments may be numpy
    arrays, where the function returns an array of intensity values I(u,v) for each position.
    If this is not possible, then the evaluation will proceed one point at a time, which tends
    to be significantly slower.

    Parameters:
        func:       The function I(u,v) describing the surface brightness profile in sky
                    coordinates (u,v).
        gsparams:   An optional `GSParams` argument. [default: None]
    """
    _req_params = { "func" : str }

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = False

    def __init__(self, func, gsparams=None):
        self._gsparams = GSParams.check(gsparams)
        self._orig_func = func
        self._initialize_func()

    def _initialize_func(self):
        if isinstance(self._orig_func, str):
            self._func = math_eval('lambda u,v : ' + self._orig_func)
        else:
            self._func = self._orig_func

    def func(self):
        return self._func

    def sb(self, u, v):
        try:
            return self._func(u,v)
        except Exception as e:
            # If that didn't work, do each point separately.
            try:
                return np.array([self._func(u1,v1) for (u1,v1) in zip(u,v)])
            except Exception:
                # If this also fails, raise the original error, since it's probably more relevant.
                raise e from None

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, UserFunction) and
                 self._orig_func == other._orig_func and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.UserFunction", self._orig_func, self.gsparams))

    def __repr__(self):
        return 'galsim.UserFunction(func=%r, gsparams=%r)'%(self._orig_func, self.gsparams)

    def __str__(self):
        return 'galsim.UserFunction(%s)'%self._orig_func

    # TODO:
    @property
    def _maxk(self):
        return UserFunction._mock_inf

    @property
    def _stepk(self):
        return UserFunction._mock_inf

    @property
    def _max_sb(self):
        return UserFunction._mock_inf

    def _xValue(self, pos):
        if isinstance(pos, galsim.Position):
            from .deprecated import depr
            depr("_xValue(pos)", 2.8, "_xValue(x,y)")
            x,y = pos.x, pos.y
        else:
            x,y = pos
        return self._sbp.xValue(x,y)

    def _kValue(self, kpos):
        if isinstance(kpos, galsim.Position):
            from .deprecated import depr
            depr("_kValue(kpos)", 2.8, "_kValue(x,y)")
            kx,ky = kpos.x, kpos.y
        else:
            kx,ky = kpos
        return self._sbp.kValue(kx,ky)

    def _xValue(self, pos):
        if pos.x == 0. and pos.y == 0.:
            return UserFunction._mock_inf
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
        return UserFunction(flux=flux, gsparams=self.gsparams)
