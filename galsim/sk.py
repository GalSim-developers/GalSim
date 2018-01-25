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
"""@file sk.py
This file implements the atmospheric PSF "second kick".
"""

import numpy as np

import galsim

from . import _galsim
from .gsobject import GSObject


class SK(GSObject):
    """
    """
    def __init__(self, lam, r0, diam, obscuration=0, L0=np.inf, kcrit=None, flux=1,
                 scale_unit=galsim.arcsec, gsparams=None):
        # We lose stability if L0 gets too large.  This should be close enough to infinity for
        # all practical purposes though.
        if L0 > 1e10:
            L0 = 1e10
        if isinstance(scale_unit, str):
            scale_unit = galsim.angle.get_angle_unit(scale_unit)
        if kcrit is None:
            kcrit = 2*np.pi/r0
        # Need _scale_unit for repr roundtriping.
        self._scale_unit = scale_unit
        scale = scale_unit/galsim.arcsec
        self._sbp = galsim._galsim.SBSK(lam, r0, diam, obscuration, L0, kcrit, flux, scale,
                                        gsparams)

    @property
    def lam(self):
        return self._sbp.getLam()

    @property
    def r0(self):
        return self._sbp.getR0()

    @property
    def diam(self):
        return self._sbp.getDiam()

    @property
    def obscuration(self):
        return self._sbp.getObscuration()

    @property
    def _sbairy(self):
        return self._sbp.getAiry()

    @property
    def L0(self):
        return self._sbp.getL0()

    @property
    def kcrit(self):
        return self._sbp.getKCrit()

    @property
    def scale_unit(self):
        return self._scale_unit
        # Type conversion makes the following not repr-roundtrip-able, so we store init input as a
        # hidden attribute.
        # return galsim.AngleUnit(self._sbvk.getScale())

    @property
    def half_light_radius(self):
        return self._sbp.getHalfLightRadius()

    def _structure_function(self, rho):
        return self._sbp.structureFunction(rho)

    def __eq__(self, other):
        return (isinstance(other, galsim.SK) and
        self.lam == other.lam and
        self.r0 == other.r0 and
        self.diam == other.diam and
        self.obscuration == other.obscuration and
        self.L0 == other.L0 and
        self.kcrit == other.kcrit and
        self.flux == other.flux and
        self.scale_unit == other.scale_unit and
        self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.SK", self.lam, self.r0, self.diam, self.obscuration, self.L0,
                     self.kcrit, self.flux, self.scale_unit, self.gsparams))

    def __repr__(self):
        out = "galsim.SK("
        out += "lam=%r"%self.lam
        out += ", r0=%r"%self.r0
        out += ", diam=%r"%self.diam
        if self.obscuration != 0.0:
            out += ", obscuration=%r"%self.obscuration
        out += ", L0=%r"%self.L0
        out += ", kcrit=%r"%self.kcrit
        if self.flux != 1:
            out += ", flux=%r"%self.flux
        if self.scale_unit != galsim.arcsec:
            out += ", scale_unit=%r"%self.scale_unit
        out += ", gsparams=%r)"%self.gsparams
        return out

    def __str__(self):
        return "galsim.SK(lam=%r, r0=%r, kcrit=%r)"%(self.lam, self.r0, self.kcrit)

_galsim.SBSK.__getinitargs__ = lambda self: (
    self.getLam(), self.getR0(), self.getDiam(), self.getObscuration(), self.getL0(),
    self.getKCrit(), self.getFlux(), self.getScale(), self.getGSParams())
# _galsim.SBSK.__getstate__ = lambda self: None
_galsim.SBSK.__repr__ = lambda self: \
    "galsim._galsim.SBSK(%r, %r, %r, %r, %r, %r, %r, %r, %r)"%self.__getinitargs__()
