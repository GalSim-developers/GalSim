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
"""@file second_kick.py

This file implements the atmospheric PSF "second kick", which is an analytic calculation for the
surface brightness profile of an atmospheric PSF convolved with an Airy PSF in the infinite exposure
limit.  The atmospheric turbulence spectrum follows the von Karman formulation with an optional
truncation scale.  Generally, this will be used in conjunction with a PhaseScreenPSF being drawn
using geometric photon-shooting.  In this case, the PhaseScreenPSF will simulate the effects of the
low frequency turbulence modes, which can be treated purely using refraction, while the SecondKick
handles the high frequency modes."""

import numpy as np

import galsim

from . import _galsim
from .gsobject import GSObject


class SecondKick(GSObject):
    """Class describing the expectation value of the high-k turbulence portion of an atmospheric PSF
    convolved by an Airy PSF.  The power spectrum of atmospheric phase fluctuations is assumed to
    follow the von Karman prescription, but possibly modified by the addition of a critical scale
    below which the power is zero.  (See the VonKarman docstring for more details).  As an
    expectation value, this profile is formally only exact in the infinite-exposure limit.  However,
    at least for large apertures, we have found that this expectation value is approached rapidly,
    and can be applied for even fairly small exposure times.

    The intended use for this profile is as a correction to applying the geometric approximation to
    PhaseScreenPSF objects when drawing using geometric photon shooting.  The geometric
    approximation is only valid for length scales larger than some critical scale where the effects
    of interference are unimportant.  For smaller length scales, interference (diffraction) must be
    handled using an optical paradigm that acknowledges the wave nature of light, such as Fourier
    optics.

    Fourier optics calculations are many orders of magnitude slower than geometric optics
    calculations for typical flux levels, however, so we implement a scale-splitting algorithm first
    described in Peterson et al. (2015) for the LSST PhoSim package.  Essentially, phase
    fluctuations below a critical mode in Fourier space, labeled `kcrit`, are handled by the fast
    geometric optics calculations present in PhaseScreenPSF.  Fluctuations for Fourier modes above
    `kcrit` are then calculated analytically by SecondKick.  Because very many oscillations of these
    high-k modes both fit within a given telescope aperture and pass by the aperture during a
    moderate length exposure time, we can use the same analytic expectation value calculation for
    the high-k component of all PSFs across a field of view, thus incurring the somewhat expensive
    calculation for Fourier optics only once.

    There are two limiting cases for this profile that may helpful for readers trying to understand
    how this class works.  When kcrit = 0, then all turbulent modes are included, and this surface
    brightness profile becomes identical to the convolution of an Airy profile and a Von Karman
    profile.  In contrast, when kcrit = inf, then none of the turbulent modes are included, and this
    surface brightness profile is just an Airy profile.  In other words, the full effect of an Airy
    profile, and additionally some portion (which depends on kcrit) of a VonKarman profile are
    modeled.

    For more details, we refer the reader to the original implementation described in

        Peterson et al.  2015  ApJSS  vol. 218

    @param lam               Wavelength in nanometers
    @param r0                Fried parameter in meters.
    @param diam              Aperture diameter in meters.
    @param obscuration       Linear dimension of central obscuration as fraction of aperture
                             linear dimension. [0., 1.).  [default: 0.0]
    @param L0                Outer scale in meters.  [default: 25.0]
    @param kcrit             Critical Fourier mode (in units of 1/r0) below which the turbulence
                             power spectrum will be truncated.  [default: 0.2]
    @param flux              The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param scale_unit        Units assumed when drawing this profile or evaluating xValue, kValue,
                             etc.  Should be a galsim.AngleUnit or a string that can be used to
                             construct one (e.g., 'arcsec', 'radians', etc.).
                             [default: galsim.arcsec]
    @param gsparams          An optional GSParams argument.  See the docstring for GSParams for
                             details. [default: None]
    """
    def __init__(self, lam, r0, diam, obscuration=0, L0=25.0, kcrit=0.2, flux=1,
                 scale_unit=galsim.arcsec, gsparams=None):
        # We lose stability if L0 gets too large.  This should be close enough to infinity for
        # all practical purposes though.
        if L0 > 1e10:
            L0 = 1e10
        if isinstance(scale_unit, str):
            scale_unit = galsim.angle.get_angle_unit(scale_unit)
        # Need _scale_unit for repr roundtriping.
        self._scale_unit = scale_unit
        scale = scale_unit/galsim.arcsec
        self._sbp = galsim._galsim.SBSecondKick(
            lam, r0, diam, obscuration, L0, kcrit, flux, scale, gsparams)

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
        return (isinstance(other, galsim.SecondKick) and
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
        return hash(("galsim.SecondKick", self.lam, self.r0, self.diam, self.obscuration, self.L0,
                     self.kcrit, self.flux, self.scale_unit, self.gsparams))

    def __repr__(self):
        out = "galsim.SecondKick("
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
        return "galsim.SecondKick(lam=%r, r0=%r, kcrit=%r)"%(self.lam, self.r0, self.kcrit)

_galsim.SBSecondKick.__getinitargs__ = lambda self: (
    self.getLam(), self.getR0(), self.getDiam(), self.getObscuration(), self.getL0(),
    self.getKCrit(), self.getFlux(), self.getScale(), self.getGSParams())
_galsim.SBSecondKick.__getstate__ = lambda self: None
_galsim.SBSecondKick.__repr__ = lambda self: \
    "galsim._galsim.SBSecondKick(%r, %r, %r, %r, %r, %r, %r, %r, %r)"%self.__getinitargs__()
