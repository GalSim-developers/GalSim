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
"""@file base.py
This file implements the von Karman atmospheric PSF profile.  A version which has the underlying
turbulence power spectrum truncated below a given scale is also available as a correction when using
geometric shooting through an atmospheric PhaseScreenPSF.
"""

import numpy as np

import galsim

from . import _galsim
from .gsobject import GSObject


class VonKarman(GSObject):
    """Class describing a von Karman surface brightness profile, which represents a long exposure
    atmospheric PSF.  The difference between the von Karman profile and the related Kolmogorov
    profile is that the von Karman profile includes a parameter for the outer scale of atmospheric
    turbulence, which is a physical scale beyond which fluctuations in the refractive index stop
    growing, typically between 10 and 100 meters.  Quantitatively, the von Karman phase fluctuation
    power spectrum is proportional to

        (f^2 + L0^-2)^(-11/6)

    where f is a spatial frequency and L0 is the outer scale in meters.  The Kolmogorov power
    spectrum proportional to f^(-11/3) is recovered as L0 -> infinity,

    For more information, we recommend the following references:

        Martinez et al.  2010  A&A  vol. 516
        Conan  2008  JOSA  vol. 25

    Notes
    -----

    If one blindly follows the math for converting the von Karman power spectrum into a PSF, one
    finds that the PSF contains a delta-function at the origin with fractional flux of

        exp(-0.5*0.172*(r0/L0)^(-5/3))

    In almost all cases of interest this evaluates to something tiny, often on the order of 10^-100
    or smaller.  By default, GalSim will ignore this delta function entirely.  If for some reason
    you want to keep the delta function, though, then you can pass the do_delta=True argument to the
    VonKarman initializer.

    @param lam               Wavelength in nanometers
    @param r0                Fried parameter in meters.
    @param L0                Outer scale in meters.  [default: np.inf]
    @param flux              The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param scale_unit        Units assumed when drawing this profile or evaluating xValue, kValue,
                             etc.  Should be a galsim.AngleUnit or a string that can be used to
                             construct one (e.g., 'arcsec', 'radians', etc.).
                             [default: galsim.arcsec]
    @param do_delta          Include delta-function at origin? (not recommended; see above).
                             [default: False]
    @param suppress_warning  For some combinations of r0 and L0, it may become impossible to satisfy
                             the gsparams.maxk_threshold criterion (see above).  In that case, the
                             code will emit a warning alerting the user that they may have entered a
                             non-physical regime.  However, this warning can be suppressed with this
                             keyword.  [default: False]
    @param gsparams          An optional GSParams argument.  See the docstring for GSParams for
                             details. [default: None]
    """
    def __init__(self, lam, r0, L0=np.inf, flux=1, scale_unit=galsim.arcsec,
                 do_delta=False, suppress_warning=False, gsparams=None):
        # We lose stability if L0 gets too large.  This should be close enough to infinity for
        # all practical purposes though.
        if L0 > 1e10:
            L0 = 1e10
        if isinstance(scale_unit, str):
            scale_unit = galsim.angle.get_angle_unit(scale_unit)
        # Need _scale_unit for repr roundtriping.
        self._scale_unit = scale_unit
        scale = scale_unit/galsim.arcsec
        self._sbvk = _galsim.SBVonKarman(lam, r0, L0, flux, scale, do_delta, gsparams)
        self._delta_amplitude = self._sbvk.getDeltaAmplitude()
        self._suppress_warning = suppress_warning
        if not suppress_warning:
            if self._delta_amplitude > self._sbvk.getGSParams().maxk_threshold:
                import warnings
                warnings.warn("VonKarman delta-function component is larger than maxk_threshold.  "
                              "Please see docstring for information about this component and how to"
                              " toggle it.")


        # Add in a delta function with appropriate amplitude if requested.
        if do_delta:
            self._sbdelta = _galsim.SBDeltaFunction(self._delta_amplitude, gsparams=gsparams)
            # A bit wasteful maybe, but params should be cached so not too bad to recreate _sbvk?
            self._sbvk = _galsim.SBVonKarman(lam, r0, L0, flux-self._delta_amplitude, scale,
                                             do_delta, gsparams)

            self._sbp = _galsim.SBAdd([self._sbvk, self._sbdelta], gsparams=gsparams)
        else:
            self._sbp = self._sbvk

    @property
    def lam(self):
        return self._sbvk.getLam()

    @property
    def r0(self):
        return self._sbvk.getR0()

    @property
    def L0(self):
        return self._sbvk.getL0()

    @property
    def scale_unit(self):
        return self._scale_unit
        # Type conversion makes the following not repr-roundtrip-able, so we store init input as a
        # hidden attribute.
        # return galsim.AngleUnit(self._sbvk.getScale())

    @property
    def do_delta(self):
        return self._sbvk.getDoDelta()

    @property
    def delta_amplitude(self):
        # Don't use self._sbvk.getDeltaAmplitude, since we might have rescaled the flux of the sbvk
        # component when including a SBDeltaFunction.
        return self._delta_amplitude

    @property
    def half_light_radius(self):
        return self._sbvk.getHalfLightRadius()

    def structureFunction(self, rho):
        return self._sbvk.structureFunction(rho)

    def __eq__(self, other):
        return (isinstance(other, galsim.VonKarman) and
        self.lam == other.lam and
        self.r0 == other.r0 and
        self.L0 == other.L0 and
        self.flux == other.flux and
        self.scale_unit == other.scale_unit and
        self.do_delta == other.do_delta and
        self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.VonKarman", self.lam, self.r0, self.L0, self.flux, self.scale_unit,
                     self.do_delta, self.gsparams))

    def __repr__(self):
        out = "galsim.VonKarman(lam=%r, r0=%r, L0=%r"%(self.lam, self.r0, self.L0)
        out += ", flux=%r"%self.flux
        if self.scale_unit != galsim.arcsec:
            out += ", scale_unit=%r"%self.scale_unit
        if self.do_delta:
            out += ", do_delta=True"
        out += ", gsparams=%r"%self.gsparams
        out += ")"
        return out

    def __str__(self):
        return "galsim.VonKarman(lam=%r, r0=%r, L0=%r)"%(self.lam, self.r0, self.L0)

_galsim.SBVonKarman.__getinitargs__ = lambda self: (
    self.getLam(), self.getR0(), self.getL0(), self.getFlux(), self.getScale(),
    self.getDoDelta(), self.getGSParams())
_galsim.SBVonKarman.__getstate__ = lambda self: None
_galsim.SBVonKarman.__repr__ = lambda self: \
    "galsim._galsim.SBVonKarman(%r, %r, %r, %r, %r, %r, %r)"%self.__getinitargs__()
