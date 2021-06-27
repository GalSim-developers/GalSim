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

from . import _galsim
from .gsobject import GSObject
from .gsparams import GSParams
from .utilities import lazy_property, doc_inherit
from .angle import arcsec, AngleUnit, radians
from .deltafunction import DeltaFunction

class SecondKick(GSObject):
    """Class describing the expectation value of the high-k turbulence portion of an atmospheric
    PSF convolved by an `Airy` PSF.

    The power spectrum of atmospheric phase fluctuations is assumed to follow the von Karman
    prescription, but possibly modified by the addition of a critical scale below which the power
    is zero.  (See the `VonKarman` docstring for more details).

    As an expectation value, this profile is formally only exact in the infinite-exposure limit.
    However, at least for large apertures, we have found that this expectation value is approached
    rapidly, and can be applied for even fairly small exposure times.

    The intended use for this profile is as a correction to applying the geometric approximation to
    `PhaseScreenPSF` objects when drawing using geometric photon shooting. In this case, the
    `PhaseScreenPSF` will simulate the effects of the low frequency turbulence modes, which can be
    treated purely using refraction, while the SecondKick handles the high frequency modes.

    The geometric approximation is only valid for length scales larger than some critical scale
    where the effects of interference are unimportant.  For smaller length scales, interference
    (diffraction) must be handled using an optical paradigm that acknowledges the wave nature of
    light, such as Fourier optics.

    Fourier optics calculations are many orders of magnitude slower than geometric optics
    calculations for typical flux levels, however, so we implement a scale-splitting algorithm first
    described in Peterson et al. (2015) for the LSST PhoSim package.  Essentially, phase
    fluctuations below a critical mode in Fourier space, labeled ``kcrit``, are handled by the fast
    geometric optics calculations present in `PhaseScreenPSF`.  Fluctuations for Fourier modes above
    ``kcrit`` are then calculated analytically by SecondKick.  Because very many oscillations of
    these high-k modes both fit within a given telescope aperture and pass by the aperture during a
    moderate length exposure time, we can use the same analytic expectation value calculation for
    the high-k component of all PSFs across a field of view, thus incurring the somewhat expensive
    calculation for Fourier optics only once.

    There are two limiting cases for this profile that may helpful for readers trying to understand
    how this class works.  When kcrit = 0, then all turbulent modes are included, and this surface
    brightness profile becomes identical to the convolution of an `Airy` profile and a Von Karman
    profile.  In contrast, when kcrit = inf, then none of the turbulent modes are included, and this
    surface brightness profile is just an `Airy` profile.  In other words, the full effect of an
    `Airy` profile, and additionally some portion (which depends on kcrit) of a `VonKarman` profile
    are modeled.

    For more details, we refer the reader to the original implementation described in

        Peterson et al.  2015  ApJSS  vol. 218

    Parameters:
        lam:            Wavelength in nanometers
        r0:             Fried parameter in meters.
        diam:           Aperture diameter in meters.
        obscuration:    Linear dimension of central obscuration as fraction of aperture
                        linear dimension. [0., 1.).  [default: 0.0]
        kcrit:          Critical Fourier mode (in units of 1/r0) below which the turbulence
                        power spectrum will be truncated.  [default: 0.2]
        flux:           The flux (in photons/cm^2/s) of the profile. [default: 1]
        scale_unit:     Units assumed when drawing this profile or evaluating xValue, kValue,
                        etc.  Should be a `galsim.AngleUnit` or a string that can be used to
                        construct one (e.g., 'arcsec', 'radians', etc.). [default: galsim.arcsec]
        gsparams:       An optional `GSParams` argument. [default: None]
    """
    _req_params = { "lam" : float, "r0" : float, "diam" : float }
    _opt_params = { "obscuration" : float, "kcrit" : float, "flux" : float, "scale_unit" : str }

    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, lam, r0, diam, obscuration=0, kcrit=0.2, flux=1,
                 scale_unit=arcsec, gsparams=None):
        if isinstance(scale_unit, str):
            self._scale_unit = AngleUnit.from_name(scale_unit)
        else:
            self._scale_unit = scale_unit
        self._scale = radians / self._scale_unit

        self._flux = float(flux)
        self._r0 = float(r0)
        self._lam = float(lam)
        self._diam = float(diam)
        self._obscuration = float(obscuration)
        self._kcrit = float(kcrit)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _sbs(self):
        lam_over_r0 = (1.e-9*self._lam/self._r0)*self._scale
        return _galsim.SBSecondKick(lam_over_r0, self._kcrit, self._flux, self._gsparams._gsp)

    @lazy_property
    def _sba(self):
        lam_over_diam = (1.e-9*self._lam/self._diam)*self._scale
        return _galsim.SBAiry(lam_over_diam, self._obscuration, 1., self._gsparams._gsp)

    @lazy_property
    def _sbd(self):
        return _galsim.SBDeltaFunction(self._sbs.getDelta(), self._gsparams._gsp)

    @lazy_property
    def _sbp(self):
        full_sbs = _galsim.SBAdd([self._sbs, self._sbd], self._gsparams._gsp)
        return _galsim.SBConvolve([full_sbs, self._sba], False, self._gsparams._gsp)

    @property
    def lam(self):
        """The input lam value.
        """
        return self._lam

    @property
    def r0(self):
        """The input r0 value.
        """
        return self._r0;

    @property
    def diam(self):
        """The input diam value.
        """
        return self._diam;

    @property
    def obscuration(self):
        """The input obscuration value.
        """
        return self._obscuration;

    @property
    def kcrit(self):
        """The input kcrit value.
        """
        return self._kcrit;

    @property
    def scale_unit(self):
        """The input scale_unit value.
        """
        return self._scale_unit

    def _structure_function(self, rho):
        return self._sbs.structureFunction(rho)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, SecondKick) and
                 self.lam == other.lam and
                 self.r0 == other.r0 and
                 self.diam == other.diam and
                 self.obscuration == other.obscuration and
                 self.kcrit == other.kcrit and
                 self.flux == other.flux and
                 self.scale_unit == other.scale_unit and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.SecondKick", self.lam, self.r0, self.diam, self.obscuration,
                     self.kcrit, self.flux, self.scale_unit, self.gsparams))

    def __repr__(self):
        out = "galsim.SecondKick("
        out += "lam=%r"%self.lam
        out += ", r0=%r"%self.r0
        out += ", diam=%r"%self.diam
        if self.obscuration != 0.0:
            out += ", obscuration=%r"%self.obscuration
        out += ", kcrit=%r"%self.kcrit
        if self.flux != 1:
            out += ", flux=%r"%self.flux
        if self.scale_unit != arcsec:
            out += ", scale_unit=%r"%self.scale_unit
        out += ", gsparams=%r)"%self.gsparams
        return out

    def __str__(self):
        return "galsim.SecondKick(lam=%r, r0=%r, kcrit=%r)"%(self.lam, self.r0, self.kcrit)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        d.pop('_sba',None)
        d.pop('_sbd',None)
        d.pop('_sbs',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        return self._sbp.maxK()

    @property
    def _stepk(self):
        return self._sbp.stepK()

    @property
    def _max_sb(self):
        return DeltaFunction._mock_inf

    def _xValue(self, pos):
        return self._sbp.xValue(pos._p)

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _shoot(self, photons, rng):
        self._sbp.shoot(photons._pa, rng._rng)

    def _drawKImage(self, image, jac=None):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.drawK(image._image, image.scale, _jac)

    @doc_inherit
    def withFlux(self, flux):
        return SecondKick(lam=self.lam, r0=self.r0, diam=self.diam, obscuration=self.obscuration,
                          kcrit=self.kcrit, flux=flux, scale_unit=self.scale_unit,
                          gsparams=self.gsparams)
