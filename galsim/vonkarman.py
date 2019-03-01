# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
This file implements the von Karman atmospheric PSF profile.
"""

import numpy as np

from . import _galsim
from .gsobject import GSObject
from .gsparams import GSParams
from .utilities import lazy_property, doc_inherit
from .position import PositionD
from .angle import arcsec, AngleUnit
from .errors import GalSimError, convert_cpp_errors, galsim_warn
from .errors import GalSimIncompatibleValuesError


class VonKarman(GSObject):
    """Class describing a von Karman surface brightness profile, which represents a long exposure
    atmospheric PSF.  The difference between the von Karman profile and the related Kolmogorov
    profile is that the von Karman profile includes a parameter for the outer scale of atmospheric
    turbulence, which is a physical scale beyond which fluctuations in the refractive index stop
    growing, typically between 10 and 100 meters.  Quantitatively, the von Karman phase fluctuation
    power spectrum at spatial frequency f is proportional to

        r0^(-5/3) (f^2 + L0^-2)^(-11/6)

    where r0 is the Fried parameter which sets the overall turbulence amplitude and L0 is the outer
    scale in meters.  The Kolmogorov power spectrum proportional to r0^(-5/3) f^(-11/3) is recovered
    as L0 -> infinity.

    For more information, we recommend the following references:

        Martinez et al.  2010  A&A  vol. 516
        Conan  2008  JOSA  vol. 25

    Notes
    -----

    If one blindly follows the math for converting the von Karman power spectrum into a PSF, one
    finds that the PSF contains a delta-function at the origin with fractional flux of

        exp(-0.5*0.172*(r0/L0)^(-5/3))

    In almost all cases of interest this evaluates to something tiny, often on the order of 10^-100
    or smaller.  By default, GalSim will ignore this delta function entirely since it usually
    doesn't make any difference, but can complicate some calculations like drawing using
    method='real_space' or by formally requiring huge Fourier transforms for drawing using
    method='fft'.  If for some reason you want to keep the delta function, though, then you can pass
    the do_delta=True argument to the VonKarman initializer.

    @param lam               Wavelength in nanometers.
    @param r0                Fried parameter at specified wavelength `lam` in meters.  Exactly one
                             of r0 and r0_500 should be specified.
    @param r0_500            Fried parameter at 500 nm in meters.  Exactly one of r0 and r0_500
                             should be specified.
    @param L0                Outer scale in meters.  [default: 25.0]
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
    _req_params = { "lam" : float }
    _opt_params = { "L0" : float, "flux" : float, "scale_unit" : str, "do_delta" : bool }
    _single_params = [ { "r0" : float, "r0_500" : float } ]
    _takes_rng = False

    _has_hard_edges = False
    _is_axisymmetric = True
    #_is_analytic_x = True  # = not do_delta  defined below.
    _is_analytic_k = True

    def __init__(self, lam, r0=None, r0_500=None, L0=25.0, flux=1, scale_unit=arcsec,
                 do_delta=False, suppress_warning=False, gsparams=None):
        # We lose stability if L0 gets too large.  This should be close enough to infinity for
        # all practical purposes though.
        if L0 > 1e10:
            L0 = 1e10

        if r0 is not None and r0_500 is not None:
            raise GalSimIncompatibleValuesError(
                "Only one of r0 and r0_500 may be specified",
                r0=r0, r0_500=r0_500)
        if r0 is None and r0_500 is None:
            raise GalSimIncompatibleValuesError(
                "Either r0 or r0_500 must be specified",
                r0=r0, r0_500=r0_500)
        if r0_500 is not None:
            r0 = r0_500 * (lam/500.)**1.2

        if isinstance(scale_unit, str):
            self._scale_unit = AngleUnit.from_name(scale_unit)
        else:
            self._scale_unit = scale_unit
        self._scale = self._scale_unit/arcsec

        self._lam = float(lam)
        self._r0 = float(r0)
        self._L0 = float(L0)
        self._flux = float(flux)
        self._do_delta = bool(do_delta)
        self._gsparams = GSParams.check(gsparams)
        self._suppress = bool(suppress_warning)
        self._sbvk  # Make this now, so we get the warning if appropriate.

    @lazy_property
    def _sbvk(self):
        with convert_cpp_errors():
            sbvk = _galsim.SBVonKarman(self._lam, self._r0, self._L0, self._flux,
                                       self._scale, self._do_delta, self._gsparams._gsp)

        self._delta = sbvk.getDelta()
        if not self._suppress:
            if self._delta > self._gsparams.maxk_threshold:
                galsim_warn("VonKarman delta-function component is larger than maxk_threshold.  "
                            "Please see docstring for information about this component and how "
                            "to toggle it.")
        if self._do_delta:
            with convert_cpp_errors():
                sbvk = _galsim.SBVonKarman(self._lam, self._r0, self._L0,
                                           self._flux-self._delta, self._scale,
                                           self._do_delta, self._gsparams._gsp)
        return sbvk

    @lazy_property
    def _sbp(self):
        # Add in a delta function with appropriate amplitude if requested.
        if self._do_delta:
            sbvk = self._sbvk
            with convert_cpp_errors():
                sbdelta = _galsim.SBDeltaFunction(self._delta, self._gsparams._gsp)
                return _galsim.SBAdd([sbvk, sbdelta], self._gsparams._gsp)
        else:
            return self._sbvk

    @property
    def lam(self):
        return self._lam

    @property
    def r0(self):
        return self._r0

    @property
    def r0_500(self):
        return self._r0*(self._lam/500.)**(-1.2)

    @property
    def L0(self):
        return self._L0

    @property
    def scale_unit(self):
        return self._scale_unit

    @property
    def do_delta(self):
        return self._do_delta

    @property
    def _is_analytic_x(self):
        return not self._do_delta

    @property
    def delta_amplitude(self):
        self._sbvk  # This is where _delta is calculated.
        return self._delta

    @property
    def half_light_radius(self):
        return self._sbvk.getHalfLightRadius()

    def _structure_function(self, rho):
        return self._sbvk.structureFunction(rho)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, VonKarman) and
                 self.lam == other.lam and
                 self.r0 == other.r0 and
                 self.L0 == other.L0 and
                 self.flux == other.flux and
                 self.scale_unit == other.scale_unit and
                 self.do_delta == other.do_delta and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.VonKarman", self.lam, self.r0, self.L0, self.flux, self.scale_unit,
                     self.do_delta, self.gsparams))

    def __repr__(self):
        out = "galsim.VonKarman(lam=%r, r0=%r, L0=%r"%(self.lam, self.r0, self.L0)
        out += ", flux=%r"%self.flux
        if self.scale_unit != arcsec:
            out += ", scale_unit=%r"%self.scale_unit
        if self.do_delta:
            out += ", do_delta=True"
        if self._suppress:
            out += ", suppress_warning=True"
        out += ", gsparams=%r"%self.gsparams
        out += ")"
        return out

    def __str__(self):
        return "galsim.VonKarman(lam=%r, r0=%r, L0=%r)"%(self.lam, self.r0, self.L0)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        d.pop('_sbvk',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        return self._sbvk.maxK()

    @property
    def _stepk(self):
        return self._sbvk.stepK()

    @property
    def _max_sb(self):
        return self._sbvk.xValue(PositionD(0,0)._p)

    @doc_inherit
    def _xValue(self, pos):
        return self._sbvk.xValue(pos._p)

    @doc_inherit
    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    @doc_inherit
    def _drawReal(self, image):
        self._sbvk.draw(image._image, image.scale)

    @doc_inherit
    def _shoot(self, photons, rng):
        self._sbp.shoot(photons._pa, rng._rng)

    @doc_inherit
    def _drawKImage(self, image):
        self._sbp.drawK(image._image, image.scale)
