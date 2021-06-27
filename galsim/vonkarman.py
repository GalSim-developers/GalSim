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
from .position import _PositionD
from .angle import arcsec, AngleUnit
from .errors import GalSimError, convert_cpp_errors, galsim_warn
from .errors import GalSimIncompatibleValuesError


class VonKarman(GSObject):
    r"""Class describing a von Karman surface brightness profile, which represents a long exposure
    atmospheric PSF.  The difference between the von Karman profile and the related `Kolmogorov`
    profile is that the von Karman profile includes a parameter for the outer scale of atmospheric
    turbulence, which is a physical scale beyond which fluctuations in the refractive index stop
    growing, typically between 10 and 100 meters.  Quantitatively, the von Karman phase fluctuation
    power spectrum at spatial frequency f is proportional to

    .. math::

        P(f) = r_0^{-5/3} \left(f^2 + L_0^{-2}\right)^{-11/6}

    where :math:`r_0` is the Fried parameter which sets the overall turbulence amplitude and
    :math:`L_0` is the outer scale in meters.
    The Kolmogorov power spectrum proportional to :math:`r_0^{-5/3} f^{-11/3}` is recovered
    as :math:`L_0 \rightarrow \infty`.

    For more information, we recommend the following references:

        Martinez et al.  2010  A&A  vol. 516
        Conan  2008  JOSA  vol. 25

    .. note::

        If one blindly follows the math for converting the von Karman power spectrum into a PSF, one
        finds that the PSF contains a delta-function at the origin with fractional flux of:

        .. math::

            F_\mathrm{delta} = e^{-0.086 (r_0/L_0)^{-5/3}}

        In almost all cases of interest this evaluates to something tiny, often on the order of
        :math:`10^{-100}` or smaller.  By default, GalSim will ignore this delta function entirely
        since it usually doesn't make any difference, but can complicate some calculations like
        drawing using method='real_space' or by formally requiring huge Fourier transforms for
        drawing using method='fft'.  If for some reason you want to keep the delta function
        though, then you can pass the do_delta=True argument to the VonKarman initializer.

    Parameters:
        lam:                Wavelength in nanometers.
        r0:                 Fried parameter at specified wavelength ``lam`` in meters.  Exactly one
                            of r0 and r0_500 should be specified.
        r0_500:             Fried parameter at 500 nm in meters.  Exactly one of r0 and r0_500
                            should be specified.
        L0:                 Outer scale in meters.  [default: 25.0]
        flux:               The flux (in photons/cm^2/s) of the profile. [default: 1]
        scale_unit:         Units assumed when drawing this profile or evaluating xValue, kValue,
                            etc.  Should be a `galsim.AngleUnit` or a string that can be used to
                            construct one (e.g., 'arcsec', 'radians', etc.).
                            [default: galsim.arcsec]
        force_stepk:        By default, VonKarman will derive a value of stepk from a computed
                            real-space surface brightness profile and gsparams settings.  Although
                            this profile is cached for future instantiations of identical VonKarman
                            objects, it is relatively slow to compute for the first instance and
                            can dominate the compute time when drawing many VonKaman's with
                            different parameters using method 'fft', 'sb', or 'no_pixel', a
                            situation that may occur, e.g., in a fitting context.  This keyword
                            enables a user to bypass the real-space profile computation by directly
                            specifying a stepk value.  Note that if the ``.half_light_radius``
                            property is queried, or the object is drawn using method 'phot' or
                            'real_space', then the real-space profile calculation is performed (if
                            not cached) at that point.  [default: 0.0, which means do not force a
                            value of stepk]
        do_delta:           Include delta-function at origin? (not recommended; see above).
                            [default: False]
        suppress_warning:   For some combinations of r0 and L0, it may become impossible to satisfy
                            the gsparams.maxk_threshold criterion (see above).  In that case, the
                            code will emit a warning alerting the user that they may have entered a
                            non-physical regime.  However, this warning can be suppressed with this
                            keyword.  [default: False]
        gsparams:           An optional `GSParams` argument. [default: None]
    """
    _req_params = { "lam" : float }
    _opt_params = { "L0" : float, "flux" : float, "scale_unit" : str, "do_delta" : bool }
    _single_params = [ { "r0" : float, "r0_500" : float } ]

    _has_hard_edges = False
    _is_axisymmetric = True
    #_is_analytic_x = True  # = not do_delta  defined below.
    _is_analytic_k = True

    def __init__(self, lam, r0=None, r0_500=None, L0=25.0, flux=1, scale_unit=arcsec,
                 force_stepk=0.0, do_delta=False, suppress_warning=False, gsparams=None):
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
        self._force_stepk = force_stepk
        self._sbvk  # Make this now, so we get the warning if appropriate.

    @lazy_property
    def _sbvk(self):
        with convert_cpp_errors():
            sbvk = _galsim.SBVonKarman(self._lam, self._r0, self._L0, self._flux,
                                       self._scale, self._do_delta, self._gsparams._gsp,
                                       self._force_stepk)

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
                                           self._do_delta, self._gsparams._gsp,
                                           self._force_stepk)
        return sbvk

    @lazy_property
    def _sbp(self):
        # Add in a delta function with appropriate amplitude if requested.
        if self._do_delta:
            sbvk = self._sbvk
            sbdelta = _galsim.SBDeltaFunction(self._delta, self._gsparams._gsp)
            return _galsim.SBAdd([sbvk, sbdelta], self._gsparams._gsp)
        else:
            return self._sbvk

    @property
    def lam(self):
        """The input lam value.
        """
        return self._lam

    @property
    def r0(self):
        """The input r0 value.
        """
        return self._r0

    @property
    def r0_500(self):
        """The input r0_500 value.
        """
        return self._r0*(self._lam/500.)**(-1.2)

    @property
    def L0(self):
        """The input L0 value.
        """
        return self._L0

    @property
    def scale_unit(self):
        """The input scale_units.
        """
        return self._scale_unit

    @property
    def force_stepk(self):
        """Forced value of stepk or 0.0.
        """
        return self._force_stepk

    @property
    def do_delta(self):
        """Whether to include the delta function at the center.
        """
        return self._do_delta

    @property
    def _is_analytic_x(self):
        return not self._do_delta

    @property
    def delta_amplitude(self):
        """The amplitude of the delta function at the center.
        """
        self._sbvk  # This is where _delta is calculated.
        return self._delta

    @property
    def half_light_radius(self):
        """The half-light radius.
        """
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
                 self.force_stepk == other.force_stepk and
                 self.do_delta == other.do_delta and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.VonKarman", self.lam, self.r0, self.L0, self.flux, self.scale_unit,
                     self.force_stepk, self.do_delta, self.gsparams))

    def __repr__(self):
        out = "galsim.VonKarman(lam=%r, r0=%r, L0=%r"%(self.lam, self.r0, self.L0)
        out += ", flux=%r"%self.flux
        if self.scale_unit != arcsec:
            out += ", scale_unit=%r"%self.scale_unit
        if self.force_stepk:
            out += ", force_stepk=%r"%self.force_stepk
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
        return self._sbvk.xValue(_PositionD(0,0)._p)

    def _xValue(self, pos):
        return self._sbvk.xValue(pos._p)

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _drawReal(self, image, jac=None, offset=(0.,0.), flux_scaling=1.):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        dx,dy = offset
        self._sbp.draw(image._image, image.scale, _jac, dx, dy, flux_scaling)

    def _shoot(self, photons, rng):
        self._sbp.shoot(photons._pa, rng._rng)

    def _drawKImage(self, image, jac=None):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.drawK(image._image, image.scale, _jac)

    @doc_inherit
    def withFlux(self, flux):
        return VonKarman(lam=self.lam, r0=self.r0, L0=self.L0, flux=flux,
                         scale_unit=self.scale_unit,
                         force_stepk=self.force_stepk, do_delta=self.do_delta,
                         suppress_warning=self._suppress, gsparams=self.gsparams)
