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
from .utilities import lazy_property, doc_inherit
from .errors import GalSimIncompatibleValuesError


class Exponential(GSObject):
    r"""A class describing an exponential profile.

    Surface brightness profile with

    .. math::
        I(r) \sim e^{-r/r_0}

    where :math:`r_0` is the ``scale_radius``.  This is a special case of the `Sersic` profile,
    but is given a separate class since the Fourier transform has closed form and can be generated
    without lookup tables.

    An Exponential can be initialized using one (and only one) of two possible size parameters:
    ``scale_radius`` or ``half_light_radius``.  Exactly one of these two is required.

    Parameters:
        half_light_radius:  The half-light radius of the profile.  Typically given in arcsec.
                            [One of ``scale_radius`` or ``half_light_radius`` is required.]
        scale_radius:       The scale radius of the profile.  Typically given in arcsec.
                            [One of ``scale_radius`` or ``half_light_radius`` is required.]
        flux:               The flux (in photons/cm^2/s) of the profile. [default: 1]
        gsparams:           An optional `GSParams` argument. [default: None]
    """
    _opt_params = { "flux" : float }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]

    # The half-light-radius is not analytic, but can be calculated numerically
    # by iterative solution of equation:
    #     (re / r0) = ln[(re / r0) + 1] + ln(2)
    _hlr_factor = 1.6783469900166605
    _one_third = 1./3.
    _inv_twopi = 0.15915494309189535

    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, half_light_radius=None, scale_radius=None, flux=1., gsparams=None):
        if half_light_radius is not None:
            if scale_radius is not None:
                raise GalSimIncompatibleValuesError(
                    "Only one of scale_radius and half_light_radius may be specified",
                    half_light_radius=half_light_radius, scale_radius=scale_radius)
            else:
                scale_radius = half_light_radius / Exponential._hlr_factor
        elif scale_radius is None:
                raise GalSimIncompatibleValuesError(
                    "Either scale_radius or half_light_radius must be specified",
                    half_light_radius=half_light_radius, scale_radius=scale_radius)
        self._r0 = float(scale_radius)
        self._flux = float(flux)
        self._gsparams = GSParams.check(gsparams)
        self._inv_r0 = 1./self._r0
        self._norm = self._flux * Exponential._inv_twopi * self._inv_r0**2

    @lazy_property
    def _sbp(self):
        return _galsim.SBExponential(self._r0, self._flux, self.gsparams._gsp)

    @property
    def scale_radius(self):
        """The scale radius of the profile.
        """
        return self._r0
    @property
    def half_light_radius(self):
        """The half-light radius of the profile.
        """
        return self._r0 * Exponential._hlr_factor

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Exponential) and
                 self.scale_radius == other.scale_radius and
                 self.flux == other.flux and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.Exponential", self.scale_radius, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Exponential(scale_radius=%r, flux=%r, gsparams=%r)'%(
            self.scale_radius, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Exponential(scale_radius=%s'%self.scale_radius
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        return (self.gsparams.maxk_threshold ** -Exponential._one_third) / self.scale_radius

    @property
    def _stepk(self):
        return self._sbp.stepK()

    @property
    def _max_sb(self):
        return self._norm

    def _xValue(self, pos):
        r = math.sqrt(pos.x**2 + pos.y**2)
        return self._norm * math.exp(-r * self._inv_r0)

    def _kValue(self, kpos):
        ksqp1 = (kpos.x**2 + kpos.y**2) * self._r0**2 + 1.
        return self._flux / (ksqp1 * math.sqrt(ksqp1))

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
        return Exponential(scale_radius=self.scale_radius, flux=flux, gsparams=self.gsparams)
