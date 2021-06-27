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
from .errors import GalSimRangeError, GalSimIncompatibleValuesError, convert_cpp_errors


class Spergel(GSObject):
    r"""A class describing a Spergel profile.

    The Spergel surface brightness profile is characterized by three properties: its Spergel index
    ``nu``, its ``flux``, and either the ``half_light_radius`` or ``scale_radius``.  Given these
    properties, the surface brightness profile scales as

    .. math::
        I(r) \sim \left(\frac{r}{r_0}\right)^\nu K_\nu\left(\frac{r}{r_0}\right)

    where :math:`r_0` is the ``scale_radius`` and :math:`K_\nu` is the modified Bessel function of
    the second kind.

    The Spergel profile is intended as a generic galaxy profile, somewhat like a `Sersic` profile,
    but with the advantage of being analytic in both real space and Fourier space.  The Spergel
    index :math:`\nu` plays a similar role to the Sersic index :math:`n`, in that it adjusts the
    relative peakiness of the profile core and the relative prominence of the profile wings.
    At :math:`\nu = 0.5`, the Spergel profile is equivalent to an `Exponential` profile (or
    alternatively an :math`n = 1` `Sersic` profile).  At :math:`\nu = -0.6` (and in the radial
    range near the half-light radius), the Spergel profile is similar to a `DeVaucouleurs` profile
    or :math:`n = 4` `Sersic` profile.

    Note that for :math:`\nu <= 0`, the Spergel profile surface brightness diverges at the origin.
    This may lead to rendering problems if the profile is not convolved by either a PSF or a pixel
    and the profile center is precisely on a pixel center.

    Due to its analytic Fourier transform and depending on the indices :math:`n` and :math:`\nu`,
    the Spergel profile can be considerably faster to draw than the roughly equivalent `Sersic`
    profile.  For example, the :math:`\nu = -0.6` Spergel profile is roughly 3x faster to draw than
    an :math:`n = 4` `Sersic` profile once the `Sersic` profile cache has been set up.  However, if
    not taking advantage of the cache, for example, if drawing `Sersic` profiles with :math:`n`
    continuously varying near 4.0 and Spergel profiles with :math:`\nu` continuously varying near
    -0.6, then the Spergel profiles are about 50x faster to draw.  At the other end of the galaxy
    profile spectrum, the :math:`\nu = 0.5` Spergel profile, :math:`n = 1` `Sersic` profile, and
    the `Exponential` profile all take about the same amount of time to draw if cached, and the
    Spergel profile is about 2x faster than the `Sersic` profile if uncached.

    For more information, refer to

        D. N. Spergel, "ANALYTICAL GALAXY PROFILES FOR PHOTOMETRIC AND LENSING ANALYSIS,"
        ASTROPHYS J SUPPL S 191(1), 58-65 (2010) [doi:10.1088/0067-0049/191/1/58].

    The allowed range of values for the ``nu`` parameter is -0.85 <= ``nu`` <= 4.  An exception
    will be thrown if you provide a value outside that range.  The lower limit is set above the
    theoretical lower limit of -1 due to numerical difficulties integrating the *very* peaky
    ``nu`` < -0.85 profiles.  The upper limit is set to avoid numerical difficulties evaluating the
    modified Bessel function of the second kind.

    A Spergel profile can be initialized using one (and only one) of two possible size parameters:
    ``scale_radius`` or ``half_light_radius``.  Exactly one of these two is required.

    Parameters:
        nu:                 The Spergel index, nu.
        half_light_radius:  The half-light radius of the profile.  Typically given in arcsec.
                            [One of ``scale_radius`` or ``half_light_radius`` is required.]
        scale_radius:       The scale radius of the profile.  Typically given in arcsec.
                            [One of ``scale_radius`` or ``half_light_radius`` is required.]
        flux:               The flux (in photons/cm^2/s) of the profile. [default: 1]
        gsparams:           An optional `GSParams` argument. [default: None]
    """
    _req_params = { "nu" : float }
    _opt_params = { "flux" : float}
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]

    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    # Constrain range of allowed Spergel index nu.  Spergel (2010) Table 1 lists values of nu
    # from -0.9 to +0.85. We found that nu = -0.9 is too tricky for the GKP integrator to
    # handle, however, so the lower limit is -0.85 instead.  The upper limit is set by the
    # cyl_bessel_k function, which runs into overflow errors for nu larger than about 4.0.
    _minimum_nu = -0.85
    _maximum_nu = 4.0

    def __init__(self, nu, half_light_radius=None, scale_radius=None,
                 flux=1., gsparams=None):
        self._nu = float(nu)
        self._flux = float(flux)
        self._gsparams = GSParams.check(gsparams)

        if self._nu < Spergel._minimum_nu:
            raise GalSimRangeError("Requested Spergel index is too small",
                                   self._nu, Spergel._minimum_nu, Spergel._maximum_nu)
        if self._nu > Spergel._maximum_nu:
            raise GalSimRangeError("Requested Spergel index is too large",
                                   self._nu, Spergel._minimum_nu, Spergel._maximum_nu)

        # Parse the radius options
        if half_light_radius is not None:
            if scale_radius is not None:
                raise GalSimIncompatibleValuesError(
                    "Only one of scale_radius or half_light_radius may be specified",
                    half_light_radius=half_light_radius, scale_radius=scale_radius)
            self._hlr = float(half_light_radius)
            with convert_cpp_errors():
                self._r0 = self._hlr / _galsim.SpergelCalculateHLR(self._nu)
        elif scale_radius is not None:
            self._r0 = float(scale_radius)
            self._hlr = 0.
        else:
            raise GalSimIncompatibleValuesError(
                "Either scale_radius or half_light_radius must be specified for Spergel",
                half_light_radius=half_light_radius, scale_radius=scale_radius)

    @lazy_property
    def _sbp(self):
        with convert_cpp_errors():
            return _galsim.SBSpergel(self._nu, self._r0, self._flux, self.gsparams._gsp)

    @property
    def nu(self):
        """The Spergel index, nu
        """
        return self._nu

    @property
    def scale_radius(self):
        """The scale radius
        """
        return self._r0

    @property
    def half_light_radius(self):
        """The half-light radius
        """
        if self._hlr == 0.:
            with convert_cpp_errors():
                self._hlr = self._r0 * _galsim.SpergelCalculateHLR(self._nu)
        return self._hlr

    def calculateIntegratedFlux(self, r):
        """Return the integrated flux out to a given radius, r"""
        return self._sbp.calculateIntegratedFlux(float(r))

    def calculateFluxRadius(self, f):
        """Return the radius within which the total flux is f"""
        return self._sbp.calculateFluxRadius(float(f))

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Spergel) and
                 self.nu == other.nu and
                 self.scale_radius == other.scale_radius and
                 self.flux == other.flux and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.Spergel", self.nu, self.scale_radius, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Spergel(nu=%r, scale_radius=%r, flux=%r, gsparams=%r)'%(
            self.nu, self.scale_radius, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Spergel(nu=%s, half_light_radius=%s'%(self.nu, self.half_light_radius)
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
        # (1+k^2)^(-1-nu) = maxk_threshold
        return math.sqrt(self.gsparams.maxk_threshold ** (-1./(1.+self._nu)) - 1.0) / self._r0

    @property
    def _stepk(self):
        R = self.calculateFluxRadius(1.0 - self.gsparams.folding_threshold) * self._r0
        # Go to at least 5*hlr
        R = max(R, self.gsparams.stepk_minimum_hlr * self.half_light_radius)
        return math.pi / R

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

    def _xValue(self, pos):
        return self._sbp.xValue(pos._p)

    def _kValue(self, kpos):
        ksq = (kpos.x**2 + kpos.y**2) * self._r0**2
        return self._flux * (1.+ksq)**(-1.-self._nu)

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
        return Spergel(nu=self.nu, scale_radius=self.scale_radius, flux=flux,
                       gsparams=self.gsparams)
