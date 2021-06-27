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

import math

from . import _galsim
from .gsobject import GSObject
from .gsparams import GSParams
from .utilities import lazy_property, doc_inherit
from .exponential import Exponential
from .angle import Angle
from .errors import GalSimRangeError, GalSimIncompatibleValuesError, convert_cpp_errors


class InclinedExponential(GSObject):
    r"""A class describing an inclined exponential profile.

    The Inclined Exponential surface brightness profile is characterized by three properties: its
    inclination angle (where 0 degrees = face-on and 90 degrees = edge-on), its scale radius, and
    its scale height. The 3D light distribution function is:

    .. math::
        I(R,z) \sim \mathrm{sech}^2 (z/h_s) \, \exp\left(-R/R_s\right)

    where :math:`z` is the distance along the minor axis, :math:`R` is the radial distance from the
    minor axis, :math:`R_s` is the scale radius of the disk, and :math:`h_s` is the scale height of
    the disk. The 2D light distribution function is then determined from the scale height and
    radius here, along with the inclination angle.

    In this implementation, the profile is inclined along the y-axis. This means that it will likely
    need to be rotated in most circumstances.

    At present, this profile is not enabled for photon-shooting.

    A profile can be initialized using one (and only one) of two possible size parameters:
    ``scale_radius`` or ``half_light_radius``.  Exactly one of these two is required. Similarly,
    at most one of ``scale_height`` and ``scale_h_over_r`` is required; if neither is given, the
    default of scale_h_over_r = 0.1 will be used. Note that if half_light_radius and
    scale_h_over_r are supplied (or the default value of scale_h_over_r is used),
    scale_h_over_r will be assumed to refer to the scale radius, not the half-light radius.

    Parameters:
        inclination:        The inclination angle, which must be a `galsim.Angle` instance
        scale_radius:       The scale radius of the exponential disk.  Typically given in
                            arcsec. This can be compared to the 'scale_radius' parameter of the
                            `galsim.Exponential` class, and in the face-on case, the same scale
                            radius will result in the same 2D light distribution as with that
                            class.
        half_light_radius:  The half-light radius of the exponential disk, as an alternative to
                            the scale radius.
        scale_height:       The scale height of the exponential disk.  Typically given in arcsec.
                            [default: None]
        scale_h_over_r:     In lieu of the scale height, you may also specify the ratio of the
                            scale height to the scale radius. [default: 0.1]
        flux:               The flux (in photons) of the profile. [default: 1]
        gsparams:           An optional `GSParams` argument. [default: None]
    """
    _req_params = { "inclination" : Angle }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _opt_params = { "scale_height" : float, "scale_h_over_r" : float, "flux" : float }

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, inclination, half_light_radius=None, scale_radius=None, scale_height=None,
                 scale_h_over_r=None, flux=1., gsparams=None):

        # Check that the scale/half-light radius is valid
        if scale_radius is not None:
            if not scale_radius > 0.:
                raise GalSimRangeError("scale_radius must be > 0.", scale_radius, 0.)
            if half_light_radius is not None:
                raise GalSimIncompatibleValuesError(
                    "Only one of scale_radius and half_light_radius may be specified",
                    half_light_radius=half_light_radius, scale_radius=scale_radius)
            self._r0 = float(scale_radius)
        elif half_light_radius is not None:
            if not half_light_radius > 0.:
                raise GalSimRangeError("half_light_radius must be > 0.", half_light_radius, 0.)
            # Use the factor from the Exponential class
            self._r0 = float(half_light_radius) / Exponential._hlr_factor
        else:
            raise GalSimIncompatibleValuesError(
                "Either scale_radius or half_light_radius must be specified",
                half_light_radius=half_light_radius, scale_radius=scale_radius)

        # Check that the height specification is valid
        if scale_height is not None:
            if not scale_height > 0.:
                raise GalSimRangeError("scale_height must be > 0.", scale_height, 0.)
            if scale_h_over_r is not None:
                raise GalSimIncompatibleValuesError(
                    "Only one of scale_height and scale_h_over_r may be specified.",
                    scale_height=scale_height, scale_h_over_r=scale_h_over_r)
            self._h0 = float(scale_height)
        else:
            if scale_h_over_r is None:
                # Use the default scale_h_over_r
                scale_h_over_r = 0.1
            elif not scale_h_over_r > 0.:
                raise GalSimRangeError("half_light_radius must be > 0.", scale_h_over_r, 0.)
            self._h0 = float(self._r0) * float(scale_h_over_r)

        # Explicitly check for angle type, so we can give more informative error if eg. a float is
        # passed
        if not isinstance(inclination, Angle):
            raise TypeError("Input inclination should be an Angle")

        self._inclination = inclination
        self._flux = float(flux)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _sbp(self):
        return _galsim.SBInclinedExponential(self._inclination.rad, self._r0,
                                             self._h0, self._flux, self.gsparams._gsp)

    @property
    def inclination(self):
        """The inclination angle.
        """
        return self._inclination
    @property
    def scale_radius(self):
        """The scale radius of the exponential disk.
        """
        return self._r0
    @property
    def scale_height(self):
        """The scale height of the disk.
        """
        return self._h0

    @property
    def disk_half_light_radius(self):
        """The half-light radius of the exponential disk.
        """
        return self._r0 * Exponential._hlr_factor
    @property
    def scale_h_over_r(self):
        """The ratio scale_height / scale_radius
        """
        return self._h0 / self._r0

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, InclinedExponential) and
                 (self.inclination == other.inclination) and
                 (self.scale_radius == other.scale_radius) and
                 (self.scale_height == other.scale_height) and
                 (self.flux == other.flux) and
                 (self.gsparams == other.gsparams)))

    def __hash__(self):
        return hash(("galsim.InclinedExponential", self.inclination, self.scale_radius,
                    self.scale_height, self.flux, self.gsparams))

    def __repr__(self):
        return ('galsim.InclinedExponential(inclination=%r, scale_radius=%r, scale_height=%r, '
                'flux=%r, gsparams=%r)')%(
                self.inclination, self.scale_radius, self.scale_height, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.InclinedExponential(inclination=%s, scale_radius=%s, scale_height=%s' % (
                self.inclination, self.scale_radius, self.scale_height)
        if self.flux != 1.0:
            s += ', flux=%s' % self.flux
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
        return self._sbp.maxK()

    @property
    def _stepk(self):
        return self._sbp.stepK()

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _drawKImage(self, image, jac=None):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.drawK(image._image, image.scale, _jac)

    @doc_inherit
    def withFlux(self, flux):
        return InclinedExponential(inclination=self.inclination, scale_radius=self.scale_radius,
                                   scale_height=self.scale_height, flux=flux,
                                   gsparams=self.gsparams)


class InclinedSersic(GSObject):
    r"""A class describing an inclined sersic profile. This class is general, and so for certain
    special cases, more specialized classes will be more efficient. For the case where n==1
    with no truncation, the `InclinedExponential` class will be much more efficient. For the case
    where the inclination angle is zero (face-on), the `Sersic` class will be slightly more
    efficient.

    The InclinedSersic surface brightness profile is characterized by four properties: its
    Sersic index ``n``, its inclination angle (where 0 degrees = face-on and 90 degrees = edge-on),
    its scale radius, and its scale height. The 3D light distribution function is:

    .. math::
        I(R,z) \sim \mathrm{sech}^2 (z/h_s) \, \exp\left(-(R/R_s)^{1/n}\right)

    where :math:`z` is the distance along the minor axis, :math:`R` is the radial distance from the
    minor axis, :math:`R_s` is the scale radius of the disk, and :math:`h_s` is the scale height of
    the disk. The 2D light distribution function is then determined from the scale height and
    radius here, along with the inclination angle.

    In this implementation, the profile is inclined along the y-axis. This means that it will likely
    need to be rotated in most circumstances.

    At present, this profile is not enabled for photon-shooting.

    The allowed range of values for the ``n`` parameter is 0.3 <= n <= 6.2.  An exception will be
    thrown if you provide a value outside that range, matching the range of the `Sersic` profile.

    This class shares the caching of Hankel transformations with the `Sersic` class; see that
    class for documentation on efficiency considerations with regards to caching.

    A profile can be initialized using one (and only one) of two possible size parameters:
    ``scale_radius`` or ``half_light_radius``.  Exactly one of these two is required. Similarly,
    at most one of ``scale_height`` and ``scale_h_over_r`` is required; if neither is given, the
    default of scale_h_over_r = 0.1 will be used. Note that if half_light_radius and
    scale_h_over_r are supplied (or the default value of scale_h_over_r is used),
    scale_h_over_r will be assumed to refer to the scale radius, not the half-light radius.

    Parameters:
        n:                  The Sersic index, n.
        inclination:        The inclination angle, which must be a `galsim.Angle` instance
        scale_radius:       The scale radius of the disk.  Typically given in arcsec.
                            This can be compared to the 'scale_radius' parameter of the
                            `galsim.Sersic` class, and in the face-on case, the same scale
                            radius will result in the same 2D light distribution as with that
                            class. Exactly one of this and half_light_radius must be provided.
        half_light_radius:  The half-light radius of disk when seen face-on. Exactly one of this
                            and scale_radius must be provided.
        scale_height:       The scale height of the exponential disk.  Typically given in arcsec.
                            [default: None]
        scale_h_over_r:     In lieu of the scale height, you may specify the ratio of the
                            scale height to the scale radius. [default: 0.1]
        flux:               The flux (in photons) of the profile. [default: 1]
        trunc:              An optional truncation radius at which the profile is made to drop to
                            zero, in the same units as the size parameter.
                            [default: 0, indicating no truncation]
        flux_untruncated:   Should the provided ``flux`` and ``half_light_radius`` refer to the
                            untruncated profile? See the documentation of the `Sersic` class for
                            more details. [default: False]
        gsparams:           An optional `GSParams` argument. [default: None]
    """
    _req_params = { "inclination" : Angle, "n" : float }
    _opt_params = { "scale_height" : float, "scale_h_over_r" : float, "flux" : float,
                    "trunc" : float, "flux_untruncated" : bool }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float }]
    _takes_rng = False

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, n, inclination, half_light_radius=None, scale_radius=None, scale_height=None,
                 scale_h_over_r=None, flux=1., trunc=0., flux_untruncated=False, gsparams=None):

        self._flux = float(flux)
        self._n = float(n)
        self._inclination = inclination
        self._trunc = float(trunc)
        self._gsparams = GSParams.check(gsparams)

        # Check that trunc is valid
        if trunc < 0.:
            raise GalSimRangeError("trunc must be >= 0. (zero implying no truncation).", trunc, 0.)

        # Parse the radius options
        if scale_radius is not None:
            if not scale_radius > 0.:
                raise GalSimRangeError("scale_radius must be > 0.", scale_radius, 0.)
            if half_light_radius is not None:
                raise GalSimIncompatibleValuesError(
                    "Only one of scale_radius and half_light_radius may be specified.",
                    half_light_radius=half_light_radius, scale_radius=scale_radius)
            self._r0 = float(scale_radius)
            self._hlr = 0.
        elif half_light_radius is not None:
            if not half_light_radius > 0.:
                raise GalSimRangeError("half_light_radius must be > 0.", half_light_radius, 0.)
            self._hlr = float(half_light_radius)
            if self._trunc == 0. or flux_untruncated:
                with convert_cpp_errors():
                    self._r0 = self._hlr / _galsim.SersicHLR(self._n, 1.)
            else:
                if self._trunc <= math.sqrt(2.) * self._hlr:
                    raise GalSimRangeError("Sersic trunc must be > sqrt(2) * half_light_radius",
                                           self._trunc, math.sqrt(2.) * self._hlr)
                with convert_cpp_errors():
                    self._r0 = _galsim.SersicTruncatedScale(self._n, self._hlr, self._trunc)
        else:
            raise GalSimIncompatibleValuesError(
                "Either scale_radius or half_light_radius must be specified",
                half_light_radius=half_light_radius, scale_radius=scale_radius)

        # Parse the height options
        if scale_height is not None:
            if not scale_height > 0.:
                raise GalSimRangeError("scale_height must be > 0.", scale_height, 0.)
            if scale_h_over_r is not None:
                raise GalSimIncompatibleValuesError(
                    "Only one of scale_height and scale_h_over_r may be specified",
                    scale_height=scale_height, scale_h_over_r=scale_h_over_r)
            self._h0 = float(scale_height)
        else:
            if scale_h_over_r is None:
                scale_h_over_r = 0.1
            elif not scale_h_over_r > 0.:
                raise GalSimRangeError("half_light_radius must be > 0.", scale_h_over_r, 0.)
            self._h0 = float(scale_h_over_r) * self._r0

        # Explicitly check for angle type, so we can give more informative error if eg. a float is
        # passed
        if not isinstance(inclination, Angle):
            raise TypeError("Input inclination should be an Angle")

        # If flux_untrunctated, then the above picked the right radius, but the flux needs
        # to be updated.
        if self._trunc > 0.:
            with convert_cpp_errors():
                self._flux_fraction = _galsim.SersicIntegratedFlux(self._n, self._trunc/self._r0)
            if flux_untruncated:
                self._flux *= self._flux_fraction
                self._hlr = 0.  # This will be updated by getHalfLightRadius if necessary.
        else:
            self._flux_fraction = 1.

    @lazy_property
    def _sbp(self):
        with convert_cpp_errors():
            return  _galsim.SBInclinedSersic(self._n, self._inclination.rad, self._r0, self._h0,
                                             self._flux, self._trunc, self.gsparams._gsp)

    @property
    def n(self):
        """The Sersic parameter n.
        """
        return self._n
    @property
    def inclination(self):
        """The inclination angle.
        """
        return self._inclination
    @property
    def scale_radius(self):
        """The scale radius of the exponential disk.
        """
        return self._r0
    @property
    def scale_height(self):
        """The scale height of the disk.
        """
        return self._h0
    @property
    def trunc(self):
        """The truncation radius (if any).
        """
        return self._trunc
    @property
    def scale_h_over_r(self):
        """The ratio scale_height / scale_radius.
        """
        return self._h0 / self._r0

    @property
    def disk_half_light_radius(self):
        """The half-light radius of the exponential disk.
        """
        if self._hlr == 0.:
            with convert_cpp_errors():
                self._hlr = self._r0 * _galsim.SersicHLR(self._n, self._flux_fraction)
        return self._hlr

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, InclinedSersic) and
                 (self.n == other.n) and
                 (self.inclination == other.inclination) and
                 (self.scale_radius == other.scale_radius) and
                 (self.scale_height == other.scale_height) and
                 (self.flux == other.flux) and
                 (self.trunc == other.trunc) and
                 (self.gsparams == other.gsparams)))

    def __hash__(self):
        return hash(("galsim.InclinedSersic", self.n, self.inclination, self.scale_radius,
                    self.scale_height, self.flux, self.trunc, self.gsparams))
    def __repr__(self):
        return ('galsim.InclinedSersic(n=%r, inclination=%r, scale_radius=%r, scale_height=%r, '
                'flux=%r, trunc=%r, gsparams=%r)')%(
                self.n, self.inclination, self.scale_radius, self.scale_height, self.flux,
                self.trunc, self.gsparams)

    def __str__(self):
        s = 'galsim.InclinedSersic(n=%s, inclination=%s, scale_radius=%s, scale_height=%s' % (
                self.n, self.inclination, self.scale_radius, self.scale_height)
        if self.flux != 1.0:
            s += ', flux=%s' % self.flux
        if self.trunc != 0.:
            s += ', trunc=%s' % self.trunc
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
        return self._sbp.maxK()

    @property
    def _stepk(self):
        return self._sbp.stepK()

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _drawKImage(self, image, jac=None):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.drawK(image._image, image.scale, _jac)

    @doc_inherit
    def withFlux(self, flux):
        return InclinedSersic(n=self.n, inclination=self.inclination,
                              scale_radius=self.scale_radius, scale_height=self.scale_height,
                              flux=flux, trunc=self.trunc, gsparams=self.gsparams)
