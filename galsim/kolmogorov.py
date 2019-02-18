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

import numpy as np
import math

from . import _galsim
from .gsobject import GSObject
from .gsparams import GSParams
from .utilities import lazy_property, doc_inherit
from .position import PositionD
from .errors import GalSimIncompatibleValuesError, convert_cpp_errors


class Kolmogorov(GSObject):
    """A class describing a Kolmogorov surface brightness profile, which represents a long
    exposure atmospheric PSF.

    For more information, refer to

        http://en.wikipedia.org/wiki/Atmospheric_seeing#The_Kolmogorov_model_of_turbulence

    Initialization
    --------------

    The Kolmogorov profile is normally defined in terms of the ratio lambda / r0, where lambda is
    the wavelength of the light (say in the middle of the bandpass you are using) and r0 is the
    Fried parameter.  Typical values for the Fried parameter are on the order of 0.1 m for
    most observatories and up to 0.2 m for excellent sites. The values are usually quoted at
    lambda = 500nm and r0 depends on wavelength as [r0 ~ lambda^(6/5)].

    The natural units for this ratio is radians, which is not normally a convenient unit to use for
    other GSObject dimensions.  Assuming that the other sky coordinates you are using are all in
    arcsec (e.g. the pixel scale when you draw the image, the size of the galaxy, etc.), then you
    should convert this to arcsec as well:

        >>> lam = 700  # nm
        >>> r0 = 0.15 * (lam/500)**1.2  # meters
        >>> lam_over_r0 = (lam * 1.e-9) / r0  # radians
        >>> lam_over_r0 *= 206265  # Convert to arcsec
        >>> psf = galsim.Kolmogorov(lam_over_r0)

    To make this process a bit simpler, we recommend instead providing the wavelength and Fried
    parameter separately using the parameters `lam` (in nm) and either `r0` or `r0_500` (in m).
    GalSim will then convert this to any of the normal kinds of angular units using the
    `scale_unit` parameter:

        >>> psf = galsim.Kolmogorov(lam=lam, r0=r0, scale_unit=galsim.arcsec)

    or

        >>> psf = galsim.Kolmogorov(lam=lam, r0_500=0.15, scale_unit=galsim.arcsec)

    When drawing images, the scale_unit should match the unit used for the pixel scale or the WCS.
    e.g. in this case, a pixel scale of 0.2 arcsec/pixel would be specified as `pixel_scale=0.2`.

    A Kolmogorov object may also be initialized using `fwhm` or `half_light_radius`.  Exactly one
    of these four ways to define the size is required.

    The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 arcsec (e.g., Racine 1996, PASP 699, 108).

    @param lam_over_r0      The parameter that governs the scale size of the profile.
                            See above for details about calculating it. [One of `lam_over_r0`,
                            `fwhm`, `half_light_radius`, or `lam` (along with either `r0` or
                            `r0_500`) is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of `lam_over_r0`, `fwhm`, `half_light_radius`, or `lam` (along
                            with either `r0` or `r0_500`) is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `lam_over_r0`, `fwhm`, `half_light_radius`, or `lam` (along
                            with either `r0` or `r0_500`) is required.]
    @param lam              Lambda (wavelength) in units of nanometers.  Must be supplied with
                            either `r0` or `r0_500`, and in this case, image scales (`scale`)
                            should be specified in units of `scale_unit`.
    @param r0               The Fried parameter in units of meters.  Must be supplied with `lam`,
                            and in this case, image scales (`scale`) should be specified in units
                            of `scale_unit`.
    @param r0_500           The Fried parameter in units of meters at 500 nm.  The Fried parameter
                            at the given wavelength, `lam`, will be computed using the standard
                            relation r0 = r0_500 * (lam/500)**1.2.
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param scale_unit       Units to use for the sky coordinates when calculating lam/r0 if these
                            are supplied separately.  Note that the results of calling methods like
                            getFWHM() will be returned in units of `scale_unit` as well.  Should be
                            either a galsim.AngleUnit or a string that can be used to construct one
                            (e.g., 'arcsec', 'radians', etc.).  [default: galsim.arcsec]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, Kolmogorov has the following access properties:

        >>> lam_over_r0 = kolm.lam_over_r0
        >>> fwhm = kolm.fwhm
        >>> hlr = kolm.half_light_radius
    """
    _req_params = {}
    _opt_params = { "flux" : float, "r0" : float, "r0_500" : float, "scale_unit" : str }
    # Note that this is not quite right; it's true that exactly one of these 4 should be supplied,
    # but if lam is supplied then r0 is required.  Errors in which parameters are used may be
    # caught either by config or by the python code itself, depending on the particular error.
    _single_params = [ { "lam_over_r0" : float, "fwhm" : float, "half_light_radius" : float,
                         "lam" : float } ]
    _takes_rng = False

    # The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).
    # In SBKolmogorov.cpp we refine this factor to 0.975865
    _fwhm_factor = 0.975865
    # Similarly, SBKolmogorov calculates the relation between lambda/r0 and half-light radius
    _hlr_factor = 0.554811

    # This constant comes from the standard form of the Kolmogorov spectrum from
    # from Racine, 1996 PASP, 108, 699 (who in turn is quoting Fried, 1966, JOSA, 56, 1372):
    # T(k) = exp(-1/2 D(k))
    # D(k) = 6.8839 (lambda/r0 k/2Pi)^(5/3)
    #
    # We convert this into T(k) = exp(-(k/k0)^5/3) for efficiency,
    # which implies 1/2 6.8839 (lambda/r0 / 2Pi)^5/3 = (1/k0)^5/3
    # k0 * lambda/r0 = 2Pi * (6.8839 / 2)^-3/5 = 2.992934
    _k0_factor = 2.992934

    # The value in real space at (x,y) = (0,0) is analytic:
    # int( flux (k/2pi) exp(-(k/k0)**(5/3)), k=0..inf)
    # = flux * k0^2 * (3/5) Gamma(6/5) / 2pi
    _xzero = 0.08767865636723461

    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, lam_over_r0=None, fwhm=None, half_light_radius=None, lam=None, r0=None,
                 r0_500=None, flux=1., scale_unit=None, gsparams=None):

        from .angle import arcsec, radians, AngleUnit

        self._flux = float(flux)
        self._gsparams = GSParams.check(gsparams)

        if fwhm is not None :
            if any(item is not None for item in (lam_over_r0, lam, r0, r0_500, half_light_radius)):
                raise GalSimIncompatibleValuesError(
                    "Only one of lam_over_r0, fwhm, half_light_radius, or lam (with r0 or r0_500) "
                    "may be specified",
                    fwhm=fwhm, lam_over_r0=lam_over_r0, lam=lam, r0=r0, r0_500=r0_500,
                    half_light_radius=half_light_radius)
            self._lor0 = float(fwhm) / Kolmogorov._fwhm_factor
        elif half_light_radius is not None:
            if any(item is not None for item in (lam_over_r0, lam, r0, r0_500)):
                raise GalSimIncompatibleValuesError(
                    "Only one of lam_over_r0, fwhm, half_light_radius, or lam (with r0 or r0_500) "
                    "may be specified",
                    fwhm=fwhm, lam_over_r0=lam_over_r0, lam=lam, r0=r0, r0_500=r0_500,
                    half_light_radius=half_light_radius)
            self._lor0 = float(half_light_radius) / Kolmogorov._hlr_factor
        elif lam_over_r0 is not None:
            if any(item is not None for item in (lam, r0, r0_500)):
                raise GalSimIncompatibleValuesError(
                    "Cannot specify lam, r0 or r0_500 in conjunction with lam_over_r0.",
                    lam_over_r0=lam_over_r0, lam=lam, r0=r0, r0_500=r0_500)
            self._lor0 = float(lam_over_r0)
        else:
            if lam is None or (r0 is None and r0_500 is None):
                raise GalSimIncompatibleValuesError(
                    "One of lam_over_r0, fwhm, half_light_radius, or lam (with r0 or r0_500) "
                    "must be specified",
                    fwhm=fwhm, lam_over_r0=lam_over_r0, lam=lam, r0=r0, r0_500=r0_500,
                    half_light_radius=half_light_radius)
            # In this case we're going to use scale_unit, so parse it in case of string input:
            if scale_unit is None:
                scale_unit = arcsec
            elif isinstance(scale_unit, str):
                scale_unit = AngleUnit.from_name(scale_unit)
            if r0 is None:
                r0 = r0_500 * (lam/500.)**1.2
            self._lor0 = (1.e-9*float(lam)/float(r0))*(radians/scale_unit)

        self._k0 = Kolmogorov._k0_factor / self._lor0

    @lazy_property
    def _sbp(self):
        with convert_cpp_errors():
            return _galsim.SBKolmogorov(self._lor0, self._flux, self.gsparams._gsp)

    @property
    def lam_over_r0(self): return self._lor0

    @property
    def fwhm(self):
        """Return the FWHM of this Kolmogorov profile.
        """
        return self._lor0 * Kolmogorov._fwhm_factor

    @property
    def half_light_radius(self):
        """Return the half light radius of this Kolmogorov profile.
        """
        return self._lor0 * Kolmogorov._hlr_factor

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Kolmogorov) and
                 self.lam_over_r0 == other.lam_over_r0 and
                 self.flux == other.flux and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.Kolmogorov", self.lam_over_r0, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Kolmogorov(lam_over_r0=%r, flux=%r, gsparams=%r)'%(
            self.lam_over_r0, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Kolmogorov(lam_over_r0=%s'%self.lam_over_r0
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
        # exp(-k^(5/3)) = kvalue_accuracy
        return (-math.log(self.gsparams.kvalue_accuracy)) ** 0.6 * self._k0

    @property
    def _stepk(self):
        return self._sbp.stepK()

    @property
    def _max_sb(self):
        return self._flux * self._k0**2 * Kolmogorov._xzero

    @doc_inherit
    def _xValue(self, pos):
        return self._sbp.xValue(pos._p)

    @doc_inherit
    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    @doc_inherit
    def _drawReal(self, image):
        self._sbp.draw(image._image, image.scale)

    @doc_inherit
    def _shoot(self, photons, rng):
        self._sbp.shoot(photons._pa, rng._rng)

    @doc_inherit
    def _drawKImage(self, image):
        self._sbp.drawK(image._image, image.scale)
