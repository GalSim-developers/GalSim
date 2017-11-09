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

import galsim

from . import _galsim
from .gsobject import GSObject


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
                            ralation r0 = r0_500 * (lam/500)**1.2.
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

    def __init__(self, lam_over_r0=None, fwhm=None, half_light_radius=None, lam=None, r0=None,
                 r0_500=None, flux=1., scale_unit=galsim.arcsec, gsparams=None):

        if fwhm is not None :
            if any(item is not None for item in (lam_over_r0, lam, r0, r0_500, half_light_radius)):
                raise TypeError(
                        "Only one of lam_over_r0, fwhm, half_light_radius, or lam (with r0 or "+
                        "r0_500) may be specified for Kolmogorov")
            else:
                lam_over_r0 = fwhm / Kolmogorov._fwhm_factor
        elif half_light_radius is not None:
            if any(item is not None for item in (lam_over_r0, lam, r0, r0_500)):
                raise TypeError(
                        "Only one of lam_over_r0, fwhm, half_light_radius, or lam (with r0 or "+
                        "r0_500) may be specified for Kolmogorov")
            else:
                lam_over_r0 = half_light_radius / Kolmogorov._hlr_factor
        elif lam_over_r0 is not None:
            if any(item is not None for item in (lam, r0, r0_500)):
                raise TypeError("Cannot specify lam, r0 or r0_500 in conjunction with lam_over_r0.")
        else:
            if lam is None or (r0 is None and r0_500 is None):
                raise TypeError(
                        "One of lam_over_r0, fwhm, half_light_radius, or lam (with r0 or "+
                        "r0_500) must be specified for Kolmogorov")
            # In this case we're going to use scale_unit, so parse it in case of string input:
            if isinstance(scale_unit, str):
                scale_unit = galsim.AngleUnit.from_name(scale_unit)
            if r0 is None:
                r0 = r0_500 * (lam/500.)**1.2
            lam_over_r0 = (1.e-9*lam/r0)*(galsim.radians/scale_unit)

        self._gsparams = galsim.GSParams.check(gsparams)
        self._sbp = _galsim.SBKolmogorov(lam_over_r0, flux, self.gsparams._gsp)

    @property
    def lam_over_r0(self): return self._sbp.getLamOverR0()
    @property
    def half_light_radius(self): return self.lam_over_r0 * Kolmogorov._hlr_factor
    @property
    def fwhm(self): return self.lam_over_r0 * Kolmogorov._fwhm_factor

    def __eq__(self, other):
        return (isinstance(other, galsim.Kolmogorov) and
                self.lam_over_r0 == other.lam_over_r0 and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

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

_galsim.SBKolmogorov.__getinitargs__ = lambda self: (
        self.getLamOverR0(), self.getFlux(), self.getGSParams())
