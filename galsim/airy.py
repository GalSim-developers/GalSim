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
"""@file airy.py
"""

import galsim

from . import _galsim
from .gsobject import GSObject


class Airy(GSObject):
    """A class describing the surface brightness profile for an Airy disk (perfect
    diffraction-limited PSF for a circular aperture), with an optional central obscuration.

    For more information, refer to

        http://en.wikipedia.org/wiki/Airy_disc

    Initialization
    --------------

    The Airy profile is defined in terms of the diffraction angle, which is a function of the
    ratio lambda / D, where lambda is the wavelength of the light (say in the middle of the
    bandpass you are using) and D is the diameter of the telescope.

    The natural units for this value is radians, which is not normally a convenient unit to use for
    other GSObject dimensions.  Assuming that the other sky coordinates you are using are all in
    arcsec (e.g. the pixel scale when you draw the image, the size of the galaxy, etc.), then you
    should convert this to arcsec as well:

        >>> lam = 700  # nm
        >>> diam = 4.0    # meters
        >>> lam_over_diam = (lam * 1.e-9) / diam  # radians
        >>> lam_over_diam *= 206265  # Convert to arcsec
        >>> airy = galsim.Airy(lam_over_diam)

    To make this process a bit simpler, we recommend instead providing the wavelength and diameter
    separately using the parameters `lam` (in nm) and `diam` (in m).  GalSim will then convert this
    to any of the normal kinds of angular units using the `scale_unit` parameter:

        >>> airy = galsim.Airy(lam=lam, diam=diam, scale_unit=galsim.arcsec)

    When drawing images, the scale_unit should match the unit used for the pixel scale or the WCS.
    e.g. in this case, a pixel scale of 0.2 arcsec/pixel would be specified as `pixel_scale=0.2`.

    @param lam_over_diam    The parameter that governs the scale size of the profile.
                            See above for details about calculating it.
    @param lam              Lambda (wavelength) in units of nanometers.  Must be supplied with
                            `diam`, and in this case, image scales (`scale`) should be specified in
                            units of `scale_unit`.
    @param diam             Telescope diameter in units of meters.  Must be supplied with
                            `lam`, and in this case, image scales (`scale`) should be specified in
                            units of `scale_unit`.
    @param obscuration      The linear dimension of a central obscuration as a fraction of the
                            pupil dimension.  [default: 0]
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param scale_unit       Units to use for the sky coordinates when calculating lam/diam if these
                            are supplied separately.  Note that the results of calling methods like
                            getFWHM() will be returned in units of `scale_unit` as well.  Should be
                            either a galsim.AngleUnit or a string that can be used to construct one
                            (e.g., 'arcsec', 'radians', etc.).  [default: galsim.arcsec]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, Airy has the following access properties:

        >>> lam_over_diam = airy_obj.lam_over_diam
        >>> fwhm = airy_obj.fwhm
        >>> hlr = airy_obj.half_light_radius

    The latter two are only available if the obscuration is 0.
    """
    _req_params = { }
    _opt_params = { "flux" : float , "obscuration" : float, "diam" : float,
                    "scale_unit" : str }
    # Note that this is not quite right; it's true that either lam_over_diam or lam should be
    # supplied, but if lam is supplied then diam is required.  Errors in which parameters are used
    # may be caught either by config or by the python code itself, depending on the particular
    # error.
    _single_params = [{ "lam_over_diam" : float , "lam" : float } ]
    _takes_rng = False

    # For an unobscured Airy, we have the following factor which can be derived using the
    # integral result given in the Wikipedia page (http://en.wikipedia.org/wiki/Airy_disk),
    # solved for half total flux using the free online tool Wolfram Alpha.
    # At www.wolframalpha.com:
    # Type "Solve[BesselJ0(x)^2+BesselJ1(x)^2=1/2]" ... and divide the result by pi
    _hlr_factor = 0.5348321477242647
    _fwhm_factor = 1.028993969962188

    def __init__(self, lam_over_diam=None, lam=None, diam=None, obscuration=0., flux=1.,
                 scale_unit=galsim.arcsec, gsparams=None):
        # Parse arguments: either lam_over_diam in arbitrary units, or lam in nm and diam in m.
        # If the latter, then get lam_over_diam in units of `scale_unit`, as specified in
        # docstring.
        if lam_over_diam is not None:
            if lam is not None or diam is not None:
                raise TypeError("If specifying lam_over_diam, then do not specify lam or diam")
        else:
            if lam is None or diam is None:
                raise TypeError("If not specifying lam_over_diam, then specify lam AND diam")
            # In this case we're going to use scale_unit, so parse it in case of string input:
            if isinstance(scale_unit, str):
                scale_unit = galsim.AngleUnit.from_name(scale_unit)
            lam_over_diam = (1.e-9*lam/diam)*(galsim.radians/scale_unit)

        self._gsparams = galsim.GSParams.check(gsparams)
        self._sbp = _galsim.SBAiry(lam_over_diam, obscuration, flux, self.gsparams._gsp)

    @property
    def lam_over_diam(self): return self._sbp.getLamOverD()
    @property
    def obscuration(self): return self._sbp.getObscuration()

    @property
    def half_light_radius(self):
        if self.obscuration == 0.:
            return self.lam_over_diam * Airy._hlr_factor
        else:
            # In principle can find the half light radius as a function of lam_over_diam and
            # obscuration too, but it will be much more involved...!
            raise NotImplementedError("Half light radius calculation not implemented for Airy "+
                                      "objects with non-zero obscuration.")

    @property
    def fwhm(self):
        # As above, likewise, FWHM only easy to define for unobscured Airy
        if self.obscuration == 0.:
            return self.lam_over_diam * Airy._fwhm_factor
        else:
            # In principle can find the FWHM as a function of lam_over_diam and obscuration too,
            # but it will be much more involved...!
            raise NotImplementedError("FWHM calculation not implemented for Airy "+
                                      "objects with non-zero obscuration.")


    def __eq__(self, other):
        return (isinstance(other, galsim.Airy) and
                self.lam_over_diam == other.lam_over_diam and
                self.obscuration == other.obscuration and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Airy", self.lam_over_diam, self.obscuration, self.flux,
                     self.gsparams))

    def __repr__(self):
        return 'galsim.Airy(lam_over_diam=%r, obscuration=%r, flux=%r, gsparams=%r)'%(
            self.lam_over_diam, self.obscuration, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Airy(lam_over_diam=%s'%self.lam_over_diam
        if self.obscuration != 0.:
            s += ', obscuration=%s'%self.obscuration
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBAiry.__getinitargs__ = lambda self: (
        self.getLamOverD(), self.getObscuration(), self.getFlux(), self.getGSParams())
