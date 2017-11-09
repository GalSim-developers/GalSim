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


class Moffat(GSObject):
    """A class describing a Moffat surface brightness profile.

    The Moffat surface brightness profile is I(R) ~ [1 + (r/scale_radius)^2]^(-beta).  The
    GalSim representation of a Moffat profile also includes an optional truncation beyond a given
    radius.

    For more information, refer to

        http://home.fnal.gov/~neilsen/notebook/astroPSF/astroPSF.html

    Initialization
    --------------

    A Moffat can be initialized using one (and only one) of three possible size parameters:
    `scale_radius`, `fwhm`, or `half_light_radius`.  Exactly one of these three is required.

    @param beta             The `beta` parameter of the profile.
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius`, `fwhm`, or `half_light_radius` is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius`, `fwhm`, or `half_light_radius` is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of `scale_radius`, `fwhm`, or `half_light_radius` is required.]
    @param trunc            An optional truncation radius at which the profile is made to drop to
                            zero, in the same units as the size parameter.
                            [default: 0, indicating no truncation]
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, Moffat has the following access properties:

        >>> beta = moffat_obj.beta
        >>> rD = moffat_obj.scale_radius
        >>> fwhm = moffat_obj.fwhm
        >>> hlr = moffat_obj.half_light_radius
    """
    _req_params = { "beta" : float }
    _opt_params = { "trunc" : float , "flux" : float }
    _single_params = [ { "scale_radius" : float, "half_light_radius" : float, "fwhm" : float } ]
    _takes_rng = False

    # The conversion from hlr or fwhm to scale radius is complicated for Moffat, especially
    # since we allow it to be truncated, which matters for hlr.  So we do these calculations
    # in the C++-layer constructor.
    def __init__(self, beta, scale_radius=None, half_light_radius=None, fwhm=None, trunc=0.,
                 flux=1., gsparams=None):
        self._gsparams = galsim.GSParams.check(gsparams)
        self._sbp = _galsim.SBMoffat(beta, scale_radius, half_light_radius, fwhm,
                                     trunc, flux, self.gsparams._gsp)

    @property
    def beta(self): return self._sbp.getBeta()
    @property
    def scale_radius(self): return self._sbp.getScaleRadius()
    @property
    def half_light_radius(self): return self._sbp.getHalfLightRadius()
    @property
    def fwhm(self): return self._sbp.getFWHM()
    @property
    def trunc(self): return self._sbp.getTrunc()

    def __eq__(self, other):
        return (isinstance(other, galsim.Moffat) and
                self.beta == other.beta and
                self.scale_radius == other.scale_radius and
                self.trunc == other.trunc and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Moffat", self.beta, self.scale_radius, self.trunc, self.flux,
                     self.gsparams))

    def __repr__(self):
        return 'galsim.Moffat(beta=%r, scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)'%(
            self.beta, self.scale_radius, self.trunc, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Moffat(beta=%s, scale_radius=%s'%(self.beta, self.scale_radius)
        if self.trunc != 0.:
            s += ', trunc=%s'%self.trunc
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBMoffat.__getinitargs__ = lambda self: (
        self.getBeta(), self.getScaleRadius(), None, None, self.getTrunc(),
        self.getFlux(), self.getGSParams())
