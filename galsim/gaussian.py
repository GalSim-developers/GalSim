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


class Gaussian(GSObject):
    """A class describing a 2D Gaussian surface brightness profile.

    The Gaussian surface brightness profile is characterized by two properties, its `flux`
    and the characteristic size `sigma` where the radial profile of the circular Gaussian
    drops off as `exp[-r^2 / (2. * sigma^2)]`.

    Initialization
    --------------

    A Gaussian can be initialized using one (and only one) of three possible size parameters:
    `sigma`, `fwhm`, or `half_light_radius`.  Exactly one of these three is required.

    @param sigma            The value of sigma of the profile.  Typically given in arcsec.
                            [One of `sigma`, `fwhm`, or `half_light_radius` is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of `sigma`, `fwhm`, or `half_light_radius` is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `sigma`, `fwhm`, or `half_light_radius` is required.]
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, Gaussian has the following access properties:

        >>> sigma = gauss.sigma
        >>> fwhm = gauss.fwhm
        >>> hlr = gauss.half_light_radius
    """
    # Initialization parameters of the object, with type information, to indicate
    # which attributes are allowed / required in a config file for this object.
    # _req_params are required
    # _opt_params are optional
    # _single_params are a list of sets for which exactly one in the list is required.
    # _takes_rng indicates whether the constructor should be given the current rng.
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = [ { "sigma" : float, "half_light_radius" : float, "fwhm" : float } ]
    _takes_rng = False

    # The FWHM of a Gaussian is 2 sqrt(2 ln2) sigma
    _fwhm_factor = 2.3548200450309493
    # The half-light-radius is sqrt(2 ln2) sigma
    _hlr_factor =  1.1774100225154747

    def __init__(self, half_light_radius=None, sigma=None, fwhm=None, flux=1., gsparams=None):
        if fwhm is not None :
            if sigma is not None or half_light_radius is not None:
                raise TypeError(
                        "Only one of sigma, fwhm, and half_light_radius may be " +
                        "specified for Gaussian")
            else:
                sigma = fwhm / Gaussian._fwhm_factor
        elif half_light_radius is not None:
            if sigma is not None:
                raise TypeError(
                        "Only one of sigma, fwhm, and half_light_radius may be " +
                        "specified for Gaussian")
            else:
                sigma = half_light_radius / Gaussian._hlr_factor
        elif sigma is None:
                raise TypeError(
                        "One of sigma, fwhm, or half_light_radius must be " +
                        "specified for Gaussian")

        self._gsparams = galsim.GSParams.check(gsparams)
        self._sbp = _galsim.SBGaussian(sigma, flux, self.gsparams._gsp)

    @property
    def sigma(self): return self._sbp.getSigma()
    @property
    def half_light_radius(self): return self.sigma * Gaussian._hlr_factor
    @property
    def fwhm(self): return self.sigma * Gaussian._fwhm_factor

    def __eq__(self, other):
        return (isinstance(other, galsim.Gaussian) and
                self.sigma == other.sigma and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Gaussian", self.sigma, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Gaussian(sigma=%r, flux=%r, gsparams=%r)'%(
            self.sigma, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Gaussian(sigma=%s'%self.sigma
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBGaussian.__getinitargs__ = lambda self: (
        self.getSigma(), self.getFlux(), self.getGSParams())
