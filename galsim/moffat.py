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
from .errors import GalSimRangeError, GalSimIncompatibleValuesError, convert_cpp_errors


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

    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    # The conversion from hlr or fwhm to scale radius is complicated for Moffat, especially
    # since we allow it to be truncated, which matters for hlr.  So we do these calculations
    # in the C++-layer constructor.
    def __init__(self, beta, scale_radius=None, half_light_radius=None, fwhm=None, trunc=0.,
                 flux=1., gsparams=None):
        self._beta = float(beta)
        self._trunc = float(trunc)
        self._flux = float(flux)
        self._gsparams = GSParams.check(gsparams)

        if self._trunc == 0. and self._beta <= 1.1:
            raise GalSimRangeError("Moffat profiles with beta <= 1.1 must be truncated",
                                   beta, 1.1)
        if self._trunc < 0.:
            raise GalSimRangeError("Moffat trunc must be >= 0", self._trunc, 0.)

        # Parse the radius options
        if half_light_radius is not None:
            if scale_radius is not None or fwhm is not None:
                raise GalSimIncompatibleValuesError(
                    "Only one of scale_radius, half_light_radius, or fwhm may be specified",
                    half_light_radius=half_light_radius, scale_radius=scale_radius, fwhm=fwhm)
            self._hlr = float(half_light_radius)
            if self._trunc > 0. and self._trunc <= math.sqrt(2.) * self._hlr:
                raise GalSimRangeError("Moffat trunc must be > sqrt(2) * half_light_radius.",
                                       self._trunc, math.sqrt(2.) * self._hlr)
            with convert_cpp_errors():
                self._r0 = _galsim.MoffatCalculateSRFromHLR(self._hlr, self._trunc, self._beta)
            self._fwhm = 0.
        elif fwhm is not None:
            if scale_radius is not None:
                raise GalSimIncompatibleValuesError(
                    "Only one of scale_radius, half_light_radius, or fwhm may be specified",
                    half_light_radius=half_light_radius, scale_radius=scale_radius, fwhm=fwhm)
            self._fwhm = float(fwhm)
            self._r0 = self._fwhm / (2. * math.sqrt(2.**(1./self._beta) - 1.))
            self._hlr = 0.
        elif scale_radius is not None:
            self._r0 = float(scale_radius)
            self._hlr = 0.
            self._fwhm = 0.
        else:
            raise GalSimIncompatibleValuesError(
                "One of scale_radius, half_light_radius, or fwhm must be specified",
                half_light_radius=half_light_radius, scale_radius=scale_radius, fwhm=fwhm)

    @lazy_property
    def _sbp(self):
        with convert_cpp_errors():
            return _galsim.SBMoffat(self._beta, self._r0, self._trunc, self._flux,
                                    self.gsparams._gsp)

    def getFWHM(self):
        """Return the FWHM for this Moffat profile.
        """

    def getHalfLightRadius(self):
        """Return the half light radius for this Moffat profile.
        """

    @property
    def beta(self): return self._beta
    @property
    def scale_radius(self): return self._r0
    @property
    def trunc(self): return self._trunc

    @property
    def half_light_radius(self):
        if self._hlr == 0.:
            self._hlr = self._sbp.getHalfLightRadius()
        return self._hlr

    @lazy_property
    def fwhm(self):
        if self._fwhm == 0.:
            self._fwhm = self._r0 * (2. * math.sqrt(2.**(1./self._beta) - 1.))
        return self._fwhm

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Moffat) and
                 self.beta == other.beta and
                 self.scale_radius == other.scale_radius and
                 self.trunc == other.trunc and
                 self.flux == other.flux and
                 self.gsparams == other.gsparams))

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
    def _has_hard_edges(self):
        return self._trunc != 0.

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

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
