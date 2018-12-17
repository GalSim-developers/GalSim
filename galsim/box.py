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
from .errors import convert_cpp_errors


class Box(GSObject):
    """A class describing a box profile.  This is just a 2D top-hat function, where the
    width and height are allowed to be different.

    Initialization
    --------------

    @param width            The width of the Box.
    @param height           The height of the Box.
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, Box has the following access properties:

        >>> width = box.width
        >>> height = box.height

    """
    _req_params = { "width" : float, "height" : float }
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False

    _has_hard_edges = True
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, width, height, flux=1., gsparams=None):
        self._width = float(width)
        self._height = float(height)
        self._flux = float(flux)
        self._gsparams = GSParams.check(gsparams)
        self._norm = self._flux / (self._width * self._height)

    @lazy_property
    def _sbp(self):
        with convert_cpp_errors():
            return _galsim.SBBox(self._width, self._height, self._flux, self.gsparams._gsp)

    @property
    def width(self): return self._width
    @property
    def height(self): return self._height

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Box) and
                 self.width == other.width and
                 self.height == other.height and
                 self.flux == other.flux and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.Box", self.width, self.height, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Box(width=%r, height=%r, flux=%r, gsparams=%r)'%(
            self.width, self.height, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Box(width=%s, height=%s'%(self.width, self.height)
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
        return 2. / (self.gsparams.maxk_threshold * min(self.width, self.height))

    @property
    def _stepk(self):
        return math.pi / max(self.width, self.height)

    @property
    def _max_sb(self):
        return self._norm

    @doc_inherit
    def _xValue(self, pos):
        if 2.*abs(pos.x) < self._width and 2.*abs(pos.y) < self._height:
            return self._norm
        else:
            return 0.

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


class Pixel(Box):
    """A class describing a pixel profile.  This is just a 2D square top-hat function.

    This class is typically used to represent a pixel response function.  It is used internally by
    the drawImage() function, but there may be cases where the user would want to use this profile
    directly.

    Initialization
    --------------

    @param scale            The linear scale size of the pixel.  Typically given in arcsec.
    @param flux             The flux (in photons/cm^2/s) of the profile.  This should almost
                            certainly be left at the default value of 1. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, Pixel has the following access property:

        >>> scale = pixel.scale

    """
    _req_params = { "scale" : float }
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False

    def __init__(self, scale, flux=1., gsparams=None):
        super(Pixel, self).__init__(width=scale, height=scale, flux=flux, gsparams=gsparams)

    @property
    def scale(self): return self.width

    def __repr__(self):
        return 'galsim.Pixel(scale=%r, flux=%r, gsparams=%r)'%(
            self.scale, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Pixel(scale=%s'%self.scale
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s


class TopHat(GSObject):
    """A class describing a radial tophat profile.  This profile is a constant value within some
    radius, and zero outside this radius.

    Initialization
    --------------

    @param radius           The radius of the TopHat, where the surface brightness drops to 0.
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, TopHat has the following access property:

        >>> radius = tophat.radius

    """
    _req_params = { "radius" : float }
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False

    _has_hard_edges = True
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, radius, flux=1., gsparams=None):
        self._radius = float(radius)
        self._flux = float(flux)
        self._gsparams = GSParams.check(gsparams)
        self._rsq = self._radius**2
        self._norm = self._flux / (math.pi * self._rsq)

    @lazy_property
    def _sbp(self):
        with convert_cpp_errors():
            return _galsim.SBTopHat(self._radius, self._flux, self.gsparams._gsp)

    @property
    def radius(self): return self._radius

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, TopHat) and
                 self.radius == other.radius and
                 self.flux == other.flux and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.TopHat", self.radius, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.TopHat(radius=%r, flux=%r, gsparams=%r)'%(
            self.radius, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.TopHat(radius=%s'%self.radius
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
        return math.pi / self._radius

    @property
    def _max_sb(self):
        return self._norm

    @doc_inherit
    def _xValue(self, pos):
        rsq = pos.x**2 + pos.y**2
        if rsq < self._rsq:
            return self._norm
        else:
            return 0.

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
