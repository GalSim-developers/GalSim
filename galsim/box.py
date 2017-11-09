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


class Pixel(GSObject):
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
        self._gsparams = galsim.GSParams.check(gsparams)
        self._sbp = _galsim.SBBox(scale, scale, flux, self.gsparams._gsp)

    @property
    def scale(self): return self._sbp.getWidth()

    def __eq__(self, other):
        return (isinstance(other, galsim.Pixel) and
                self.scale == other.scale and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Pixel", self.scale, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Pixel(scale=%r, flux=%r, gsparams=%r)'%(
            self.scale, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Pixel(scale=%s'%self.scale
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s


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

    def __init__(self, width, height, flux=1., gsparams=None):
        width = float(width)
        height = float(height)
        self._gsparams = galsim.GSParams.check(gsparams)
        self._sbp = _galsim.SBBox(width, height, flux, self.gsparams._gsp)

    @property
    def width(self): return self._sbp.getWidth()
    @property
    def height(self): return self._sbp.getHeight()

    def __eq__(self, other):
        return (isinstance(other, galsim.Box) and
                self.width == other.width and
                self.height == other.height and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

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

_galsim.SBBox.__getinitargs__ = lambda self: (
        self.getWidth(), self.getHeight(), self.getFlux(), self.getGSParams())


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

    def __init__(self, radius, flux=1., gsparams=None):
        radius = float(radius)
        self._gsparams = galsim.GSParams.check(gsparams)
        self._sbp = _galsim.SBTopHat(radius, flux=flux, gsparams=self.gsparams._gsp)

    @property
    def radius(self): return self._sbp.getRadius()

    def __eq__(self, other):
        return (isinstance(other, galsim.TopHat) and
                self.radius == other.radius and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

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

_galsim.SBTopHat.__getinitargs__ = lambda self: (
        self.getRadius(), self.getFlux(), self.getGSParams())
