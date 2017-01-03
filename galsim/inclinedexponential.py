# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
"""@file inclinedexponential.py

InclinedExponential is a class representing an exponential profile inclined to the LOS.
"""

import galsim
from galsim import GSObject
from . import _galsim


class InclinedExponential(GSObject):
    """A class describing an inclined exponential profile.

    The Inclined Exponential surface brightness profile is characterized by three properties: its
    inclination angle (where 0 degrees = face-on and 90 degrees = edge-on), its scale radius, and
    its scale height. The 3D light distribution function is:

        I(R,z) = I_0 / (2h_s) * sech^2 (z/h_s) * exp(-R/R_s)

    where z is the distance along the minor axis, R is the radial distance from the minor axis,
    R_s is the scale radius of the disk, h_s is the scale height of the disk, and I_0 is the central
    surface brightness of the face-on disk. The 2D light distribution function is then determined
    from the scale height and radius here, along with the inclination angle.

    In this implementation, the profile is inclined along the y-axis. This means that it will likely
    need to be rotated in most circumstances.

    At present, this profile is not enabled for photon-shooting.

    Initialization
    --------------

    @param inclination      The inclination angle, which must be a galsim.Angle instance
    @param scale_radius     The scale radius of the exponential disk.  Typically given in arcsec.
                            This can be compared to the 'scale_radius' parameter of the
                            galsim.Exponential class, and in the face-on case, the same same scale
                            radius will result in the same 2D light distribution as with that
                            class.
    @param scale_height     The scale height of the exponential disk.  Typically given in arcsec.
                            [default: None]
    @param scale_h_over_r   In lieu of the scale height, you may also specify the ratio of the
                            scale height to the scale radius. [default: 0.1]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, InclinedExponential has the following access methods:

        >>> inclination = inclined_exponential_obj.getInclination()
        >>> r0 = inclined_exponential_obj.getScaleRadius()
        >>> h0 = inclined_exponential_obj.getScaleHeight()
    """
    _req_params = { "inclination" : galsim.Angle, "scale_radius" : float }
    _opt_params = { "scale_height" : float, "scale_h_over_r" : float, "flux" : float }
    _takes_rng = False

    def __init__(self, inclination, scale_radius, scale_height=None, scale_h_over_r=0.1,
                 flux=1., gsparams=None):

        if scale_height is None:
            scale_height = scale_h_over_r * scale_radius

        # Explicitly check for angle type, so we can give more informative error if eg. a float is
        # passed
        if not isinstance(inclination, galsim.Angle):
            raise TypeError("Input inclination should be an Angle")

        self._inclination = inclination
        GSObject.__init__(self, _galsim.SBInclinedExponential(
                inclination.rad(), scale_radius, scale_height, flux, gsparams))
        self._gsparams = gsparams

    def getInclination(self):
        """Return the inclination angle for this profile as a galsim.Angle instance.
        """
        return self._inclination

    def getScaleRadius(self):
        """Return the scale radius for this profile.
        """
        return self.SBProfile.getScaleRadius()

    def getScaleHeight(self):
        """Return the scale height for this profile.
        """
        return self.SBProfile.getScaleHeight()

    def getScaleHOverR(self):
        """Return the scale height over scale radius for this profile.
        """
        return self.SBProfile.getScaleHeight()/self.SBProfile.getScaleRadius()

    @property
    def inclination(self): return self._inclination
    @property
    def scale_radius(self): return self.getScaleRadius()
    @property
    def scale_height(self): return self.getScaleHeight()
    @property
    def scale_h_over_r(self): return self.getScaleHOverR()

    def __eq__(self, other):
        return ((isinstance(other, galsim.InclinedExponential) and
                 (self.inclination == other.inclination) and
                 (self.scale_radius == other.scale_radius) and
                 (self.scale_height == other.scale_height) and
                 (self.flux == other.flux) and
                 (self._gsparams == other._gsparams)))

    def __hash__(self):
        return hash(("galsim.InclinedExponential", self.inclination, self.scale_radius,
                    self.scale_height, self.flux, self._gsparams))

    def __repr__(self):
        return ('galsim.InclinedExponential(inclination=%r, scale_radius=%r, scale_height=%r, '+
               'flux=%r, gsparams=%r)') % (
            self.inclination, self.scale_radius, self.scale_height, self.flux, self._gsparams)

    def __str__(self):
        s = 'galsim.InclinedExponential(inclination=%s, scale_radius=%s, scale_height=%s' % (
                self.inclination, self.scale_radius, self.scale_height)
        if self.flux != 1.0:
            s += ', flux=%s' % self.flux
        s += ')'
        return s

_galsim.SBInclinedExponential.__getinitargs__ = lambda self: (
        self.getInclination(), self.getScaleRadius(), self.getScaleHeight(), self.getFlux(),
        self.getGSParams())
_galsim.SBInclinedExponential.__getstate__ = lambda self: None
_galsim.SBInclinedExponential.__repr__ = lambda self: \
        'galsim._galsim.SBInclinedExponential(%r, %r, %r, %r, %r)' % self.__getinitargs__()
