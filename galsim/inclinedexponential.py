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
"""@file inclinedexponential.py

InclinedExponential is a class representing an exponential profile inclined to the LOS.
"""

from galsim import GSObject
import galsim

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

    A profile can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required. Similarly,
    at most one of `scale_height' and `scale_h_over_r' is required; if neither is given, the
    default of scale_h_over_r = 0.1 will be used. Note that if half_light_radius and
    scale_h_over_r are supplied (or the default value of scale_h_over_r is used),
    scale_h_over_r will be assumed to refer to the scale radius, not the half-light radius.

    Initialization
    --------------

    @param inclination          The inclination angle, which must be a galsim.Angle instance
    @param scale_radius         The scale radius of the exponential disk.  Typically given in
                                arcsec. This can be compared to the 'scale_radius' parameter of the
                                galsim.Exponential class, and in the face-on case, the same scale
                                radius will result in the same 2D light distribution as with that
                                class.
    @param half_light_radius    The half-light radius of the exponential disk, as an alternative to
                                the scale radius.
    @param scale_height         The scale height of the exponential disk.  Typically given in arcsec.
                                [default: None]
    @param scale_h_over_r       In lieu of the scale height, you may also specify the ratio of the
                                scale height to the scale radius. [default: 0.1]
    @param flux                 The flux (in photons) of the profile. [default: 1]
    @param gsparams             An optional GSParams argument.  See the docstring for GSParams for
                                details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, InclinedExponential has the following access
    properties:

        >>> inclination = inclined_exponential_obj.inclination
        >>> r0 = inclined_exponential_obj.scale_radius
        >>> rh = inclined_exponential_obj.half_light_radius
        >>> h0 = inclined_exponential_obj.scale_height
    """
    _req_params = { "inclination" : galsim.Angle }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _opt_params = { "scale_height" : float, "scale_h_over_r" : float, "flux" : float }
    _takes_rng = False

    def __init__(self, inclination, half_light_radius=None, scale_radius=None, scale_height=None,
                 scale_h_over_r=None, flux=1., gsparams=None):

        # Check that the scale/half-light radius is valid
        if scale_radius is not None:
            if not scale_radius > 0.:
                raise ValueError("scale_radius must be > zero.")
        elif half_light_radius is not None:
            if not half_light_radius > 0.:
                raise ValueError("half_light_radius must be > zero.")
        else:
            raise TypeError(
                    "Either scale_radius or half_light_radius must be " +
                    "specified for InclinedExponential")

        # Check that we have exactly one of scale_radius and half_light_radius,
        # then get scale_radius
        if half_light_radius is not None:
            if scale_radius is not None:
                raise TypeError(
                        "Only one of scale_radius and half_light_radius may be " +
                        "specified for InclinedExponential")
            else:
                # Use the factor from the Exponential class
                scale_radius = half_light_radius / galsim.Exponential._hlr_factor

        # Check that the height specification is valid
        if scale_height is not None:
            if not scale_height > 0.:
                raise ValueError("scale_height must be > zero.")
        elif scale_h_over_r is not None:
            if not scale_h_over_r > 0.:
                raise ValueError("half_light_radius must be > zero.")
        else:
            # Use the default scale_h_over_r
            scale_h_over_r = 0.1

        # Check that we have exactly one of scale_height and scale_h_over_r,
        # then get scale_height
        if scale_h_over_r is not None:
            if scale_height is not None:
                raise TypeError(
                        "Only one of scale_height and scale_h_over_r may be " +
                        "specified for InclinedExponential")
            else:
                scale_height = scale_radius * scale_h_over_r

        # Explicitly check for angle type, so we can give more informative error if eg. a float is
        # passed
        if not isinstance(inclination, galsim.Angle):
            raise TypeError("Input inclination should be an Angle")

        self._sbp = _galsim.SBInclinedExponential(
                inclination, scale_radius, scale_height, flux, gsparams)
        self._gsparams = gsparams

    def getInclination(self):
        """Return the inclination angle for this profile as a galsim.Angle instance.
        """
        from .deprecated import depr
        depr("inclined_exp.getInclination()", 1.5, "inclined_exp.inclination")
        return self.inclination

    def getScaleRadius(self):
        """Return the scale radius for this profile.
        """
        from .deprecated import depr
        depr("inclined_exp.getScaleRadius()", 1.5, "inclined_exp.scale_radius")
        return self.scale_radius

    def getHalfLightRadius(self):
        """Return the half light radius for this Exponential profile.
        """
        from .deprecated import depr
        depr("inclined_exp.getHalfLightRadius()", 1.5, "inclined_exp.half_light_radius")
        return self.half_light_radius

    def getScaleHeight(self):
        """Return the scale height for this profile.
        """
        from .deprecated import depr
        depr("inclined_exp.getScaleHeight()", 1.5, "inclined_exp.scale_height")
        return self.scale_height

    def getScaleHOverR(self):
        """Return the scale height over scale radius for this profile.
        """
        from .deprecated import depr
        depr("inclined_exp.getScaleHOverR()", 1.5, "inclined_exp.scale_h_over_r")
        return self.scale_h_over_r

    @property
    def inclination(self): return self._sbp.getInclination()
    @property
    def scale_radius(self): return self._sbp.getScaleRadius()
    @property
    def half_light_radius(self): return self.scale_radius * galsim.Exponential._hlr_factor
    @property
    def scale_height(self): return self._sbp.getScaleHeight()
    @property
    def scale_h_over_r(self): return self.scale_height / self.scale_radius

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
        return ('galsim.InclinedExponential(inclination=%r, scale_radius=%r, scale_height=%r, ' +
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
