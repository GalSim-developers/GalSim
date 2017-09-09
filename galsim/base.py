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
"""@file base.py
This file implements many of the basic kinds of surface brightness profiles in GalSim:

    Kolmogorov
    Pixel
    Box
    TopHat
    Sersic
    Exponential
    DeVaucouleurs
    Spergel
    DeltaFunction

These are all relatively simple profiles, most being radially symmetric.  They are all subclasses
of GSObject, which defines much of the top-level interface to these objects.  See gsobject.py for
details about the GSObject class.

For a description of units conventions for scale radii for our base classes see
`doc/GalSim_Quick_Reference.pdf`, section 2.2.  In short, any system that will ensure consistency
between the scale radii used to specify the size of the GSObject and between the pixel scale of the
Image is acceptable.
"""

import numpy as np
import math

from . import _galsim
from .gsobject import GSObject
from .gsparams import GSParams
from .angle import arcsec, radians, AngleUnit

class Sersic(GSObject):
    """A class describing a Sersic profile.

    The Sersic surface brightness profile is characterized by three properties: its Sersic index
    `n`, its `flux`, and either the `half_light_radius` or `scale_radius`.  Given these properties,
    the surface brightness profile scales as I(r) ~ exp[-(r/scale_radius)^{1/n}], or
    I(r) ~ exp[-b*(r/half_light_radius)^{1/n}] (where b is calculated to give the right
    half-light radius).

    For more information, refer to

        http://en.wikipedia.org/wiki/Sersic_profile

    Initialization
    --------------

    The allowed range of values for the `n` parameter is 0.3 <= n <= 6.2.  An exception will be
    thrown if you provide a value outside that range.  Below n=0.3, there are severe numerical
    problems.  Above n=6.2, we found that the code begins to be inaccurate when sheared or
    magnified (at the level of upcoming shear surveys), so we do not recommend extending beyond
    this.  See Issues #325 and #450 for more details.

    Sersic profile calculations take advantage of Hankel transform tables that are precomputed for a
    given value of n when the Sersic profile is initialized.  Making additional objects with the
    same n can therefore be many times faster than making objects with different values of n that
    have not been used before.  Moreover, these Hankel transforms are only cached for a maximum of
    100 different n values at a time.  For this reason, for large sets of simulations, it is worth
    considering the use of only discrete n values rather than allowing it to vary continuously.  For
    more details, see https://github.com/GalSim-developers/GalSim/issues/566.

    Note that if you are building many Sersic profiles using truncation, the code will be more
    efficient if the truncation is always the same multiple of `scale_radius`, since it caches
    many calculations that depend on the ratio `trunc/scale_radius`.

    A Sersic can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param n                The Sersic index, n.
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param trunc            An optional truncation radius at which the profile is made to drop to
                            zero, in the same units as the size parameter.
                            [default: 0, indicating no truncation]
    @param flux_untruncated Should the provided `flux` and `half_light_radius` refer to the
                            untruncated profile? See below for more details. [default: False]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Flux of a truncated profile
    ---------------------------

    If you are truncating the profile, the optional parameter, `flux_untruncated`, specifies
    whether the `flux` and `half_light_radius` specifications correspond to the untruncated
    profile (`True`) or to the truncated profile (`False`, default).  The impact of this parameter
    is a little subtle, so we'll go through a few examples to show how it works.

    First, let's examine the case where we specify the size according to the half-light radius.
    If `flux_untruncated` is True (and `trunc > 0`), then the profile will be identical
    to the version without truncation up to the truncation radius, beyond which it drops to 0.
    In this case, the actual half-light radius will be different from the specified half-light
    radius.  The half_light_radius property will return the true half-light radius.  Similarly,
    the actual flux will not be the same as the specified value; the true flux is also returned
    by the flux property.

    Example:

        >>> sersic_obj1 = galsim.Sersic(n=3.5, half_light_radius=2.5, flux=40.)
        >>> sersic_obj2 = galsim.Sersic(n=3.5, half_light_radius=2.5, flux=40., trunc=10.)
        >>> sersic_obj3 = galsim.Sersic(n=3.5, half_light_radius=2.5, flux=40., trunc=10., \\
                                        flux_untruncated=True)

        >>> sersic_obj1.xValue(galsim.PositionD(0.,0.))
        237.3094228615618
        >>> sersic_obj2.xValue(galsim.PositionD(0.,0.))
        142.54505376530574    # Normalization and scale radius adjusted (same half-light radius)
        >>> sersic_obj3.xValue(galsim.PositionD(0.,0.))
        237.30942286156187

        >>> sersic_obj1.xValue(galsim.PositionD(10.0001,0.))
        0.011776164687304694
        >>> sersic_obj2.xValue(galsim.PositionD(10.0001,0.))
        0.0
        >>> sersic_obj3.xValue(galsim.PositionD(10.0001,0.))
        0.0

        >>> sersic_obj1.half_light_radius
        2.5
        >>> sersic_obj2.half_light_radius
        2.5
        >>> sersic_obj3.half_light_radius
        1.9795101383056892    # The true half-light radius is smaller than the specified value

        >>> sersic_obj1.flux
        40.0
        >>> sersic_obj2.flux
        40.0
        >>> sersic_obj3.flux
        34.56595186009519     # Flux is missing due to truncation

        >>> sersic_obj1.scale_radius
        0.003262738739834598
        >>> sersic_obj2.scale_radius
        0.004754602453641744  # the scale radius needed adjustment to accommodate HLR
        >>> sersic_obj3.scale_radius
        0.003262738739834598  # the scale radius is still identical to the untruncated case

    When the truncated Sersic scale is specified with `scale_radius`, the behavior between the
    three cases (untruncated, `flux_untruncated=True` and `flux_untruncated=False`) will be
    somewhat different from above.  Since it is the scale radius that is being specified, and since
    truncation does not change the scale radius the way it can change the half-light radius, the
    scale radius will remain unchanged in all cases.  This also results in the half-light radius
    being the same between the two truncated cases (although different from the untruncated case).
    The flux normalization is the only difference between `flux_untruncated=True` and
    `flux_untruncated=False` in this case.

    Example:

        >>> sersic_obj1 = galsim.Sersic(n=3.5, scale_radius=0.05, flux=40.)
        >>> sersic_obj2 = galsim.Sersic(n=3.5, scale_radius=0.05, flux=40., trunc=10.)
        >>> sersic_obj3 = galsim.Sersic(n=3.5, scale_radius=0.05, flux=40., trunc=10., \\
                                        flux_untruncated=True)

        >>> sersic_obj1.xValue(galsim.PositionD(0.,0.))
        1.010507575186637
        >>> sersic_obj2.xValue(galsim.PositionD(0.,0.))
        5.786692612210923     # Normalization adjusted to accomodate the flux within trunc radius
        >>> sersic_obj3.xValue(galsim.PositionD(0.,0.))
        1.010507575186637

        >>> sersic_obj1.half_light_radius
        38.311372735390016
        >>> sersic_obj2.half_light_radius
        5.160062547614234
        >>> sersic_obj3.half_light_radius
        5.160062547614234     # For the truncated cases, the half-light radii are the same

        >>> sersic_obj1.flux
        40.0
        >>> sersic_obj2.flux
        40.0
        >>> sersic_obj3.flux
        6.985044085834393     # Flux is missing due to truncation

        >>> sersic_obj1.scale_radius
        0.05
        >>> sersic_obj2.scale_radius
        0.05
        >>> sersic_obj3.scale_radius
        0.05

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, Sersic has the following access properties:

        >>> n = sersic_obj.n
        >>> r0 = sersic_obj.scale_radius
        >>> hlr = sersic_obj.half_light_radius
    """
    _req_params = { "n" : float }
    _opt_params = { "flux" : float, "trunc" : float, "flux_untruncated" : bool }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False

    # The conversion from hlr to scale radius is complicated for Sersic, especially since we
    # allow it to be truncated.  So we do these calculations in the C++-layer constructor.
    def __init__(self, n, half_light_radius=None, scale_radius=None,
                 flux=1., trunc=0., flux_untruncated=False, gsparams=None):
        self._gsparams = GSParams.check(gsparams)
        self._sbp = _galsim.SBSersic(n, scale_radius, half_light_radius,flux, trunc,
                                     flux_untruncated, self.gsparams._gsp)

    @property
    def n(self): return self._sbp.getN()
    @property
    def scale_radius(self): return self._sbp.getScaleRadius()
    @property
    def half_light_radius(self): return self._sbp.getHalfLightRadius()
    @property
    def trunc(self): return self._sbp.getTrunc()

    def __eq__(self, other):
        return (isinstance(other, Sersic) and
                self.n == other.n and
                self.scale_radius == other.scale_radius and
                self.trunc == other.trunc and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.SBSersic", self.n, self.scale_radius, self.trunc, self.flux,
                     self.gsparams))

    def __repr__(self):
        return 'galsim.Sersic(n=%r, scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)'%(
            self.n, self.scale_radius, self.trunc, self.flux, self.gsparams)

    def __str__(self):
        # Note: for the repr, we use the scale_radius, since that should just flow as is through
        # the constructor, so it should be exact.  But most people use half_light_radius
        # for Sersics, so use that in the looser str() function.
        s = 'galsim.Sersic(n=%s, half_light_radius=%s'%(self.n, self.half_light_radius)
        if self.trunc != 0.:
            s += ', trunc=%s'%self.trunc
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBSersic.__getinitargs__ = lambda self: (
        self.getN(), self.getScaleRadius(), None, self.getFlux(), self.getTrunc(),
        False, self.getGSParams())


class Exponential(GSObject):
    """A class describing an exponential profile.

    Surface brightness profile with I(r) ~ exp[-r/scale_radius].  This is a special case of
    the Sersic profile, but is given a separate class since the Fourier transform has closed form
    and can be generated without lookup tables.

    Initialization
    --------------

    An Exponential can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, Exponential has the following access properties:

        >>> r0 = exp_obj.scale_radius
        >>> hlr = exp_obj.half_light_radius
    """
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False

    # The half-light radius Factor is not analytic, but can be calculated by iterative solution
    # of the equation:
    #     (re / r0) = ln[(re / r0) + 1] + ln(2)
    _hlr_factor = 1.6783469900166605

    def __init__(self, half_light_radius=None, scale_radius=None, flux=1., gsparams=None):
        if half_light_radius is not None:
            if scale_radius is not None:
                raise TypeError(
                        "Only one of scale_radius and half_light_radius may be " +
                        "specified for Exponential")
            else:
                scale_radius = half_light_radius / Exponential._hlr_factor
        elif scale_radius is None:
                raise TypeError(
                        "Either scale_radius or half_light_radius must be " +
                        "specified for Exponential")
        self._gsparams = GSParams.check(gsparams)
        self._sbp = _galsim.SBExponential(scale_radius, flux, self.gsparams._gsp)

    @property
    def scale_radius(self): return self._sbp.getScaleRadius()
    @property
    def half_light_radius(self): return self.scale_radius * Exponential._hlr_factor

    def __eq__(self, other):
        return (isinstance(other, Exponential) and
                self.scale_radius == other.scale_radius and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Exponential", self.scale_radius, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Exponential(scale_radius=%r, flux=%r, gsparams=%r)'%(
            self.scale_radius, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Exponential(scale_radius=%s'%self.scale_radius
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBExponential.__getinitargs__ = lambda self: (
        self.getScaleRadius(), self.getFlux(), self.getGSParams())


class DeVaucouleurs(GSObject):
    """A class describing DeVaucouleurs profile objects.

    Surface brightness profile with I(r) ~ exp[-(r/scale_radius)^{1/4}].  This is completely
    equivalent to a Sersic with n=4.

    For more information, refer to

        http://en.wikipedia.org/wiki/De_Vaucouleurs'_law


    Initialization
    --------------

    A DeVaucouleurs can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param scale_radius     The value of scale radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param trunc            An optional truncation radius at which the profile is made to drop to
                            zero, in the same units as the size parameter.
                            [default: 0, indicating no truncation]
    @param flux_untruncated Should the provided `flux` and `half_light_radius` refer to the
                            untruncated profile? See the docstring for Sersic for more details.
                            [default: False]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, DeVaucouleurs has the following access properties:

        >>> r0 = devauc_obj.scale_radius
        >>> hlr = devauc_obj.half_light_radius
    """
    _req_params = {}
    _opt_params = { "flux" : float, "trunc" : float, "flux_untruncated" : bool }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False

    def __init__(self, half_light_radius=None, scale_radius=None, flux=1., trunc=0.,
                 flux_untruncated=False, gsparams=None):
        self._gsparams = GSParams.check(gsparams)
        self._sbp = _galsim.SBSersic(4, scale_radius, half_light_radius, flux,
                                     trunc, flux_untruncated, self.gsparams._gsp)

    @property
    def scale_radius(self): return self._sbp.getScaleRadius()
    @property
    def half_light_radius(self): return self._sbp.getHalfLightRadius()
    @property
    def trunc(self): return self._sbp.getTrunc()

    def __eq__(self, other):
        return (isinstance(other, DeVaucouleurs) and
                self.scale_radius == other.scale_radius and
                self.trunc == other.trunc and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.DeVaucouleurs", self.scale_radius, self.trunc, self.flux,
                     self.gsparams))

    def __repr__(self):
        return 'galsim.DeVaucouleurs(scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)'%(
            self.scale_radius, self.trunc, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.DeVaucouleurs(half_light_radius=%s'%self.half_light_radius
        if self.trunc != 0.:
            s += ', trunc=%s'%self.trunc
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s


class Spergel(GSObject):
    """A class describing a Spergel profile.

    The Spergel surface brightness profile is characterized by three properties: its Spergel index
    `nu`, its `flux`, and either the `half_light_radius` or `scale_radius`.  Given these properties,
    the surface brightness profile scales as I(r) ~ (r/scale_radius)^{nu} * K_{nu}(r/scale_radius),
    where K_{nu} is the modified Bessel function of the second kind.

    The Spergel profile is intended as a generic galaxy profile, somewhat like a Sersic profile, but
    with the advantage of being analytic in both real space and Fourier space.  The Spergel index
    `nu` plays a similar role to the Sersic index `n`, in that it adjusts the relative peakiness of
    the profile core and the relative prominence of the profile wings.  At `nu = 0.5`, the Spergel
    profile is equivalent to an Exponential profile (or alternatively an `n = 1.0` Sersic profile).
    At `nu = -0.6` (and in the radial range near the half-light radius), the Spergel profile is
    similar to a DeVaucouleurs profile or `n = 4.0` Sersic profile.

    Note that for `nu <= 0.0`, the Spergel profile surface brightness diverges at the origin.  This
    may lead to rendering problems if the profile is not convolved by either a PSF or a pixel and
    the profile center is precisely on a pixel center.

    Due to its analytic Fourier transform and depending on the indices `n` and `nu`, the Spergel
    profile can be considerably faster to draw than the roughly equivalent Sersic profile.  For
    example, the `nu = -0.6` Spergel profile is roughly 3x faster to draw than an `n = 4.0` Sersic
    profile once the Sersic profile cache has been set up.  However, if not taking advantage of
    the cache, for example, if drawing Sersic profiles with `n` continuously varying near 4.0 and
    Spergel profiles with `nu` continuously varying near -0.6, then the Spergel profiles are about
    50x faster to draw.  At the other end of the galaxy profile spectrum, the `nu = 0.5` Spergel
    profile, `n = 1.0` Sersic profile, and the Exponential profile all take about the same amount
    of time to draw if cached, and the Spergel profile is about 2x faster than the Sersic profile
    if uncached.

    For more information, refer to

        D. N. Spergel, "ANALYTICAL GALAXY PROFILES FOR PHOTOMETRIC AND LENSING ANALYSIS,"
        ASTROPHYS J SUPPL S 191(1), 58-65 (2010) [doi:10.1088/0067-0049/191/1/58].

    Initialization
    --------------

    The allowed range of values for the `nu` parameter is -0.85 <= nu <= 4.  An exception will be
    thrown if you provide a value outside that range.  The lower limit is set above the theoretical
    lower limit of -1 due to numerical difficulties integrating the *very* peaky nu < -0.85
    profiles.  The upper limit is set to avoid numerical difficulties evaluating the modified
    Bessel function of the second kind.

    A Spergel profile can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param nu               The Spergel index, nu.
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, the Spergel profile has the following access
    properties:

        >>> nu = spergel_obj.nu
        >>> r0 = spergel_obj.scale_radius
        >>> hlr = spergel_obj.half_light_radius
    """
    _req_params = { "nu" : float }
    _opt_params = { "flux" : float}
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False

    def __init__(self, nu, half_light_radius=None, scale_radius=None,
                 flux=1., gsparams=None):
        self._gsparams = GSParams.check(gsparams)
        self._sbp = _galsim.SBSpergel(nu, scale_radius, half_light_radius, flux,
                                      self.gsparams._gsp)

    def calculateIntegratedFlux(self, r):
        """Return the integrated flux out to a given radius, r"""
        return self._sbp.calculateIntegratedFlux(float(r))

    def calculateFluxRadius(self, f):
        """Return the radius within which the total flux is f"""
        return self._sbp.calculateFluxRadius(float(f))

    @property
    def nu(self): return self._sbp.getNu()
    @property
    def scale_radius(self): return self._sbp.getScaleRadius()
    @property
    def half_light_radius(self): return self._sbp.getHalfLightRadius()

    def __eq__(self, other):
        return (isinstance(other, Spergel) and
                self.nu == other.nu and
                self.scale_radius == other.scale_radius and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Spergel", self.nu, self.scale_radius, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Spergel(nu=%r, scale_radius=%r, flux=%r, gsparams=%r)'%(
            self.nu, self.scale_radius, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Spergel(nu=%s, half_light_radius=%s'%(self.nu, self.half_light_radius)
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBSpergel.__getinitargs__ = lambda self: (
        self.getNu(), self.getScaleRadius(), None, self.getFlux(), self.getGSParams())


class DeltaFunction(GSObject):
    """A class describing a DeltaFunction surface brightness profile.

    The DeltaFunction surface brightness profile is characterized by a single property,
    its `flux'.

    Initialization
    --------------

    A DeltaFunction can be initialized with a specified flux.

    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods and Properties
    ----------------------

    DeltaFunction simply has the usual GSObject properties.
    """
    # Initialization parameters of the object, with type information, to indicate
    # which attributes are allowed / required in a config file for this object.
    # _req_params are required
    # _opt_params are optional
    # _single_params are a list of sets for which exactly one in the list is required.
    # _takes_rng indicates whether the constructor should be given the current rng.
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False

    def __init__(self, flux=1., gsparams=None):
        self._gsparams = GSParams.check(gsparams)
        self._sbp = _galsim.SBDeltaFunction(flux, self.gsparams._gsp)

    def __eq__(self, other):
        return (isinstance(other, DeltaFunction) and
                self.flux == other.flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.DeltaFunction", self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.DeltaFunction(flux=%r, gsparams=%r)'%(self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.DeltaFunction('
        if self.flux != 1.0:
            s += 'flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBDeltaFunction.__getinitargs__ = lambda self: (
        self.getFlux(), self.getGSParams())
