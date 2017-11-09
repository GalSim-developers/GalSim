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
        self._gsparams = galsim.GSParams.check(gsparams)
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
        return (isinstance(other, galsim.Sersic) and
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
        self._gsparams = galsim.GSParams.check(gsparams)
        self._sbp = _galsim.SBSersic(4, scale_radius, half_light_radius, flux,
                                     trunc, flux_untruncated, self.gsparams._gsp)

    @property
    def scale_radius(self): return self._sbp.getScaleRadius()
    @property
    def half_light_radius(self): return self._sbp.getHalfLightRadius()
    @property
    def trunc(self): return self._sbp.getTrunc()

    def __eq__(self, other):
        return (isinstance(other, galsim.DeVaucouleurs) and
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
