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

    Gaussian
    Moffat
    Airy
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

    Methods
    -------

    In addition to the usual GSObject methods, Gaussian has the following access methods:

        >>> sigma = gauss.getSigma()
        >>> fwhm = gauss.getFWHM()
        >>> hlr = gauss.getHalfLightRadius()
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

        GSObject.__init__(self, _galsim.SBGaussian(sigma, flux, gsparams))
        self._gsparams = gsparams

    def getSigma(self):
        """Return the sigma scale length for this Gaussian profile.
        """
        return self.SBProfile.getSigma()

    def getFWHM(self):
        """Return the FWHM for this Gaussian profile.
        """
        return self.SBProfile.getSigma() * Gaussian._fwhm_factor

    def getHalfLightRadius(self):
        """Return the half light radius for this Gaussian profile.
        """
        return self.SBProfile.getSigma() * Gaussian._hlr_factor

    @property
    def sigma(self): return self.getSigma()
    @property
    def half_light_radius(self): return self.getHalfLightRadius()
    @property
    def fwhm(self): return self.getFWHM()

    def __eq__(self, other):
        return (isinstance(other, galsim.Gaussian) and
                self.sigma == other.sigma and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.Gaussian", self.sigma, self.flux, self._gsparams))

    def __repr__(self):
        return 'galsim.Gaussian(sigma=%r, flux=%r, gsparams=%r)'%(
            self.sigma, self.flux, self._gsparams)

    def __str__(self):
        s = 'galsim.Gaussian(sigma=%s'%self.sigma
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBGaussian.__getinitargs__ = lambda self: (
        self.getSigma(), self.getFlux(), self.getGSParams())
# SBProfile defines __getstate__ and __setstate__.  We don't actually want to use those here.
# Just the __getinitargs__ is sufficient.  We need to define getstate to override the base class
# definition.  (And then setstate will never be called for these, so don't need that one.)
_galsim.SBGaussian.__getstate__ = lambda self: None
_galsim.SBGaussian.__repr__ = lambda self: \
        'galsim._galsim.SBGaussian(%r, %r, %r)'%self.__getinitargs__()


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

    Methods
    -------

    In addition to the usual GSObject methods, Moffat has the following access methods:

        >>> beta = moffat_obj.getBeta()
        >>> rD = moffat_obj.getScaleRadius()
        >>> fwhm = moffat_obj.getFWHM()
        >>> hlr = moffat_obj.getHalfLightRadius()
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
        GSObject.__init__(self, _galsim.SBMoffat(beta, scale_radius, half_light_radius, fwhm,
                                                 trunc, flux, gsparams))
        self._gsparams = gsparams

    def getBeta(self):
        """Return the beta parameter for this Moffat profile.
        """
        return self.SBProfile.getBeta()

    def getScaleRadius(self):
        """Return the scale radius for this Moffat profile.
        """
        return self.SBProfile.getScaleRadius()

    def getFWHM(self):
        """Return the FWHM for this Moffat profile.
        """
        return self.SBProfile.getFWHM()

    def getHalfLightRadius(self):
        """Return the half light radius for this Moffat profile.
        """
        return self.SBProfile.getHalfLightRadius()

    def getTrunc(self):
        """Return the truncation radius for this Moffat profile.
        """
        return self.SBProfile.getTrunc()

    @property
    def beta(self): return self.getBeta()
    @property
    def scale_radius(self): return self.getScaleRadius()
    @property
    def half_light_radius(self): return self.getHalfLightRadius()
    @property
    def fwhm(self): return self.getFWHM()
    @property
    def trunc(self): return self.getTrunc()

    def __eq__(self, other):
        return (isinstance(other, galsim.Moffat) and
                self.beta == other.beta and
                self.scale_radius == other.scale_radius and
                self.trunc == other.trunc and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.Moffat", self.beta, self.scale_radius, self.trunc, self.flux,
                     self._gsparams))

    def __repr__(self):
        return 'galsim.Moffat(beta=%r, scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)'%(
            self.beta, self.scale_radius, self.trunc, self.flux, self._gsparams)

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
_galsim.SBMoffat.__getstate__ = lambda self: None
_galsim.SBMoffat.__repr__ = lambda self: \
        'galsim._galsim.SBMoffat(%r, %r, %r, %r, %r, %r, %r)'%self.__getinitargs__()


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

    Methods
    -------

    In addition to the usual GSObject methods, Airy has the following access methods:

        >>> lam_over_diam = airy_obj.getLamOverD()
        >>> fwhm = airy_obj.getFWHM()
        >>> hlr = airy_obj.getHalfLightRadius()

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
                scale_unit = galsim.angle.get_angle_unit(scale_unit)
            lam_over_diam = (1.e-9*lam/diam)*(galsim.radians/scale_unit)

        GSObject.__init__(self, _galsim.SBAiry(lam_over_diam, obscuration, flux, gsparams))
        self._gsparams = gsparams

    def getHalfLightRadius(self):
        """Return the half light radius of this Airy profile (only supported for
        obscuration = 0.).
        """
        if self.SBProfile.getObscuration() == 0.:
            return self.SBProfile.getLamOverD() * Airy._hlr_factor
        else:
            # In principle can find the half light radius as a function of lam_over_diam and
            # obscuration too, but it will be much more involved...!
            raise NotImplementedError("Half light radius calculation not implemented for Airy "+
                                      "objects with non-zero obscuration.")

    def getFWHM(self):
        """Return the FWHM of this Airy profile (only supported for obscuration = 0.).
        """
        # As above, likewise, FWHM only easy to define for unobscured Airy
        if self.SBProfile.getObscuration() == 0.:
            return self.SBProfile.getLamOverD() * Airy._fwhm_factor
        else:
            # In principle can find the FWHM as a function of lam_over_diam and obscuration too,
            # but it will be much more involved...!
            raise NotImplementedError("FWHM calculation not implemented for Airy "+
                                      "objects with non-zero obscuration.")

    def getLamOverD(self):
        """Return the `lam_over_diam` parameter of this Airy profile.
        """
        return self.SBProfile.getLamOverD()

    def getObscuration(self):
        """Return the `obscuration` parameter of this Airy profile.
        """
        return self.SBProfile.getObscuration()

    @property
    def lam_over_diam(self): return self.getLamOverD()
    @property
    def half_light_radius(self): return self.getHalfLightRadius()
    @property
    def fwhm(self): return self.getFWHM()
    @property
    def obscuration(self): return self.getObscuration()

    def __eq__(self, other):
        return (isinstance(other, galsim.Airy) and
                self.lam_over_diam == other.lam_over_diam and
                self.obscuration == other.obscuration and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.Airy", self.lam_over_diam, self.obscuration, self.flux,
                     self._gsparams))

    def __repr__(self):
        return 'galsim.Airy(lam_over_diam=%r, obscuration=%r, flux=%r, gsparams=%r)'%(
            self.lam_over_diam, self.obscuration, self.flux, self._gsparams)

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
_galsim.SBAiry.__getstate__ = lambda self: None
_galsim.SBAiry.__repr__ = lambda self: \
        'galsim._galsim.SBAiry(%r, %r, %r, %r)'%self.__getinitargs__()


class Kolmogorov(GSObject):
    """A class describing a Kolmogorov surface brightness profile, which represents a long
    exposure atmospheric PSF.

    For more information, refer to

        http://en.wikipedia.org/wiki/Atmospheric_seeing#The_Kolmogorov_model_of_turbulence

    Initialization
    --------------

    The Kolmogorov profile is normally defined in terms of the ratio lambda / r0, where lambda is
    the wavelength of the light (say in the middle of the bandpass you are using) and r0 is the
    Fried parameter.  Typical values for the Fried parameter are on the order of 0.1 m for
    most observatories and up to 0.2 m for excellent sites. The values are usually quoted at
    lambda = 500nm and r0 depends on wavelength as [r0 ~ lambda^(6/5)].

    The natural units for this ratio is radians, which is not normally a convenient unit to use for
    other GSObject dimensions.  Assuming that the other sky coordinates you are using are all in
    arcsec (e.g. the pixel scale when you draw the image, the size of the galaxy, etc.), then you
    should convert this to arcsec as well:

        >>> lam = 700  # nm
        >>> r0 = 0.15 * (lam/500)**1.2  # meters
        >>> lam_over_r0 = (lam * 1.e-9) / r0  # radians
        >>> lam_over_r0 *= 206265  # Convert to arcsec
        >>> psf = galsim.Kolmogorov(lam_over_r0)

    To make this process a bit simpler, we recommend instead providing the wavelength and Fried
    parameter separately using the parameters `lam` (in nm) and either `r0` or `r0_500` (in m).
    GalSim will then convert this to any of the normal kinds of angular units using the
    `scale_unit` parameter:

        >>> psf = galsim.Kolmogorov(lam=lam, r0=r0, scale_unit=galsim.arcsec)

    or

        >>> psf = galsim.Kolmogorov(lam=lam, r0_500=0.15, scale_unit=galsim.arcsec)

    When drawing images, the scale_unit should match the unit used for the pixel scale or the WCS.
    e.g. in this case, a pixel scale of 0.2 arcsec/pixel would be specified as `pixel_scale=0.2`.

    A Kolmogorov object may also be initialized using `fwhm` or `half_light_radius`.  Exactly one
    of these four ways to define the size is required.

    The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 arcsec (e.g., Racine 1996, PASP 699, 108).

    @param lam_over_r0      The parameter that governs the scale size of the profile.
                            See above for details about calculating it. [One of `lam_over_r0`,
                            `fwhm`, `half_light_radius`, or `lam` (along with either `r0` or
                            `r0_500`) is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of `lam_over_r0`, `fwhm`, `half_light_radius`, or `lam` (along
                            with either `r0` or `r0_500`) is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `lam_over_r0`, `fwhm`, `half_light_radius`, or `lam` (along
                            with either `r0` or `r0_500`) is required.]
    @param lam              Lambda (wavelength) in units of nanometers.  Must be supplied with
                            either `r0` or `r0_500`, and in this case, image scales (`scale`)
                            should be specified in units of `scale_unit`.
    @param r0               The Fried parameter in units of meters.  Must be supplied with `lam`,
                            and in this case, image scales (`scale`) should be specified in units
                            of `scale_unit`.
    @param r0_500           The Fried parameter in units of meters at 500 nm.  The Fried parameter
                            at the given wavelength, `lam`, will be computed using the standard
                            ralation r0 = r0_500 * (lam/500)**1.2.
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param scale_unit       Units to use for the sky coordinates when calculating lam/r0 if these
                            are supplied separately.  Note that the results of calling methods like
                            getFWHM() will be returned in units of `scale_unit` as well.  Should be
                            either a galsim.AngleUnit or a string that can be used to construct one
                            (e.g., 'arcsec', 'radians', etc.).  [default: galsim.arcsec]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Kolmogorov has the following access methods:

        >>> lam_over_r0 = kolm.getLamOverR0()
        >>> fwhm = kolm.getFWHM()
        >>> hlr = kolm.getHalfLightRadius()
    """
    _req_params = {}
    _opt_params = { "flux" : float, "r0" : float, "r0_500" : float, "scale_unit" : str }
    # Note that this is not quite right; it's true that exactly one of these 4 should be supplied,
    # but if lam is supplied then r0 is required.  Errors in which parameters are used may be
    # caught either by config or by the python code itself, depending on the particular error.
    _single_params = [ { "lam_over_r0" : float, "fwhm" : float, "half_light_radius" : float,
                         "lam" : float } ]
    _takes_rng = False

    # The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).
    # In SBKolmogorov.cpp we refine this factor to 0.975865
    _fwhm_factor = 0.975865
    # Similarly, SBKolmogorov calculates the relation between lambda/r0 and half-light radius
    _hlr_factor = 0.554811

    def __init__(self, lam_over_r0=None, fwhm=None, half_light_radius=None, lam=None, r0=None,
                 r0_500=None, flux=1., scale_unit=galsim.arcsec, gsparams=None):

        if fwhm is not None :
            if any(item is not None for item in (lam_over_r0, lam, r0, r0_500, half_light_radius)):
                raise TypeError(
                        "Only one of lam_over_r0, fwhm, half_light_radius, or lam (with r0 or "+
                        "r0_500) may be specified for Kolmogorov")
            else:
                lam_over_r0 = fwhm / Kolmogorov._fwhm_factor
        elif half_light_radius is not None:
            if any(item is not None for item in (lam_over_r0, lam, r0, r0_500)):
                raise TypeError(
                        "Only one of lam_over_r0, fwhm, half_light_radius, or lam (with r0 or "+
                        "r0_500) may be specified for Kolmogorov")
            else:
                lam_over_r0 = half_light_radius / Kolmogorov._hlr_factor
        elif lam_over_r0 is not None:
            if any(item is not None for item in (lam, r0, r0_500)):
                raise TypeError("Cannot specify lam, r0 or r0_500 in conjunction with lam_over_r0.")
        else:
            if lam is None or (r0 is None and r0_500 is None):
                raise TypeError(
                        "One of lam_over_r0, fwhm, half_light_radius, or lam (with r0 or "+
                        "r0_500) must be specified for Kolmogorov")
            # In this case we're going to use scale_unit, so parse it in case of string input:
            if isinstance(scale_unit, str):
                scale_unit = galsim.angle.get_angle_unit(scale_unit)
            if r0 is None:
                r0 = r0_500 * (lam/500.)**1.2
            lam_over_r0 = (1.e-9*lam/r0)*(galsim.radians/scale_unit)

        GSObject.__init__(self, _galsim.SBKolmogorov(lam_over_r0, flux, gsparams))
        self._gsparams = gsparams

    def getLamOverR0(self):
        """Return the `lam_over_r0` parameter of this Kolmogorov profile.
        """
        return self.SBProfile.getLamOverR0()

    def getFWHM(self):
        """Return the FWHM of this Kolmogorov profile.
        """
        return self.SBProfile.getLamOverR0() * Kolmogorov._fwhm_factor

    def getHalfLightRadius(self):
        """Return the half light radius of this Kolmogorov profile.
        """
        return self.SBProfile.getLamOverR0() * Kolmogorov._hlr_factor

    @property
    def lam_over_r0(self): return self.getLamOverR0()
    @property
    def half_light_radius(self): return self.getHalfLightRadius()
    @property
    def fwhm(self): return self.getFWHM()

    def __eq__(self, other):
        return (isinstance(other, galsim.Kolmogorov) and
                self.lam_over_r0 == other.lam_over_r0 and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.Kolmogorov", self.lam_over_r0, self.flux, self._gsparams))

    def __repr__(self):
        return 'galsim.Kolmogorov(lam_over_r0=%r, flux=%r, gsparams=%r)'%(
            self.lam_over_r0, self.flux, self._gsparams)

    def __str__(self):
        s = 'galsim.Kolmogorov(lam_over_r0=%s'%self.lam_over_r0
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBKolmogorov.__getinitargs__ = lambda self: (
        self.getLamOverR0(), self.getFlux(), self.getGSParams())
_galsim.SBKolmogorov.__getstate__ = lambda self: None
_galsim.SBKolmogorov.__repr__ = lambda self: \
        'galsim._galsim.SBKolmogorov(%r, %r, %r)'%self.__getinitargs__()


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

    Methods
    -------

    In addition to the usual GSObject methods, Pixel has the following access method:

        >>> scale = pixel.getScale()

    """
    _req_params = { "scale" : float }
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False

    def __init__(self, scale, flux=1., gsparams=None):
        GSObject.__init__(self, _galsim.SBBox(scale, scale, flux, gsparams))
        self._gsparams = gsparams

    def getScale(self):
        """Return the pixel scale.
        """
        return self.SBProfile.getWidth()

    @property
    def scale(self): return self.getScale()

    def __eq__(self, other):
        return (isinstance(other, galsim.Pixel) and
                self.scale == other.scale and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.Pixel", self.scale, self.flux, self._gsparams))

    def __repr__(self):
        return 'galsim.Pixel(scale=%r, flux=%r, gsparams=%r)'%(
            self.scale, self.flux, self._gsparams)

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

    Methods
    -------

    In addition to the usual GSObject methods, Box has the following access methods:

        >>> width = box.getWidth()
        >>> height = box.getHeight()

    """
    _req_params = { "width" : float, "height" : float }
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False

    def __init__(self, width, height, flux=1., gsparams=None):
        width = float(width)
        height = float(height)
        GSObject.__init__(self, _galsim.SBBox(width, height, flux, gsparams))
        self._gsparams = gsparams

    def getWidth(self):
        """Return the width of the box in the x dimension.
        """
        return self.SBProfile.getWidth()

    def getHeight(self):
        """Return the height of the box in the y dimension.
        """
        return self.SBProfile.getHeight()

    @property
    def width(self): return self.getWidth()
    @property
    def height(self): return self.getHeight()

    def __eq__(self, other):
        return (isinstance(other, galsim.Box) and
                self.width == other.width and
                self.height == other.height and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.Box", self.width, self.height, self.flux, self._gsparams))

    def __repr__(self):
        return 'galsim.Box(width=%r, height=%r, flux=%r, gsparams=%r)'%(
            self.width, self.height, self.flux, self._gsparams)

    def __str__(self):
        s = 'galsim.Box(width=%s, height=%s'%(self.width, self.height)
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBBox.__getinitargs__ = lambda self: (
        self.getWidth(), self.getHeight(), self.getFlux(), self.getGSParams())
_galsim.SBBox.__getstate__ = lambda self: None
_galsim.SBBox.__repr__ = lambda self: \
        'galsim._galsim.SBBox(%r, %r, %r, %r)'%self.__getinitargs__()


class TopHat(GSObject):
    """A class describing a radial tophat profile.  This profile is a constant value within some
    radius, and zero outside this radius.

    Initialization
    --------------

    @param radius           The radius of the TopHat, where the surface brightness drops to 0.
    @param flux             The flux (in photons/cm^2/s) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, TopHat has the following access method:

        >>> radius = tophat.getRadius()

    """
    _req_params = { "radius" : float }
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False

    def __init__(self, radius, flux=1., gsparams=None):
        radius = float(radius)
        GSObject.__init__(self, _galsim.SBTopHat(radius, flux=flux, gsparams=gsparams))
        self._gsparams = gsparams

    def getRadius(self):
        """Return the radius of the tophat profile.
        """
        return self.SBProfile.getRadius()

    @property
    def radius(self): return self.getRadius()

    def __eq__(self, other):
        return (isinstance(other, galsim.TopHat) and
                self.radius == other.radius and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.TopHat", self.radius, self.flux, self._gsparams))

    def __repr__(self):
        return 'galsim.TopHat(radius=%r, flux=%r, gsparams=%r)'%(
            self.radius, self.flux, self._gsparams)

    def __str__(self):
        s = 'galsim.TopHat(radius=%s'%self.radius
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBTopHat.__getinitargs__ = lambda self: (
        self.getRadius(), self.getFlux(), self.getGSParams())
_galsim.SBTopHat.__getstate__ = lambda self: None
_galsim.SBTopHat.__repr__ = lambda self: \
        'galsim._galsim.SBTopHat(%r, %r, %r)'%self.__getinitargs__()


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
    radius.  The getHalfLightRadius() method will return the true half-light radius.  Similarly,
    the actual flux will not be the same as the specified value; the true flux is also returned
    by the getFlux() method.

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

        >>> sersic_obj1.getHalfLightRadius()
        2.5
        >>> sersic_obj2.getHalfLightRadius()
        2.5
        >>> sersic_obj3.getHalfLightRadius()
        1.9795101383056892    # The true half-light radius is smaller than the specified value

        >>> sersic_obj1.getFlux()
        40.0
        >>> sersic_obj2.getFlux()
        40.0
        >>> sersic_obj3.getFlux()
        34.56595186009519     # Flux is missing due to truncation

        >>> sersic_obj1.getScaleRadius()
        0.003262738739834598
        >>> sersic_obj2.getScaleRadius()
        0.004754602453641744  # the scale radius needed adjustment to accommodate HLR
        >>> sersic_obj3.getScaleRadius()
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

        >>> sersic_obj1.getHalfLightRadius()
        38.311372735390016
        >>> sersic_obj2.getHalfLightRadius()
        5.160062547614234
        >>> sersic_obj3.getHalfLightRadius()
        5.160062547614234     # For the truncated cases, the half-light radii are the same

        >>> sersic_obj1.getFlux()
        40.0
        >>> sersic_obj2.getFlux()
        40.0
        >>> sersic_obj3.getFlux()
        6.985044085834393     # Flux is missing due to truncation

        >>> sersic_obj1.getScaleRadius()
        0.05
        >>> sersic_obj2.getScaleRadius()
        0.05
        >>> sersic_obj3.getScaleRadius()
        0.05

    Methods
    -------

    In addition to the usual GSObject methods, Sersic has the following access methods:

        >>> n = sersic_obj.getN()
        >>> r0 = sersic_obj.getScaleRadius()
        >>> hlr = sersic_obj.getHalfLightRadius()
    """
    _req_params = { "n" : float }
    _opt_params = { "flux" : float, "trunc" : float, "flux_untruncated" : bool }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False

    # The conversion from hlr to scale radius is complicated for Sersic, especially since we
    # allow it to be truncated.  So we do these calculations in the C++-layer constructor.
    def __init__(self, n, half_light_radius=None, scale_radius=None,
                 flux=1., trunc=0., flux_untruncated=False, gsparams=None):
        GSObject.__init__(self, _galsim.SBSersic(n, scale_radius, half_light_radius,flux, trunc,
                                                 flux_untruncated, gsparams))
        self._gsparams = gsparams

    def getN(self):
        """Return the Sersic index `n` for this profile.
        """
        return self.SBProfile.getN()

    def getHalfLightRadius(self):
        """Return the half light radius for this Sersic profile.
        """
        return self.SBProfile.getHalfLightRadius()

    def getScaleRadius(self):
        """Return the scale radius for this Sersic profile.
        """
        return self.SBProfile.getScaleRadius()

    def getTrunc(self):
        """Return the truncation radius for this Sersic profile.
        """
        return self.SBProfile.getTrunc()

    @property
    def n(self): return self.getN()
    @property
    def scale_radius(self): return self.getScaleRadius()
    @property
    def half_light_radius(self): return self.getHalfLightRadius()
    @property
    def trunc(self): return self.getTrunc()

    def __eq__(self, other):
        return (isinstance(other, galsim.Sersic) and
                self.n == other.n and
                self.scale_radius == other.scale_radius and
                self.trunc == other.trunc and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.SBSersic", self.n, self.scale_radius, self.trunc, self.flux,
                     self._gsparams))

    def __repr__(self):
        return 'galsim.Sersic(n=%r, scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)'%(
            self.n, self.scale_radius, self.trunc, self.flux, self._gsparams)

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
_galsim.SBSersic.__getstate__ = lambda self: None
_galsim.SBSersic.__repr__ = lambda self: \
        'galsim._galsim.SBSersic(%r, %r, %r, %r, %r, %r, %r)'%self.__getinitargs__()


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

    Methods
    -------

    In addition to the usual GSObject methods, Exponential has the following access methods:

        >>> r0 = exp_obj.getScaleRadius()
        >>> hlr = exp_obj.getHalfLightRadius()
    """
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False

    # The half-light-radius is not analytic, but can be calculated numerically.
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
        GSObject.__init__(self, _galsim.SBExponential(scale_radius, flux, gsparams))
        self._gsparams = gsparams

    def getScaleRadius(self):
        """Return the scale radius for this Exponential profile.
        """
        return self.SBProfile.getScaleRadius()

    def getHalfLightRadius(self):
        """Return the half light radius for this Exponential profile.
        """
        # Factor not analytic, but can be calculated by iterative solution of equation:
        #  (re / r0) = ln[(re / r0) + 1] + ln(2)
        return self.SBProfile.getScaleRadius() * Exponential._hlr_factor

    @property
    def scale_radius(self): return self.getScaleRadius()
    @property
    def half_light_radius(self): return self.getHalfLightRadius()

    def __eq__(self, other):
        return (isinstance(other, galsim.Exponential) and
                self.scale_radius == other.scale_radius and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.Exponential", self.scale_radius, self.flux, self._gsparams))

    def __repr__(self):
        return 'galsim.Exponential(scale_radius=%r, flux=%r, gsparams=%r)'%(
            self.scale_radius, self.flux, self._gsparams)

    def __str__(self):
        s = 'galsim.Exponential(scale_radius=%s'%self.scale_radius
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBExponential.__getinitargs__ = lambda self: (
        self.getScaleRadius(), self.getFlux(), self.getGSParams())
_galsim.SBExponential.__getstate__ = lambda self: None
_galsim.SBExponential.__repr__ = lambda self: \
        'galsim._galsim.SBExponential(%r, %r, %r)'%self.__getinitargs__()


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

    Methods
    -------

    In addition to the usual GSObject methods, DeVaucouleurs has the following access methods:

        >>> r0 = devauc_obj.getScaleRadius()
        >>> hlr = devauc_obj.getHalfLightRadius()
    """
    _req_params = {}
    _opt_params = { "flux" : float, "trunc" : float, "flux_untruncated" : bool }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False

    def __init__(self, half_light_radius=None, scale_radius=None, flux=1., trunc=0.,
                 flux_untruncated=False, gsparams=None):
        GSObject.__init__(self, _galsim.SBSersic(4, scale_radius, half_light_radius, flux,
                                                 trunc, flux_untruncated, gsparams))
        self._gsparams = gsparams

    def getHalfLightRadius(self):
        """Return the half light radius for this DeVaucouleurs profile.
        """
        return self.SBProfile.getHalfLightRadius()

    def getScaleRadius(self):
        """Return the scale radius for this DeVaucouleurs profile.
        """
        return self.SBProfile.getScaleRadius()

    def getTrunc(self):
        """Return the truncation radius for this DeVaucouleurs profile.
        """
        return self.SBProfile.getTrunc()

    @property
    def scale_radius(self): return self.getScaleRadius()
    @property
    def half_light_radius(self): return self.getHalfLightRadius()
    @property
    def trunc(self): return self.getTrunc()

    def __eq__(self, other):
        return (isinstance(other, galsim.DeVaucouleurs) and
                self.scale_radius == other.scale_radius and
                self.trunc == other.trunc and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.DeVaucouleurs", self.scale_radius, self.trunc, self.flux,
                     self._gsparams))

    def __repr__(self):
        return 'galsim.DeVaucouleurs(scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)'%(
            self.scale_radius, self.trunc, self.flux, self._gsparams)

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

    Methods
    -------

    In addition to the usual GSObject methods, the Spergel profile has the following access methods:

        >>> nu = spergel_obj.getNu()
        >>> r0 = spergel_obj.getScaleRadius()
        >>> hlr = spergel_obj.getHalfLightRadius()
    """
    _req_params = { "nu" : float }
    _opt_params = { "flux" : float}
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False

    def __init__(self, nu, half_light_radius=None, scale_radius=None,
                 flux=1., gsparams=None):
        GSObject.__init__(self, _galsim.SBSpergel(nu, scale_radius, half_light_radius, flux,
                                                  gsparams))
        self._gsparams = gsparams

    def getNu(self):
        """Return the Spergel index `nu` for this profile.
        """
        return self.SBProfile.getNu()

    def getHalfLightRadius(self):
        """Return the half light radius for this Spergel profile.
        """
        return self.SBProfile.getHalfLightRadius()

    def getScaleRadius(self):
        """Return the scale radius for this Spergel profile.
        """
        return self.SBProfile.getScaleRadius()

    @property
    def nu(self): return self.getNu()
    @property
    def scale_radius(self): return self.getScaleRadius()
    @property
    def half_light_radius(self): return self.getHalfLightRadius()

    def __eq__(self, other):
        return (isinstance(other, galsim.Spergel) and
                self.nu == other.nu and
                self.scale_radius == other.scale_radius and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.Spergel", self.nu, self.scale_radius, self.flux, self._gsparams))

    def __repr__(self):
        return 'galsim.Spergel(nu=%r, scale_radius=%r, flux=%r, gsparams=%r)'%(
            self.nu, self.scale_radius, self.flux, self._gsparams)

    def __str__(self):
        s = 'galsim.Spergel(nu=%s, half_light_radius=%s'%(self.nu, self.half_light_radius)
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBSpergel.__getinitargs__ = lambda self: (
        self.getNu(), self.getScaleRadius(), None, self.getFlux(), self.getGSParams())
_galsim.SBSpergel.__getstate__ = lambda self: None
_galsim.SBSpergel.__repr__ = lambda self: \
        'galsim._galsim.SBSpergel(%r, %r, %r, %r, %r)'%self.__getinitargs__()

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

    Methods
    -------

    DeltaFunction simply has the usual GSObject methods.
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
        GSObject.__init__(self, _galsim.SBDeltaFunction(flux, gsparams))
        self._gsparams = gsparams

    def __eq__(self, other):
        return (isinstance(other, galsim.DeltaFunction) and
                self.flux == other.flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.DeltaFunction", self.flux, self._gsparams))

    def __repr__(self):
        return 'galsim.DeltaFunction(flux=%r, gsparams=%r)'%(
            self.flux, self._gsparams)

    def __str__(self):
        s = 'galsim.DeltaFunction('
        if self.flux != 1.0:
            s += 'flux=%s'%self.flux
        s += ')'
        return s

_galsim.SBDeltaFunction.__getinitargs__ = lambda self: (
        self.getFlux(), self.getGSParams())
_galsim.SBDeltaFunction.__getstate__ = lambda self: None
_galsim.SBDeltaFunction.__repr__ = lambda self: \
        'galsim._galsim.SBDeltaFunction(%r, %r)'%self.__getinitargs__()
