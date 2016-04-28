# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
"""@file phase_psf.py
Utilities for creating PSFs from phase screens.  Essentially evaluates the Fourier optics
diffraction equation:

PSF(x, y) = int( |FT(aperture(u, v) * exp(i * phase(u, v, x, y, t)))|^2, dt)

where x, y are focal plane coordinates and u, v are pupil plane coordinates

The main classes of note are:

AtmosphericScreen
  Class implementing phase(u, v, x, y, t) for von Karman type turbulence, with possibly evolving
  "non-frozen-flow" phases.

PhaseScreenList
  Python sequence type to hold multiple phase screens, for instance to simulate turbulence at
  different altitudes.  A key method is makePSF(), which will take the list of phase screens,
  add them together linearly (Fraunhofer approximation), and evaluate the above diffraction
  equation to yield a PhaseScreenPSF object.

PhaseScreenPSF
  A GSObject holding the evaluated PSF.

Atmosphere
  Convenience function to quickly assemble multiple AtmosphericScreens into a PhaseScreenList.
"""

from itertools import izip, chain

import numpy as np
import galsim
import utilities
from galsim import GSObject


class Aperture(object):
    """ Class representing a telescope aperture embedded in a larger pupil plane array.

    There are several options for setting the size and resolution of the pupil plane array.

    The first option, which is implemented by the default constructor, is to set the size either
    directly with `pupil_plane_size` or automatically based on the aperture diameter `diam`, and set
    the resolution either directly with `pupil_scale` or implicitly via `npix`.  Note that setting
    the pupil plane array size via `diam` also depends on the value of `oversampling`.

    The second way to set the pupil plane array size and resolution is via the alternative
    constructor `Aperture.fromPhaseScreenList`.  This constructor will examine the supplied
    PhaseScreenList argument and wavelength argument to pick good values for size and resolution.
    These values will then be modified via the `oversampling` and `pad_factor` keywords, and can be
    overridden entirely with `pupil_plane_size` and `pupil_scale`.

    The last way to set the pupil plane array size and resolution is via the alternative constructor
    `Aperture.fromGSObject`.  This constructor will examine the supplied GSObject and wavelength to
    pick good values for size and resolution.  These values will then be modified via the
    `oversampling` and `pad_factor` keywords, and can be overridden entirely with `pupil_plane_size`
    and `pupil_scale`.

    Note that a good Aperture can also be constructed by the PhaseScreenPSF constructor directly
    (which internally uses `Aperture.fromPhaseScreenList`), or equivalently by
    PhaseScreenList.makePSF.

    All three Aperture construction mechanisms also accept keywords to control the pupil geometry,
    such as the size of a central obstruction or a description of supporting struts.

    The constructed object has two key attributes:
        `illuminated`  a boolean array indicating which positions in the pupil plane are exposed to
                       the sky.
        `rho`          complex array of unit-disc-scaled pupil coordinates for defining Zernike
                       polynomials.

    Each element of `rho` encodes the corresponding coordinate as (x, y) => x + 1j * y.

    @param diam              Aperture diameter in meters.
    @param circular_pupil    Adopt a circular pupil? [default: True].
    @param obscuration       Fractional linear circular obscuration of pupil. [default: 0.]
    @param nstruts           Number of radial support struts to add to the central obscuration.
                             [default: 0]
    @param strut_thick       Thickness of support struts as a fraction of pupil diameter.
                             [default: 0.05]
    @param strut_angle       Angle made between the vertical and the first strut in the CCW
                             direction; must be an Angle instance.  [default: 0. * galsim.degrees]
    @param oversampling      Optional oversampling factor for the PSF produced with this aperture.
                             Setting `oversampling < 1` will produce aliasing in the PSF (not good).
                             Usually `oversampling` should be somewhat larger than 1.  1.5 is
                             usually a safe choice.  Note that if `pupil_plane_size` is specified
                             directly, then this keyword is ignored.  [default: 1.5]
    """
    def __init__(self, diam, lam=None, circular_pupil=True, obscuration=0.0,
                 nstruts=0, strut_thick=0.05, strut_angle=0.0*galsim.degrees,
                 oversampling=1.5, pad_factor=1.5,
                 screen_list=None,
                 pupil_plane_im=None, pupil_angle=0.0*galsim.degrees,
                 _pupil_plane_size=None, _pupil_plane_scale=None):

        self.diam = diam  # Always need to explicitly specify an aperture diameter.

        # You can either set geometric properties, or use a pupil image, but not both, so check for
        # that here.  One caveat is that we allow sanity checking the sampling of a pupil_image by
        # comparing it to the sampling we would have used for an (obscured) Airy profile.  So it's
        # okay to specify an obscuration with a pupil_plane_im, for example, but not struts.
        is_default_geom = (circular_pupil == True and
                           nstruts == 0 and
                           strut_thick == 0.05 and
                           strut_angle == 0.0*galsim.degrees)
        if not is_default_geom and pupil_plane_im is not None:
            raise ValueError("Can't specify both geometric parameters and pupil_plane_im.")

        if pupil_plane_im is not None:  # Use image of pupil plane
            # Note that there's a small ambiguity when specifying only pupil_plane_im and a diam.
            # We need to infer the pupil plane size & scale, but we can only figure this out to the
            # relative precision of 1 pixel in the pupil_plane_im.  To optionally get more precision
            # (for unit tests), we use the _pupil_plane_scale keyword to increase this precision.
            self._load_pupil_plane(pupil_plane_im, pupil_angle, obscuration=obscuration,
                                   _pupil_plane_scale=_pupil_plane_scale)
        else:  # Use geometric parameters.
            if obscuration >= 1.:
                raise ValueError("Pupil fully obscured! obscuration = {:0} (>= 1)"
                                 .format(obscuration))
            # When setting the pupil plane size and scale, defer to the private kwargs if available,
            # otherwise, try to guess good values based on
            # 1) Any screen list that was provided, or
            # 2) An Airy function.

            if _pupil_plane_size is None:
                # Setting the pupil plane size is easy.  Just go for the Nyquist limit, adjusted by
                # oversampling.
                _pupil_plane_size = 2.0*diam*oversampling
            self.pupil_plane_size = _pupil_plane_size

            if _pupil_plane_scale is None:
                if lam is None:
                    raise ValueError("Must provide lam if not providing pupil_plane_im.")
                if screen_list is not None:
                    stepk = screen_list.stepK(lam=lam, diam=diam, obscuration=obscuration)
                else:
                    airy = galsim.Airy(diam=diam, lam=lam, obscuration=obscuration)
                    stepk = airy.stepK()
                scale = (stepk * lam*1.e-9 * (galsim.radians / galsim.arcsec) /
                         (2 * np.pi * pad_factor))
                self.npix = galsim._galsim.goodFFTSize(int(np.ceil(self.pupil_plane_size/scale)))
                _pupil_plane_scale = _pupil_plane_size/self.npix
            else:
                self.npix = int(np.ceil(self.pupil_plane_size/_pupil_plane_scale))
            # Make sure pupil_plane_size is an integer multiple of pupil_plane_scale.
            self.pupil_plane_scale = self.pupil_plane_size/self.npix

            # With the array parameters set, we're ready to actually parametrically draw the pupil
            # plane.
            self._generate_pupil_plane(circular_pupil, obscuration,
                                       nstruts, strut_thick, strut_angle)

    def _generate_pupil_plane(self, circular_pupil=True, obscuration=0.,
                              nstruts=0, strut_thick=0.05, strut_angle=0.*galsim.degrees):
        # Save params for str/repr
        self._circular_pupil = circular_pupil
        self._obscuration = obscuration
        self._nstruts = nstruts
        self._strut_thick = strut_thick
        self._strut_angle = strut_angle

        u = np.fft.fftshift(np.fft.fftfreq(self.npix, 1./self.pupil_plane_size))
        u, v = np.meshgrid(u, u)
        rsqr = u**2 + v**2

        radius = 0.5*self.diam
        if circular_pupil:
            self.illuminated = (rsqr < radius**2)
            if obscuration > 0.:
                self.illuminated *= rsqr >= (radius*obscuration)**2
        else:
            self.illuminated = (np.abs(u) < radius) & (np.abs(v) < radius)
            if obscuration > 0.:
                self.illuminated *= ((np.abs(u) >= radius*obscuration) *
                                     (np.abs(v) >= radius*obscuration))

        if nstruts > 0:
            if not isinstance(strut_angle, galsim.Angle):
                raise TypeError("Input kwarg strut_angle must be a galsim.Angle instance.")
            # Add the initial rotation if requested, converting to radians.
            if strut_angle.rad != 0.:
                u, v = utilities.rotate_xy(u, v, -strut_angle)
            rotang = 360. * galsim.degrees / nstruts
            # Then loop through struts setting to zero the regions which lie under the strut
            for istrut in xrange(nstruts):
                u, v = utilities.rotate_xy(u, v, -rotang)
                self.illuminated *= ((np.abs(u) >= radius * strut_thick) + (v < 0.0))

    def _load_pupil_plane(self, pupil_plane_im, pupil_angle, obscuration=0.0,
                          _pupil_plane_scale=None):
        # Handle multiple types of input: NumPy array, galsim.Image, or string for filename with
        # image.
        if isinstance(pupil_plane_im, np.ndarray):
            # Make it into an image.
            pupil_plane_im = galsim.Image(pupil_plane_im)
        elif isinstance(pupil_plane_im, galsim.Image):
            # Make sure not to overwrite input image.
            pupil_plane_im = pupil_plane_im.copy()
        else:
            # Read in image of pupil plane from file.
            pupil_plane_im = galsim.fits.read(pupil_plane_im)

        # Sanity checks
        if pupil_plane_im.array.shape[0] != pupil_plane_im.array.shape[1]:
            raise ValueError("We require square input pupil plane arrays!")
        if pupil_plane_im.array.shape[0] % 2 == 1:
            raise ValueError("Even-sized input arrays are required for the pupil plane!")

        self.npix = pupil_plane_im.array.shape[0]
        u = np.fft.fftshift(np.fft.fftfreq(self.npix))
        u, v = np.meshgrid(u, u)
        r = np.hypot(u, v)
        rmax_illum = np.max(r*(pupil_plane_im.array > 0))
        # Figure out the scale given the diam and illuminated pixels.
        self.pupil_plane_size = self.diam / (2.0 * rmax_illum)
        self.pupil_plane_scale = self.pupil_plane_size / self.npix
        # We only know rmax_illum to the precision of 1 pixel or so; i.e. to the precision of
        # 1./self.npix.  So if the _pupil_plane_scale is set, assume that's more accurate, but make
        # sure it's consistent with what we just calculated.
        if _pupil_plane_scale is not None:
            assert abs(self.pupil_plane_scale/_pupil_plane_scale-1.0) < 2./(self.npix*rmax_illum)
            self.pupil_plane_scale = _pupil_plane_scale
            self.pupil_plane_size = self.npix * self.pupil_plane_scale

        # At this point, we can compare the sampling derived from the diameter and the image to the
        # sampling that would have been used for a reference Airy profile.
        # Conveniently, the physical scale for sampling the aperture for an Airy does not depend on
        # the wavelength!  So just use 500nm.
        airy = galsim.Airy(lam=500.0, diam=self.diam, obscuration=obscuration)
        stepk = airy.stepK()
        scale = stepk * 500.0*1.e-9 * (galsim.radians / galsim.arcsec) / (2 * np.pi)
        if scale < self.pupil_plane_scale:
            import warnings
            ratio = self.pupil_plane_scale / scale
            warnings.warn("Input pupil plane image may not be sampled well enough!\n"
                          "Consider increasing sampling by a factor %f, and/or check "
                          "OpticalPSF outputs for signs of folding in real space."%ratio)

        if pupil_angle.rad() == 0.:
            self.illuminated = pupil_plane_im.array.astype(bool)
        else:
            # Rotate the pupil plane image as required based on the `pupil_angle`, being careful to
            # ensure that the image is one of the allowed types.  We ignore the scale.
            int_im = galsim.InterpolatedImage(galsim.Image(pupil_plane_im.array, scale=1.,
                                                           dtype=np.float64),
                                              x_interpolant='linear', calculate_stepk=False,
                                              calculate_maxk=False)
            int_im = int_im.rotate(pupil_angle)
            new_im = galsim.ImageF(pupil_plane_im.array.shape[1], pupil_plane_im.array.shape[0])
            new_im = int_im.drawImage(image=new_im, scale=1., method='no_pixel')
            pp_arr = new_im.array
            # Restore hard edges that might have been lost during the interpolation.  To do this, we
            # check the maximum value of the entries.  Values after interpolation that are >half
            # that maximum value are kept as nonzero (True), but those that are <half the maximum
            # value are set to zero (False).
            max_pp_val = np.max(pp_arr)
            pp_arr[pp_arr < 0.5*max_pp_val] = 0.
            self.illuminated = pp_arr.astype(bool)

    # Used in Aperture.__str__ and OpticalPSF.__str__
    def _geometry_str(self):
        s = ""
        if not self._circular_pupil:
            s += ", circular_pupil=False"
        if self._obscuration != 0.0:
            s += ", obscuration=%s"%self._obscuration
        if self._nstruts != 0:
            s += ", nstruts=%s"%self._nstruts
            if self._strut_thick != 0.05:
                s += ", strut_thick=%s"%self._strut_thick
            if self._strut_angle != 0*galsim.degrees:
                s += ", strut_angle=%s"%self._strut_angle
        return s

    def __str__(self):
        s = "galsim.Aperture(diam=%r"%self.diam
        if hasattr(self, '_circular_pupil'):  # Pupil was created geometrically, so use that here.
            s += self._geometry_str()
        s += ", _pupil_plane_scale=%s"%self.pupil_plane_scale
        s += ", _pupil_plane_size=%s"%self.pupil_plane_size
        s += ")"
        return s

    # Used in Aperture.__repr__ and OpticalPSF.__repr__
    def _geometry_repr(self):
        s = ""
        if not self._circular_pupil:
            s += ", circular_pupil=False"
        if self._obscuration != 0.0:
            s += ", obscuration=%r"%self._obscuration
        if self._nstruts != 0:
            s += ", nstruts=%r"%self._nstruts
            if self._strut_thick != 0.05:
                s += ", strut_thick=%r"%self._strut_thick
            if self._strut_angle != 0*galsim.degrees:
                s += ", strut_angle=%r"%self._strut_angle
        s += ", _pupil_plane_scale=%r"%self.pupil_plane_scale
        s += ", _pupil_plane_size=%r"%self.pupil_plane_size
        return s

    def __repr__(self):
        s = "galsim.Aperture(diam=%r"%self.diam
        if hasattr(self, '_circular_pupil'):  # Pupil was created geometrically, so use that here.
            s += self._geometry_repr()
        else:  # Pupil was created from image, so use that instead.
            s += ", pupil_plane_im=array(%r"%self.illuminated.tolist()+", dtype='float')"
            s += ", _pupil_plane_scale=%r"%self.pupil_plane_scale
        s += ")"
        return s

    def __eq__(self, other):
        return (isinstance(other, galsim.Aperture) and
                self.diam == other.diam and
                self.pupil_plane_scale == other.pupil_plane_scale and
                np.array_equal(self.illuminated, other.illuminated))

    def __hash__(self):
        # Cache since self.illuminated may be large.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.Aperture", self.diam, self.pupil_plane_scale))
            self._hash ^= hash(tuple(self.illuminated.ravel()))
        return self._hash

    @property
    def rho(self):
        """ Pupil plane coordinate array.

        Each element encodes the coordinate, (normalized to a unit disk) as a complex number:
        (x, y) => x + 1j * y.

        Computed on demand and cached for reuse.
        """
        if not hasattr(self, '_rho') or self._rho is None:
            u = np.fft.fftshift(np.fft.fftfreq(self.npix, self.diam/self.pupil_plane_size/2.0))
            u, v = np.meshgrid(u, u)
            self._rho = u + 1j * v
        return self._rho

    # Some quick notes for Josh:
    # - Relation between real-space grid with size L and pitch dL (dimensions of angle) and
    #   corresponding Fourier grid with size 2*maxK and pitch stepK (dimensions of inverse angle):
    #     stepK = 2*pi/L
    #     maxK = pi/dL
    # - Relation between aperture of size N*dx and pitch dx (dimensions of length, not angle!) and
    #   Fourier grid:
    #     dx = stepK * lambda / (2 * pi)
    #     N*dx = maxK * lambda / pi
    # - Implies relation between aperture grid and real-space grid:
    #     dx = lambda/L
    #     N*dx = lambda/dL
    def _stepK(self, wave, scale_unit=galsim.arcsec):
        """Return the Fourier grid spacing for this aperture at given wavelength.

        @param wave        Wavelength in nanometers.
        @param scale_unit  Inverse units in which to return result [default: galsim.arcsec]
        @returns           Fourier grid spacing.
        """
        return 2*np.pi*self.pupil_plane_scale/(wave*1e-9) * scale_unit/galsim.radians

    def _maxK(self, lam, scale_unit=galsim.arcsec):
        """Return the Fourier grid half-size for this aperture at given wavelength.

        @param lam         Wavelength in nanometers.
        @param scale_unit  Inverse units in which to return result [default: galsim.arcsec]
        @returns           Fourier grid half-size.
        """
        return np.pi*self.pupil_plane_size/(wave*1e-9) * scale_unit/galsim.radians

    def _sky_scale(self, lam, scale_unit=galsim.arcsec):
        """Return the image scale for this aperture at given wavelength.
        @param lam         Wavelength in nanometers.
        @param scale_unit  Units in which to return result [default: galsim.arcsec]
        @returns           Image scale.
        """
        return (lam*1e-9) / self.pupil_plane_size * galsim.radians/scale_unit

    def _sky_size(self, lam, scale_unit=galsim.arcsec):
        return (lam*1e-9) / self.pupil_plane_scale * galsim.radians/scale_unit


class AtmosphericScreen(object):
    """ An atmospheric phase screen that can drift in the wind and evolves ("boils") over time.  The
    initial phases and fractional phase updates are drawn from a von Karman power spectrum, which is
    defined by a Fried parameter that effectively sets the amplitude of the turbulence, and an outer
    scale that sets the scale beyond which the turbulence power goes (smoothly) to zero.

    @param screen_size   Physical extent of square phase screen in meters.  This should be large
                         enough to accommodate the desired field-of-view of the telescope as well as
                         the meta-pupil defined by the wind speed and exposure time.  Note that
                         the screen will have periodic boundary conditions, so the code will run
                         with a smaller sized screen, though this may introduce artifacts into PSFs
                         or PSF correlations functions. Note that screen_size may be tweaked by the
                         initializer to ensure screen_size is a multiple of screen_scale.
    @param screen_scale  Physical pixel scale of phase screen in meters.  A fraction of the Fried
                         parameter is usually sufficiently small, but users should test the effects
                         of this parameter to ensure robust results. [Default: half of r0_500]
    @param altitude      Altitude of phase screen in km.  This is with respect to the telescope, not
                         sea-level.  [Default: 0.0]
    @param time_step     Interval to use when advancing the screen in time in seconds.
                         [Default: 0.03]
    @param r0_500        Fried parameter setting the amplitude of turbulence; contributes to "size"
                         of the resulting atmospheric PSF.  Specified at wavelength 500 nm, in units
                         of meters.  [Default: 0.2]
    @param L0            Outer scale in meters.  The turbulence power spectrum will smoothly
                         approach a constant at scales larger than L0.  Set to `None` or `np.inf`
                         for a power spectrum without an outer scale.  [Default: 25.0]
    @param vx            x-component wind velocity in meters/second.  [Default: 0.]
    @param vy            y-component wind velocity in meters/second.  [Default: 0.]
    @param alpha         Square root of fraction of phase that is "remembered" between time_steps
                         (i.e., alpha**2 is the fraction remembered). The fraction sqrt(1-alpha**2)
                         is then the amount of turbulence freshly generated in each step.  Setting
                         alpha=1.0 results in a frozen-flow atmosphere.  Note that computing PSFs
                         from frozen-flow atmospheres may be significantly faster than computing
                         PSFs with non-frozen-flow atmospheres.  [Default: 1.0]
    @param rng           Random number generator as a galsim.BaseDeviate().  If None, then use the
                         clock time or system entropy to seed a new generator.  [Default: None]

    Relevant SPIE paper:
    "Remembrance of phases past: An autoregressive method for generating realistic atmospheres in
    simulations"
    Srikar Srinath, Univ. of California, Santa Cruz;
    Lisa A. Poyneer, Lawrence Livermore National Lab.;
    Alexander R. Rudy, UCSC; S. Mark Ammons, LLNL
    Published in Proceedings Volume 9148: Adaptive Optics Systems IV
    September 2014
    """
    def __init__(self, screen_size, screen_scale=None, altitude=0.0, time_step=0.03,
                 r0_500=0.2, L0=25.0, vx=0.0, vy=0.0, alpha=1.0, rng=None,
                 _orig_rng=None, _tab2d=None, _psi=None, _screen=None, _origin=None):

        if screen_scale is None:
            screen_scale = 0.5 * r0_500
        self.npix = galsim._galsim.goodFFTSize(int(np.ceil(screen_size/screen_scale)))
        self.screen_scale = screen_scale
        self.screen_size = screen_size
        self.altitude = altitude
        self.time_step = time_step
        self.r0_500 = r0_500
        if L0 == np.inf:  # Allow np.inf as synonym for None.
            L0 = None
        self.L0 = L0
        self.vx = vx
        self.vy = vy
        self.alpha = alpha

        if rng is None:
            rng = galsim.BaseDeviate()

        # Should only be using private constructor variables when reconstituting from
        # eval(repr(obj)).
        if _orig_rng is not None:
            self.orig_rng = _orig_rng
            self.rng = rng
            self.tab2d = _tab2d
            # Last two might get quickly deleted if alpha==1, but that's okay.
            self.psi = _psi
            self.screen = _screen
            self.origin = _origin
        else:
            self.orig_rng = rng
            self._init_psi()
            self.reset()

        # Free some RAM for frozen-flow screen.
        if self.alpha == 1.0:
            del self.psi, self.screen

    def __str__(self):
        return "galsim.AtmosphericScreen(altitude=%s)" % self.altitude

    def __repr__(self):
        s = ("galsim.AtmosphericScreen(%r, %r, altitude=%r, time_step=%r, r0_500=%r, L0=%r, " +
             "vx=%r, vy=%r, alpha=%r, rng=%r, _origin=array(%r), _orig_rng=%r, _tab2d=%r") % (
                self.screen_size, self.screen_scale, self.altitude, self.time_step, self.r0_500,
                self.L0, self.vx, self.vy, self.alpha, self.rng, self.origin, self.orig_rng,
                self.tab2d)
        if self.alpha == 1.0:
            s += ", _screen=array(%r, dtype=%s)" % (self.screen.to_list(), self.screen.dtype)
            s += ", _psi=array(%r, dtype=%s)" % (self.screen.to_list(), self.screen.dtype)
        s += ")"
        return s

    def __eq__(self, other):
        # This is a bit draconian since two phase screens with different `time_step`s but otherwise
        # equivalent attributes at least start out equal.  However, I like the idea of comparing
        # rng, orig_rng, and time_step better than comparing self.tab2d, so I'm going with that.
        return (isinstance(other, galsim.AtmosphericScreen) and
                self.screen_size == other.screen_size and
                self.screen_scale == other.screen_scale and
                self.altitude == other.altitude and
                self.r0_500 == other.r0_500 and
                self.L0 == other.L0 and
                self.vx == other.vx and
                self.vy == other.vy and
                self.alpha == other.alpha and
                self.orig_rng == other.orig_rng and
                self.rng == other.rng and
                self.time_step == other.time_step and
                np.array_equal(self.origin, other.origin))

    # No hash since this is a mutable class
    __hash__ = None

    def __ne__(self, other): return not self == other

    # Note the magic number 0.00058 is actually ... wait for it ...
    # (5 * (24/5 * gamma(6/5))**(5/6) * gamma(11/6)) / (6 * pi**(8/3) * gamma(1/6)) / (2 pi)**2
    # It nearly impossible to figure this out from a single source, but it can be derived from a
    # combination of Roddier (1981), Sasiela (1994), and Noll (1976).  (These atmosphere people
    # sure like to work alone... )
    kolmogorov_constant = np.sqrt(0.00058)

    def _init_psi(self):
        """Assemble 2D von Karman sqrt power spectrum.
        """
        fx = np.fft.fftfreq(self.npix, self.screen_scale)
        fx, fy = np.meshgrid(fx, fx)

        L0_inv = 1./self.L0 if self.L0 is not None else 0.0
        old_settings = np.seterr(all='ignore')
        self.psi = (1./self.screen_size*self.kolmogorov_constant*(self.r0_500**(-5.0/6.0)) *
                    (fx*fx + fy*fy + L0_inv*L0_inv)**(-11.0/12.0) *
                    self.npix * np.sqrt(np.sqrt(2.0)))
        np.seterr(**old_settings)
        self.psi *= 500.0  # Multiply by 500 here so we can divide by arbitrary lam later.
        self.psi[0, 0] = 0.0

    def _random_screen(self):
        """Generate a random phase screen with power spectrum given by self.psi**2"""
        gd = galsim.GaussianDeviate(self.rng)
        noise = utilities.rand_arr(self.psi.shape, gd)
        return np.fft.ifft2(np.fft.fft2(noise)*self.psi).real

    def advance(self):
        """Advance phase screen realization by self.time_step."""
        # Moving the origin of the aperture in the opposite direction of the wind is equivalent to
        # moving the screen with the wind.
        self.origin -= (self.vx*self.time_step, self.vy*self.time_step)
        # "Boil" the atmsopheric screen if alpha not 1.
        if self.alpha != 1.0:
            self.screen = self.alpha*self.screen + np.sqrt(1.-self.alpha**2)*self._random_screen()
            wrap_screen = np.pad(self.screen, [(0,1), (0,1)], mode='wrap')
            self.tab2d = galsim.LookupTable2D(self._xs, self._ys, wrap_screen, edge_mode='wrap')

    def advance_by(self, dt):
        """Advance phase screen by specified amount of time.

        @param dt  Amount of time in seconds by which to update the screen.
        @returns   The actual amount of time updated, which will differ from `dt` when `dt` is not a
                   multiple of self.time_step.
        """
        if dt < 0:
            raise ValueError("Cannot advance phase screen backwards in time.")
        _nstep = int(np.round(dt/self.time_step))
        if _nstep == 0:
            _nstep = 1
        for i in xrange(_nstep):
            self.advance()
        return _nstep*self.time_step  # return the time *actually* advanced

    # Note -- use **kwargs here so that AtmosphericScreen.stepK and OpticalScreen.stepK
    # can use the same signature, even though they depend on different parameters.
    def stepK(self, **kwargs):
        """Return an appropriate stepk for this atmospheric layer.

        @param lam         Wavelength in nanometers.
        @param scale_unit  Sky coordinate units of output profile. [Default: galsim.arcsec]
        @returns  Good pupil scale size in meters.
        """
        lam = kwargs['lam']
        obj = galsim.Kolmogorov(lam=lam, r0=self.r0_500 * (lam/500.0)**(6./5))
        return obj.stepK()

    def wavefront(self, aper, theta_x=0.0*galsim.degrees, theta_y=0.0*galsim.degrees):
        """ Compute wavefront due to phase screen.

        Wavefront here indicates the distance by which the physical wavefront lags or leads the
        ideal plane wave (pre-optics) or spherical wave (post-optics).

        @param aper     `galsim.Aperture` over which to compute wavefront.
        @param theta_x  x-component of field angle corresponding to center of output array.
        @param theta_y  y-component of field angle corresponding to center of output array.
        @returns        Wavefront lag or lead in nanometers over aperture.
        """
        xs = ys = np.arange(aper.npix)*aper.pupil_plane_scale
        xs += self.origin[0] + 1000*self.altitude*theta_x.tan() - 0.5*aper.pupil_plane_scale*(aper.npix-1)
        ys += self.origin[1] + 1000*self.altitude*theta_y.tan() - 0.5*aper.pupil_plane_scale*(aper.npix-1)
        return self.tab2d(xs, ys)

    def reset(self):
        """Reset phase screen back to time=0."""
        self.rng = self.orig_rng.duplicate()
        self.origin = np.array([0.0, 0.0])

        # Only need to reset/create tab2d if not frozen or doesn't already exist
        if self.alpha != 1.0 or not hasattr(self, 'tab2d'):
            self.screen = self._random_screen()
            wrap_screen = np.pad(self.screen, [(0,1), (0,1)], mode='wrap')
            self._xs = np.linspace(-0.5*self.screen_size, 0.5*self.screen_size, self.npix+1)
            self._ys = self._xs
            self.tab2d = galsim.LookupTable2D(self._xs, self._ys, wrap_screen, edge_mode='wrap')


# Some utilities for working with Zernike polynomials
# Combinations.  n choose r.
def _nCr(n, r):
    from math import factorial
    return factorial(n) / (factorial(r)*factorial(n-r))


# This function stolen from https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
def _noll_to_zern(j):
    """
    Convert linear Noll index to tuple of Zernike indices.
    j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike
    index.
    @param [in] j Zernike mode Noll index
    @return (n, m) tuple of Zernike indices
    @see <https://oeis.org/A176988>.
    """
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")

    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n

    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1) % 2)) / 2.0))
    return (n, m)


def _zern_norm(n, m):
    """Normalization coefficient for zernike (n, m).

    Defined such that \int_{unit disc} Z(n1, m1) Z(n2, m2) dA = \pi if n1==n2 and m1==m2 else 0.0
    """
    if m == 0:
        return np.sqrt(1./(n+1))
    else:
        return np.sqrt(1./(2.*n+2))


def _zern_rho_coefs(n, m):
    """Compute coefficients of radial part of Zernike (n, m).
    """
    kmax = (n-abs(m))/2
    A = [0]*(n+1)
    for k in xrange(kmax+1):
        val = (-1)**k * _nCr(n-k, k) * _nCr(n-2*k, kmax-k) / _zern_norm(n, m)
        A[n-2*k] = val
    return A


def _zern_coef_array(n, m, shape=None):
    """Assemble coefficient array array for evaluating Zernike (n, m) as the real part of a
    bivariate polynomial in abs(rho)^2 and rho, where rho is a complex array indicating position on
    a unit disc.
    """
    if shape is None:
        shape = ((n//2)+1, abs(m)+1)
    out = np.zeros(shape, dtype=np.complex128)
    coefs = np.array(_zern_rho_coefs(n, m), dtype=np.complex128)
    if m < 0:
        coefs *= -1j

    for i, c in enumerate(coefs[abs(m)::2]):
        out[i, abs(m)] = c
    return out


def horner(x, coef):
    """Evaluate univariate polynomial using Horner's method.

    I.e., take A + Bx + Cx^2 + Dx^3 and evaluate it as
    A + x(B + x(C + x(D)))

    @param x     Where to evaluate polynomial.
    @param coef  Polynomial coefficients of increasing powers of x.
    @returns     Polynomial evaluation.  Will take on the shape of x if x is an ndarray.
    """
    result = 0
    for c in coef[::-1]:
        result = result*x + c
    return result


def horner2d(x, y, coefs):
    """Evaluate bivariate polynomial using nested Horner's method.

    @param x      Where to evaluate polynomial (first covariate).  Must be same shape as y.
    @param y      Where to evaluate polynomial (second covariate).  Must be same shape as x.
    @param coefs  2D array-like of coefficients in increasing powers of x and y.
                  The first axis corresponds to increasing the power of y, and the second to
                  increasing the power of x.
    @returns      Polynomial evaluation.  Will take on the shape of x and y if these are ndarrays.
    """
    result = 0
    for coef in coefs[::-1]:
        result = result*x + horner(y, coef)
    return result


class OpticalScreen(object):
    """
    @param aberrations   Zernike polynomial aberrations sequence in waves.
    @param lam_0         Reference wavelength in nanometers at which Zernike aberrations are being
                         specified.  [Default: 500]
    """
    def __init__(self, tip=0.0, tilt=0.0, defocus=0.0, astig1=0.0, astig2=0.0, coma1=0.0, coma2=0.0,
                 trefoil1=0.0, trefoil2=0.0, spher=0.0, aberrations=None, lam_0=500.0):
        if aberrations is None:
            aberrations = np.zeros(12)
            aberrations[2] = tip
            aberrations[3] = tilt
            aberrations[4] = defocus
            aberrations[5] = astig1
            aberrations[6] = astig2
            aberrations[7] = coma1
            aberrations[8] = coma2
            aberrations[9] = trefoil1
            aberrations[10] = trefoil2
            aberrations[11] = spher
        else:
            # Make sure no individual aberrations were passed in, since they will be ignored.
            if any([tip, tilt, defocus, astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher]):
                raise TypeError("Cannot pass in individual aberrations and array!")
            # Aberrations were passed in, so check for right number of entries.
            if len(aberrations) <= 2:
                raise ValueError("Aberrations keyword must have length > 2")
            # Check for non-zero value in first two places.  Probably a mistake.
            if aberrations[0] != 0.0:
                import warnings
                warnings.warn(
                    "Detected non-zero value in aberrations[0] -- this value is ignored!")

        self.aberrations = aberrations
        self.lam_0 = lam_0

        maxn = max(_noll_to_zern(j)[0] for j in range(1, len(self.aberrations)))
        shape = (maxn//2+1, maxn+1)  # (max power of |rho|^2,  max power of rho)
        self.coef_array = np.zeros(shape, dtype=np.complex128)

        for j, ab in enumerate(self.aberrations):
            if j == 0:
                continue
            self.coef_array += _zern_coef_array(*_noll_to_zern(j), shape=shape) * ab

    def __str__(self):
        return "galsim.OpticalScreen(lam_0=%s)" % self.lam_0

    def __repr__(self):
        s = "galsim.OpticalScreen(lam_0=%s" % self.lam_0
        if any(self.aberrations):
            s += ", aberrations=["+",".join(self.aberrations)+"]"
        s += ")"
        return s

    def __eq__(self, other):
        return (isinstance(other, galsim.OpticalScreen) and
                np.array_equal(self.aberrations/self.lam_0, other.aberrations/other.lam_0))

    def __ne__(self, other): return not self == other

    # This screen is immutable, so make a hash for it.
    def __hash__(self):
        return hash(("galsim.AtmosphericScreen", tuple((self.aberattions/self.lam_0).ravel())))

    # Note -- use **kwargs here so that AtmosphericScreen.stepK and OpticalScreen.stepK
    # can use the same signature, even though they depend on different parameters.
    def stepK(self, **kwargs):
        """Return an appropriate stepK for this phase screen.

        @param lam         Wavelength in nanometers.
        @param diam        Aperture diameter in meters.
        @param obscuration Fractional linear aperture obscuration. [Default: 0.0]
        @param scale_unit  Sky coordinate units of output profile. [Default: galsim.arcsec]
        @returns  Good pupil scale size in meters.
        """
        lam = kwargs['lam']
        diam = kwargs['diam']
        obscuration = kwargs.get('obscuration', 0.0)
        # Use an Airy for get appropriate stepK.
        obj = galsim.Airy(lam=lam, diam=diam, obscuration=obscuration)
        return obj.stepK()

    def wavefront(self, aper, theta_x=0.0*galsim.degrees, theta_y=0.0*galsim.degrees):
        """ Compute wavefront due to phase screen.

        Wavefront here indicates the distance by which the physical wavefront lags or leads the
        ideal plane wave (pre-optics) or spherical wave (post-optics).

        @param aper     `galsim.Aperture` over which to compute wavefront.
        @param theta_x  x-component of field angle corresponding to center of output array.
        @param theta_y  y-component of field angle corresponding to center of output array.
        @returns        Wavefront lag or lead in nanometers over aperture.
        """
        # ignore theta_x, theta_y
        r = aper.rho[aper.illuminated]
        rsqr = np.abs(r)**2
        wf = np.zeros(aper.illuminated.shape, dtype=np.float64)
        wf[aper.illuminated] = horner2d(rsqr, r, self.coef_array).real
        return wf * self.lam_0


class PhaseScreenList(object):
    """ List of phase screens that can be turned into a PSF.  Screens can be either atmospheric
    layers or optical phase screens.  Generally, one would assemble a PhaseScreenList object using
    the function `Atmosphere`.  Layers can be added, removed, appended, etc. just like items can be
    manipulated in a python list.  For example:

        # Create an atmosphere with three layers.
        >>> screens = galsim.PhaseScreenList([galsim.AtmosphericScreen(...),
                                              galsim.AtmosphericScreen(...),
                                              galsim.AtmosphericScreen(...)])
        # Add another layer
        >>> screens.append(galsim.AtmosphericScreen(...))
        # Remove the second layer
        >>> del screens[1]
        # Switch the first and second layer.  Silly, but works...
        >>> screens[0], screens[1] = screens[1], screens[0]

    Note that creating new PhaseScreenLists from old PhaseScreenLists copies the wrapped phase
    screens by reference, not value.  Thus, advancing the screens in one list will also advance the
    screens in a copy:

        >>> more_screens = screens[0:2]
        >>> more_screens.advance()
        >>> assert more_screens[0] == screens[0]

        >>> more_screens = galsim.PhaseScreenList(screens)
        >>> screens.reset()
        >>> psf = screens.makePSF(exptime=exptime, ...)        # starts at t=0
        >>> psf2 = more_screens.makePSF(exptime=exptime, ...)  # starts at t=exptime
        >>> assert psf != psf2

    Methods
    -------
    makePSF()          Obtain a PSF from this set of phase screens.  See PhaseScreenPSF docstring
                       for more details.
    advance()          Advance each phase screen in list by self.time_step.
    advance_by()       Advance each phase screen in list by specified amount.
    reset()            Reset each phase screen to t=0.
    wavefront()        Compute the cumulative wavefront due to all screens.

    @param layers  Sequence of phase screens.
    """
    def __init__(self, layers):
        self._layers = list(layers)
        self._update_attrs()  # for now, just updating self.time_step

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, index):
        import numbers
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._layers[index])
        elif isinstance(index, numbers.Integral):
            return self._layers[index]
        else:
            msg = "{cls.__name__} indices must be integers"
            raise TypeError(msg.format(cls=cls))

    def __setitem__(self, index, layer):
        self._layers[index] = layer
        self._update_attrs()

    def __delitem__(self, index):
        del self._layers[index]
        self._update_attrs()

    def append(self, layer):
        self._layers.append(layer)
        self._update_attrs()

    def extend(self, layers):
        self._layers.extend(layers)
        self._update_attrs()

    def __str__(self):
        return "galsim.PhaseScreenList([%s])" % ",".join(str(l) for l in self._layers)

    def __repr__(self):
        return "galsim.PhaseScreenList(%r)" % self._layers

    def __eq__(self, other):
        return self._layers == other._layers

    def __ne__(self, other): return not self == other

    def _update_attrs(self):
        # Update object attributes for current set of layers.  Currently the only attribute is
        # self.time_step.
        # Could have made self.time_step a @property instead of defining _update_attrs(), but then
        # failures would occur late rather than early, which makes debugging more difficult.

        # Each layer must have same value for time_step or no attr time_step.
        time_step = set([l.time_step for l in self if hasattr(l, 'time_step')])
        if len(time_step) == 0:
            self.time_step = None
        elif len(time_step) == 1:
            self.time_step = time_step.pop()
        else:
            raise ValueError("Layer time steps must all be identical or None")

    def advance(self):
        """Advance each phase screen in list by self.time_step."""
        for layer in self:
            try:
                layer.advance()
            except AttributeError:
                # Time indep phase screen.
                pass

    def advance_by(self, dt):
        """Advance each phase screen in list by specified amount of time.

        @param dt  Amount of time in seconds by which to update the screens.
        @returns   The actual amount of time updated, which can potentially (though not necessarily)
                   differ from `dt` when `dt` is not a multiple of self.time_step.
        """
        for layer in self:
            try:
                out = layer.advance_by(dt)
            except AttributeError:
                # Time indep phase screen
                pass
        return out

    def reset(self):
        """Reset phase screens back to time=0."""
        for layer in self:
            try:
                layer.reset()
            except AttributeError:
                # Time indep phase screen
                pass

    def wavefront(self, *args, **kwargs):
        """ Compute wavefront due to phase screen.

        Wavefront here indicates the distance by which the physical wavefront lags or leads the
        ideal plane wave (pre-optics) or spherical wave (post-optics).

        @param aper     `galsim.Aperture` over which to compute wavefront.
        @param theta_x  x-component of field angle corresponding to center of output array.
        @param theta_y  y-component of field angle corresponding to center of output array.
        @returns        Wavefront lag or lead in nanometers over aperture.
        """
        return np.sum(layer.wavefront(*args, **kwargs) for layer in self)

    def makePSF(self, lam, **kwargs):
        """Compute one PSF or multiple PSFs from the current PhaseScreenList, depending on the type
        of (`theta_x`, `theta_y`) or `theta`.  If (`theta_x`, `theta_y`) or `theta` are iterable,
        then return PSFs at the implied field angles in a list.  If `theta_x` and `theta_y` are
        scalars or `theta` is a single tuple, then return a single PSF at the specified field angle.

        Note that this method advances each PhaseScreen in the list, so consecutive calls with the
        same arguments will generally return different PSFs.  Use PhaseScreenList.reset() to reset
        the time to t=0.  See galsim.PhaseScreenPSF docstring for more details.

        @param lam               Wavelength in nanometers at which to compute PSF.
        @param exptime           Time in seconds overwhich to accumulate evolving instantaneous PSF.
                                 [Default: 0.0]
        @param flux              Flux of output PSF [Default: 1.0]
        @param theta_x           x-component of field angle at which to evaluate phase screens and
                                 resulting PSF.  [Default: 0.0*galsim.arcmin]
        @param theta_y           y-component of field angle at which to evaluate phase screens and
                                 resulting PSF.  [Default: 0.0*galsim.arcmin]
        @param theta             Alternative field angle specification.  Single tuple or iterable of
                                 tuples (theta_x, theta_y).
        @param scale_unit        Units to use for the sky coordinates of the output profile.
                                 [Default: galsim.arcsec]
        @param interpolant       Either an Interpolant instance or a string indicating which
                                 interpolant should be used.  Options are 'nearest', 'sinc',
                                 'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the
                                 integer order to use. [default: galsim.Quintic()]
        @param aper              Aperture to use to compute PSF(s).
        @param gsparams          An optional GSParams argument.  See the docstring for GSParams for
                                 details. [default: None]

        The following are optional keywords to use to setup the aperture if `aper` is not provided.

        @param diam              Diameter in meters of aperture used to compute PSF from phases.
        @param pupil_plane_scale       Sampling resolution of the pupil plane in meters.  Either
                                 `pupil_plane_scale` or `npix` must be specified.
        @param pupil_plane_size  Size of the pupil plane in meters.  Note, this may be (in fact, it
                                 usually *should* be) larger than the aperture diameter.
                                 [default: 2.*diam]
        @param circular_pupil    Adopt a circular pupil? [default: True].
        @param obscuration       Fractional linear circular obscuration of pupil. [default: 0.]
        @param nstruts           Number of radial support struts to add to the central obscuration.
                                 [default: 0]
        @param strut_thick       Thickness of support struts as a fraction of pupil diameter.
                                 [default: 0.05]
        @param strut_angle       Angle made between the vertical and the first strut in the CCW
                                 direction; must be an Angle instance.
                                 [default: 0. * galsim.degrees]
        @param oversampling      Optional oversampling factor for the InterpolatedImage. Setting
                                 `oversampling < 1` will produce aliasing in the PSF (not good).
                                 Usually `oversampling` should be somewhat larger than 1.  1.5 is
                                 usually a safe choice.  [default: 1.5]
        @param pad_factor        Additional multiple by which to zero-pad the PSF image to avoid
                                 folding compared to what would be employed for a simple Airy.  Note
                                 that `pad_factor` may need to be increased for stronger
                                 aberrations, i.e., when the equivalent Zernike coefficients become
                                 larger than order unity.  [default: 1.5]
        """
        # Assemble theta as an iterable over 2-tuples of Angles.
        # 5 possible input kwargs cases for theta, theta_x, theta_y.
        # 1) All undefined, in which case set to [(Angle(0), Angle(0))]
        # 2) theta = tuple(Angle, Angle), in which case just need to listify
        # 3) theta_x = Angle, theta_y = Angle, in which case need to listify and zip
        # 4) theta_x/y = [Angle, Angle, ...], in which case need to zip
        # 5) theta = [(Angle, Angle), (Angle, Angle), ...], in which case we're already done.
        single = False
        if 'theta' not in kwargs:  # Case 1, 3, or 4
            theta_x = kwargs.pop('theta_x', 0.0*galsim.arcmin)
            theta_y = kwargs.pop('theta_y', 0.0*galsim.arcmin)
            if not hasattr(theta_x, '__iter__'):
                single = True
            else:
                theta = izip(theta_x, theta_y)
        else:  # Case 2 or 5
            theta = kwargs.pop('theta')
            # 2-tuples are iterable, so to check whether theta is indicating a single pointing, or a
            # generator of pointings we need to look at the first item.  If the first item is
            # iterable itself, then assume theta is an interable of 2-tuple field angles.  We then
            # replace the consumed tuple at the beginning of the generator and go on.  If the first
            # item is scalar, then assume that it's the x-component of a single field angle.
            theta = iter(theta)
            th0 = theta.next()
            if hasattr(th0, '__iter__'):
                theta = chain([th0], theta)
            else:
                theta_x, theta_y = th0, theta.next()
                single = True

        if single:
            return PhaseScreenPSF(self, lam, theta_x=theta_x, theta_y=theta_y, **kwargs)
        else:
            kwargs['_eval_now'] = False
            PSFs = []
            for theta_x, theta_y in theta:
                PSFs.append(PhaseScreenPSF(self, lam, theta_x=theta_x, theta_y=theta_y, **kwargs))

            flux = kwargs.get('flux', 1.0)
            gsparams = kwargs.get('gsparams', None)
            _nstep = PSFs[0]._nstep
            # For non-frozen-flow AtmosphericScreens, it can take much longer to update the
            # atmospheric layers than it does to create an instantaneous PSF, so we exchange the
            # order of the PSF and time loops so we're not recomputing screens needlessly when we go
            # from PSF1 to PSF2 and so on.  For frozen-flow AtmosphericScreens, there's not much
            # difference with either loop order, so we just always make the PSF loop the inner loop.
            for i in xrange(_nstep):
                for PSF in PSFs:
                    PSF._step()
                self.advance()

            suppress_warning = kwargs.pop('suppress_warning', False)
            for PSF in PSFs:
                PSF._finalize(flux, gsparams, suppress_warning)
            return PSFs

    @property
    def r0_500_effective(self):
        """Effective r0_500 for set of screens in list that define an r0_500 attribute."""
        return sum(l.r0_500**(-5./3) for l in self if hasattr(l, 'r0_500'))**(-3./5)

    def stepK(self, **kwargs):
        """Return an appropriate stepK for this list of phase screens.

        The required set of parameters depends on the types of the individual PhaseScreens in the
        PhaseScreenList.  See the documentation for the individual PhaseScreen.pupil_plane_scale methods
        for more details.

        @returns  stepK.
        """
        # Generically, Galsim propagates stepK() for convolutions using
        #   stepk = sum(s**-2 for s in stepks)**(-0.5)
        # We're not actually doing convolution between screens here, though.  In fact, the right
        # relation for Kolmogorov screens uses exponents -5./3 and -3./5:
        #   stepk = sum(s**(-5./3) for s in stepks)**(-3./5)
        # Since most of the layers in a PhaseScreenList are likely to be (nearly) Kolmogorov
        # screens, we'll use that relation.
        return sum(layer.stepK(**kwargs)**(-5./3) for layer in self)**(-3./5)


class PhaseScreenPSF(GSObject):
    """A PSF surface brightness profile constructed by integrating over time the instantaneous PSF
    derived from a set of phase screens and an aperture.

    There are at least three ways construct a PhaseScreenPSF given a PhaseScreenList.  The following
    two statements are equivalent:
        >>> psf = screen_list.makePSF(...)
        >>> psf = PhaseScreenPSF(screen_list, ...)

    The third option is to use screen_list.makePSF() to obtain multiple PSFs at different field
    angles from the same PhaseScreenList over the same simulated time interval.  Depending on the
    details of PhaseScreenList and other arguments, this may be significantly faster than manually
    looping over makePSF().
        >>> psfs = screen_list.makePSF(..., theta_x=[...], theta_y=[...])

    Note that constructing a PhaseScreenPSF advances each PhaseScreen in `screen_list`, so PSFs
    constructed consecutively with the same arguments will generally be different.  Use
    PhaseScreenList.reset() to reset the time to t=0.

        >>> screen_list.reset()
        >>> psf1 = screen_list.makePSF(...)
        >>> psf2 = screen_list.makePSF(...)
        >>> assert psf1 != psf2
        >>> screen_list.reset()
        >>> psf3 = screen_list.makePSF(...)
        >>> assert psf1 == psf3

    Computing a PSF from a phase screen also requires an Aperture be specified.  This can be done
    either directly via the `aper` keyword, or by setting a number of keywords that will be passed
    to the `Aperture.fromPhaseScreenList` constructor.  The `aper` keyword always takes precedence.

    @param screen_list       PhaseScreenList object from which to create PSF.
    @param lam               Wavelength in nanometers at which to compute PSF.
    @param exptime           Time in seconds overwhich to accumulate evolving instantaneous PSF.
                             [Default: 0.0]
    @param flux              Flux of output PSF [Default: 1.0]
    @param theta_x           x-component of field angle at which to evaluate phase screens and
                             resulting PSF.  [Default: 0.0*galsim.arcmin]
    @param theta_y           y-component of field angle at which to evaluate phase screens and
                             resulting PSF.  [Default: 0.0*galsim.arcmin]
    @param scale_unit        Units to use for the sky coordinates of the output profile.
                             [Default: galsim.arcsec]
    @param interpolant       Either an Interpolant instance or a string indicating which interpolant
                             should be used.  Options are 'nearest', 'sinc', 'linear', 'cubic',
                             'quintic', or 'lanczosN' where N should be the integer order to use.
                             [default: galsim.Quintic()]
    @param aper              Aperture to use to compute PSF(s).
    @param gsparams          An optional GSParams argument.  See the docstring for GSParams for
                             details. [default: None]

    The following are optional keywords to use to setup the aperture if `aper` is not provided.

    @param diam              Diameter in meters of aperture used to compute PSF from phases.
    @param pupil_plane_scale       Sampling resolution of the pupil plane in meters.  Either `pupil_plane_scale`
                             or `npix` must be specified.
    @param pupil_plane_size  Size of the pupil plane in meters.  Note, this may be (in fact, it
                             usually *should* be) larger than the aperture diameter.
                             [default: 2.*diam]
    @param circular_pupil    Adopt a circular pupil? [default: True].
    @param obscuration       Fractional linear circular obscuration of pupil. [default: 0.]
    @param nstruts           Number of radial support struts to add to the central obscuration.
                             [default: 0]
    @param strut_thick       Thickness of support struts as a fraction of pupil diameter.
                             [default: 0.05]
    @param strut_angle       Angle made between the vertical and the first strut in the CCW
                             direction; must be an Angle instance.  [default: 0. * galsim.degrees]
    @param oversampling      Optional oversampling factor for the InterpolatedImage. Setting
                             `oversampling < 1` will produce aliasing in the PSF (not good).
                             Usually `oversampling` should be somewhat larger than 1.  1.5 is
                             usually a safe choice.  [default: 1.5]
    @param pad_factor        Additional multiple by which to zero-pad the PSF image to avoid
                             folding compared to what would be employed for a simple Airy.  Note
                             that `pad_factor` may need to be increased for stronger aberrations,
                             i.e. when the equivalent Zernike coefficients become larger than
                             order unity.  [default: 1.5]
    """
    def __init__(self, screen_list, lam, exptime=0.0, flux=1.0,
                 theta_x=0.0*galsim.arcmin, theta_y=0.0*galsim.arcmin,
                 interpolant=None, aper=None, scale_unit=galsim.arcsec,
                 gsparams=None, _eval_now=True, _bar=None, suppress_warning=False,
                 **kwargs):
        # Hidden `_bar` kwarg can be used with astropy.console.utils.ProgressBar to print out a
        # progress bar during long calculations.

        if not isinstance(screen_list, PhaseScreenList):
            screen_list = PhaseScreenList(screen_list)
        self.screen_list = screen_list
        self.lam = float(lam)
        self.exptime = float(exptime)
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.scale_unit = scale_unit
        self.interpolant = interpolant

        if aper is None:
            aper = Aperture(lam=lam, screen_list=screen_list, **kwargs)

        self.aper = aper
        self.scale = aper._sky_scale(self.lam, self.scale_unit)

        self.img = np.zeros(self.aper.illuminated.shape, dtype=np.float64)

        if self.exptime < 0:
            raise ValueError("Cannot integrate PSF for negative time.")
        if self.screen_list.time_step is None:
            self._nstep = 1
        else:
            self._nstep = int(np.round(self.exptime/self.screen_list.time_step))
        # Generate at least one time sample
        if self._nstep == 0:
            self._nstep = 1

        # PhaseScreenList.makePSFs() optimizes multiple PSF evaluation by iterating over PSFs inside
        # of the normal iterate over time loop.  So only do the time loop here and now if we're not
        # doing a makePSFs().
        if _eval_now:
            for i in xrange(self._nstep):
                self._step()
                self.screen_list.advance()
                if _bar is not None:
                    _bar.update()
            self._finalize(flux, gsparams, suppress_warning)

    def __str__(self):
        return ("galsim.PhaseScreenPSF(%s, lam=%s, exptime=%s)" %
                (self.screen_list, self.lam, self.exptime))

    def __repr__(self):
        outstr = ("galsim.PhaseScreenPSF(%r, lam=%r, exptime=%r, flux=%r, theta_x=%r, " +
                  "theta_y=%r, scale_unit=%r, interpolant=%r, gsparams=%r)")
        return outstr % (self.screen_list, self.lam, self.exptime, self.flux, self.theta_x,
                         self.theta_y, self.scale_unit, self.interpolant, self.gsparams)

    def __eq__(self, other):
        # Even if two PSFs were generated with different sets of parameters, they will act
        # identically if their img and interpolant match.
        return (self.img == other.img and
                self.interpolant == other.interpolant and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.PhaseScreenPSF", self.ii))

    def _step(self):
        """Compute the current instantaneous PSF and add it to the developing integrated PSF."""
        wf = self.screen_list.wavefront(self.aper, self.theta_x, self.theta_y)
        expwf = self.aper.illuminated * np.exp(2j * np.pi * wf / self.lam)
        ftexpwf = np.fft.fft2(np.fft.fftshift(expwf))
        self.img += np.abs(ftexpwf)**2

    def _finalize(self, flux, gsparams, suppress_warning):
        """Take accumulated integrated PSF image and turn it into a proper GSObject."""
        self.img = np.fft.fftshift(self.img)
        self.img *= (flux / (self.img.sum() * self.scale**2))
        self.img = galsim.ImageD(self.img.astype(np.float64), scale=self.scale)

        self.ii = galsim.InterpolatedImage(
                self.img, x_interpolant=self.interpolant, calculate_stepk=True, calculate_maxk=True,
                use_true_center=False, normalization='sb', gsparams=gsparams)

        GSObject.__init__(self, self.ii)

        if not suppress_warning:
            specified_stepk = 2*np.pi/(self.img.array.shape[0]*self.scale)
            specified_maxk = np.pi/self.scale
            observed_stepk = self.SBProfile.stepK()
            observed_maxk = self.SBProfile.maxK()

            if observed_stepk < specified_stepk:
                import warnings
                warnings.warn(
                    "The calculated stepk (%g) for PhasePSF is smaller "%observed_stepk +
                    "than what was used to build the wavefront (%g). "%specified_stepk +
                    "This could lead to aliasing problems. ") # +
                    # "Using pad_factor >= %f is recommended."%(pad_factor * stepk / final_stepk))
            # if observed_maxk < 0.5*specified_maxk:
            #     import warnings
            #     warnings.warn(
            #         "The calculated maxk (%g) for PhasePSF is much smaller "%observed_maxk +
            #         "than what was used to build the wavefront (%g). "%specified_maxk +
            #         "This could indicate that oversampling is set too small.")


def _listify(arg):
    """Turn argument into a list if not already iterable."""
    return [arg] if not hasattr(arg, '__iter__') else arg


def _lod_to_dol(lod, N=None):
    """ Generate dicts from dict of lists (with broadcasting).
    Specifically, generate "scalar-valued" kwargs dictionaries from a kwarg dictionary with values
    that are length-N lists, or possibly length-1 lists or scalars that should be broadcasted up to
    length-N lists.
    """
    if N is None:
        N = max(len(v) for v in lod.values() if hasattr(v, '__len__'))
    # Loop through broadcast range
    for i in xrange(N):
        out = {}
        for k, v in lod.iteritems():
            try:
                out[k] = v[i]
            except IndexError:  # It's list-like, but too short.
                if len(v) != 1:
                    raise ValueError("Cannot broadcast kwargs of different non-length-1 lengths.")
                out[k] = v[0]
            except TypeError:  # Value is not list-like, so broadcast it in its entirety.
                out[k] = v
            except:
                raise "Cannot broadcast kwarg {1}={2}".format(k, v)
        yield out


def Atmosphere(screen_size, rng=None, **kwargs):
    """Create an atmosphere as a list of turbulent phase screens at different altitudes.  The
    atmosphere model can then be used to simulate atmospheric PSFs.

    Simulating an atmospheric PSF is typically accomplished by first representing the 3-dimensional
    turbulence in the atmosphere as a series of discrete 2-dimensional phase screens.  These screens
    may blow around in the wind, and may or may not also evolve in time.  This function allows one
    to quickly assemble a list of atmospheric phase screens into a galsim.PhaseScreenList object,
    which can then be used to evaluate PSFs through various columns of atmosphere at different field
    angles.

    The atmospheric screens currently available both produce turbulence that follows a von Karman
    power spectrum.  Specifically, the phase power spectrum in each screen can be written

    psi(nu) = 0.023 r0^(-5/3) (nu^2 + 1/L0^2)^(11/6)

    where psi(nu) is the power spectral density at spatial frequency nu, r0 is the Fried parameter
    (which has dimensions of length) and sets the amplitude of the turbulence, and L0 is the outer
    scale (also dimensions of length) which effectively cuts off the power spectrum at large scales
    (small nu).  Typical values for r0 are ~0.1 to 0.2 meters, which corresponds roughly to PSF
    FWHMs of ~0.5 to 1.0 arcsec for optical wavelengths.  Note that r0 is a function of wavelength,
    scaling like r0 ~ wavelength^(6/5).  To reduce confusion, the input parameter here is named
    r0_500 and refers explicitly to the Fried parameter at a wavelength of 500 nm.  The outer scale
    is typically in the 10s of meters and does not vary with wavelength.

    To create multiple layers, simply specify keyword arguments as length-N lists instead of scalars
    (works for all arguments except `time_step` and `rng`).  If, for any of these keyword arguments,
    you want to use the same value for each layer, then you can just specify the argument as a
    scalar and the function will automatically broadcast it into a list with length equal to the
    longest found keyword argument list.  Note that it is an error to specify two keywords with
    lists of different lengths (unless the of one of them is length is 1).

    The one exception to the above is the keyword `r0_500`.  The effective Fried parameter for a set
    of atmospheric layers is r0_500_effective = (sum(r**(-5./3) for r in r0_500s))**(-3./5).
    Providing `r0_500` as a scalar or length-1 list will result in broadcasting such that the
    effective Fried parameter for the whole set of layers equals the input argument.

    As an example, the following code approximately creates the atmosphere used by Jee+Tyson(2011)
    for their study of atmospheric PSFs for LSST.  Note this code takes about ~3-4 minutes to run on
    a fast laptop, and will consume about (8192**2 pixels) * (8 bytes) * (6 screens) ~ 3 GB of
    RAM in its final state, and more at intermediate states.

        >>> altitude = [0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
        >>> r0_500_effective = 0.16  # m
        >>> weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
        >>> r0_500 = [r0_500_effective * w**(-3./5) for w in weights]
        >>> speed = np.random.uniform(0, 20, size=6)  # m/s
        >>> direction = [np.random.uniform(0, 360)*galsim.degrees for i in xrange(6)]
        >>> npix = 8192
        >>> screen_scale = 0.5 * r0_500_effective
        >>> atm = galsim.Atmosphere(r0_500=r0_500, screen_size=screen_scale*npix, time_step=0.005,
                                    altitude=altitude, L0=25.0, speed=speed,
                                    direction=direction, screen_scale=screen_scale)

    Once the atmosphere is constructed, a 15-sec exposure PSF (using an 8.4 meter aperture and
    default settings) takes about 150 sec to generate on a fast laptop.

        >>> psf = atm.makePSF(diam=8.4, exptime=15.0, obscuration=0.6)

    Many factors will affect the timing of results, of course, including aperture diameter, gsparams
    settings, pad_factor and oversampling options to makePSF, time_step and exposure time, frozen
    vs. non-frozen atmospheric layers, and so on.

    @param r0_500        Fried parameter setting the amplitude of turbulence; contributes to "size"
                         of the resulting atmospheric PSF.  Specified at wavelength 500 nm, in units
                         of meters.  [Default: 0.2]
    @param screen_size   Physical extent of square phase screen in meters.  This should be large
                         enough to accommodate the desired field-of-view of the telescope as well as
                         the meta-pupil defined by the wind speed and exposure time.  Note that
                         the screen will have periodic boundary conditions, so the code will run
                         with a smaller sized screen, though this may introduce artifacts into PSFs
                         or PSF correlations functions. Note that screen_size may be tweaked by the
                         initializer to ensure screen_size is a multiple of screen_scale.
    @param time_step     Interval to use when advancing the screen in time in seconds.
                         [Default: 0.03]
    @param altitude      Altitude of phase screen in km.  This is with respect to the telescope, not
                         sea-level.  [Default: 0.0]
    @param L0            Outer scale in meters.  The turbulence power spectrum will smoothly
                         approach a constant at scales larger than L0.  Set to `None` or `np.inf`
                         for a power spectrum without an outer scale.  [Default: 25.0]
    @param speed         Wind speed in meters/second.  [Default: 0.0]
    @param direction     Wind direction as galsim.Angle [Default: 0.0 * galsim.degrees]
    @param alpha         Fraction of phase that is "remembered" between time_steps.  The fraction
                         1-alpha is then the amount of turbulence freshly generated in each step.
                         [Default: 1.0]
    @param screen_scale  Physical pixel scale of phase screen in meters.  A fraction of the Fried
                         parameter is usually sufficiently small, but users should test the effects
                         of this parameter to ensure robust results.
                         [Default: half of r0_500 for each screen]
    @param rng           Random number generator as a galsim.BaseDeviate().  If None, then use the
                         clock time or system entropy to seed a new generator.  [Default: None]
    """
    # Fill in screen_size here, since there isn't a default in AtmosphericScreen
    kwargs['screen_size'] = _listify(screen_size)

    # Set default r0_500 here, so that by default it gets broadcasted below such that the
    # _total_ r0_500 from _all_ screens is 0.2 m.
    if 'r0_500' not in kwargs:
        kwargs['r0_500'] = [0.2]
    kwargs['r0_500'] = _listify(kwargs['r0_500'])

    # Turn speed, direction into vx, vy
    if 'speed' in kwargs:
        kwargs['speed'] = _listify(kwargs['speed'])
        if 'direction' not in kwargs:
            kwargs['direction'] = [0*galsim.degrees]*len(kwargs['speed'])
        kwargs['vx'], kwargs['vy'] = zip(*[v*d.sincos()
                                           for v, d in zip(kwargs['speed'],
                                                           kwargs['direction'])])
        del kwargs['speed']
        del kwargs['direction']

    # Determine broadcast size
    nmax = max(len(v) for v in kwargs.values() if hasattr(v, '__len__'))

    # Broadcast r0_500 here, since logical combination of indiv layers' r0s is complex:
    if len(kwargs['r0_500']) == 1:
        kwargs['r0_500'] = [nmax**(3./5) * kwargs['r0_500'][0]] * nmax

    return PhaseScreenList(AtmosphericScreen(rng=rng, **kw) for kw in _lod_to_dol(kwargs, nmax))


#  Args not yet implemented:
#  suppress_warning, max_size
#  Also pickling.
class OpticalPSF(GSObject):
    _req_params = {}
    _opt_params = {
        "diam": float,
        "defocus": float,
        "astig1": float,
        "astig2": float,
        "coma1": float,
        "coma2": float,
        "trefoil1": float,
        "trefoil2": float,
        "spher": float,
        "circular_pupil": bool,
        "obscuration": float,
        "oversampling": float,
        "pad_factor": float,
        "suppress_warning": bool,
        "max_size": float,
        "interpolant": str,
        "flux": float,
        "nstruts": int,
        "strut_thick": float,
        "strut_angle": galsim.Angle,
        "pupil_plane_im": str,
        "pupil_angle": galsim.Angle,
        "scale_unit": str}
    _single_params = [{"lam_over_diam": float, "lam": float}]
    _takes_rng = False

    def __init__(self, lam_over_diam=None, lam=None, diam=None, tip=0., tilt=0., defocus=0.,
                 astig1=0., astig2=0., coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0.,
                 aberrations=None, circular_pupil=True, obscuration=0., interpolant=None,
                 oversampling=1.5, pad_factor=1.5, flux=1., nstruts=0, strut_thick=0.05,
                 strut_angle=0.*galsim.degrees, pupil_plane_im=None,
                 pupil_angle=0.*galsim.degrees, scale_unit=galsim.arcsec, gsparams=None,
                 _pupil_plane_scale=None, _pupil_plane_size=None):
        # Need to handle lam/diam vs. lam_over_diam here since lam by itself is needed for
        # OpticalScreen.
        if lam_over_diam is not None:
            if lam is not None or diam is not None:
                raise TypeError("If specifying lam_over_diam, then do not specify lam or diam")
            lam = 500.  # Arbitrary
            diam = lam*1.e-9 / lam_over_diam * galsim.radians / scale_unit
        else:
            if lam is None or diam is None:
                raise TypeError("If not specifying lam_over_diam, then specify lam AND diam")

        # Make the optical screen.
        optics_screen = galsim.OpticalScreen(
                defocus=defocus, astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2,
                trefoil1=trefoil1, trefoil2=trefoil2, spher=spher, aberrations=aberrations,
                lam_0=lam)
        self._screens = galsim.PhaseScreenList([optics_screen])

        # Make the aperture.
        self._aper = galsim.Aperture(
                diam, lam=lam, circular_pupil=circular_pupil, obscuration=obscuration,
                nstruts=nstruts, strut_thick=strut_thick, strut_angle=strut_angle,
                oversampling=oversampling, pad_factor=pad_factor,
                pupil_plane_im=pupil_plane_im, pupil_angle=pupil_angle,
                _pupil_plane_scale=_pupil_plane_scale, _pupil_plane_size=_pupil_plane_size)

        # Finally, put together to make the PSF.
        self._psf = galsim.PhaseScreenPSF(self._screens, lam=lam, flux=flux, aper=self._aper,
                                          interpolant=interpolant, scale_unit=scale_unit,
                                          gsparams=gsparams)
        GSObject.__init__(self, self._psf)

    def __str__(self):
        screen = self._psf.screen_list[0]
        s = "galsim.OpticalPSF(lam=%s, diam=%s" % (screen.lam_0, self._aper.diam)
        if any(screen.aberrations):
            s += ", aberrations=[" + ",".join(str(ab) for ab in screen.aberrations) + "]"
        if hasattr(self._aper, '_circular_pupil'):
            s += self._aper._geometry_str()
        if self._psf.flux != 1.0:
            s += ", flux=%s" % self._psf.flux
        s += ")"
        return s

    def __repr__(self):
        screen = self._psf.screen_list[0]
        s = "galsim.OpticalPSF(lam=%s, diam=%s" % (screen.lam_0, self._aper.diam)
        if any(screen.aberrations):
            s += ", aberrations=[" + ",".join(str(ab) for ab in screen.aberrations) + "]"
        if hasattr(self._aper, '_circular_pupil'):
            s += self._aper._geometry_repr()
        if self._psf.flux != 1.0:
            s += ", flux=%s" % self._psf.flux
        s += ")"
        return s

    def __eq__(self, other):
        # Should it be possible for an OpticalPSF to be equal to a PhaseScreenPSF?  It seems simpler
        # to just vote no, so I'm doing that for now, though I'm certainly open to changing this.
        return (isinstance(other, galsim.OpticalPSF) and
                self._psf == other._psf)

    def __hash__(self):
        return hash(("galsim.OpticalPSF", self._psf))
