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

import numpy as np
import galsim
import utilities
from galsim import GSObject


def listify(arg):
    """Turn argument into a list if not already iterable."""
    return [arg] if not hasattr(arg, '__iter__') else arg


def broadcast(arg, n):
    """Turn length-1 iterable into length-n list."""
    return [arg[0]]*n if len(arg) == 1 else arg


def generate_pupil(npix, pupil_plane_size, diam=None, obscuration=None):
    """ Generate a pupil transmission array (0's and 1's) for a circular aperture and potentially a
    central circular obscuration.

    @param npix              Number of pixels across pupil array.
    @param pupil_plane_size  Physical size of the array defining the pupil in meters.
    @param diam              Diameter of aperture in meters.
    @param obscuration       Fractional linear obscuration.
    @returns array of 0's and 1's indicating pupil transmission function.
    """
    aper = np.ones((npix, npix), dtype=np.float64)
    if diam is not None:
        radius = 0.5*diam
        u = np.fft.fftshift(np.fft.fftfreq(npix, 1./pupil_plane_size))
        u, v = np.meshgrid(u, u)
        rsqr = u**2 + v**2
        aper[rsqr > radius**2] = 0.0
        if obscuration is not None:
            aper[rsqr < (radius*obscuration)**2] = 0.0
    return aper


class PhaseScreen(object):
    # ABC for phase screens.  Subclasses should implement:
    # advance()
    # advance_by()
    # reset()
    # path_difference()
    # pupil_scale()
    # Non-evolving screens, such as one representing optical aberrations, can probably accept the
    # default advance(), advance_by(), and reset() methods which are all no-ops.  All subclasses
    # will need to implement their own path_difference() and pupil_scale() though.
    """ Abstract base class for a phase screen to use in generating a PSF using Fourier optics.
    Not intended to be instantiated directly.

    @param screen_size   Physical extent of square phase screen in meters.  This should be large
                         enough to accommodate the desired field-of-view of the telescope as well as
                         the meta-pupil defined by the wind velocity and exposure time.  Note that
                         the screen will have periodic boundary conditions, so the code will run
                         with a smaller sized screen, though this may introduce artifacts into PSFs
                         or PSF correlations functions. Note that screen_size may be tweaked by the
                         initializer to ensure screen_size is a multiple of screen_scale.
    @param screen_scale  Physical pixel scale of phase screen in meters.  A fraction of the Fried
                         parameter is usually sufficiently small, but users should test the effects
                         of this parameter to ensure robust results.
    @param altitude      Altitude of phase screen in km.  This is with respect to the telescope, not
                         sea-level.  [Default: 0.0]
    """
    def __init__(self, screen_size, screen_scale, altitude=0.0):
        self.npix = int(np.ceil(screen_size/screen_scale))
        self.screen_scale = screen_scale
        self.screen_size = self.screen_scale * self.npix
        self.altitude = altitude

    def advance(self):
        """Advance phase screen realization by self.time_step."""
        # Default is a no-op, which would be appropriate for an optics phase screen, for example.
        # For an atmsopheric phase screen, this should update the atmospheric layer to account for
        # wind, boiling, etc.
        pass

    def advance_by(self, dt):
        """Advance phase screen by specified amount of time.

        @param dt  Amount of time in seconds by which to update the screen.
        @returns   The actual amount of time updated, which can potentially (though not necessarily)
                   differ from `dt` when `dt` is not a multiple of self.time_step.
        """
        return dt

    def reset(self):
        """Reset phase screen back to time=0."""
        # For time-independent screens, this is a no-op.
        pass

    def path_difference(self, nx, scale, theta_x=None, theta_y=None):
        """ Compute effective pathlength differences due to phase screen.

        @param nx       Size of output array
        @param scale    Scale of output array pixels in meters
        @param theta_x  Field angle corresponding to center of output array.
        @param theta_y  Ditto.
        @returns   Array of pathlength differences in nanometers.  Multiply by 2pi/wavelength to get
                   array of phase differences.
        """
        # This should return an nx-by-nx pixel array with scale `scale` (in meters) representing the
        # effective difference in path length (nanometers) for rays originating from different
        # points in the pupil plane.  The `theta_x` and `theta_y` params indicate the position on
        # the focal plane, or equivalently the position on the sky from which the rays originate.
        raise NotImplementedError

    def pupil_scale(self, lam, scale_unit=galsim.arcsec):
        """Compute a good pupil_scale in meters for this atmospheric layer.

        @param lam         Wavelength in nanometers.
        @param scale_unit  Sky coordinate units of output profile. [Default: galsim.arcsec]
        @returns  Good pupil scale size in meters.
        """
        raise NotImplementedError


class AtmosphericScreen(PhaseScreen):
    """ An atmospheric phase screen that can drift in the wind and evolves ("boils") over time.  The
    initial phases and fractional phase updates are drawn from a von Karman power spectrum, which is
    defined by a Fried parameter that effectively sets the amplitude of the turbulence, and an outer
    scale that sets the scale beyond which the turbulence power goes (smoothly) to zero.

    @param screen_size   Physical extent of square phase screen in meters.  This should be large
                         enough to accommodate the desired field-of-view of the telescope as well as
                         the meta-pupil defined by the wind velocity and exposure time.  Note that
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
                 r0_500=0.2, L0=25.0, vx=0.0, vy=0.0, alpha=1.0, rng=None):

        if screen_scale is None:
            screen_scale = 0.5 * r0_500
        super(AtmosphericScreen, self).__init__(screen_size, screen_scale, altitude)

        self.time_step = time_step
        self.r0_500 = r0_500
        self.L0 = L0
        self.vx = vx
        self.vy = vy
        self.alpha = alpha

        if rng is None:
            rng = galsim.BaseDeviate()
        self.orig_rng = rng

        self._init_psi()

        self.reset()

        # Free some RAM for frozen-flow screen
        if self.alpha == 1.0:
            del self.psi, self.screen

    def __str__(self):
        return "galsim.AtmosphericScreen(altitude=%s)" % self.altitude

    def __repr__(self):
        outstr = ("galsim.AtmosphericScreen(%r, %r, altitude=%r, time_step=%r, " +
                  "r0_500=%r, L0=%r, vx=%r, vy=%r, alpha=%r, rng=%r)")
        return outstr % (self.screen_size, self.screen_scale, self.altitude, self.time_step,
                         self.r0_500, self.L0, self.vx, self.vy, self.alpha, self.rng)

    def __eq__(self, other):
        sL0 = self.L0 if self.L0 is not None else np.inf
        oL0 = other.L0 if other.L0 is not None else np.inf
        return (self.screen_size == other.screen_size and
                self.screen_scale == other.screen_scale and
                self.altitude == other.altitude and
                self.r0_500 == other.r0_500 and
                sL0 == oL0 and
                self.vx == other.vx and
                self.vy == other.vy and
                self.alpha == other.alpha and
                self.rng == other.rng)

    def __ne__(self, other):
        return not self == other

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
        self.psi = (1./self.screen_size*self.kolmogorov_constant*(self.r0_500**(-5.0/6.0)) *
                    (fx*fx + fy*fy + L0_inv*L0_inv)**(-11.0/12.0) *
                    self.npix * np.sqrt(np.sqrt(2.0)))
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
            self.tab2d = galsim.LookupTable2D(self._x0, self._y0, self._dx, self._dy, self.screen,
                                              edge_mode='wrap')

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

    # Both types of atmospheric screens determine their pupil scales (essentially stepK()) from the
    # Kolmogorov profile with matched Fried parameter r0.
    def pupil_scale(self, lam, scale_unit=galsim.arcsec):
        """Compute a good pupil_scale in meters for this atmospheric layer.

        @param lam         Wavelength in nanometers.
        @param scale_unit  Sky coordinate units of output profile. [Default: galsim.arcsec]
        @returns  Good pupil scale size in meters.
        """
        obj = galsim.Kolmogorov(lam=lam, r0=self.r0_500 * (lam/500.0)**(6./5))
        return obj.stepK() * lam*1.e-9 * galsim.radians / scale_unit

    def path_difference(self, nx, scale, theta_x=0.0*galsim.degrees, theta_y=0.0*galsim.degrees):
        """ Compute effective pathlength differences due to phase screen.

        @param nx       Size of output array
        @param scale    Scale of output array pixels in meters
        @param theta_x  x-component of field angle corresponding to center of output array.
        @param theta_y  y-component of field angle corresponding to center of output array.
        @returns   Array of pathlength differences in nanometers.  Multiply by 2pi/wavelength to get
                   array of phase differences.
        """
        xmin = self.origin[0] + 1000*self.altitude*theta_x.tan() - 0.5*scale*(nx-1)
        xmax = xmin + scale*(nx-1)
        ymin = self.origin[1] + 1000*self.altitude*theta_y.tan() - 0.5*scale*(nx-1)
        ymax = ymin + scale*(nx-1)
        return self.tab2d.eval_grid(xmin, xmax, nx, ymin, ymax, nx)

    def reset(self):
        """Reset phase screen back to time=0."""
        self.rng = self.orig_rng.duplicate()
        self.origin = np.array([0.0, 0.0])

        # Only need to reset/create tab2d if not frozen or doesn't already exist
        if self.alpha != 1.0 or not hasattr(self, 'tab2d'):
            self.screen = self._random_screen()
            self._x0 = self._y0 = -0.5*(self.npix-1)*self.screen_scale
            self._dx = self._dy = self.screen_scale
            self.tab2d = galsim.LookupTable2D(self._x0, self._y0, self._dx, self._dy, self.screen,
                                              edge_mode='wrap')


class PhaseScreenList(object):
    """ List of phase screens that can be turned into a PSF.  Screens can be either atmospheric
    layers or optical phase screens.  Generally, one would assemble a PhaseScreenList object using
    the function `Atmosphere`.  Layers can be added, removed, appended, etc. just like items can be
    manipulated in a python list.  For example:

    # Create an atmosphere with three layers.
    > screens = galsim.PhaseScreenList([galsim.AtmosphericScreen(...),
                                        galsim.AtmosphericScreen(...),
                                        galsim.AtmosphericScreen(...)])
    # Add another layer
    > screens.append(galsim.AtmosphericScreen(...))
    # Remove the second layer
    > del screens[1]
    # Switch the first and second layer.  Silly, but works...
    > screens[0], screens[1] = screens[1], screens[0]

    Note that creating new PhaseScreenLists from old PhaseScreenLists copies the wrapped phase
    screens by reference, not value.  Thus, advancing the screens in one list will also advance the
    screens in a copy:

    > more_screens = screens[0:2]
    > more_screens.advance()
    > assert more_screens[0] == screens[0]

    > more_screens = galsim.PhaseScreenList(screens)
    > screens.reset()
    > psf = screens.makePSF(exptime=exptime, ...)        # starts at t=0
    > psf2 = more_screens.makePSF(exptime=exptime, ...)  # starts at t=exptime
    > assert psf != psf2

    Methods
    -------
    makePSF()          Obtain a PSF from this set of phase screens.  See PhaseScreenPSF docstring
                       for more details.
    advance()          Advance each phase screen in list by self.time_step.
    advance_by()       Advance each phase screen in list by specified amount.
    reset()            Reset each phase screen to t=0.
    path_difference()  Compute the cumulative pathlength difference due to all screens.

    @param layers  Sequence of phase screens.
    """
    def __init__(self, layers):
        self._layers = layers
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
        return "galsim.PhaseScreenList(%s)" % self._layers

    def __repr__(self):
        return "galsim.PhaseScreenList(%r)" % self._layers

    def __eq__(self, other):
        return (len(self) == len(other) and
                all(sl == ol for sl, ol in zip(self._layers, other._layers)))

    def __ne__(self, other):
        return not self == other

    def _update_attrs(self):
        # Update object attributes for current set of layers.  Currently the only attribute is
        # self.time_step.
        # Could have made self.time_step a @property instead of defining _update_attrs(), but then
        # failures would occur late rather than early, which makes debugging more difficult.

        # Must have unique time_step or time_step is None (for time-indep screen)
        time_step = set([l.time_step for l in self if l.time_step is not None])
        if len(time_step) == 0:
            self.time_step = None
        elif len(time_step) == 1:
            self.time_step = time_step.pop()
        else:
            raise ValueError("Layer time steps must all be identical or None")

    def advance(self):
        """Advance each phase screen in list by self.time_step."""
        for layer in self:
            layer.advance()

    def advance_by(self, dt):
        """Advance each phase screen in list by specified amount of time.

        @param dt  Amount of time in seconds by which to update the screens.
        @returns   The actual amount of time updated, which can potentially (though not necessarily)
                   differ from `dt` when `dt` is not a multiple of self.time_step.
        """
        for layer in self:
            out = layer.advance_by(dt)
        return out

    def reset(self):
        """Reset phase screens back to time=0."""
        for layer in self:
            layer.reset()

    def path_difference(self, *args, **kwargs):
        """ Compute cumulative effective pathlength differences due to phase screens.

        @param nx       Size of output array
        @param scale    Scale of output array pixels in meters
        @param theta_x  Field angle corresponding to center of output array.
        @param theta_y  Ditto.
        @returns   Array of pathlength differences in nanometers.  Multiply by 2pi/wavelength to get
                   array of phase differences.
        """
        return np.sum(layer.path_difference(*args, **kwargs) for layer in self)

    def makePSF(self, diam, **kwargs):
        """Compute one PSF or multiple PSFs from the current PhaseScreenList, depending on the type
        of `theta_x` and `theta_y`.  If `theta_x` and `theta_y` are iterable, then return PSFs at
        the implied field angles in a list.  If `theta_x` and `theta_y` are scalars, return a single
        PSF at the specified field angle.

        Note that this method advances each PhaseScreen in the list, so consecutive calls with the
        same arguments will generally return different PSFs.  Use PhaseScreenList.reset() to reset
        the time to t=0.  See galsim.PhaseScreenPSF docstring for more details.

        @param diam             Diameter in meters of aperture used to compute PSF from phases.
        @param lam              Wavelength in nanometers used to compute PSF.  [Default: 500]
        @param exptime          Time in seconds overwhich to accumulate evolving instantaneous PSFs
                                [Default: 0.0]
        @param flux             Flux of output PSF [Default: 1.0]
        @param theta_x          Iterable or scalar for x-component of field angle at which to
                                evaluate phase screens and the resulting PSF(s).
                                [Default: 0.0*galsim.arcmin]
        @param theta_y          Iterable or scalar for y-component of field angle at which to
                                evaluate phase screens and the resulting PSF(s).
                                [Default: 0.0*galsim.arcmin]
        @param scale_unit       Units to use for the sky coordinates of the output profile.
                                [Default: galsim.arcsec]
        @param obscuration      Linear dimension of central obscuration as fraction of pupil linear
                                dimension, [0., 1.).  [Default: 0]
        @param interpolant      Either an Interpolant instance or a string indicating which
                                interpolant should be used.  Options are 'nearest', 'sinc',
                                'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the
                                integer order to use. [Default: galsim.Quintic()]
        @param pad_factor       Additional multiple by which to zero-pad wavefront images before
                                Fourier Transforming to produce the PSF.  [Default: 1.5]
        @param oversampling     Optional oversampling factor for the wavefront image.  Higher
                                sampling reduces folding of the PSF and allows one to better model
                                the PSF wings.  Usually `oversampling` should be somewhat larger
                                than 1.  1.5 is usually a safe choice.  [Default: 1.5]
        @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                                details. [default: None]
        """
        theta_x = kwargs.get('theta_x', 0.0*galsim.arcmin)
        theta_y = kwargs.get('theta_y', 0.0*galsim.arcmin)
        if not hasattr(theta_x, '__iter__') and not hasattr(theta_y, '__iter__'):
            return PhaseScreenPSF(self, diam, **kwargs)
        else:
            kwargs['_eval_now'] = False
            PSFs = []
            for theta_x, theta_y in zip(kwargs.pop('theta_x'), kwargs.pop('theta_y')):
                PSFs.append(PhaseScreenPSF(self, diam, theta_x=theta_x, theta_y=theta_y, **kwargs))

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

            for PSF in PSFs:
                PSF._finalize(flux, gsparams)
            return PSFs

    @property
    def r0_500_effective(self):
        """Effective r0_500 for set of screens in list that define an r0_500 attribute."""
        return sum(l.r0_500**(-5./3) for l in self if hasattr(l, 'r0_500'))**(-3./5)


class PhaseScreenPSF(GSObject):
    """A PSF surface brightness profile constructed by integrating over time the instantaneous PSF
    derived from a set of phase screens and an aperture.

    There are at least three ways construct a PhaseScreenPSF given a PhaseScreenList.  The following
    two statements are equivalent:
    > psf = screen_list.makePSF(...)
    > psf = PhaseScreenPSF(screen_list, ...)

    The third option is to use screen_list.makePSF() to obtain multiple PSFs at different field
    angles from the same PhaseScreenList over the same simulated time interval.  Depending on the
    details of PhaseScreenList and other arguments, this may be significantly faster than manually
    looping over makePSF().
    > psfs = screen_list.makePSF(..., theta_x=[...], theta_y=[...])

    Note that constructing a PhaseScreenPSF advances each PhaseScreen in `screen_list`, so PSFs
    constructed consecutively with the same arguments will generally be different.  Use
    PhaseScreenList.reset() to reset the time to t=0.

    > screen_list.reset()
    > psf1 = screen_list.makePSF(...)
    > psf2 = screen_list.makePSF(...)
    > assert psf1 != psf2
    > screen_list.reset()
    > psf3 = screen_list.makePSF(...)
    > assert psf1 == psf3

    @param screen_list      PhaseScreenList object from which to create PSF.
    @param diam             Diameter in meters of aperture used to compute PSF from phases.
    @param lam              Wavelength in nanometers used to compute PSF.  [Default: 500]
    @param exptime          Time in seconds overwhich to accumulate evolving instantaneous PSF.
                            [Default: 0.0]
    @param flux             Flux of output PSF [Default: 1.0]
    @param theta_x          x-component of field angle at which to evaluate phase screens and
                            resulting PSF.  [Default: 0.0*galsim.arcmin]
    @param theta_y          y-component of field angle at which to evaluate phase screens and
                            resulting PSF.  [Default: 0.0*galsim.arcmin]
    @param scale_unit       Units to use for the sky coordinates of the output profile.
                            [Default: galsim.arcsec]
    @param obscuration      Linear dimension of central obscuration as fraction of pupil linear
                            dimension, [0., 1.).  [Default: 0]
    @param interpolant      Either an Interpolant instance or a string indicating which interpolant
                            should be used.  Options are 'nearest', 'sinc', 'linear', 'cubic',
                            'quintic', or 'lanczosN' where N should be the integer order to use.
                            [default: galsim.Quintic()]
    @param pad_factor       Additional multiple by which to zero-pad the PSF image to avoid folding
                            compared to what would be employed for a simple Airy.  Note that
                            `pad_factor` may need to be increased for stronger aberrations, i.e.
                            those larger than order unity.  [default: 1.5]
    @param oversampling     Optional oversampling factor for the wrapped InterpolatedImage. Setting
                            `oversampling < 1` will produce aliasing in the PSF (not good).
                            Usually `oversampling` should be somewhat larger than 1.  1.5 is
                            usually a safe choice.  [default: 1.5]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """
    def __init__(self, screen_list, diam, lam=500., exptime=0.0, flux=1.0,
                 theta_x=0.0*galsim.arcmin, theta_y=0.0*galsim.arcmin,
                 scale_unit=galsim.arcsec, interpolant=None,
                 obscuration=0.0,
                 pad_factor=1.5, oversampling=1.5,
                 _pupil_plane_size=None, _pupil_scale=None,
                 gsparams=None, _eval_now=True, _bar=None):
        # Hidden `_bar` kwarg can be used with astropy.console.utils.ProgressBar to print out a
        # progress bar during long calculations.

        self.screen_list = screen_list
        self.lam = lam
        self.exptime = exptime
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.scale_unit = scale_unit
        self.interpolant = interpolant
        self.diam = diam
        self.obscuration = obscuration
        self.pad_factor = pad_factor
        self.oversampling = oversampling

        if _pupil_scale is None:
            # Generically, Galsim propagates stepK() for convolutions using
            #   scale = sum(s**-2 for s in scales)**(-0.5)
            # We're not actually doing convolution between screens here, though.  In fact, the right
            # relation for Kolmogorov screens uses exponents -5./3 and -3./5:
            #   scale = sum(s**(-5./3) for s in scales)**(-3./5)
            # Since most of the layers in a PhaseScreenList are likely to be (nearly) Kolmogorov
            # screens, we'll use that relation.
            _pupil_scale = (sum(layer.pupil_scale(lam)**(-5./3) for layer in screen_list))**(-3./5)
            _pupil_scale /= oversampling
        self._pupil_scale = _pupil_scale
        # Note _pupil_plane_size sets the size of the array defining the pupil, which will generally
        # be somewhat larger than the diameter of the pupil itself.
        if _pupil_plane_size is None:
            _pupil_plane_size = self.diam * self.pad_factor
        self._npix = int(np.ceil(_pupil_plane_size/self._pupil_scale))
        self._pupil_plane_size = self._npix * self._pupil_scale

        self.scale = 1e-9*self.lam/self._pupil_plane_size * galsim.radians / self.scale_unit

        self.aper = generate_pupil(self._npix, self._pupil_plane_size, self.diam, self.obscuration)
        self.img = np.zeros_like(self.aper)

        if self.exptime < 0:
            raise ValueError("Cannot integrate PSF for negative time.")
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
            self._finalize(flux, gsparams)

    def __str__(self):
        return ("galsim.PhaseScreenPSF(%s, lam=%s, exptime=%s)" %
                (self.screen_list, self.lam, self.exptime))

    def __repr__(self):
        outstr = ("galsim.PhaseScreenPSF(%r, lam=%r, exptime=%r, flux=%r, theta_x=%r, " +
                  "theta_y=%r, scale_unit=%r, interpolant=%r, diam=%r, obscuration=%r, " +
                  "pad_factor=%r, oversampling=%r, gsparam=%r)")
        return outstr % (self.screen_list, self.lam, self.exptime, self.flux, self.theta_x,
                         self.theta_y, self.scale_unit, self.interpolant, self.diam,
                         self.obscuration, self.pad_factor, self.oversampling, self.gsparams)

    def __eq__(self, other):
        # Even if two PSFs were generated with different sets of parameters, they will act
        # identically if their img and interpolant match.
        return (self.img == other.img and
                self.interpolant == other.interpolant)

    def __ne__(self, other):
        return not self == other

    def _step(self):
        """Compute the current instantaneous PSF and add it to the developing integrated PSF."""
        path_difference = self.screen_list.path_difference(self._npix, self._pupil_scale,
                                                           self.theta_x, self.theta_y)
        wf = self.aper * np.exp(2j * np.pi * path_difference / self.lam)
        ftwf = np.fft.ifft2(np.fft.ifftshift(wf))
        self.img += np.abs(ftwf)**2

    def _finalize(self, flux, gsparams):
        """Take accumulated integrated PSF image and turn it into a proper GSObject."""
        del self.aper  # don't need this any more, save some RAM
        self.img = np.fft.fftshift(self.img)
        self.img *= (flux / (self.img.sum() * self.scale**2))
        self.img = galsim.ImageD(self.img.astype(np.float64), scale=self.scale)

        ii = galsim.InterpolatedImage(
            self.img, x_interpolant=self.interpolant, calculate_stepk=True, calculate_maxk=True,
            use_true_center=False, normalization='sb', gsparams=gsparams
        )
        GSObject.__init__(self, ii)


def Atmosphere(r0_500=0.2, screen_size=30.0, time_step=0.03, altitude=0.0, L0=25.0,
               velocity=0.0, direction=0.0*galsim.degrees, alpha=1.0, screen_scale=None, rng=None):
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

    > altitude = [0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
    > r0_500_effective = 0.16  # m
    > weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    > r0_500 = [r0_500_effective * w**(-3./5) for w in weights]
    > velocity = np.random.uniform(0, 20, size=6)  # m/s
    > direction = [np.random.uniform(0, 360)*galsim.degrees for i in xrange(6)]
    > npix = 8192
    > screen_scale = 0.5 * r0_500_effective
    > atm = galsim.Atmosphere(r0_500=r0_500, screen_size=screen_scale*npix, time_step=0.005,
                              altitude=altitude, L0=25.0, velocity=velocity, direction=direction,
                              screen_scale=screen_scale)

    Once the atmosphere is constructed, a 15-sec exposure PSF (using an 8.4 meter aperture and
    default settings) takes about 150 sec to generate on a fast laptop.

    > psf = atm.makePSF(exptime=15.0, diam=8.4, obscuration=0.6)

    Many factors will affect the timing of results, of course, including aperture diameter, gsparams
    settings, pad_factor and oversampling options to makePSF, time_step and exposure time, frozen
    vs. non-frozen atmospheric layers, and so on.

    @param r0_500        Fried parameter setting the amplitude of turbulence; contributes to "size"
                         of the resulting atmospheric PSF.  Specified at wavelength 500 nm, in units
                         of meters.  [Default: 0.2]
    @param screen_size   Physical extent of square phase screen in meters.  This should be large
                         enough to accommodate the desired field-of-view of the telescope as well as
                         the meta-pupil defined by the wind velocity and exposure time.  Note that
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
    @param velocity      Wind speed in meters/second.  [Default: 0.0]
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
    # Listify
    r0_500s, sizes, altitudes, L0s, velocities, directions, alphas, scales = (
        listify(i) for i in (
            r0_500, screen_size, altitude, L0, velocity, direction, alpha, screen_scale))

    # Broadcast
    n_layers = max(len(i) for i in (
        r0_500s, sizes, altitudes, L0s, velocities, directions, alphas, scales))
    if n_layers > 1:
        sizes, altitudes, L0s, velocities, directions, alphas, scales = (
            broadcast(i, n_layers) for i in (
                sizes, altitudes, L0s, velocities, directions, alphas, scales))
    # Broadcast r0_500 separately, since combination of indiv layers' r0s is more complex:
    if len(r0_500s) == 1:
        r0_500s = [n_layers**(3./5) * r0_500s[0]] * n_layers

    if any(len(i) != n_layers for i in (
        r0_500s, sizes, L0s, velocities, directions, alphas, scales)):
        raise ValueError("r0_500, screen_size, L0, velocity, direction, alpha, " +
                         "screen_scale not broadcastable")

    # decompose velocities
    vxs, vys = zip(*[v*d.sincos() for v, d in zip(velocities, directions)])

    layers = [AtmosphericScreen(size, scale, alt, time_step=time_step, r0_500=r0_500, L0=L,
                                vx=vx, vy=vy, alpha=a, rng=rng)
              for r0, size, alt, L, vx, vy, a, scale
              in zip(r0_500s, sizes, altitudes, L0s, vxs, vys, alphas, scales)]
    return PhaseScreenList(layers)
