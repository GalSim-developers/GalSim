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
"""@file atmpsf.py
"""

import copy
import numpy as np
import galsim
import utilities
from galsim import GSObject


def listify(arg):
    """Turns argument into a list if not already iterable."""
    return [arg] if not hasattr(arg, '__iter__') else arg


def broadcast(arg, n):
    """Turn length-1 iterable into length-n list."""
    return [arg[0]]*n if len(arg) == 1 else arg


def generate_pupil(nu, pupil_size, diam=None, obscuration=None):
    """ Generate a pupil transmission array (0's and 1's) for a circular aperture and potentially a
    central circular obscuration.

    @param nu          Number of pixels in square output array.
    @param pupil_size  Physical size of pupil array in meters.
    @param diam        Diameter of aperture in meters.
    @param obscuration Fractional linear obscuration.
    @returns array of 0's and 1's indicating pupil transmission function.
    """
    aper = np.ones((nu, nu), dtype=np.float64)
    if diam is not None:
        radius = 0.5*diam
        u = np.fft.fftshift(np.fft.fftfreq(nu, 1./pupil_size))
        u, v = np.meshgrid(u, u)
        rsqr = u**2 + v**2
        aper = rsqr <= radius**2
        if obscuration is not None:
            aper *= rsqr >= (radius*obscuration)**2
    return aper


class PhaseScreen(object):
    """ Abstract base class for a phase screen to use in generating a PSF using Fourier optics.
    PhaseScreen subclasses need to implement the methods below.
    """
    def __init__(self, screen_size, screen_scale, altitude):
        self.npix = int(np.ceil(screen_size/screen_scale))
        self.screen_scale = screen_scale
        self.screen_size = self.screen_scale * self.npix
        self.altitude = altitude

    def advance(self):
        # Default is a no-op, which would be appropriate for an optics phase screen, for example.
        # For an atmsopheric phase screen, this should update the atmospheric layer to account for
        # wind, boiling, etc.
        pass

    def advance_by(self, dt):
        return dt

    def reset(self):
        # For time-dependent screens, should reset state to t=0.
        # For time-independent screens, this is a no-op.
        pass

    def path_difference(self, nx, scale, theta_x=None, theta_y=None):
        # This should return an nx-by-nx pixel array with scale `scale` (in meters) representing the
        # effective difference in path length (nanometers) for rays originating from different
        # points in the pupil plane.  The `theta_x` and `theta_y` params indicate the position on
        # the focal plane, or equivalently the position on the sky from which the rays originate.
        raise NotImplementedError

    def pupil_scale(self, lam, scale_unit=galsim.arcsec):
        raise NotImplementedError


class AtmosphericScreen(PhaseScreen):
    """ A phase screen representing an atmospheric layer.
    """
    def __init__(self, screen_size, screen_scale=None, altitude=0.0, time_step=0.03,
                 r0_500=0.2, L0_inv=1./25.0, vx=0.0, vy=0.0, rng=None):

        if screen_scale is None:
            screen_scale = 0.5 * r0_500
        super(AtmosphericScreen, self).__init__(screen_size, screen_scale, altitude)

        if rng is None:
            rng = galsim.BaseDeviate()
        self.rng = rng
        self.orig_rng = copy.deepcopy(rng)

        self.time_step = time_step
        self.altitude = altitude
        self.r0_500 = r0_500
        self.L0_inv = L0_inv
        self.vx = vx
        self.vy = vy

    def advance_by(self, dt):
        _nstep = int(np.ceil(dt/self.time_step))
        for i in xrange(_nstep):
            self.advance()
        return _nstep*self.time_step  # return the time *actually* advanced

    # Collect a few methods common to both FrozenAtmosphericScreen and ARAtmosphericScreen
    def _freq(self):
        fx = np.fft.fftfreq(self.npix, self.screen_scale)
        return np.meshgrid(fx, fx)

    def _init_screen(self, fx, fy):
        self.psi = (1./self.screen_size*np.sqrt(0.00058)*(self.r0_500**(-5.0/6.0)) *
                    (fx*fx + fy*fy + self.L0_inv*self.L0_inv)**(-11.0/12.0) *
                    self.npix * np.sqrt(np.sqrt(2.0))) * 500.0
        self.psi[0, 0] = 0.0
        self._phaseFT = self._noiseFT()
        self.screen = np.fft.ifft2(self._phaseFT).real

    def _noiseFT(self):
        gd = galsim.GaussianDeviate(self.rng)
        noise = utilities.rand_arr(self.psi.shape, gd)
        return np.fft.fft2(noise)*self.psi

    # Both types of atmospheric screens determine their pupil scales (essentially stepK()) from the
    # Kolmogorov profile with the same Fried parameter r0.
    def pupil_scale(self, lam, scale_unit=galsim.arcsec):
        obj = galsim.Kolmogorov(lam=lam, r0=self.r0_500 * (lam/500.0)**(6./5))
        return obj.stepK() * lam*1.e-9 * galsim.radians / scale_unit


class FrozenAtmosphericScreen(AtmosphericScreen):
    """ An atmospheric phase screen that can drift in the wind, but otherwise does not evolve with
    time.  The phases are drawn from a von Karman power spectrum, which is defined by a Fried
    parameter that effectively sets the amplitude of the turbulence, and an outer scale that sets
    scale beyond which the turbulence power goes (smoothly) to zero.

    @param screen_size    How large in meters should the screen be?  This should be large enough to
                          accommodate the desired field-of-view of the telescope as well as the
                          meta-pupil defined by the wind velocity and exposure time.  Note that the
                          screen will have periodic boundary conditions, so it's technically
                          possible to use a smaller sized screen than technically necessary, though
                          this may introduce artifacts into PSFs or PSF correlations functions.
    @param screen_scale   How finely should the phase screen be sampled in meters?  A fraction of
                          the Fried parameter is usually sufficiently small, but users should test
                          the effects of this parameter to ensure robust results.
                          [Default: 0.5*r0_500]
    @param altitude       The altitude of the screen with respect to the telescope in km.
                          [Default: 0.0]
    @param time_step      Time interval in seconds over which atmosphere is propagated before the
                          PSF image is incrementally integrated.  [Default: 0.03]
    @param r0_500         The Fried parameter in meters *at wavelength 500 nm*.  [Default: 0.2]
    @param L0_inv         Inverse outer scale in 1/meters.  [Default: 1./25].
    @param vx             Wind velocity in x-direction in meters/second. [Default: 0]
    @param vy             Wind velocity in y-direction in meters/second. [Default: 0]
    @param rng            Random number generator (galsim.BaseDeviate or subclass) used to
                          initialize phase screen.  Default of None will create a BaseDeviate with
                          random seed from the system entropy or clock time.
    """
    def __init__(self, screen_size, screen_scale, altitude=0.0, time_step=0.03,
                 r0_500=0.2, L0_inv=1./25.0, vx=0.0, vy=0.0, rng=None):
        super(FrozenAtmosphericScreen, self).__init__(screen_size, screen_scale, altitude,
                                                      time_step, r0_500, L0_inv,
                                                      vx, vy, rng)

        # Note that _init_screen is here instead of in superclass since ARAtmosphericScreen reuses
        # fx, fy, but we don't want to store these in the class or recompute them.  So instead, we
        # just provide the `_freq` method in the superclass that each AtmosphericScreen subclass can
        # use.
        fx, fy = self._freq()
        self._init_screen(fx, fy)
        # Use a LookupTable2D to interpolate/extrapolate with periodic boundary conditions.
        x0 = y0 = -self.screen_size/2.0
        dx = dy = self.screen_scale
        self.tab2d = galsim.LookupTable2D(x0, y0, dx, dy, self.screen, edge_mode='wrap')
        # To handle wind, we will interpolate the LookupTable2D with an offset origin.
        self.origin = np.r_[0.0, 0.0]

    def __str__(self):
        return "galsim.FrozenAtmosphericScreen(altitude=%s)" % self.altitude

    def __repr__(self):
        outstr = ("galsim.FrozenAtmosphericScreen(%r, %r, altitude=%r, time_step=%r, " +
                  "r0_500=%r, L0_inv=%r, vx=%r, vy=%r, rng=%r)")
        return outstr % (self.screen_size, self.screen_scale, self.altitude, self.time_step,
                         self.r0_500, self.L0_inv, self.vx, self.vy, self.rng)

    def __eq__(self, other):
        return (self.screen_size == other.screen_size and
                self.screen_scale == other.screen_scale and
                self.altitude == other.altitude and
                self.r0_500 == other.r0_500 and
                self.L0_inv == other.L0_inv and
                self.vx == other.vx and
                self.vy == other.vy and
                self.tab2d == other.tab2d and
                np.array_equal(self.origin, other.origin))

    def __ne__(self, other):
        return not self == other

    def advance(self):
        # If wind blows right, then origin moves left, so use minus sign.
        self.origin -= (self.vx*self.time_step, self.vy*self.time_step)

    def advance_by(self, dt):
        self.origin -= (self.vx*dt, self.vy*dt)
        return dt

    def path_difference(self, nx, scale, theta_x=0.0*galsim.degrees, theta_y=0.0*galsim.degrees):
        xmin = self.origin[0] + 1000*self.altitude*theta_x.tan() - 0.5*scale*(nx-1)
        xmax = xmin + scale*(nx-1)
        ymin = self.origin[1] + 1000*self.altitude*theta_y.tan() - 0.5*scale*(nx-1)
        ymax = ymin + scale*(nx-1)

        return self.tab2d.eval_grid(xmin, xmax, nx, ymin, ymax, nx)

    def reset(self):
        self.rng = copy.deepcopy(self.orig_rng)
        self.origin = np.r_[0.0, 0.0]


class ARAtmosphericScreen(AtmosphericScreen):
    """Auto-regressive atmospheric screen.

    Relevant SPIE paper:
    "Remembrance of phases past: An autoregressive method for generating realistic atmospheres in
    simulations"
    Srikar Srinath, Univ. of California, Santa Cruz;
    Lisa A. Poyneer, Lawrence Livermore National Lab.;
    Alexander R. Rudy, UCSC; S. Mark Ammons, LLNL
    Published in Proceedings Volume 9148: Adaptive Optics Systems IV
    September 2014
    """
    def __init__(self, screen_size, screen_scale, altitude=0.0, time_step=0.03,
                 r0_500=0.2, L0_inv=1./25.0, vx=0.0, vy=0.0, alpha_mag=0.999, rng=None):

        super(ARAtmosphericScreen, self).__init__(screen_size, screen_scale, altitude,
                                                  time_step, r0_500, L0_inv,
                                                  vx, vy, rng)

        self.alpha_mag = alpha_mag
        fx, fy = self._freq()
        self._init_screen(fx, fy)

        # Work wind directly into "boiling" phase updating.
        alpha_phase = -(fx*vx + fy*vy) * time_step
        self.alpha = alpha_mag * np.exp(2j*np.pi*alpha_phase)

        self.noise_frac = np.sqrt(1.0 - np.abs(self.alpha**2)[0, 0])

        # Ignore boiling for *really* large alpha, but don't let amplitudes decay
        if self.noise_frac < 1.e-12:
            self.alpha /= np.abs(self.alpha)

    def __str__(self):
        return "galsim.ARAtmosphericScreen(altitude=%s)" % self.altitude

    def __repr__(self):
        outstr = ("galsim.ARAtmosphericScreen(%r, %r, altitude=%r, time_step=%r, " +
                  "r0_500=%r, L0_inv=%r, vx=%r, vy=%r, alpha_mag=%r, rng=%r)")
        return outstr % (self.screen_size, self.screen_scale, self.altitude, self.time_step,
                         self.r0_500, self.L0_inv, self.vx, self.vy, self.alpha_mag, self.rng)

    def __eq__(self, other):
        return (self.screen_size == other.screen_size and
                self.screen_scale == other.screen_scale and
                self.altitude == other.altitude and
                self.r0_500 == other.r0_500 and
                self.L0_inv == other.L0_inv and
                self.vx == other.vx and
                self.vy == other.vy and
                self.alpha_mag == other.alpha_mag and
                self.rng == other.rng)

    def __ne__(self, other):
        return not self == other

    def advance(self):
        # For ARAtmosphericScreen, since we use Fourier methods to update the screen each step
        # anyway, it's easy to simultaneously move the screen with the wind.
        if self.noise_frac < 1.e-12:
            # Frozen flow (but slower than FrozenAtmosphericScreen, since using Fourier methods to
            # move the screen in the wind).
            self._phaseFT = self.alpha*self._phaseFT
        else:
            # Boiling, do a fractional random phase update.
            self._phaseFT = self.alpha*self._phaseFT + self._noiseFT()*self.noise_frac
        self.screen = np.fft.ifft2(self._phaseFT).real

    def path_difference(self, nx, scale, theta_x=0.0*galsim.degrees, theta_y=0.0*galsim.degrees):
        img = galsim.Image(np.ascontiguousarray(self.screen), scale=self.screen_scale)
        ii = galsim.InterpolatedImage(img, calculate_stepk=False, calculate_maxk=False,
                                      normalization='sb')
        ii = ii.shift(1000*self.altitude * theta_x.tan(), 1000*self.altitude * theta_y.tan())
        return ii.drawImage(nx=nx, ny=nx, scale=scale, method='sb').array

    def reset(self):
        self.rng = copy.deepcopy(self.orig_rng)
        fx, fy = self._freq()
        self._init_screen(fx, fy)


class PhaseScreenList(object):
    """ List of phase screens that can be turned into a PSF.  Screens can be either atmospheric
    layers or optical phase screens.  Generally, one would assemble a PhaseScreenList object using
    the function `Atmosphere` or `OpticalPSF`.
    """
    def __init__(self, layers):
        self._layers = layers
        self._update_attrs()  # for now, just updating self.time_step

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, position):
        return self._layers[position]

    def __setitem__(self, position, layer):
        self._layers[position] = layer
        self._update_attrs()

    def __delitem__(self, position):
        del self._layers[position]
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
        time_step = {l.time_step for l in self if l.time_step is not None}
        if len(time_step) == 0:
            self.time_step = None
        elif len(time_step) == 1:
            self.time_step = time_step.pop()
        else:
            raise ValueError("Layer time steps must all be identical or None")

    def advance(self):
        for layer in self:
            layer.advance()

    def getPSF(self, *args, **kwargs):
        return PhaseScreenPSF(self, *args, **kwargs)

    def getPSFs(self, **kwargs):
        kwargs['_eval_now'] = False
        PSFs = []
        for theta_x, theta_y in zip(kwargs.pop('theta_x'), kwargs.pop('theta_y')):
            PSFs.append(PhaseScreenPSF(self, theta_x=theta_x, theta_y=theta_y, **kwargs))

        flux = kwargs.pop('flux', 1.0)
        gsparams = kwargs.pop('gsparams', None)
        _nstep = PSFs[0]._nstep
        for i in xrange(_nstep):
            for PSF in PSFs:
                PSF._step()
            self.advance()

        for PSF in PSFs:
            PSF._finalize(flux, gsparams)
        return PSFs

    def path_difference(self, *args, **kwargs):
        return sum(layer.path_difference(*args, **kwargs) for layer in self)

    def reset(self):
        for layer in self:
            layer.reset()


class PhaseScreenPSF(GSObject):
    def __init__(self, screen_list, lam=500., exptime=0.0, flux=1.0,
                 theta_x=0.0*galsim.arcmin, theta_y=0.0*galsim.arcmin,
                 scale_unit=galsim.arcsec, interpolant=None,
                 diam=8.4, obscuration=0.6,
                 pad_factor=1.0, oversampling=1.5,
                 _pupil_size=None, _pupil_scale=None,
                 gsparams=None, _eval_now=True, _bar=None):

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
            # We're not actually doing convolution here, and, in fact, the right relation for
            # Kolmogorov screens uses exponents -5./3 and -3./5, which is just slightly different.
            # Since most of the layers in a PhaseScreenList are will likely be Kolmogorov screens,
            # we'll use that relation.
            _pupil_scale = (sum(layer.pupil_scale(lam)**(-5./3) for layer in screen_list))**(-3./5)
            _pupil_scale /= oversampling
        self._pupil_scale = _pupil_scale
        if _pupil_size is None:
            _pupil_size = self.diam * self.pad_factor
        self._nu = int(np.ceil(_pupil_size/self._pupil_scale))
        self._pupil_size = self._nu * self._pupil_scale

        self.scale = 1e-9*self.lam/self._pupil_size * galsim.radians / self.scale_unit

        self.aper = generate_pupil(self._nu, self._pupil_size, self.diam, self.obscuration)
        self.img = np.zeros_like(self.aper, dtype=np.float64)

        self._nstep = int(np.ceil(self.exptime/self.screen_list.time_step))
        # Generate at least one time sample
        if self._nstep == 0:
            self._nstep = 1

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
        return (self.screen_list == other.screen_list and
                self.lam == other.lam and
                self.exptime == other.exptime and
                self.theta_x == other.theta_x and
                self.theta_y == other.theta_y and
                self.scale_unit == other.scale_unit and
                self.interpolant == other.interpolant and
                self.diam == other.diam and
                self.obscuration == other.obscuration and
                self._pupil_scale == other._pupil_scale and
                self._pupil_size == other.pupil_size)

    def __ne__(self, other):
        return not self == other

    def _step(self):
        path_difference = self.screen_list.path_difference(self._nu, self._pupil_scale,
                                                           self.theta_x, self.theta_y)
        wf = self.aper * np.exp(2j * np.pi * path_difference / self.lam)
        ftwf = np.fft.ifft2(np.fft.ifftshift(wf))
        self.img += np.abs(ftwf)**2

    def _finalize(self, flux, gsparams):
        del self.aper  # save some RAM
        self.img = np.fft.fftshift(self.img)
        self.img *= (flux / (self.img.sum() * self.scale**2))
        self.img = galsim.ImageD(self.img.astype(np.float64), scale=self.scale)

        ii = galsim.InterpolatedImage(
            self.img, x_interpolant=self.interpolant, calculate_stepk=True, calculate_maxk=True,
            use_true_center=False, normalization='sb', gsparams=gsparams
        )
        GSObject.__init__(self, ii)


def Atmosphere(time_step=0.03, screen_size=10.0, screen_scale=0.1, altitude=0.0,
               r0_500=0.2, L0=25.0, velocity=0.0, direction=0.0*galsim.degrees, alpha_mag=0.997,
               rng=None, frozen=True):

    if rng is None:
        rng = galsim.BaseDeviate()

    # Listify
    altitudes, r0_500s, L0s, velocities, directions, alpha_mags = (
        listify(i) for i in (altitude, r0_500, L0, velocity, direction, alpha_mag)
    )

    # Broadcast
    n_layers = max(len(i) for i in (altitudes, r0_500s, L0s, velocities, directions, alpha_mags))
    if n_layers > 1:
        altitudes, L0s, velocities, directions, alpha_mags = (
            broadcast(i, n_layers) for i in (altitudes, L0s, velocities, directions, alpha_mags)
        )
    # Broadcast r0_500 separately, since combination of indiv layers' r0s is more complex:
    if len(r0_500s) == 1:
        r0_500s = [n_layers**(3./5) * r0_500s[0]] * n_layers
    if any(len(i) != n_layers for i in (r0_500s, L0s, velocities, directions, alpha_mags)):
        raise ValueError("r0_500, L0, velocity, direction, alpha_mag not broadcastable")
    # r0_500_effective = (sum(r**(-5./3) for r in r0_500s))**(-3./5)

    # Invert L0, with `L0 is None` interpretted as L0 = infinity => L0_inv = 0.0
    L0_invs = [1./L if L is not None else 0.0 for L in L0s]

    # decompose velocities
    vxs, vys = zip(*[v*d.sincos() for v, d in zip(velocities, directions)])

    if frozen:
        layers = [
            FrozenAtmosphericScreen(
                screen_size, screen_scale, alt, time_step=time_step,
                r0_500=r0, L0_inv=L0_inv, vx=vx, vy=vy, rng=rng)
            for alt, r0, L0_inv, vx, vy in zip(altitudes, r0_500s, L0_invs, vxs, vys)]
    else:
        layers = [
            ARAtmosphericScreen(
                screen_size, screen_scale, alt, time_step=time_step,
                r0_500=r0, L0_inv=L0_inv, vx=vx, vy=vy, alpha_mag=amag, rng=rng)
            for alt, r0, L0_inv, vx, vy, amag
            in zip(altitudes, r0_500s, L0_invs, vxs, vys, alpha_mags)]

    return PhaseScreenList(layers)
