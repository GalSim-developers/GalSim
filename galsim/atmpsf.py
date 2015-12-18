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
Module for generating atmospheric PSFs using an autoregressive phase screen generator.

Relevant SPIE paper:
"Remembrance of phases past: An autoregressive method for generating realistic atmospheres in
simulations"
Srikar Srinath, Univ. of California, Santa Cruz;
Lisa A. Poyneer, Lawrence Livermore National Lab.;
Alexander R. Rudy, UCSC; S. Mark Ammons, LLNL
Published in Proceedings Volume 9148: Adaptive Optics Systems IV
September 2014

"""

import numpy as np
import galsim
import utilities
from galsim import GSObject


class Atmosphere(object):
    def __init__(self, time_step=0.03, screen_size=10.0, screen_scale=0.1, altitude=0.0,
                 r0_500=0.2, L0=25.0, velocity=0.0, direction=0.0*galsim.degrees, alpha_mag=0.997,
                 rng=None, frozen=True):
        self.time_step = time_step
        self.npix = int(np.ceil(screen_size/screen_scale))
        self.screen_scale = screen_scale
        # redefine screen_size to make sure it's consistent with npix and screen_scale
        self.screen_size = self.screen_scale * self.npix
        if rng is None:
            rng = galsim.BaseDeviate()
        self.rng = rng

        # Listify
        altitude, r0_500, L0, velocity, direction, alpha_mag = map(
            lambda i: [i] if not hasattr(i, '__iter__') else i,
            (altitude, r0_500, L0, velocity, direction, alpha_mag)
        )
        self.altitude = altitude
        # Broadcast
        self.n_layers = len(self.altitude) if hasattr(self.altitude, '__iter__') else 1
        if self.n_layers > 1:
            L0, velocity, direction, alpha_mag = map(
                lambda i: [i[0]]*self.n_layers if len(i) == 1 else i,
                (L0, velocity, direction, alpha_mag)
            )
        # Broadcast r0_500 separately, since combination of indiv layers' r0s is more complex:
        if len(r0_500) == 1:
            r0_500 = [self.n_layers**(3./5) * r0_500[0]] * self.n_layers
        if any(len(i) != self.n_layers for i in (r0_500, L0, velocity, direction, alpha_mag)):
            raise ValueError("r0_500, L0, velocity, direction, alpha_mag not broadcastable")

        self.r0_500_effective = (np.sum(r**(-5./3) for r in r0_500))**(-3./5)
        self.r0_500 = r0_500

        # Invert L0, with `L0 is None` interpretted as L0 = infinity => L0_inv = 0.0
        self.L0_inv = [1./L00 if L00 is not None else 0.0 for L00 in L0]

        # decompose velocities
        self.vx, self.vy = zip(*[v*d.sincos() for v, d in zip(velocity, direction)])

        if frozen:
            self.layers = [
                FrozenPhaseScreen(
                    self.time_step, self.screen_size, self.screen_scale, alt,
                    r, L, vx0, vy0, self.rng
                )
                for r, L, vx0, vy0, alt
                in zip(self.r0_500, self.L0_inv, self.vx, self.vy, self.altitude)
            ]
        else:
            self.alpha_mag = alpha_mag
            self.layers = [
                ARPhaseScreen(
                    self.time_step, self.screen_size, self.screen_scale, alt,
                    r, L, vx0, vy0, amag, self.rng
                )
                for r, L, vx0, vy0, amag, alt
                in zip(self.r0_500, self.L0_inv, self.vx, self.vy, self.alpha_mag, self.altitude)
            ]

    def advance(self):
        for layer in self.layers:
            layer.advance()

    def getPSF(self, **kwargs):
        return AtmosphericPSF(self, **kwargs)

    def path_difference(self, *args, **kwargs):
        return np.sum(layer.path_difference(*args, **kwargs) for layer in self.layers)


class PhaseScreen(object):
    def __init__(self, time_step=0.03, screen_size=10.0, screen_scale=0.1, altitude=0.0,
                 r0_500=0.2, L0_inv=1./25.0, vx=0.0, vy=0.0, rng=None, **kwargs):
        if rng is None:
            rng = galsim.BaseDeviate()
        self.rng = rng

        self.npix = int(np.ceil(screen_size/screen_scale))
        self.screen_scale = screen_scale
        self.screen_size = self.screen_scale * self.npix
        self.altitude = altitude
        self.time_step = time_step
        self.r0_500 = r0_500
        self.L0_inv = L0_inv
        self.vx = vx
        self.vy = vy

        fx = np.fft.fftfreq(self.npix, self.screen_scale)
        fx, fy = np.meshgrid(fx, fx)

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


class FrozenPhaseScreen(PhaseScreen):
    def __init__(self, time_step=0.03, screen_size=10.0, screen_scale=0.1, altitude=0.0,
                 r0_500=0.2, L0_inv=1./25.0, vx=0.0, vy=0.0, rng=None):

        super(FrozenPhaseScreen, self).__init__(time_step, screen_size, screen_scale, altitude,
                                                r0_500, L0_inv, vx, vy, rng)

        self.img = galsim.Image(np.ascontiguousarray(self.screen), scale=self.screen_scale)
        self.ii = galsim.InterpolatedImage(self.img, calculate_stepk=False, calculate_maxk=False,
                                           normalization='sb', x_interpolant='Linear')

    def advance(self):
        self.ii = self.ii.shift(self.vx*self.time_step, self.vy*self.time_step)

    def path_difference(self, nx, scale, theta_x=0.0*galsim.degrees, theta_y=0.0*galsim.degrees):
        ii_t = self.ii.shift(self.altitude*theta_x.tan(), self.altitude*theta_y.tan())
        return ii_t.drawImage(nx=nx, ny=nx, scale=scale, method='sb').array


class ARPhaseScreen(PhaseScreen):
    def __init__(self, time_step=0.03, screen_size=10.0, screen_scale=0.1, altitude=0.0,
                 r0_500=0.2, L0_inv=1./25.0, vx=0.0, vy=0.0, alpha_mag=0.999, rng=None):

        super(ARPhaseScreen, self).__init__(time_step, screen_size, screen_scale, altitude,
                                            r0_500, L0_inv, vx, vy, rng)
        self.alpha_mag = alpha_mag
        fx = np.fft.fftfreq(self.npix, self.screen_scale)
        fx, fy = np.meshgrid(fx, fx)
        alpha_phase = -(fx*vx + fy*vy) * time_step
        self.alpha = alpha_mag * np.exp(2j*np.pi*alpha_phase)

        self.noise_frac = np.sqrt(1.0 - np.abs(self.alpha**2)[0, 0])

        # Ignore boiling for *really* large alpha, but don't let amplitudes decay
        if self.noise_frac < 1.e-10:
            self.alpha /= np.abs(self.alpha)

    def advance(self):
        if self.noise_frac < 1.e-10:
            # Frozen flow
            self._phaseFT = self.alpha*self._phaseFT
        else:
            # Boiling
            self._phaseFT = self.alpha*self._phaseFT + self._noiseFT()*self.noise_frac
        self.screen = np.fft.ifft2(self._phaseFT).real

    def path_difference(self, nx, scale, theta_x=0.0*galsim.degrees, theta_y=0.0*galsim.degrees):
        img = galsim.Image(np.ascontiguousarray(self.screen), scale=self.screen_scale)
        ii = galsim.InterpolatedImage(img, calculate_stepk=False, calculate_maxk=False,
                                      normalization='sb')
        ii = ii.shift(self.altitude * theta_x.tan(), self.altitude * theta_y.tan())
        return ii.drawImage(nx=nx, ny=nx, scale=scale, method='sb').array


class AtmosphericPSF(GSObject):
    def __init__(self, atmosphere, lam=500.0, exptime=15.0, flux=1.0,
                 theta_x=0.0*galsim.degrees, theta_y=0.0*galsim.degrees,
                 scale_unit=galsim.arcsec, interpolant=None,
                 diam=10.0, obscuration=None,
                 pad_factor=1.0, pupil_size=None,
                 oversample_factor=1.0, pupil_scale=None,
                 _bar=None, verbose=False):

        if pupil_scale is None:
            obj = galsim.Kolmogorov(lam=lam, r0=atmosphere.r0_500_effective*(lam/500.)**(6./5))
            pupil_scale = obj.stepK() * lam*1e-9 * galsim.radians / scale_unit / oversample_factor
        if pupil_size is None:
            pupil_size = diam * pad_factor

        n_u = int(np.ceil(pupil_size/pupil_scale))
        pupil_size = n_u * pupil_scale
        self.scale = 1e-9*lam/pupil_size * galsim.radians / scale_unit
        img = np.zeros((n_u, n_u), dtype=np.float64)

        if verbose:
            print "pupil_size: ", pupil_size
            print "pupil_scale: ", pupil_scale
            print "n_u: ", n_u

        aper = np.ones_like(img)
        if diam is not None:
            u = np.fft.fftshift(np.fft.fftfreq(n_u, 1./pupil_size))
            u, v = np.meshgrid(u, u)
            r = np.hypot(u, v)
            aper = r <= 0.5*diam
            if obscuration is not None:
                aper[r <= 0.5*diam*obscuration] = 0.0

        nstep = int(np.ceil(exptime/atmosphere.time_step))
        if nstep == 0:
            nstep = 1

        for i in xrange(nstep):
            path_difference = atmosphere.path_difference(n_u, pupil_scale, theta_x, theta_y)
            wf = aper * np.exp(2j * np.pi * path_difference / lam)
            ftwf = np.fft.ifft2(np.fft.ifftshift(wf))
            img += np.abs(ftwf)**2
            atmosphere.advance()
            if _bar is not None:
                _bar.update()

        img = np.fft.fftshift(img)
        img *= (flux / (img.sum() * self.scale**2))
        img = galsim.ImageD(img.astype(np.float64), scale=self.scale)

        ii = galsim.InterpolatedImage(
            img, x_interpolant=interpolant, calculate_stepk=True, calculate_maxk=True,
            use_true_center=False, normalization='sb'
        )
        GSObject.__init__(self, ii)


# class AtmosphericPSF(GSObject):
#     """ Create an atmosphere PSF by summing over a phase cube in time steps.
#     @param lam              Lambda (wavelength) in units of nanometers.  Must be supplied with `r0`,
#                             and in this case, image scales (`scale`) should be specified in units of
#                             `scale_unit`.
#     @param r0_500           Like r0, but specified for lambda=500nm instead of at provided lambda.
#     @param L0               Outer scale in meters of von Karman spectrum.  [Default: None (= inf)]
#     @param weights          Relative weights for each turbulent layer in terms of the layers'
#                             refractive index fluctuations C_n^2.  The Fried parameter for each layer
#                             is then set via r0_i = r0_effective * weight_i^(-3/5).
#     @param alpha_mag        Magnitude of autoregressive parameter.  (1-alpha) is the fraction of the
#                             phase from the prior time step that is "forgotten" and replaced by
#                             Gaussian noise.  [Default: 0.999]
#     @param exptime          Exposure time in seconds. [Default: 0]
#     @param time_step        Interval between PSF images in seconds. [Default: 0.03]
#     @param velocity         Velocity magnitude of each phase screen layer in meters / second.
#                             [Default: 0]
#     @param direction        CCW relative to +x as galsim.Angle.  [Default: 0*galsim.degrees]
#     @param phase_generator  AtmosphericPhaseGenerator object.  [Default: None]
#     @param interpolant      Either an Interpolant instance or a string indicating which interpolant
#                             should be used.  Options are 'nearest', 'sinc', 'linear', 'cubic',
#                             'quintic', or 'lanczosN' where N should be the integer order to use.
#                             [Default: galsim.Quintic()]
#     @param flux             Total flux of the profile. [Default: 1.]
#     @param scale_unit       Units used to define the diffraction limit and draw images, if the user
#                             has supplied a separate value for `lam` and `r0`.  Should be either a
#                             galsim.AngleUnit, or a string that can be used to construct one (e.g.,
#                             'arcsec', 'radians', etc.).
#                             [Default: galsim.arcsec]
#     @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
#                             details. [Default: None]
#     """
#     def __init__(self, lam=None, r0_500=None, L0=None, weights=None, alpha_mag=None, exptime=0.0,
#                  time_step=0.03, velocity=None, direction=None, phase_generator=None,
#                  interpolant=None, oversampling=1.5, flux=1.0, scale_unit=galsim.arcsec,
#                  gsparams=None, diam=None, obscuration=None, screen_size=None, rng=None,
#                  _bar=None):
#         import itertools
#
#         nstep = int(np.ceil(exptime/time_step))
#         if nstep == 0:
#             nstep = 1
#
#         if phase_generator is not None:
#             if any(item is not None for item in (r0_500, L0, weights, alpha_mag)):
#                 raise ValueError("Cannot specify r0_500, L0, weights, or alpha_mag"
#                                  " when specifying phase_generator")
#         else:
#             if alpha_mag is None:
#                 alpha_mag = 0.999
#             if direction is None:
#                 direction = 0.0 * galsim.degrees
#             if velocity is None:
#                 velocity = 0.0
#             if weights is not None:
#                 if hasattr(r0_500, '__iter__'):
#                     raise ValueError("Cannot specify both weights and list of r0_500s.")
#                 r0_500 = [r0_500 * w**(-3./5) for w in weights]
#
#             # Listify
#             r0_500, velocity, direction, alpha_mag = map(
#                 lambda i: [i] if not hasattr(i, '__iter__') else i,
#                 (r0_500, velocity, direction, alpha_mag)
#             )
#
#             if lam is None:
#                 lam = 500.0  # arbitrarily set wavelength = 500nm
#             r0_500_effective = (sum(r**(-5./3) for r in r0_500)**(-3./5))
#
#             # Sampling the phase screen is roughly analogous to sampling the PSF in Fourier space.
#             # We can use a Kolmogorov profile to get a rough estimate of what stepK is needed to
#             # avoid aliasing.
#             kolm = galsim.Kolmogorov(lam=lam, r0=r0_500_effective*(lam/500.0)**(6./5))
#             screen_scale = kolm.stepK()/(2*np.pi)  # 2pi b/c np and GalSim FFT conventions differ.
#             # an arbitrary additional factor of 4 to account for the fact that a stochastic
#             # atmospheric PSF can have significant fluctuations at relatively large radii.
#             screen_scale /= 4.0
#             # We'll hard code screen_size = 10 meters since that aperture covers all planned
#             # ground-based optical weak lensing experiments.
#             if screen_size is None:
#                 screen_size = 10.0
#             phase_generator = AtmosphericPhaseGenerator(
#                 time_step=time_step, screen_size=screen_size, screen_scale=screen_scale,
#                 r0_500=r0_500, L0=L0, alpha_mag=alpha_mag, velocity=velocity, direction=direction,
#                 rng=rng)
#         self.phase_generator = phase_generator
#
#         scale = 1e-9*lam/self.phase_generator.screen_size * galsim.radians / scale_unit
#         # Generate PSFs for each time step
#         nx, ny = phase_generator.screens[0].shape
#         im_grid = np.zeros((nx, ny), dtype=np.float64)
#         aper = np.ones_like(im_grid)
#         if diam is not None:
#             x = np.fft.fftfreq(nx, 1./phase_generator.screen_size)
#             x, y = np.meshgrid(x, x)
#             r = np.hypot(x, y)
#             if obscuration is not None:
#                 aper = (r <= 0.5*diam) & (r >= 0.5*diam*obscuration)
#             else:
#                 aper = r <= 0.5*diam
#
#         for i, screens in itertools.izip(xrange(nstep), phase_generator):
#             if _bar is not None:  # Hidden ProgressBar option
#                 _bar.update()
#             # The wavefront to use is exp(2 pi i screen)
#             wf = np.exp(2j * np.pi * np.sum(screens, axis=0) / lam) * aper
#             # Calculate the image array via FFT.
#             # Copied from galsim.optics.psf method (hacky)
#             ftwf = np.fft.ifft2(np.fft.ifftshift(wf))
#             im = np.abs(ftwf)**2
#             # Add this PSF instance to stack to get the finite-exposure PSF
#             im_grid += im
#
#         im_grid = np.fft.fftshift(im_grid)
#         im_grid *= (flux / (im_grid.sum() * scale**2))
#         out_im = galsim.InterpolatedImage(
#             galsim.Image(im_grid.astype(np.float64), scale=scale),
#             x_interpolant=interpolant, calculate_stepk=True, calculate_maxk=True,
#             use_true_center=False, normalization='sb')
#         GSObject.__init__(self, out_im)
