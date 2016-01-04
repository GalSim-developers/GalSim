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

import copy
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
        self.orig_rng = copy.deepcopy(rng)

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

        self.alpha_mag = alpha_mag
        self.frozen = frozen
        self._initialize_screens()

    def _initialize_screens(self):
        if self.frozen:
            self.layers = [
                FrozenPhaseScreen(
                    self.time_step, self.screen_size, self.screen_scale, alt,
                    r, L, vx0, vy0, self.rng
                )
                for r, L, vx0, vy0, alt
                in zip(self.r0_500, self.L0_inv, self.vx, self.vy, self.altitude)
            ]
        else:
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

    def getPSFs(self, **kwargs):
        kwargs['_eval_now'] = False
        PSFs = []
        for theta_x, theta_y in zip(kwargs.pop('theta_x'), kwargs.pop('theta_y')):
            PSFs.append(AtmosphericPSF(self, theta_x=theta_x, theta_y=theta_y, **kwargs))

        nstep = PSFs[0]._nstep
        atm = PSFs[0].atmosphere
        flux = kwargs.pop('flux', 1.0)
        gsparams = kwargs.pop('gsparams', None)
        for i in xrange(nstep):
            for PSF in PSFs:
                PSF._step()
            atm.advance()
        for PSF in PSFs:
            PSF._finalize(flux, gsparams)
        return PSFs

    def path_difference(self, *args, **kwargs):
        return np.sum(layer.path_difference(*args, **kwargs) for layer in self.layers)

    def reset(self):
        self.rng = copy.deepcopy(self.orig_rng)
        self._initialize_screens()


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

    def advance(self):
        raise NotImplementedError

    def path_difference(self, nx, scale, theta_x=None, theta_y=None):
        raise NotImplementedError


class FrozenPhaseScreen(PhaseScreen):
    def __init__(self, time_step=0.03, screen_size=10.0, screen_scale=0.1, altitude=0.0,
                 r0_500=0.2, L0_inv=1./25.0, vx=0.0, vy=0.0, rng=None):

        super(FrozenPhaseScreen, self).__init__(time_step, screen_size, screen_scale, altitude,
                                                r0_500, L0_inv, vx, vy, rng)
        x0 = y0 = -self.screen_size/2.0
        dx = dy = self.screen_scale
        self.tab2d = galsim.LookupTable2D(x0, y0, dx, dy, self.screen, edge_mode='wrap')
        self.origin = np.r_[0.0, 0.0]

    def advance(self):
        """ For frozen screen, easier to adjust telescope bore-sight location rather than actually
        move the screen (c.f. ARPhaseScreen).
        """
        self.origin += (self.vx*self.time_step, self.vy*self.time_step)

    def path_difference(self, nx, scale, theta_x=0.0*galsim.degrees, theta_y=0.0*galsim.degrees):
        xmin = self.origin[0] + self.altitude*theta_x.tan() - 0.5*scale*(nx-1)
        xmax = xmin + scale*(nx-1)
        ymin = self.origin[1] + self.altitude*theta_y.tan() - 0.5*scale*(nx-1)
        ymax = ymin + scale*(nx-1)

        return self.tab2d.eval_grid(xmin, xmax, nx, ymin, ymax, nx)


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
        """ For AR screen, easy to actually move the screen, rather than the telescope bore-sight.
        """
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
                 pad_factor=1.0, oversampling=1.5,
                 _pupil_size=None, _pupil_scale=None,
                 _bar=None, gsparams=None, _eval_now=True):

        # Store initialization variables
        self.atmosphere = atmosphere
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
            obj = galsim.Kolmogorov(lam=self.lam,
                                    r0=self.atmosphere.r0_500_effective*(self.lam/500.)**(6./5))
            _pupil_scale = (obj.stepK() * self.lam*1e-9 *
                            galsim.radians / self.scale_unit / self.oversampling)
        self._pupil_scale = _pupil_scale

        if _pupil_size is None:
            _pupil_size = self.diam * self.pad_factor
        self._nu = int(np.ceil(_pupil_size/self._pupil_scale))
        self._pupil_size = self._nu * self._pupil_scale

        self.scale = 1e-9*self.lam/self._pupil_size * galsim.radians / self.scale_unit
        self.img = np.zeros((self._nu, self._nu), dtype=np.float64)

        self.aper = self._generate_pupil()

        self._nstep = int(np.ceil(self.exptime/self.atmosphere.time_step))
        # Generate at least one time sample
        if self._nstep == 0:
            self._nstep = 1

        # When generating many PSFs from the same atmosphere, it's more efficient to loop over
        # the PSFs and then increment the atmosphere than to fully compute single PSFs one at a
        # time.  The _eval_now flag enables (in a hidden way) the above computation strategy.
        if _eval_now:
            for i in xrange(self._nstep):
                self._step()
                self.atmosphere.advance()
                if _bar is not None:
                    _bar.update()

            self._finalize(flux, gsparams)

    def _generate_pupil(self):
        aper = np.ones((self._nu, self._nu), dtype=np.float64)
        if self.diam is not None:
            radius = 0.5*self.diam
            u = np.fft.fftshift(np.fft.fftfreq(self._nu, 1./self._pupil_size))
            u, v = np.meshgrid(u, u)
            rsqr = u**2 + v**2
            aper = rsqr <= radius**2
            if self.obscuration is not None:
                aper[rsqr <= (radius*self.obscuration)**2] = 0.0
        return aper

    def _step(self):
        path_difference = self.atmosphere.path_difference(self._nu, self._pupil_scale,
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
