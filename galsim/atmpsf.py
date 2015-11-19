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


class AtmosphericPhaseGenerator(object):
    """ Use a lag-1 autoregressive model to propagate a set of 2D turbulent phase screens with von
    Karman spectra.  Successive iterations yield screens propagated with constant velocity for each
    layer and also replace a small amount of turbulence with a newly generated screen to effect
    phase "boiling".  The number of atmosphere layers is determined from the length of the `r0`,
    `L0`, `velocity`, `direction`, or `alpha_mag` arguments, if they are lists.  The length of these
    lists must all either be equal the number of layers or equal to 1.  In the latter case, the
    length-1 list is broadcast to length-N.

    @param time_step in seconds [Default: 0.03]
    @param screen_size in meters [Default: 10]
    @param screen_scale in meters [Default: 0.1]
    @param r0 in meters [Default: 0.2]
    @param L0 in meters [Default: 25.0]
    @param velocity in meters/second [Default: 0]
    @param direction CCW relative to +x as galsim.Angle [Default: 0*galsim.degrees]
    @param alpha_mag [Default: 0.999]
    @param rng BaseDeviate instance to provide random number generation

    Some suggestions for choices of wind velocities come from data hosted by NOAA. The Global Data
    Assimilation System (GDAS), run by the NOAA National Center for Environmental Prediction (NCEP),
    puts out various datasets. GDAS produces analyses every six hours, giving a large number of
    parameters tabulated at 24 altitudes in the atmosphere covering the entire Earth. The following
    is an example of wind velocities extracted from a file for 2014, Feb 13, 6h UT at the longitude
    and latitude of CTIO. (Wind velocities are m/s). Altitudes are approximate.
        Altitude (km)    velocity (m/s)    direction (deg.)
            1               0.98                -66
            10              0.67                116
            30              18.0                 27
    Data is found here: ftp://arlftp.arlhq.noaa.gov/pub/archives/gdas1/
    Data documentation: http://ready.arl.noaa.gov/gdas1.php
    """
    def __init__(self, time_step=0.03, screen_size=10.0, screen_scale=0.1, r0=0.2, L0=25.0,
                 velocity=0.0, direction=0*galsim.degrees, alpha_mag=0.999, rng=None):
        if rng is None:
            rng = galsim.BaseDeviate()
        self.rng = rng

        self.npix = int(np.ceil(screen_size/screen_scale))
        self.screen_scale = screen_scale
        # in case screen_scale doesn't divide screen_size
        self.screen_size = self.screen_scale * self.npix

        # Listify
        r0, L0, velocity, direction, alpha_mag = map(
            lambda i: [i] if not hasattr(i, '__iter__') else i,
            (r0, L0, velocity, direction, alpha_mag)
        )

        # Broadcast
        n_layers = max(map(len, [r0, L0, velocity, direction, alpha_mag]))
        if n_layers > 1:
            r0, L0, velocity, direction, alpha_mag = map(
                lambda i: [i[0]]*n_layers if len(i) == 1 else i,
                (r0, L0, velocity, direction, alpha_mag)
            )

        if any(len(i) != n_layers for i in (r0, L0, velocity, direction, alpha_mag)):
            raise ValueError("r0, L0, velocity, direction, alpha_mag not broadcastable")

        # Invert L0, with `L0 is None` interpretted as L0 = infinity => L0_inv = 0.0
        L0_inv = [1./L00 if L00 is not None else 0.0 for L00 in L0]

        # decompose velocities
        vx, vy = zip(*[v*d.sincos() for v, d in zip(velocity, direction)])

        # setup frequency grid
        # probe frequencies between -1/(2 * screen_scale) to 1/(2 * screen_scale),
        # in steps of 1/screen_size.
        fx = np.fft.fftfreq(self.npix, self.screen_scale)
        fx, fy = np.meshgrid(fx, fx)

        # setup phase screens states
        self.powerlaw = np.empty((n_layers, self.npix, self.npix), dtype=np.float64)
        self.alpha = np.empty((n_layers, self.npix, self.npix), dtype=np.complex128)
        self._phaseFT = np.empty((n_layers, self.npix, self.npix), dtype=np.complex128)
        self.screens = np.zeros_like(self.powerlaw)

        for i, (r00, L00_inv, vx0, vy0, amag0) in enumerate(zip(r0, L0_inv, vx, vy, alpha_mag)):
            # I believe the magic number below is 0.00058 ~= 0.023 / (2*pi)**2, where
            # 0.023 ~= (5 * (24/5 * gamma(6/5))**(5/6) * gamma(11/6) /
            #          (6 * pi**8/3 * gamma(1/6))
            # I used a combination of Roddier (1981), Noll (1976), and Sasiela (1994) to figure
            # this out.
            pl = (1./self.screen_size*np.sqrt(0.00058)*(r00**(-5.0/6.0)) *
                  (fx*fx + fy*fy + L00_inv*L00_inv)**(-11.0/12.0) *
                  self.npix * np.sqrt(np.sqrt(2.0)))
            pl[0, 0] = 0.0
            self.powerlaw[i] = pl
            self._phaseFT[i] = self._noiseFT(pl)
            self.screens[i] = np.fft.ifft2(self._phaseFT[i]).real

            # make array for the alpha parameter and populate it
            alpha_phase = -(fx*vx0 + fy*vy0) * time_step
            self.alpha[i] = amag0 * np.exp(2j*np.pi*alpha_phase)

    def _noiseFT(self, powerlaw):
        """  Return Fourier transform of Gaussian noise drawn from specified powerlaw.
        """
        gd = galsim.GaussianDeviate(self.rng)
        noise = utilities.rand_arr(powerlaw.shape, gd)
        return np.fft.fft2(noise)*powerlaw

    def next(self):
        """ Propagate phase screens with wind, update boiling, and return new screens.
        """
        for i, (pl, phFT, alpha) in enumerate(zip(self.powerlaw, self._phaseFT, self.alpha)):
            noisescalefac = np.sqrt(1. - np.abs(alpha**2))
            self._phaseFT[i] = alpha*phFT + self._noiseFT(pl)*noisescalefac
            self.screens[i] = np.fft.ifft2(self._phaseFT[i]).real
        return self.screens

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


class AtmosphericPSF(GSObject):
    """ Create an atmosphere PSF by summing over a phase cube in time steps.
    @param lam              Lambda (wavelength) in units of nanometers.  Must be supplied with `r0`,
                            and in this case, image scales (`scale`) should be specified in units of
                            `scale_unit`.
    @param r0               Fried parameter (in meters) setting the amplitude of turbulence.  If a
                            scalar, then this is the effective Fried parameter for combination of
                            all layers.  If a list, then each element specifies the Fried parameter
                            for a particular layer.  Note that the Fried parameter adds over layers
                            like r0_effective^(-5/3) = sum r0_i^(-5/3).
    @param L0               Outer scale in meters of von Karman spectrum.  [Default: None (= inf)]
    @param lam_over_r0      Ratio of wavelength to Fried parameter.  Can be a scalar, which
                            then specifies the net turbulence across all layers, or a list which
                            then indicates the turbulence for each individual layer.
    @param fwhm             Full width at half max (FWHM) of the PSF in the infinite exposure limit
                            in arcseconds.
    @param half_light_radius  The half-light-radius of the PSF in the infinite exposure limit.
                            Typically given in arcsec.
    @param weights          Relative weights for each turbulent layer in terms of the layers'
                            refractive index fluctuations C_n^2.  The Fried parameter for each layer
                            is then set via r0_i = r0_effective * weight_i^(-3/5).
    @param alpha_mag        Magnitude of autoregressive parameter.  (1-alpha) is the fraction of the
                            phase from the prior time step that is "forgotten" and replaced by
                            Gaussian noise.  [Default: 0.999]
    @param exptime          Exposure time in seconds. [Default: 0]
    @param time_step        Interval between PSF images in seconds. [Default: 0.03]
    @param velocity         Velocity magnitude of each phase screen layer in meters / second.
                            [Default: 0]
    @param direction        CCW relative to +x as galsim.Angle.  [Default: 0*galsim.degrees]
    @param phase_generator  AtmosphericPhaseGenerator object.  [Default: None]
    @param interpolant      Either an Interpolant instance or a string indicating which interpolant
                            should be used.  Options are 'nearest', 'sinc', 'linear', 'cubic',
                            'quintic', or 'lanczosN' where N should be the integer order to use.
                            [Default: galsim.Quintic()]
    @param flux             Total flux of the profile. [Default: 1.]
    @param scale_unit       Units used to define the diffraction limit and draw images, if the user
                            has supplied a separate value for `lam` and `r0`.  Should be either a
                            galsim.AngleUnit, or a string that can be used to construct one (e.g.,
                            'arcsec', 'radians', etc.).
                            [Default: galsim.arcsec]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [Default: None]
    """
    def __init__(self, lam=None, r0=None, L0=None, lam_over_r0=None, fwhm=None,
                 half_light_radius=None, weights=None, alpha_mag=None, exptime=0.0, time_step=0.03,
                 velocity=None, direction=None, phase_generator=None, interpolant=None,
                 oversampling=1.5, flux=1.0, scale_unit=galsim.arcsec, gsparams=None,
                 diam=None, obscuration=None, screen_size=None):
        import itertools

        nstep = int(np.ceil(exptime/time_step))
        if nstep == 0:
            nstep = 1

        if phase_generator is not None:
            if any(item is not None for item in (r0, L0, lam_over_r0, fwhm, weights, alpha_mag)):
                raise ValueError("Cannot specify r0, L0, lam_over_r0, fwhm, weights, or alpha_mag"
                                 " when specifying phase_generator")
        else:
            if fwhm is not None:
                lam_over_r0 = fwhm / 0.975865
            if half_light_radius is not None:
                lam_over_r0 = half_light_radius / 0.554811
            if alpha_mag is None:
                alpha_mag = 0.999
            if direction is None:
                direction = 0.0 * galsim.degrees
            if velocity is None:
                velocity = 0.0
            if weights is not None:
                if r0 is not None:
                    if hasattr(r0, '__len__'):
                        raise ValueError("Cannot specify both weights and list of r0s.")
                    r0 = [r0 * w**(-3./5) for w in weights]
                elif lam_over_r0 is not None:
                    if hasattr(lam_over_r0, '__len__'):
                        raise ValueError("Cannot specify both weights and list of lam_over_r0s.")
                    lam_over_r0 = [lam_over_r0 * w**(3./5) for w in weights]
            # Listify
            r0, velocity, direction, alpha_mag, lam_over_r0 = map(
                lambda i: [i] if not hasattr(i, '__iter__') else i,
                (r0, velocity, direction, alpha_mag, lam_over_r0)
            )

            if lam is None:
                lam = 800.  # arbitrarily set wavelength = 800nm
                r0 = [lam*1.e-9 / lor0 * galsim.radians / scale_unit for lor0 in lam_over_r0]
            # Should have lam as a scalar, and r0 as a list at this point.
            r0_effective = (sum(r**(-5./3) for r in r0)**(-3./5))

            # Sampling the phase screen is roughly analogous to sampling the PSF in Fourier space.
            # We can use a Kolmogorov profile to get a rough estimate of what stepK is needed to
            # avoid aliasing.
            kolm = galsim.Kolmogorov(lam=lam, r0=r0_effective)
            screen_scale = kolm.stepK()/(2*np.pi)  # 2pi b/c np and GalSim FFT conventions differ.
            # an arbitrary additional factor of 4 to account for the fact that a stochastic
            # atmospheric PSF can have significant fluctuations at relatively large radii.
            screen_scale /= 4.0
            # We'll hard code screen_size = 10 meters since that aperturn covers all planned
            # ground-based optical weak lensing experiments.
            if screen_size is None:
                screen_size = 10.0
            print "screen_scale: {}".format(screen_scale)
            print "screen_size: {}".format(screen_size)
            phase_generator = AtmosphericPhaseGenerator(
                time_step=time_step, screen_size=screen_size, screen_scale=screen_scale, r0=r0,
                L0=L0, alpha_mag=alpha_mag, velocity=velocity, direction=direction)
        self.phase_generator = phase_generator

        scale = 1e-9*lam/self.phase_generator.screen_size * galsim.radians / scale_unit
        # Generate PSFs for each time step
        nx, ny = phase_generator.screens[0].shape
        im_grid = np.zeros((nx, ny), dtype=np.float64)
        aper = np.ones_like(im_grid)
        if diam is not None:
            x = np.fft.fftfreq(nx, 1./phase_generator.screen_size)
            x, y = np.meshgrid(x, x)
            r = np.hypot(x, y)
            if obscuration is not None:
                aper = (r <= 0.5*diam) & (r >= 0.5*diam*obscuration)
            else:
                aper = r < 0.5*diam

        for i, screens in itertools.izip(xrange(nstep), phase_generator):
            # The wavefront to use is exp(2 pi i screen)
            wf = np.exp(2j * np.pi * np.sum(screens, axis=0)) * aper
            # Calculate the image array via FFT.
            # Copied from galsim.optics.psf method (hacky)
            ftwf = np.fft.ifft2(np.fft.ifftshift(wf))
            im = np.abs(ftwf)**2
            # Add this PSF instance to stack to get the finite-exposure PSF
            im_grid += im

        im_grid = np.fft.fftshift(im_grid)
        im_grid *= (flux / (im_grid.sum() * scale**2))
        out_im = galsim.InterpolatedImage(
            galsim.Image(im_grid.astype(np.float64), scale=scale),
            x_interpolant=interpolant, calculate_stepk=True, calculate_maxk=True,
            use_true_center=False, normalization='sb')
        GSObject.__init__(self, out_im)
