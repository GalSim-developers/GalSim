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

class AtmosphericPhaseCube(object):
    """ Create a phase cube using an autoregressive model.
    @param exptime in seconds
    @param time_step in seconds [Default: 0.03]
    @param screen_size in meters [Default: 10]
    @param screen_scale in meters [Default: 0.1]
    @param r0 in meters [Default: 0.2]
    @param alpha [Default: 0.999]
    @param velocity in meters/second [Default: 0]
    @param direction CCW relative to +x as galsim.Angle [Default: 0*galsim.degrees]
    """
    def __init__(self, exptime, time_step=0.03, screen_size=10.0, screen_scale=0.1,
                 r0=0.2, alpha_mag=0.999, velocity=0.0, direction=0*galsim.degrees):

        self.n = int(np.ceil(screen_size/screen_scale))
        self.nsteps = int(np.ceil(exptime/time_step))
        self.paramcube = np.array([r0, velocity, direction.rad()])
        self.paramcube.shape=(1, 3) #HACK
        self.screen_scale = screen_scale
        self.pl, self.alpha = create_multilayer_arbase(self.n, screen_scale, 1./time_step,
                                                       self.paramcube, alpha_mag)
        self._phaseFT = None
        self.screens = [[] for x in self.paramcube]

    def get_ar_atmos(self):
        shape = self.alpha.shape
        newphFT = []
        newphase = []
        for i, powerlaw, alpha in zip(range(shape[0]), self.pl, self.alpha):
            noise = np.random.normal(size=shape[1:3])
            noisescalefac = np.sqrt(1. - np.abs(alpha**2))
            noiseFT = np.fft.fft2(noise)*powerlaw
            if self._phaseFT is None:
                newphFT.append(noiseFT)
            else:
                newphFT.append(alpha*self._phaseFT[i] + noiseFT*noisescalefac)
            newphase.append(np.fft.ifft2(newphFT[i]).real)
        return np.array(newphFT), np.array(newphase)

    def run(self):
        for j in range(self.n):
            self._phaseFT, screens = self.get_ar_atmos()
            for i, item in enumerate(screens):
                self.screens[i].append(item)


def create_multilayer_arbase(n, pscale, rate, paramcube, alpha_mag,
                             boiling_only=False):
    """
    Function to create the starting phase screen to be used for an
    autoregressive atmosphere model. A powerlaw scales random noise
    generated to make it look like Kolmogorov turbulence.  alpha is
    the autoregressive parameter to scale the current phase.

    @param n          Number of pixels across the screen
    @param pscale     Pixel scale
    @param rate       A0 system rate (Hz)
    @param paramcube  Parameter array describing each layer of the atmosphere
                      to be modeled.  Each row contains a tuple of
                      (r0 (m), velocity (m/s), direction (deg))
                      describing the corresponding layer.
    @param alpha      magnitude of autoregressive parameter.  (1-alpha)
                      is the fraction of the phase from the prior time step
                      that is "forgotten" and replaced by Gaussian noise.
    @param boiling_only Flag to set all screen velocities to zero.
    """
    n_layers = len(paramcube)

    cp_r0s = paramcube[:, 0]      # r0 in meters
    cp_vels = paramcube[:, 1]     # m/s,  change to [0,0,0] to get pure boiling

    if boiling_only:
        cp_vels *= 0
    cp_dirs = paramcube[:, 2]*np.pi/180.   # in radians

    # decompose velocities
    cp_vels_x = cp_vels*np.cos(cp_dirs)
    cp_vels_y = cp_vels*np.sin(cp_dirs)

    screensize_meters = n*pscale # extent is given by aperture size and sampling
    deltaf = 1./screensize_meters   # spatial frequency delta

    # This is very similar to numpy.fftfreq, so we can probably use that, but for now
    # just copy over the original code from Srikar:
    #fx, fy = gg.generate_grids(n, scalefac=deltaf, freqshift=True)
    fx = np.zeros((n,n))
    for j in np.arange(n):
        fx[:,j] = j - (j > n/2)*n
    fx = fx * deltaf
    fy = fx.transpose()

    powerlaw = []
    alpha = []
    for i in range(n_layers):
        factor1 = 2*np.pi/screensize_meters*np.sqrt(0.00058)*(cp_r0s[i]**(-5.0/6.0))
        factor2 = (fx*fx + fy*fy)**(-11.0/12.0)
        factor3 = n*np.sqrt(np.sqrt(2.))
        powerlaw.append(factor1*factor2*factor3)
        powerlaw[-1][0][0] = 0.0

        # make array for the alpha parameter and populate it
        # phase of alpha = -2pi(k*vx + l*vy)*T/Nd where T is sampling interval
        # N is WFS grid, d is subap size in meters = pscale*m, k = 2pi*fx
        # fx, fy are k/Nd and l/Nd respectively
        alpha_phase = -2*np.pi*(fx*cp_vels_x[i] + fy*cp_vels_y[i])/rate
        try:
            alpha.append(alpha_mag[i]*(np.cos(alpha_phase) +
                                       1j*np.sin(alpha_phase)))
        except TypeError:
            # Just have a scalar for alpha_mag
            alpha.append(alpha_mag*(np.cos(alpha_phase) +
                                    1j*np.sin(alpha_phase)))

    powerlaw = np.array(powerlaw)
    alpha = np.array(alpha)

    return powerlaw, alpha


class AtmosphericPSF(GSObject):
    """ Create an atmosphere PSF by summing over a phase cube in time steps.
    @param lam              Lambda (wavelength) in units of nanometers.  Must be supplied with
                            `diam`, and in this case, image scales (`scale`) should be specified in
                            units of `scale_unit`.
    @param r0               Fried parameter for each layer of turbulence in meters.
                            This parameter sets the amplitude of turbulence for each
                            layer. [Default: 0.2]
    @param lam_over_r0      Lambda / Fried parameter
    @param fwhm             Full width at half max (FWHM) of the PSF in the infinite
                            exposure limit in arcseconds. [Default: 0.8]
    @param alpha            Magnitude of autoregressive parameter.  (1-alpha)
                            is the fraction of the phase from the prior time step
                            that is "forgotten" and replaced by Gaussian noise.
                            [Default: 0.999]
    @param exptime          Exposure time in seconds.
    @param time_step        Interval between PSF images in seconds. [Default: 0.03]
    @param velocity         Velocity magnitude of each phase screen layer in
                            meters / second. [Default: 0]
    @param direction        CCW relative to +x as galsim.Angle
                            [Default: 0*galsim.degrees]
    @param phase_cube       [Default: None]
    @param start_time       Start time in seconds for the simulation relative to the
                            beginning of the phase cube, if `phase_cube` is provided.
                            [Default: 0]
    @param stop_time        Stop time in seconds for the simulation relative to the
                            beginning of the phase cube, if `phase_cube` is provided.
                            [Default: None]
    @param interpolant      Either an Interpolant instance or a string indicating which interpolant
                            should be used.  Options are 'nearest', 'sinc', 'linear', 'cubic',
                            'quintic', or 'lanczosN' where N should be the integer order to use.
                            [default: galsim.Quintic()]
    @param oversampling     Optional oversampling factor for the InterpolatedImage. Setting
                            `oversampling < 1` will produce aliasing in the PSF (not good).
                            Usually `oversampling` should be somewhat larger than 1.  1.5 is
                            usually a safe choice.  [default: 1.5]
    @param flux             Total flux of the profile. [default: 1.]
    @param scale_unit       Units used to define the diffraction limit and draw images, if the user
                            has supplied a separate value for `lam` and `diam`.  Should be either a
                            galsim.AngleUnit, or a string that can be used to construct one (e.g.,
                            'arcsec', 'radians', etc.).
                            [default: galsim.arcsec]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """
    def __init__(self, lam=None, r0=0.2, lam_over_r0=None, fwhm=0.8,
                 alpha=0.999, exptime=None, time_step=0.03, velocity=0,
                 direction=0*galsim.degrees, phase_cube=None, start_time=0.,
                 stop_time=None, interpolant=galsim.Quintic(), oversampling=1.5,
                 flux=1., scale_unit=galsim.arcsec, gsparams=None):
        if phase_cube is None:
            ### Setup a new phase screen generator
            phase_cube = AtmosphericPhaseCube(exptime=exptime, time_step=time_step,
                screen_size=10., screen_scale=0.1, r0=r0, alpha=alpha,
                velocity=velocity, direction=direction)
        ### Generate the phase screens for every time step
        phase_cube.run()

        ### Generate PSFs for each time step
        for i, screen in enumerate(phase_cube.screens):
            ### The wavefront to use is exp(2 pi i screen)
            wf = np.exp(2j * np.pi * np.array(screen))
            ### Calculate the image array via FFT.
            ### Copied from galsim.optics.psf method (hacky)
            ftwf = np.fft.fft2(wf)
            im = np.abs(ftwf)**2
            im = utilities.roll2d(im, (im.shape[0] / 2, im.shape[1] / 2))
            im *= (flux / (im.sum() * phase_cube.screen_scale**2))
