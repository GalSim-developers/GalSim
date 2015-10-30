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
