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
from galsim.deprecated import depr

def Noise_setRNG(self, rng):
    """Deprecated method to set the BaseDeviate used to generate random numbers for
    the current noise model.
    """
    depr('setRNG', 1.1, 'noise = noise.copy(rng=rng)')
    self._setRNG(rng)

def Noise_setVariance(self, variance):
    """A deprecated method that is roughly equivalent to
    `noise = noise.withVariance(variance)`.
    """
    depr('setVariance', 1.1, 'noise = noise.withVariance(variance)')
    self._setVariance(variance)

def Noise_scaleVariance(self, variance_ratio):
    """A deprecated method that is roughly equivalent to `noise = noise * variance_ratio`.
    """
    depr('scaleVariance', 1.1, 'noise *= variance_ratio')
    self._scaleVariance(variance_ratio)

galsim._galsim.BaseNoise.setRNG = Noise_setRNG
galsim._galsim.BaseNoise.setVariance = Noise_setVariance
galsim._galsim.BaseNoise.scaleVariance = Noise_scaleVariance

def GaussianNoise_setSigma(self, sigma):
    """Deprecated method to set the value of sigma
    """
    depr('setSigma', 1.1, 'noise = galsim.GaussianNoise(noise.rng, sigma)')
    self._setSigma(sigma)

galsim._galsim.GaussianNoise.setSigma = GaussianNoise_setSigma

def PoissonNoise_setSkyLevel(self, sky_level):
    """Deprecated method to set the value of sky_level
    """
    depr('setSkyLevel', 1.1, 'noise = galsim.PoissonNoise(noise.rng, sky_level)')
    self._setSkyLevel(sky_level)

galsim._galsim.PoissonNoise.setSkyLevel = PoissonNoise_setSkyLevel

def CCDNoise_setSkyLevel(self, sky_level):
    """Deprecated method to set the value of sky_level
    """
    depr('setSkyLevel', 1.1,
         'noise = galsim.CCDNoise(noise.rng, sky_level, noise.gain, noise.read_noise)')
    self._setSkyLevel(sky_level)

def CCDNoise_setGain(self, gain):
    """Deprecated method to set the value of gain
    """
    depr('setGain', 1.1,
         'noise = galsim.CCDNoise(noise.rng, noise.sky_level, gain, noise.read_noise)')
    self._setGain(gain)

def CCDNoise_setReadNoise(self, read_noise):
    """Deprecated method to set the value of read_noise
    """
    depr('setReadNoise', 1.1,
         'noise = galsim.CCDNoise(noise.rng, noise.sky_level, noise.gain, read_noise)')
    self._setReadNoise(read_noise)

galsim._galsim.CCDNoise.setSkyLevel = CCDNoise_setSkyLevel
galsim._galsim.CCDNoise.setGain = CCDNoise_setGain
galsim._galsim.CCDNoise.setReadNoise = CCDNoise_setReadNoise

