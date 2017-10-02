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

def Noise_getRNG(self):
    depr("noise.getRNG()", 1.5, "noise.rng")
    return self.rng

def Noise_applyToView(self, im_view):
    depr("noise.applyToView(image.image.view())", 1.5, "noise.applyTo(image)")
    self._applyToView(im_view)

galsim._galsim.BaseNoise.setRNG = Noise_setRNG
galsim._galsim.BaseNoise.setVariance = Noise_setVariance
galsim._galsim.BaseNoise.scaleVariance = Noise_scaleVariance
galsim._galsim.BaseNoise.getRNG = Noise_getRNG
galsim._galsim.BaseNoise.applyToView = Noise_applyToView

def GaussianNoise_setSigma(self, sigma):
    """Deprecated method to set the value of sigma
    """
    depr('setSigma', 1.1, 'noise = galsim.GaussianNoise(noise.rng, sigma)')
    self._setSigma(sigma)

def GaussianNoise_getSigma(self):
    depr("gn.getSigma()", 1.5, "gn.sigma")
    return self.sigma

galsim._galsim.GaussianNoise.setSigma = GaussianNoise_setSigma
galsim._galsim.GaussianNoise.getSigma = GaussianNoise_getSigma

def PoissonNoise_setSkyLevel(self, sky_level):
    """Deprecated method to set the value of sky_level
    """
    depr('setSkyLevel', 1.1, 'noise = galsim.PoissonNoise(noise.rng, sky_level)')
    self._setSkyLevel(sky_level)

def PoissonNoise_getSkyLevel(self):
    depr("pn.getSkyLevel()", 1.5, "pn.sky_level")
    return self.sky_level

galsim._galsim.PoissonNoise.setSkyLevel = PoissonNoise_setSkyLevel
galsim._galsim.PoissonNoise.getSkyLevel = PoissonNoise_getSkyLevel

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

def CCDNoise_getSkyLevel(self):
    depr("ccdn.getSkyLevel()", 1.5, "ccdn.sky_level")
    return self.sky_level

def CCDNoise_getGain(self):
    depr("ccdn.getGain()", 1.5, "ccdn.gain")
    return self.gain

def CCDNoise_getReadNoise(self):
    depr("ccdn.getReadNoise()", 1.5, "ccdn.read_noise")
    return self.read_noise

galsim._galsim.CCDNoise.setSkyLevel = CCDNoise_setSkyLevel
galsim._galsim.CCDNoise.setGain = CCDNoise_setGain
galsim._galsim.CCDNoise.setReadNoise = CCDNoise_setReadNoise
galsim._galsim.CCDNoise.getSkyLevel = CCDNoise_getSkyLevel
galsim._galsim.CCDNoise.getGain = CCDNoise_getGain
galsim._galsim.CCDNoise.getReadNoise = CCDNoise_getReadNoise

def VGNoise_setVariance(self, variance):
    raise RuntimeError("Changing the variance is not allowed for VariableGaussianNoise")

def VGNoise_scaleVariance(self, variance):
    raise RuntimeError("Changing the variance is not allowed for VariableGaussianNoise")

def VGNoise_getVarImage(self):
    depr("vgn.getVarImage()", 1.5, "vgn.var_image")
    return self.var_image

galsim.noise.VariableGaussianNoise.setVariance = VGNoise_setVariance
galsim.noise.VariableGaussianNoise.scaleVariance = VGNoise_scaleVariance
galsim.noise.VariableGaussianNoise.getVarImage = VGNoise_getVarImage
