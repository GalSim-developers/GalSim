# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file noise.py
Module which adds the addNoise and addNoiseSNR methods to the galsim.Image classes at the Python
layer.
"""

from . import _galsim

def addNoise(image, noise):
    """Noise addition Image method, adding noise according to a supplied noise model.

    >>> Image.addNoise(noise)

    Noise following supplied model will be added to the image.

    @param  image  The image on which to add the noise.
    @param  noise  Instantiated noise model (currently CCDNoise, UniformDeviate, BinomialDeviate,
                   GaussianDeviate and PoissonDeviate are supported).

    If the supplied noise model object does not have an applyTo() method, then this will raise an
    AttributeError exception.
    """
    noise.applyTo(image.view())

def addNoiseSNR(image, noise, snr, preserve_flux=False):
    """Adds CCDNoise to an image in a way that achieves the specified signal-to-noise ratio.
    
    >>> Image.addNoiseSNR(noise, snr, preserve_flux)  
                                                   
    Noise following the suppled model will be added to the image modifying either the flux of the
    object (if `preserve_flux=True`) or the variance of the noise (if `preserve_flux=False`) such
    that the given signal-to-noise ratio, `snr`, is reached.

    If `preserve_flux=False` (the default), the flux of the input image will be rescaled to achieve
    the desired signal-to-noise ratio (useful if adding noise separately to multiple galaxies with
    the same sky_level).
    
    If `preserve_flux=True`, then the variance of the noise model is modified.

    The definition of SNR is equivalent to the one used by Great08.  Taking a weighted integral 
    of the flux:
        S = sum W(x,y) I(x,y) / sum W(x,y)
        N^2 = Var(S) = sum W(x,y)^2 Var(I(x,y)) / (sum W(x,y))^2
    and assuming that Var(I(x,y)) is constant
        Var(I(x,y)) = noise_var
    We then assume that we are using a matched filter for W, so W(x,y) = I(x,y).  Then a few things 
    cancel and we find that
        snr = S/N = sqrt( sum I(x,y)^2 / noise_var )
    and therefore, for a given I(x,y) and snr,
        noise_var = sum I(x,y)^2/snr^2.

    Not that for noise models such as Poisson and CCDNoise, the constant Var(I(x,y)) assumption
    is only approximate, since the flux of the object adds to the Poisson noise in those pixels.
    Thus, the real S/N on the final image will be slightly lower than the target `snr` value, 
    and this effect will be larger for brighter objects.
    """
    import numpy
    noise_var = noise.getVariance()
    if preserve_flux:
        new_noise_var = numpy.sum(image.array**2)/snr/snr
        noise.setVariance(new_noise_var)
        image.addNoise(noise)
        noise.setVariance(noise_var)  # Revert to condition on input.
    else:
        sn_meas = numpy.sqrt( numpy.sum(image.array**2)/noise_var )
        flux = snr/sn_meas
        image *= flux
        image.addNoise(noise)

# inject addNoise and addNoiseSNR as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.addNoise = addNoise
    Class.addNoiseSNR = addNoiseSNR

for Class in _galsim.ImageView.itervalues():
    Class.addNoise = addNoise
    Class.addNoiseSNR = addNoiseSNR

del Class # cleanup public namespace

# Then add docstrings for C++ layer Noise classes

# BaseNoise methods used by derived classes
_galsim.BaseNoise.getRNG.__func__.__doc__ = """
Get the galsim.BaseDeviate used to generate random numbers for the current noise model.
"""
_galsim.BaseNoise.setRNG.__func__.__doc__ = """
Set the galsim.BaseDeviate used to generate random numbers for the current noise model.
"""
_galsim.BaseNoise.getVariance.__func__.__doc__ = "Get variance in current noise model."
_galsim.BaseNoise.setVariance.__func__.__doc__ = "Set variance in current noise model."
_galsim.BaseNoise.scaleVariance.__func__.__doc__ = "Scale variance in current noise model."

# Make op* and op*= work to adjust the overall variance of a BaseNoise object
def Noise_imul(self, other):
    self.scaleVariance(other)
    return self

def Noise_mul(self, other):
    ret = self.copy()
    Noise_imul(ret, other)
    return ret

# Likewise for op/ and op/=
def Noise_idiv(self, other):
    self.scaleVariance(1. / other)
    return self

def Noise_div(self, other):
    ret = self.copy()
    Noise_idiv(ret, other)
    return ret

_galsim.BaseNoise.__imul__ = Noise_imul
_galsim.BaseNoise.__mul__ = Noise_mul
_galsim.BaseNoise.__rmul__ = Noise_mul
_galsim.BaseNoise.__div__ = Noise_div
_galsim.BaseNoise.__truediv__ = Noise_div
 
# GaussianNoise docstrings
_galsim.GaussianNoise.__doc__ = """
Class implementing simple Gaussian noise.

This is a simple noise model where each pixel in the image gets Gaussian noise with a
given sigma.

Initialization
--------------

    >>> gaussian_noise = galsim.GaussianNoise(rng, sigma=1.)

Parameters:

    rng       A BaseDeviate instance to use for generating the random numbers.
    sigma     The rms noise on each pixel [default `sigma = 1.`].

Methods
-------
To add noise to every element of an image, use the syntax image.addNoise(gaussian_noise).
"""

_galsim.GaussianNoise.applyTo.__func__.__doc__ = """
Add Gaussian noise to an input Image.

Calling
-------

    >>> gaussian_noise.applyTo(image)

On output the Image instance image will have been given additional Gaussian noise according 
to the given GaussianNoise instance.

Note: The syntax image.addNoise(gaussian_noise) is preferred.
"""
_galsim.GaussianNoise.getSigma.__func__.__doc__ = "Get sigma in current noise model."
_galsim.GaussianNoise.setSigma.__func__.__doc__ = "Set sigma in current noise model."

def GaussianNoise_copy(self):
    return _galsim.GaussianNoise(self.getRNG(),self.getSigma())
_galsim.GaussianNoise.copy = GaussianNoise_copy


# PoissonNoise docstrings
_galsim.PoissonNoise.__doc__ = """
Class implementing Poisson noise according to the flux (and sky level) present in the image.

The PoissonNoise class encapsulates a simple version of the noise model of a normal CCD image
where each pixel has Poisson noise corresponding to the number of electrons in each pixel
(including an optional extra sky level).

Note that if the image to which you are adding noise already has a sky level on it,
then you should not provide the sky level here as well.  The sky level here corresponds
to a level that is taken to be already subtracted from the image, but that originally contributed to
the addition of Poisson noise.

Initialization
--------------

    >>> poisson_noise = galsim.PoissonNoise(rng, sky_level=0.)

Parameters:

    rng         A BaseDeviate instance to use for generating the random numbers.
    sky_level   The sky level in electrons per pixel that was originally in the input image, 
                but which is taken to have already been subtracted off [default `sky_level = 0.`].

Methods
-------
To add noise to every element of an image, use the syntax image.addNoise(poisson_noise).
"""

_galsim.PoissonNoise.applyTo.__func__.__doc__ = """
Add Poisson noise to an input Image.

Calling
-------

    >>> galsim.PoissonNoise.applyTo(image)

On output the Image instance image will have been given additional Poisson noise according 
to the given PoissonNoise instance.

Note: the syntax image.addNoise(poisson_noise) is preferred.
"""
_galsim.PoissonNoise.getSkyLevel.__func__.__doc__ = "Get sky level in current noise model."
_galsim.PoissonNoise.setSkyLevel.__func__.__doc__ = "Set sky level in current noise model."

def PoissonNoise_copy(self):
    return _galsim.PoissonNoise(self.getRNG(),self.getSkyLevel())
_galsim.PoissonNoise.copy = PoissonNoise_copy


# CCDNoise docstrings
_galsim.CCDNoise.__doc__ = """
Class implementing a basic CCD noise model.

The CCDNoise class encapsulates the noise model of a normal CCD image.  The noise has two
components: first, Poisson noise corresponding to the number of electrons in each pixel
(including an optional extra sky level); second, Gaussian read noise.

Note that if the image to which you are adding noise already has a sky level on it,
then you should not provide the sky level here as well.  The sky level here corresponds
to a level is taken to be already subtracted from the image, but which was present
for the Poisson noise.

Initialization
--------------

    >>> ccd_noise = galsim.CCDNoise(rng, sky_level=0., gain=1., read_noise=0.)  

Parameters:

    rng         A BaseDeviate instance to use for generating the random numbers.
    sky_level   The sky level in electrons per pixel that was originally in the input image, 
                but which is taken to have already been subtracted off [default `sky_level = 0.`].
    gain        The gain for each pixel in electrons per ADU; setting gain<=0 will shut off the
                Poisson noise, and the Gaussian rms will take the value read_noise as being in 
                units of ADU rather than electrons [default `gain = 1.`].
    read_noise  The read noise on each pixel in electrons (gain > 0.) or ADU (gain <= 0.)
                setting read_noise=0. will shut off the Gaussian noise [default `read_noise = 0.`].

Methods
-------
To add noise to every element of an image, use the syntax image.addNoise(ccd_noise).
"""

_galsim.CCDNoise.applyTo.__func__.__doc__ = """
Add noise to an input Image.

Calling
-------

    >>> ccd_noise.applyTo(image)

On output the Image instance image will have been given additional stochastic noise according to 
the gain and read noise settings of the given CCDNoise instance.

Note: the syntax image.addNoise(ccd_noise) is preferred.
"""
_galsim.CCDNoise.getSkyLevel.__func__.__doc__ = "Get sky level in current noise model."
_galsim.CCDNoise.getGain.__func__.__doc__ = "Get gain in current noise model."
_galsim.CCDNoise.getReadNoise.__func__.__doc__ = "Get read noise in current noise model."
_galsim.CCDNoise.setSkyLevel.__func__.__doc__ = "Set sky level in current noise model."
_galsim.CCDNoise.setGain.__func__.__doc__ = "Set gain in current noise model."
_galsim.CCDNoise.setReadNoise.__func__.__doc__ = "Set read noise in current noise model."

def CCDNoise_copy(self):
    return _galsim.CCDNoise(self.getRNG(),self.getSkyLevel(),self.getGain(),self.getReadNoise())
_galsim.CCDNoise.copy = CCDNoise_copy


# DeviateNoise docstrings
_galsim.DeviateNoise.__doc__ = """
Class implementing noise with an arbitrary BaseDeviate object.

The DeviateNoise class provides a way to treat an arbitrary deviate as the noise model for 
each pixel in an image.

Initialization
--------------

    >>> dev_noise = galsim.DeviateNoise(dev)

Parameters:

    dev         A BaseDeviate subclass to use as the noise deviate for each pixel.

Methods
-------
To add noise to every element of an image, use the syntax image.addNoise(dev_noise).
"""

_galsim.DeviateNoise.applyTo.__func__.__doc__ = """
Add noise to an input Image.

Calling
-------

    >>> dev_noise.applyTo(image)

On output the Image instance image will have been given additional noise according to 
the given DeviateNoise instance.

To add deviates to every element of an image, the syntax image.addNoise() is preferred.
"""

def DeviateNoise_copy(self):
    return _galsim.DeviateNoise(self.getRNG())
_galsim.DeviateNoise.copy = DeviateNoise_copy

