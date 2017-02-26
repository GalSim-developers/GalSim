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
"""@file noise.py
Module which adds the addNoise() and addNoiseSNR() methods to the Image classes at the Python
layer.
"""

import galsim
from . import _galsim
from ._galsim import BaseNoise, GaussianNoise, PoissonNoise, CCDNoise
from ._galsim import DeviateNoise, VarGaussianNoise
from .utilities import set_func_doc
import numpy as np

def addNoise(self, noise):
    # This will be inserted into the Image class as a method.  So self = image.
    """Add noise to the image according to a supplied noise model.

    @param noise        The noise (BaseNoise) model to use.
    """
    noise.applyTo(self)

def addNoiseSNR(self, noise, snr, preserve_flux=False):
    # This will be inserted into the Image class as a method.  So self = image.
    """Adds noise to the image in a way that achieves the specified signal-to-noise ratio.

    The given SNR (`snr`) can be achieved either by scaling the flux of the object while keeping the
    noise level fixed, or the flux can be preserved and the noise variance changed.  This is set
    using the parameter `preserve_flux`.

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

    Note that for noise models such as Poisson and CCDNoise, the constant Var(I(x,y)) assumption
    is only approximate, since the flux of the object adds to the Poisson noise in those pixels.
    Thus, the real S/N on the final image will be slightly lower than the target `snr` value,
    and this effect will be larger for brighter objects.

    Also, this function relies on noise.getVariance() to determine how much variance the
    noise model will add.  Thus, it will not work for noise models that do not have a well-
    defined variance, such as VariableGaussianNoise.

    @param noise        The noise (BaseNoise) model to use.
    @param snr          The desired signal-to-noise after the noise is applied.
    @param preserve_flux  Whether to preserve the flux of the object (True) or the variance of
                        the noise model (False) to achieve the desired SNR. [default: False]

    @returns the variance of the noise that was applied to the image.
    """
    noise_var = noise.getVariance()
    sumsq = np.sum(self.array**2, dtype=float)
    if preserve_flux:
        new_noise_var = sumsq/snr/snr
        noise = noise.withVariance(new_noise_var)
        self.addNoise(noise)
        return new_noise_var
    else:
        sn_meas = np.sqrt( sumsq/noise_var )
        flux = snr/sn_meas
        self *= flux
        self.addNoise(noise)
        return noise_var

galsim.Image.addNoise = addNoise
galsim.Image.addNoiseSNR = addNoiseSNR

# Then add docstrings for C++ layer Noise classes

# BaseNoise methods used by derived classes
set_func_doc(_galsim.BaseNoise.getRNG, """
Get the BaseDeviate used to generate random numbers for the current noise model.
""")

set_func_doc(_galsim.BaseNoise.getVariance, "Get variance in current noise model.")

def Noise_withVariance(self, variance):
    """Return a new noise object (of the same type as the current one) with the specified variance.

    @param variance     The desired variance in the noise.

    @returns a new Noise object with the given variance.
    """
    ret = self.copy()
    ret._setVariance(variance)
    return ret

def Noise_withScaledVariance(self, variance_ratio):
    """Return a new noise object with the variance scaled up by the specified factor.

    This is equivalent to noise * variance_ratio.

    @param variance_ratio   The factor by which to scale the variance of the correlation
                            function profile.

    @returns a new Noise object whose variance has been scaled by the given amount.
    """
    ret = self.copy()
    ret._scaleVariance(variance_ratio)
    return ret

_galsim.BaseNoise.withVariance = Noise_withVariance
_galsim.BaseNoise.withScaledVariance = Noise_withScaledVariance

# Make op* and op*= work to adjust the overall variance of a BaseNoise object
def Noise_mul(self, variance_ratio):
    """Multiply the variance of the noise by `variance_ratio`.

    @param variance_ratio   The factor by which to scale the variance of the correlation
                            function profile.

    @returns a new Noise object whose variance has been scaled by the given amount.
    """
    return self.withScaledVariance(variance_ratio)

# Likewise for op/
def Noise_div(self, variance_ratio):
    """Equivalent to self * (1/variance_ratio)"""
    return self.withScaledVariance(1./variance_ratio)

_galsim.BaseNoise.__mul__ = Noise_mul
_galsim.BaseNoise.__rmul__ = Noise_mul
_galsim.BaseNoise.__div__ = Noise_div
_galsim.BaseNoise.__truediv__ = Noise_div

# Quick and dirty.  Just check reprs are equal.
_galsim.BaseNoise.__eq__ = lambda self, other: repr(self) == repr(other)
_galsim.BaseNoise.__ne__ = lambda self, other: not self.__eq__(other)
_galsim.BaseNoise.__hash__ = lambda self: hash(repr(self))

# GaussianNoise docstrings
_galsim.GaussianNoise.__doc__ = """
Class implementing simple Gaussian noise.

This is a simple noise model where each pixel in the image gets Gaussian noise with a
given `sigma`.

Initialization
--------------

@param rng          A BaseDeviate instance to use for generating the random numbers.
@param sigma        The rms noise on each pixel. [default: 1.]

Methods
-------

To add noise to every element of an image, use the syntax `image.addNoise(gaussian_noise)`.

Attributes
----------

    noise.rng           # The internal random number generator (read-only)
    noise.sigma         # The value of the constructor parameter sigma (read-only)
"""

def GaussianNoise_applyTo(self, image):
    """
    Add Gaussian noise to an input Image.

    Calling
    -------

        >>> gaussian_noise.applyTo(image)

    On output the Image instance `image` will have been given additional Gaussian noise according
    to the given GaussianNoise instance.

    Note: The syntax `image.addNoise(gaussian_noise)` is preferred.
    """
    self.applyToView(image.image.view())
_galsim.GaussianNoise.applyTo = GaussianNoise_applyTo

set_func_doc(_galsim.GaussianNoise.getSigma, "Get `sigma` in current noise model.")

def GaussianNoise_copy(self, rng=None):
    """Returns a copy of the Gaussian noise model.

    By default, the copy will share the BaseDeviate random number generator with the parent
    instance.  However, you can provide a new rng to use in the copy if you want with

        >>> noise_copy = noise.copy(rng=new_rng)
    """
    if rng is None: rng = self.rng
    return _galsim.GaussianNoise(rng, self.getSigma())

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

@param rng          A BaseDeviate instance to use for generating the random numbers.
@param sky_level    The sky level in electrons per pixel that was originally in the input image,
                    but which is taken to have already been subtracted off. [default: 0.]

Methods
-------

To add noise to every element of an image, use the syntax `image.addNoise(poisson_noise)`.

Attributes
----------

    noise.rng           # The internal random number generator (read-only)
    noise.sky_level     # The value of the constructor parameter sky_level (read-only)
"""

def PoissonNoise_applyTo(self, image):
    """
    Add Poisson noise to an input Image.

    Calling
    -------

        >>> galsim.PoissonNoise.applyTo(image)

    On output the Image instance `image` will have been given additional Poisson noise according
    to the given PoissonNoise instance.

    Note: the syntax `image.addNoise(poisson_noise)` is preferred.
    """
    self.applyToView(image.image.view())
_galsim.PoissonNoise.applyTo = PoissonNoise_applyTo

set_func_doc(_galsim.PoissonNoise.getSkyLevel, "Get sky level in current noise model.")

def PoissonNoise_copy(self, rng=None):
    """Returns a copy of the Poisson noise model.

    By default, the copy will share the BaseDeviate random number generator with the parent
    instance.  However, you can provide a new rng to use in the copy if you want with

        >>> noise_copy = noise.copy(rng=new_rng)
    """
    if rng is None: rng = self.rng
    return _galsim.PoissonNoise(rng, self.getSkyLevel())

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

The units here are slightly confusing.  We try to match the most common way that each of
these quantities is usually reported.  Note: ADU stands for Analog/Digital Units; they are the
units of the numbers in the final image.  Some places use the term "counts" for this.

- sky_level is normally measured from the image itself, so it is normally quoted in ADU/pixel.
- gain is a property of the detector and is normally measured in the laboratory.  The units
  are normally e-/ADU.  This is backwards what might be more intuitive, ADU/e-, but that's
  how astronomers use the term gain, so we follow suit here.
- read_noise is also a property of the detector and is usually quoted in e-/pixel.

If you are manually applying the quantum efficiency of the detector (e-/photon), then this would
normally be applied before the noise.  However, it is also fine to fold in the quantum efficiency
into the gain to give units photons/ADU.  Either way is acceptable.  Just make sure you give
the read noise in photons as well in this case.

Initialization
--------------

    >>> ccd_noise = galsim.CCDNoise(rng, sky_level=0., gain=1., read_noise=0.)

Parameters:

@param rng          A BaseDeviate instance to use for generating the random numbers.
@param sky_level    The sky level in ADU per pixel that was originally in the input image,
                    but which is taken to have already been subtracted off. [default: 0.]
@param gain         The gain for each pixel in electrons per ADU; setting `gain<=0` will shut off
                    the Poisson noise, and the Gaussian rms will take the value `read_noise` as
                    being in units of ADU rather than electrons. [default: 1.]
@param read_noise   The read noise on each pixel in electrons (gain > 0.) or ADU (gain <= 0.).
                    Setting `read_noise=0`. will shut off the Gaussian noise. [default: 0.]

Methods
-------

To add noise to every element of an image, use the syntax `image.addNoise(ccd_noise)`.

Attributes
----------

    noise.rng           # The internal random number generator (read-only)
    noise.sky_level     # The value of the constructor parameter sky_level (read-only)
    noise.gain          # The value of the constructor parameter gain (read-only)
    noise.read_noise    # The value of the constructor parameter read_noise (read-only)
"""

def CCDNoise_applyTo(self, image):
    """
    Add CCD noise to an input Image.

    Calling
    -------

        >>> ccd_noise.applyTo(image)

    On output the Image instance `image` will have been given additional stochastic noise according
    to the gain and read noise settings of the given CCDNoise instance.

    Note: the syntax `image.addNoise(ccd_noise)` is preferred.
    """
    self.applyToView(image.image.view())
_galsim.CCDNoise.applyTo = CCDNoise_applyTo

set_func_doc(_galsim.CCDNoise.getSkyLevel, "Get sky level in current noise model.")
set_func_doc(_galsim.CCDNoise.getGain, "Get gain in current noise model.")
set_func_doc(_galsim.CCDNoise.getReadNoise, "Get read noise in current noise model.")

def CCDNoise_copy(self, rng=None):
    """Returns a copy of the CCD noise model.

    By default, the copy will share the BaseDeviate random number generator with the parent
    instance.  However, you can provide a new rng to use in the copy if you want with

        >>> noise_copy = noise.copy(rng=new_rng)
    """
    if rng is None: rng = self.rng
    return _galsim.CCDNoise(rng, self.getSkyLevel(), self.getGain(), self.getReadNoise())

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

@param dev         A BaseDeviate subclass to use as the noise deviate for each pixel.

Methods
-------

To add noise to every element of an image, use the syntax `image.addNoise(dev_noise)`.

Attributes
----------

    noise.rng       # The internal random number generator (read-only)
"""

def DeviateNoise_applyTo(self, image):
    """
    Add noise according to a random deviate to an input Image.

    Calling
    -------

        >>> dev_noise.applyTo(image)

    On output the Image instance `image` will have been given additional noise according to
    the given DeviateNoise instance.

    To add deviates to every element of an image, the syntax `image.addNoise()` is preferred.
    """
    self.applyToView(image.image.view())
_galsim.DeviateNoise.applyTo = DeviateNoise_applyTo

def DeviateNoise_copy(self, rng=None):
    """Returns a copy of the Deviate noise model.

    By default, the copy will share the BaseDeviate random number generator with the parent
    instance.  However, you can provide a new rng to use in the copy if you want with

        >>> noise_copy = noise.copy(rng=new_rng)
    """
    if rng is None: rng = self.rng
    else: rng = self.rng.duplicate().reset(rng)
    return _galsim.DeviateNoise(rng)

_galsim.DeviateNoise.copy = DeviateNoise_copy

# VariableGaussianNoise is a thin wrapper of the C++ VarGaussianNoise
# This way the python layer can have the argument be a galsim.Image object,
# but the C++ version can take a C++ BaseImage object.

class VariableGaussianNoise(_galsim.BaseNoise):
    """
    Class implementing Gaussian noise that has a different variance in each pixel.

    Initialization
    --------------

        >>> variable_noise = galsim.VariableGaussianNoise(rng, var_image)

    Parameters:

    @param rng          A BaseDeviate instance to use for generating the random numbers.
    @param var_image    The variance of the noise to apply to each pixel.  This image must be the
                        same shape as the image for which you eventually call addNoise().

    Methods
    -------
    To add noise to every element of an image, use the syntax `image.addNoise(variable_noise)`.

    Attributes
    ----------

        noise.rng           # The internal random number generator (read-only)
        noise.var_image     # The value of the constructor parameter var_image (read-only)
    """
    def __init__(self, rng, var_image):
        if not isinstance(rng, galsim.BaseDeviate):
            raise TypeError(
                "Supplied rng argument not a galsim.BaseDeviate or derived class instance.")

        # Make sure var_image is an ImageF, converting dtype if necessary
        var_image = galsim.ImageF(var_image)

        # Make the noise object using the image.image as needed in the C++ layer.
        self.noise = _galsim.VarGaussianNoise(rng, var_image.image)

    def applyTo(self, image):
        """
        Add VariableGaussian noise to an input Image.

        Calling
        -------

            >>> variable_noise.applyTo(image)

        On output the Image instance `image` will have been given additional Gaussian noise
        according to the variance image of the given VariableGaussianNoise instance.

        Note: The syntax `image.addNoise(variable_noise)` is preferred.
        """
        self.noise.applyToView(image.image.view())

    def applyToView(self, image_view):
        self.noise.applyToView(image_view)

    def getVarImage(self):
        return galsim.Image(self.noise.getVarImage())

    def getRNG(self):
        return self.noise.getRNG()

    @property
    def rng(self): return self.getRNG()
    @property
    def var_image(self): return self.getVarImage()

    def copy(self, rng=None):
        """Returns a copy of the variable Gaussian noise model.

        By default, the copy will share the BaseDeviate random number generator with the parent
        instance.  However, you can provide a new rng to use in the copy if you want with

            >>> noise_copy = noise.copy(rng=new_rng)
        """
        if rng is None: rng = self.rng
        return VariableGaussianNoise(rng, self.getVarImage())

    def getVariance(self):
        raise RuntimeError("No single variance value for VariableGaussianNoise")

    def withVariance(self, variance):
        raise RuntimeError("Changing the variance is not allowed for VariableGaussianNoise")

    def withScaledVariance(self, variance):
        # This one isn't undefined like withVariance, but it's inefficient.  Better to
        # scale the values in the image before constructing VariableGaussianNoise.
        raise RuntimeError("Changing the variance is not allowed for VariableGaussianNoise")

    def setVariance(self, variance):
        raise RuntimeError("Changing the variance is not allowed for VariableGaussianNoise")

    def scaleVariance(self, variance):
        raise RuntimeError("Changing the variance is not allowed for VariableGaussianNoise")

    def __repr__(self):
        return 'galsim.VariableGaussianNoise(rng=%r, var_image%r)'%(self.rng, self.var_image)

    def __str__(self):
        return 'galsim.VariableGaussianNoise(var_image%s)'%(self.var_image)

# Enable pickling of the boost-python wrapped classes
_galsim.GaussianNoise.__getinitargs__ = lambda self: (self.rng, self.sigma)
_galsim.PoissonNoise.__getinitargs__ = lambda self: (self.rng, self.sky_level)
_galsim.CCDNoise.__getinitargs__ = \
        lambda self: (self.rng, self.sky_level, self.gain, self.read_noise)
_galsim.DeviateNoise.__getinitargs__ = lambda self: (self.rng, )
_galsim.VarGaussianNoise.__getinitargs__ = lambda self: (self.rng, self.var_image)

# Make repr and str functions
_galsim.GaussianNoise.__repr__ = \
        lambda self: 'galsim.GaussianNoise(rng=%r, sigma=%r)'%(self.rng, self.sigma)
_galsim.PoissonNoise.__repr__ = \
        lambda self: 'galsim.PoissonNoise(rng=%r, sky_level=%r)'%(self.rng, self.sky_level)
_galsim.CCDNoise.__repr__ = \
        lambda self: 'galsim.CCDNoise(rng=%r, sky_level=%r, gain=%r, read_noise=%r)'%(
            self.rng, self.sky_level, self.gain, self.read_noise)
_galsim.DeviateNoise.__repr__ = \
        lambda self: 'galsim.DeviateNoise(dev=%r)'%(self.rng)
_galsim.VarGaussianNoise.__repr__ = \
        lambda self: 'galsim.VarGaussianNoise(rng=%r, var_image%r)'%(self.rng, self.var_image)

_galsim.GaussianNoise.__str__ = \
        lambda self: 'galsim.GaussianNoise(sigma=%s)'%(self.sigma)
_galsim.PoissonNoise.__str__ = \
        lambda self: 'galsim.PoissonNoise(sky_level=%s)'%(self.sky_level)
_galsim.CCDNoise.__str__ = \
        lambda self: 'galsim.CCDNoise(sky_level=%s, gain=%s, read_noise=%s)'%(
            self.sky_level, self.gain, self.read_noise)
_galsim.DeviateNoise.__str__ = \
        lambda self: 'galsim.DeviateNoise(dev=%s)'%(self.rng)
_galsim.VarGaussianNoise.__str__ = \
        lambda self: 'galsim.VarGaussianNoise(var_image%s)'%(self.var_image)
