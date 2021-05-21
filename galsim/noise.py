# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

import numpy as np
import math

from .image import Image, ImageD
from .utilities import doc_inherit
from .errors import GalSimError, GalSimIncompatibleValuesError


def addNoise(self, noise):
    # This will be inserted into the Image class as a method.  So self = image.
    """Add noise to the image according to a supplied noise model.

    Parameters:
        noise:      The noise (`BaseNoise`) model to use.
    """
    noise.applyTo(self)

# This will be inserted into the Image class as a method.  So self = image.
def addNoiseSNR(self, noise, snr, preserve_flux=False):
    r"""Adds noise to the image in a way that achieves the specified signal-to-noise ratio.

    The specified ``snr`` (signal-to-noise ratio, or :math:`S/N`) can be achieved either by scaling
    the flux of the object while keeping the noise level fixed, or the flux can be preserved and
    the noise variance changed.  This choice is specified using the parameter ``preserve_flux``,
    where False means the former and True means the latter.

    The definition of :math:`S/N` is equivalent to the one used by Great08.  We take the signal to
    be a weighted sum of the pixel values:

    .. math::
        S &= \frac{\sum W(x,y) I(x,y)}{\sum W(x,y)} \\
        N^2 = Var(S) &= \frac{\sum W(x,y)^2 Var(I(x,y))}{(\sum W(x,y))^2}

    and assume that :math:`Var(I(x,y))` is constant, :math:`V \equiv Var(I(x,y))`.
    We then assume that we are using a matched filter for :math:`W`, so :math:`W(x,y) = I(x,y)`.
    Then a few things cancel and we find that

    .. math::
        (S/N)^2 = \frac{\sum I(x,y)^2}{V}

    and therefore, for a given :math:`I(x,y)` and :math:`S/N` (aka ``snr``)

    .. math::
        V = \frac{\sum I(x,y)^2}{(S/N)^2}

    .. note::
        For noise models such as `PoissonNoise` and `CCDNoise`, the assumption of constant
        :math:`Var(I(x,y))` is only approximate, since the flux of the object adds to the Poisson
        noise in those pixels.  Thus, the real :math:`S/N` on the final image will be slightly
        lower than the target ``snr`` value, and this effect will be larger for brighter objects.

    This function relies on `BaseNoise.getVariance` to determine how much variance the noise model
    will add.  Thus, it will not work for noise models that do not have a well-defined variance,
    such as `VariableGaussianNoise`.

    Parameters:
        noise:          The noise (`BaseNoise`) model to use.
        snr:            The desired signal-to-noise after the noise is applied.
        preserve_flux:  Whether to preserve the flux of the object (``True``) or the variance of
                        the noise model (``False``) to achieve the desired snr. [default: False]

    Returns:
        the variance of the noise that was applied to the image.
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

Image.addNoise = addNoise
Image.addNoiseSNR = addNoiseSNR


class BaseNoise(object):
    """Base class for all noise classes.

    This class should not be constructed directly.  Rather, you should instantiate one of the
    subclasses:

    * `GaussianNoise`
    * `PoissonNoise`
    * `CCDNoise`
    * `VariableGaussianNoise`
    * `DeviateNoise`

    which define what kind of noise you want to implement.  This base class mostly just serves as
    a way to check if an object is a valid noise object with::

        >>> isinstance(noise, galsim.BaseNoise)
    """
    def __init__(self, rng=None):
        from .random import BaseDeviate
        if rng is None:
            self._rng = BaseDeviate()
        else:
            if not isinstance(rng, BaseDeviate):
                raise TypeError("rng must be a galsim.BaseDeviate instance.")
            self._rng = rng

    @property
    def rng(self):
        """The `BaseDeviate` of this noise object.
        """
        return self._rng

    def getVariance(self):
        """Get variance in current noise model."""
        return self._getVariance()

    def _getVariance(self):
        raise NotImplementedError("Cannot call getVariance on a pure BaseNoise object")

    def withVariance(self, variance):
        """Return a new noise object (of the same type as the current one) with the specified
        variance.

        Parameters:
            variance:   The desired variance in the noise.

        Returns:
            a new Noise object with the given variance.
        """
        return self._withVariance(variance)

    def _withVariance(self, variance):
        raise NotImplementedError("Cannot call withVariance on a pure BaseNoise object")

    def withScaledVariance(self, variance_ratio):
        """Return a new noise object with the variance scaled up by the specified factor.

        This is equivalent to noise * variance_ratio.

        Parameters:
            variance_ratio: The factor by which to scale the variance of the correlation
                            function profile.

        Returns:
            a new Noise object whose variance has been scaled by the given amount.
        """
        return self._withScaledVariance(variance_ratio)

    def _withScaledVariance(self, variance_ratio):
        raise NotImplementedError("Cannot call withScaledVariance on a pure BaseNoise object")

    def __mul__(self, variance_ratio):
        """Multiply the variance of the noise by ``variance_ratio``.

        Parameters:
            variance_ratio: The factor by which to scale the variance of the correlation
                            function profile.

        Returns:
            a new Noise object whose variance has been scaled by the given amount.
        """
        return self.withScaledVariance(variance_ratio)

    def __div__(self, variance_ratio):
        """Equivalent to self * (1/variance_ratio)"""
        return self.withScaledVariance(1./variance_ratio)

    __rmul__ = __mul__
    __truediv__ = __div__

    def applyTo(self, image):
        """Add noise to an input `Image`.

        e.g.::

            >>> noise.applyTo(image)

        On output the `Image` instance ``image`` will have been given additional noise according
        to the current noise model.

        Note: This is equivalent to the alternate syntax::

            >>> image.addNoise(noise)

        which may be more convenient or clearer.
        """
        if not isinstance(image, Image):
            raise TypeError("Provided image must be a galsim.Image")
        return self._applyTo(image)

    def _applyTo(self, image):
        raise NotImplementedError("Cannot call applyTo on a pure BaseNoise object")

    def __eq__(self, other):
        # Quick and dirty.  Just check reprs are equal.
        return self is other or repr(self) == repr(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None


class GaussianNoise(BaseNoise):
    """Class implementing simple Gaussian noise.

    This is a simple noise model where each pixel in the image gets Gaussian noise with a
    given ``sigma``.

    Example:

    The following will add Gaussian noise to every element of an image::

        >>> gaussian_noise = galsim.GaussianNoise(rng, sigma=1.0)
        >>> image.addNoise(gaussian_noise)

    Parameters:
        rng:        A `BaseDeviate` instance to use for generating the random numbers.
        sigma:      The rms noise on each pixel. [default: 1.]

    Attributes:
        rng:        The internal random number generator (read-only)
        sigma:      The value of the constructor parameter sigma (read-only)
    """
    def __init__(self, rng=None, sigma=1.):
        from .random import GaussianDeviate
        BaseNoise.__init__(self, rng)
        self._sigma = sigma
        self._gd = GaussianDeviate(self.rng, sigma=sigma)

    @property
    def sigma(self):
        """The input sigma value.
        """
        return self._sigma

    def _applyTo(self, image):
        self._gd.clearCache()
        noise_array = np.empty(np.prod(image.array.shape), dtype=float)
        self._gd.generate(noise_array)
        image.array[:,:] += noise_array.reshape(image.array.shape).astype(image.dtype)

    def _getVariance(self):
        return self.sigma**2

    def _withVariance(self, variance):
        return GaussianNoise(self.rng, math.sqrt(variance))

    def _withScaledVariance(self, variance_ratio):
        return GaussianNoise(self.rng, self.sigma * math.sqrt(variance_ratio))

    def copy(self, rng=None):
        """Returns a copy of the Gaussian noise model.

        By default, the copy will share the `BaseDeviate` random number generator with the parent
        instance.  However, you can provide a new rng to use in the copy if you want with::

            >>> noise_copy = noise.copy(rng=new_rng)
        """
        if rng is None: rng = self.rng
        return GaussianNoise(rng, self.sigma)

    def __repr__(self):
        return 'galsim.GaussianNoise(rng=%r, sigma=%r)'%(self.rng, self.sigma)

    def __str__(self):
        return 'galsim.GaussianNoise(sigma=%s)'%(self.sigma)


class PoissonNoise(BaseNoise):
    """Class implementing Poisson noise according to the flux (and sky level) present in the image.

    The PoissonNoise class encapsulates a simple version of the noise model of a normal CCD image
    where each pixel has Poisson noise corresponding to the number of electrons in each pixel
    (including an optional extra sky level).

    Note that if the image to which you are adding noise already has a sky level on it,
    then you should not provide the sky level here as well.  The sky level here corresponds
    to a level that is taken to be already subtracted from the image, but that originally
    contributed to the addition of Poisson noise.

    Example:

    The following will add Poisson noise to every element of an image::

        >>> poisson_noise = galsim.PoissonNoise(rng, sky_level=0.)
        >>> image.addNoise(poisson_noise)

    Parameters:
        rng:        A `BaseDeviate` instance to use for generating the random numbers.
        sky_level:  The sky level in electrons per pixel that was originally in the input image,
                    but which is taken to have already been subtracted off. [default: 0.]

    Attributes:
        rng:        The internal random number generator (read-only)
        sky_level:  The value of the constructor parameter sky_level (read-only)
    """
    def __init__(self, rng=None, sky_level=0.):
        from .random import PoissonDeviate
        BaseNoise.__init__(self, rng)
        self._sky_level = sky_level
        self._pd = PoissonDeviate(self.rng)

    @property
    def sky_level(self):
        """The input sky_level.
        """
        return self._sky_level

    def _applyTo(self, image):
        noise_array = np.empty(np.prod(image.array.shape), dtype=float)
        noise_array.reshape(image.array.shape)[:,:] = image.array

        # Minor subtlety for integer images.  It's a bit more consistent to convert to an
        # integer with the sky still added and then subtract off the sky.  But this isn't quite
        # right if the sky has a fractional part.  So only subtract off the integer part of the
        # sky at the end.  For float images, you get the same answer either way, so it doesn't
        # matter.
        frac_sky = self.sky_level - image.dtype(self.sky_level)
        int_sky = self.sky_level - frac_sky

        if self.sky_level != 0.:
            noise_array += self.sky_level
        # Make sure no negative values
        noise_array = noise_array.clip(0.)
        # The noise_image now has the expectation values for each pixel with the sky added.
        self._pd.generate_from_expectation(noise_array)
        # Subtract off the sky, since we don't want it in the final image.
        if frac_sky != 0.:
            noise_array -= frac_sky
        # Noise array is now the correct value for each pixel.
        np.copyto(image.array, noise_array.reshape(image.array.shape), casting='unsafe')
        if int_sky != 0.:
            image -= int_sky

    def _getVariance(self):
        return self.sky_level

    def _withVariance(self, variance):
        return PoissonNoise(self.rng, variance)

    def _withScaledVariance(self, variance_ratio):
        return PoissonNoise(self.rng, self.sky_level * variance_ratio)

    def copy(self, rng=None):
        """Returns a copy of the Poisson noise model.

        By default, the copy will share the `BaseDeviate` random number generator with the parent
        instance.  However, you can provide a new rng to use in the copy if you want with::

            >>> noise_copy = noise.copy(rng=new_rng)
        """
        if rng is None: rng = self.rng
        return PoissonNoise(rng, self.sky_level)

    def __repr__(self):
        return 'galsim.PoissonNoise(rng=%r, sky_level=%r)'%(self.rng, self.sky_level)

    def __str__(self):
        return 'galsim.PoissonNoise(sky_level=%s)'%(self.sky_level)


class CCDNoise(BaseNoise):
    """Class implementing a basic CCD noise model.

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

    If you are manually applying the quantum efficiency of the detector (e-/photon), then this
    would normally be applied before the noise.  However, it is also fine to fold in the quantum
    efficiency into the gain to give units photons/ADU.  Either way is acceptable.  Just make sure
    you give the read noise in photons as well in this case.

    Example:

    The following will add CCD noise to every element of an image::

        >>> ccd_noise = galsim.CCDNoise(rng, sky_level=0., gain=1., read_noise=0.)
        >>> image.addNoise(ccd_noise)

    Parameters:
        rng:            A `BaseDeviate` instance to use for generating the random numbers.
        sky_level:      The sky level in ADU per pixel that was originally in the input image,
                        but which is taken to have already been subtracted off. [default: 0.]
        gain:           The gain for each pixel in electrons per ADU; setting ``gain<=0`` will shut
                        off the Poisson noise, and the Gaussian rms will take the value
                        ``read_noise`` as being in units of ADU rather than electrons. [default: 1.]
        read_noise:     The read noise on each pixel in electrons (gain > 0.) or ADU (gain <= 0.).
                        Setting ``read_noise=0``. will shut off the Gaussian noise. [default: 0.]

    Attributes:
        rng:            The internal random number generator (read-only)
        sky_level:      The value of the constructor parameter sky_level (read-only)
        gain:           The value of the constructor parameter gain (read-only)
        read_noise:     The value of the constructor parameter read_noise (read-only)
    """
    def __init__(self, rng=None, sky_level=0., gain=1., read_noise=0.):
        from .random import PoissonDeviate, GaussianDeviate
        BaseNoise.__init__(self, rng)
        self._sky_level = float(sky_level)
        self._gain = float(gain)
        self._read_noise = float(read_noise)
        self._pd = PoissonDeviate(self.rng)
        if gain > 0.:
            self._gd = GaussianDeviate(self.rng, sigma=self.read_noise / self.gain)
        else:
            self._gd = GaussianDeviate(self.rng, sigma=self.read_noise)

    @property
    def sky_level(self):
        """The input sky_level.
        """
        return self._sky_level

    @property
    def gain(self):
        """The input gain.
        """
        return self._gain

    @property
    def read_noise(self):
        """The input read_noise.
        """
        return self._read_noise

    def _applyTo(self, image):
        noise_array = np.empty(np.prod(image.array.shape), dtype=float)
        noise_array.reshape(image.array.shape)[:,:] = image.array

        # cf. PoissonNoise._applyTo function
        frac_sky = self.sky_level - image.dtype(self.sky_level)  # 0 if dtype = float
        int_sky = self.sky_level - frac_sky

        if self.sky_level != 0.:
            noise_array += self.sky_level

        # First add the poisson noise from the signal + sky:
        if self.gain > 0.:
            noise_array *= self.gain  # convert to electrons
            noise_array = noise_array.clip(0.)
            # The noise_image now has the expectation values for each pixel with the sky added.
            self._pd.generate_from_expectation(noise_array)
            # Subtract off the sky, since we don't want it in the final image.
            noise_array /= self.gain

        # Now add the read noise:
        if self.read_noise > 0.:
            self._gd.clearCache()
            self._gd.add_generate(noise_array)

        if frac_sky != 0.:
            noise_array -= frac_sky
        np.copyto(image.array, noise_array.reshape(image.array.shape), casting='unsafe')
        if int_sky != 0.:
            image -= int_sky

    def _getVariance(self):
        if self.gain > 0.:
            return self.sky_level/self.gain + (self.read_noise / self.gain)**2
        else:
            return self.read_noise**2

    def _withVariance(self, variance):
        current_var = self._getVariance()
        if current_var > 0.:
            return self._withScaledVariance(variance / current_var)
        else:
            return CCDNoise(self.rng, sky_level=variance)

    def _withScaledVariance(self, variance_ratio):
        return CCDNoise(self.rng, gain=self.gain,
                        sky_level = self.sky_level * variance_ratio,
                        read_noise = self.read_noise * math.sqrt(variance_ratio))

    def copy(self, rng=None):
        """Returns a copy of the CCD noise model.

        By default, the copy will share the `BaseDeviate` random number generator with the parent
        instance.  However, you can provide a new rng to use in the copy if you want with::

            >>> noise_copy = noise.copy(rng=new_rng)
        """
        if rng is None: rng = self.rng
        return CCDNoise(rng, self.sky_level, self.gain, self.read_noise)

    def __repr__(self):
        return 'galsim.CCDNoise(rng=%r, sky_level=%r, gain=%r, read_noise=%r)'%(
                self.rng, self.sky_level, self.gain, self.read_noise)

    def __str__(self):
        return 'galsim.CCDNoise(sky_level=%r, gain=%r, read_noise=%r)'%(
                self.sky_level, self.gain, self.read_noise)


class DeviateNoise(BaseNoise):
    """Class implementing noise with an arbitrary `BaseDeviate` object.

    The DeviateNoise class provides a way to treat an arbitrary deviate as the noise model for
    each pixel in an image.

    The following will add noise according to a given random deviate to every element of an image::

        >>> dev_noise = galsim.DeviateNoise(dev)
        >>> image.addNoise(dev_noise)

    Parameters:
        dev:    A `BaseDeviate` subclass to use as the noise deviate for each pixel.

    Attributes:
        rng:    The internal random number generator (read-only)
    """
    def __init__(self, dev):
        BaseNoise.__init__(self, dev)

    def _applyTo(self, image):
        noise_array = np.empty(np.prod(image.array.shape), dtype=float)
        self._rng.generate(noise_array)
        image.array[:,:] += noise_array.reshape(image.array.shape).astype(image.dtype)

    def _getVariance(self):
        raise GalSimError("No single variance value for DeviateNoise")

    def _withVariance(self, variance):
        raise GalSimError("Changing the variance is not allowed for DeviateNoise")

    def _withScaledVariance(self, variance):
        raise GalSimError("Changing the variance is not allowed for DeviateNoise")

    def copy(self, rng=None):
        """Returns a copy of the `DeviateNoise` instance.

        By default, the copy will share the `BaseDeviate` random number generator with the parent
        instance.  However, you can provide a new rng to use in the copy if you want with::

            >>> noise_copy = noise.copy(rng=new_rng)
        """
        if rng is None:
            dev = self.rng
        else:
            # Slightly different this time, since we want to make sure that we keep the same
            # kind of deviate, but just reset it to follow the given rng.
            dev = self.rng.duplicate()
            dev.reset(rng)
        return DeviateNoise(dev)

    def __repr__(self):
        return 'galsim.DeviateNoise(dev=%r)'%self.rng

    def __str__(self):
        return 'galsim.DeviateNoise(dev=%s)'%self.rng


class VariableGaussianNoise(BaseNoise):
    """
    Class implementing Gaussian noise that has a different variance in each pixel.

    The following will add variable Gaussian noise to every element of an image::

        >>> variable_noise = galsim.VariableGaussianNoise(rng, var_image)
        >>> image.addNoise(variable_noise)

    Parameters:
        rng:        A `BaseDeviate` instance to use for generating the random numbers.
        var_image:  The variance of the noise to apply to each pixel.  This image must be the
                    same shape as the image for which you eventually call addNoise().

    Attributes:
        rng:        The internal random number generator (read-only)
        var_image:  The value of the constructor parameter var_image (read-only)
    """
    def __init__(self, rng, var_image):
        from .random import GaussianDeviate
        BaseNoise.__init__(self, rng)
        self._gd = GaussianDeviate(rng)

        # Make sure var_image is an ImageD, converting dtype if necessary
        self._var_image = ImageD(var_image)

    @property
    def var_image(self):
        """The input var_image.
        """
        return self._var_image

    # Repeat this here, since we want to add an extra sanity check, which should go in the
    # non-underscore version.
    @doc_inherit
    def applyTo(self, image):
        if not isinstance(image, Image):
            raise TypeError("Provided image must be a galsim.Image")
        if image.array.shape != self.var_image.array.shape:
            raise GalSimIncompatibleValuesError(
                "Provided image shape does not match the shape of var_image",
                image=image, var_image=self.var_image)
        return self._applyTo(image)

    def _applyTo(self, image):
        noise_array = self.var_image.array.flatten()  # NB. Makes a copy! (which is what we want)
        self._gd.generate_from_variance(noise_array)
        image.array[:,:] += noise_array.reshape(image.array.shape).astype(image.dtype)

    def copy(self, rng=None):
        """Returns a copy of the variable Gaussian noise model.

        By default, the copy will share the `BaseDeviate` random number generator with the parent
        instance.  However, you can provide a new rng to use in the copy if you want with::

            >>> noise_copy = noise.copy(rng=new_rng)
        """
        if rng is None: rng = self.rng
        return VariableGaussianNoise(rng, self.var_image)

    def _getVariance(self):
        raise GalSimError("No single variance value for VariableGaussianNoise")

    def _withVariance(self, variance):
        raise GalSimError("Changing the variance is not allowed for VariableGaussianNoise")

    def _withScaledVariance(self, variance):
        # This one isn't undefined like withVariance, but it's inefficient.  Better to
        # scale the values in the image before constructing VariableGaussianNoise.
        raise GalSimError("Changing the variance is not allowed for VariableGaussianNoise")

    def __repr__(self):
        return 'galsim.VariableGaussianNoise(rng=%r, var_image%r)'%(self.rng, self.var_image)

    def __str__(self):
        return 'galsim.VariableGaussianNoise(var_image%s)'%(self.var_image)
