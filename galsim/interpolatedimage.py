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

from .gsobject import GSObject
from .gsparams import GSParams
from .image import Image
from .bounds import _BoundsI
from .position import PositionD, _PositionD
from .interpolant import Quintic, Interpolant, SincInterpolant
from .utilities import convert_interpolant, lazy_property, doc_inherit, basestring
from .random import BaseDeviate
from . import _galsim
from . import fits
from .errors import GalSimError, GalSimRangeError, GalSimValueError, GalSimUndefinedBoundsError
from .errors import GalSimIncompatibleValuesError, convert_cpp_errors, galsim_warn

class InterpolatedImage(GSObject):
    """A class describing non-parametric profiles specified using an `Image`, which can be
    interpolated for the purpose of carrying out transformations.

    The InterpolatedImage class is useful if you have a non-parametric description of an object as
    an `Image`, that you wish to manipulate / transform using `GSObject` methods such as
    `GSObject.shear`, `GSObject.magnify`, `GSObject.shift`, etc.  Note that when convolving an
    InterpolatedImage, the use of real-space convolution is not recommended, since it is typically
    a great deal slower than Fourier-space convolution for this kind of object.

    There are three options for determining the flux of the profile.

    1. You can simply specify a ``flux`` value explicitly.
    2. If you set ``normalization = 'flux'``, the flux will be taken as the sum of the pixel values
       in the input image.  This corresponds to an image that was drawn with
       ``drawImage(method='no_pixel')``.  This is the default if flux is not given.
    3. If you set ``normalization = 'sb'``, the pixel values are treated as samples of the surface
       brightness profile at each location.  This corresponds to an image drawn with
       ``drawImage(method='sb')``.  The flux is then the sum of the pixels in the input image
       multiplied by the pixel area.

    You can also use images that were drawn with one of the pixel-integrating methods ('auto',
    'fft', or 'real_space'); however, the resulting profile will not correspond to the one
    that was used to call `GSObject.drawImage`.  The integration over the pixel is equivalent to
    convolving the original profile by a `Pixel` and then drawing with ``method='no_pixel'``.  So
    if you use such an image with InterpolatedImage, the resulting profile will include the `Pixel`
    convolution already.  As such, if you use it as a PSF for example, then the final objects
    convolved by this PSF will already include the pixel convolution, so you should draw them
    using ``method='no_pixel'``.

    If the input `Image` has a ``scale`` or ``wcs`` associated with it, then there is no need to
    specify one as a parameter here.  But if one is provided, that will override any ``scale`` or
    ``wcs`` that is native to the `Image`.

    The user may optionally specify an interpolant, ``x_interpolant``, for real-space manipulations
    (e.g., shearing, resampling).  If none is specified, then by default, a `Quintic` interpolant is
    used.  The user may also choose to specify two quantities that can affect the Fourier space
    convolution: the k-space interpolant (``k_interpolant``) and the amount of padding to include
    around the original images (``pad_factor``).  The default values for ``x_interpolant``,
    ``k_interpolant``, and ``pad_factor`` were chosen based on the tests of branch #389 to reach
    good accuracy without being excessively slow.  Users should be particularly wary about changing
    ``k_interpolant`` and ``pad_factor`` from the defaults without careful testing.  The user is
    given complete freedom to choose interpolants and pad factors, and no warnings are raised when
    the code is modified to choose some combination that is known to give significant error.  More
    details can be found in http://arxiv.org/abs/1401.2636, especially table 1, and in comment
    https://github.com/GalSim-developers/GalSim/issues/389#issuecomment-26166621 and the following
    comments.

    The user can choose to pad the image with a noise profile if desired.  To do so, specify
    the target size for the noise padding in ``noise_pad_size``, and specify the kind of noise
    to use in ``noise_pad``.  The ``noise_pad`` option may be a Gaussian random noise of some
    variance, or a Gaussian but correlated noise field that is specified either as a
    `BaseCorrelatedNoise` instance, an `Image` (from which a correlated noise model is derived), or
    a string (interpreted as a filename of an image to use for deriving a `CorrelatedNoise`).
    The user can also pass in a random number generator to be used for noise generation.  Finally,
    the user can pass in a ``pad_image`` for deterministic image padding.

    By default, the InterpolatedImage recalculates the Fourier-space step and number of points to
    use for further manipulations, rather than using the most conservative possibility.  For typical
    objects representing galaxies and PSFs this can easily make the difference between several
    seconds (conservative) and 0.04s (recalculated).  However, the user can turn off this option,
    and may especially wish to do so when using images that do not contain a high S/N object - e.g.,
    images of noise fields.


    Example::

        >>> interpolated_image = galsim.InterpolatedImage(
                image, x_interpolant=None, k_interpolant=None, normalization='flux', scale=None,
                wcs=None, flux=None, pad_factor=4., noise_pad_size=0, noise_pad=0., use_cache=True,
                pad_image=None, rng=None, calculate_stepk=True, calculate_maxk=True,
                use_true_center=True, offset=None, hdu=None)

    Initializes ``interpolated_image`` as an InterpolatedImage instance.

    For comparison of the case of padding with noise or zero when the image itself includes noise,
    compare ``im1`` and ``im2`` from the following code snippet (which can be executed from the
    examples/ directory)::

        >>> image = galsim.fits.read('data/147246.0_150.416558_1.998697_masknoise.fits')
        >>> int_im1 = galsim.InterpolatedImage(image)
        >>> int_im2 = galsim.InterpolatedImage(image, noise_pad='data/blankimg.fits')
        >>> im1 = galsim.ImageF(1000,1000)
        >>> im2 = galsim.ImageF(1000,1000)
        >>> int_im1.drawImage(im1, method='no_pixel')
        >>> int_im2.drawImage(im2, method='no_pixel')

    Examination of these two images clearly shows how padding with a correlated noise field that is
    similar to the one in the real data leads to a more reasonable appearance for the result when
    re-drawn at a different size.

    Parameters:
        image:              The `Image` from which to construct the object.
                            This may be either an `Image` instance or a string indicating a fits
                            file from which to read the image.  In the latter case, the ``hdu``
                            kwarg can be used to specify a particular HDU in that file.
        x_interpolant:      Either an `Interpolant` instance or a string indicating which real-space
                            interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                            'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                            to use. [default: galsim.Quintic()]
        k_interpolant:      Either an `Interpolant` instance or a string indicating which k-space
                            interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                            'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                            to use.  We strongly recommend leaving this parameter at its default
                            value; see text above for details.  [default: galsim.Quintic()]
        normalization:      Two options for specifying the normalization of the input `Image`:

                            - "flux" or "f" means that the sum of the pixels is normalized
                              to be equal to the total flux.
                            - "surface brightness" or "sb" means that the pixels sample
                              the surface brightness distribution at each location.

                            This is overridden if you specify an explicit flux value.
                            [default: "flux"]
        scale:              If provided, use this as the pixel scale for the `Image`; this will
                            override the pixel scale stored by the provided `Image`, in any.
                            If ``scale`` is ``None``, then take the provided image's pixel scale.
                            [default: None]
        wcs:                If provided, use this as the wcs for the image.  At most one of
                            ``scale`` or ``wcs`` may be provided. [default: None]
        flux:               Optionally specify a total flux for the object, which overrides the
                            implied flux normalization from the `Image` itself. [default: None]
        pad_factor:         Factor by which to pad the `Image` with zeros.  We strongly recommend
                            leaving this parameter at its default value; see text above for
                            details.  [default: 4]
        noise_pad_size:     If provided, the image will be padded out to this size (in arcsec) with
                            the noise specified by ``noise_pad``. This is important if you are
                            planning to whiten the resulting image.  You want to make sure that the
                            noise-padded image is larger than the postage stamp onto which you are
                            drawing this object.  [default: None]
        noise_pad:          Noise properties to use when padding the original image with
                            noise.  This can be specified in several ways:

                            a) as a float, which is interpreted as being a variance to use when
                               padding with uncorrelated Gaussian noise;
                            b) as a `galsim.BaseCorrelatedNoise`, which contains information about
                               the desired noise power spectrum - any random number generator passed
                               to the ``rng`` keyword will take precedence over that carried in
                               an input `BaseCorrelatedNoise` instance;
                            c) as an `Image` of a noise field, which is used to calculate
                               the desired noise power spectrum; or
                            d) as a string which is interpreted as a filename containing an
                               example noise field with the proper noise power spectrum (as an
                               `Image` in the first HDU).

                            It is important to keep in mind that the calculation of the correlation
                            function that is internally stored within a `BaseCorrelatedNoise`
                            object is a non-negligible amount of overhead, so the recommended means
                            of specifying a correlated noise field for padding are (b) or (d).  In
                            the case of (d), if the same file is used repeatedly, then the
                            ``use_cache`` keyword (see below) can be used to prevent the need for
                            repeated `CorrelatedNoise` initializations.  [default: 0, i.e., pad
                            with zeros]
        use_cache:          Specify whether to cache ``noise_pad`` read in from a file to save
                            having to build a CorrelatedNoise object repeatedly from the same image.
                            [default: True]
        rng:                If padding by noise, the user can optionally supply the random noise
                            generator to use for drawing random numbers as ``rng`` (may be any kind
                            of `BaseDeviate` object).  Such a user-input random number generator
                            takes precedence over any stored within a user-input
                            `BaseCorrelatedNoise` instance (see the ``noise_pad`` parameter).
                            If ``rng=None``, one will be automatically created, using the time as a
                            seed. [default: None]
        pad_image:          `Image` to be used for deterministically padding the original image.
                            This can be specified in two ways:

                            - as an `Image`; or
                            - as a string which is interpreted as a filename containing an
                              image to use (in the first HDU).

                            The ``pad_image`` scale or wcs is ignored.  It uses the same scale or
                            wcs for both the ``image`` and the ``pad_image``.
                            The user should be careful to ensure that the image used for padding
                            has roughly zero mean.  The purpose of this keyword is to allow for a
                            more flexible representation of some noise field around an object; if
                            the user wishes to represent the sky level around an object, they
                            should do that after they have drawn the final image instead.
                            [default: None]
        calculate_stepk:    Specify whether to perform an internal determination of the extent of
                            the object being represented by the InterpolatedImage; often this is
                            useful in choosing an optimal value for the stepsize in the Fourier
                            space lookup table.
                            If you know a priori an appropriate maximum value for ``stepk``, then
                            you may also supply that here instead of a bool value, in which case
                            the ``stepk`` value is still calculated, but will not go above the
                            provided value.
                            [default: True]
        calculate_maxk:     Specify whether to perform an internal determination of the highest
                            spatial frequency needed to accurately render the object being
                            represented by the InterpolatedImage; often this is useful in choosing
                            an optimal value for the extent of the Fourier space lookup table.
                            If you know a priori an appropriate maximum value for ``maxk``, then
                            you may also supply that here instead of a bool value, in which case
                            the ``maxk`` value is still calculated, but will not go above the
                            provided value.
                            [default: True]
        use_true_center:    Similar to the same parameter in the `GSObject.drawImage` function,
                            this sets whether to use the true center of the provided image as the
                            center of the profile (if ``use_true_center=True``) or the nominal
                            center given by image.center (if ``use_true_center=False``)
                            [default: True]
        offset:             The location in the input image to use as the center of the profile.
                            This should be specified relative to the center of the input image
                            (either the true center if ``use_true_center=True``, or the nominal
                            center if ``use_true_center=False``).  [default: None]
        gsparams:           An optional `GSParams` argument. [default: None]
        hdu:                When reading in an `Image` from a file, this parameter can be used to
                            select a particular HDU in the file. [default: None]
    """
    _req_params = { 'image' : str }
    _opt_params = {
        'x_interpolant' : str ,
        'k_interpolant' : str ,
        'normalization' : str ,
        'scale' : float ,
        'flux' : float ,
        'pad_factor' : float ,
        'noise_pad_size' : float ,
        'noise_pad' : str ,
        'pad_image' : str ,
        'calculate_stepk' : bool ,
        'calculate_maxk' : bool ,
        'use_true_center' : bool ,
        'hdu' : int
    }
    _takes_rng = True
    _cache_noise_pad = {}

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, image, x_interpolant=None, k_interpolant=None, normalization='flux',
                 scale=None, wcs=None, flux=None, pad_factor=4., noise_pad_size=0, noise_pad=0.,
                 rng=None, pad_image=None, calculate_stepk=True, calculate_maxk=True,
                 use_cache=True, use_true_center=True, offset=None, gsparams=None,
                 _force_stepk=0., _force_maxk=0., hdu=None):

        from .wcs import BaseWCS, PixelScale
        from .random import BaseDeviate

        # If the "image" is not actually an image, try to read the image as a file.
        if isinstance(image, str):
            image = fits.read(image, hdu=hdu)
        elif not isinstance(image, Image):
            raise TypeError("Supplied image must be an Image or file name")

        # it must have well-defined bounds, otherwise seg fault in SBInterpolatedImage constructor
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Supplied image does not have bounds defined.")

        # check what normalization was specified for the image: is it an image of surface
        # brightness, or flux?
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise GalSimValueError("Invalid normalization requested.", normalization,
                                   ('flux', 'f', 'surface brightness', 'sb'))

        # Store the image as an attribute and make sure we don't change the original image
        # in anything we do here.  (e.g. set scale, etc.)
        self._image = image.view(dtype=np.float64, contiguous=True)
        self._image.setCenter(0,0)
        self._gsparams = GSParams.check(gsparams)

        # Set up the interpolants if none was provided by user, or check that the user-provided ones
        # are of a valid type
        if x_interpolant is None:
            self._x_interpolant = Quintic(gsparams=self._gsparams)
        else:
            self._x_interpolant = convert_interpolant(x_interpolant).withGSParams(self._gsparams)
        if k_interpolant is None:
            self._k_interpolant = Quintic(gsparams=self._gsparams)
        else:
            self._k_interpolant = convert_interpolant(k_interpolant).withGSParams(self._gsparams)

        # Set the wcs if necessary
        if scale is not None:
            if wcs is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both scale and wcs to InterpolatedImage", scale=scale, wcs=wcs)
            self._image.wcs = PixelScale(scale)
        elif wcs is not None:
            if not isinstance(wcs, BaseWCS):
                raise TypeError("wcs parameter is not a galsim.BaseWCS instance")
            self._image.wcs = wcs
        elif self._image.wcs is None:
            raise GalSimIncompatibleValuesError(
                "No information given with Image or keywords about pixel scale!",
                scale=scale, wcs=wcs, image=image)

        # Figure out the offset to apply based on the original image (not the padded one).
        # We will apply this below in _sbp.
        offset = self._parse_offset(offset)
        self._offset = self._adjust_offset(self._image.bounds, offset, None, use_true_center)

        im_cen = image.true_center if use_true_center else image.center
        self._wcs = self._image.wcs.local(image_pos=im_cen)

        # Build the fully padded real-space image according to the various pad options.
        self._buildRealImage(pad_factor, pad_image, noise_pad_size, noise_pad, rng, use_cache)
        self._image_flux = np.sum(self._image.array, dtype=np.float64)

        # I think the only things that will mess up if flux == 0 are the
        # calculateStepK and calculateMaxK functions, and rescaling the flux to some value.
        if (calculate_stepk or calculate_maxk or flux is not None) and self._image_flux == 0.:
            raise GalSimValueError("This input image has zero total flux. It does not define a "
                                   "valid surface brightness profile.", image)

        # Process the different options for flux, stepk, maxk
        self._flux = self._getFlux(flux, normalization)
        self._stepk = self._getStepK(calculate_stepk, _force_stepk)
        self._maxk = self._getMaxK(calculate_maxk, _force_maxk)

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        ret._x_interpolant = self._x_interpolant.withGSParams(ret._gsparams, **kwargs)
        ret._k_interpolant = self._k_interpolant.withGSParams(ret._gsparams, **kwargs)
        return ret

    @lazy_property
    def _sbp(self):
        min_scale = self._wcs._minScale()
        max_scale = self._wcs._maxScale()
        self._sbii = _galsim.SBInterpolatedImage(
                self._xim._image, self._image.bounds._b, self._pad_image.bounds._b,
                self._x_interpolant._i, self._k_interpolant._i,
                self._stepk*min_scale,
                self._maxk*max_scale,
                self.gsparams._gsp)

        self._sbp = self._sbii  # Temporary.  Will overwrite this with the return value.

        # Apply the offset
        prof = self
        if self._offset != _PositionD(0,0):
            # Opposite direction of what drawImage does.
            prof = prof._shift(-self._offset.x, -self._offset.y)

        # If the user specified a flux, then set to that flux value.
        if self._flux != self._image_flux:
            flux_ratio = self._flux / self._image_flux
        else:
            flux_ratio = 1.

        # Bring the profile from image coordinates into world coordinates
        # Note: offset needs to happen first before the transformation, so can't bundle it here.
        prof = self._wcs._profileToWorld(prof, flux_ratio, _PositionD(0,0))

        return prof._sbp

    @property
    def x_interpolant(self):
        """The real-space `Interpolant` for this profile.
        """
        return self._x_interpolant

    @property
    def k_interpolant(self):
        """The Fourier-space `Interpolant` for this profile.
        """
        return self._k_interpolant

    @property
    def image(self):
        """The underlying `Image` being interpolated.
        """
        return self._image

    def _buildRealImage(self, pad_factor, pad_image, noise_pad_size, noise_pad, rng, use_cache):
        # Check that given pad_image is valid:
        if pad_image is not None:
            if isinstance(pad_image, basestring):
                pad_image = fits.read(pad_image).view(dtype=np.float64)
            elif isinstance(pad_image, Image):
                pad_image = pad_image.view(dtype=np.float64, contiguous=True)
            else:
                raise TypeError("Supplied pad_image must be an Image.", pad_image)

        if pad_factor <= 0.:
            raise GalSimRangeError("Invalid pad_factor <= 0 in InterpolatedImage", pad_factor, 0.)

        # Convert noise_pad_size from arcsec to pixels according to the local wcs.
        # Use the minimum scale, since we want to make sure noise_pad_size is
        # as large as we need in any direction.
        if noise_pad_size:
            if noise_pad_size < 0:
                raise GalSimValueError("noise_pad_size may not be negative", noise_pad_size)
            if not noise_pad:
                raise GalSimIncompatibleValuesError(
                        "Must provide noise_pad if noise_pad_size > 0",
                        noise_pad=noise_pad, noise_pad_size=noise_pad_size)
            noise_pad_size = int(math.ceil(noise_pad_size / self._wcs._minScale()))
            noise_pad_size = Image.good_fft_size(noise_pad_size)
        else:
            if noise_pad:
                raise GalSimIncompatibleValuesError(
                        "Must provide noise_pad_size if noise_pad != 0",
                        noise_pad=noise_pad, noise_pad_size=noise_pad_size)

        # The size of the final padded image is the largest of the various size specifications
        pad_size = max(self._image.array.shape)
        if pad_factor > 1.:
            pad_size = int(math.ceil(pad_factor * pad_size))
        if noise_pad_size:
            pad_size = max(pad_size, noise_pad_size)
        if pad_image:
            pad_image.setCenter(0,0)
            pad_size = max(pad_size, *pad_image.array.shape)
        # And round up to a good fft size
        pad_size = Image.good_fft_size(pad_size)

        self._xim = Image(pad_size, pad_size, dtype=np.float64, wcs=self._wcs)
        self._xim.setCenter(0,0)

        # If requested, fill (some of) this image with noise padding.
        nz_bounds = self._image.bounds
        if noise_pad:
            # This is a bit involved, so pass this off to another helper function.
            b = self._buildNoisePadImage(noise_pad_size, noise_pad, rng, use_cache)
            nz_bounds += b

        # The the user gives us a pad image to use, fill the relevant portion with that.
        if pad_image:
            #assert self._xim.bounds.includes(pad_image.bounds)
            self._xim[pad_image.bounds] = pad_image
            nz_bounds += pad_image.bounds

        # Now place the given image in the center of the padding image:
        #assert self._xim.bounds.includes(self._image.bounds)
        self._xim[self._image.bounds] = self._image
        self._xim.wcs = self._wcs

        # And update the _image to be that portion of the full real image rather than the
        # input image.
        self._image = self._xim[self._image.bounds]

        # These next two allow for easy pickling/repring.  We don't need to serialize all the
        # zeros around the edge.  But we do need to keep any non-zero padding as a pad_image.
        self._pad_image = self._xim[nz_bounds]
        #self._pad_factor = (max(self._xim.array.shape)-1.e-6) / max(self._image.array.shape)
        self._pad_factor = pad_factor

    def _buildNoisePadImage(self, noise_pad_size, noise_pad, rng, use_cache):
        """A helper function that builds the ``pad_image`` from the given ``noise_pad``
        specification.
        """
        from .random import BaseDeviate
        from .noise import GaussianNoise
        from .correlatednoise import BaseCorrelatedNoise, CorrelatedNoise

        # Make sure we make rng a BaseDeviate if rng is None
        rng1 = BaseDeviate(rng)

        # Figure out what kind of noise to apply to the image
        try:
            noise_pad = float(noise_pad)
        except (TypeError, ValueError):
            if isinstance(noise_pad, BaseCorrelatedNoise):
                noise = noise_pad.copy(rng=rng1)
            elif isinstance(noise_pad, Image):
                noise = CorrelatedNoise(noise_pad, rng1)
            elif use_cache and noise_pad in InterpolatedImage._cache_noise_pad:
                noise = InterpolatedImage._cache_noise_pad[noise_pad]
                if rng:
                    # Make sure that we are using a specified RNG by resetting that in this cached
                    # CorrelatedNoise instance, otherwise preserve the cached RNG
                    noise = noise.copy(rng=rng1)
            elif isinstance(noise_pad, basestring):
                noise = CorrelatedNoise(fits.read(noise_pad), rng1)
                if use_cache:
                    InterpolatedImage._cache_noise_pad[noise_pad] = noise
            else:
                raise GalSimValueError(
                    "Input noise_pad must be a float/int, a CorrelatedNoise, Image, or filename "
                    "containing an image to use to make a CorrelatedNoise.", noise_pad)

        else:
            if noise_pad < 0.:
                raise GalSimRangeError("Noise variance may not be negative.", noise_pad, 0.)
            noise = GaussianNoise(rng1, sigma = np.sqrt(noise_pad))

        # Find the portion of xim to fill with noise.
        # It's allowed for the noise padding to not cover the whole pad image
        half_size = noise_pad_size // 2
        b = _BoundsI(-half_size, -half_size + noise_pad_size-1,
                     -half_size, -half_size + noise_pad_size-1)
        #assert self._xim.bounds.includes(b)
        noise_image = self._xim[b]
        # Add the noise
        noise_image.addNoise(noise)
        return b

    def _getFlux(self, flux, normalization):
        # If the user specified a surface brightness normalization for the input Image, then
        # need to rescale flux by the pixel area to get proper normalization.
        if flux is None:
            flux = self._image_flux
            if normalization.lower() in ('surface brightness','sb'):
                flux *= self._wcs.pixelArea()
        return flux

    def _getStepK(self, calculate_stepk, _force_stepk):
        # GalSim cannot automatically know what stepK and maxK are appropriate for the
        # input image.  So it is usually worth it to do a manual calculation (below).
        #
        # However, there is also a hidden option to force it to use specific values of stepK and
        # maxK (caveat user!).  The values of _force_stepk and _force_maxk should be provided in
        # terms of physical scale, e.g., for images that have a scale length of 0.1 arcsec, the
        # stepK and maxK should be provided in units of 1/arcsec.  Then we convert to the 1/pixel
        # units required by the C++ layer below.  Also note that profile recentering for even-sized
        # images (see the ._adjust_offset step below) leads to automatic reduction of stepK slightly
        # below what is provided here, while maxK is preserved.
        if _force_stepk > 0.:
            return _force_stepk
        elif calculate_stepk:
            if calculate_stepk is True:
                im = self._image
            else:
                # If not a bool, then value is max_stepk
                R = int(math.ceil(math.pi / calculate_stepk))
                b = _BoundsI(-R, R, -R, R)
                b = self._image.bounds & b
                im = self._image[b]
            thresh = (1.-self.gsparams.folding_threshold) * self._image_flux
            R = _galsim.CalculateSizeContainingFlux(self._image._image, thresh)
        else:
            R = np.max(self._image.array.shape) / 2. - 0.5
        return self._getSimpleStepK(R)

    def _getSimpleStepK(self, R):
        min_scale = self._wcs._minScale()
        # Add xInterp range in quadrature just like convolution:
        R2 = self._x_interpolant.xrange
        R = math.hypot(R, R2)
        stepk = math.pi / (R * min_scale)
        return stepk

    def _getMaxK(self, calculate_maxk, _force_maxk):
        max_scale = self._wcs._maxScale()
        if _force_maxk > 0.:
            return _force_maxk
        elif calculate_maxk:
            self._maxk = 0.
            self._sbp
            if calculate_maxk is True:
                self._sbii.calculateMaxK(0.)
            else:
                # If not a bool, then value is max_maxk
                self._sbii.calculateMaxK(float(calculate_maxk))
            self.__dict__.pop('_sbp')  # Need to remake it.
            return self._sbii.maxK() / max_scale
        else:
            return self._x_interpolant.krange / max_scale

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, InterpolatedImage) and
                 self._xim == other._xim and
                 self.x_interpolant == other.x_interpolant and
                 self.k_interpolant == other.k_interpolant and
                 self.flux == other.flux and
                 self._offset == other._offset and
                 self.gsparams == other.gsparams and
                 self._stepk == other._stepk and
                 self._maxk == other._maxk))

    def __hash__(self):
        # Definitely want to cache this, since the size of the image could be large.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.InterpolatedImage", self.x_interpolant, self.k_interpolant))
            self._hash ^= hash((self.flux, self._stepk, self._maxk, self._pad_factor))
            self._hash ^= hash((self._xim.bounds, self._image.bounds, self._pad_image.bounds))
            # A common offset is 0.5,0.5, and *sometimes* this produces the same hash as 0,0
            # (which is also common).  I guess because they are only different in 2 bits.
            # This mucking of the numbers seems to help make the hash more reliably different for
            # these two cases.  Note: "sometiems" because of this:
            # https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
            self._hash ^= hash((self._offset.x * 1.234, self._offset.y * 0.23424))
            self._hash ^= hash(self._gsparams)
            self._hash ^= hash(self._xim.wcs)
            # Just hash the diagonal.  Much faster, and usually is unique enough.
            # (Let python handle collisions as needed if multiple similar IIs are used as keys.)
            self._hash ^= hash(tuple(np.diag(self._pad_image.array)))
        return self._hash

    def __repr__(self):
        s = 'galsim.InterpolatedImage(%r, %r, %r'%(
                self._image, self.x_interpolant, self.k_interpolant)
        # Most things we keep even if not required, but the pad_image is large, so skip it
        # if it's really just the same as the main image.
        if self._pad_image.bounds != self._image.bounds:
            s += ', pad_image=%r'%(self._pad_image)
        s += ', pad_factor=%f, flux=%r, offset=%r'%(self._pad_factor, self.flux, self._offset)
        s += ', use_true_center=False, gsparams=%r, _force_stepk=%r, _force_maxk=%r)'%(
                self.gsparams, self._stepk, self._maxk)
        return s

    def __str__(self): return 'galsim.InterpolatedImage(image=%s, flux=%s)'%(self.image, self.flux)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbii',None)
        d.pop('_sbp',None)
        # Only pickle _pad_image.  Not _xim or _image
        d['_xim_bounds'] = self._xim.bounds
        d['_image_bounds'] = self._image.bounds
        d.pop('_xim',None)
        d.pop('_image',None)
        return d

    def __setstate__(self, d):
        xim_bounds = d.pop('_xim_bounds')
        image_bounds = d.pop('_image_bounds')
        self.__dict__ = d
        if self._pad_image.bounds == xim_bounds:
            self._xim = self._pad_image
        else:
            self._xim = Image(xim_bounds, wcs=self._wcs, dtype=np.float64)
            self._xim[self._pad_image.bounds] = self._pad_image
        self._image = self._xim[image_bounds]

    @property
    def _centroid(self):
        return PositionD(self._sbp.centroid())

    @property
    def _positive_flux(self):
        return self._sbp.getPositiveFlux()

    @property
    def _negative_flux(self):
        return self._sbp.getNegativeFlux()

    @lazy_property
    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

    def _xValue(self, pos):
        return self._sbp.xValue(pos._p)

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _shoot(self, photons, rng):
        with convert_cpp_errors():
            self._sbp.shoot(photons._pa, rng._rng)
        photons.flux *= self._flux_per_photon

    def _drawReal(self, image, jac=None, offset=(0.,0.), flux_scaling=1.):
        dx,dy = offset
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.draw(image._image, image.scale, _jac, dx, dy, flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.drawK(image._image, image.scale, _jac)


def _InterpolatedImage(image, x_interpolant=Quintic(), k_interpolant=Quintic(),
                       use_true_center=True, offset=None, gsparams=None,
                       force_stepk=0., force_maxk=0.):
    """Approximately equivalent to `InterpolatedImage`, but with fewer options and no sanity checks.

    Some notable reductions in functionality relative to `InterpolatedImage`:

    1. There are no padding options. The image must be provided with all padding already applied.
    2. The stepk and maxk values will not be calculated.  If you want to use values for these other
       than the default, you may provide them as force_stepk and force_maxk.  Otherwise
       stepk ~= 2pi / image_size and maxk ~= x_interpolant.krange() / pixel_scale.
    3. The flux is just the flux of the image.  It cannot be rescaled to a different flux value.
    4. The input image must have a defined wcs.

    Parameters:
        image:              The `Image` from which to construct the object.
        x_interpolant:      An `Interpolant` instance for real-space interpolation
                            [default: Quintic]
        k_interpolant:      An `Interpolant` instance for k-space interpolation [default: Quintic]
        use_true_center:    Whether to use the true center of the provided image as the center
                            of the profile. [default: True]
        offset:             The location in the input image to use as the center of the profile.
                            [default: None]
        gsparams:           An optional `GSParams` argument. [default: None]
        force_stepk:        A stepk value to use rather than the default value. [default: 0.]
        force_maxk:         A maxk value to use rather than the default value. [default: 0.]

    Returns:
        an `InterpolatedImage` instance
    """
    ret = InterpolatedImage.__new__(InterpolatedImage)

    # We need to set all the various attributes that are expected to be in an InterpolatedImage:
    ret._image = image.view(dtype=np.float64, contiguous=True)
    ret._gsparams = GSParams.check(gsparams)
    ret._x_interpolant = x_interpolant.withGSParams(ret._gsparams)
    ret._k_interpolant = k_interpolant.withGSParams(ret._gsparams)

    offset = ret._parse_offset(offset)
    ret._offset = ret._adjust_offset(ret._image.bounds, offset, None, use_true_center)
    im_cen = ret._image.true_center if use_true_center else ret._image.center
    ret._wcs = ret._image.wcs.local(image_pos = im_cen)
    ret._pad_factor = 1.
    ret._image_flux = np.sum(ret._image.array, dtype=np.float64)
    ret._flux = ret._image_flux

    # If image isn't a good fft size, we may still need to pad it out.
    size = max(ret._image.array.shape)
    pad_size = Image.good_fft_size(size)
    if size == pad_size:
        ret._xim = ret._image
    else:
        ret._xim = Image(pad_size, pad_size, dtype=np.float64)
        ret._xim.setCenter(ret._image.center)
        ret._xim[ret._image.bounds] = ret._image
        ret._xim.wcs = ret._wcs
        ret._image = ret._xim[ret._image.bounds]
    ret._pad_image = ret._image

    if force_stepk == 0.:
        ret._stepk = ret._getSimpleStepK(np.max(ret._image.array.shape) / 2. - 0.5)
    else:
        ret._stepk = force_stepk
    if force_maxk == 0.:
        ret._maxk = ret._x_interpolant.krange / ret._wcs._maxScale()
    else:
        ret._maxk = force_maxk
    return ret


class InterpolatedKImage(GSObject):
    """A class describing non-parametric profiles specified by samples of their complex Fourier
    transform.

    The InterpolatedKImage class is useful if you have a non-parametric description of the Fourier
    transform of the profile (provided as either a complex `Image` or two images giving the real
    and imaginary parts) that you wish to manipulate / transform using `GSObject` methods such as
    `GSObject.shear`, `GSObject.magnify`, `GSObject.shift`, etc.  Note that neither real-space
    convolution nor photon-shooting of InterpolatedKImages is currently implemented.  Please submit
    an issue at http://github.com/GalSim-developers/GalSim/issues if you require either of these
    use cases.

    The images required for creating an InterpolatedKImage are precisely those returned by the
    `GSObject.drawKImage` method.  The ``a`` and ``b`` objects in the following command will
    produce essentially equivalent images when drawn with the `GSObject.drawImage` method::

    >>> a = returns_a_GSObject()
    >>> b = galsim.InterpolatedKImage(a.drawKImage())

    The input ``kimage`` must have dtype=numpy.complex64 or dtype=numpy.complex128, which are also
    known as `ImageCF` and `ImageCD` objects respectively.
    The only wcs permitted is a simple `PixelScale` (or `OffsetWCS`), in which case ``kimage.scale``
    is used for the ``stepk`` value unless overridden by the ``stepk`` initialization argument.

    Furthermore, the complex-valued Fourier profile given by ``kimage`` must be Hermitian, since it
    represents a real-valued real-space profile.  (To see an example of valid input to
    InterpolatedKImage, you can look at the output of `GSObject.drawKImage`).

    The user may optionally specify an interpolant, ``k_interpolant``, for Fourier-space
    manipulations (e.g., shearing, resampling).  If none is specified, then by default, a `Quintic`
    interpolant is used.  The `Quintic` interpolant has been found to be a good compromise between
    speed and accuracy for real-and Fourier-space interpolation of objects specified by samples of
    their real-space profiles (e.g., in `InterpolatedImage`), though no extensive testing has been
    performed for objects specified by samples of their Fourier-space profiles (e.g., this
    class).

    Example::

        >>> interpolated_kimage = galsim.InterpolatedKImage(kimage, k_interpolant=None, stepk=0.,
                                                            gsparams=None)

    Initializes ``interpolated_kimage`` as an InterpolatedKImage instance.

    Parameters:
        kimage:         The complex `Image` corresponding to the Fourier-space samples.
        k_interpolant:  Either an `Interpolant` instance or a string indicating which k-space
                        interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                        'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                        to use.  [default: galsim.Quintic()]
        stepk:          By default, the stepk value (the sampling frequency in Fourier-space)
                        of the profile is set by the ``scale`` attribute of the supplied images.
                        This keyword allows the user to specify a coarser sampling in Fourier-
                        space, which may increase efficiency at the expense of decreasing the
                        separation between neighboring copies of the DFT-rendered real-space
                        profile.  (See the `GSParams` docstring for the parameter
                        ``folding_threshold`` for more information). [default: kimage.scale]
        gsparams:       An optional `GSParams` argument. [default: None]
        real_kimage:    Optionally, rather than provide kimage, you may provide the real
                        and imaginary parts separately.  These separate real-valued images
                        may be strings, in which case they refer to FITS files from which
                        to read the images. [default: None]
        imag_kimage:    The imaginary image corresponding to real_kimage. [default: None]
        real_hdu:       When reading in real_kimage from a file, this parameter can be used to
                        select a particular HDU in the file. [default: None]
        imag_hdu:       When reading in imag_kimage from a file, this parameter can be used to
                        select a particular HDU in the file. [default: None]
    """
    _req_params = { 'real_kimage' : str,
                    'imag_kimage' : str }
    _opt_params = {
        'k_interpolant' : str, 'stepk': float,
        'real_hdu': int, 'imag_hdu': int,
    }

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, kimage=None, k_interpolant=None, stepk=None, gsparams=None,
                 real_kimage=None, imag_kimage=None, real_hdu=None, imag_hdu=None):
        if kimage is None:
            if real_kimage is None or imag_kimage is None:
                raise GalSimIncompatibleValuesError(
                    "Must provide either kimage or real_kimage/imag_kimage",
                    kimage=kimage, real_kimage=real_kimage, imag_kimage=imag_kimage)

            # If the "image" is not actually an image, try to read the image as a file.
            if isinstance(real_kimage, str):
                real_kimage = fits.read(real_kimage, hdu=real_hdu)
            elif not isinstance(real_kimage, Image):
                raise TypeError("real_kimage must be either an Image or a file name")
            if isinstance(imag_kimage, str):
                imag_kimage = fits.read(imag_kimage, hdu=imag_hdu)
            elif not isinstance(imag_kimage, Image):
                raise TypeError("imag_kimage must be either an Image or a file name")

            # make sure real_kimage, imag_kimage are congruent.
            if real_kimage.bounds != imag_kimage.bounds:
                raise GalSimIncompatibleValuesError(
                    "Real and Imag kimages must have same bounds.",
                    real_kimage=real_kimage, imag_kimage=imag_kimage)
            if real_kimage.wcs != imag_kimage.wcs:
                raise GalSimIncompatibleValuesError(
                    "Real and Imag kimages must have same scale/wcs.",
                    real_kimage=real_kimage, imag_kimage=imag_kimage)

            kimage = real_kimage + 1j*imag_kimage
        else:
            if real_kimage is not None or imag_kimage is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both kimage and real_kimage/imag_kimage",
                    kimage=kimage, real_kimage=real_kimage, imag_kimage=imag_kimage)
            if not isinstance(kimage, Image):
                raise TypeError("kimage must be a galsim.Image isntance")
            if not kimage.iscomplex:
                raise GalSimValueError("Supplied kimage is not complex", kimage)

        # Make sure wcs is a PixelScale.
        if kimage.wcs is not None and not kimage.wcs._isPixelScale:
            raise GalSimValueError("kimage wcs must be PixelScale or None.", kimage.wcs)

        if not kimage.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Supplied image does not have bounds defined.")

        self._gsparams = GSParams.check(gsparams)

        # Check for Hermitian symmetry properties of kimage
        shape = kimage.array.shape
        # If image is even-sized, ignore first row/column since in this case not every pixel has
        # a symmetric partner to which to compare.
        bd = _BoundsI(kimage.xmin + (1 if shape[1]%2==0 else 0),
                      kimage.xmax,
                      kimage.ymin + (1 if shape[0]%2==0 else 0),
                      kimage.ymax)
        if not (np.allclose(kimage[bd].real.array,
                            kimage[bd].real.array[::-1,::-1]) and
                np.allclose(kimage[bd].imag.array,
                            -kimage[bd].imag.array[::-1,::-1])):
            raise GalSimIncompatibleValuesError(
                "Real and Imag kimages must form a Hermitian complex matrix.", kimage=kimage)

        # Make sure the image is complex128 and contiguous
        self._kimage = kimage.view(dtype=np.complex128, contiguous=True)
        self._kimage.setCenter(0,0)

        if stepk is None:
            if self._kimage.scale is None:
                # Defaults to 1.0 if no scale is set.
                self._kimage.scale = 1.
            self._stepk = self._kimage.scale
        elif stepk < kimage.scale:
            galsim_warn(
                "Provided stepk is smaller than kimage.scale; overriding with kimage.scale.")
            self._stepk = kimage.scale
        else:
            self._stepk = stepk

        # set up k_interpolant if none was provided by user, or check that the user-provided one
        # is of a valid type
        if k_interpolant is None:
            self._k_interpolant = Quintic(gsparams=self._gsparams)
        else:
            self._k_interpolant = convert_interpolant(k_interpolant).withGSParams(self._gsparams)

    @property
    def kimage(self):
        """The underlying `Image` being interpolated.
        """
        return self._kimage

    @property
    def k_interpolant(self):
        """The Fourier-space `Interpolant` for this profile.
        """
        return self._k_interpolant

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        ret._k_interpolant = self._k_interpolant.withGSParams(ret._gsparams, **kwargs)
        return ret

    @lazy_property
    def _sbp(self):
        stepk_image = self.stepk / self.kimage.scale  # usually 1, but could be larger

        # C++ layer needs Bounds that look like 0, N/2, -N/2, N/2-1
        # So find the biggest N that works like that.
        b = self._kimage.bounds
        N = min(b.xmax*2, -b.ymin*2, b.ymax*2+1)
        b = _BoundsI(0, N//2, -(N//2), N//2-1)
        posx_kimage = self._kimage[b]
        self._sbiki = _galsim.SBInterpolatedKImage(
                posx_kimage._image, stepk_image, self.k_interpolant._i, self.gsparams._gsp)

        scale = self.kimage.scale
        jac = np.array((1./scale, 0., 0., 1./scale))
        _jac = jac.__array_interface__['data'][0]
        if scale != 1.:
            return _galsim.SBTransform(self._sbiki, _jac, 0., 0., scale**2, self.gsparams._gsp)
        else:
            return self._sbiki

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, InterpolatedKImage) and
                 np.array_equal(self.kimage.array, other.kimage.array) and
                 self.kimage.scale == other.kimage.scale and
                 self.k_interpolant == other.k_interpolant and
                 self.stepk == other.stepk and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        # Definitely want to cache this, since the kimage could be large.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.InterpolatedKImage", self.k_interpolant, self._stepk,
                               self.gsparams))
            self._hash ^= hash(tuple(self.kimage.array.ravel()))
            self._hash ^= hash((self.kimage.bounds, self.kimage.wcs))
        return self._hash

    def __repr__(self):
        return ('galsim.InterpolatedKImage(\n%r,\n%r, stepk=%r, gsparams=%r)')%(
                self.kimage, self.k_interpolant, self.stepk, self.gsparams)

    def __str__(self):
        return 'galsim.InterpolatedKImage(kimage=%s)'%(self.kimage)

    def __getstate__(self):
        # The SBInterpolatedKImage is picklable, but that is pretty inefficient, due to the large
        # images being written as strings.  Better to pickle the intermediate products and then
        # call init again on the other side.  There's still an image to be pickled, but at least
        # it will be through the normal pickling rules, rather than the repr.
        d = self.__dict__.copy()
        d.pop('_sbiki',None)
        d.pop('_sbp',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        return self._sbp.maxK()

    @property
    def _centroid(self):
        with convert_cpp_errors():
            return PositionD(self._sbp.centroid())

    @property
    def _flux(self):
        return self._sbp.getFlux()

    @property
    def _positive_flux(self):
        return self._sbp.getPositiveFlux()

    @property
    def _negative_flux(self):
        return self._sbp.getNegativeFlux()

    @lazy_property
    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _drawKImage(self, image, jac=None):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.drawK(image._image, image.scale, _jac)


def _InterpolatedKImage(kimage, k_interpolant, gsparams):
    """Approximately equivalent to `InterpolatedKImage`, but with fewer options and no sanity
    checks.

    Parameters:
        kimage:         The complex `Image` corresponding to the Fourier-space samples.
        k_interpolant:  An `Interpolant` instance indicating which k-space interpolant should be
                        used.
        gsparams:       An optional `GSParams` argument. [default: None]
     """
    ret = InterpolatedKImage.__new__(InterpolatedKImage)
    ret._kimage = kimage.view(dtype=np.complex128, contiguous=True)
    ret._kimage.shift(-kimage.center)
    ret._stepk = kimage.scale
    ret._gsparams = GSParams.check(gsparams)
    ret._k_interpolant = k_interpolant.withGSParams(gsparams)
    return ret
