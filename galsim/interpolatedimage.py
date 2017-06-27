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
"""@file interpolatedimage.py

InterpolatedImage is a class that allows one to treat an image as a profile.
"""

from past.builtins import basestring
import galsim
from galsim import GSObject
from . import _galsim
from ._galsim import Interpolant
from ._galsim import Nearest, Linear, Cubic, Quintic, Lanczos, SincInterpolant, Delta
import numpy as np

class InterpolatedImage(GSObject):
    """A class describing non-parametric profiles specified using an Image, which can be
    interpolated for the purpose of carrying out transformations.

    The InterpolatedImage class is useful if you have a non-parametric description of an object as
    an Image, that you wish to manipulate / transform using GSObject methods such as shear(),
    magnify(), shift(), etc.  Note that when convolving an InterpolatedImage, the use of real-space
    convolution is not recommended, since it is typically a great deal slower than Fourier-space
    convolution for this kind of object.

    There are three options for determining the flux of the profile.  First, you can simply
    specify a `flux` value explicitly.  Or there are two ways to get the flux from the image
    directly.  If you set `normalization = 'flux'`, the flux will be taken as the sum of the
    pixel values.  This corresponds to an image that was drawn with `drawImage(method='no_pixel')`.
    This is the default if flux is not given.  The other option, `normalization = 'sb'` treats
    the pixel values as samples of the surface brightness profile at each location.  This
    corresponds to an image drawn with `drawImage(method='sb')`.

    You can also use images that were drawn with one of the pixel-integrating methods ('auto',
    'fft', or 'real_space'); however, the resulting profile will not correspond to the one
    that was used to call `drawImage`.  The integration over the pixel is equivalent to convolving
    the original profile by a Pixel and then drawing with `method='no_pixel'`.  So if you use
    such an image with InterpolatedImage, the resulting profile will include the Pixel convolution
    already.  As such, if you use it as a PSF for example, then the final objects convolved by
    this PSF will already include the pixel convolution, so you should draw them using
    `method='no_pixel'`.

    If the input Image has a `scale` or `wcs` associated with it, then there is no need to specify
    one as a parameter here.  But if one is provided, that will override any `scale` or `wcs` that
    is native to the Image.

    The user may optionally specify an interpolant, `x_interpolant`, for real-space manipulations
    (e.g., shearing, resampling).  If none is specified, then by default, a Quintic interpolant is
    used.  The user may also choose to specify two quantities that can affect the Fourier space
    convolution: the k-space interpolant (`k_interpolant`) and the amount of padding to include
    around the original images (`pad_factor`).  The default values for `x_interpolant`,
    `k_interpolant`, and `pad_factor` were chosen based on the tests of branch #389 to reach good
    accuracy without being excessively slow.  Users should be particularly wary about changing
    `k_interpolant` and `pad_factor` from the defaults without careful testing.  The user is given
    complete freedom to choose interpolants and pad factors, and no warnings are raised when the
    code is modified to choose some combination that is known to give significant error.  More
    details can be found in http://arxiv.org/abs/1401.2636, especially table 1, and in comment
    https://github.com/GalSim-developers/GalSim/issues/389#issuecomment-26166621 and the following
    comments.

    The user can choose to pad the image with a noise profile if desired.  To do so, specify
    the target size for the noise padding in `noise_pad_size`, and specify the kind of noise
    to use in `noise_pad`.  The `noise_pad` option may be a Gaussian random noise of some variance,
    or a Gaussian but correlated noise field that is specified either as a CorrelatedNoise
    instance, an Image (from which a correlated noise model is derived), or a string (interpreted
    as a filename containing an image to use for deriving a CorrelatedNoise).  The user can also
    pass in a random number generator to be used for noise generation.  Finally, the user can pass
    in a `pad_image` for deterministic image padding.

    By default, the InterpolatedImage recalculates the Fourier-space step and number of points to
    use for further manipulations, rather than using the most conservative possibility.  For typical
    objects representing galaxies and PSFs this can easily make the difference between several
    seconds (conservative) and 0.04s (recalculated).  However, the user can turn off this option,
    and may especially wish to do so when using images that do not contain a high S/N object - e.g.,
    images of noise fields.

    Initialization
    --------------

        >>> interpolated_image = galsim.InterpolatedImage(
                image, x_interpolant=None, k_interpolant=None, normalization='flux', scale=None,
                wcs=None, flux=None, pad_factor=4., noise_pad_size=0, noise_pad=0., use_cache=True,
                pad_image=None, rng=None, calculate_stepk=True, calculate_maxk=True,
                use_true_center=True, offset=None, hdu=None)

    Initializes `interpolated_image` as an InterpolatedImage instance.

    For comparison of the case of padding with noise or zero when the image itself includes noise,
    compare `im1` and `im2` from the following code snippet (which can be executed from the
    examples/ directory):

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

    @param image            The Image from which to construct the object.
                            This may be either an Image instance or a string indicating a fits
                            file from which to read the image.  In the latter case, the `hdu`
                            kwarg can be used to specify a particular HDU in that file.
    @param x_interpolant    Either an Interpolant instance or a string indicating which real-space
                            interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                            'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                            to use. [default: galsim.Quintic()]
    @param k_interpolant    Either an Interpolant instance or a string indicating which k-space
                            interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                            'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                            to use.  We strongly recommend leaving this parameter at its default
                            value; see text above for details.  [default: galsim.Quintic()]
    @param normalization    Two options for specifying the normalization of the input Image:
                              "flux" or "f" means that the sum of the pixels is normalized
                                  to be equal to the total flux.
                              "surface brightness" or "sb" means that the pixels sample
                                  the surface brightness distribution at each location.
                            This is overridden if you specify an explicit flux value.
                            [default: "flux"]
    @param scale            If provided, use this as the pixel scale for the Image; this will
                            override the pixel scale stored by the provided Image, in any.
                            If `scale` is `None`, then take the provided image's pixel scale.
                            [default: None]
    @param wcs              If provided, use this as the wcs for the image.  At most one of `scale`
                            or `wcs` may be provided. [default: None]
    @param flux             Optionally specify a total flux for the object, which overrides the
                            implied flux normalization from the Image itself. [default: None]
    @param pad_factor       Factor by which to pad the Image with zeros.  We strongly recommend
                            leaving this parameter at its default value; see text above for
                            details.  [default: 4]
    @param noise_pad_size   If provided, the image will be padded out to this size (in arcsec) with
                            the noise specified by `noise_pad`. This is important if you are
                            planning to whiten the resulting image.  You want to make sure that the
                            noise-padded image is larger than the postage stamp onto which you are
                            drawing this object.  [default: None]
    @param noise_pad        Noise properties to use when padding the original image with
                            noise.  This can be specified in several ways:
                               (a) as a float, which is interpreted as being a variance to use when
                                   padding with uncorrelated Gaussian noise;
                               (b) as a galsim.CorrelatedNoise, which contains information about the
                                   desired noise power spectrum - any random number generator passed
                                   to the `rng` keyword will take precedence over that carried in an
                                   input CorrelatedNoise instance;
                               (c) as an Image of a noise field, which is used to calculate
                                   the desired noise power spectrum; or
                               (d) as a string which is interpreted as a filename containing an
                                   example noise field with the proper noise power spectrum (as an
                                   Image in the first HDU).
                            It is important to keep in mind that the calculation of the correlation
                            function that is internally stored within a CorrelatedNoise object is a
                            non-negligible amount of overhead, so the recommended means of
                            specifying a correlated noise field for padding are (b) or (d).  In the
                            case of (d), if the same file is used repeatedly, then the `use_cache`
                            keyword (see below) can be used to prevent the need for repeated
                            CorrelatedNoise initializations.
                            [default: 0, i.e., pad with zeros]
    @param use_cache        Specify whether to cache `noise_pad` read in from a file to save having
                            to build a CorrelatedNoise object repeatedly from the same image.
                            [default: True]
    @param rng              If padding by noise, the user can optionally supply the random noise
                            generator to use for drawing random numbers as `rng` (may be any kind of
                            BaseDeviate object).  Such a user-input random number generator
                            takes precedence over any stored within a user-input CorrelatedNoise
                            instance (see the `noise_pad` parameter).
                            If `rng=None`, one will be automatically created, using the time as a
                            seed. [default: None]
    @param pad_image        Image to be used for deterministically padding the original image.  This
                            can be specified in two ways:
                               (a) as an Image; or
                               (b) as a string which is interpreted as a filename containing an
                                   image to use (in the first HDU).
                            The `pad_image` scale or wcs is ignored.  It uses the same scale or
                            wcs for both the `image` and the `pad_image`.
                            The user should be careful to ensure that the image used for padding
                            has roughly zero mean.  The purpose of this keyword is to allow for a
                            more flexible representation of some noise field around an object; if
                            the user wishes to represent the sky level around an object, they
                            should do that after they have drawn the final image instead.
                            [default: None]
    @param calculate_stepk  Specify whether to perform an internal determination of the extent of
                            the object being represented by the InterpolatedImage; often this is
                            useful in choosing an optimal value for the stepsize in the Fourier
                            space lookup table.
                            If you know a priori an appropriate maximum value for `stepk`, then
                            you may also supply that here instead of a bool value, in which case
                            the `stepk` value is still calculated, but will not go above the
                            provided value.
                            [default: True]
    @param calculate_maxk   Specify whether to perform an internal determination of the highest
                            spatial frequency needed to accurately render the object being
                            represented by the InterpolatedImage; often this is useful in choosing
                            an optimal value for the extent of the Fourier space lookup table.
                            If you know a priori an appropriate maximum value for `maxk`, then
                            you may also supply that here instead of a bool value, in which case
                            the `maxk` value is still calculated, but will not go above the
                            provided value.
                            [default: True]
    @param use_true_center  Similar to the same parameter in the GSObject.drawImage() function,
                            this sets whether to use the true center of the provided image as the
                            center of the profile (if `use_true_center=True`) or the nominal
                            center returned by image.bounds.center() (if `use_true_center=False`)
                            [default: True]
    @param offset           The location in the input image to use as the center of the profile.
                            This should be specified relative to the center of the input image
                            (either the true center if `use_true_center=True`, or the nominal
                            center if `use_true_center=False`).  [default: None]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    @param hdu              When reading in an Image from a file, this parameter can be used to
                            select a particular HDU in the file. [default: None]

    Methods
    -------

    There are no additional methods for InterpolatedImage beyond the usual GSObject methods.
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
    _single_params = []
    _takes_rng = True
    _cache_noise_pad = {}

    def __init__(self, image, x_interpolant=None, k_interpolant=None, normalization='flux',
                 scale=None, wcs=None, flux=None, pad_factor=4., noise_pad_size=0, noise_pad=0.,
                 rng=None, pad_image=None, calculate_stepk=True, calculate_maxk=True,
                 use_cache=True, use_true_center=True, offset=None, gsparams=None, dx=None,
                 _force_stepk=0., _force_maxk=0., _serialize_stepk=None, _serialize_maxk=None,
                 hdu=None):

        # Check for obsolete dx parameter
        if dx is not None and scale is None: # pragma: no cover
            from galsim.deprecated import depr
            depr('dx', 1.1, 'scale')
            scale = dx

        # If the "image" is not actually an image, try to read the image as a file.
        if not isinstance(image, galsim.Image):
            image = galsim.fits.read(image, hdu=hdu)

        # make sure image is really an image and has a float type
        if image.dtype != np.float32 and image.dtype != np.float64:
            raise ValueError("Supplied image does not have dtype of float32 or float64!")

        # it must have well-defined bounds, otherwise seg fault in SBInterpolatedImage constructor
        if not image.bounds.isDefined():
            raise ValueError("Supplied image does not have bounds defined!")

        # check what normalization was specified for the image: is it an image of surface
        # brightness, or flux?
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness', or 'sb'.") % normalization)

        # set up the interpolants if none was provided by user, or check that the user-provided ones
        # are of a valid type
        if x_interpolant is None:
            self.x_interpolant = galsim.Quintic(tol=1e-4)
        else:
            self.x_interpolant = galsim.utilities.convert_interpolant(x_interpolant)
        if k_interpolant is None:
            self.k_interpolant = galsim.Quintic(tol=1e-4)
        else:
            self.k_interpolant = galsim.utilities.convert_interpolant(k_interpolant)

        # Store the image as an attribute and make sure we don't change the original image
        # in anything we do here.  (e.g. set scale, etc.)
        self.image = image._view()
        self.use_cache = use_cache

        # Set the wcs if necessary
        if scale is not None:
            if wcs is not None:
                raise TypeError("Cannot provide both scale and wcs to InterpolatedImage")
            self.image.wcs = galsim.PixelScale(scale)
        elif wcs is not None:
            if not isinstance(wcs, galsim.BaseWCS):
                raise TypeError("wcs parameter is not a galsim.BaseWCS instance")
            self.image.wcs = wcs
        elif self.image.wcs is None:
            raise ValueError("No information given with Image or keywords about pixel scale!")

        # Set up the GaussianDeviate if not provided one, or check that the user-provided one is
        # of a valid type.
        if rng is None:
            if noise_pad: rng = galsim.BaseDeviate()
        elif not isinstance(rng, galsim.BaseDeviate):
            raise TypeError("rng provided to InterpolatedImage constructor is not a BaseDeviate")

        # Check that given pad_image is valid:
        if pad_image:
            if isinstance(pad_image, basestring):
                pad_image = galsim.fits.read(pad_image)
            if not isinstance(pad_image, galsim.Image):
                raise ValueError("Supplied pad_image is not an Image!")
            if pad_image.dtype != np.float32 and pad_image.dtype != np.float64:
                raise ValueError("Supplied pad_image is not one of the allowed types!")

        # Check that the given noise_pad is valid:
        try:
            noise_pad = float(noise_pad)
        except (TypeError, ValueError):
            pass
        if isinstance(noise_pad, float):
            if noise_pad < 0.:
                raise ValueError("Noise variance cannot be negative!")
        # There are other options for noise_pad, the validity of which will be checked in
        # the helper function self.buildNoisePadImage()

        # This will be passed to SBInterpolatedImage, so make sure it is the right type.
        pad_factor = float(pad_factor)
        if pad_factor <= 0.:
            raise ValueError("Invalid pad_factor <= 0 in InterpolatedImage")

        if use_true_center:
            im_cen = self.image.bounds.trueCenter()
        else:
            im_cen = self.image.bounds.center()

        local_wcs = self.image.wcs.local(image_pos = im_cen)
        self.min_scale = local_wcs._minScale()
        self.max_scale = local_wcs._maxScale()

        # Make sure the image fits in the noise pad image:
        if noise_pad_size:
            import math
            # Convert from arcsec to pixels according to the local wcs.
            # Use the minimum scale, since we want to make sure noise_pad_size is
            # as large as we need in any direction.
            noise_pad_size = int(math.ceil(noise_pad_size / self.min_scale))
            # Round up to a good size for doing FFTs
            noise_pad_size = galsim.Image.good_fft_size(noise_pad_size)
            if noise_pad_size <= min(self.image.array.shape):
                # Don't need any noise padding in this case.
                noise_pad_size = 0
            elif noise_pad_size < max(self.image.array.shape):
                noise_pad_size = max(self.image.array.shape)

        # See if we need to pad out the image with either a pad_image or noise_pad
        if noise_pad_size:
            new_pad_image = self.buildNoisePadImage(noise_pad_size, noise_pad, rng)

            if pad_image:
                # if both noise_pad and pad_image are set, then we need to build up a larger
                # pad_image and place the given pad_image in the center.

                # We will change the bounds here, so make a new view to avoid modifying the
                # input pad_image.
                pad_image = pad_image._view()
                pad_image.setCenter(0,0)
                new_pad_image.setCenter(0,0)
                if new_pad_image.bounds.includes(pad_image.bounds):
                    new_pad_image[pad_image.bounds] = pad_image
                else:
                    new_pad_image = pad_image

            pad_image = new_pad_image

        elif pad_image:
            # Just make sure pad_image is the right type
            pad_image = galsim.Image(pad_image, dtype=image.dtype)

        # Now place the given image in the center of the padding image:
        if pad_image:
            pad_image.setCenter(0,0)
            self.image.setCenter(0,0)
            if pad_image.bounds.includes(self.image.bounds):
                pad_image[self.image.bounds] = self.image
                pad_image.wcs = self.image.wcs
            else:
                # If padding was smaller than original image, just use the original image.
                pad_image = self.image
        else:
            pad_image = self.image

        # GalSim cannot automatically know what stepK and maxK are appropriate for the
        # input image.  So it is usually worth it to do a manual calculation (below).
        #
        # However, there is also a hidden option to force it to use specific values of stepK and
        # maxK (caveat user!).  The values of _force_stepk and _force_maxk should be provided in
        # terms of physical scale, e.g., for images that have a scale length of 0.1 arcsec, the
        # stepK and maxK should be provided in units of 1/arcsec.  Then we convert to the 1/pixel
        # units required by the C++ layer below.  Also note that profile recentering for even-sized
        # images (see the ._fix_center step below) leads to automatic reduction of stepK slightly
        # below what is provided here, while maxK is preserved.
        if _force_stepk > 0.:
            calculate_stepk = False
            _force_stepk *= self.min_scale
        if _force_maxk > 0.:
            calculate_maxk = False
            _force_maxk *= self.max_scale

        # Due to floating point rounding errors, for pickling it's necessary to store the exact
        # _force_maxk and _force_stepk used to create the SBInterpolatedImage, as opposed to the
        # values before being scaled by self.min_scale and self.max_scale.  So we do that via the
        # _serialize_maxk and _serialize_stepk hidden kwargs, which should only get used during
        # pickling.
        if _serialize_stepk is not None:
            calculate_stepk = False
            _force_stepk = _serialize_stepk
        if _serialize_maxk is not None:
            calculate_maxk = False
            _force_maxk = _serialize_maxk

        # Save these values for pickling
        self._pad_image = pad_image
        self._pad_factor = pad_factor
        self._gsparams = gsparams

        # Make the SBInterpolatedImage out of the image.
        sbii = galsim._galsim.SBInterpolatedImage(
                pad_image.image, self.x_interpolant, self.k_interpolant, pad_factor,
                _force_stepk, _force_maxk, gsparams)

        # I think the only things that will mess up if getFlux() == 0 are the
        # calculateStepK and calculateMaxK functions, and rescaling the flux to some value.
        if (calculate_stepk or calculate_maxk or flux is not None) and sbii.getFlux() == 0.:
            raise RuntimeError("This input image has zero total flux. "
                               "It does not define a valid surface brightness profile.")

        if calculate_stepk:
            if calculate_stepk is True:
                sbii.calculateStepK()
            else:
                # If not a bool, then value is max_stepk
                sbii.calculateStepK(max_stepk=calculate_stepk)
        if calculate_maxk:
            if calculate_maxk is True:
                sbii.calculateMaxK()
            else:
                # If not a bool, then value is max_maxk
                sbii.calculateMaxK(max_maxk=calculate_maxk)

        # If the user specified a surface brightness normalization for the input Image, then
        # need to rescale flux by the pixel area to get proper normalization.
        if flux is None and normalization.lower() in ['surface brightness','sb']:
            flux = sbii.getFlux() * local_wcs.pixelArea()

        # Save this intermediate profile
        self._sbii = sbii
        self._stepk = sbii.stepK() / self.min_scale
        self._maxk = sbii.maxK() / self.max_scale
        self._flux = flux

        self._serialize_stepk = sbii.stepK()
        self._serialize_maxk = sbii.maxK()

        prof = GSObject(sbii)

        # Make sure offset is a PositionD
        offset = prof._parse_offset(offset)

        # Apply the offset, and possibly fix the centering for even-sized images
        # Note reverse=True, since we want to fix the center in the opposite sense of what the
        # draw function does.
        prof = prof._fix_center(self.image.bounds, offset, use_true_center, reverse=True)

        # Save the offset we will need when pickling.
        if hasattr(prof, 'offset'):
            self._offset = -prof.offset
        else:
            self._offset = None

        # Bring the profile from image coordinates into world coordinates
        prof = local_wcs._profileToWorld(prof)

        # If the user specified a flux, then set to that flux value.
        if flux is not None:
            prof = prof.withFlux(float(flux))

        # Now, in order for these to pickle correctly if they are the "original" object in a
        # Transform object, we need to hide the current transformation.  An easy way to do that
        # is to hide the SBProfile in an SBAdd object.
        sbp = galsim._galsim.SBAdd([prof.SBProfile])

        GSObject.__init__(self, sbp)

    def buildNoisePadImage(self, noise_pad_size, noise_pad, rng):
        """A helper function that builds the `pad_image` from the given `noise_pad` specification.
        """
        # Make it with the same dtype as the image
        pad_image = galsim.Image(noise_pad_size, noise_pad_size, dtype=self.image.dtype)

        # Figure out what kind of noise to apply to the image
        if isinstance(noise_pad, float):
            noise = galsim.GaussianNoise(rng, sigma = np.sqrt(noise_pad))
        elif isinstance(noise_pad, galsim.correlatednoise._BaseCorrelatedNoise):
            noise = noise_pad.copy(rng=rng)
        elif isinstance(noise_pad,galsim.Image):
            noise = galsim.CorrelatedNoise(noise_pad, rng)
        elif self.use_cache and noise_pad in InterpolatedImage._cache_noise_pad:
            noise = InterpolatedImage._cache_noise_pad[noise_pad]
            if rng:
                # Make sure that we are using a specified RNG by resetting that in this cached
                # CorrelatedNoise instance, otherwise preserve the cached RNG
                noise = noise.copy(rng=rng)
        elif isinstance(noise_pad, basestring):
            noise = galsim.CorrelatedNoise(galsim.fits.read(noise_pad), rng)
            if self.use_cache:
                InterpolatedImage._cache_noise_pad[noise_pad] = noise
        else:
            raise ValueError(
                "Input noise_pad must be a float/int, a CorrelatedNoise, Image, or filename "+
                "containing an image to use to make a CorrelatedNoise!")
        # Add the noise
        pad_image.addNoise(noise)

        return pad_image

    def __eq__(self, other):
        return (isinstance(other, galsim.InterpolatedImage) and
                self._pad_image == other._pad_image and
                self.x_interpolant == other.x_interpolant and
                self.k_interpolant == other.k_interpolant and
                self._pad_factor == other._pad_factor and
                self._flux == other._flux and
                self._offset == other._offset and
                self._gsparams == other._gsparams and
                self._stepk == other._stepk and
                self._maxk == other._maxk)

    def __hash__(self):
        # Definitely want to cache this, since the size of the image could be large.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.InterpolatedImage", self.x_interpolant, self.k_interpolant,
                               self._pad_factor, self._flux, self._offset, self._gsparams,
                               self._stepk, self._maxk))
            self._hash ^= hash(tuple(self._pad_image.array.ravel()))
            self._hash ^= hash((self._pad_image.bounds, self._pad_image.wcs))
        return self._hash

    def __repr__(self):
        return ('galsim.InterpolatedImage(%r, %r, %r, pad_factor=%r, flux=%r, offset=%r, '+
                'use_true_center=False, gsparams=%r, _force_stepk=%r, _force_maxk=%r)')%(
                self._pad_image, self.x_interpolant, self.k_interpolant,
                self._pad_factor, self._flux, self._offset, self._gsparams,
                self._stepk, self._maxk)

    def __str__(self): return 'galsim.InterpolatedImage(image=%s, flux=%s)'%(self.image, self.flux)

    def __getstate__(self):
        # The SBInterpolatedImage and the SBProfile both are picklable, but they are pretty
        # inefficient, due to the large images being written as strings.  Better to pickle
        # the intermediate products and then call init again on the other side.  There's still
        # an image to be pickled, but at least it will be through the normal pickling rules,
        # rather than the repr.
        d = self.__dict__.copy()
        del d['_sbii']
        del d['image']
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._pad_image,
                      x_interpolant=self.x_interpolant, k_interpolant=self.k_interpolant,
                      pad_factor=self._pad_factor, flux=self._flux,
                      offset=self._offset, use_true_center=False, gsparams=self._gsparams,
                      _serialize_stepk=self._serialize_stepk,
                      _serialize_maxk=self._serialize_maxk)


class InterpolatedKImage(GSObject):
    """A class describing non-parametric profiles specified by samples of their complex Fourier
    transform.

    The InterpolatedKImage class is useful if you have a non-parametric description of the Fourier
    transform of the profile (provided as either a complex Image or two Images giving the real and
    imaginary parts) that you wish to manipulate / transform using GSObject methods such as
    shear(), magnify(), shift(), etc.  Note that neither real-space convolution nor photon-shooting
    of InterpolatedKImages is currently implemented.  Please submit an issue at
    http://github.com/GalSim-developers/GalSim/issues if you require either of these use cases.

    The images required for creating an InterpolatedKImage are precisely those returned by the
    GSObject `.drawKImage()` method.  The `a` and `b` objects in the following command will produce
    essentially equivalent images when drawn with the `.drawImage()` method:

    >>> a = returns_a_GSObject()
    >>> b = galsim.InterpolatedKImage(a.drawKImage())

    The input `kimage` must have dtype=numpy.complex64 or dtype=numpy.complex128, which are also
    known as ImageCF and ImageCD objects respectively.
    The only wcs permitted is a simple PixelScale (or OffsetWCS), in which case `kimage.scale` is
    used for the `stepk` value unless overridden by the `stepk` initialization argument.

    Furthermore, the complex-valued Fourier profile given by `kimage` must be Hermitian, since it
    represents a real-valued real-space profile.  (To see an example of valid input to
    `InterpolatedKImage`, you can look at the output of `drawKImage`).

    The user may optionally specify an interpolant, `k_interpolant`, for Fourier-space
    manipulations (e.g., shearing, resampling).  If none is specified, then by default, a Quintic
    interpolant is used.  The Quintic interpolant has been found to be a good compromise between
    speed and accuracy for real-and Fourier-space interpolation of objects specified by samples of
    their real-space profiles (e.g., in InterpolatedImage), though no extensive testing has been
    performed for objects specified by samples of their Fourier-space profiles (e.g., this
    class).

    Initialization
    --------------

        >>> interpolated_kimage = galsim.InterpolatedKImage(kimage, k_interpolant=None, stepk=0.,
                                                            gsparams=None)

    Initializes `interpolated_kimage` as an InterpolatedKImage instance.

    @param kimage           The complex Image corresponding to the Fourier-space samples.
    @param k_interpolant    Either an Interpolant instance or a string indicating which k-space
                            interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                            'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                            to use.  [default: galsim.Quintic()]
    @param stepk            By default, the stepk value (the sampling frequency in Fourier-space)
                            of the underlying SBProfile is set by the `scale` attribute of the
                            supplied images.  This keyword allows the user to specify a coarser
                            sampling in Fourier-space, which may increase efficiency at the expense
                            of decreasing the separation between neighboring copies of the
                            DFT-rendered real-space profile.  (See the GSParams docstring for the
                            parameter `folding_threshold` for more information).
                            [default: kimage.scale]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    @param real_kimage      Optionally, rather than provide kimage, you may provide the real
                            and imaginary parts separately.  These separate real-valued images
                            may be strings, in which case they refer to FITS files from which
                            to read the images. [default: None]
    @param imag_kimage      The imaginary image corresponding to real_kimage. [default: None]
    @param real_hdu         When reading in real_kimage from a file, this parameter can be used to
                            select a particular HDU in the file. [default: None]
    @param imag_hdu         When reading in imag_kimage from a file, this parameter can be used to
                            select a particular HDU in the file. [default: None]

    Methods
    -------

    There are no additional methods for InterpolatedKImage beyond the usual GSObject methods.
    """
    _req_params = { 'real_kimage' : str,
                    'imag_kimage' : str }
    _opt_params = {
        'k_interpolant' : str, 'stepk': float,
        'real_hdu': int, 'imag_hdu': int,
    }
    _single_params = []
    _takes_rng = False

    def __init__(self, kimage=None, k_interpolant=None, stepk=None, gsparams=None,
                 real_kimage=None, imag_kimage=None, real_hdu=None, imag_hdu=None):
        if isinstance(kimage,galsim.Image) and isinstance(k_interpolant,galsim.Image):
            from .deprecated import depr
            depr('InterpolatedKImage(re,im,...)', 1.5,
                 'either InterpolatedKImage(re + 1j * im, ...) or '
                 'InterpolatedKImage(real_kimage=re, imag_kimage=im)')
            # This won't work if they call InterpolatedKImage(re,im, k_interpolant=kinterp)
            # But I don't see an easy way around that, so I guess that use case is not
            # backwards compatible.  Sorry..
            real_kimage = kimage
            imag_kimage = k_interpolant
            kimage = None
            k_interpolant = None

        if kimage is None:
            if real_kimage is None or imag_kimage is None:
                raise ValueError("Must provide either kimage or real_kimage/imag_kimage")

            # If the "image" is not actually an image, try to read the image as a file.
            if not isinstance(real_kimage, galsim.Image):
                real_kimage = galsim.fits.read(real_kimage, hdu=real_hdu)
            if not isinstance(imag_kimage, galsim.Image):
                imag_kimage = galsim.fits.read(imag_kimage, hdu=imag_hdu)

            # make sure real_kimage, imag_kimage are really `Image`s, are floats, and are
            # congruent.
            if not isinstance(real_kimage, galsim.Image):
                raise ValueError("Supplied real_kimage is not an Image instance")
            if not isinstance(imag_kimage, galsim.Image):
                raise ValueError("Supplied imag_kimage is not an Image instance")
            if real_kimage.bounds != imag_kimage.bounds:
                raise ValueError("Real and Imag kimages must have same bounds.")
            if real_kimage.wcs != imag_kimage.wcs:
                raise ValueError("Real and Imag kimages must have same scale/wcs.")

            kimage = real_kimage + 1j*imag_kimage
        else:
            if real_kimage is not None or imag_kimage is not None:
                raise ValueError("Cannot provide both kimage and real_kimage/imag_kimage")
            if not kimage.iscomplex:
                raise ValueError("Supplied kimage is not complex")

        # Make sure wcs is a PixelScale.
        if kimage.wcs is not None and not kimage.wcs.isPixelScale():
            raise ValueError("kimage wcs must be PixelScale or None.")

        self._kimage = kimage.copy()

        # Check for Hermitian symmetry properties of kimage
        shape = kimage.array.shape
        # If image is even-sized, ignore first row/column since in this case not every pixel has
        # a symmetric partner to which to compare.
        bd = galsim.BoundsI(kimage.xmin + (1 if shape[1]%2==0 else 0),
                            kimage.xmax,
                            kimage.ymin + (1 if shape[0]%2==0 else 0),
                            kimage.ymax)
        if not (np.allclose(kimage[bd].real.array,
                            kimage[bd].real.array[::-1,::-1]) and
                np.allclose(kimage[bd].imag.array,
                            -kimage[bd].imag.array[::-1,::-1])):
            raise ValueError("Real and Imag kimages must form a Hermitian complex matrix.")

        if stepk is None:
            stepk = kimage.scale
        else:
            if stepk < kimage.scale:
                import warnings
                warnings.warn(
                    "Provided stepk is smaller than kimage.scale; overriding with kimage.scale.")
                stepk = kimage.scale

        # Default to dk=1
        if stepk is None:
            stepk = 1.
            self._kimage.scale = 1.

        self._stepk = stepk

        stepk_image = stepk / self._kimage.scale  # usually 1, but could be larger

        self._gsparams = gsparams

        # set up k_interpolant if none was provided by user, or check that the user-provided one
        # is of a valid type
        if k_interpolant is None:
            self.k_interpolant = galsim.Quintic(tol=1e-4)
        else:
            self.k_interpolant = galsim.utilities.convert_interpolant(k_interpolant)

        sbiki = _galsim.SBInterpolatedKImage(
                self._kimage.image, stepk_image, self.k_interpolant, gsparams)
        self._sbiki = sbiki

        if kimage.wcs is not None:
            sbp = _galsim.SBTransform(sbiki, 1./kimage.scale, 0., 0., 1./kimage.scale,
                                      galsim.PositionD(0.,0.), kimage.scale**2, gsparams)
        else:
            sbp = sbiki
        sbp = _galsim.SBAdd([sbp])

        GSObject.__init__(self, sbp)

    def __eq__(self, other):
        return (isinstance(other, galsim.InterpolatedKImage) and
                np.array_equal(self._kimage.array, other._kimage.array) and
                self._kimage.scale == other._kimage.scale and
                self.k_interpolant == other.k_interpolant and
                self._stepk == other._stepk and
                self._gsparams == other._gsparams)

    def __hash__(self):
        # Definitely want to cache this, since the kimage could be large.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.InterpolatedKImage", self.k_interpolant, self._stepk,
                               self._gsparams))
            self._hash ^= hash(tuple(self._kimage.array.ravel()))
            self._hash ^= hash((self._kimage.bounds, self._kimage.wcs))
        return self._hash

    def __repr__(self):
        return ('galsim.InterpolatedKImage(\n%r,\n%r, stepk=%r, gsparams=%r)')%(
                self._kimage, self.k_interpolant, self._stepk, self._gsparams)

    def __str__(self):
        return 'galsim.InterpolatedKImage(kimage=%s)'%(self._kimage)

    def __getstate__(self):
        # The SBInterpolatedKImage and the SBProfile both are picklable, but they are pretty
        # inefficient, due to the large images being written as strings.  Better to pickle
        # the intermediate products and then call init again on the other side.  There's still
        # an image to be pickled, but at least it will be through the normal pickling rules,
        # rather than the repr.
        d = self.__dict__.copy()
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._kimage, self.k_interpolant, stepk=self._stepk, gsparams=self._gsparams)


def _InterpolatedKImage(kimage, k_interpolant, gsparams):
    """Approximately equivalent to InterpolatedKImage, but with fewer options and no sanity checks.
    """
    ret = InterpolatedKImage.__new__(InterpolatedKImage)
    ret._kimage = kimage.copy()
    ret._stepk = kimage.scale
    ret._gsparams = gsparams
    ret.k_interpolant = k_interpolant
    ret._sbiki = _galsim.SBInterpolatedKImage(
            ret._kimage.image, 1.0, ret.k_interpolant, gsparams)
    sbp = _galsim.SBTransform(ret._sbiki, 1./kimage.scale, 0., 0., 1./kimage.scale,
                              galsim.PositionD(0.,0.), kimage.scale**2, gsparams)
    ret.SBProfile = _galsim.SBAdd([sbp])
    return ret


_galsim.SBInterpolatedImage.__getinitargs__ = lambda self: (
        self.getImage(), self.getXInterp(), self.getKInterp(), self.getPadFactor(),
        self.stepK(), self.maxK(), self.getGSParams())
_galsim.SBInterpolatedImage.__getstate__ = lambda self: None
_galsim.SBInterpolatedImage.__repr__ = lambda self: \
        'galsim._galsim.SBInterpolatedImage(%r, %r, %r, %r, %r, %r, %r)'%self.__getinitargs__()

_galsim.SBInterpolatedKImage.__getinitargs__ = lambda self: (
        self._getKData(), self.stepK(), self.maxK(), self.getKInterp(), self.getGSParams())
_galsim.SBInterpolatedKImage.__getstate__ = lambda self: None
_galsim.SBInterpolatedKImage.__repr__ = lambda self: (
        'galsim._galsim.SBInterpolatedKImage(%r, %r, %r, %r, %r)'
        %self.__getinitargs__())

_galsim.Interpolant.__getinitargs__ = lambda self: (self.makeStr(), self.getTol())
_galsim.Delta.__getinitargs__ = lambda self: (self.getTol(), )
_galsim.Nearest.__getinitargs__ = lambda self: (self.getTol(), )
_galsim.SincInterpolant.__getinitargs__ = lambda self: (self.getTol(), )
_galsim.Linear.__getinitargs__ = lambda self: (self.getTol(), )
_galsim.Cubic.__getinitargs__ = lambda self: (self.getTol(), )
_galsim.Quintic.__getinitargs__ = lambda self: (self.getTol(), )
_galsim.Lanczos.__getinitargs__ = lambda self: (self.getN(), self.conservesDC(), self.getTol())

_galsim.Interpolant.__repr__ = lambda self: 'galsim.Interpolant(%r, %r)'%self.__getinitargs__()
_galsim.Delta.__repr__ = lambda self: 'galsim.Delta(%r)'%self.getTol()
_galsim.Nearest.__repr__ = lambda self: 'galsim.Nearest(%r)'%self.getTol()
_galsim.SincInterpolant.__repr__ = lambda self: 'galsim.SincInterpolant(%r)'%self.getTol()
_galsim.Linear.__repr__ = lambda self: 'galsim.Linear(%r)'%self.getTol()
_galsim.Cubic.__repr__ = lambda self: 'galsim.Cubic(%r)'%self.getTol()
_galsim.Quintic.__repr__ = lambda self: 'galsim.Quintic(%r)'%self.getTol()
_galsim.Lanczos.__repr__ = lambda self: 'galsim.Lanczos(%r, %r, %r)'%self.__getinitargs__()

# Quick and dirty.  Just check reprs are equal.
_galsim.Interpolant.__eq__ = lambda self, other: repr(self) == repr(other)
_galsim.Interpolant.__ne__ = lambda self, other: not self.__eq__(other)
_galsim.Interpolant.__hash__ = lambda self: hash(repr(self))
