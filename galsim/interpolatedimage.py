# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

import galsim
from galsim import GSObject
from . import _galsim
from ._galsim import Interpolant, Interpolant2d, InterpolantXY
from ._galsim import Nearest, Linear, Cubic, Quintic, Lanczos, SincInterpolant, Delta

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
                use_true_center=True, offset=None)

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
                            file from which to read the image.
    @param x_interpolant    Either an Interpolant2d (or Interpolant) instance or a string indicating
                            which real-space interpolant should be used.  Options are 'nearest',
                            'sinc', 'linear', 'cubic', 'quintic', or 'lanczosN' where N should be
                            the integer order to use. [default: Quintic]
    @param k_interpolant    Either an Interpolant2d (or Interpolant) instance or a string indicating
                            which k-space interpolant should be used.  Options are 'nearest',
                            'sinc', 'linear', 'cubic', 'quintic', or 'lanczosN' where N should be
                            the integer order to use.  We strongly recommend leaving this parameter
                            at its default value; see text above for details.  [default:
                            Quintic]
    @param normalization    Two options for specifying the normalization of the input Image:
                              "flux" or "f" means that the sum of the pixels is normalized
                                  to be equal to the total flux.
                              "surface brightness" or "sb" means that the pixels sample
                                  the surface brightness distribution at each location.
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
                                   example noise field with the proper noise power spectrum.
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
                                   image to use.
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

    Methods
    -------

    There are no additional methods for InterpolatedImage beyond the usual GSObject methods.
    """

    # Initialization parameters of the object, with type information
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
        'use_true_center' : bool
    }
    _single_params = []
    _takes_rng = True
    _takes_logger = False
    _cache_noise_pad = {}

    # --- Public Class methods ---
    def __init__(self, image, x_interpolant=None, k_interpolant=None, normalization='flux',
                 scale=None, wcs=None, flux=None, pad_factor=4., noise_pad_size=0, noise_pad=0.,
                 rng=None, pad_image=None, calculate_stepk=True, calculate_maxk=True,
                 use_cache=True, use_true_center=True, offset=None, gsparams=None, dx=None,
                 _force_stepk=None, _force_maxk=None):
        # Check for obsolete dx parameter
        if dx is not None and scale is None: scale = dx

        import numpy

        # first try to read the image as a file.  If it's not either a string or a valid
        # pyfits hdu or hdulist, then an exception will be raised, which we ignore and move on.
        try:
            image = galsim.fits.read(image)
        except:
            pass

        # make sure image is really an image and has a float type
        if not isinstance(image, galsim.Image):
            raise ValueError("Supplied image is not an Image instance")
        if image.dtype != numpy.float32 and image.dtype != numpy.float64:
            raise ValueError("Supplied image is not an image of floats or doubles!")

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
            self.x_interpolant = galsim.InterpolantXY(galsim.Quintic(tol=1e-4))
        else:
            self.x_interpolant = galsim.utilities.convert_interpolant_to_2d(x_interpolant)
        if k_interpolant is None:
            self.k_interpolant = galsim.InterpolantXY(galsim.Quintic(tol=1e-4))
        else:
            self.k_interpolant = galsim.utilities.convert_interpolant_to_2d(k_interpolant)

        # Store the image as an attribute and make sure we don't change the original image
        # in anything we do here.  (e.g. set scale, etc.)
        self.image = image.view()
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
            import numpy
            if isinstance(pad_image, str):
                pad_image = galsim.fits.read(pad_image)
            if not isinstance(pad_image, galsim.Image):
                raise ValueError("Supplied pad_image is not an Image!")
            if pad_image.dtype != numpy.float32 and pad_image.dtype != numpy.float64:
                raise ValueError("Supplied pad_image is not one of the allowed types!")

        # Check that the given noise_pad is valid:
        try:
            noise_pad = float(noise_pad)
        except:
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

        # Make sure the image fits in the noise pad image:
        if noise_pad_size:
            import math
            # Convert from arcsec to pixels according to the local wcs.
            # Use the minimum scale, since we want to make sure noise_pad_size is
            # as large as we need in any direction.
            scale = local_wcs.minLinearScale()
            noise_pad_size = int(math.ceil(noise_pad_size / scale))
            # Round up to a good size for doing FFTs
            noise_pad_size = galsim._galsim.goodFFTSize(noise_pad_size)
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
                pad_image = pad_image.view()  
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
            else:
                # If padding was smaller than original image, just use the original image.
                pad_image = self.image
        else:
            pad_image = self.image

        # Make the SBInterpolatedImage out of the image.
        sbinterpolatedimage = galsim._galsim.SBInterpolatedImage(
                pad_image.image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                pad_factor=pad_factor, gsparams=gsparams)

        # GalSim cannot automatically know what stepK and maxK are appropriate for the 
        # input image.  So it is usually worth it to do a manual calculation here.
        #
        # However, there is also a hidden option to force it to use specific values of stepK and
        # maxK (caveat user!).  The values of _force_stepk and _force_maxk should be provided in
        # terms of physical scale, e.g., for images that have a scale length of 0.1 arcsec, the
        # stepK and maxK should be provided in units of 1/arcsec.  Then we convert to the 1/pixel
        # units required by the C++ layer below.  Also note that profile recentering for even-sized
        # images (see the ._fix_center step below) leads to automatic reduction of stepK slightly
        # below what is provided here, while maxK is preserved.
        if _force_stepk is not None:
            calculate_stepk = False
            sbinterpolatedimage.forceStepK(_force_stepk*image.scale)
        if _force_maxk is not None:
            calculate_maxk = False
            sbinterpolatedimage.forceMaxK(_force_maxk*image.scale)
        if calculate_stepk:
            if calculate_stepk is True:
                sbinterpolatedimage.calculateStepK()
            else:
                # If not a bool, then value is max_stepk
                sbinterpolatedimage.calculateStepK(max_stepk=calculate_stepk)
        if calculate_maxk:
            if calculate_maxk is True:
                sbinterpolatedimage.calculateMaxK()
            else:
                # If not a bool, then value is max_maxk
                sbinterpolatedimage.calculateMaxK(max_maxk=calculate_maxk)

        # Initialize the SBProfile
        GSObject.__init__(self, sbinterpolatedimage)

        # Make sure offset is a PositionD
        offset = self._parse_offset(offset)

        # Apply the offset, and possibly fix the centering for even-sized images
        # Note reverse=True, since we want to fix the center in the opposite sense of what the 
        # draw function does.
        prof = self._fix_center(self.image, offset, use_true_center, reverse=True)

        # Bring the profile from image coordinates into world coordinates
        prof = local_wcs.toWorld(prof)

        # If the user specified a flux, then set to that flux value.
        if flux is not None:
            prof = prof.withFlux(float(flux))
        # If the user specified a surface brightness normalization for the input Image, then
        # need to rescale flux by the pixel area to get proper normalization.
        elif normalization.lower() in ['surface brightness','sb']:
            prof *= local_wcs.pixelArea()

        GSObject.__init__(self, prof)

    def buildNoisePadImage(self, noise_pad_size, noise_pad, rng):
        """A helper function that builds the `pad_image` from the given `noise_pad` specification.
        """
        import numpy as np

        # Make it with the same dtype as the image
        pad_image = galsim.Image(noise_pad_size, noise_pad_size, dtype=self.image.dtype)

        # Figure out what kind of noise to apply to the image
        if isinstance(noise_pad, float):
            noise = galsim.GaussianNoise(rng, sigma = np.sqrt(noise_pad))
        elif isinstance(noise_pad, galsim.correlatednoise._BaseCorrelatedNoise):
            noise = noise_pad.copy()
            if rng: # Let a user supplied RNG take precedence over that in user CN
                noise.setRNG(rng)
        elif isinstance(noise_pad,galsim.Image):
            noise = galsim.CorrelatedNoise(noise_pad, rng)
        elif self.use_cache and noise_pad in InterpolatedImage._cache_noise_pad:
            noise = InterpolatedImage._cache_noise_pad[noise_pad]
            if rng:
                # Make sure that we are using a specified RNG by resetting that in this cached
                # CorrelatedNoise instance, otherwise preserve the cached RNG
                noise.setRNG(rng)
        elif isinstance(noise_pad, str):
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

