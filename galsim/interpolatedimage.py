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
"""@file interpolatedimage.py 

InterpolatedImage is a class that allows one to treat an image as a profile.
"""

import galsim
from galsim import GSObject, goodFFTSize


class InterpolatedImage(GSObject):
    """A class describing non-parametric profiles specified using an Image, which can be 
    interpolated for the purpose of carrying out transformations.

    The InterpolatedImage class is useful if you have a non-parametric description of an object as 
    an Image, that you wish to manipulate / transform using GSObject methods such as applyShear(),
    applyMagnification(), applyShift(), etc.  The input Image can be any BaseImage (i.e., Image,
    ImageView, or ConstImageView).  Note that when convolving an InterpolatedImage, the use of
    real-space convolution is not recommended, since it is typically a great deal slower than 
    Fourier-space convolution for this kind of object.

    The constructor needs to know how the Image was drawn: is it an Image of flux or of surface
    brightness?  Since our default for drawing Images using draw() and drawShoot() is that
    `normalization == 'flux'` (i.e., sum of pixel values equals the object flux), the default for 
    the InterpolatedImage class is to assume the same flux normalization.  However, the user can 
    specify 'surface brightness' normalization if desired, or alternatively, can instead specify 
    the desired flux for the object.

    If the input Image has a scale associated with it, then there is no need to specify an input
    scale `dx`.

    The user may optionally specify an interpolant, `x_interpolant`, for real-space manipulations
    (e.g., shearing, resampling).  If none is specified, then by default, a quintic interpolant is
    used.  The user may also choose to specify two quantities that can affect the Fourier space 
    convolution: the k-space interpolant (`k_interpolant`) and the amount of padding to include 
    around the original images (`pad_factor`).  The default values for `x_interpolant`,
    `k_interpolant`, and `pad_factor` were chosen based on the tests of branch #389 to reach good
    accuracy without being excessively slow.  Users should be particularly wary about changing 
    `k_interpolant` and `pad_factor` from the defaults without careful testing.  The user is given 
    complete freedom to choose interpolants and pad factors, and no warnings are raised when the 
    code is modified to choose some combination that is known to give significant error.  More 
    details can be found in devel/modules/finterp.pdf, especially table 1, in the GalSim 
    repository, and in comment 
    https://github.com/GalSim-developers/GalSim/issues/389#issuecomment-26166621 and following
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
                image, x_interpolant = None, k_interpolant = None, normalization = 'flux',
                dx = None, flux = None, pad_factor = 4., noise_pad_size = 0, noise_pad = 0.,
                use_cache = True, pad_image = None, rng = None, calculate_stepk = True,
                calculate_maxk = True, use_true_center = True, offset = None)

    Initializes interpolated_image as a galsim.InterpolatedImage() instance.

    For comparison of the case of padding with noise or zero when the image itself includes noise,
    compare `im1` and `im2` from the following code snippet (which can be executed from the
    examples/ directory):

        image = galsim.fits.read('data/147246.0_150.416558_1.998697_masknoise.fits')
        int_im1 = galsim.InterpolatedImage(image)
        int_im2 = galsim.InterpolatedImage(image, noise_pad='../tests/blankimg.fits')
        im1 = galsim.ImageF(1000,1000)
        im2 = galsim.ImageF(1000,1000)
        int_im1.draw(im1)
        int_im2.draw(im2)

    Examination of these two images clearly shows how padding with a correlated noise field that is
    similar to the one in the real data leads to a more reasonable appearance for the result when
    re-drawn at a different size.

    @param image           The Image from which to construct the object.
                           This may be either an Image (or ImageView) instance or a string
                           indicating a fits file from which to read the image.
    @param x_interpolant   Either an Interpolant2d (or Interpolant) instance or a string indicating
                           which real-space interpolant should be used.  Options are 'nearest',
                           'sinc', 'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the
                           integer order to use. (Default `x_interpolant = galsim.Quintic()`)
    @param k_interpolant   Either an Interpolant2d (or Interpolant) instance or a string indicating
                           which k-space interpolant should be used.  Options are 'nearest', 'sinc',
                           'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the integer
                           order to use.  We strongly recommend leaving this parameter at its
                           default value; see text above for details.  (Default `k_interpolant =
                           galsim.Quintic()`)
    @param normalization   Two options for specifying the normalization of the input Image:
                              "flux" or "f" means that the sum of the pixels is normalized
                                  to be equal to the total flux.
                              "surface brightness" or "sb" means that the pixels sample
                                  the surface brightness distribution at each location.
                              (Default `normalization = "flux"`)
    @param dx              If provided, use this as the pixel scale for the Image; this will
                           override the pixel scale stored by the provided Image, in any.  If `dx`
                           is `None`, then take the provided image's pixel scale.
                           (Default `dx = None`.)
    @param flux            Optionally specify a total flux for the object, which overrides the
                           implied flux normalization from the Image itself.
    @param pad_factor      Factor by which to pad the Image with zeros.  We strongly recommend 
                           leaving this parameter at its default value; see text above for details. 
                           (Default `pad_factor = 4`)
    @param noise_pad_size  If provided, the image will be padded out to this size (in arcsec) with 
                           the noise specified by `noise_pad`. This is important if you are 
                           planning to whiten the resulting image.  You want to make sure that the 
                           noise-padded image is larger than the postage stamp onto which you are 
                           drawing this object.  [Default `noise_pad_size = None`.]
    @param noise_pad       Noise properties to use when padding the original image with
                           noise.  This can be specified in several ways:
                               (a) as a float, which is interpreted as being a variance to use when
                                   padding with uncorrelated Gaussian noise; 
                               (b) as a galsim.CorrelatedNoise, which contains information about the
                                   desired noise power spectrum - any random number generator passed
                                   to the `rng` keyword will take precedence over that carried in an
                                   input galsim.CorrelatedNoise;
                               (c) as a galsim.Image of a noise field, which is used to calculate
                                   the desired noise power spectrum; or
                               (d) as a string which is interpreted as a filename containing an
                                   example noise field with the proper noise power spectrum.
                           It is important to keep in mind that the calculation of the correlation
                           function that is internally stored within a galsim.CorrelatedNoise is a 
                           non-negligible amount of overhead, so the recommended means of specifying
                           a correlated noise field for padding are (b) or (d).  In the case of (d),
                           if the same file is used repeatedly, then the `use_cache` keyword (see 
                           below) can be used to prevent the need for repeated 
                           galsim.CorrelatedNoise initializations.
                           (Default `noise_pad = 0.`, i.e., pad with zeros.)
    @param use_cache       Specify whether to cache noise_pad read in from a file to save having
                           to build a CorrelatedNoise object repeatedly from the same image.
                           (Default `use_cache = True`)
    @param rng             If padding by noise, the user can optionally supply the random noise
                           generator to use for drawing random numbers as `rng` (may be any kind of
                           `galsim.BaseDeviate` object).  Such a user-input random number generator
                           takes precedence over any stored within a user-input CorrelatedNoise 
                           instance (see the `noise_pad` param).
                           If `rng=None`, one will be automatically created, using the time as a
                           seed. (Default `rng = None`)
    @param pad_image       Image to be used for deterministically padding the original image.  This
                           can be specified in two ways:
                               (a) as a galsim.Image; or
                               (b) as a string which is interpreted as a filename containing an
                                   image to use.
                           The `pad_image` scale is ignored, and taken to be equal to that
                           of the `image`.
                           The user should be careful to ensure that the image used for padding has 
                           roughly zero mean.  The purpose of this keyword is to allow for a more 
                           flexible representation of some noise field around an object; if the 
                           user wishes to represent the sky level around an object, they should do 
                           that after they have drawn the final image instead.  
                           (Default `pad_image = None`.)
    @param calculate_stepk Specify whether to perform an internal determination of the extent of 
                           the object being represented by the InterpolatedImage; often this is 
                           useful in choosing an optimal value for the stepsize in the Fourier 
                           space lookup table.  
                           If you know a priori an appropriate maximum value for stepk, then 
                           you may also supply that here instead of a bool value, in which case
                           the stepk value is still calculated, but will not go above the
                           provided value. 
                           (Default `calculate_stepk = True`)
    @param calculate_maxk  Specify whether to perform an internal determination of the highest 
                           spatial frequency needed to accurately render the object being 
                           represented by the InterpolatedImage; often this is useful in choosing 
                           an optimal value for the extent of the Fourier space lookup table.
                           If you know a priori an appropriate maximum value for maxk, then 
                           you may also supply that here instead of a bool value, in which case
                           the maxk value is still calculated, but will not go above the
                           provided value. 
                           (Default `calculate_maxk = True`)
    @param use_true_center Similar to the same parameter in the GSObject.draw function, this
                           sets whether to use the true center of the provided image as the 
                           center of the profile (if `use_true_center=True`) or the nominal
                           center returned by `image.bounds.center()` (if `use_true_center=False`)
                           [Default `use_true_center = True`]
    @param offset          The location in the input image to use as the center of the profile.
                           This should be specified relative to the center of the input image 
                           (either the true center if use_true_center=True, or the nominal center 
                           if use_true_center=False).  [Default `offset = None`]
    @param gsparams        You may also specify a gsparams argument.  See the docstring for
                           galsim.GSParams using help(galsim.GSParams) for more information about
                           this option.

    Methods
    -------
    The InterpolatedImage is a GSObject, and inherits all of the GSObject methods (draw(),
    drawShoot(), applyShear() etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = { 'image' : str }
    _opt_params = {
        'x_interpolant' : str ,
        'k_interpolant' : str ,
        'normalization' : str ,
        'dx' : float ,
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
    def __init__(self, image, x_interpolant = None, k_interpolant = None, normalization = 'flux',
                 dx = None, flux = None, pad_factor = 4., noise_pad_size=0, noise_pad = 0.,
                 rng = None, pad_image = None, calculate_stepk=True, calculate_maxk=True,
                 use_cache=True, use_true_center=True, offset=None, gsparams=None):

        # first try to read the image as a file.  If it's not either a string or a valid
        # pyfits hdu or hdulist, then an exception will be raised, which we ignore and move on.
        try:
            image = galsim.fits.read(image)
        except:
            pass

        # make sure image is really an image and has a float type
        if not isinstance(image, galsim.BaseImageF) and not isinstance(image, galsim.BaseImageD):
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

        # Make sure we don't change the original image in anything we do to it here.
        # (e.g. set scale, etc.)
        image = image.view()

        # Check for input dx, and check whether Image already has one set.  At the end of this
        # code block, either an exception will have been raised, or the input image will have a
        # valid scale set.
        if dx is None:
            dx = image.scale
            if dx == 0:
                raise ValueError("No information given with Image or keywords about pixel scale!")
        else:
            if type(dx) != float:
                dx = float(dx)
            if dx <= 0.0:
                raise ValueError("dx may not be <= 0.0")
            image.scale = dx

        # Store the image as an attribute
        self.orig_image = image
        self.use_cache = use_cache

        # Set up the GaussianDeviate if not provided one, or check that the user-provided one is
        # of a valid type.
        if rng is None:
            if noise_pad: rng = galsim.BaseDeviate()
        elif not isinstance(rng, galsim.BaseDeviate):
            raise TypeError("rng provided to InterpolatedImage constructor is not a BaseDeviate")

        # Check that given pad_image is valid:
        if pad_image:
            if isinstance(pad_image, str):
                pad_image = galsim.fits.read(pad_image)
            if ( not isinstance(pad_image, galsim.BaseImageF) and 
                 not isinstance(pad_image, galsim.BaseImageD) ):
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

        # Make sure the image fits in the noise pad image:
        if noise_pad_size:
            import math
            # Convert from arcsec to pixels according to the image scale.
            noise_pad_size = int(math.ceil(noise_pad_size / image.scale))
            # Round up to a good size for doing FFTs
            noise_pad_size = goodFFTSize(noise_pad_size)
            if noise_pad_size <= min(image.array.shape):
                # Don't need any noise padding in this case.
                noise_pad_size = 0
            elif noise_pad_size < max(image.array.shape):
                noise_pad_size = max(image.array.shape)

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
            if ( isinstance(image, galsim.BaseImageF) and 
                 not isinstance(pad_image, galsim.BaseImageF) ):
                pad_image = galsim.ImageF(pad_image)
            elif ( isinstance(image, galsim.BaseImageD) and 
                   not isinstance(pad_image, galsim.BaseImageD) ):
                pad_image = galsim.ImageD(pad_image)

        # Now place the given image in the center of the padding image:
        if pad_image:
            pad_image.setCenter(0,0)
            image.setCenter(0,0)
            if pad_image.bounds.includes(image.bounds):
                pad_image[image.bounds] = image
            else:
                # If padding was smaller than original image, just use the original image.
                pad_image = image
        else:
            pad_image = image

        # Make the SBInterpolatedImage out of the image.
        sbinterpolatedimage = galsim.SBInterpolatedImage(
                pad_image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                dx=dx, pad_factor=pad_factor, gsparams=gsparams)

        # GalSim cannot automatically know what stepK and maxK are appropriate for the 
        # input image.  So it is usually worth it to do a manual calculation here.
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

        # If the user specified a flux, then set to that flux value.
        if flux != None:
            if type(flux) != flux:
                flux = float(flux)
            sbinterpolatedimage.setFlux(flux)
        # If the user specified a flux normalization for the input Image, then since
        # SBInterpolatedImage works in terms of surface brightness, have to rescale the values to
        # get proper normalization.
        elif flux is None and normalization.lower() in ['flux','f'] and dx != 1.:
            sbinterpolatedimage.scaleFlux(1./(dx**2))
        # If the input Image normalization is 'sb' then since that is the SBInterpolated default
        # assumption, no rescaling is needed.

        # Initialize the SBProfile
        GSObject.__init__(self, sbinterpolatedimage)

        # Apply the offset, and possibly fix the centering for even-sized images
        # Note reverse=True, since we want to fix the center in the opposite sense of what the 
        # draw function does.
        prof = self._fix_center(image, dx, offset, use_true_center, reverse=True)
        GSObject.__init__(self, prof.SBProfile)


    def buildNoisePadImage(self, noise_pad_size, noise_pad, rng):
        """A helper function that builds the pad_image from the given noise_pad specification.
        """
        import numpy as np
        if isinstance(self.orig_image, galsim.BaseImageF):
            pad_image = galsim.ImageF(noise_pad_size, noise_pad_size)
        if isinstance(self.orig_image, galsim.BaseImageD):
            pad_image = galsim.ImageD(noise_pad_size, noise_pad_size)

        # Figure out what kind of noise to apply to the image
        if isinstance(noise_pad, float):
            noise = galsim.GaussianNoise(rng, sigma = np.sqrt(noise_pad))
        elif isinstance(noise_pad, galsim.correlatednoise._BaseCorrelatedNoise):
            noise = noise_pad.copy()
            if rng: # Let a user supplied RNG take precedence over that in user CN
                noise.setRNG(rng)
        elif isinstance(noise_pad,galsim.BaseImageF) or isinstance(noise_pad,galsim.BaseImageD):
            noise = galsim.CorrelatedNoise(rng, noise_pad)
        elif self.use_cache and noise_pad in InterpolatedImage._cache_noise_pad:
            noise = InterpolatedImage._cache_noise_pad[noise_pad]
            if rng:
                # Make sure that we are using a specified RNG by resetting that in this cached
                # CorrelatedNoise instance, otherwise preserve the cached RNG
                noise.setRNG(rng)
        elif isinstance(noise_pad, str):
            noise = galsim.CorrelatedNoise(rng, galsim.fits.read(noise_pad))
            if self.use_cache: 
                InterpolatedImage._cache_noise_pad[noise_pad] = noise
        else:
            raise ValueError(
                "Input noise_pad must be a float/int, a CorrelatedNoise, Image, or filename "+
                "containing an image to use to make a CorrelatedNoise!")
        # Add the noise
        pad_image.addNoise(noise)

        return pad_image

