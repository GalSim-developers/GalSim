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
from galsim import GSObject


class InterpolatedImage(GSObject):
    """A class describing non-parametric objects specified using an Image, which can be interpolated
    for the purpose of carrying out transformations.

    The input Image and optional interpolants are used to create an SBInterpolatedImage.  The
    InterpolatedImage class is useful if you have a non-parametric description of an object as an
    Image, that you wish to manipulate / transform using GSObject methods such as applyShear(),
    applyMagnification(), applyShift(), etc.  The input Image can be any BaseImage (i.e., Image,
    ImageView, or ConstImageView).

    The constructor needs to know how the Image was drawn: is it an Image of flux or of surface
    brightness?  Since our default for drawing Images using draw() and drawShoot() is that
    `normalization == 'flux'` (i.e., sum of pixel values equals the object flux), the default for 
    the InterpolatedImage class is to assume the same flux normalization.  However, the user can 
    specify 'surface brightness' normalization if desired, or alternatively, can instead specify 
    the desired flux for the object.

    If the input Image has a scale associated with it, then there is no need to specify an input
    scale `dx`.

    The user may optionally specify an interpolant, `x_interpolant`, for real-space manipulations
    (e.g., shearing, resampling).  If none is specified, then by default, a 5th order Lanczos
    interpolant is used.  The user may also choose to specify two quantities that can affect the
    Fourier space convolution: the k-space interpolant (`k_interpolant`) and the amount of padding
    to include around the original images (`pad_factor`).  The default values for `x_interpolant`,
    `k_interpolant`, and `pad_factor` were chosen based on preliminary tests suggesting that they
    lead to a high degree of accuracy without being excessively slow.  Users should be particularly
    wary about changing `k_interpolant` and `pad_factor` from the defaults without careful testing.
    The user is given complete freedom to choose interpolants and pad factors, and no warnings are
    raised when the code is modified to choose some combination that is known to give significant
    error.  More details can be found in devel/modules/finterp.pdf, especially table 1, in the
    GalSim repository.

    The user can choose to have the image padding use zero (default), Gaussian random noise of some
    variance, or a Gaussian but correlated noise field that is specified either as a 
    CorrelatedNoise instance, an Image (from which a correlated noise model is derived), or a string
    (interpreted as a filename containing an image to use for deriving a CorrelatedNoise).  The user
    can also pass in a random number generator to be used for noise generation.  Finally, the user
    can pass in a `pad_image` for deterministic image padding.

    By default, the InterpolatedImage recalculates the Fourier-space step and number of points to
    use for further manipulations, rather than using the most conservative possibility.  For typical
    objects representing galaxies and PSFs this can easily make the difference between several
    seconds (conservative) and 0.04s (recalculated).  However, the user can turn off this option,
    and may especially wish to do so when using images that do not contain a high S/N object - e.g.,
    images of noise fields.

    Initialization
    --------------
    
        >>> interpolated_image = galsim.InterpolatedImage(image, x_interpolant = None,
                                                          k_interpolant = None,
                                                          normalization = 'f', dx = None,
                                                          flux = None, pad_factor = 0.,
                                                          noise_pad = 0., rng = None,
                                                          pad_image = None,
                                                          calculate_stepk = True,
                                                          calculate_maxk = True,
                                                          use_cache = True)

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
    @param pad_factor      Factor by which to pad the Image when creating the SBInterpolatedImage;
                           `pad_factor <= 0` results in the use of the default value, 4.  We
                           strongly recommend leaving this parameter at its default value; see text
                           above for details.
                           (Default `pad_factor = 0`, unless a `pad_image` is passed in, which
                           results in a default value of `pad_factor = 1`.)
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
                           a correlated noise field for padding are (b) or (d). In the case of (d),
                           if the same file is used repeatedly, then the `use_cache` keyword (see 
                           below) can be used to prevent the need for repeated 
                           galsim.CorrelatedNoise initializations.
                           (Default `noise_pad = 0.`, i.e., pad with zeros.)
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
                           The size of the image that is passed in is taken to specify the amount of
                           padding, and so the `pad_factor` keyword should be equal to 1, i.e., no
                           padding.  The `pad_image` scale is ignored, and taken to be equal to that
                           of the `image`. Note that `pad_image` can be used together with
                           `noise_pad`.  However, the user should be careful to ensure that the
                           image used for padding has roughly zero mean.  The purpose of this
                           keyword is to allow for a more flexible representation of some noise
                           field around an object; if the user wishes to represent the sky level
                           around an object, they should do that when they have drawn the final
                           image instead.  (Default `pad_image = None`.)
    @param calculate_stepk Specify whether to perform an internal determination of the extent of 
                           the object being represented by the InterpolatedImage; often this is 
                           useful in choosing an optimal value for the stepsize in the Fourier 
                           space lookup table. (Default `calculate_stepk = True`)
    @param calculate_maxk  Specify whether to perform an internal determination of the highest 
                           spatial frequency needed to accurately render the object being 
                           represented by the InterpolatedImage; often this is useful in choosing 
                           an optimal value for the extent of the Fourier space lookup table.
                           (Default `calculate_maxk = True`)
    @param use_cache       Specify whether to cache noise_pad read in from a file to save having
                           to build a CorrelatedNoise object repeatedly from the same image.
                           (Default `use_cache = True`)
    @param use_true_center Similar to the same parameter in the GSObject.draw function, this
                           sets whether to use the true center of the provided image as the 
                           center of the profile (if `use_true_center=True`) or the nominal
                           center returned by `image.bounds.center()` (if `use_true_center=False`)
                           [default `use_true_center = True`]
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
        'noise_pad' : str ,
        'pad_image' : str ,
        'calculate_stepk' : bool ,
        'calculate_maxk' : bool,
        'use_true_center' : bool
    }
    _single_params = []
    _takes_rng = True
    _cache_noise_pad = {}

    # --- Public Class methods ---
    def __init__(self, image, x_interpolant = None, k_interpolant = None, normalization = 'flux',
                 dx = None, flux = None, pad_factor = 0., noise_pad = 0., rng = None,
                 pad_image = None, calculate_stepk=True, calculate_maxk=True,
                 use_cache=True, use_true_center=True, gsparams=None):

        import numpy as np

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
        if not image.getBounds().isDefined():
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
            # Don't change the original image.  Make a new view if we need to set the scale.
            image = image.view()
            image.setScale(dx)
            if dx == 0.0:
                raise ValueError("dx may not be 0.0")

        # Set up the GaussianDeviate if not provided one, or check that the user-provided one is
        # of a valid type.
        if rng is None:
            gaussian_deviate = galsim.GaussianDeviate()
        elif isinstance(rng, galsim.BaseDeviate):
            # Even if it's already a GaussianDeviate, we still want to make a new Gaussian deviate
            # that would generate the same sequence, because later we change the sigma and we don't
            # want to change it for the original one that was passed in.  So don't distinguish
            # between GaussianDeviate and the other BaseDeviates here.
            gaussian_deviate = galsim.GaussianDeviate(rng)
        else:
            raise TypeError("rng provided to InterpolatedImage constructor is not a BaseDeviate")

        # decide about deterministic image padding
        specify_size = False
        padded_size = image.getPaddedSize(pad_factor)
        if pad_image:
            specify_size = True
            if isinstance(pad_image, str):
                pad_image = galsim.fits.read(pad_image)
            if ( not isinstance(pad_image, galsim.BaseImageF) and 
                 not isinstance(pad_image, galsim.BaseImageD) ):
                raise ValueError("Supplied pad_image is not one of the allowed types!")

            # If an image was supplied directly or from a file, check its size:
            #    Cannot use if too small.
            #    Use to define the final image size otherwise.
            deltax = (1+pad_image.getXMax()-pad_image.getXMin())-(1+image.getXMax()-image.getXMin())
            deltay = (1+pad_image.getYMax()-pad_image.getYMin())-(1+image.getYMax()-image.getYMin())
            if deltax < 0 or deltay < 0:
                raise RuntimeError("Image supplied for padding is too small!")
            if pad_factor != 1. and pad_factor != 0.:
                import warnings
                msg =  "Warning: ignoring specified pad_factor because user also specified\n"
                msg += "         an image to use directly for the padding."
                warnings.warn(msg)
        elif noise_pad:
            if isinstance(image, galsim.BaseImageF):
                pad_image = galsim.ImageF(padded_size, padded_size)
            if isinstance(image, galsim.BaseImageD):
                pad_image = galsim.ImageD(padded_size, padded_size)

        # now decide about noise padding
        # First, see if the input is consistent with a float.
        # i.e. it could be an int, or a str that converts to a number.
        try:
            noise_pad = float(noise_pad)
        except:
            pass
        if isinstance(noise_pad, float):
            if noise_pad < 0.:
                raise ValueError("Noise variance cannot be negative!")
            elif noise_pad > 0.:
                # Note: make sure the sigma is properly set to sqrt(noise_pad).
                gaussian_deviate.setSigma(np.sqrt(noise_pad))
                pad_image.addNoise(galsim.DeviateNoise(gaussian_deviate))
        else:
            if isinstance(noise_pad, galsim.correlatednoise._BaseCorrelatedNoise):
                cn = noise_pad.copy()
                if rng: # Let a user supplied RNG take precedence over that in user CN
                    cn.setRNG(gaussian_deviate)
            elif isinstance(noise_pad,galsim.BaseImageF) or isinstance(noise_pad,galsim.BaseImageD):
                cn = galsim.CorrelatedNoise(gaussian_deviate, noise_pad)
            elif use_cache and noise_pad in InterpolatedImage._cache_noise_pad:
                cn = InterpolatedImage._cache_noise_pad[noise_pad]
                if rng:
                    # Make sure that we are using a specified RNG by resetting that in this cached
                    # CorrelatedNoise instance, otherwise preserve the cached RNG
                    cn.setRNG(gaussian_deviate)
            elif isinstance(noise_pad, str):
                cn = galsim.CorrelatedNoise(gaussian_deviate, galsim.fits.read(noise_pad))
                if use_cache: 
                    InterpolatedImage._cache_noise_pad[noise_pad] = cn
            else:
                raise ValueError(
                    "Input noise_pad must be a float/int, a CorrelatedNoise, Image, or filename "+
                    "containing an image to use to make a CorrelatedNoise!")
            pad_image.addNoise(cn)

        # Now we have to check: was the padding determined using pad_factor?  Or by passing in an
        # image for padding?  Treat these cases differently:
        # (1) If the former, then we can simply have the C++ handle the padding process.
        # (2) If the latter, then we have to do the padding ourselves, and pass the resulting image
        # to the C++ with pad_factor explicitly set to 1.
        if specify_size is False:
            # Make the SBInterpolatedImage out of the image.
            sbinterpolatedimage = galsim.SBInterpolatedImage(
                    image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                    dx=dx, pad_factor=pad_factor, pad_image=pad_image, gsparams=gsparams)
            self.x_size = padded_size
            self.y_size = padded_size
        else:
            # Leave the original image as-is.  Instead, we shift around the image to be used for
            # padding.  Find out how much x and y margin there should be on lower end:
            x_marg = int(np.round(0.5*deltax))
            y_marg = int(np.round(0.5*deltay))
            # Now reset the pad_image to contain the original image in an even way
            pad_image = pad_image.view()
            pad_image.setScale(dx)
            pad_image.setOrigin(image.getXMin()-x_marg, image.getYMin()-y_marg)
            # Set the central values of pad_image to be equal to the input image
            pad_image[image.bounds] = image
            sbinterpolatedimage = galsim.SBInterpolatedImage(
                    pad_image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                    dx=dx, pad_factor=1., gsparams=gsparams)
            self.x_size = 1+pad_image.getXMax()-pad_image.getXMin()
            self.y_size = 1+pad_image.getYMax()-pad_image.getYMin()

        # GalSim cannot automatically know what stepK and maxK are appropriate for the 
        # input image.  So it is usually worth it to do a manual calculation here.
        if calculate_stepk:
            sbinterpolatedimage.calculateStepK()
        if calculate_maxk:
            sbinterpolatedimage.calculateMaxK()

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

        # Fix the center to be in the right place.
        # Note the minus sign in front of image.scale, since we want to fix the center in the 
        # opposite sense of what the draw function does.
        if use_true_center:
            prof = self._fix_center(image, -image.scale)
            GSObject.__init__(self, prof.SBProfile)
            


