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
"""@file hsm.py
Routines for adaptive moment estimation and PSF correction.

This file contains the python interface to C++ routines for estimation of second moments of images,
and for carrying out PSF correction using a variety of algorithms.  The algorithms are described in
Hirata & Seljak (2003; MNRAS, 343, 459), and were tested/characterized using real data in Mandelbaum
et al. (2005; MNRAS, 361, 1287).  We also define a python-level container for the outputs of these
codes, ShapeData, analogous to the C++-level CppShapeData.  Note that these routines for
moment measurement and shear estimation are not accessible via config, only via python.  There are a
number of default settings for the code (often governing the tradeoff between accuracy and speed)
that can be adjusting using an optional `hsmparams` argument as described below.

The moments that are estimated are "adaptive moments" (see the first paper cited above for details);
that is, they use an elliptical Gaussian weight that is matched to the image of the object being
measured.  The observed moments can be represented as a Gaussian sigma and a Shear object
representing the shape.

The PSF correction includes several algorithms, three that are re-implementations of methods
originated by others and one that was originated by Hirata & Seljak:

- One from Kaiser, Squires, & Broadhurts (1995), "KSB"

- One from Bernstein & Jarvis (2002), "BJ"

- One that represents a modification by Hirata & Seljak (2003) of methods in Bernstein & Jarvis
(2002), "LINEAR"

- One method from Hirata & Seljak (2003), "REGAUSS" (re-Gaussianization)

These methods return shear (or shape) estimators, which may not in fact satisfy conditions like
|e|<=1, and so they are represented simply as e1/e2 or g1/g2 (depending on the method) rather than
using a Shear object, which IS required to satisfy |e|<=1.

These methods are all based on correction of moments, but with different sets of assumptions.  For
more detailed discussion on all of these algorithms, see the relevant papers above.

Users can find a listing of the parameters that can be adjusted using the `hsmparams` keyword, along
with default values, using help(galsim.hsm.HSMParams).
"""


from . import _galsim
import galsim
from ._galsim import HSMParams
import numpy as np


class ShapeData(object):
    """A class to contain the outputs of using the HSM shape and moments measurement routines.

    At the C++ level, we have a container for the outputs of the HSM shape measurement routines.
    The ShapeData class is the analogous object at the python level.  It contains the following
    information about moment measurement (from either EstimateShear() or FindAdaptiveMom()):

    - image_bounds: a BoundsI object describing the image.

    - moments_status: the status flag resulting from moments measurement; -1 indicates no attempt to
      measure, 0 indicates success.

    - observed_shape: a Shear object representing the observed shape based on adaptive
      moments.

    - moments_sigma: size sigma=(det M)^(1/4) from the adaptive moments, in units of pixels; -1 if
      not measured.

    - moments_amp: total image intensity for best-fit elliptical Gaussian from adaptive moments.
      Normally, this field is simply equal to the image flux (for objects that follow a Gaussian
      light distribution, otherwise it is something approximating the flux).  However, if the image
      was drawn using `drawImage(method='sb')` then moments_amp relates to the flux via
      flux=(moments_amp)*(pixel scale)^2.

    - moments_centroid: a PositionD object representing the centroid based on adaptive moments, in
      units of pixels.  The indexing convention is defined with respect to the BoundsI object
      defining the bounds of the input Image, i.e., the center of the lower left pixel is
      `(image.xmin, image.ymin)`.  An object drawn at the center of the image should generally have
      moments_centroid equal to image.true_center.

    - moments_rho4: the weighted radial fourth moment of the image.

    - moments_n_iter: number of iterations needed to get adaptive moments, or 0 if not measured.

    If EstimateShear() was used, then the following fields related to PSF-corrected shape will also
    be populated:

    - correction_status: the status flag resulting from PSF correction; -1 indicates no attempt to
      measure, 0 indicates success.

    - corrected_e1, corrected_e2, corrected_g1, corrected_g2: floats representing the estimated
      shear after removing the effects of the PSF.  Either e1/e2 or g1/g2 will differ from the
      default values of -10, with the choice of shape to use determined by the correction method
      (since the correction method determines what quantity is estimated, either the shear or the
      distortion).  After a measurement is made, the type of shape measurement is stored in the
      ShapeData structure as 'meas_type' (a string that equals either 'e' or 'g').

    - corrected_shape_err: shape measurement uncertainty sigma_gamma per component.  The estimate of
      the uncertainty will only be non-zero if an estimate of the sky variance was passed to
      EstimateShear().

    - correction_method: a string indicating the method of PSF correction (will be "None" if
      PSF-correction was not carried out).

    - resolution_factor: Resolution factor R_2;  0 indicates object is consistent with a PSF, 1
      indicates perfect resolution.

    - psf_sigma: size sigma=(det M)^(1/4) of the PSF from the adaptive moments, in units of pixels;
      -1 if not measured.

    - psf_shape: a Shear object representing the observed PSF shape based on adaptive moments.

    - error_message: a string containing any error messages from the attempt to carry out
      PSF-correction.

    The ShapeData object can be initialized completely empty, or can be returned from the
    routines that measure object moments (FindAdaptiveMom()) and carry out PSF correction
    (EstimateShear()).
    """
    def __init__(self, *args, **kwargs):
        # arg checking: require either a CppShapeData, or nothing
        if len(args) > 1:
            raise TypeError("Too many arguments to initialize ShapeData!")
        elif len(args) == 1:
            if isinstance(args[0], ShapeData):
                data = args[0]
                self.observed_shape = data.observed_shape
                self.psf_shape = data.psf_shape
            elif isinstance(args[0], _galsim.CppShapeData):
                data = args[0]
                self.observed_shape = galsim.Shear(e1=data.observed_e1, e2=data.observed_e2)
                self.psf_shape = galsim.Shear(e1=data.psf_e1, e2=data.psf_e2)
            else:
                raise TypeError("Non keyword argument must be a ShapeData or CppShapeData!")
            self.image_bounds = data.image_bounds
            self.moments_status = data.moments_status
            self.moments_sigma = data.moments_sigma
            self.moments_amp = data.moments_amp
            self.moments_centroid = data.moments_centroid
            self.moments_rho4 = data.moments_rho4
            self.moments_n_iter = data.moments_n_iter
            self.correction_status = data.correction_status
            self.corrected_e1 = data.corrected_e1
            self.corrected_e2 = data.corrected_e2
            self.corrected_g1 = data.corrected_g1
            self.corrected_g2 = data.corrected_g2
            self.meas_type = data.meas_type
            self.corrected_shape_err = data.corrected_shape_err
            self.correction_method = data.correction_method
            self.resolution_factor = data.resolution_factor
            self.psf_sigma = data.psf_sigma
            # We use "None" in CppShapeData to indicate no error messages to avoid problems on
            # (some) Macs using zero-length strings.  Here, we revert that back to "".
            if data.error_message == "None":
                self.error_message = ""
            else:
                self.error_message = data.error_message
        else:
            self.image_bounds = _galsim.BoundsI()
            self.moments_status = -1
            self.observed_shape = galsim.Shear()
            self.moments_sigma = -1.0
            self.moments_amp = -1.0
            self.moments_centroid = _galsim.PositionD()
            self.moments_rho4 = -1.0
            self.moments_n_iter = 0
            self.correction_status = -1
            self.corrected_e1 = -10.
            self.corrected_e2 = -10.
            self.corrected_g1 = -10.
            self.corrected_g2 = -10.
            self.meas_type = "None"
            self.corrected_shape_err = -1.0
            self.correction_method = "None"
            self.resolution_factor = -1.0
            self.psf_sigma = -1.0
            self.psf_shape = galsim.Shear()
            self.error_message = ""

        self.image_bounds = kwargs.pop('image_bounds', self.image_bounds)
        self.moments_status = kwargs.pop('moments_status', self.moments_status)
        self.observed_shape = kwargs.pop('observed_shape', self.observed_shape)
        self.moments_sigma = kwargs.pop('moments_sigma', self.moments_sigma)
        self.moments_amp = kwargs.pop('moments_amp', self.moments_amp)
        self.moments_centroid = kwargs.pop('moments_centroid', self.moments_centroid)
        self.moments_rho4 = kwargs.pop('moments_rho4', self.moments_rho4)
        self.moments_n_iter = kwargs.pop('moments_n_iter', self.moments_n_iter)
        self.correction_status = kwargs.pop('correction_status', self.correction_status)
        self.corrected_e1 = kwargs.pop('corrected_e1', self.corrected_e1)
        self.corrected_e2 = kwargs.pop('corrected_e2', self.corrected_e2)
        self.corrected_g1 = kwargs.pop('corrected_g1', self.corrected_g1)
        self.corrected_g2 = kwargs.pop('corrected_g2', self.corrected_g2)
        self.meas_type = kwargs.pop('meas_type', self.meas_type)
        self.corrected_shape_err = kwargs.pop('corrected_shape_err', self.corrected_shape_err)
        self.correction_method = kwargs.pop('correction_method', self.correction_method)
        self.resolution_factor = kwargs.pop('resolution_factor', self.resolution_factor)
        self.psf_sigma = kwargs.pop('psf_sigma', self.psf_sigma)
        self.psf_shape = kwargs.pop('psf_shape', self.psf_shape)
        self.error_message = kwargs.pop('error_message', self.error_message)
        if kwargs:
            raise TypeError(
                "ShapeData constructor got unexpected extra argument(s): %s"%kwargs.keys())

    def __repr__(self):
        s = 'galsim.hsm.ShapeData('
        if self.image_bounds.isDefined(): s += 'image_bounds=%r, '%self.image_bounds
        if self.moments_status != -1: s += 'moments_status=%r, '%self.moments_status
        # Always include this one:
        s += 'observed_shape=%r'%self.observed_shape
        if self.moments_sigma != -1: s += ', moments_sigma=%r'%self.moments_sigma
        if self.moments_amp != -1: s += ', moments_amp=%r'%self.moments_amp
        if self.moments_centroid != galsim.PositionD():
            s += ', moments_centroid=%r'%self.moments_centroid
        if self.moments_rho4 != -1: s += ', moments_rho4=%r'%self.moments_rho4
        if self.moments_n_iter != 0: s += ', moments_n_iter=%r'%self.moments_n_iter
        if self.correction_status != -1: s += ', correction_status=%r'%self.correction_status
        if self.corrected_e1 != -10.: s += ', corrected_e1=%r'%self.corrected_e1
        if self.corrected_e2 != -10.: s += ', corrected_e2=%r'%self.corrected_e2
        if self.corrected_g1 != -10.: s += ', corrected_g1=%r'%self.corrected_g1
        if self.corrected_g2 != -10.: s += ', corrected_g2=%r'%self.corrected_g2
        if self.meas_type != 'None': s += ', meas_type=%r'%self.meas_type
        if self.corrected_shape_err != -1.:
            s += ', corrected_shape_err=%r'%self.corrected_shape_err
        if self.correction_method != 'None': s += ', correction_method=%r'%self.correction_method
        if self.resolution_factor != -1.: s += ', resolution_factor=%r'%self.resolution_factor
        if self.psf_sigma != -1.: s += ', psf_sigma=%r'%self.psf_sigma
        if self.psf_shape != galsim.Shear(): s += ', psf_shape=%r'%self.psf_shape
        if self.error_message != "": s += ', error_message=%r'%self.error_message
        s += ')'
        return s

    # Quick and dirty.  Just check reprs are equal.
    def __eq__(self, other): return repr(self) == repr(other)
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(repr(self))

_galsim.CppShapeData.__getinitargs__ = lambda self: (
        self.image_bounds, self.moments_status, self.observed_e1, self.observed_e2,
        self.moments_sigma, self.moments_amp, self.moments_centroid, self.moments_rho4,
        self.moments_n_iter, self.correction_status, self.corrected_e1, self.corrected_e2,
        self.corrected_g1, self.corrected_g2, self.meas_type, self.corrected_shape_err,
        self.correction_method, self.resolution_factor, self.psf_sigma,
        self.psf_e1, self.psf_e2, self.error_message)

_galsim.CppShapeData.__repr__ = lambda self: \
        ('galsim._galsim.CppShapeData(' + 21*'%r,' + '%r)')%self.__getinitargs__()

_galsim.CppShapeData.__eq__ = lambda self, other: repr(self) == repr(other)
_galsim.CppShapeData.__ne__ = lambda self, other: not self.__eq__(other)
_galsim.CppShapeData.__hash__ = lambda self: hash(repr(self))

_galsim.HSMParams.__getinitargs__ = lambda self: (
        self.nsig_rg, self.nsig_rg2, self.max_moment_nsig2, self.regauss_too_small,
        self.adapt_order, self.convergence_threshold, self.max_mom2_iter, self.num_iter_default,
        self.bound_correct_wt, self.max_amoment, self.max_ashift,
        self.ksb_moments_max, self.ksb_sig_weight, self.ksb_sig_factor, self.failed_moments)

_galsim.HSMParams.__repr__ = lambda self: \
        ('galsim.hsm.HSMParams(' + 14*'%r,' + '%r)')%self.__getinitargs__()

_galsim.HSMParams.__eq__ = lambda self, other: repr(self) == repr(other)
_galsim.HSMParams.__ne__ = lambda self, other: not self.__eq__(other)
_galsim.HSMParams.__hash__ = lambda self: hash(repr(self))


# A helper function for taking input weight and badpix Images, and returning a weight Image in the
# format that the C++ functions want
def _convertMask(image, weight=None, badpix=None):
    """Convert from input weight and badpix images to a single mask image needed by C++ functions.

    This is used by EstimateShear() and FindAdaptiveMom().
    """
    # if no weight image was supplied, make an int array (same size as gal image) filled with 1's
    if weight is None:
        mask = galsim.ImageI(bounds=image.bounds, init_value=1)

    else:
        # if weight image was supplied, check if it has the right bounds and is non-negative
        if weight.bounds != image.bounds:
            raise ValueError("Weight image does not have same bounds as the input Image!")

        # also make sure there are no negative values
        if np.any(weight.array < 0) == True:
            raise ValueError("Weight image cannot contain negative values!")

        # if weight is an ImageI, then we can use it as the mask image:
        if weight.dtype == np.int32:
            if not badpix:
                mask = weight
            else:
                # If we need to mask bad pixels, we'll need a copy anyway.
                mask = galsim.ImageI(weight)

        # otherwise, we need to convert it to the right type
        else:
            mask = galsim.ImageI(bounds=image.bounds, init_value=0)
            mask.array[weight.array > 0.] = 1

    # if badpix image was supplied, identify the nonzero (bad) pixels and set them to zero in weight
    # image; also check bounds
    if badpix is not None:
        if badpix.bounds != image.bounds:
            raise ValueError("Badpix image does not have the same bounds as the input Image!")
        mask.array[badpix.array != 0] = 0

    # if no pixels are used, raise an exception
    if mask.array.sum() == 0:
        raise RuntimeError("No pixels are being used!")

    # finally, return the Image for the weight map
    return mask._image.view()


# A simpler helper function to force images to be of type ImageF or ImageD
def _convertImage(image):
    """Convert the given image to the correct format needed to pass to the C++ layer.

    This is used by EstimateShear() and FindAdaptiveMom().
    """
    # if weight is not of type float/double, convert to float/double
    if (image.dtype == np.int16 or image.dtype == np.uint16):
        image = galsim.ImageF(image)

    if (image.dtype == np.int32 or image.dtype == np.uint32):
        image = galsim.ImageD(image)

    # Return this as an ImageView
    return image._image.view()


def EstimateShear(gal_image, PSF_image, weight=None, badpix=None, sky_var=0.0,
                  shear_est="REGAUSS", recompute_flux="FIT", guess_sig_gal=5.0,
                  guess_sig_PSF=3.0, precision=1.0e-6, guess_centroid=None,
                  strict=True, hsmparams=None):
    """Carry out moments-based PSF correction routines.

    Carry out PSF correction using one of the methods of the HSM package (see references in
    docstring for file hsm.py) to estimate galaxy shears, correcting for the convolution by the
    PSF.

    This method works from Image inputs rather than GSObject inputs, which provides
    additional flexibility (e.g., it is possible to work from an Image that was read from file and
    corresponds to no particular GSObject), but this also means that users who wish to apply it to
    simple combinations of GSObjects (e.g., a Convolve) must take the additional step of drawing
    their GSObjects into Images.

    This routine assumes that (at least locally) the WCS can be approximated as a PixelScale, with
    no distortion or non-trivial remapping. Any non-trivial WCS gets completely ignored.

    Note that the method will fail if the PSF or galaxy are too point-like to easily fit an
    elliptical Gaussian; when running on batches of many galaxies, it may be preferable to set
    `strict=False` and catch errors explicitly, as in the second example below.

    This function has a number of keyword parameters, many of which a typical user will not need to
    change from the default.

    Example usage
    -------------

    Typical application to a single object:

        >>> galaxy = galsim.Gaussian(flux=1.0, sigma=1.0)
        >>> galaxy = galaxy.shear(g1=0.05, g2=0.0)  # shears the Gaussian by (0.05, 0) using the
        >>>                                         # |g| = (a-b)/(a+b) definition
        >>> psf = galsim.Kolmogorov(flux=1.0, fwhm=0.7)
        >>> final = galsim.Convolve(galaxy, psf)
        >>> final_image = final.drawImage(scale=0.2)
        >>> final_epsf_image = psf.drawImage(scale=0.2)
        >>> result = galsim.hsm.EstimateShear(final_image, final_epsf_image)

    After running the above code, `result.observed_shape` is a galsim.Shear object with a value of
    `(0.0438925349133, -2.85747392701e-18)` and `result.corrected_e1`, `result_corrected_e2` are
    `(0.09934103488922119, -3.746108423463568e-10)`, compared with the expected `(0.09975, 0)` for a perfect
    PSF correction method.

    The code below gives an example of how one could run this routine on a large batch of galaxies,
    explicitly catching and tracking any failures:

        >>> n_image = 100
        >>> n_fail = 0
        >>> for i=0, range(n_image):
        >>>     #...some code defining this_image, this_final_epsf_image...
        >>>     result = galsim.hsm.EstimateShear(this_image, this_final_epsf_image, strict=False)
        >>>     if result.error_message != "":
        >>>         n_fail += 1
        >>> print "Number of failures: ", n_fail

    @param gal_image        The Image of the galaxy being measured.
    @param PSF_image        The Image for the PSF.
    @param weight           The optional weight image for the galaxy being measured.  Can be an int
                            or a float array.  Currently, GalSim does not account for the variation
                            in non-zero weights, i.e., a weight map is converted to an image with 0
                            and 1 for pixels that are not and are used.  Full use of spatial
                            variation in non-zero weights will be included in a future version of
                            the code.
    @param badpix           The optional bad pixel mask for the image being used.  Zero should be
                            used for pixels that are good, and any nonzero value indicates a bad
                            pixel.
    @param sky_var          The variance of the sky level, used for estimating uncertainty on the
                            measured shape. [default: 0.]
    @param shear_est        A string indicating the desired method of PSF correction: 'REGAUSS',
                            'LINEAR', 'BJ', or 'KSB'. The first three options return an e-type
                            distortion, whereas the last option returns a g-type shear.  [default:
                            'REGAUSS']
    @param recompute_flux   A string indicating whether to recompute the object flux, which
                            should be 'NONE' (for no recomputation), 'SUM' (for recomputation via
                            an unweighted sum over unmasked pixels), or 'FIT' (for
                            recomputation using the Gaussian + quartic fit). [default: 'FIT']
    @param guess_sig_gal    Optional argument with an initial guess for the Gaussian sigma of the
                            galaxy (in pixels). [default: 5.]
    @param guess_sig_PSF    Optional argument with an initial guess for the Gaussian sigma of the
                            PSF (in pixels). [default: 3.]
    @param precision        The convergence criterion for the moments. [default: 1e-6]
    @param guess_centroid   An initial guess for the object centroid (useful in
                            case it is not located at the center, which is used if this keyword is
                            not set).  The convention for centroids is such that the center of
                            the lower-left pixel is (image.xmin, image.ymin).
                            [default: gal_image.true_center]
    @param strict           Whether to require success. If `strict=True`, then there will be a
                            `RuntimeError` exception if shear estimation fails.  If set to `False`,
                            then information about failures will be silently stored in the output
                            ShapeData object. [default: True]
    @param hsmparams        The hsmparams keyword can be used to change the settings used by
                            EstimateShear() when estimating shears; see HSMParams documentation
                            using help(galsim.hsm.HSMParams) for more information. [default: None]

    @returns a ShapeData object containing the results of shape measurement.
    """
    # prepare inputs to C++ routines: ImageView for galaxy, PSF, and weight map
    gal_image_view = _convertImage(gal_image)
    PSF_image_view = _convertImage(PSF_image)
    weight_view = _convertMask(gal_image, weight=weight, badpix=badpix)

    if guess_centroid is None:
        guess_centroid = gal_image.true_center
    try:
        result = _galsim._EstimateShearView(gal_image_view, PSF_image_view, weight_view,
                                            sky_var = sky_var,
                                            shear_est = shear_est.upper(),
                                            recompute_flux = recompute_flux.upper(),
                                            guess_sig_gal = guess_sig_gal,
                                            guess_sig_PSF = guess_sig_PSF,
                                            precision = precision,
                                            guess_centroid = guess_centroid,
                                            hsmparams = hsmparams)
    except RuntimeError as err:
        if (strict == True):
            raise
        else:
            result = ShapeData(error_message = str(err))
    return ShapeData(result)

def FindAdaptiveMom(object_image, weight=None, badpix=None, guess_sig=5.0, precision=1.0e-6,
                    guess_centroid=None, strict=True, round_moments=False, hsmparams=None):
    """Measure adaptive moments of an object.

    This method estimates the best-fit elliptical Gaussian to the object (see Hirata & Seljak 2003
    for more discussion of adaptive moments).  This elliptical Gaussian is computed iteratively
    by initially guessing a circular Gaussian that is used as a weight function, computing the
    weighted moments, recomputing the moments using the result of the previous step as the weight
    function, and so on until the moments that are measured are the same as those used for the
    weight function.  FindAdaptiveMom() can be used either as a free function, or as a method of the
    Image class.

    This routine assumes that (at least locally) the WCS can be approximated as a PixelScale, with
    no distortion or non-trivial remapping. Any non-trivial WCS gets completely ignored.

    Like EstimateShear(), FindAdaptiveMom() works on Image inputs, and fails if the object is small
    compared to the pixel scale.  For more details, see EstimateShear().

    Example usage
    -------------

        >>> my_gaussian = galsim.Gaussian(flux=1.0, sigma=1.0)
        >>> my_gaussian_image = my_gaussian.drawImage(scale=0.2, method='no_pixel')
        >>> my_moments = galsim.hsm.FindAdaptiveMom(my_gaussian_image)

    OR

        >>> my_moments = my_gaussian_image.FindAdaptiveMom()

    Assuming a successful measurement, the most relevant pieces of information are
    `my_moments.moments_sigma`, which is `|det(M)|^(1/4)` [=`sigma` for a circular Gaussian] and
    `my_moments.observed_shape`, which is a Shear.  In this case, `my_moments.moments_sigma` is
    precisely 5.0 (in units of pixels), and `my_moments.observed_shape` is consistent with zero.

    Methods of the Shear class can be used to get the distortion `e`, the shear `g`, the conformal
    shear `eta`, and so on.

    As an example of how to use the optional `hsmparams` argument, consider cases where the input
    images have unusual properties, such as being very large.  This could occur when measuring the
    properties of a very over-sampled image such as that generated using

        >>> my_gaussian = galsim.Gaussian(sigma=5.0)
        >>> my_gaussian_image = my_gaussian.drawImage(scale=0.01, method='no_pixel')

    If the user attempts to measure the moments of this very large image using the standard syntax,

        >>> my_moments = my_gaussian_image.FindAdaptiveMom()

    then the result will be a RuntimeError due to moment measurement failing because the object is
    so large.  While the list of all possible settings that can be changed is accessible in the
    docstring of the HSMParams class, in this case we need to modify `max_amoment` which
    is the maximum value of the moments in units of pixel^2.  The following measurement, using the
    default values for every parameter except for `max_amoment`, will be
    successful:

        >>> new_params = galsim.hsm.HSMParams(max_amoment=5.0e5)
        >>> my_moments = my_gaussian_image.FindAdaptiveMom(hsmparams=new_params)

    @param object_image     The Image for the object being measured.
    @param weight           The optional weight image for the object being measured.  Can be an int
                            or a float array.  Currently, GalSim does not account for the variation
                            in non-zero weights, i.e., a weight map is converted to an image with 0
                            and 1 for pixels that are not and are used.  Full use of spatial
                            variation in non-zero weights will be included in a future version of
                            the code. [default: None]
    @param badpix           The optional bad pixel mask for the image being used.  Zero should be
                            used for pixels that are good, and any nonzero value indicates a bad
                            pixel. [default: None]
    @param guess_sig        Optional argument with an initial guess for the Gaussian sigma of the
                            object (in pixels). [default: 5.0]
    @param precision        The convergence criterion for the moments. [default: 1e-6]
    @param guess_centroid   An initial guess for the object centroid (useful in case it is not
                            located at the center, which is used if this keyword is not set).  The
                            convention for centroids is such that the center of the lower-left pixel
                            is (image.xmin, image.ymin).
                            [default: object_image.true_center]
    @param strict           Whether to require success. If `strict=True`, then there will be a
                            `RuntimeError` exception if shear estimation fails.  If set to `False`,
                            then information about failures will be silently stored in the output
                            ShapeData object. [default: True]
    @param round_moments    Use a circular weight function instead of elliptical.
                            [default: False]
    @param hsmparams        The hsmparams keyword can be used to change the settings used by
                            FindAdaptiveMom when estimating moments; see HSMParams documentation
                            using help(galsim.hsm.HSMParams) for more information. [default: None]

    @returns a ShapeData object containing the results of moment measurement.
    """
    # prepare inputs to C++ routines: ImageView for the object being measured and the weight map.
    object_image_view = _convertImage(object_image)
    weight_view = _convertMask(object_image, weight=weight, badpix=badpix)

    if guess_centroid is None:
        guess_centroid = object_image.true_center

    try:
        result = _galsim._FindAdaptiveMomView(object_image_view, weight_view,
                                              guess_sig = guess_sig, precision =  precision,
                                              guess_centroid = guess_centroid,
                                              round_moments = round_moments,
                                              hsmparams = hsmparams)
    except RuntimeError as err:
        if (strict == True):
            raise
        else:
            result = ShapeData(error_message = str(err))
    return ShapeData(result)

# make FindAdaptiveMom a method of Image class
galsim.Image.FindAdaptiveMom = FindAdaptiveMom
