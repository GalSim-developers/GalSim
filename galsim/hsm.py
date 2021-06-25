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

from . import _galsim
from .position import PositionD
from .bounds import BoundsI
from .shear import Shear
from .image import Image, ImageI, ImageF, ImageD
from .errors import GalSimError, GalSimValueError, GalSimHSMError, GalSimIncompatibleValuesError

class ShapeData(object):
    """A class to contain the outputs of the HSM shape and moments measurement routines.

    The ShapeData class contains the following information about moment measurement (from either
    `EstimateShear` or `FindAdaptiveMom`:

    - ``image_bounds``: a `BoundsI` object describing the image.

    - ``moments_status``: the status flag resulting from moments measurement; -1 indicates no
      attempt to measure, 0 indicates success.

    - ``observed_shape``: a `Shear` object representing the observed shape based on adaptive
      moments.

    - ``moments_sigma``: size ``sigma=(det M)^(1/4)`` from the adaptive moments, in units of pixels;
      -1 if not measured.

    - ``moments_amp``: total image intensity for best-fit elliptical Gaussian from adaptive moments.
      Normally, this field is simply equal to the image flux (for objects that follow a Gaussian
      light distribution, otherwise it is something approximating the flux).  However, if the image
      was drawn using ``drawImage(method='sb')`` then moments_amp relates to the flux via
      ``flux=(moments_amp)*(pixel scale)^2``.

    - ``moments_centroid``: a `PositionD` object representing the centroid based on adaptive
      moments, in units of pixels.  The indexing convention is defined with respect to the `BoundsI`
      object defining the bounds of the input `Image`, i.e., the center of the lower left pixel is
      ``(image.xmin, image.ymin)``.  An object drawn at the center of the image should generally
      have moments_centroid equal to ``image.true_center``.

    - ``moments_rho4``: the weighted radial fourth moment of the image.

    - ``moments_n_iter``: number of iterations needed to get adaptive moments, or 0 if not measured.

    If `EstimateShear` was used, then the following fields related to PSF-corrected shape will also
    be populated:

    - ``correction_status``: the status flag resulting from PSF correction; -1 indicates no attempt
      to measure, 0 indicates success.

    - ``corrected_e1``, ``corrected_e2``, ``corrected_g1``, ``corrected_g2``: floats representing
      the estimated shear after removing the effects of the PSF.  Either e1/e2 or g1/g2 will differ
      from the default values of -10, with the choice of shape to use determined by the correction
      method (since the correction method determines what quantity is estimated, either the shear or
      the distortion).  After a measurement is made, the type of shape measurement is stored in the
      ShapeData structure as 'meas_type' (a string that equals either 'e' or 'g').

    - ``corrected_shape_err``: shape measurement uncertainty sigma_gamma per component.  The
      estimate of the uncertainty will only be non-zero if an estimate of the sky variance was
      passed to `EstimateShear`.

    - ``correction_method``: a string indicating the method of PSF correction (will be "None" if
      PSF-correction was not carried out).

    - ``resolution_factor``: Resolution factor R_2;  0 indicates object is consistent with a PSF, 1
      indicates perfect resolution.

    - ``psf_sigma``: size ``sigma=(det M)^(1/4)`` of the PSF from the adaptive moments, in units of
      pixels; -1 if not measured.

    - ``psf_shape``: a `Shear` object representing the observed PSF shape based on adaptive moments.

    - ``error_message``: a string containing any error messages from the attempt to carry out
      PSF-correction.

    The `ShapeData` object can be initialized completely empty, or can be returned from the
    routines that measure object moments (`FindAdaptiveMom`) and carry out PSF correction
    (`EstimateShear`).
    """
    def __init__(self, image_bounds=BoundsI(), moments_status=-1,
                 observed_shape=Shear(), moments_sigma=-1.0, moments_amp=-1.0,
                 moments_centroid=PositionD(), moments_rho4=-1.0, moments_n_iter=0,
                 correction_status=-10, corrected_e1=-10., corrected_e2=-10.,
                 corrected_g1=-10., corrected_g2=-10., meas_type="None",
                 corrected_shape_err=-1.0, correction_method="None",
                 resolution_factor=-1.0, psf_sigma=-1.0,
                 psf_shape=Shear(), error_message=""):

        # Avoid empty string, which can caus problems in C++ layer.
        if error_message == "": error_message = "None"

        if not isinstance(image_bounds, BoundsI):
            raise TypeError("image_bounds must be a BoundsI instance")

        # The others will raise an appropriate TypeError from the call to _galsim.ShapeData
        # when converting to int, float, etc.
        self._data = _galsim.ShapeData(
            image_bounds._b, int(moments_status), observed_shape.e1, observed_shape.e2,
            float(moments_sigma), float(moments_amp), moments_centroid._p,
            float(moments_rho4), int(moments_n_iter), int(correction_status),
            float(corrected_e1), float(corrected_e2), float(corrected_g1), float(corrected_g2),
            str(meas_type), float(corrected_shape_err), str(correction_method),
            float(resolution_factor), float(psf_sigma), psf_shape.e1, psf_shape.e2,
            str(error_message))

    @property
    def image_bounds(self): return BoundsI(self._data.image_bounds)
    @property
    def moments_status(self): return self._data.moments_status

    @property
    def observed_shape(self):
        return Shear(e1=self._data.observed_e1, e2=self._data.observed_e2)

    @property
    def moments_sigma(self): return self._data.moments_sigma
    @property
    def moments_amp(self): return self._data.moments_amp
    @property
    def moments_centroid(self): return PositionD(self._data.moments_centroid)
    @property
    def moments_rho4(self): return self._data.moments_rho4
    @property
    def moments_n_iter(self): return self._data.moments_n_iter
    @property
    def correction_status(self): return self._data.correction_status
    @property
    def corrected_e1(self): return self._data.corrected_e1
    @property
    def corrected_e2(self): return self._data.corrected_e2
    @property
    def corrected_g1(self): return self._data.corrected_g1
    @property
    def corrected_g2(self): return self._data.corrected_g2
    @property
    def meas_type(self): return self._data.meas_type
    @property
    def corrected_shape_err(self): return self._data.corrected_shape_err
    @property
    def correction_method(self): return self._data.correction_method
    @property
    def resolution_factor(self): return self._data.resolution_factor
    @property
    def psf_sigma(self): return self._data.psf_sigma

    @property
    def psf_shape(self):
        return Shear(e1=self._data.psf_e1, e2=self._data.psf_e2)

    @property
    def error_message(self):
        # We use "None" in C++ ShapeData to indicate no error messages to avoid problems on
        # (some) Macs using zero-length strings.  Here, we revert that back to "".
        if self._data.error_message == "None":
            return ""
        else:
            return self._data.error_message

    def __repr__(self):
        s = 'galsim.hsm.ShapeData('
        if self.image_bounds.isDefined(): s += 'image_bounds=%r, '%self.image_bounds
        if self.moments_status != -1: s += 'moments_status=%r, '%self.moments_status
        # Always include this one:
        s += 'observed_shape=%r'%self.observed_shape
        if self.moments_sigma != -1: s += ', moments_sigma=%r'%self.moments_sigma
        if self.moments_amp != -1: s += ', moments_amp=%r'%self.moments_amp
        if self.moments_centroid != PositionD():
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
        if self.psf_shape != Shear(): s += ', psf_shape=%r'%self.psf_shape
        if self.error_message != "": s += ', error_message=%r'%self.error_message
        s += ')'
        return s

    def __eq__(self, other):
        return (self is other or
                (isinstance(other,ShapeData) and self._getinitargs() == other._getinitargs()))
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(("galsim.hsm.ShapeData", self._getinitargs()))

    def _getinitargs(self):
        return (self.image_bounds, self.moments_status, self.observed_shape,
                self.moments_sigma, self.moments_amp, self.moments_centroid, self.moments_rho4,
                self.moments_n_iter, self.correction_status, self.corrected_e1, self.corrected_e2,
                self.corrected_g1, self.corrected_g2, self.meas_type, self.corrected_shape_err,
                self.correction_method, self.resolution_factor, self.psf_sigma,
                self.psf_shape, self.error_message)

    def __getstate__(self):
        return self._getinitargs()

    def __setstate__(self, state):
        self.__init__(*state)


class HSMParams(object):
    """A collection of parameters that govern how the HSM functions operate.

    HSMParams stores a set of numbers that determine how the moments/shape estimation
    routines make speed/accuracy tradeoff decisions and/or store their results.

    Parameters:
        nsig_rg:                A parameter used to optimize convolutions by cutting off the galaxy
                                profile.  In the first step of the re-Gaussianization method of PSF
                                correction, a Gaussian approximation to the pre-seeing galaxy is
                                calculated. If ``nsig_rg > 0``, then this approximation is cut off
                                at ``nsig_rg`` sigma to save computation time in convolutions.
                                [default: 3.0]

        nsig_rg2:               A parameter used to optimize convolutions by cutting off the PSF
                                residual profile.  In the re-Gaussianization method of PSF
                                correction, a 'PSF residual' (the difference between the true PSF
                                and its best-fit Gaussian approximation) is constructed. If
                                ``nsig_rg2 > 0``, then this PSF residual is cut off at ``nsig_rg2``
                                sigma to save computation time in convolutions. [default: 3.6]

        max_moment_nsig2:       A parameter for optimizing calculations of adaptive moments by
                                cutting off profiles. This parameter is used to decide how many
                                sigma^2 into the Gaussian adaptive moment to extend the moment
                                calculation, with the weight being defined as 0 beyond this point.
                                i.e., if max_moment_nsig2 is set to 25, then the Gaussian is
                                extended to ``(r^2/sigma^2)=25``, with proper accounting for
                                elliptical geometry.  If this parameter is set to some very large
                                number, then the weight is never set to zero and the exponential
                                function is always called. Note: GalSim script
                                devel/modules/test_mom_timing.py was used to choose a value of 25 as
                                being optimal, in that for the cases that were tested, the speedups
                                were typically factors of several, but the results of moments and
                                shear estimation were changed by <10^-5.  Not all possible cases
                                were checked, and so for use of this code for unusual cases, we
                                recommend that users check that this value does not affect accuracy,
                                and/or set it to some large value to completely disable this
                                optimization. [default: 25.0]

        regauss_too_small:      A parameter for how strictly the re-Gaussianization code treats
                                small galaxies. If this parameter is 1, then the re-Gaussianization
                                code does not impose a cut on the apparent resolution before trying
                                to measure the PSF-corrected shape of the galaxy; if 0, then it is
                                stricter.  Using the default value of 1 prevents the
                                re-Gaussianization PSF correction from completely failing at the
                                beginning, before trying to do PSF correction, due to the crudest
                                possible PSF correction (Gaussian approximation) suggesting that
                                the galaxy is very small.  This could happen for some usable
                                galaxies particularly when they have very non-Gaussian surface
                                brightness profiles -- for example, if there's a prominent bulge
                                that the adaptive moments attempt to fit, ignoring a more
                                extended disk.  Setting a value of 1 is useful for keeping galaxies
                                that would have failed for that reason.  If they later turn out to
                                be too small to really use, this will be reflected in the final
                                estimate of the resolution factor, and they can be rejected after
                                the fact. [default: 1]

        adapt_order:            The order to which circular adaptive moments should be calculated
                                for KSB method. This parameter only affects calculations using the
                                KSB method of PSF correction.  Warning: deviating from default
                                value of 2 results in code running more slowly, and results have
                                not been significantly tested. [default: 2]

        convergence_threshold:  Accuracy (in x0, y0, and sigma, each as a fraction of sigma)
                                when calculating adaptive moments. [default: 1.e-6]

        max_mom2_iter:          Maximum number of iterations to use when calculating adaptive
                                moments.  This should be sufficient in nearly all situations, with
                                the possible exception being very flattened profiles. [default: 400]

        num_iter_default:       Number of iterations to report in the output ShapeData structure
                                when code fails to converge within max_mom2_iter iterations.
                                [default: -1]

        bound_correct_wt:       Maximum shift in centroids and sigma between iterations for
                                adaptive moments. [default: 0.25]

        max_amoment:            Maximum value for adaptive second moments before throwing
                                exception.  Very large objects might require this value to be
                                increased. [default: 8000]

        max_ashift:             Maximum allowed x / y centroid shift (units: pixels) between
                                successive iterations for adaptive moments before throwing
                                exception. [default: 15]

        ksb_moments_max:        Use moments up to ksb_moments_max order for KSB method of PSF
                                correction. [default: 4]

        ksb_sig_weight:         The width of the weight function (in pixels) to use for the KSB
                                method.  Normally, this is derived from the measured moments of the
                                galaxy image; this keyword overrides this calculation.  Can be
                                combined with ksb_sig_factor. [default: 0.0]

        ksb_sig_factor:         Factor by which to multiply the weight function width for the KSB
                                method (default: 1.0).  Can be combined with ksb_sig_weight.
                                [default: 1.0]

        failed_moments:         Value to report for ellipticities and resolution factor if shape
                                measurement fails. [default: -1000.]

    After construction, all of the above are available as read-only attributes.
    """
    def __init__(self, nsig_rg=3.0, nsig_rg2=3.6, max_moment_nsig2=25.0, regauss_too_small=1,
                 adapt_order=2, convergence_threshold=1.e-6, max_mom2_iter=400,
                 num_iter_default=-1, bound_correct_wt=0.25, max_amoment=8000., max_ashift=15.,
                 ksb_moments_max=4, ksb_sig_weight=0.0, ksb_sig_factor=1.0, failed_moments=-1000.):

        self._nsig_rg = float(nsig_rg)
        self._nsig_rg2 = float(nsig_rg2)
        self._max_moment_nsig2 = float(max_moment_nsig2)
        self._regauss_too_small = int(regauss_too_small)
        self._adapt_order = int(adapt_order)
        self._convergence_threshold = float(convergence_threshold)
        self._max_mom2_iter = int(max_mom2_iter)
        self._num_iter_default = int(num_iter_default)
        self._bound_correct_wt = float(bound_correct_wt)
        self._max_amoment = float(max_amoment)
        self._max_ashift = float(max_ashift)
        self._ksb_moments_max = int(ksb_moments_max)
        self._ksb_sig_weight = float(ksb_sig_weight)
        self._ksb_sig_factor = float(ksb_sig_factor)
        self._failed_moments = float(failed_moments)
        self._make_hsmp()

    def _make_hsmp(self):
        self._hsmp = _galsim.HSMParams(*self._getinitargs())

    def _getinitargs(self):
        return (self.nsig_rg, self.nsig_rg2, self.max_moment_nsig2, self.regauss_too_small,
                self.adapt_order, self.convergence_threshold, self.max_mom2_iter,
                self.num_iter_default, self.bound_correct_wt, self.max_amoment, self.max_ashift,
                self.ksb_moments_max, self.ksb_sig_weight, self.ksb_sig_factor,
                self.failed_moments)

    @property
    def nsig_rg(self): return self._nsig_rg
    @property
    def nsig_rg2(self): return self._nsig_rg2
    @property
    def max_moment_nsig2(self): return self._max_moment_nsig2
    @property
    def regauss_too_small(self): return self._regauss_too_small
    @property
    def adapt_order(self): return self._adapt_order
    @property
    def convergence_threshold(self): return self._convergence_threshold
    @property
    def max_mom2_iter(self): return self._max_mom2_iter
    @property
    def num_iter_default(self): return self._num_iter_default
    @property
    def bound_correct_wt(self): return self._bound_correct_wt
    @property
    def max_amoment(self): return self._max_amoment
    @property
    def max_ashift(self): return self._max_ashift
    @property
    def ksb_moments_max(self): return self._ksb_moments_max
    @property
    def ksb_sig_weight(self): return self._ksb_sig_weight
    @property
    def ksb_sig_factor(self): return self._ksb_sig_factor
    @property
    def failed_moments(self): return self._failed_moments

    @staticmethod
    def check(hsmparams, default=None):
        """Checks that hsmparams is either a valid HSMParams instance or None.

        In the former case, it returns hsmparams, in the latter it returns default
        (HSMParams.default if no other default specified).
        """
        if hsmparams is None:
            return default if default is not None else HSMParams.default
        elif not isinstance(hsmparams, HSMParams):
            raise TypeError("Invalid HSMParams: %s"%hsmparams)
        else:
            return hsmparams

    def __repr__(self):
        return ('galsim.hsm.HSMParams(' + 14*'%r,' + '%r)')%self._getinitargs()

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, HSMParams) and self._getinitargs() == other._getinitargs()))
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return hash(('galsim.hsm.HSMParams', self._getinitargs()))

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_hsmp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._make_hsmp()

# We use the default a lot, so make it a class attribute.
HSMParams.default = HSMParams()


# A helper function for taking input weight and badpix Images, and returning a weight Image in the
# format that the C++ functions want
def _convertMask(image, weight=None, badpix=None):
    # Convert from input weight and badpix images to a single mask image needed by C++ functions.
    # This is used by EstimateShear() and FindAdaptiveMom().

    # if no weight image was supplied, make an int array (same size as gal image) filled with 1's
    if weight is None:
        mask = ImageI(bounds=image.bounds, init_value=1)

    else:
        # if weight image was supplied, check if it has the right bounds and is non-negative
        if weight.bounds != image.bounds:
            raise GalSimIncompatibleValuesError(
                "Weight image does not have same bounds as the input Image.",
                weight=weight, image=image)

        # also make sure there are no negative values
        if np.any(weight.array < 0) == True:
            raise GalSimValueError("Weight image cannot contain negative values.", weight)

        # if weight is an ImageI, then we can use it as the mask image:
        if weight.dtype == np.int32:
            if not badpix:
                mask = weight
            else:
                # If we need to mask bad pixels, we'll need a copy anyway.
                mask = ImageI(weight)

        # otherwise, we need to convert it to the right type
        else:
            mask = ImageI(bounds=image.bounds, init_value=0)
            mask.array[weight.array > 0.] = 1

    # if badpix image was supplied, identify the nonzero (bad) pixels and set them to zero in weight
    # image; also check bounds
    if badpix is not None:
        if badpix.bounds != image.bounds:
            raise GalSimIncompatibleValuesError(
                "Badpix image does not have the same bounds as the input Image.",
                badpix=badpix, image=image)
        mask.array[badpix.array != 0] = 0

    # if no pixels are used, raise an exception
    if mask.array.sum() == 0:
        raise GalSimHSMError("No pixels are being used!")

    # finally, return the Image for the weight map
    return mask


# A simpler helper function to force images to be of type ImageF or ImageD
def _convertImage(image):
    # Convert the given image to the correct format needed to pass to the C++ layer.
    # This is used by EstimateShear() and FindAdaptiveMom().

    # if weight is not of type float/double, convert to float/double
    if (image.dtype == np.int16 or image.dtype == np.uint16):
        image = ImageF(image)

    if (image.dtype == np.int32 or image.dtype == np.uint32):
        image = ImageD(image)

    return image


def EstimateShear(gal_image, PSF_image, weight=None, badpix=None, sky_var=0.0,
                  shear_est="REGAUSS", recompute_flux="FIT", guess_sig_gal=5.0,
                  guess_sig_PSF=3.0, precision=1.0e-6, guess_centroid=None,
                  strict=True, hsmparams=None):
    """Carry out moments-based PSF correction routines.

    Carry out PSF correction using one of the methods of the HSM package (see references in
    docstring for file hsm.py) to estimate galaxy shears, correcting for the convolution by the
    PSF.

    This method works from `Image` inputs rather than `GSObject` inputs, which provides
    additional flexibility (e.g., it is possible to work from an `Image` that was read from file and
    corresponds to no particular `GSObject`), but this also means that users who wish to apply it to
    compount `GSObject` classes (e.g., `Convolve`) must take the additional step of drawing
    their `GSObject` into `Image` instances.

    This routine assumes that (at least locally) the WCS can be approximated as a `PixelScale`, with
    no distortion or non-trivial remapping. Any non-trivial WCS gets completely ignored.

    Note that the method will fail if the PSF or galaxy are too point-like to easily fit an
    elliptical Gaussian; when running on batches of many galaxies, it may be preferable to set
    ``strict=False`` and catch errors explicitly, as in the second example below.

    This function has a number of keyword parameters, many of which a typical user will not need to
    change from the default.

    Example:

    Typical application to a single object::

        >>> galaxy = galsim.Gaussian(flux=1.0, sigma=1.0)
        >>> galaxy = galaxy.shear(g1=0.05, g2=0.0)  # shears the Gaussian by (0.05, 0) using the
        >>>                                         # |g| = (a-b)/(a+b) definition
        >>> psf = galsim.Kolmogorov(flux=1.0, fwhm=0.7)
        >>> final = galsim.Convolve(galaxy, psf)
        >>> final_image = final.drawImage(scale=0.2)
        >>> final_epsf_image = psf.drawImage(scale=0.2)
        >>> result = galsim.hsm.EstimateShear(final_image, final_epsf_image)

    After running the above code, ``result.observed_shape`` is a `Shear` object with a value of
    ``(0.0438925349133, -2.85747392701e-18)`` and ``result.corrected_e1``, ``result_corrected_e2``
    are ``(0.09934103488922119, -3.746108423463568e-10)``, compared with the expected ``(0.09975,
    0)`` for a perfect PSF correction method.

    The code below gives an example of how one could run this routine on a large batch of galaxies,
    explicitly catching and tracking any failures::

        >>> n_image = 100
        >>> n_fail = 0
        >>> for i=0, range(n_image):
        >>>     #...some code defining this_image, this_final_epsf_image...
        >>>     result = galsim.hsm.EstimateShear(this_image, this_final_epsf_image, strict=False)
        >>>     if result.error_message != "":
        >>>         n_fail += 1
        >>> print "Number of failures: ", n_fail

    Parameters:
        gal_image:      The `Image` of the galaxy being measured.
        PSF_image:      The `Image` for the PSF.
        weight:         The optional weight image for the galaxy being measured.  Can be an int
                        or a float array.  Currently, GalSim does not account for the variation
                        in non-zero weights, i.e., a weight map is converted to an image with 0
                        and 1 for pixels that are not and are used.  Full use of spatial
                        variation in non-zero weights will be included in a future version of
                        the code.

        badpix:         The optional bad pixel mask for the image being used.  Zero should be
                        used for pixels that are good, and any nonzero value indicates a bad
                        pixel.

        sky_var:        The variance of the sky level, used for estimating uncertainty on the
                        measured shape. [default: 0.]

        shear_est:      A string indicating the desired method of PSF correction: 'REGAUSS',
                        'LINEAR', 'BJ', or 'KSB'. The first three options return an e-type
                        distortion, whereas the last option returns a g-type shear.  [default:
                        'REGAUSS']

        recompute_flux: A string indicating whether to recompute the object flux, which
                        should be 'NONE' (for no recomputation), 'SUM' (for recomputation via
                        an unweighted sum over unmasked pixels), or 'FIT' (for
                        recomputation using the Gaussian + quartic fit). [default: 'FIT']

        guess_sig_gal:  Optional argument with an initial guess for the Gaussian sigma of the
                        galaxy (in pixels). [default: 5.]

        guess_sig_PSF:  Optional argument with an initial guess for the Gaussian sigma of the
                        PSF (in pixels). [default: 3.]

        precision:      The convergence criterion for the moments. [default: 1e-6]

        guess_centroid: An initial guess for the object centroid (useful in
                        case it is not located at the center, which is used if this keyword is
                        not set).  The convention for centroids is such that the center of
                        the lower-left pixel is (image.xmin, image.ymin).
                        [default: gal_image.true_center]

        strict:         Whether to require success. If ``strict=True``, then there will be a
                        ``GalSimHSMError`` exception if shear estimation fails.  If set to
                        ``False``, then information about failures will be silently stored in
                        the output ShapeData object. [default: True]

        hsmparams:      The hsmparams keyword can be used to change the settings used by
                        `EstimateShear` when estimating shears; see `HSMParams` documentation
                        for more information. [default: None]

    Returns:
        a `ShapeData` object containing the results of shape measurement.
    """
    # prepare inputs to C++ routines: ImageF or ImageD for galaxy, PSF, and ImageI for weight map
    gal_image = _convertImage(gal_image)
    PSF_image = _convertImage(PSF_image)
    weight = _convertMask(gal_image, weight=weight, badpix=badpix)
    hsmparams = HSMParams.check(hsmparams)

    if guess_centroid is None:
        guess_centroid = gal_image.true_center
    try:
        result = ShapeData()
        _galsim._EstimateShearView(result._data,
                                   gal_image._image, PSF_image._image, weight._image,
                                   float(sky_var), shear_est.upper(), recompute_flux.upper(),
                                   float(guess_sig_gal), float(guess_sig_PSF), float(precision),
                                   guess_centroid._p, hsmparams._hsmp)
        return result
    except RuntimeError as err:
        if (strict == True):
            raise GalSimHSMError(str(err))
        else:
            return ShapeData(error_message = str(err))

def FindAdaptiveMom(object_image, weight=None, badpix=None, guess_sig=5.0, precision=1.0e-6,
                    guess_centroid=None, strict=True, round_moments=False, hsmparams=None):
    """Measure adaptive moments of an object.

    This method estimates the best-fit elliptical Gaussian to the object (see Hirata & Seljak 2003
    for more discussion of adaptive moments).  This elliptical Gaussian is computed iteratively
    by initially guessing a circular Gaussian that is used as a weight function, computing the
    weighted moments, recomputing the moments using the result of the previous step as the weight
    function, and so on until the moments that are measured are the same as those used for the
    weight function.  `FindAdaptiveMom` can be used either as a free function, or as a method of the
    `Image` class.

    This routine assumes that (at least locally) the WCS can be approximated as a `PixelScale`, with
    no distortion or non-trivial remapping. Any non-trivial WCS gets completely ignored.

    Like `EstimateShear`, `FindAdaptiveMom` works on `Image` inputs, and fails if the object is
    small compared to the pixel scale.  For more details, see `EstimateShear`.

    Example::

        >>> my_gaussian = galsim.Gaussian(flux=1.0, sigma=1.0)
        >>> my_gaussian_image = my_gaussian.drawImage(scale=0.2, method='no_pixel')
        >>> my_moments = galsim.hsm.FindAdaptiveMom(my_gaussian_image)

    or::

        >>> my_moments = my_gaussian_image.FindAdaptiveMom()

    Assuming a successful measurement, the most relevant pieces of information are
    ``my_moments.moments_sigma``, which is ``|det(M)|^(1/4)`` (= ``sigma`` for a circular Gaussian)
    and ``my_moments.observed_shape``, which is a `Shear`.  In this case,
    ``my_moments.moments_sigma`` is precisely 5.0 (in units of pixels), and
    ``my_moments.observed_shape`` is consistent with zero.

    Methods of the `Shear` class can be used to get the distortion ``e``, the shear ``g``, the
    conformal shear ``eta``, and so on.

    As an example of how to use the optional ``hsmparams`` argument, consider cases where the input
    images have unusual properties, such as being very large.  This could occur when measuring the
    properties of a very over-sampled image such as that generated using::

        >>> my_gaussian = galsim.Gaussian(sigma=5.0)
        >>> my_gaussian_image = my_gaussian.drawImage(scale=0.01, method='no_pixel')

    If the user attempts to measure the moments of this very large image using the standard syntax,
    ::

        >>> my_moments = my_gaussian_image.FindAdaptiveMom()

    then the result will be a ``GalSimHSMError`` due to moment measurement failing because the
    object is so large.  While the list of all possible settings that can be changed is accessible
    in the docstring of the `HSMParams` class, in this case we need to modify ``max_amoment`` which
    is the maximum value of the moments in units of pixel^2.  The following measurement, using the
    default values for every parameter except for ``max_amoment``, will be
    successful::

        >>> new_params = galsim.hsm.HSMParams(max_amoment=5.0e5)
        >>> my_moments = my_gaussian_image.FindAdaptiveMom(hsmparams=new_params)

    Parameters:
        object_image:       The `Image` for the object being measured.
        weight:             The optional weight image for the object being measured.  Can be an int
                            or a float array.  Currently, GalSim does not account for the variation
                            in non-zero weights, i.e., a weight map is converted to an image with 0
                            and 1 for pixels that are not and are used.  Full use of spatial
                            variation in non-zero weights will be included in a future version of
                            the code. [default: None]
        badpix:             The optional bad pixel mask for the image being used.  Zero should be
                            used for pixels that are good, and any nonzero value indicates a bad
                            pixel. [default: None]
        guess_sig:          Optional argument with an initial guess for the Gaussian sigma of the
                            object (in pixels). [default: 5.0]
        precision:          The convergence criterion for the moments. [default: 1e-6]
        guess_centroid:     An initial guess for the object centroid (useful in case it is not
                            located at the center, which is used if this keyword is not set).  The
                            convention for centroids is such that the center of the lower-left pixel
                            is (image.xmin, image.ymin).
                            [default: object_image.true_center]
        strict:             Whether to require success. If ``strict=True``, then there will be a
                            ``GalSimHSMError`` exception if shear estimation fails.  If set to
                            ``False``, then information about failures will be silently stored in
                            the output ShapeData object. [default: True]
        round_moments:      Use a circular weight function instead of elliptical.
                            [default: False]
        hsmparams:          The hsmparams keyword can be used to change the settings used by
                            FindAdaptiveMom when estimating moments; see `HSMParams` documentation
                            for more information. [default: None]

    Returns:
        a `ShapeData` object containing the results of moment measurement.
    """
    # prepare inputs to C++ routines: ImageF or ImageD for galaxy, PSF, and ImageI for weight map
    object_image = _convertImage(object_image)
    weight = _convertMask(object_image, weight=weight, badpix=badpix)
    hsmparams = HSMParams.check(hsmparams)

    if guess_centroid is None:
        guess_centroid = object_image.true_center

    try:
        result = ShapeData()
        _galsim._FindAdaptiveMomView(result._data,
                                     object_image._image, weight._image,
                                     float(guess_sig), float(precision), guess_centroid._p,
                                     bool(round_moments), hsmparams._hsmp)
        return result
    except RuntimeError as err:
        if (strict == True):
            raise GalSimHSMError(str(err))
        else:
            return ShapeData(error_message = str(err))

# make FindAdaptiveMom a method of Image class
Image.FindAdaptiveMom = FindAdaptiveMom
