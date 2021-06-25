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

from .image import Image
from .random import BaseDeviate
from .gsobject import GSObject
from .gsparams import GSParams
from .wcs import PixelScale
from . import utilities
from .errors import GalSimError, GalSimValueError, GalSimRangeError, GalSimUndefinedBoundsError
from .errors import GalSimIncompatibleValuesError, galsim_warn

def whitenNoise(self, noise):
    # This will be inserted into the Image class as a method.  So self = image.
    """Whiten the noise in the image assuming that the noise currently in the image can be described
    by the `BaseCorrelatedNoise` object ``noise``.  See `BaseCorrelatedNoise.whitenImage` for more
    details of how this method works.

    Parameters:
        noise:      The `BaseCorrelatedNoise` model to use when figuring out how much noise to add
                    to make the final noise white.

    Returns:
        the theoretically calculated variance of the combined noise fields in the updated image.
    """
    return noise.whitenImage(self)

def symmetrizeNoise(self, noise, order=4):
    # This will be inserted into the Image class as a method.  So self = image.
    """Impose N-fold symmetry (where N=``order`` is an even integer >=4) on the noise in a square
    image assuming that the noise currently in the image can be described by the
    `BaseCorrelatedNoise` object ``noise``.  See `BaseCorrelatedNoise.symmetrizeImage` for more
    details of how this method works.

    Parameters:
        noise:      The `BaseCorrelatedNoise` model to use when figuring out how much noise to add
                    to make the final noise have symmetry at the desired order.
        order:      Desired symmetry order.  Must be an even integer larger than 2.
                    [default: 4]

    Returns:
        the theoretically calculated variance of the combined noise fields in the updated image.
    """
    return noise.symmetrizeImage(self, order=order)

# Now inject whitenNoise and symmetrizeNoise as methods of the Image class.
Image.whitenNoise = whitenNoise
Image.symmetrizeNoise = symmetrizeNoise

class BaseCorrelatedNoise(object):
    """A Base Class describing 2D correlated Gaussian random noise fields.

    A BaseCorrelatedNoise will not generally be instantiated directly.  This is recommended as the
    current ``BaseCorrelatedNoise.__init__`` interface does not provide any guarantee that the input
    `GSObject` represents a physical correlation function, e.g. a profile that is an even function
    (two-fold rotationally symmetric in the plane) and peaked at the origin.  The proposed pattern
    is that users instead instantiate derived classes, such as the `CorrelatedNoise`, which are
    able to guarantee the above.

    The BaseCorrelatedNoise is therefore here primarily to define the way in which derived classes
    (currently only `CorrelatedNoise` and `UncorrelatedNoise`) store the random deviate, noise
    correlation function profile and allow operations with it, generate images containing noise with
    these correlation properties, and generate covariance matrices according to the correlation
    function.

    Parameters:
        rng:        A `BaseDeviate` instance to use for generating the random numbers.
        gsobject:   The `GSObject` defining the correlation function.

                    .. warning:

                        The user is responsible for ensuring this profile is 2-fold rotationally
                        symmetric.  This is **not** checked.

        wcs:        The wcs for the image to define the phyical relationship between the pixels.
                    [default: PixelScale(1.0)]
    """
    def __init__(self, rng, gsobject, wcs=PixelScale(1.0)):
        if rng is not None and not isinstance(rng, BaseDeviate):
            raise TypeError(
                "Supplied rng argument not a galsim.BaseDeviate or derived class instance.")

        if rng is None: rng = BaseDeviate()
        self._rng = rng
        # Act as a container for the GSObject used to represent the correlation funcion.
        self._profile = gsobject
        self.wcs = wcs

        # When applying normal or whitening noise to an image, we normally do calculations.
        # If _profile_for_cached is profile, then it means that we can use the stored values in
        # _rootps_cache, _rootps_whitening_cache, and/or _rootps_symmetrizing_cache and avoid having
        # to redo the calculations.
        # So for now, we start out with _profile_for_cached = None, and _rootps_cache,
        # _rootps_whitening_cache, _rootps_symmetrizing_cache empty.
        self._profile_for_cache = None
        self._rootps_cache = {}
        self._rootps_whitening_cache = {}
        self._rootps_symmetrizing_cache = {}
        # Also set up the cache for a stored value of the variance, needed for efficiency once the
        # noise field can get convolved with other GSObjects making is_analytic_x False
        self._variance_cached = None

    @property
    def rng(self):
        """The `BaseDeviate` for this object.
        """
        return self._rng

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._profile.gsparams

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current object with the given `GSParams`.
        """
        if gsparams == self.gsparams: return self
        return BaseCorrelatedNoise(self.rng, self._profile.withGSParams(gsparams, **kwargs),
                                   self.wcs)

    # Make "+" work in the intuitive sense (variances being additive, correlation functions add as
    # you would expect)
    def __add__(self, other):
        from . import wcs
        if not wcs.compatible(self.wcs, other.wcs):
            galsim_warn("Adding two CorrelatedNoise objects with incompatible WCS. "
                        "The result will have the WCS of the first object.")
        return BaseCorrelatedNoise(self.rng, self._profile + other._profile, self.wcs)

    def __sub__(self, other):
        from . import wcs
        if not wcs.compatible(self.wcs, other.wcs):
            galsim_warn("Subtracting two CorrelatedNoise objects with incompatible WCS. "
                        "The result will have the WCS of the first object.")
        return BaseCorrelatedNoise(self.rng, self._profile - other._profile, self.wcs)

    def __mul__(self, variance_ratio):
        return self.withScaledVariance(variance_ratio)
    def __div__(self, variance_ratio):
        return self.withScaledVariance(1./variance_ratio)
    __truediv__ = __div__

    def copy(self, rng=None):
        """Returns a copy of the correlated noise model.

        By default, the copy will share the `BaseDeviate` random number generator with the parent
        instance.  However, you can provide a new rng to use in the copy if you want with::

            >>> cn_copy = cn.copy(rng=new_rng)
        """
        if rng is None:
            rng = self.rng
        return BaseCorrelatedNoise(rng, self._profile, self.wcs)

    def __repr__(self):
        return "galsim.BaseCorrelatedNoise(%r,%r,%r)"%(
                self.rng, self._profile, self.wcs)

    def __str__(self):
        return "galsim.BaseCorrelatedNoise(%s,%s)"%(self._profile, self.wcs)

    # Quick and dirty.  Just check reprs are equal.
    def __eq__(self, other): return self is other or repr(self) == repr(other)
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(repr(self))

    def _clear_cache(self):
        """Check if the profile has changed and clear caches if appropriate.
        """
        if self._profile_for_cache is not self._profile:
            self._rootps_cache.clear()
            self._rootps_whitening_cache.clear()
            self._rootps_symmetrizing_cache.clear()
            self._variance_cached = None
        # Set profile_for_cache for next time.
        self._profile_for_cache = self._profile

    def applyTo(self, image):
        """Apply this correlated Gaussian random noise field to an input `Image`.

        To add deviates to every element of an image, the syntax::

            >>> image.addNoise(correlated_noise)

        is preferred.  However, this is equivalent to calling this instance's `applyTo` method as
        follows::

            >>> correlated_noise.applyTo(image)

        On output the `Image` instance ``image`` will have been given additional noise according to
        the given `BaseCorrelatedNoise` instance ``correlated_noise``.  Normally, ``image.scale``
        is used to determine the input pixel separation, and if ``image.scale <= 0`` a pixel scale
        of 1 is assumed.  If the image has a non-uniform WCS, the local uniform approximation at
        the center of the image will be used.

        Note that the correlations defined in a correlated_noise object are defined in terms of
        world coordinates (i.e. typically arcsec on the sky).  Some care is thus required if you
        apply correlated noise to an image with a non-trivial WCS.  The correlations will have a
        specific direction and scale in world coordinates, so if you apply them to an image with
        a WCS that has a rotation or a different pixel scale than the original, the resulting
        correlations will have the correct direction and scale in world coordinates, but a
        different direction and/or scale in image coordinates.

        If you want to override this behavior, you can view your image with the WCS of the
        correlation function and apply the noise to that.  For example::

            >>> image = galsim.Image(nx, ny, wcs=complicated_wcs)
            >>> noise = galsim.getCOSMOSNoise(rng=rng)
            >>> image.view(wcs=noise.wcs).addNoise(noise)

        This will create noise whose pixel-to-pixel correlations match those of the original
        correlated noise image (in this case, the COSMOS images).  If the input image has no WCS
        set, then it will be treated as having the same WCS as the noise.

        Note that the correlated noise field in ``image`` will be periodic across its boundaries:
        this is due to the fact that the internals of the `BaseCorrelatedNoise` currently use a
        relatively simple implementation of noise generation using the Fast Fourier Transform.
        If you wish to avoid this property being present in your final ``image`` you should add the
        noise to an ``image`` of greater extent than you need, and take a subset.

        Parameters:
            image:      The input `Image` object.
        """
        # Note that this uses the (fast) method of going via the power spectrum and FFTs to generate
        # noise according to the correlation function represented by this instance.  An alternative
        # would be to use the covariance matrices and eigendecomposition.  However, it is O(N^6)
        # operations for an NxN image!  FFT-based noise realization is O(2 N^2 log[N]) so we use it
        # for noise generation applications.

        # Check that the input has defined bounds
        if not isinstance(image, Image):
            raise TypeError("Input image argument must be a galsim.Image.")
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Input image argument must have defined bounds.")

        # If the profile has changed since last time (or if we have never been here before),
        # clear out the stored values.
        self._clear_cache()

        if image.wcs is None:
            wcs = self.wcs
        else:
            wcs = image.wcs.local(image.true_center)

        # Then retrieve or redraw the sqrt(power spectrum) needed for making the noise field
        rootps = self._get_update_rootps(image.array.shape, wcs)

        # Finally generate a random field in Fourier space with the right PS
        noise_array = _generate_noise_from_rootps(self.rng, image.array.shape, rootps)

        # Add it to the image
        image.array[:,:] += noise_array
        return image

    def whitenImage(self, image):
        """Apply noise designed to whiten correlated Gaussian random noise in an input `Image`.

        On input, The `Image`, ``image``, is assumed to have correlated noise described by this
        `BaseCorrelatedNoise` instance.

        On output ``image`` will have been given additional (correlated) noise designed to whiten
        the noise profile.

        Note: the syntax ``image.whitenNoise(noise)`` is normally preferred, but it is equivalent
        to::

            >>> correlated_noise.whitenImage(image)

        If the ``image`` originally contained noise with a correlation function described by the
        ``correlated_noise`` instance, the combined noise after using the whitenImage() method
        will be approximately uncorrelated.  Tests using COSMOS noise fields suggest ~0.3% residual
        off-diagonal covariances after whitening, relative to the variance, although results may
        vary depending on the precise correlation function of the noise field.
        (See ``devel/external/hst/compare_whitening_subtraction.py`` for the COSMOS tests.)

        Note that the code doesn't check that the "if" above is true: the user MUST make sure this
        is the case for the final noise to be uncorrelated.

        Normally, ``image.scale`` is used to determine the input `Image` pixel separation, and if
        ``image.wcs`` is None, it will use the wcs of the noise.  If the image has a non-uniform
        WCS, the local uniform approximation at the center of the image will be used.

        If you are interested in a theoretical calculation of the variance in the final noise field
        after whitening, the whitenImage() method in fact returns this variance.  For example::

            >>> variance = correlated_noise.whitenImage(image)

        **Example**:

        To see noise whitening in action, let us use a model of the correlated noise in COSMOS
        as returned by the `getCOSMOSNoise` function.  Let's initialize and add noise to an image::

            >>> cn = galsim.getCOSMOSNoise()
            >>> image = galsim.ImageD(256, 256, scale=0.03)
            >>> # The scale should match the COSMOS default since didn't specify another
            >>> image.addNoise(cn)

        The ``image`` will then contain a realization of a random noise field with COSMOS-like
        correlation.  Using the whitenImage() method, we can now add more noise to ``image``
        with a power spectrum specifically designed to make the combined noise fields uncorrelated::

            >>> cn.whitenImage(image)

        Of course, this whitening comes at the cost of adding further noise to the image, but
        the algorithm is designed to make this additional noise (nearly) as small as possible.

        Parameters:
            image:      The input `Image` object.

        Returns:
            the theoretically calculated variance of the combined noise fields in the updated image.
        """
        # Note that this uses the (fast) method of going via the power spectrum and FFTs to generate
        # noise according to the correlation function represented by this instance.  An alternative
        # would be to use the covariance matrices and eigendecomposition.  However, it is O(N^6)
        # operations for an NxN image!  FFT-based noise realization is O(2 N^2 log[N]) so we use it
        # for noise generation applications.

        # Check that the input has defined bounds
        if not isinstance(image, Image):
            raise TypeError("Input image not a galsim.Image object")
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Input image argument must have defined bounds.")

        # If the profile has changed since last time (or if we have never been here before),
        # clear out the stored values.
        self._clear_cache()

        if image.wcs is None:
            wcs = self.wcs
        else:
            wcs = image.wcs.local(image.true_center)

        # Then retrieve or redraw the sqrt(power spectrum) needed for making the whitening noise,
        # and the total variance of the combination
        rootps_whitening, variance = self._get_update_rootps_whitening(image.array.shape, wcs)

        # Finally generate a random field in Fourier space with the right PS and add to image
        noise_array = _generate_noise_from_rootps(self.rng, image.array.shape, rootps_whitening)
        image += Image(noise_array)

        # Return the variance to the interested user
        return variance

    def symmetrizeImage(self, image, order=4):
        """Apply noise designed to impose N-fold symmetry on the existing noise in a (square) input
        `Image`.

        On input, The `Image`, ``image``, is assumed to have correlated noise described by this
        `BaseCorrelatedNoise` instance.

        On output ``image`` will have been given additional (correlated) noise designed to
        symmetrize the noise profile.

        When called for a non-square image, this method will raise an exception, unlike the noise
        whitening routines.

        The ``order`` of the symmetry can be supplied as a keyword argument, with the default being
        4 because this is presumably the minimum required for the anisotropy of noise correlations
        to not affect shear statistics.

        Note: the syntax ``image.symmetrizeNoise(noise, order)`` is preferred, but it is equivalent
        to::

            >>> correlated_noise.symmetrizeImage(image, order=order)

        If the ``image`` originally contained noise with a correlation function described by the
        ``correlated_noise`` instance, the combined noise after using the symmetrizeImage() method
        will have a noise correlation function with N-fold symmetry, where ``N=order``.

        Note that the code doesn't check that the "if" above is true: the user MUST make sure this
        is the case for the final noise correlation function to be symmetric in the requested way.

        Normally, ``image.scale`` is used to determine the input pixel separation, and if
        ``image.wcs`` is None, it will use the wcs of the noise.  If the image has a non-uniform
        WCS, the local uniform approximation at the center of the image will be used.

        If you are interested in a theoretical calculation of the variance in the final noise field
        after imposing symmetry, the symmetrizeImage() method in fact returns this variance.
        For example::

            >>> variance = correlated_noise.symmetrizeImage(image, order=order)

        For context, in comparison with the `whitenImage` method for the case of noise
        correlation functions that are roughly like those in the COSMOS HST data, the amount of
        noise added to impose N-fold symmetry is usually much less than what is added to fully
        whiten the noise.  The usage of symmetrizeImage() is totally analogous to the usage of
        `whitenImage`.

        Parameters:
            image:      The square input `Image` object.
            order:      The order at which to require the noise to be symmetric.  All noise fields
                        are already 2-fold symmetric, so ``order`` should be an even integer >2.
                        [default: 4].

        Returns:
            the theoretically calculated variance of the combined noise fields in the updated image.
        """
        # Check that the input has defined bounds
        if not isinstance(image, Image):
            raise TypeError("Input image not a galsim.Image object")
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Input image argument must have defined bounds.")

        # Check that the input is square in shape.
        if image.array.shape[0] != image.array.shape[1]:
            raise GalSimValueError("Input image must be square.", image.array.shape)

        # Check that the input order is an allowed value.
        if order % 2 != 0 or order <= 2:
            raise GalSimValueError("Order must be an even number >=4.", order)

        # If the profile has changed since last time (or if we have never been here before),
        # clear out the stored values.  Note that this cache is not the same as the one used for
        # whitening.
        self._clear_cache()

        if image.wcs is None:
            wcs = self.wcs
        else:
            wcs = image.wcs.local(image.true_center)

        # Then retrieve or redraw the sqrt(power spectrum) needed for making the symmetrizing noise,
        # and the total variance of the combination.
        rootps_symmetrizing, variance = self._get_update_rootps_symmetrizing(
            image.array.shape, wcs, order)

        # Finally generate a random field in Fourier space with the right PS and add to image.
        noise_array = _generate_noise_from_rootps(self.rng, image.array.shape, rootps_symmetrizing)
        image += Image(noise_array)

        # Return the variance to the interested user
        return variance

    def expand(self, scale):
        """Scale the linear scale of correlations in this noise model by ``scale``.

        Scales the linear dimensions of the image by the factor ``scale``, e.g.
        ``half_light_radius`` <-- ``half_light_radius * scale``.

        Parameters:
            scale:      The linear rescaling factor to apply.

        Returns:
            a new `BaseCorrelatedNoise` object with the specified expansion.
        """
        return BaseCorrelatedNoise(self.rng, self._profile.expand(scale), self.wcs)

    def dilate(self, scale):
        """Apply the appropriate changes to the scale and variance for when the object has
        an applied dilation.

        Parameters:
            scale:  The linear dilation scale factor.

        Returns:
            a new `BaseCorrelatedNoise` object with the specified dilation.
        """
        # Expansion changes the flux by scale**2, dilate reverses that to conserve flux,
        # so the variance needs to change by scale**-4.
        return BaseCorrelatedNoise(self.rng, self._profile.expand(scale) / scale**4, self.wcs)

    def magnify(self, mu):
        """Apply the appropriate changes to the scale and variance for when the object has
        an applied magnification ``mu``.

        Parameters:
            mu:     The lensing magnification

        Returns:
            a new `BaseCorrelatedNoise` object with the specified magnification.
        """
        return BaseCorrelatedNoise(self.rng, self._profile.magnify(mu), self.wcs)

    def lens(self, g1, g2, mu):
        """Apply the appropriate changes for when the object has an applied shear and magnification.

        Parameters:
            g1:     First component of lensing (reduced) shear to apply to the object.
            g2:     Second component of lensing (reduced) shear to apply to the object.
            mu:     Lensing magnification to apply to the object.

        Returns:
            a new `BaseCorrelatedNoise` object with the specified shear and magnification.
        """
        return BaseCorrelatedNoise(self.rng, self._profile.lens(g1,g2,mu), self.wcs)

    def rotate(self, theta):
        """Apply a rotation ``theta`` to this correlated noise model.

        Parameters:
            theta:  Rotation angle (`Angle` object, positive means anticlockwise).

        Returns:
            a new `BaseCorrelatedNoise` object with the specified rotation.
        """
        return BaseCorrelatedNoise(self.rng, self._profile.rotate(theta), self.wcs)

    def shear(self, *args, **kwargs):
        """Apply a shear to this correlated noise model, where arguments are either a `Shear`,
        or arguments that will be used to initialize one.

        For more details about the allowed keyword arguments, see the `Shear` docstring.

        Parameters:
            shear:  The shear to be applied. Or, as described above, you may instead supply
                    parameters do construct a shear directly.  eg. ``corr.shear(g1=g1,g2=g2)``.

        Returns:
            a new `BaseCorrelatedNoise` object with the specified shear.
        """
        return BaseCorrelatedNoise(self.rng, self._profile.shear(*args,**kwargs), self.wcs)

    def transform(self, dudx, dudy, dvdx, dvdy):
        """Apply an arbitrary jacobian transformation to this correlated noise model.

        Parameters:
            dudx:   du/dx, where (x,y) are the current coords, and (u,v) are the new coords.
            dudy:   du/dy, where (x,y) are the current coords, and (u,v) are the new coords.
            dvdx:   dv/dx, where (x,y) are the current coords, and (u,v) are the new coords.
            dvdy:   dv/dy, where (x,y) are the current coords, and (u,v) are the new coords.

        Returns:
            a new `BaseCorrelatedNoise` object with the specified transformation.
        """
        return BaseCorrelatedNoise(self.rng, self._profile.transform(dudx,dudy,dvdx,dvdy),
                                   self.wcs)

    def getVariance(self):
        """Return the point variance of this noise field, equal to its correlation function value at
        zero distance.

        This is the variance of values in an image filled with noise according to this model.
        """
        from .position import _PositionD
        # Test whether we can simply return the zero-lag correlation function value, which gives the
        # variance of an image of noise generated according to this model
        if self._profile.is_analytic_x:
            variance = self._profile.xValue(_PositionD(0., 0.))
        else:
            # If the profile has changed since last time (or if we have never been here before),
            # clear out the stored values.
            self._clear_cache()

            # Then use cached version or rebuild if necessary
            if self._variance_cached is not None:
                variance = self._variance_cached
            else:
                imtmp = Image(1, 1, dtype=float)
                # GalSim internals handle this correctly w/out folding
                self.drawImage(imtmp, scale=1.)
                variance = imtmp(1, 1)
                self._variance_cached = variance # Store variance for next time
        return variance

    def withVariance(self, variance):
        """Set the point variance of the noise field, equal to its correlation function value at
        zero distance, to an input ``variance``.  The rest of the correlated noise field is scaled
        proportionally.

        Parameters:
            variance:   The desired point variance in the noise.

        Returns:
            a `BaseCorrelatedNoise` object with the new variance.
        """
        if variance <= 0.:
            raise GalSimValueError("variance must be > 0 in withVariance", variance)
        variance_ratio = variance / self.getVariance()
        return self * variance_ratio

    def withScaledVariance(self, variance_ratio):
        """Scale the entire correlated noise field by the given factor.

        This is equivalent to cn * variance_ratio.

        Parameters:
            variance_ratio:     The factor by which to scale the variance of the correlation
                                function profile.

        Returns:
            a `BaseCorrelatedNoise` object whose variance and covariances have been scaled up by
            the given factor.
        """
        return BaseCorrelatedNoise(self.rng, self._profile * variance_ratio, self.wcs)

    def convolvedWith(self, gsobject, gsparams=None):
        """Convolve the correlated noise model with an input `GSObject`.

        The resulting correlated noise model will then give a statistical description of the noise
        field that would result from convolving noise generated according to the initial correlated
        noise with a kernel represented by ``gsobject`` (e.g. a PSF).

        The practical purpose of this method is that it allows us to model what is happening to
        noise in the images from Hubble Space Telescope that we use for simulating PSF convolved
        galaxies with the `RealGalaxy` class.

        This modifies the representation of the correlation function, but leaves the random number
        generator unchanged.

        **Examples**:

        The following command simply applies a Moffat PSF with slope parameter beta=3. and
        FWHM=0.7::

            >>> cn = cn.convolvedWith(galsim.Moffat(beta=3., fwhm=0.7))

        Often we will want to convolve with more than one function.  For example, if we wanted to
        simulate how a noise field would look if convolved with a ground-based PSF (such as the
        Moffat above) and then rendered onto a new (typically larger) pixel grid, the following
        example command demonstrates the syntax::

            >>> cn = cn.convolvedWith(
            ...    galsim.Convolve([galsim.Deconvolve(galsim.Pixel(0.03)),
            ...                     galsim.Pixel(0.2), galsim.Moffat(3., fwhm=0.7),

        Note, we also deconvolve by the original pixel, which should be the pixel size of the
        image from which the ``correlated_noise`` was made.  This command above is functionally
        equivalent to::

            >>> cn = cn.convolvedWith(galsim.Deconvolve(galsim.Pixel(0.03)))
            >>> cn = cn.convolvedWith(galsim.Pixel(0.2))
            >>> cn = cn.convolvedWith(galsim.Moffat(beta=3., fwhm=0.7))

        as is demanded for a linear operation such as convolution.

        Parameters:
            gsobject:   A `GSObject` or derived class instance representing the function
                        with which the user wants to convolve the correlated noise model.
            gsparams:   An optional `GSParams` argument. [default: None]

        Returns:
            the new `BaseCorrelatedNoise` of the convolved profile.
        """
        from .convolve import Convolve, AutoCorrelate
        conv = Convolve([self._profile, AutoCorrelate(gsobject,gsparams=gsparams)],
                        gsparams=gsparams)
        return BaseCorrelatedNoise(self.rng, conv, self.wcs)

    def drawImage(self, image=None, scale=None, wcs=None, dtype=None, add_to_image=False):
        """A method for drawing profiles storing correlation functions.

        This is a mild reimplementation of the `GSObject.drawImage` method.  The ``method`` is
        automatically set to 'sb' and cannot be changed, and the ``gain`` is set to unity.
        Also, not all the normal parameters of the `GSObject` method are available.

        If ``scale`` and ``wcs`` are not set, and the ``image`` has no ``wcs`` attribute, then this
        will use the wcs of the `BaseCorrelatedNoise` object.

        Parameters:
            image:          If provided, this will be the image on which to draw the profile.
                            If ``image`` is None, then an automatically-sized `Image` will be
                            created.  If ``image`` is given, but its bounds are undefined (e.g. if
                            it was constructed with ``image = galsim.Image()``), then it will be
                            resized appropriately based on the profile's size [default: None].
            scale:          If provided, use this as the pixel scale for the image.  [default: None]
            wcs:            If provided, use this as the wcs for the image (possibly overriding any
                            existing ``image.wcs``).  At most one of ``scale`` or ``wcs`` may be
                            provided.  [default: None]  Note: If no WCS is provided either via
                            ``scale``, ``wcs`` or ``image.wcs``, then the noise object's wcs will
                            be used.
            dtype:          The data type to use for an automatically constructed image.  Only
                            valid if ``image`` is None. [default: None, which means to use
                            numpy.float32]
            add_to_image:   Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing.
                            Note: This requires that ``image`` be provided and that it have defined
                            bounds. [default: False]

        Returns:
            an `Image` of the correlation function.
        """
        wcs = self._profile._determine_wcs(scale, wcs, image, self.wcs)

        return self._profile.drawImage(
            image=image, wcs=wcs, dtype=dtype, method='sb', gain=1.,
            add_to_image=add_to_image, use_true_center=False)

    def drawKImage(self, image=None, nx=None, ny=None, bounds=None, scale=None,
                   add_to_image=False):
        """A method for drawing profiles storing correlation functions (i.e., power spectra) in
        Fourier space.

        This is a mild reimplementation of the `GSObject.drawKImage` method.  The ``gain`` is
        automatically set to unity and cannot be changed.  Also, not all the normal parameters of
        the `GSObject` method are available.

        If ``scale`` is not set, and ``image`` has no ``wcs`` attribute, then this will use the
        wcs of the `BaseCorrelatedNoise` object, which must be a `PixelScale`.

        Parameters:
            image:          If provided, this will be the `Image` onto which to draw the k-space
                            image.  If ``image`` is None, then an automatically-sized image will be
                            created.  If ``image`` is given, but its bounds are undefined, then it
                            will be resized appropriately based on the profile's size.
                            [default: None]
            nx:             If provided and ``image`` is None, use to set the x-direction size of
                            the image.  Must be accompanied by ``ny``.
            ny:             If provided and ``image`` is None, use to set the y-direction size of
                            the image.  Must be accompanied by ``nx``.
            bounds:         If provided and ``image`` is None, use to set the bounds of the image.
            scale:          If provided, use this as the pixel scale, dk, for the images.
                            If ``scale`` is None and ``image`` is given, then take the provided
                            images' pixel scale (which must be equal).
                            If ``scale`` is None and ``image`` is None, then use the Nyquist scale.
                            If ``scale <= 0`` (regardless of ``image``), then use the Nyquist scale.
                            [default: None]
            add_to_image:   Whether to add to the existing images rather than clear out
                            anything in the image before drawing.
                            Note: This requires that ``image`` be provided and that it has defined
                            bounds. [default: False]

        Returns:
            the tuple of `Image` instances, ``(re, im)`` (created if necessary)
        """
        return self._profile.drawKImage(
                image=image, nx=nx, ny=ny, bounds=bounds, scale=scale, add_to_image=add_to_image)

    def _get_update_rootps(self, shape, wcs):
        # Internal utility function for querying the rootps cache, used by applyTo(),
        # whitenImage(), and symmetrizeImage() methods.

        # Query using the rfft2/irfft2 half-sized shape (shape[0], shape[1] // 2 + 1)
        half_shape = (shape[0], shape[1] // 2 + 1)
        key = (half_shape, wcs)

        # Use the cached value if possible.
        rootps = self._rootps_cache.get(key, None)

        # If not, draw the correlation function to the desired size and resolution, then DFT to
        # generate the required array of the square root of the power spectrum
        if rootps is None:
            # Draw this correlation function into an array.  If this is not done at the same wcs as
            # the original image from which the CF derives, even if the image is rotated, then this
            # step requires interpolation and the newcf (used to generate the PS below) is thus
            # approximate at some level
            newcf = Image(shape[1], shape[0], wcs=wcs, dtype=float)
            self.drawImage(newcf)

            # Since we just drew it, save the variance value for posterity.
            var = newcf(newcf.bounds.center)
            self._variance_cached = var

            if var <= 0.:  # pragma: no cover   This should be impossible...
                raise GalSimError("CorrelatedNoise found to have negative variance.")

            # Then calculate the sqrt(PS) that will be used to generate the actual noise.  First do
            # the power spectrum (PS)
            ps = np.fft.rfft2(newcf.array)

            # The PS we expect should be *purely* +ve, but there are reasons why this is not the
            # case.  One is that the PS is calculated from a correlation function CF that has not
            # been rolled to be centred on the [0, 0] array element.  Another reason is due to the
            # approximate nature of the CF rendered above.  Thus an abs(ps) will be necessary when
            # calculating the sqrt().
            # This all means that the performance of correlated noise fields should always be tested
            # for any given scientific application that requires high precision output.  An example
            # of such a test is the generation of noise whitened images of sheared RealGalaxies in
            # Section 9.2 of the GalSim paper (Rowe, Jarvis, Mandelbaum et al. 2014)

            # Given all the above, it might make sense to warn the user if we do detect a PS that
            # doesn't "look right" (i.e. has strongly negative values where these are not expected).
            # This is the subject of Issue #587 on GalSim's GitHub repository page (see
            # https://github.com/GalSim-developers/GalSim/issues/587)

            # For now we just take the sqrt(abs(PS)):
            rootps = np.sqrt(np.abs(ps))

            # Save this in the cache
            self._rootps_cache[key] = rootps

        return rootps

    def _get_update_rootps_whitening(self, shape, wcs, headroom=1.05):
        # Internal utility function for querying the rootps_whitening cache, used by the
        # whitenImage() method, and calculate and update it if not present.

        # Returns: rootps_whitening, variance

        # Query using the rfft2/irfft2 half-sized shape (shape[0], shape[1] // 2 + 1)
        half_shape = (shape[0], shape[1] // 2 + 1)

        key = (half_shape, wcs)

        # Use the cached values if possible.
        rootps_whitening, variance = self._rootps_whitening_cache.get(key, (None,None))

        # If not, calculate the whitening power spectrum as (almost) the smallest power spectrum
        # that when added to rootps**2 gives a flat resultant power that is nowhere negative.
        # Note that rootps = sqrt(power spectrum), and this procedure therefore works since power
        # spectra add (rather like variances).  The resulting power spectrum will be all positive
        # (and thus physical).
        if rootps_whitening is None:

            rootps = self._get_update_rootps(shape, wcs)
            ps_whitening = -rootps * rootps
            ps_whitening += np.abs(np.min(ps_whitening)) * headroom # Headroom adds a little extra
            rootps_whitening = np.sqrt(ps_whitening)                # variance, for "safety"

            # Finally calculate the theoretical combined variance to output alongside the image
            # to be generated with the rootps_whitening.  Note that although we use the [0, 0]
            # element we could use any as the PS should be flat.
            variance = rootps[0, 0]**2 + ps_whitening[0, 0]

            # Then add all this and the relevant wcs to the _rootps_whitening_cache
            self._rootps_whitening_cache[key] = (rootps_whitening, variance)

        return rootps_whitening, variance

    def _get_update_rootps_symmetrizing(self, shape, wcs, order, headroom=1.02):
        # Internal utility function for querying the ``rootps_symmetrizing`` cache, used by the
        # symmetrizeImage() method, and calculate and update it if not present.

        # Returns: rootps_symmetrizing, variance

        # Query using the rfft2/irfft2 half-sized shape (shape[0], shape[1] // 2 + 1)
        half_shape = (shape[0], shape[1] // 2 + 1)

        key = (half_shape, wcs, order)

        # Use the cached values if possible.
        rootps_symmetrizing, variance = self._rootps_symmetrizing_cache.get(key, (None,None))

        # If not, calculate the symmetrizing power spectrum as (almost) the smallest power spectrum
        # that when added to rootps**2 gives a power that has N-fold symmetry, where `N=order`.
        # Note that rootps = sqrt(power spectrum), and this procedure therefore works since power
        # spectra add (rather like variances).  The resulting power spectrum will be all positive
        # (and thus physical).
        if rootps_symmetrizing is None:

            rootps = self._get_update_rootps(shape, wcs)
            ps_actual = rootps * rootps
            # This routine will get a PS that is a symmetrized version of `ps_actual` at the desired
            # order, that also satisfies the requirement of being >= ps_actual for all k values.
            ps_symmetrized = self._get_symmetrized_ps(ps_actual, order)
            ps_symmetrizing = ps_symmetrized * headroom - ps_actual # add a little extra variance
            rootps_symmetrizing = np.sqrt(ps_symmetrizing)

            # Finally calculate the theoretical combined variance to output alongside the image to
            # be generated with the rootps_symmetrizing.
            # Here, unlike in _get_update_rootps_whitening, the final power spectrum is not flat, so
            # we have to take the mean power instead of just using the [0, 0] element.
            # Note that the mean of the power spectrum (fourier space) is the zero lag value in
            # real space, which is the desired variance.
            variance = np.mean(rootps**2 + ps_symmetrizing)

            # Then add all this and the relevant wcs to the _rootps_symmetrizing_cache
            self._rootps_symmetrizing_cache[key] = (rootps_symmetrizing, variance)

        return rootps_symmetrizing, variance

    def _get_symmetrized_ps(self, ps, order):
        # Internal utility function for taking an input power spectrum and generating a version of
        # it with symmetry at a given order.

        # We make an image of the PS and turn it into an galsim.InterpolatedImage in order to carry
        # out the necessary rotations using well-tested interpolation routines.  We will also
        # require the output to be strictly >= the input noise power spectrum, so that it should be
        # possible to generate noise with power equal to the difference between the two power
        # spectra.

        from .interpolatedimage import InterpolatedImage
        from .angle import radians
        # Initialize a temporary copy of the original PS array, expanded to full size rather than
        # the compact halfcomplex format that the PS is supplied in, which we will turn into an
        # InterpolatedImage
        # Check for an input ps which was even-sized along the y axis, which needs special treatment
        do_expansion = False
        if ps.shape[0] % 2 == 0:
            do_expansion = True
        # Then roll the PS by half its size in the leading dimension, centering it in that dimension
        # (we will construct the expanded array to be centred in the other dimension)
        ps_rolled = utilities.roll2d(ps, (ps.shape[0] // 2, 0))
        # Then create and fill an expanded-size tmp_arr with this PS
        if not do_expansion:
            tmp_arr = np.zeros((ps_rolled.shape[0], 2 * ps_rolled.shape[1] - 1)) # Both dims now odd
            # Fill the first half, the RHS...
            tmp_arr[:, ps.shape[1]-1:] = ps_rolled
            # Then do the LHS of tmp_arr, straightforward enough, fill with the inverted RHS
            tmp_arr[:, :ps.shape[1]-1] = ps_rolled[:, 1:][::-1, ::-1]
        else:
            # For the even-sized leading dimension ps, we have to do a tiny bit more work than
            # the odd case...
            tmp_arr = np.zeros((ps_rolled.shape[0] + 1, 2 * ps_rolled.shape[1] - 1))
            tmp_arr[:-1, ps_rolled.shape[1]-1:] = ps_rolled
            tmp_arr[1:, :ps_rolled.shape[1]-1] = ps_rolled[:, 1:][::-1, ::-1]
            # Then one tiny element breaks the symmetry of the above, so fix this
            tmp_arr[-1, tmp_arr.shape[1] // 2] = tmp_arr[0, tmp_arr.shape[1] // 2]

        # Also initialize the array in which to build up the symmetrized PS.
        final_arr = tmp_arr.copy()
        tmp_im = Image(tmp_arr, scale=1)
        tmp_obj = InterpolatedImage(tmp_im, calculate_maxk=False, calculate_stepk=False)

        # Now loop over the rotations by 2pi/order.
        for i_rot in range(order):
            # For the first one, we don't rotate at all.
            if i_rot > 0:
                # For later ones, rotate by 2pi/order, and draw it back into a new image.
                tmp_obj = tmp_obj.rotate(2.*np.pi*radians/order)
                tmp_im = Image(tmp_arr.shape[1], tmp_arr.shape[0], scale=1)
                tmp_obj.drawImage(tmp_im, scale=1, method='sb')
                final_arr[tmp_im.array > final_arr] = tmp_im.array[tmp_im.array > final_arr]

        # Now simply take the halfcomplex, compact stored part that we are interested in,
        # remembering that the kx=ky=0 element is still in the centre
        final_arr = final_arr[:, final_arr.shape[1]//2:]
        # If we extended the array to be odd-sized along y, we have to go back to an even subarray
        if do_expansion: final_arr = final_arr[:-1, :]
        # Finally roll back the leading dimension
        final_arr = utilities.roll2d(final_arr, (-(final_arr.shape[0] // 2), 0))
        # final_arr now contains the halfcomplex compact format PS of the maximum of the set of PS
        # images rotated by 2pi/order, which (a) should be symmetric at the required order and
        # (b) be the minimal array that is symmetric at that order and >= the original PS.  So we do
        # not have to add any more noise to ensure that the target symmetrized PS is always >= the
        # original one.
        return final_arr

###
# Now a standalone utility function for generating noise according to an input (square rooted)
# Power Spectrum
#
def _generate_noise_from_rootps(rng, shape, rootps):
    # Utility function for generating a NumPy array containing a Gaussian random noise field with
    # a user-specified power spectrum also supplied as a NumPy array.

    # shape is the shape of the output array, needed because of the use of Hermitian symmetry to
    #       increase inverse FFT efficiency using the np.fft.irfft2 function (gets sent
    #       to the kwarg s of np.fft.irfft2)
    # rootps is a NumPy array containing the square root of the discrete power spectrum ordered
    #        in two dimensions according to the usual DFT pattern for np.fft.rfft2 output

    # Returns a NumPy array (contiguous) of the requested shape, filled with the noise field.

    from .random import GaussianDeviate
    # Quickest to create Gaussian rng each time needed, so do that here...
    # Note sigma scaling: 1/sqrt(2) needed so <|gaussvec|**2> = product(shape)
    # shape needed because of the asymmetry in the 1/N^2 division in the NumPy FFT/iFFT
    gd = GaussianDeviate(rng, sigma=np.sqrt(.5 * shape[0] * shape[1]))

    # Fill a couple of arrays with this noise
    gvec_real = utilities.rand_arr((shape[0], shape[1]//2+1), gd)
    gvec_imag = utilities.rand_arr((shape[0], shape[1]//2+1), gd)
    # Prepare a complex vector upon which to impose Hermitian symmetry
    gvec = gvec_real + 1J * gvec_imag
    # Now impose requirements of Hermitian symmetry on random Gaussian halfcomplex array, and ensure
    # self-conjugate elements (e.g. [0, 0]) are purely real and multiplied by sqrt(2) to compensate
    # for lost variance, see https://github.com/GalSim-developers/GalSim/issues/563
    # First do the bits necessary for both odd and even shapes:
    gvec[-1:shape[0]//2:-1, 0] = np.conj(gvec[1:(shape[0]+1)//2, 0])
    rt2 = np.sqrt(2.)
    gvec[0, 0] = rt2 * gvec[0, 0].real
    # Then make the changes necessary for even sized arrays
    if shape[1] % 2 == 0: # x dimension even
        gvec[-1:shape[0]//2:-1, shape[1]//2] = np.conj(gvec[1:(shape[0]+1)//2, shape[1]//2])
        gvec[0, shape[1]//2] = rt2 * gvec[0, shape[1]//2].real
    if shape[0] % 2 == 0: # y dimension even
        gvec[shape[0]//2, 0] = rt2 * gvec[shape[0]//2, 0].real
        # Both dimensions even
        if shape[1] % 2 == 0:
            gvec[shape[0]//2, shape[1]//2] = rt2 * gvec[shape[0]//2, shape[1]//2].real
    # Finally generate and return noise using the irfft
    return np.fft.irfft2(gvec * rootps, s=shape)

###
# Then we define the CorrelatedNoise, which generates a correlation function by estimating it
# directly from images:
#
class CorrelatedNoise(BaseCorrelatedNoise):
    """A class that represents 2D correlated noise fields calculated from an input `Image`.

    This class stores an internal representation of a 2D, discrete correlation function, and allows
    a number of subsequent operations including interpolation, shearing, magnification and rendering
    of the correlation function profile into an output `Image`.

    The class also allows correlated Gaussian noise fields to be generated according to the
    correlation function, and added to an `Image`: see `BaseCorrelatedNoise.applyTo`.

    It also provides methods for whitening or imposing N-fold symmetry on pre-existing noise that
    shares the same spatial correlations: see `BaseCorrelatedNoise.whitenImage` and
    `BaseCorrelatedNoise.symmetrizeImage`, respectively.

    It also allows the combination of multiple correlation functions by addition, and for the
    scaling of the total variance they represent by scalar factors.

    Parameters:
        image:              The image from which to derive the correlated noise profile
        rng:                A `BaseDeviate` instance to use for generating the random numbers.
        scale:              If provided, use this as the pixel scale.  Normally, the scale (or wcs)
                            is taken from the image.wcs field, but you may override that by
                            providing either scale or wcs.  [default: use image.wcs if defined,
                            else 1.0, unless ``wcs`` is provided]
        wcs:                If provided, use this as the wcs for the image.  At most one of
                            ``scale`` or ``wcs`` may be provided. [default: None]
        x_interpolant:      The interpolant to use for interpolating the image of the correlation
                            function. (See below.) [default: galsim.Linear()]
        correct_periodicity: Whether to correct for the effects of periodicity.  (See below.)
                            [default: True]
        subtract_mean:      Whether to subtract off the mean value from the image before computing
                            the correlation function. [default: False]
        gsparams:           An optional `GSParams` argument. [default: None]

    **Basic example**::

        >>> cn = galsim.CorrelatedNoise(image, rng=rng)

    Instantiates a CorrelatedNoise using the pixel scale information contained in ``image.scale``
    (assumes the scale is unity if ``image.scale <= 0.``) by calculating the correlation function
    in the input ``image``.  The input ``rng`` must be a `BaseDeviate` or derived class instance,
    setting the random number generation for the noise.

    **Optional Inputs**::

        >>> cn = galsim.CorrelatedNoise(image, rng=rng, scale=0.2)

    The example above instantiates a CorrelatedNoise, but forces the use of the pixel scale
    ``scale`` to set the units of the internal lookup table.::

        >>> cn = galsim.CorrelatedNoise(image, rng=rng, x_interpolant=galsim.Lanczos(5))

    The example above instantiates a CorrelatedNoise, but forces use of a non-default interpolant
    for interpolation of the internal lookup table in real space.

    The default ``x_interpolant`` is ``galsim.Linear()``, which uses bilinear interpolation.
    The use of this interpolant is an approximation that gives good empirical results without
    requiring internal convolution of the correlation function profile by a `Pixel` object when
    applying correlated noise to images: such an internal convolution has been found to be
    computationally costly in practice, requiring the Fourier transform of very large arrays.

    The use of the bilinear interpolants means that the representation of correlated noise will be
    noticeably inaccurate in at least the following two regimes:

      i)  If the pixel scale of the desired final output (e.g. the target image of
          `BaseCorrelatedNoise.drawImage`, `BaseCorrelatedNoise.applyTo` or
          `BaseCorrelatedNoise.whitenImage`) is small relative to the separation between pixels
          in the ``image`` used to instantiate ``cn`` as shown above.
      ii) If the CorrelatedNoise instance ``cn`` was instantiated with an image of scale comparable
          to that in the final output, and ``cn`` has been rotated or otherwise transformed (e.g.
          via the `BaseCorrelatedNoise.rotate`, `BaseCorrelatedNoise.shear` methods; see below).

    Conversely, the approximation will work best in the case where the correlated noise used to
    instantiate the ``cn`` is taken from an input image for which ``image.scale`` is smaller than
    that in the desired output.  This is the most common use case in the practical treatment of
    correlated noise when simulating galaxies from space telescopes, such as COSMOS.

    Changing from the default bilinear interpolant is made possible, but not recommended.  For more
    information please see the discussion on https://github.com/GalSim-developers/GalSim/pull/452.

    There is also an option to switch off an internal correction for assumptions made about the
    periodicity in the input noise image.  If you wish to turn this off you may, e.g.::

        >>> cn = galsim.CorrelatedNoise(image, rng=rng, correct_periodicity=False)

    The default and generally recommended setting is ``correct_periodicity=True``.

    Users should note that the internal calculation of the discrete correlation function in
    ``image`` will assume that ``image`` is periodic across its boundaries, introducing a dilution
    bias in the estimate of inter-pixel correlations that increases with separation.  Unless you
    know that the noise in ``image`` is indeed periodic (perhaps because you generated it to be so),
    you will not generally wish to use the ``correct_periodicity=False`` option.

    By default, the image is not mean subtracted before the correlation function is estimated.  To
    do an internal mean subtraction, you can set the ``subtract_mean`` keyword to ``True``, e.g.::

        >>> cn = galsim.CorrelatedNoise(image, rng=rng, subtract_mean=True)

    Using the ``subtract_mean`` option will introduce a small underestimation of variance and other
    correlation function values due to a bias on the square of the sample mean.  This bias reduces
    as the input image becomes larger, and in the limit of uncorrelated noise tends to the constant
    term ``variance/N**2`` for an N x N sized ``image``.

    It is therefore recommended that a background/sky subtraction is applied to the ``image`` before
    it is given as an input to the `CorrelatedNoise`, allowing the default ``subtract_mean=False``.
    If such a background model is global or based on large regions on sky then assuming that the
    image has a zero population mean will be reasonable, and won't introduce a bias in covariances
    from an imperfectly-estimated sample mean subtraction.  If this is not possible, just be aware
    that ``subtract_mean=True`` will bias the correlation function low to some level.

    You may also specify a gsparams argument.  See the docstring for `GSParams` for more
    information about this option.

    **Methods**:

    The main way that a CorrelatedNoise is used is to add correlated noise to an image.
    The syntax::

        >>> image.addNoise(cn)

    is preferred, although::

        >>> cn.applyTo(image)

    is equivalent.  See the `Image.addNoise` method docstring for more information.  The
    ``image.scale`` is used to get the pixel scale of the input image unless this is <= 0, in which
    case a scale of 1 is assumed.

    A number of methods familiar from `GSObject` instances have also been implemented directly as
    ``cn`` methods, so that the following commands are all legal::

        >>> image = cn.drawImage(im, scale)
        >>> cn = cn.shear(s)
        >>> cn = cn.expand(m)
        >>> cn = cn.rotate(theta * galsim.degrees)
        >>> cn = cn.transform(dudx, dudy, dvdx, dvdy)

    See the individual method docstrings for more details.  The ``shift`` method is not available
    since a correlation function must always be centred and peaked at the origin.

    The methods::

        >>> var = cn.getVariance()
        >>> cn1 = cn.withVariance(variance)
        >>> cn2 = cn.withScaledVariance(variance_ratio)

    can be used to get and set the point variance of the correlated noise, equivalent to the zero
    separation distance correlation function value.

    The `BaseCorrelatedNoise.withVariance` method scales the whole internal correlation function so
    that its point variance matches ``variance``.

    Similarly, `BaseCorrelatedNoise.withScaledVariance` scales the entire function by the given
    factor.

    **Arithmetic Operators**:

    Addition, multiplication and division operators are defined to work in an intuitive way for
    correlation functions.

    Addition works simply to add the internally-stored correlation functions, so that::

        >>> cn3 = cn2 + cn1
        >>> cn4 += cn5

    provides a representation of the correlation function of two linearly summed fields represented
    by the individual correlation function operands.

    What happens to the internally stored random number generators in the examples above?  For all
    addition operations it is the `BaseDeviate` belonging to the instance on the *left-hand side*
    of the operator that is retained.

    In the example above therefore, it is the random number generator from ``cn2`` that will be
    stored and used by ``cn3``, and ``cn4`` will retain its random number generator after in-place
    addition of ``cn5``.  The random number generator of ``cn5`` is not affected by the operation.

    The multiplication and division operators, e.g.::

        >>> cn1 /= 3.
        >>> cn2 = cn1 * 3

    scale the overall correlation function by a scalar operand.  The random number generators are
    not affected by these scaling operations.
    """
    def __init__(self, image, rng=None, scale=None, wcs=None, x_interpolant=None,
                 correct_periodicity=True, subtract_mean=False, gsparams=None):
        from .wcs import BaseWCS
        from .interpolant import Linear
        from .interpolatedimage import InterpolatedImage

        # Check that the input image is in fact a galsim.ImageSIFD class instance
        if not isinstance(image, Image):
            raise TypeError("Input image not a galsim.Image object")
        # Build a noise correlation function (CF) from the input image, using DFTs
        # Calculate the power spectrum then a (preliminary) CF
        ft_array = np.fft.rfft2(image.array)
        ps_array = np.abs(ft_array)**2 # Using timeit abs() seems to have the slight speed edge over
                                       # all other options tried, cf. results described by MJ in
                                       # the optics.psf() function in optics.py

        # Need to normalize ps due to one-directional 1/N^2 in FFT conventions and the fact that
        # we *squared* the ft_array to get ps_array:
        ps_array /= np.product(image.array.shape)

        if subtract_mean: # Quickest non-destructive way to make the PS correspond to the
                          # mean-subtracted case
            ps_array[0, 0] = 0.

        # Then calculate the CF by inverse DFT
        cf_array_prelim = np.fft.irfft2(ps_array, s=image.array.shape)

        store_rootps = True # Currently the ps_array above corresponds to cf, but this may change...

        # Apply a correction for the DFT assumption of periodicity unless user requests otherwise
        if correct_periodicity:
            cf_array_prelim *= _cf_periodicity_dilution_correction(cf_array_prelim.shape)
            store_rootps = False

        # Roll CF array to put the centre in image centre.  Remember that numpy stores data [y,x]
        cf_array_prelim = utilities.roll2d(
            cf_array_prelim, (cf_array_prelim.shape[0] // 2, cf_array_prelim.shape[1] // 2))

        # The underlying C++ object is expecting the CF to be represented by an odd-dimensioned
        # array with the central pixel denoting the zero-distance correlation (variance), even
        # even if the input image was even-dimensioned on one or both sides.
        # We therefore copy-paste and zero pad the CF calculated above to ensure that these
        # expectations are met.
        #
        # Determine the largest dimension of the input image, and use it to generate an empty CF
        # array for final output, padding by one to make odd if necessary:
        cf_array = np.zeros((
            1 + 2 * (cf_array_prelim.shape[0] // 2),
            1 + 2 * (cf_array_prelim.shape[1] // 2))) # using integer division

        # Then put the data from the prelim CF into this array
        cf_array[0:cf_array_prelim.shape[0], 0:cf_array_prelim.shape[1]] = cf_array_prelim
        # Then copy-invert-paste data from the leftmost column to the rightmost column, and lowest
        # row to the uppermost row, if the original CF had even dimensions in the x and y
        # directions, respectively (remembering again that NumPy stores data [y,x] in arrays)
        if cf_array_prelim.shape[1] % 2 == 0: # first do x
            lhs_column = cf_array[:, 0]
            cf_array[:, cf_array_prelim.shape[1]] = lhs_column[::-1] # inverts order as required
        if cf_array_prelim.shape[0] % 2 == 0: # then do y
            bottom_row = cf_array[0, :]
            cf_array[cf_array_prelim.shape[0], :] = bottom_row[::-1] # inverts order as required

        # Wrap correlation function in an image
        cf_image = Image(np.ascontiguousarray(cf_array))

        # Set the wcs if necessary
        if wcs is not None:
            if scale is not None:
                raise GalSimIncompatibleValuesError("Cannot provide both wcs and scale",
                                                    scale=scale, wcs=scale)
            if not isinstance(wcs, BaseWCS):
                raise TypeError("wcs must be a BaseWCS instance")
            if not wcs._isUniform:
                raise GalSimValueError("Cannot provide non-uniform wcs", wcs)
            cf_image.wcs = wcs
        elif scale is not None:
            cf_image.scale = scale
        elif image is not None and image.wcs is not None:
            cf_image.wcs = image.wcs.local(image.true_center)

        # If wcs is still None at this point or is a PixelScale <= 0., use scale=1.
        if cf_image.wcs is None or (cf_image.wcs._isPixelScale and cf_image.wcs.scale <= 0):
            cf_image.scale = 1.

        # If x_interpolant not specified on input, use bilinear
        if x_interpolant is None:
            x_interpolant = Linear()
        else:
            x_interpolant = utilities.convert_interpolant(x_interpolant)

        # Then initialize...
        cf_object = InterpolatedImage(
            cf_image, x_interpolant=x_interpolant, normalization="sb",
            calculate_stepk=False, calculate_maxk=False, #<-these internal calculations do not seem
            gsparams=gsparams)                           #  to do very well with often sharp-peaked
                                                         #  correlation function images...
        BaseCorrelatedNoise.__init__(self, rng, cf_object, cf_image.wcs)

        if store_rootps:
            # If it corresponds to the CF above, store in the cache
            self._profile_for_cached = self._profile
            shape = ps_array.shape
            half_shape = (shape[0], shape[1] // 2 + 1)
            key = (half_shape, cf_image.wcs)
            self._rootps_cache[key] = np.sqrt(ps_array)

        self._image = image

    def __str__(self):
        return "galsim.CorrelatedNoise(%s, wcs=%s)"%(self._image, self.wcs)


def _cf_periodicity_dilution_correction(cf_shape):
    """Return an array containing the correction factor required for wrongly assuming periodicity
    around noise field edges in an DFT estimate of the discrete correlation function.

    Uses the result calculated by MJ on GalSim Pull Request #366.
    See https://github.com/GalSim-developers/GalSim/pull/366.

    Returns a 2D NumPy array with the same shape as the input parameter tuple ``cf_shape``.  This
    array contains the correction factor by which elements in the naive CorrelatedNoise estimate of
    the discrete correlation function should be multiplied to correct for the erroneous assumption
    of periodic boundaries in an input noise field.

    Note this should be applied only to correlation functions that have *not* been rolled to place
    the origin at the array centre.  The convention used here is that the lower left corner is the
    [0, 0] origin, following standard FFT conventions (see e.g numpy.fft.fftfreq).  You should
    therefore only apply this correction before using galsim.utilities.roll2d() to recentre the
    image of the correlation function.
    """
    # First calculate the Delta_x, Delta_y
    deltax, deltay = np.meshgrid( # Remember NumPy array shapes are [y, x]
        np.fft.fftfreq(cf_shape[1]) * float(cf_shape[1]),
        np.fft.fftfreq(cf_shape[0]) * float(cf_shape[0]))
    # Then get the dilution correction
    correction = (
        cf_shape[1] * cf_shape[0] / (cf_shape[1] - np.abs(deltax)) / (cf_shape[0] - np.abs(deltay)))
    return correction


# Free function for returning a COSMOS noise field correlation function
def getCOSMOSNoise(file_name=None, rng=None, cosmos_scale=0.03, variance=0., x_interpolant=None,
                   gsparams=None):
    """Returns a representation of correlated noise in the HST COSMOS F814W unrotated science coadd
    images.

    See http://cosmos.astro.caltech.edu/astronomer/hst.html for information about the COSMOS survey,
    and Leauthaud et al (2007) for detailed information about the unrotated F814W coadds used for
    weak lensing science.

    This function uses a stacked estimate of the correlation function in COSMOS noise fields.
    The correlation function was computed by the GalSim team as described in::

        GalSim/devel/external/hst/make_cosmos_cfimage.py

    The resulting file is distributed with GalSim as::

        os.path.join('galsim.meta_data.share_dir', 'acs_I_unrot_sci_20_cf.fits')

    Parameters:
        file_name:      If provided, override the usual location of the file with the given
                        file name.  [default: None]
        rng:            If provided, a random number generator to use as the random number
                        generator of the resulting noise object. (may be any kind of
                        `BaseDeviate` object) [default: None, in which case, one will be
                        automatically created, using the time as a seed.]
        cosmos_scale:   COSMOS ACS F814W coadd image pixel scale in the units you are using to
                        describe `GSObject` instances and image scales in GalSim. [default: 0.03
                        (arcsec), see below for more information.]
        variance:       Scales the correlation function so that its point variance, equivalent
                        to its value at zero separation distance, matches this value.
                        [default: 0., which means to use the variance in the original COSMOS
                        noise fields.]
        x_interpolant:  Forces use of a non-default interpolant for interpolation of the
                        internal lookup table in real space.  See below for more details.
                        [default: galsim.Linear()]
        gsparams:       An optional `GSParams` argument. [default: None]

    Returns:
        a `BaseCorrelatedNoise` instance representing correlated noise in F814W COSMOS images.

    The default ``x_interpolant`` is a ``galsim.Linear()``, which uses bilinear interpolation.
    The use of this interpolant is an approximation that gives good empirical results without
    requiring internal convolution of the correlation function profile by a `Pixel` object when
    applying correlated noise to images: such an internal convolution has been found to be
    computationally costly in practice, requiring the Fourier transform of very large arrays.

    The use of the bilinear interpolants means that the representation of correlated noise will be
    noticeably inaccurate in at least the following two regimes:

    1. If the pixel scale of the desired final output (e.g. the target image of
       `BaseCorrelatedNoise.drawImage`, `BaseCorrelatedNoise.applyTo` or
       `BaseCorrelatedNoise.whitenImage`) is small relative to the separation between pixels in
       the ``image`` used to instantiate ``cn`` as shown above.
    2. If the `BaseCorrelatedNoise` instance ``cn`` was instantiated with an image of scale
       comparable to that in the final output, and ``cn`` has been rotated or otherwise transformed
       (e.g.  via the `BaseCorrelatedNoise.rotate`, `BaseCorrelatedNoise.shear` methods; see below).

    Conversely, the approximation will work best in the case where the correlated noise used to
    instantiate the ``cn`` is taken from an input image for which ``image.scale`` is smaller than
    that in the desired output.  This is the most common use case in the practical treatment of
    correlated noise when simulating galaxies from COSMOS, for which this function is expressly
    designed.

    Changing from the default bilinear interpolant is made possible, but not recommended.  For more
    information please see the discussion on https://github.com/GalSim-developers/GalSim/pull/452.

    You may also specify a gsparams argument.  See the docstring for `GSParams` for more
    information about this option.

    .. note::

        The ACS coadd images in COSMOS have a pixel scale of 0.03 arcsec, and so the pixel scale
        ``cosmos_scale`` adopted in the representation of of the correlation function takes a
        default value ``cosmos_scale = 0.03``

        If you wish to use other units, ensure that the input keyword ``cosmos_scale`` takes the
        value corresponding to 0.03 arcsec in your chosen system.

    **Example**:

    The following commands use this function to generate a 300 pixel x 300 pixel image of noise with
    HST COSMOS correlation properties (substitute in your own file and path for the ``filestring``)::

        >>> rng = galsim.BaseDeviate(123456)
        >>> noise = galsim.getCOSMOSNoise(rng=rng)
        >>> image = galsim.ImageD(nx, ny, scale=0.03)
        >>> image.addNoise(cf)

    If your image has some other pixel scale or a complicated WCS, then the applied noise will
    have the correct correlations in world coordinates, which may not be what you wanted if you
    expected the pixel-to-pixel correlations to match the COSMOS noise profile.  However, in
    this case, you would want to view your image with the COSMOS pixel scale when you apply
    the noise::

        >>> image = galsim.Image(nx, ny, wcs=complicated_wcs)
        >>> noise = galsim.getCOSMOSNoise(rng=rng)
        >>> image.view(wcs=noise.wcs).addNoise(noise)

    The FITS file ``out.fits`` should then contain an image of randomly-generated, COSMOS-like noise.
    """
    from . import meta_data
    from . import fits
    from .interpolant import Linear
    from .interpolatedimage import InterpolatedImage
    # First try to read in the image of the COSMOS correlation function stored in the repository
    import os
    if file_name is None:
        file_name = os.path.join(meta_data.share_dir,'acs_I_unrot_sci_20_cf.fits')
    if not os.path.isfile(file_name):
        raise OSError("The file %r does not exist."%(file_name))
    try:
        cfimage = fits.read(file_name)
    except (IOError, OSError, AttributeError, TypeError) as e:
        raise OSError("Unable to read COSMOSNoise file %s.\n%r"%(file_name,e))

    # Then check for negative variance before doing anything time consuming
    if variance < 0:
        raise GalSimRangeError("Specified variance must be zero or positive.",
                               variance, 0, None)

    # If x_interpolant not specified on input, use bilinear
    if x_interpolant is None:
        x_interpolant = Linear()
    else:
        x_interpolant = utilities.convert_interpolant(x_interpolant)

    # Use this info to then generate a correlated noise model DIRECTLY: note this is non-standard
    # usage, but tolerated since we can be sure that the input cfimage is appropriately symmetric
    # and peaked at the origin
    ii = InterpolatedImage(cfimage, scale=cosmos_scale, normalization="sb",
                           calculate_stepk=False, calculate_maxk=False,
                           x_interpolant=x_interpolant, gsparams=gsparams)
    ret = BaseCorrelatedNoise(rng, ii, PixelScale(cosmos_scale))
    # If the input keyword variance is non-zero, scale the correlation function to have this
    # variance
    if variance > 0.:
        ret = ret.withVariance(variance)
    return ret

class UncorrelatedNoise(BaseCorrelatedNoise):
    """A class that represents 2D correlated noise fields that are actually (at least initially)
    uncorrelated.  Subsequent applications of things like `BaseCorrelatedNoise.shear` or
    `BaseCorrelatedNoise.convolvedWith` will induce correlations.

    The noise is characterized by a variance in each image pixel and a pixel size and shape.
    The ``variance`` value refers to the noise variance in each pixel.  If the pixels are square
    (the usual case), you can specify the size using the ``scale`` parameter.  If not, they
    are effectively specified using the local wcs function that defines the pixel shape.  i.e.::

        >>> world_pix = wcs.toWorld(Pixel(1.))

    should return the pixel profile in world coordinates.

    Parameters:
        variance:       The noise variance value to model as being uniform and uncorrelated
                        over the whole image.
        rng:            If provided, a random number generator to use as the random number
                        generator of the resulting noise object. (may be any kind of
                        `BaseDeviate` object) [default: None, in which case, one will be
                        automatically created, using the time as a seed.]
        scale:          If provided, use this as the pixel scale.  [default: 1.0, unless ``wcs`` is
                        provided]
        wcs:            If provided, use this as the wcs for the image.  At most one of ``scale``
                        or ``wcs`` may be provided. [default: None]
        gsparams:       An optional `GSParams` argument. [default: None]
    """
    def __init__(self, variance, rng=None, scale=None, wcs=None, gsparams=None):
        from .wcs import BaseWCS, PixelScale
        from .box import Pixel
        from .convolve import AutoConvolve
        if variance < 0:
            raise GalSimRangeError("Specified variance must be zero or positive.",
                                   variance, 0, None)

        if wcs is not None:
            if scale is not None:
                raise GalSimIncompatibleValuesError("Cannot provide both wcs and scale",
                                                    scale=scale, wcs=wcs)
            if not isinstance(wcs, BaseWCS):
                raise TypeError("wcs must be a BaseWCS instance")
            if not wcs._isUniform:
                raise GalSimValueError("Cannot provide non-uniform wcs", wcs)
        elif scale is not None:
            wcs = PixelScale(scale)
        else:
            wcs = PixelScale(1.0)

        # Save the things that won't get saved by the base class, for use in repr.
        self.variance = variance
        self._gsparams = GSParams.check(gsparams)

        # Need variance == xvalue(0,0) after autoconvolution
        # So the Pixel needs to have an amplitude of sigma at (0,0)
        import math
        sigma = math.sqrt(variance)
        pix = Pixel(scale=1.0, flux=sigma, gsparams=gsparams)
        cf = AutoConvolve(pix, real_space=True, gsparams=gsparams)
        world_cf = wcs.profileToWorld(cf)
        # This gets the shape right, but not the amplitude.  Need to rescale by the pixel area
        world_cf *= wcs.pixelArea()
        BaseCorrelatedNoise.__init__(self, rng, world_cf, wcs)

    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams and not kwargs: return self
        gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        return UncorrelatedNoise(self.variance, self.rng, wcs=self.wcs, gsparams=gsparams)

    def __repr__(self):
        return "galsim.UncorrelatedNoise(%r, %r, wcs=%r, gsparams=%r)"%(
            self.variance, self.rng, self.wcs, self._gsparams)

    def __str__(self):
        return "galsim.UncorrelatedNoise(variance=%r, wcs=%s)"%(self.variance, self.wcs)


class CovarianceSpectrum(object):
    """Class to hold a `ChromaticSum` noise covariance spectrum (which is a generalization of a
    power spectrum or equivalently a correlation function).

    Analogous to how a `galsim.CorrelatedNoise` object stores the variance and covariance of a
    `galsim.Image` object, a `galsim.CovarianceSpectrum` stores the variance and covariance of the
    Fourier mode amplitudes in different components of a `ChromaticSum`.

    Note that the covariance in question exists between different `SED` components of the
    `ChromaticSum`, and not between different Fourier modes, which are assumed to be uncorrelated.
    This structure arises naturally for a `ChromaticRealGalaxy` (see devel/modules/CGNotes.pdf for
    more details).

    Parameters:
        Sigma:     A dictionary whose keys are tuples numerically indicating a pair of
                    `ChromaticSum` components whose Fourier mode amplitude covariances are
                    described by the corresponding `GSObject` values.
        SEDs:      `SED` instances of associated `ChromaticSum` components.
    """
    def __init__(self, Sigma, SEDs):
        self.Sigma = Sigma
        self.SEDs = SEDs

    def __mul__(self, variance_ratio):
        return self.withScaledVariance(variance_ratio)

    def transform(self, dudx, dudy, dvdx, dvdy):
        Sigma = {}
        for k, v in self.Sigma.items():
            Sigma[k] = v.transform(dudx, dudy, dvdx, dvdy)
        return CovarianceSpectrum(Sigma, self.SEDs)

    def withScaledVariance(self, variance_ratio):
        Sigma = {}
        for k, v in self.Sigma.items():
            Sigma[k] = v * variance_ratio
        return CovarianceSpectrum(Sigma, self.SEDs)

    def toNoise(self, bandpass, PSF, wcs, rng=None):
        """Derive the `CorrelatedNoise` object for the associated `ChromaticSum` when convolved
        with ``PSF`` and drawn through ``bandpass`` onto pixels with specified ``wcs``.

        Parameters:
            bandpass:     `Bandpass` object representing filter image is drawn through.
            PSF:          output chromatic PSF to convolve by.
            wcs:          WCS of output pixel scale.
            rng:          Random number generator to forward to resulting CorrelatedNoise object.

        Returns:
            CorrelatedNoise object.
        """
        import numpy as np
        from .convolve import Convolve
        from .box import Pixel
        from .interpolatedimage import InterpolatedKImage
        NSED = len(self.SEDs)
        maxk = np.min([PSF.evaluateAtWavelength(bandpass.blue_limit).maxk,
                       PSF.evaluateAtWavelength(bandpass.red_limit).maxk])
        stepk = np.max([PSF.evaluateAtWavelength(bandpass.blue_limit).stepk,
                        PSF.evaluateAtWavelength(bandpass.red_limit).stepk])
        nk = 2*int(np.ceil(maxk/stepk))

        PSF_eff_kimgs = np.empty((NSED, nk, nk), dtype=np.complex128)
        for i, sed in enumerate(self.SEDs):
            # Assume that PSF does not yet include pixel contribution, so add it in.
            conv = Convolve(PSF, Pixel(wcs.scale, gsparams=PSF.gsparams)) * sed
            PSF_eff_kimgs[i] = conv.drawKImage(bandpass, nx=nk, ny=nk, scale=stepk).array
        pkout = np.zeros((nk, nk), dtype=np.float64)
        for i in range(NSED):
            for j in range(i, NSED):
                s = self.Sigma[(i, j)].drawKImage(nx=nk, ny=nk, scale=stepk).array
                pkout += (np.conj(PSF_eff_kimgs[i]) * s * PSF_eff_kimgs[j] *
                          (2 if i != j else 1)).real
        pk = Image(pkout + 0j, scale=stepk, dtype=complex)  # imag part should be zero
        iki = InterpolatedKImage(pk)
        iki *= wcs.pixelArea()**2  # determined this empirically
        return BaseCorrelatedNoise(rng, iki, wcs)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, CovarianceSpectrum) and
                 self.SEDs == other.SEDs and
                 self.Sigma == other.Sigma))
    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self):
        return hash(("galsim.CovarianceSpectrum", tuple(self.SEDs),
                     frozenset(self.Sigma.items())))

    def __repr__(self):
        return "galsim.CovarianceSpectrum(%r, %r)" % (self.Sigma, self.SEDs)

    def __str__(self):
        sigma_str = '{' + ', '.join([':'.join((str(k),str(v))) for k,v in self.Sigma.items()]) + '}'
        seds_str = '[' + ', '.join([str(s) for s in self.SEDs]) + ']'
        return "galsim.CovarianceSpectrum(%s, %s)" % (sigma_str, seds_str)
