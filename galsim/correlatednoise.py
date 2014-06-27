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
"""@file correlatednoise.py
Python layer documentation and functions for handling correlated noise in GalSim.
"""

import numpy as np
import galsim
from . import base
from . import utilities

class _BaseCorrelatedNoise(galsim.BaseNoise):
    """A Base Class describing 2D correlated Gaussian random noise fields.

    A _BaseCorrelatedNoise will not generally be instantiated directly.  This is recommended as the
    current `_BaseCorrelatedNoise.__init__` interface does not provide any guarantee that the input
    GSObject represents a physical correlation function, e.g. a profile that is an even function
    (two-fold rotationally symmetric in the plane) and peaked at the origin.  The proposed pattern
    is that users instead instantiate derived classes, such as the CorrelatedNoise, which are able
    to guarantee the above.

    The _BaseCorrelatedNoise is therefore here primarily to define the way in which derived classes
    (currently only CorrelatedNoise and UncorrelatedNoise) store the random deviate, noise
    correlation function profile and allow operations with it, generate images containing noise with
    these correlation properties, and generate covariance matrices according to the correlation
    function.
    """
    def __init__(self, rng, gsobject):

        if rng is not None and not isinstance(rng, galsim.BaseDeviate):
            raise TypeError(
                "Supplied rng argument not a galsim.BaseDeviate or derived class instance.")
        if not isinstance(gsobject, base.GSObject):
            raise TypeError(
                "Supplied gsobject argument not a galsim.GSObject or derived class instance.")

        # Initialize the BaseNoise with our input random deviate (which may be None).
        galsim.BaseNoise.__init__(self, rng)
        # Act as a container for the GSObject used to represent the correlation funcion.
        self._profile = gsobject

        # When applying normal or whitening noise to an image, we normally do calculations.
        # If _profile_for_stored is profile, then it means that we can use the stored values in
        # _rootps_store and/or _rootps_whitening_store and avoid having to redo the calculations.
        # So for now, we start out with _profile_for_stored = None and _rootps_store and
        # _rootps_whitening_store empty.
        self._profile_for_stored = None
        self._rootps_store = []
        self._rootps_whitening_store = []
        # Also set up the cache for a stored value of the variance, needed for efficiency once the
        # noise field can get convolved with other GSObjects making isAnalyticX() False
        self._variance_stored = None

    # Make "+" work in the intuitive sense (variances being additive, correlation functions add as
    # you would expect)
    def __add__(self, other):
        return _BaseCorrelatedNoise(self.getRNG(), self._profile + other._profile)

    def __sub__(self, other):
        return _BaseCorrelatedNoise(self.getRNG(), self._profile - other._profile)

    # NB: op* and op/ already work to adjust the overall variance of an object using the operator
    # methods defined in BaseNoise.

    def copy(self):
        """Returns a copy of the correlated noise model.

        The copy will share the BaseDeviate random number generator with the parent instance.
        Use the setRNG() method after copying if you wish to use a different random number
        sequence.
        """
        return _BaseCorrelatedNoise(self.getRNG(), self._profile.copy())

    def applyTo(self, image):
        """Apply this correlated Gaussian random noise field to an input Image.

        Calling
        -------
        To add deviates to every element of an image, the syntax

            >>> image.addNoise(correlated_noise)

        is preferred.  However, this is equivalent to calling this instance's applyTo() method as
        follows

            >>> correlated_noise.applyTo(image)

        On output the Image instance `image` will have been given additional noise according to the
        given CorrelatedNoise instance `correlated_noise`.  Normally, `image.scale` is used to
        determine the input Image pixel separation, and if `image.scale <= 0` a pixel scale of 1 is
        assumed.  If the image has a non-trivial WCS, it must at least be "uniform", i.e.,
        `image.wcs.isUniform() == True`.

        Note that the correlated noise field in `image` will be periodic across its boundaries: this
        is due to the fact that the internals of the CorrelatedNoise currently use a relatively
        simple implementation of noise generation using the Fast Fourier Transform.  If you wish to
        avoid this property being present in your final `image` you should applyTo() an `image` of
        greater extent than you need, and take a subset.

        @param image  The input Image object.
        """
        # Note that this uses the (fast) method of going via the power spectrum and FFTs to generate
        # noise according to the correlation function represented by this instance.  An alternative
        # would be to use the covariance matrices and eigendecomposition.  However, it is O(N^6)
        # operations for an NxN image!  FFT-based noise realization is O(2 N^2 log[N]) so we use it
        # for noise generation applications.

        # Check that the input has defined bounds
        if not isinstance(image, galsim.Image):
            raise TypeError("Input image argument must be a galsim.Image.")
        if not image.bounds.isDefined():
            raise ValueError("Input image argument must have defined bounds.")

        # If the profile has changed since last time (or if we have never been here before),
        # clear out the stored values.
        if self._profile_for_stored is not self._profile:
            self._rootps_store = []
            self._rootps_whitening_store = []
            self._variance_stored = None
        # Set profile_for_stored for next time.
        self._profile_for_stored = self._profile

        if image.wcs is not None and not image.wcs.isUniform():
            raise NotImplementedError("Sorry, correlated noise cannot be applied to an "+
                                      "image with a non-uniform WCS.")

        # Then retrieve or redraw the sqrt(power spectrum) needed for making the noise field
        rootps = self._get_update_rootps(image.array.shape, image.wcs)

        # Finally generate a random field in Fourier space with the right PS
        noise_array = _generate_noise_from_rootps(self.getRNG(), image.array.shape, rootps)

        # Add it to the image
        image += galsim.Image(noise_array, wcs=image.wcs)
        return image

    def applyToView(self, image_view):
        raise RuntimeError(
            "CorrelatedNoise can only be applied to a regular Image, not an ImageView")

    def applyWhiteningTo(self, image):
        """Apply noise designed to whiten correlated Gaussian random noise in an input Image.

        On output the Image instance `image` will have been given additional noise according to
        a specified CorrelatedNoise instance, designed to whiten any correlated noise that may have
        originally existed in `image`.

        Calling
        -------

            >>> correlated_noise.applyWhiteningTo(image)

        If the `image` originally contained noise with a correlation function described by the
        `correlated_noise` instance, the combined noise after using the applyWhiteningTo() method
        will be approximately uncorrelated.  Tests using COSMOS noise fields suggest ~0.3% residual
        off-diagonal covariances after whitening, relative to the variance, although results may
        vary depending on the precise correlation function of the noise field.
        (See `devel/external/hst/compare_whitening_subtraction.py` for the COSMOS tests.)

        Note that the code doesn't check that the "if" above is true: the user MUST make sure this
        is the case for the final noise to be uncorrelated.

        Normally, `image.scale` is used to determine the input Image pixel separation, and if
        `image.wcs` is None, a pixel scale of 1 is assumed.  If the image has a non-trivial WCS, it
        must at least be "uniform", i.e., `image.wcs.isUniform() == True`.

        If you are interested in a theoretical calculation of the variance in the final noise field
        after whitening, the applyWhiteningTo() method in fact returns this variance.  For example:

            >>> variance = correlated_noise.applyWhiteningTo(image)

        Example
        -------
        To see noise whitening in action, let us use a model of the correlated noise in COSMOS
        as returned by the getCOSMOSNoise() function.  Let's initialize and add noise to an image:

            >>> cosmos_file='YOUR/REPO/PATH/GalSim/examples/data/acs_I_unrot_sci_20_cf.fits'
            >>> cn = galsim.getCOSMOSNoise(cosmos_file)
            >>> image = galsim.ImageD(256, 256, scale=0.03)
                  # The scale should match the COSMOS default since didn't specify another
            >>> image.addNoise(cn)

        The `image` will then contain a realization of a random noise field with COSMOS-like
        correlation.  Using the applyWhiteningTo() method, we can now add more noise to `image`
        with a power spectrum specifically designed to make the combined noise fields uncorrelated:

            >>> cn.applyWhiteningTo(image)

        Of course, this whitening comes at the cost of adding further noise to the image, but
        the algorithm is designed to make this additional noise (nearly) as small as possible.

        @param image The input Image object.

        @returns the theoretically calculated variance of the combined noise fields in the
                 updated image.
        """
        # Note that this uses the (fast) method of going via the power spectrum and FFTs to generate
        # noise according to the correlation function represented by this instance.  An alternative
        # would be to use the covariance matrices and eigendecomposition.  However, it is O(N^6)
        # operations for an NxN image!  FFT-based noise realization is O(2 N^2 log[N]) so we use it
        # for noise generation applications.

        # Check that the input has defined bounds
        if not isinstance(image, galsim.Image):
            raise TypeError("Input image not a galsim.Image object")
        if not image.bounds.isDefined():
            raise ValueError("Input image argument must have defined bounds.")

        # If the profile has changed since last time (or if we have never been here before),
        # clear out the stored values.
        if self._profile_for_stored is not self._profile:
            self._rootps_store = []
            self._rootps_whitening_store = []
            self._variance_stored = None
        # Set profile_for_stored for next time.
        self._profile_for_stored = self._profile

        # Then retrieve or redraw the sqrt(power spectrum) needed for making the whitening noise,
        # and the total variance of the combination
        rootps_whitening, variance = self._get_update_rootps_whitening(image.array.shape, image.wcs)

        # Finally generate a random field in Fourier space with the right PS and add to image
        noise_array = _generate_noise_from_rootps(
            self.getRNG(), image.array.shape, rootps_whitening)
        image += galsim.Image(noise_array)

        # Return the variance to the interested user
        return variance

    def expand(self, scale):
        """Scale the linear scale of correlations in this noise model by `scale`.

        Scales the linear dimensions of the image by the factor `scale`, e.g.
        `half_light_radius` <-- `half_light_radius * scale`.

        @param scale    The linear rescaling factor to apply.

        @returns a new CorrelatedNoise object with the specified expansion.
        """
        return _BaseCorrelatedNoise(self.getRNG(), self._profile.expand(scale))

    def createExpanded(self, scale):
        """This is an obsolete synonym for expand(scale)"""
        return self.expand(scale)

    def applyExpansion(self, scale):
        """This is an obsolete method that is roughly equivalent to obj = obj.expand(scale)"""
        new_obj = self.expand(scale)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def dilate(self, scale):
        """Apply the appropriate changes to the scale and variance for when the object has
        an applied dilation.

        @param scale    The linear dilation scale factor.

        @returns a new CorrelatedNoise object with the specified dilation.
        """
        # Expansion changes the flux by scale**2, dilate reverses that to conserve flux,
        # so the variance needs to change by scale**-4.
        return _BaseCorrelatedNoise(self.getRNG(), self._profile.expand(scale) * (1./scale**4))

    def createDilated(self, scale):
        """This is an obsolete synonym for dilate(scale)"""
        return self.dilate(scale)

    def applyDilation(self, scale):
        """This is an obsolete method that is roughly equivalent to obj = obj.dilate(scale)"""
        new_obj = self.dilate(scale)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def magnify(self, mu):
        """Apply the appropriate changes to the scale and variance for when the object has
        an applied magnification `mu`.

        @param mu       The lensing magnification

        @returns a new CorrelatedNoise object with the specified magnification.
        """
        return _BaseCorrelatedNoise(self.getRNG(), self._profile.magnify(mu))

    def createMagnified(self, mu):
        """This is an obsolete synonym for magnify(mu)"""
        return self.magnify(mu)

    def applyMagnification(self, mu):
        """This is an obsolete method that is roughly equivalent to obj = obj.magnify(mu)"""
        new_obj = self.magnify(mu)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def lens(self, g1, g2, mu):
        """Apply the appropriate changes for when the object has an applied shear and magnification.

        @param g1       First component of lensing (reduced) shear to apply to the object.
        @param g2       Second component of lensing (reduced) shear to apply to the object.
        @param mu       Lensing magnification to apply to the object.

        @returns a new CorrelatedNoise object with the specified shear and magnification.
        """
        return _BaseCorrelatedNoise(self.getRNG(), self._profile.lens(g1,g2,mu))

    def createLensed(self, g1, g2, mu):
        """This is an obsolete synonym for lens(g1,g2,mu)"""
        return self.lens(g1,g2,mu)

    def applyLensing(self, g1, g2, mu):
        """This is an obsolete method that is roughly equivalent to obj = obj.lens(g1,g2,mu)"""
        new_obj = self.lens(g1,g2,mu)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def rotate(self, theta):
        """Apply a rotation `theta` to this correlated noise model.

        @param theta    Rotation angle (Angle object, positive means anticlockwise).

        @returns a new CorrelatedNoise object with the specified rotation.
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        return _BaseCorrelatedNoise(self.getRNG(), self._profile.rotate(theta))

    def createRotated(self, theta):
        """This is an obsolete synonym for rotate(theta)"""
        return self.rotate(theta)

    def applyRotation(self, theta):
        """This is an obsolete method that is roughly equivalent to obj = obj.rotate(theta)"""
        new_obj = self.rotate(theta)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def shear(self, *args, **kwargs):
        """Apply a shear to this correlated noise model, where arguments are either a Shear,
        or arguments that will be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for Shear
        (for doxygen documentation, see galsim.shear.Shear).

        @param shear    The shear to be applied. Or, as described above, you may instead supply
                        parameters do construct a shear directly.  eg. `corr.shear(g1=g1,g2=g2)`.

        @returns a new CorrelatedNoise object with the specified shear.
        """
        return _BaseCorrelatedNoise(self.getRNG(), self._profile.shear(*args,**kwargs))

    def createSheared(self, *args, **kwargs):
        """This is an obsolete synonym for shear(shear)"""
        return self.shear(*args,**kwargs)

    def applyShear(self, *args, **kwargs):
        """This is an obsolete method that is roughly equivalent to obj = obj.shear(shear)"""
        new_obj = self.shear(*args, **kwargs)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def transform(self, dudx, dudy, dvdx, dvdy):
        """Apply an arbitrary jacobian transformation to this correlated noise model.

        @param dudx     du/dx, where (x,y) are the current coords, and (u,v) are the new coords.
        @param dudy     du/dy, where (x,y) are the current coords, and (u,v) are the new coords.
        @param dvdx     dv/dx, where (x,y) are the current coords, and (u,v) are the new coords.
        @param dvdy     dv/dy, where (x,y) are the current coords, and (u,v) are the new coords.

        @returns a new CorrelatedNoise object with the specified transformation.
        """
        return _BaseCorrelatedNoise(self.getRNG(), self._profile.transform(dudx,dudy,dvdx,dvdy))

    def createTransformed(self, dudx, dudy, dvdx, dvdy):
        """This is an obsolete synonym for transform(dudx,dudy,dvdx,dvdy)"""
        return self.transform(dudx,dudy,dvdx,dvdy)

    def applyTransformation(self, dudx, dudy, dvdx, dvdy):
        """This is an obsolete method that is roughly equivalent to obj = obj.transform(...)"""
        new_obj = self.transform(dudx,dudy,dvdx,dvdy)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def getVariance(self):
        """Return the point variance of this noise field, equal to its correlation function value at
        zero distance.

        This is the variance of values in an image filled with noise according to this model.
        """
        # Test whether we can simply return the zero-lag correlation function value, which gives the
        # variance of an image of noise generated according to this model
        if self._profile.isAnalyticX():
            variance = self._profile.xValue(galsim.PositionD(0., 0.))
        else:
            # If the profile has changed since last time (or if we have never been here before),
            # clear out the stored values.
            if self._profile_for_stored is not self._profile:
                self._rootps_store = []
                self._rootps_whitening_store = []
                self._variance_stored = None
            # Set profile_for_stored for next time.
            self._profile_for_stored = self._profile
            # Then use cached version or rebuild if necessary
            if self._variance_stored is not None:
                variance = self._variance_stored
            else:
                imtmp = galsim.ImageD(1, 1)
                # GalSim internals handle this correctly w/out folding
                self.drawImage(imtmp, scale=1.)
                variance = imtmp.at(1, 1)
                self._variance_stored = variance # Store variance for next time
        return variance

    def withVariance(self, variance):
        """Set the point variance of the noise field, equal to its correlation function value at
        zero distance, to an input `variance`.  The rest of the correlated noise field is scaled
        proportionally.

        @param variance     The desired point variance in the noise.

        @returns a CorrelatedNoise object with the new variance.
        """
        variance_ratio = variance / self.getVariance()
        return self * variance_ratio

    def withScaledVariance(self, variance_ratio):
        """Scale the entire correlated noise field by the given factor.

        This is equivalent to cn * variance_ratio.

        @param variance_ratio   The factor by which to scale the variance of the correlation
                                function profile.

        @returns a CorrelatedNoise object whose variance and covariances have been scaled up by
                 the given factor.
        """
        return _BaseCorrelatedNoise(self.getRNG(), self._profile * variance_ratio)

    def setVariance(self, variance):
        """This is an obsolete method that is roughly equivalent to
        corr = corr.withVariance(variance)
        """
        new_obj = self.withVariance(variance)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def scaleVariance(self, variance_ratio):
        """This is an obsolete method that is roughly equivalent to corr = corr * variance_ratio"""
        new_obj = self.withScaledVariance(variance_ratio)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def convolvedWith(self, gsobject, gsparams=None):
        """Convolve the correlated noise model with an input GSObject.

        The resulting correlated noise model will then give a statistical description of the noise
        field that would result from convolving noise generated according to the initial correlated
        noise with a kernel represented by `gsobject` (e.g. a PSF).

        The practical purpose of this method is that it allows us to model what is happening to
        noise in the images from Hubble Space Telescope that we use for simulating PSF convolved
        galaxies with the RealGalaxy class.

        This modifies the representation of the correlation function, but leaves the random number
        generator unchanged.

        Examples
        --------
        The following command simply applies a Moffat PSF with slope parameter beta=3. and
        FWHM=0.7:

            >>> cn = cn.convolvedWith(galsim.Moffat(beta=3., fwhm=0.7))

        Often we will want to convolve with more than one function.  For example, if we wanted to
        simulate how a noise field would look if convolved with a ground-based PSF (such as the
        Moffat above) and then rendered onto a new (typically larger) pixel grid, the following
        example command demonstrates the syntax:

            >>> cn = cn.convolvedWith(
            ...    galsim.Convolve([galsim.Deconvolve(galsim.Pixel(0.03)),
            ...                     galsim.Pixel(0.2), galsim.Moffat(3., fwhm=0.7),

        Note, we also deconvolve by the original pixel, which should be the pixel size of the
        image from which the `correlated_noise` was made.  This command above is functionally
        equivalent to

            >>> cn = cn.convolvedWith(galsim.Deconvolve(galsim.Pixel(0.03)))
            >>> cn = cn.convolvedWith(galsim.Pixel(0.2))
            >>> cn = cn.convolvedWith(galsim.Moffat(beta=3., fwhm=0.7))

        as is demanded for a linear operation such as convolution.

        @param gsobject     A GSObject or derived class instance representing the function
                            with which the user wants to convolve the correlated noise model.
        @param gsparams     An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

        @returns the new CorrelatedNoise of the convolved profile.
        """
        conv = galsim.Convolve([self._profile, galsim.AutoCorrelate(gsobject)], gsparams=gsparams)
        return _BaseCorrelatedNoise(self.getRNG(), conv)

    def convolveWith(self, gsobject, gsparams=None):
        """This is an obsolete method that is roughly equivalent to
        cn = cn.convolvedWith(gsobject,gsparams)
        """
        new_obj = self.convolvedWith(gsobject,gsparams)
        self._profile = new_obj._profile
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
        self.__class__ = new_obj.__class__

    def drawImage(self, image=None, scale=None, dtype=None, wmult=1., add_to_image=False):
        """A method for drawing profiles storing correlation functions.

        This is a mild reimplementation of the drawImage() method for GSObjects.  The `method` is
        automatically set to 'sb' and cannot be changed, and the `gain` is set to unity.
        Also, not all the normal parameters of the GSObject method are available.

        @param image        If provided, this will be the image on which to draw the profile.
                            If `image = None`, then an automatically-sized Image will be created.
                            If `image != None`, but its bounds are undefined (e.g. if it was
                            constructed with `image = galsim.Image()`), then it will be resized
                            appropriately based on the profile's size [default: None].
        @param scale        If provided, use this as the pixel scale for the image.
                            If `scale` is `None` and `image != None`, then take the provided
                            image's pixel scale.
                            If `scale` is `None` and `image == None`, then use the Nyquist scale.
                            If `scale <= 0` (regardless of `image`), then use the Nyquist scale.
                            [default: None]
        @param dtype        The data type to use for an automatically constructed image.  Only
                            valid if `image = None`. [default: None, which means to use
                            numpy.float32]
        @param wmult        A multiplicative factor by which to enlarge (in each direction) the
                            default automatically calculated FFT grid size used for any
                            intermediate calculations in Fourier space.  See the description
                            in GSObject.drawImage() for more details. [default: 1]
        @param add_to_image Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing.
                            Note: This requires that `image` be provided and that it have defined
                            bounds. [default: False]

        @returns an Image of the correlation function.
        """
        return self._profile.drawImage(
            image=image, scale=scale, dtype=dtype, method='sb', gain=1., wmult=wmult,
            add_to_image=add_to_image, use_true_center=False)

    def draw(self, image=None, scale=None, dtype=None, wmult=1., add_to_image=False):
        """An obsolete synonym of drawImage"""
        return self.drawImage(image,scale,dtype,wmult,add_to_image)

    def calculateCovarianceMatrix(self, bounds, scale):
        """Calculate the covariance matrix for an image with specified properties.

        A correlation function also specifies a covariance matrix for noise in an image of known
        dimensions and pixel scale.  The user specifies these bounds and pixel scale, and this
        method returns a covariance matrix as a square Image object, with the upper triangle
        containing the covariance values.

        @param  bounds Bounds corresponding to the dimensions of the image for which a covariance
                       matrix is required.
        @param  scale  Pixel scale of the image for which a covariance matrix is required.

        @returns the covariance matrix (as an Image).
        """
        # TODO: Allow this to take a JacobianWCS, rather than just a scale.
        return galsim._galsim._calculateCovarianceMatrix(self._profile.SBProfile, bounds, scale)

    def _get_update_rootps(self, shape, wcs):
        """Internal utility function for querying the `rootps` cache, used by applyTo() and
        applyWhiteningTo() methods.
        """
        # First check whether we can just use a stored power spectrum (no drawing necessary if so)
        use_stored = False
        # Query using the rfft2/irfft2 half-sized shape (shape[0], shape[1] // 2 + 1)
        half_shape = (shape[0], shape[1] // 2 + 1)
        for rootps_array, saved_wcs in self._rootps_store:
            if rootps_array.shape == half_shape:
                if ( (wcs is None and saved_wcs.isPixelScale() and saved_wcs.scale == 1.) or
                     wcs == saved_wcs ):
                    use_stored = True
                    rootps = rootps_array
                    break

        # If not, draw the correlation function to the desired size and resolution, then DFT to
        # generate the required array of the square root of the power spectrum
        if use_stored is False:
            newcf = galsim.ImageD(shape[1], shape[0]) # set the corr func to be the correct size
            # set the wcs...
            if wcs is None:
                newcf.scale = 1.
            else:
                newcf.wcs = wcs
            # Then draw this correlation function into an array.
            newcf = self.drawImage(newcf)

            # Since we just drew it, save the variance value for posterity.
            var = newcf(newcf.bounds.center())
            self._variance_stored = var

            if var <= 0.:
                raise RuntimeError("CorrelatedNoise found to have negative variance.")

            # Then calculate the sqrt(PS) that will be used to generate the actual noise
            ps = np.fft.rfft2(newcf.array)
            rootps = np.sqrt(np.abs(ps)) # The PS will often be purely real, but sometimes only
                                         # close (due to the approximations of interpolating), so
                                         # abs() is necessary... This approx. means that correlated
                                         # noise should always be tested for any given application

            # Then add this and the relevant wcs to the _rootps_store for later use
            self._rootps_store.append((rootps, newcf.wcs))

        return rootps

    def _get_update_rootps_whitening(self, shape, wcs, headroom=1.05):
        """Internal utility function for querying the `rootps_whitening` cache, used by the
        applyWhiteningTo() method, and calculate and update it if not present.

        @returns rootps_whitening, variance
        """
        # First check whether we can just use a stored whitening power spectrum
        use_stored = False
        # Query using the rfft2/irfft2 half-sized shape (shape[0], shape[1] // 2 + 1)
        half_shape = (shape[0], shape[1] // 2 + 1)
        for rootps_whitening_array, saved_wcs, var in self._rootps_whitening_store:
            if rootps_whitening_array.shape == half_shape:
                if ( (wcs is None and saved_wcs.isPixelScale() and saved_wcs.scale == 1.) or
                     wcs == saved_wcs ):
                    use_stored = True
                    rootps_whitening = rootps_whitening_array
                    variance = var
                    break

        # If not, calculate the whitening power spectrum as (almost) the smallest power spectrum
        # that when added to rootps**2 gives a flat resultant power that is nowhere negative.
        # Note that rootps = sqrt(power spectrum), and this procedure therefore works since power
        # spectra add (rather like variances).  The resulting power spectrum will be all positive
        # (and thus physical).
        if use_stored is False:

            rootps = self._get_update_rootps(shape, wcs)
            ps_whitening = -rootps * rootps
            ps_whitening += np.abs(np.min(ps_whitening)) * headroom # Headroom adds a little extra
            rootps_whitening = np.sqrt(ps_whitening)                # variance, for "safety"

            # Finally calculate the theoretical combined variance to output alongside the image
            # to be generated with the rootps_whitening.  Note that although we use the [0, 0]
            # element we could use any as the PS should be flat
            variance = rootps[0, 0]**2 + ps_whitening[0, 0]

            # Then add all this and the relevant wcs to the _rootps_whitening_store
            self._rootps_whitening_store.append((rootps_whitening, wcs, variance))

        return rootps_whitening, variance

###
# Now a standalone utility function for generating noise according to an input (square rooted)
# Power Spectrum
#
def _generate_noise_from_rootps(rng, shape, rootps):
    """Utility function for generating a NumPy array containing a Gaussian random noise field with
    a user-specified power spectrum also supplied as a NumPy array.

    @param rng      BaseDeviate instance to provide the random number generation
    @param shape    Shape of the output array, needed because of the use of Hermitian symmetry to
                    increase inverse FFT efficiency using the `np.fft.irfft2` function (gets sent to
                    the kwarg `s=` of `np.fft.irfft2`)
    @param rootps   NumPy array containing the square root of the discrete Power Spectrum ordered
                    in two dimensions according to the usual DFT pattern for `np.fft.rfft2` output
                    (see also `np.fft.fftfreq`)

    @returns a NumPy array (contiguous) of the requested shape, filled with the noise field.
    """
    # Sanity check on requested shape versus that of rootps
    if len(shape) != 2 or (shape[0], shape[1] // 2 + 1) != rootps.shape:
        raise ValueError("Requested shape does not match that of the supplied rootps")
    
    # Make half size Images using Hermitian symmetry to get full sized real inverse FFT
    gaussvec_real = galsim.ImageD(shape[1] // 2 + 1, shape[0]) # Remember NumPy is [y, x]
    gaussvec_imag = galsim.ImageD(shape[1] // 2 + 1, shape[0])
    #  Quickest to create Gaussian rng each time needed, so do that here...
    gn = galsim.GaussianNoise(
        rng=rng, sigma=np.sqrt(.5 * shape[0] * shape[1])) # Note sigma scaling: 1/sqrt(2) needed so
                                                          # <|gaussvec|**2> = product(shape); shape
                                                          # needed because of the asymmetry in the
                                                          # 1/N^2 division in the NumPy FFT/iFFT
    gaussvec_real.addNoise(gn)
    gaussvec_imag.addNoise(gn)
    noise_array = np.fft.irfft2((gaussvec_real.array + 1j * gaussvec_imag.array) * rootps, s=shape)
    return np.ascontiguousarray(noise_array)


###
# Then we define the CorrelatedNoise, which generates a correlation function by estimating it
# directly from images:
#
class CorrelatedNoise(_BaseCorrelatedNoise):
    """A class that represents 2D correlated noise fields calculated from an input Image.

    This class stores an internal representation of a 2D, discrete correlation function, and allows
    a number of subsequent operations including interpolation, shearing, magnification and
    rendering of the correlation function profile into an output Image.  The class also allows
    correlated Gaussian noise fields to be generated according to the correlation function, and
    added to an Image: see the applyTo() method.  It also provides a method for whitening
    pre-existing noise that shares the same spatial correlations: see the applyWhiteningTo()
    method.

    It also allows the combination of multiple correlation functions by addition, and for the
    scaling of the total variance they represent by scalar factors.

    Initialization
    --------------

    Basic example:

        >>> cn = galsim.CorrelatedNoise(image, rng=rng)

    Instantiates a CorrelatedNoise using the pixel scale information contained in `image.scale`
    (assumes the scale is unity if `image.scale <= 0.`) by calculating the correlation function
    in the input `image`.  The input `rng` must be a BaseDeviate or derived class instance,
    setting the random number generation for the noise.

    Optional Inputs: Interpolant
    ----------------------------

        >>> cn = galsim.CorrelatedNoise(image, rng=rng, scale=0.2)

    The example above instantiates a CorrelatedNoise, but forces the use of the pixel scale
    `scale` to set the units of the internal lookup table.

        >>> cn = galsim.CorrelatedNoise(image, rng=rng,
        ...     x_interpolant=galsim.InterpolantXY(galsim.Lanczos(5, tol=1.e-4))

    The example above instantiates a CorrelatedNoise, but forces use of a non-default interpolant
    for interpolation of the internal lookup table in real space.  Must be an InterpolantXY instance
    or an Interpolant instance (if the latter one-dimensional case is supplied, an InterpolantXY
    will be automatically generated from it).

    The default `x_interpolant` is `galsim.InterpolantXY(galsim.Linear(tol=1.e-4))`, which uses 
    bilinear interpolation.  The use of this interpolant is an approximation that gives good 
    empirical results without requiring internal convolution of the correlation function profile by
    a Pixel object when applying correlated noise to images: such an internal convolution has been
    found to be computationally costly in practice, requiring the Fourier transform of very large
    arrays.

    The use of the bilinear interpolants means that the representation of correlated noise will be
    noticeably inaccurate in at least the following two regimes:

      i)  If the pixel scale of the desired final output (e.g. the target image of drawImage(),
          applyTo() or applyWhiteningTo()) is small relative to the separation between pixels
          in the `image` used to instantiate `cn` as shown above.
      ii) If the CorrelatedNoise instance `cn` was instantiated with an image of scale comparable
          to that in the final output, and `cn` has been rotated or otherwise transformed (e.g.
          via the rotate(), shear() methods, see below).

    Conversely, the approximation will work best in the case where the correlated noise used to
    instantiate the `cn` is taken from an input image for which `image.scale` is smaller than that
    in the desired output.  This is the most common use case in the practical treatment of
    correlated noise when simulating galaxies from space telescopes, such as COSMOS.

    Changing from the default bilinear interpolant is made possible, but not recommended.  For more
    information please see the discussion on https://github.com/GalSim-developers/GalSim/pull/452.

    Optional Inputs: Periodicity Correction
    ---------------------------------------

    There is also an option to switch off an internal correction for assumptions made about the
    periodicity in the input noise image.  If you wish to turn this off you may, e.g.

        >>> cn = galsim.CorrelatedNoise(image, rng=rng, correct_periodicity=False)

    The default and generally recommended setting is `correct_periodicity=True`.

    Users should note that the internal calculation of the discrete correlation function in `image`
    will assume that `image` is periodic across its boundaries, introducing a dilution bias in the
    estimate of inter-pixel correlations that increases with separation.  Unless you know that the
    noise in `image` is indeed periodic (perhaps because you generated it to be so), you will not
    generally wish to use the `correct_periodicity=False` option.

    By default, the image is not mean subtracted before the correlation function is estimated.  To
    do an internal mean subtraction, you can set the `subtract_mean` keyword to `True`, e.g.

        >>> cn = galsim.CorrelatedNoise(image, rng=rng, subtract_mean=True)

    Using the `subtract_mean` option will introduce a small underestimation of variance and other
    correlation function values due to a bias on the square of the sample mean.  This bias reduces
    as the input image becomes larger, and in the limit of uncorrelated noise tends to the constant
    term `variance/N**2` for an `N`x`N` sized `image`.

    It is therefore recommended that a background/sky subtraction is applied to the `image` before
    it is given as an input to the `CorrelatedNoise`, allowing the default `subtract_mean=False`.
    If such a background model is global or based on large regions on sky then assuming that the
    image has a zero population mean will be reasonable, and won't introduce a bias in covariances
    from an imperfectly-estimated sample mean subtraction.  If this is not possible, just be aware
    that `subtract_mean=True` will bias the correlation function low to some level.

    You may also specify a gsparams argument.  See the docstring for GSParams using
    help(galsim.GSParams) for more information about this option.

    Methods and Use
    ---------------
    The main way that a CorrelatedNoise is used is to add or assign correlated noise to an image.
    This is common to all the classes that inherit from BaseNoise: to add deviates to every element
    of an image, the syntax

        >>> im.addNoise(cn)

    is preferred, although

        >>> cn.applyTo(im)

    is equivalent.  See the addNoise() method docstring for more information.  The `image.scale`
    is used to get the pixel scale of the input image unless this is <= 0, in which case a scale
    of 1 is assumed.

    Another method that may be of use is

        >>> m = cn.calculateCovarianceMatrix(im.bounds, scale)

    which can be used to generate a covariance matrix based on a user input image geometry.  See
    the calculateCovarianceMatrix() method docstring for more information.

    A number of methods familiar from GSObject instances have also been implemented directly as
    `cn` methods, so that the following commands are all legal:

        >>> image = cn.drawImage(im, scale, wmult=4)
        >>> cn = cn.shear(s)
        >>> cn = cn.expand(m)
        >>> cn = cn.rotate(theta * galsim.degrees)
        >>> cn = cn.transform(dudx, dudy, dvdx, dvdy)

    See the individual method docstrings for more details.  The shift() method is not available
    since a correlation function must always be centred and peaked at the origin.

    The BaseNoise methods

        >>> var = cn.getVariance()
        >>> cn1 = cn.withVariance(variance)
        >>> cn2 = cn.withScaledVariance(variance_ratio)

    can be used to get and set the point variance of the correlated noise, equivalent to the zero
    separation distance correlation function value.  The withVariance() method scales the whole
    internal correlation function so that its point variance matches `variance`.  Similarly,
    `withScaledVariance` scales the entire function by the given factor.

    Arithmetic Operators
    --------------------
    Addition, multiplication and division operators are defined to work in an intuitive way for
    correlation functions.

    Addition works simply to add the internally-stored correlation functions, so that

        >>> cn3 = cn2 + cn1
        >>> cn4 += cn5

    provides a representation of the correlation function of two linearly summed fields represented
    by the individual correlation function operands.

    What happens to the internally stored random number generators in the examples above?  For all
    addition operations it is the BaseDeviate belonging to the instance on the Left-Hand Side
    of the operator that is retained.

    In the example above therefore, it is the random number generator from `cn2` that will be stored
    and used by `cn3`, and `cn4` will retain its random number generator after in-place addition of
    `cn5`.  The random number generator of `cn5` is not affected by the operation.

    The multiplication and division operators, e.g.

        >>> cn1 /= 3.
        >>> cn2 = cn1 * 3

    scale the overall correlation function by a scalar operand.  The random number generators are
    not affected by these scaling operations.
    """
    def __init__(self, image, rng=None, scale=0., x_interpolant=None, correct_periodicity=True,
        subtract_mean=False, gsparams=None, dx=None):
        # Check for obsolete dx parameter
        if dx is not None and scale==0.: scale = dx

        # Check that the input image is in fact a galsim.ImageSIFD class instance
        if not isinstance(image, galsim.Image):
            raise TypeError("Input image not a galsim.Image object")
        # Build a noise correlation function (CF) from the input image, using DFTs
        # Calculate the power spectrum then a (preliminary) CF
        ft_array = np.fft.rfft2(image.array)
        ps_array = np.abs(ft_array)**2 # Using timeit abs() seems to have the slight speed edge

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
            cf_array_prelim, (cf_array_prelim.shape[0] / 2, cf_array_prelim.shape[1] / 2))

        # The underlying C++ object is expecting the CF to be represented by an odd-dimensioned
        # array with the central pixel denoting the zero-distance correlation (variance), even
        # even if the input image was even-dimensioned on one or both sides.
        # We therefore copy-paste and zero pad the CF calculated above to ensure that these
        # expectations are met.
        #
        # Determine the largest dimension of the input image, and use it to generate an empty CF
        # array for final output, padding by one to make odd if necessary:
        cf_array = np.zeros((
            1 + 2 * (cf_array_prelim.shape[0] / 2),
            1 + 2 * (cf_array_prelim.shape[1] / 2))) # using integer division

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
        cf_image = galsim.Image(np.ascontiguousarray(cf_array))

        # Correctly record the original image scale if set
        if scale > 0.:
            cf_image.scale = scale
        elif image.scale > 0.:
            cf_image.scale = image.scale
        else: # sometimes Images are instantiated with scale=0, in which case we will assume unit
              # pixel scale
            cf_image.scale = 1.

        # If x_interpolant not specified on input, use bilinear
        if x_interpolant == None:
            linear = galsim.Linear(tol=1.e-4)
            x_interpolant = galsim.InterpolantXY(linear)
        else:
            x_interpolant = utilities.convert_interpolant_to_2d(x_interpolant)

        # Then initialize...
        cf_object = galsim.InterpolatedImage(
            cf_image, x_interpolant=x_interpolant, scale=cf_image.scale, normalization="sb",
            calculate_stepk=False, calculate_maxk=False, #<-these internal calculations do not seem
            gsparams=gsparams)                           #  to do very well with often sharp-peaked
                                                         #  correlation function images...
        _BaseCorrelatedNoise.__init__(self, rng, cf_object)

        if store_rootps:
            # If it corresponds to the CF above, store useful data as a (rootps, scale) tuple for
            # efficient later use:
            self._profile_for_stored = self._profile
            self._rootps_store.append((np.sqrt(ps_array), cf_image.scale))


def _cf_periodicity_dilution_correction(cf_shape):
    """Return an array containing the correction factor required for wrongly assuming periodicity
    around noise field edges in an DFT estimate of the discrete correlation function.

    Uses the result calculated by MJ on GalSim Pull Request #366.
    See https://github.com/GalSim-developers/GalSim/pull/366.

    Returns a 2D NumPy array with the same shape as the input parameter tuple `cf_shape`.  This
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


# Make a function for returning Noise correlations
def _Image_getCorrelatedNoise(image):
    """Returns a CorrelatedNoise instance by calculating the correlation function of image pixels.
    This function is discouraged and will be deprecated in a future version.
    """
    return CorrelatedNoise(image)

# Then add this Image method to the Image class
galsim.Image.getCorrelatedNoise = _Image_getCorrelatedNoise

# Free function for returning a COSMOS noise field correlation function
def getCOSMOSNoise(file_name, rng=None, cosmos_scale=0.03, variance=0., x_interpolant=None,
                   gsparams=None, dx_cosmos=None):
    """Returns a representation of correlated noise in the HST COSMOS F814W unrotated science coadd
    images.

    See http://cosmos.astro.caltech.edu/astronomer/hst.html for information about the COSMOS survey,
    and Leauthaud et al (2007) for detailed information about the unrotated F814W coadds used for
    weak lensing science.

    This function uses a stacked estimate of the correlation function in COSMOS noise fields, the
    location of which should be input to this function via the `file_name` argument.  This image is
    stored in FITS format, and is generated as described in
    `YOUR/REPO/PATH/GalSim/devel/external/hst/make_cosmos_cfimage.py`.  The image itself can also be
    found within the GalSim repo, located at:

        /YOUR/REPO/PATH/GalSim/examples/data/acs_I_unrot_sci_20_cf.fits

    @param file_name    String containing the path and filename above but modified to match the
                        location of the GalSim repository on your system.
    @param rng          If provided, a random number generator to use as the random number
                        generator of the resulting noise object. (may be any kind of
                        BaseDeviate object) [default: None, in which case, one will be
                        automatically created, using the time as a seed.]
    @param cosmos_scale COSMOS ACS F814W coadd image pixel scale in the units you are using to
                        describe GSObjects and image scales in GalSim. [default: 0.03 (arcsec),
                        see below for more information.]
    @param variance     Scales the correlation function so that its point variance, equivalent
                        to its value at zero separation distance, matches this value.
                        [default: 0., which means to use the variance in the original COSMOS
                        noise fields.]
    @param x_interpolant  Forces use of a non-default interpolant for interpolation of the
                        internal lookup table in real space.  See below for more details.
                        [default: galsim.Linear(tol=1.e-4)]
    @param gsparams     An optional GSParams argument.  See the docstring for GSParams for
                        details. [default: None]

    @returns a _BaseCorrelatedNoise instance representing correlated noise in F814W COSMOS images.

    The default `x_interpolant` is a `galsim.InterpolantXY(galsim.Linear(tol=1.e-4))`, which uses
    bilinear interpolation.  The use of this interpolant is an approximation that gives good
    empirical results without requiring internal convolution of the correlation function profile by
    a Pixel object when applying correlated noise to images: such an internal convolution has
    been found to be computationally costly in practice, requiring the Fourier transform of very
    large arrays.

    The use of the bilinear interpolants means that the representation of correlated noise will be
    noticeably inaccurate in at least the following two regimes:

      i)  If the pixel scale of the desired final output (e.g. the target image of drawImage(),
          applyTo() or applyWhiteningTo()) is small relative to the separation between pixels
          in the `image` used to instantiate `cn` as shown above.
      ii) If the CorrelatedNoise instance `cn` was instantiated with an image of scale comparable
          to that in the final output, and `cn` has been rotated or otherwise transformed (e.g.
          via the rotate(), shear() methods, see below).

    Conversely, the approximation will work best in the case where the correlated noise used to
    instantiate the `cn` is taken from an input image for which `image.scale` is smaller than that
    in the desired output.  This is the most common use case in the practical treatment of
    correlated noise when simulating galaxies from COSMOS, for which this function is expressly
    designed.

    Changing from the default bilinear interpolant is made possible, but not recommended.  For more
    information please see the discussion on https://github.com/GalSim-developers/GalSim/pull/452.

    You may also specify a gsparams argument.  See the docstring for GSParams using
    help(galsim.GSParams) for more information about this option.

    Important note regarding units
    ------------------------------
    The ACS coadd images in COSMOS have a pixel scale of 0.03 arcsec, and so the pixel scale
    `cosmos_scale` adopted in the representation of of the correlation function takes a default
    value

        cosmos_scale = 0.03

    If you wish to use other units, ensure that the input keyword `cosmos_scale` takes the value
    corresponding to 0.03 arcsec in your chosen system.

    Example usage
    -------------
    The following commands use this function to generate a 300 pixel x 300 pixel image of noise with
    HST COSMOS correlation properties (substitute in your own file and path for the `filestring`).

        >>> file_name='/YOUR/REPO/PATH/GalSim/devel/external/hst/acs_I_unrot_sci_20_cf.fits'
        >>> import galsim
        >>> rng = galsim.UniformDeviate(123456)
        >>> cf = galsim.correlatednoise.getCOSMOSNoise(file_name, rng=rng)
        >>> im = galsim.ImageD(300, 300, scale=0.03)
        >>> cf.applyTo(im)
        >>> im.write('out.fits')

    The FITS file `out.fits` should then contain an image of randomly-generated, COSMOS-like noise.
    """
    # Check for obsolete dx_cosmos parameter
    if dx_cosmos is not None and cosmos_scale==0.03: cosmos_scale = dx_cosmos

    # First try to read in the image of the COSMOS correlation function stored in the repository
    import os
    if not os.path.isfile(file_name):
        raise IOError("The input file_name '"+str(file_name)+"' does not exist.")
    try:
        cfimage = galsim.fits.read(file_name)
    except Exception:
        # Give a vaguely helpful warning, then raise the original exception for extra diagnostics
        import warnings
        warnings.warn(
            "Function getCOSMOSNoise() unable to read FITS image from "+str(file_name)+", "+
            "more information on the error in the following Exception...")
        raise

    # Then check for negative variance before doing anything time consuming
    if variance < 0:
        raise ValueError("Input keyword variance must be zero or positive.")

    # If x_interpolant not specified on input, use bilinear
    if x_interpolant == None:
        linear = galsim.Linear(tol=1.e-4)
        x_interpolant = galsim.InterpolantXY(linear)
    else:
        x_interpolant = utilities.convert_interpolant_to_2d(x_interpolant)

    # Use this info to then generate a correlated noise model DIRECTLY: note this is non-standard
    # usage, but tolerated since we can be sure that the input cfimage is appropriately symmetric
    # and peaked at the origin
    ii = galsim.InterpolatedImage(cfimage, scale=cosmos_scale, normalization="sb",
                                  calculate_stepk=False, calculate_maxk=False,
                                  x_interpolant=x_interpolant, gsparams=gsparams)
    ret = _BaseCorrelatedNoise(rng, ii)
    # If the input keyword variance is non-zero, scale the correlation function to have this
    # variance
    if variance > 0.:
        ret = ret.withVariance(variance)
    return ret

class UncorrelatedNoise(_BaseCorrelatedNoise):
    """A class that represents 2D correlated noise fields that are actually (at least initially)
    uncorrelated.  Subsequent applications of things like shear or convolvedWith will induce
    correlations.

    Initialization
    --------------

    The noise is characterized by a variance in each image pixel and a pixel size and shape.
    The `variance` value refers to the noise variance in each pixel.  If the pixels are square
    (the usual case), you can specify the size using the `scale` parameter.  If not, they
    are effectively specified using the local wcs function that defines the pixel shape.  i.e.

        >>> world_pix = wcs.toWorld(Pixel(1.))`

    should return the pixel profile in world coordinates.

    @param variance     The noise variance value to model as being uniform and uncorrelated
                        over the whole image.
    @param rng          If provided, a random number generator to use as the random number
                        generator of the resulting noise object. (may be any kind of
                        BaseDeviate object) [default: None, in which case, one will be
                        automatically created, using the time as a seed.]
    @param scale        If provided, use this as the pixel scale.  [default: 1.0, unless `wcs` is
                        provided]
    @param wcs          If provided, use this as the wcs for the image.  At most one of `scale`
                        or `wcs` may be provided. [default: None]
    @param gsparams     An optional GSParams argument.  See the docstring for GSParams for
                        details. [default: None]


    Methods and Use
    ---------------

    This class is used in the same way as CorrelatedNoise.  See the documentation for that
    class for more details.
    """
    def __init__(self, variance, rng=None, scale=None, wcs=None, gsparams=None):
        if variance < 0:
            raise ValueError("Input keyword variance must be zero or positive.")

        if wcs is not None:
            if scale is not None:
                raise ValueError("Cannot provide both wcs and scale")
            if not isinstance(wcs, galsim.BaseWCS):
                raise TypeError("wcs must be a BaseWCS instance")
            if not wcs.isUniform():
                raise ValueError("Cannot provide non-local wcs")
        elif scale is not None:
            wcs = galsim.PixelScale(scale)
        else:
            wcs = galsim.PixelScale(1.0)

        # Need variance == xvalue(0,0)
        # Pixel has flux of f/scale^2, so use f = variance * scale^2
        image_pix = galsim.Pixel(scale=1.0, flux=variance * wcs.pixelArea(), gsparams=gsparams)
        world_pix = wcs.toWorld(image_pix)
        cf_object = galsim.AutoConvolve(world_pix)
        _BaseCorrelatedNoise.__init__(self, rng, cf_object)


