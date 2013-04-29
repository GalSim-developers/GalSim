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
    `GSObject` represents a physical correlation function, e.g. a profile that is an even function 
    (two-fold rotationally symmetric in the plane) and peaked at the origin.  The proposed pattern
    is that users instead instantiate derived classes, such as the CorrelatedNoise, which are able
    to guarantee the above.

    The _BaseCorrelatedNoise is therefore here primarily to define the way in which derived classes 
    (currently only the `CorrelatedNoise`) store the random deviate, noise correlation function
    profile and allow operations with it, generate images containing noise with these correlation
    properties, and generate covariance matrices according to the correlation function.
    """
    def __init__(self, rng, gsobject):
        
        if not isinstance(rng, galsim.BaseDeviate):
            raise TypeError(
                "Supplied rng argument not a galsim.BaseDeviate or derived class instance.")
        if not isinstance(gsobject, base.GSObject):
            raise TypeError(
                "Supplied gsobject argument not a galsim.GSObject or derived class instance.")

        # Initialize the GaussianNoise with our input random deviate/GaussianNoise
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
        ret = self.copy()
        ret += other
        return ret

    def __iadd__(self, other):
        self._profile += other._profile
        return _BaseCorrelatedNoise(self.getRNG(), self._profile)

    # Make op* and op*= work to adjust the overall variance of an object
    def __imul__(self, other):
        self._profile.scaleVariance(other)
        return self

    def __mul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    def __rmul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    # Likewise for op/ and op/=
    def __idiv__(self, other):
        self.scaleVariance(1. / other)
        return self

    def __div__(self, other):
        ret = self.copy()
        ret /= other
        return ret

    def __itruediv__(self, other):
        return __idiv__(self, other)

    def __truediv__(self, other):
        return __div__(self, other)

    def copy(self):
        """Returns a copy of the correlated noise model.

        The copy will share the galsim.BaseDeviate random number generator with the parent instance.
        Use the .setRNG() method after copying if you wish to use a different random number
        sequence.
        """
        return _BaseCorrelatedNoise(self.getRNG(), self._profile.copy())

    def applyTo(self, image):
        """Apply this correlated Gaussian random noise field to an input Image.

        Calling
        -------
        To add deviates to every element of an image, the syntax 

            >>> image.addNoise(correlated_noise)

        is preferred.  However, this is equivalent to calling this instance's .applyTo() method as
        follows

            >>> correlated_noise.applyTo(image)

        On output the Image instance `image` will have been given additional noise according to the
        given CorrelatedNoise instance `correlated_noise`.  image.getScale() is used to determine
        the input image pixel separation, and if image.getScale() <= 0 a pixel scale of 1 is
        assumed.

        Note that the correlated noise field in `image` will be periodic across its boundaries: this
        is due to the fact that the internals of the CorrelatedNoise currently use a relatively
        simple implementation of noise generation using the Fast Fourier Transform.  If you wish to
        avoid this property being present in your final `image` you should .applyTo() an `image` of
        greater extent than you need, and take a subset.

        @param image The input Image object.
        """
        # Note that this uses the (fast) method of going via the power spectrum and FFTs to generate
        # noise according to the correlation function represented by this instance.  An alternative
        # would be to use the covariance matrices and eigendecomposition.  However, it is O(N^6)
        # operations for an NxN image!  FFT-based noise realization is O(2 N^2 log[N]) so we use it
        # for noise generation applications.

        # Check that the input has defined bounds
        if not hasattr(image, "bounds"):
            raise ValueError(
                "Input image argument does not have a bounds attribute, it must be a galsim.Image "+
                "or galsim.ImageView-type object with defined bounds.")

        # If the profile has changed since last time (or if we have never been here before),
        # clear out the stored values.
        if self._profile_for_stored is not self._profile:
            self._rootps_store = []
            self._rootps_whitening_store = []
            self._variance_stored = None
        # Set profile_for_stored for next time.
        self._profile_for_stored = self._profile

        # Then retrieve or redraw the sqrt(power spectrum) needed for making the noise field
        rootps = self._get_update_rootps(image.array.shape, image.getScale())

        # Finally generate a random field in Fourier space with the right PS
        noise_array = _generate_noise_from_rootps(self.getRNG(), rootps)
        # Add it to the image
        image += galsim.ImageViewD(noise_array)
        return image

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

        image.getScale() is used to determine the input image pixel separation, and if 
        image.getScale() <= 0 a pixel scale of 1 is assumed.

        If you are interested in a theoretical calculation of the variance in the final noise field
        after whitening, the applyWhiteningTo() method in fact returns this variance.  For example:

            >>> variance = correlated_noise.applyWhiteningTo(image)

        Example
        -------
        To see noise whitening in action, let us use a model of the correlated noise in COSMOS 
        as returned by the getCOSMOSNoise() function.  Let's initialize and add noise to an image:

            >>> cosmos_file='YOUR/REPO/PATH/GalSim/examples/data/acs_I_unrot_sci_20_cf.fits'
            >>> cn = galsim.getCOSMOSNoise(galsim.BaseDeviate(), cosmos_file)
            >>> image = galsim.ImageD(256, 256)
            >>> image.setScale(0.03) # Should match the COSMOS default since didn't specify another
            >>> image.addNoise(cn)

        The `image` will then contain a realization of a random noise field with COSMOS-like
        correlation.  Using the applyWhiteningTo() method, we can now add more noise to `image`
        with a power spectrum specifically designed to make the combined noise fields uncorrelated:

            >>> cn.applyWhiteningTo(image)

        Of course, this whitening comes at the cost of adding further noise to the image, but 
        the algorithm is designed to make this additional noise (nearly) as small as possible.

        @param image The input Image object.

        @return variance  A float containing the theoretically calculated variance of the combined
                          noise fields in the updated image.
        """
        # Note that this uses the (fast) method of going via the power spectrum and FFTs to generate
        # noise according to the correlation function represented by this instance.  An alternative
        # would be to use the covariance matrices and eigendecomposition.  However, it is O(N^6)
        # operations for an NxN image!  FFT-based noise realization is O(2 N^2 log[N]) so we use it
        # for noise generation applications.

        # Check that the input has defined bounds
        if not hasattr(image, "bounds"):
            raise ValueError(
                "Input image argument does not have a bounds attribute, it must be a galsim.Image "+
                "or galsim.ImageView-type object with defined bounds.")

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
        rootps_whitening, variance = self._get_update_rootps_whitening(
            image.array.shape, image.getScale())

        # Finally generate a random field in Fourier space with the right PS and add to image
        noise_array = _generate_noise_from_rootps(self.getRNG(), rootps_whitening)
        image += galsim.ImageViewD(noise_array)

        # Return the variance to the interested user
        return variance

    def applyTransformation(self, ellipse):
        """Apply a galsim.Ellipse distortion to the correlated noise model.
           
        galsim.Ellipse objects can be initialized in a variety of ways (see documentation of this
        class, galsim.ellipse.Ellipse in the doxygen documentation, for details).

        Note that the correlation function must be peaked at the origin, and is translationally
        invariant: any X0 shift in the input ellipse is therefore ignored.

        @param ellipse The galsim.Ellipse transformation to apply.
        """
        if not isinstance(ellipse, galsim.Ellipse):
            raise TypeError("Argument to applyTransformation must be a galsim.Ellipse!")
        # Create a new ellipse without a shift
        ellipse_noshift = galsim.Ellipse(shear=ellipse.getS(), mu=ellipse.getMu())
        self._profile.applyTransformation(ellipse_noshift)

    def applyExpansion(self, scale):
        """Scale the linear scale of correlations in this noise model by scale.  
        
        Scales the linear dimensions of the image by the factor scale, e.g.
        `half_light_radius` <-- `half_light_radius * scale`.

        @param scale The linear rescaling factor to apply.
        """
        self._profile.applyMagnification(scale**2)

    def applyRotation(self, theta):
        """Apply a rotation theta to this correlated noise model.
           
        @param theta Rotation angle (Angle object, +ve anticlockwise).
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        self._profile.applyRotation(theta)

    def applyShear(self, *args, **kwargs):
        """Apply a shear to this correlated noise model, where arguments are either a galsim.Shear,
        or arguments that will be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).
        """
        self._profile.applyShear(*args, **kwargs)

    # Also add methods which create a new _BaseCorrelatedNoise with the transformations applied...
    #
    def createTransformed(self, ellipse):
        """Returns a new correlated noise model by applying a galsim.Ellipse transformation (shear,
        dilate).

        The new instance will share the galsim.BaseDeviate random number generator with the parent.
        Use the .setRNG() method after this operation if you wish to use a different random number
        sequence.

        Note that galsim.Ellipse objects can be initialized in a variety of ways (see documentation
        of this class, galsim.ellipse.Ellipse in the doxygen documentation, for details).

        Note also that the correlation function must be peaked at the origin, and is translationally
        invariant: any X0 shift in the input ellipse is therefore ignored.

        @param ellipse The galsim.Ellipse transformation to apply
        @returns The transformed object.
        """
        if not isinstance(ellipse, galsim.Ellipse):
            raise TypeError("Argument to createTransformed must be a galsim.Ellipse!")
        ret = self.copy()
        ret.applyTransformation(ellipse)
        return ret

    def createExpanded(self, scale):
        """Returns a new correlated noise model by applying an Expansion by the given scale,
        scaling the linear size by scale.

        The new instance will share the galsim.BaseDeviate random number generator with the parent.
        Use the .setRNG() method after this operation if you wish to use a different random number
        sequence.

        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`
 
        @param scale The linear rescaling factor to apply.
        @returns The rescaled object.
        """
        ret = self.copy()
        ret.applyExpansion(scale)
        return ret

    def createRotated(self, theta):
        """Returns a new correlated noise model by applying a rotation.

        The new instance will share the galsim.BaseDeviate random number generator with the parent.
        Use the .setRNG() method after this operation if you wish to use a different random number
        sequence.

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        @returns The rotated object.
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        ret = self.copy()
        ret.applyRotation(theta)
        return ret

    def createSheared(self, *args, **kwargs):
        """Returns a new correlated noise model by applying a shear, where arguments are either a
        galsim.Shear or keyword arguments that can be used to create one.

        The new instance will share the galsim.BaseDeviate random number generator with the parent.
        Use the .setRNG() method after this operation if you wish to use a different random number
        sequence.

        For more details about the allowed keyword arguments, see the documentation of galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).
        """
        ret = self.copy()
        ret.applyShear(*args, **kwargs)
        return ret

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
                self.draw(imtmp, dx=1.) # GalSim internals handle this correctly w/out folding
                variance = imtmp.at(1, 1)
                self._variance_stored = variance # Store variance for next time 
        return variance

    def scaleVariance(self, variance_ratio):
        """Multiply the variance of the noise field by variance_ratio.

        @param variance_ratio The factor by which to scale the variance of the correlation function
                              profile.
        """
        self._profile.SBProfile.scaleFlux(variance_ratio)
        self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date

    def setVariance(self, variance):
        """Set the point variance of the noise field, equal to its correlation function value at
        zero distance, to an input variance.

        @param variance  The desired point variance in the noise.
        """
        variance_ratio = variance / self.getVariance()
        self.scaleVariance(variance_ratio)

    def convolveWith(self, gsobject, gsparams=None):
        """Convolve the correlated noise model with an input GSObject.

        The resulting correlated noise model will then give a statistical description of the noise
        field that would result from convolving noise generated according to the initial correlated
        noise with a kernel represented by `gsobject` (e.g. a PSF).

        The practical purpose of this method is that it allows us to model what is happening to
        noise in the images from Hubble Space Telescope that we use for simulating PSF convolved 
        galaxies with the galsim.RealGalaxy class.

        This modifies the representation of the correlation function, but leaves the random number
        generator unchanged.

        Examples
        --------
        The following command simply applies a galsim.Moffat PSF with slope parameter `beta`=3. and
        FWHM=0.7:

            >>> correlated_noise.convolveWith(galsim.Moffat(beta=3., fwhm=0.7))

        Often we will want to convolve with more than one function.  For example, if we wanted to
        simulate how a noise field would look if convolved with a ground-based PSF (such as the 
        Moffat above) and then rendered onto a new (typically larger) pixel grid, the following
        example command demonstrates the syntax: 

            >>> correlated_noise.convolveWith(
            ...    galsim.Convolve([galsim.Deconvolve(galsim.Pixel(0.03)),
            ...                     galsim.Pixel(0.2), galsim.Moffat(3., fwhm=0.7),

        Note, we also deconvolve by the original pixel, which should be the pixel size of the 
        image from which the `correlated_noise` was made.  This command above is functionally 
        equivalent to

            >>> correlated_noise.convolveWith(galsim.Deconvolve(galsim.Pixel(0.03)))
            >>> correlated_noise.convolveWith(galsim.Pixel(0.2))
            >>> correlated_noise.convolveWith(galsim.Moffat(beta=3., fwhm=0.7))

        as is demanded for a linear operation such as convolution.

        @param gsobject  A galsim.GSObject or derived class instance representing the function with
                         which the user wants to convolve the correlated noise model.
        @param gsparams  You may also specify a gsparams argument.  See the docstring for 
                         GSObject for more information about this option.
        """
        self._profile = galsim.Convolve(
            [self._profile, galsim.AutoCorrelate(gsobject)], gsparams=gsparams)

    def draw(self, image=None, dx=None, wmult=1., add_to_image=False):
        """The draw method for profiles storing correlation functions.

        This is a very mild reimplementation of the draw() method for GSObjects.  The normalization
        is automatically set to have the behviour appropriate for a correlation function, and the 
        `gain` kwarg is automatically set to unity.

        See the general GSObject draw() method for more information the input parameters.
        """
        return self._profile.draw(
            image=image, dx=dx, gain=1., wmult=wmult, normalization="surface brightness",
            add_to_image=add_to_image, use_true_center=False)

    def calculateCovarianceMatrix(self, bounds, dx):
        """Calculate the covariance matrix for an image with specified properties.

        A correlation function also specifies a covariance matrix for noise in an image of known
        dimensions and pixel scale.  The user specifies these bounds and pixel scale, and this
        method returns a covariance matrix as a square ImageD object, with the upper triangle
        containing the covariance values.

        @param  bounds Bounds corresponding to the dimensions of the image for which a covariance
                       matrix is required.
        @param  dx     Pixel scale of the image for which a covariance matrix is required.

        @return The covariance matrix (as an ImageD)
        """
        return galsim._galsim._calculateCovarianceMatrix(self._profile.SBProfile, bounds, dx)

    def _get_update_rootps(self, shape, dx):
        """Internal utility function for querying the rootps cache, used by applyTo and 
        applyWhiteningTo methods.
        """ 
        # First check whether we can just use a stored power spectrum (no drawing necessary if so)
        use_stored = False
        for rootps_array, scale in self._rootps_store:
            if shape == rootps_array.shape:
                if ((dx <= 0. and scale == 1.) or (dx == scale)):
                    use_stored = True
                    rootps = rootps_array
                    break

        # If not, draw the correlation function to the desired size and resolution, then DFT to
        # generate the required array of the square root of the power spectrum
        if use_stored is False:
            newcf = galsim.ImageD(shape[1], shape[0]) # set the corr func to be the correct size
            # set the scale based on dx...
            if dx <= 0.:
                newcf.setScale(1.) # sometimes new Images have getScale() = 0
            else:
                newcf.setScale(dx)
            # Then draw this correlation function into an array
            self.draw(newcf, dx=None) # setting dx=None uses the newcf image scale set above

            # Then calculate the sqrt(PS) that will be used to generate the actual noise
            rootps = np.sqrt(np.abs(np.fft.fft2(newcf.array)) * np.product(shape))

            # Then add this and the relevant scale to the _rootps_store for later use
            self._rootps_store.append((rootps, newcf.getScale()))

        return rootps

    def _get_update_rootps_whitening(self, shape, dx, headroom=1.05):
        """Internal utility function for querying the rootps_whitening cache, used by the
        applyWhiteningTo method, and calculate & update it if not present.

        @return rootps_whitening, variance
        """ 
        # First check whether we can just use a stored whitening power spectrum
        use_stored = False
        for rootps_whitening_array, scale, var in self._rootps_whitening_store:
            if shape == rootps_whitening_array.shape:
                if ((dx <= 0. and scale == 1.) or (dx == scale)):
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

            rootps = self._get_update_rootps(shape, dx)
            ps_whitening = -rootps * rootps
            ps_whitening += np.abs(np.min(ps_whitening)) * headroom # Headroom adds a little extra
            rootps_whitening = np.sqrt(ps_whitening)                # variance, for "safety"

            # Finally calculate the theoretical combined variance to output alongside the image 
            # to be generated with the rootps_whitening.  The factor of product of the image shape
            # is required due to inverse FFT conventions, and note that although we use the [0, 0] 
            # element we could use any as the PS should be flat
            variance = (rootps[0, 0]**2 + ps_whitening[0, 0]) / np.product(shape)

            # Then add all this and the relevant scale dx to the _rootps_whitening_store
            self._rootps_whitening_store.append((rootps_whitening, dx, variance))

        return rootps_whitening, variance

###
# Now a standalone utility function for generating noise according to an input (square rooted)
# Power Spectrum
#
def _generate_noise_from_rootps(rng, rootps):
    """Utility function for generating a NumPy array containing a Gaussian random noise field with
    a user-specified power spectrum also supplied as a NumPy array.

    @param rng    galsim.BaseDeviate instance to provide the random number generation
    @param rootps a NumPy array containing the square root of the discrete Power Spectrum ordered
                  in two dimensions according to the usual DFT pattern (see np.fft.fftfreq)
    @return A NumPy array (contiguous) of the same shape as rootps, filled with the noise field.
    """
    # I believe it is cheaper to make two random vectors than to make a single one (for a phase)
    # and then apply cos(), sin() to it...
    gaussvec_real = galsim.ImageD(rootps.shape[1], rootps.shape[0]) # Remember NumPy is [y, x]
    gaussvec_imag = galsim.ImageD(rootps.shape[1], rootps.shape[0])
    gn = galsim.GaussianNoise(rng, sigma=1.) # Quicker to create anew each time than to save it and
                                             # then check if its rng needs to be changed or not.
    gaussvec_real.addNoise(gn)
    gaussvec_imag.addNoise(gn)
    noise_array = np.fft.ifft2((gaussvec_real.array + gaussvec_imag.array * 1j) * rootps)
    return np.ascontiguousarray(noise_array.real)


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
    added to an Image.

    It also allows the combination of multiple correlation functions by addition, and for the
    scaling of the total variance they represent by scalar factors.

    Convolution of correlation functions with a GSObject is not yet supported, but will be in the 
    near future.

    Initialization
    --------------

    Basic example:

        >>> cn = galsim.CorrelatedNoise(rng, image)

    Instantiates a CorrelatedNoise using the pixel scale information contained in image.getScale()
    (assumes the scale is unity if image.getScale() <= 0.) by calculating the correlation function
    in the input `image`.  The input `rng` must be a galsim.BaseDeviate or derived class instance,
    setting the random number generation for the noise.

    Optional Inputs
    ---------------

        >>> cn = galsim.CorrelatedNoise(rng, image, dx=0.2)

    The example above instantiates a CorrelatedNoise, but forces the use of the pixel scale `dx` to
    set the units of the internal lookup table.

        >>> cn = galsim.CorrelatedNoise(rng, image,
        ...     x_interpolant=galsim.InterpolantXY(galsim.Lanczos(5, tol=1.e-4))

    The example above instantiates a CorrelatedNoise, but forces use of a non-default interpolant
    for interpolation of the internal lookup table in real space.  Must be an InterpolantXY instance
    or an Interpolant instance (if the latter one-dimensional case is supplied, an InterpolantXY
    will be automatically generated from it).

    The default x_interpolant if `None` is set is a galsim.InterpolantXY(galsim.Linear(tol=1.e-4)),
    which uses bilinear interpolation.  Initial tests indicate the favourable performance of this
    interpolant in applications involving correlated pixel noise.

    There is also an option to switch off an internal correction for assumptions made about the
    periodicity in the input noise image.  If you wish to turn this off you may, e.g.

        >>> cn = galsim.CorrelatedNoise(rng, image, correct_periodicity=False)
    
    The default and generally recommended setting is `correct_periodicity=True`.

    Users should note that the internal calculation of the discrete correlation function in `image`
    will assume that `image` is periodic across its boundaries, introducing a dilution bias in the
    estimate of inter-pixel correlations that increases with separation.  Unless you know that the
    noise in `image` is indeed periodic (perhaps because you generated it to be so), you will not
    generally wish to use the `correct_periodicity=False` option.

    By default, the image is not mean subtracted before the correlation function is estimated.  To
    do an internal mean subtraction, you can set the `subtract_mean` keyword to `True`, e.g.

        >>> cn = galsim.CorrelatedNoise(rng, image, subtract_mean=True)

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

    Methods and Use
    ---------------
    The main way that a CorrelatedNoise is used is to add or assign correlated noise to an image.
    This is common to all the classes that inherit from BaseNoise: to add deviates to every element
    of an image, the syntax

        >>> im.addNoise(cn)

    is preferred, although

        >>> cn.applyTo(im)

    is equivalent.  See the .addNoise() method docstring for more information.  The
    image.getScale() method is used to get the pixel scale of the input image unless this is <= 0,
    in which case a scale of 1 is assumed.

    Another method that may be of use is

        >>> cn.calculateCovarianceMatrix(im.bounds, dx)

    which can be used to generate a covariance matrix based on a user input image geometry.  See
    the .calculateCovarianceMatrix() method docstring for more information.

    A number of methods familiar from GSObject instances have also been implemented directly as 
    `cn` methods, so that the following commands are all legal:

        >>> cn.draw(im, dx, wmult=4)
        >>> cn.createSheared(s)
        >>> cn.createExpanded(m)
        >>> cn.createRotated(theta * galsim.degrees)
        >>> cn.createTransformed(ellipse)
        >>> cn.applyShear(s)
        >>> cn.applyExpansion(scale)  # Behaves similarly to applyMagnification
        >>> cn.applyRotation(theta * galsim.degrees)
        >>> cn.applyTransformation(ellipse)

    See the individual method docstrings for more details.  The .applyShift() and .createShifted()
    methods are not available since a correlation function must always be centred and peaked at the
    origin.

    The BaseNoise methods

        >>> cn.getVariance()
        >>> cn.setVariance(variance)
        >>> cn.scaleVariance(variance_ratio)
 
    can be used to get and set the point variance of the correlated noise, equivalent to the zero
    separation distance correlation function value.  The .setVariance(variance) method scales the
    whole internal correlation function so that its point variance matches `variance`.

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
    addition operations it is the galsim.BaseDeviate belonging to the instance on the Left-Hand Side
    of the operator that is retained. 

    In the example above therefore, it is the random number generator from `cn2` that will be stored
    and used by `cn3`, and `cn4` will retain its random number generator after in-place addition of
    `cn5`.  The random number generator of `cn5` is not affected by the operation.

    The multiplication and division operators, e.g.

        >>> cn1 /= 3.
        >>> cn2 = cn1 * 3

    scale the overall correlation function by a scalar operand using the .scaleVariance() method
    described above.  The random number generators are not affected by these scaling operations.
    """
    def __init__(self, rng, image, dx=0., x_interpolant=None, correct_periodicity=True,
        subtract_mean=False):

        # Check that the input image is in fact a galsim.ImageSIFD class instance
        if not isinstance(image, (
            galsim.BaseImageD, galsim.BaseImageF, galsim.BaseImageS, galsim.BaseImageI)):
            raise TypeError(
                "Input image not a galsim.Image class object (e.g. ImageD, ImageViewS etc.)")
        # Build a noise correlation function (CF) from the input image, using DFTs
        # Calculate the power spectrum then a (preliminary) CF 
        ft_array = np.fft.fft2(image.array)
        ps_array = (ft_array * ft_array.conj()).real
        if subtract_mean: # Quickest non-destructive way to make the PS correspond to the
                          # mean-subtracted case
            ps_array[0, 0] = 0.
        # Note need to normalize due to one-directional 1/N^2 in FFT conventions
        cf_array_prelim = (np.fft.ifft2(ps_array)).real / np.product(image.array.shape)

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
        cf_image = galsim.ImageViewD(np.ascontiguousarray(cf_array))

        # Correctly record the original image scale if set
        if dx > 0.:
            cf_image.setScale(dx)
        elif image.getScale() > 0.:
            cf_image.setScale(image.getScale())
        else: # sometimes Images are instantiated with scale=0, in which case we will assume unit
              # pixel scale
            cf_image.setScale(1.)

        # If x_interpolant not specified on input, use bilinear
        if x_interpolant == None:
            linear = galsim.Linear(tol=1.e-4)
            x_interpolant = galsim.InterpolantXY(linear)
        else:
            x_interpolant = utilities.convert_interpolant_to_2d(x_interpolant)

        # Then initialize...
        cf_object = base.InterpolatedImage(
            cf_image, x_interpolant=x_interpolant, dx=cf_image.getScale(), normalization="sb",
            calculate_stepk=False, calculate_maxk=False) # these internal calculations do not seem
                                                         # to do very well with often sharp-peaked
                                                         # correlation function images...
        _BaseCorrelatedNoise.__init__(self, rng, cf_object)

        if store_rootps:
            # If it corresponds to the CF above, store useful data as a (rootps, dx) tuple for
            # efficient later use:
            self._profile_for_stored = self._profile
            self._rootps_store.append((np.sqrt(ps_array), cf_image.getScale()))


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
    therefore only apply this correction before using galsim.utilities.roll2d to recentre the image
    of the correlation function.
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
    """
    return CorrelatedNoise(image)

# Then add this Image method to the Image classes
for Class in galsim.Image.itervalues():
    Class.getCorrelatedNoise = _Image_getCorrelatedNoise

for Class in galsim.ImageView.itervalues():
    Class.getCorrelatedNoise = _Image_getCorrelatedNoise

for Class in galsim.ConstImageView.itervalues():
    Class.getCorrelatedNoise = _Image_getCorrelatedNoise

# Free function for returning a COSMOS noise field correlation function
def getCOSMOSNoise(rng, file_name, dx_cosmos=0.03, variance=0., x_interpolant=None):
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

    @param rng            Must be a galsim.BaseDeviate or derived class instance, provides the
                          random number generator used by the returned _BaseCorrelatedNoise
                          instance.
    @param file_name      String containing the path and filename above but modified to match the
                          location of the GalSim repository on your system.
    @param dx_cosmos      COSMOS ACS F814W coadd image pixel scale in the units you are using to
                          describe GSObjects and image scales in GalSim: defaults to 0.03 arcsec,
                          see below for more information.
    @param variance       Scales the correlation function so that its point variance, equivalent to
                          its value at zero separation distance, matches this value.  The default
                          `variance = 0.` uses the variance in the original COSMOS noise fields.
    @param x_interpolant  Forces use of a non-default interpolant for interpolation of the internal
                          lookup table in real space.  Supplied kwarg must be an InterpolantXY
                          instance or an Interpolant instance (from which an InterpolantXY will be
                          automatically generated).  Defaults to use of the bilinear interpolant
                          `galsim.InterpolantXY(galsim.Linear(tol=1.e-4))`, see below.

    @return A _BaseCorrelatedNoise instance representing correlated noise in F814W COSMOS images.

    The interpolation used if `x_interpolant=None` (default) is a
    galsim.InterpolantXY(galsim.Linear(tol=1.e-4)), which uses bilinear interpolation.  Initial
    tests indicate the favourable performance of this interpolant in applications involving
    correlated noise.

    Important note regarding units
    ------------------------------
    The ACS coadd images in COSMOS have a pixel scale of 0.03 arcsec, and so the pixel scale
    `dx_cosmos` adopted in the representation of of the correlation function takes a default value

        dx_cosmos = 0.03

    If you wish to use other units, ensure that the input keyword `dx_cosmos` takes the value
    corresponding to 0.03 arcsec in your chosen system.

    Example usage
    -------------
    The following commands use this function to generate a 300 pixel x 300 pixel image of noise with
    HST COSMOS correlation properties (substitute in your own file and path for the `filestring`).

        >>> filestring='/YOUR/REPO/PATH/GalSim/devel/external/hst/acs_I_unrot_sci_20_cf.fits'
        >>> import galsim
        >>> rng = galsim.UniformDeviate(123456)
        >>> cf = galsim.correlatednoise.getCOSMOSNoise(rng, filestring)
        >>> im = galsim.ImageD(300, 300)
        >>> im.setScale(0.03)
        >>> cf.applyTo(im)
        >>> im.write('out.fits')

    The FITS file `out.fits` should then contain an image of randomly-generated, COSMOS-like noise.
    """
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
    ret = _BaseCorrelatedNoise(rng, base.InterpolatedImage(
        cfimage, dx=dx_cosmos, normalization="sb", calculate_stepk=False, calculate_maxk=False,
        x_interpolant=x_interpolant))
    # If the input keyword variance is non-zero, scale the correlation function to have this
    # variance
    if variance > 0.:
        ret.setVariance(variance)
    return ret
