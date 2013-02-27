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

class _BaseCorrelatedNoise(galsim._galsim.BaseNoise):
    """A Base Class describing 2D correlated noise fields.

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
        
        # Act as a container for the GSObject used to represent the correlation funcion.
        self._profile = gsobject

        # When applying noise to an image, we normally do a calculation. 
        # If _profile_for_stored is profile, then it means that we can use the stored values
        # in _rootps_store and avoid having to redo the calculation.
        # So for now, we start out with _profile_for_stored = None and _rootps_store empty.
        self._profile_for_stored = None
        self._rootps_store = []

        # Cause any methods we don't want the user to have access to, since they don't make sense
        # for correlation functions and could cause errors in applyNoiseTo, to raise exceptions
        self._profile.applyShift = self._notImplemented

    # Make "+" work in the intuitive sense (variances being additive, correlation functions add as
    # you would expect)
    def __add__(self, other):
        ret = self.copy()
        ret += other
        return ret

    def __iadd__(self, other):
        self._profile += other._profile
        return _CorrFunc(self._profile)

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
        """Returns a copy of the correlation function.
        """
        return _CorrFunc(self._profile.copy())

    def applyNoiseTo(self, image, dx=0., dev=None, add_to_image=True):
        """Apply noise as a Gaussian random field with this correlation function to an input Image.

        If the optional image pixel scale `dx` is not specified, `image.getScale()` is used for the
        input image pixel separation.
        
        If an optional random deviate `dev` is supplied, the application of noise will share the
        same underlying random number generator when generating the vector of unit variance
        Gaussians that seed the (Gaussian) noise field.

        @param image The input Image object.
        @param dx    The pixel scale to adopt for the input image; should use the same units the
                     ImageCorrFunc instance for which this is a method.  If is not specified,
                     `image.getScale()` is used instead.
        @param dev   Optional BaseDeviate from which to draw pseudo-random numbers in generating
                     the noise field.
        @param add_to_image  Whether to add to the existing image rather than clear out anything
                             in the image before drawing.
                             Note: This requires that the image has defined bounds (default 
                             `add_to_image = True`).
        """
        # Note that this uses the (fast) method of going via the power spectrum and FFTs to generate
        # noise according to the correlation function represented by this instance.  An alternative
        # would be to use the covariance matrices and eigendecomposition.  However, it is O(N^6)
        # operations for an NxN image!  FFT-based noise realization is O(2 N^2 log[N]) so we use it
        # for noise generation applications.

        # Check that the input has defined bounds
        if not hasattr(image, "bounds"):
            raise ValueError(
                "Input image argument does not have a bounds attribute, it must be a galsim.Image"+
                "or galsim.ImageView-type object with defined bounds.")

        # Set up the GaussianNoise object we will need later
        if dev is None:
            dev = galsim.BaseDeviate()
        elif not isinstance(dev, galsim.BaseDeviate):
            raise TypeError(
                "Supplied input keyword dev must be a galsim.BaseDeviate or derived class "+
                "(e.g. galsim.UniformDeviate, galsim.GaussianDeviate).")
        g = galsim.GaussianNoise(dev, sigma=1.)

        # If the profile has changed since last time (or if we have never been here before),
        # clear out the stored values.
        if self._profile_for_stored is not self._profile:
            self._rootps_store = []
        # Set profile_for_stored for next time.
        self._profile_for_stored = self._profile

        # Then retrieve or redraw the sqrt(power spectrum) needed for making the noise field:

        # First check whether we can just use the stored power spectrum (no drawing necessary if so)
        use_stored = False
        for rootps_array, scale in self._rootps_store:
            if image.array.shape == rootps_array.shape:
                if ((dx <= 0. and scale == 1.) or (dx == scale)):
                    use_stored = True
                    rootps = rootps_array
                    break

        # If not, draw the correlation function to the desired size and resolution, then DFT to
        # generate the required array of the square root of the power spectrum
        if use_stored is False:
            newcf = galsim.ImageD(image.bounds) # set the correlation func to be the correct size
            # set the scale based on dx...
            if dx <= 0.:
                if image.getScale() > 0.:
                    newcf.setScale(image.getScale())
                else:
                    newcf.setScale(1.) # sometimes new Images have getScale() = 0
            else:
                newcf.setScale(dx)
            # Then draw this correlation function into an array
            self.draw(newcf, dx=None) # setting dx=None uses the newcf image scale set above

            # Then calculate the sqrt(PS) that will be used to generate the actual noise
            rootps = np.sqrt(np.abs(np.fft.fft2(newcf.array)) * np.product(image.array.shape))

            # Then add this and the relevant scale to the _rootps_store for later use
            self._rootps_store.append((rootps, newcf.getScale()))

        # Finally generate a random field in Fourier space with the right PS, and inverse DFT back,
        # including factor of sqrt(2) to account for only adding noise to the real component:
        gaussvec = galsim.ImageD(image.bounds)
        gaussvec.addNoise(g)
        noise_array = np.sqrt(2.) * np.fft.ifft2(gaussvec.array * rootps)
        # Make contiguous and add/assign to the image
        if add_to_image:
            image += galsim.ImageViewD(np.ascontiguousarray(noise_array.real))
        else:
            image = galsim.ImageViewD(np.ascontiguousarray(noise_array.real))
        return image

    def applyTransformation(self, ellipse):
        """Apply a galsim.Ellipse distortion to this correlation function.
           
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

    def applyMagnification(self, scale):
        """Scale the linear size of this _CorrFunc by scale.  
        
        Scales the linear dimensions of the image by the factor scale, e.g.
        `half_light_radius` <-- `half_light_radius * scale`.

        @param scale The linear rescaling factor to apply.
        """
        self.applyTransformation(galsim.Ellipse(np.log(scale)))

    def applyRotation(self, theta):
        """Apply a rotation theta to this object.
           
        After this call, the caller's type will still be a _CorrFunc, unlike in the GSObject base
        class implementation of this method.  This is to allow _CorrFunc methods to be available
        after transformation, such as .applyNoiseTo().

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        self._profile.applyRotation(theta)

    def applyShear(self, *args, **kwargs):
        """Apply a shear to this object, where arguments are either a galsim.Shear, or arguments
        that will be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).

        After this call, the caller's type will still be a _CorrFunc.  This is to allow _CorrFunc
        methods to be available after transformation, such as .applyNoiseTo().
        """
        self._profile.applyShear(*args, **kwargs)

    # Also add methods which create a new GSObject with the transformations applied...
    #
    def createTransformed(self, ellipse):
        """Returns a new correlation function by applying a galsim.Ellipse transformation (shear,
        dilate).

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

    def createMagnified(self, scale):
        """Returns a new correlation function by applying a magnification by the given scale,
        scaling the linear size by scale.  
 
        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`
 
        @param scale The linear rescaling factor to apply.
        @returns The rescaled object.
        """
        ret = self.copy()
        ret.applyTransformation(galsim.Ellipse(np.log(scale)))
        return ret

    def createRotated(self, theta):
        """Returns a new correlation function by applying a rotation.

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        @returns The rotated object.
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        ret = self.copy()
        ret.applyRotation(theta)
        return ret

    def createSheared(self, *args, **kwargs):
        """Returns a new correlation function by applying a shear, where arguments are either a
        galsim.Shear or keyword arguments that can be used to create one.

        For more details about the allowed keyword arguments, see the documentation of galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).
        """
        ret = self.copy()
        ret.applyShear(*args, **kwargs)
        return ret

    # Now I define some methods that are not used by this instance directly, but are used to
    # redefine the behaviour of the stored profile, or print a method saying that this method is not
    # implemented
    def scaleVariance(self, variance_ratio):
        """Multiply the overall variance of the correlation function profile by variance_ratio.

        @param variance_ratio The factor by which to scale the variance of the correlation function
                              profile.
        """
        self._profile.SBProfile.scaleFlux(variance_ratio)

    def _notImplemented(self, *args, **kwargs):
        raise NotImplementedError(
            "This method is not available for profiles that represent correlation functions.")

    def draw(self, image=None, dx=None, wmult=1., add_to_image=False):
        """The draw method for profiles storing correlation functions.

        This is a very mild reimplementation of the draw() method for GSObjects.  The normalization
        is automatically set to have the behviour appropriate for a correlation function, and the 
        `gain` kwarg is automatically set to unity.

        See the general GSObject draw() method for more information the input parameters.
        """
        return self._profile.draw(
            image=image, dx=dx, gain=1., wmult=wmult, normalization="surface brightness",
            add_to_image=add_to_image)

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

###
# Then we define the ImageCorrFunc, which generates a correlation function by estimating it directly
# from images:
#
class ImageCorrFunc(_BaseCorrelatedNoise):
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

        >>> cf = galsim.ImageCorrFunc(image)

    Instantiates an ImageCorrFunc using the pixel scale information contained in image.getScale()
    (assumes the scale is unity if image.getScale() <= 0.)

    Optional Inputs
    ---------------

        >>> cf = galsim.ImageCorrFunc(image, dx=0.2)

    The example above instantiates an ImageCorrFunc, but forces the use of the pixel scale `dx` to
    set the units of the internal lookup table.

        >>> cf = galsim.ImageCorrFunc(image,
        ...     interpolant=galsim.InterpolantXY(galsim.Lanczos(5, tol=1.e-4))

    The example above instantiates a ImageCorrFunc, but forces the use of a non-default interpolant
    for interpolation of the internal lookup table.  Must be an InterpolantXY instance or an
    Interpolant instance (if the latter one-dimensional case is supplied an InterpolantXY will be
    automatically generated from it).

    The default interpolant if `None` is set is a galsim.InterpolantXY(galsim.Linear(tol=1.e-4)),
    which uses bilinear interpolation.  Initial tests indicate the favourable performance of this
    interpolant in applications involving correlated pixel noise.

    Methods
    -------
    The main way that ImageCorrFunc is used is to add or assign correlated noise to an image.
    This is done with

        cf.applyNoiseTo(im)

    The correlation function is calculated from its pixel values using the NumPy FFT functions.
    Optionally, the pixel scale for the input `image` can be specified using the `dx` keyword
    argument.  See the .applyNoiseTo() method docstring for more information.

    If `dx` is not set the value returned by `image.getScale()` is used unless this is <= 0, in
    which case a scale of 1 is assumed.

    Another method that may be of use is

        cf.calculateCovarianceMatrix(im.bounds, dx)

    which can be used to generate a covariance matrix based on a user input image geometry.  See
    the .calculateCovarianceMatrix() method docstring for more information.

    A number of methods familiar from GSObject instance have also been implemented directly as 
    `cf` methods, so that the following commands are all legal:

        cf.draw(im, dx, wmult=4)
        cf.createSheared(s)
        cf.createMagnified(m)
        cf.createRotated(theta * galsim.degrees)
        cf.createTransformed(ellipse)
        cf.applyShear(s)
        cf.applyMagnification(m)
        cf.applyRotation(theta * galsim.degrees)
        cf.applyTransformation(ellipse)

    See the individual method docstrings for more details.

    A new method, which is in fact a more appropriately named reimplmentation of the
    .scaleFlux() method in GSObject instances, is

        cf.scaleVariance(variance_ratio)

    which scales the overall correlation function, and therefore its total variance, by a scalar
    factor `variance_ratio`.

    Arithmetic Operators
    --------------------
    Addition, multiplication and division operators are defined to work in an intuitive way for
    correlation functions.

    Addition works simply to add the internally-stored correlation functions, so that

        >>> cf2 = cf0 + cf1
        >>> cf2 += cf1

    provides a representation of the correlation function of two linearly summed fields represented
    by the individual correlation function operands.

    The multiplication and division operators scale the overall correlation function by a scalar 
    operand, using the .scaleVariance() method described above.
    """
    def __init__(self, image, dx=0., interpolant=None):

        # Check that the input image is in fact a galsim.ImageSIFD class instance
        if not isinstance(image, (
            galsim.BaseImageD, galsim.BaseImageF, galsim.BaseImageS, galsim.BaseImageI)):
            raise TypeError(
                "Input image not a galsim.Image class object (e.g. ImageD, ImageViewS etc.)")
        # Build a noise correlation function (CF) from the input image, using DFTs

        # Calculate the power spectrum then a (preliminary) CF 
        ft_array = np.fft.fft2(image.array)
        ps_array = np.abs(ft_array * ft_array.conj())
        cf_array_prelim = (np.fft.ifft2(ps_array)).real / float(np.product(np.shape(ft_array)))

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
        # row to the uppermost row, if the the original CF had even dimensions in the x and y 
        # directions, respectively (remembering again that NumPy stores data [y,x] in arrays)
        if cf_array_prelim.shape[1] % 2 == 0: # first do x
            lhs_column = cf_array[:, 0]
            cf_array[:, cf_array_prelim.shape[1]] = lhs_column[::-1] # inverts order as required
        if cf_array_prelim.shape[0] % 2 == 0: # then do y
            bottom_row = cf_array[0, :]
            cf_array[cf_array_prelim.shape[0], :] = bottom_row[::-1] # inverts order as required

        # Store power spectrum and correlation function in an image 
        original_ps_image = galsim.ImageViewD(np.ascontiguousarray(ps_array))
        original_cf_image = galsim.ImageViewD(np.ascontiguousarray(cf_array))

        # Correctly record the original image scale if set
        if dx > 0.:
            original_cf_image.setScale(dx)
        elif image.getScale() > 0.:
            original_cf_image.setScale(image.getScale())
        else: # sometimes Images are instantiated with scale=0, in which case we will assume unit
              # pixel scale
            original_image.setScale(1.)
            original_cf_image.setScale(1.)

        # If interpolant not specified on input, use bilinear
        if interpolant == None:
            linear = galsim.Linear(tol=1.e-4)
            interpolant = galsim.InterpolantXY(linear)
        else:
            if isinstance(interpolant, galsim.Interpolant):
                interpolant = galsim.InterpolantXY(interpolant)
            elif isinstance(interpolant, galsim.InterpolantXY):
                interpolant = interpolant
            else:
                raise RuntimeError(
                    'Specified interpolant is not an Interpolant or InterpolantXY instance!')

        # Then initialize...
        _CorrFunc.__init__(self, base.InterpolatedImage(
            original_cf_image, interpolant, dx=original_cf_image.getScale(), normalization="sb",
            calculate_stepk=False, calculate_maxk=False)) # these internal calculations do not seem
                                                          # to do very well with often sharp-peaked
                                                          # correlation function images...

        # Finally store useful data as a (rootps, dx) tuple for efficient later use:
        self._profile_for_stored = self._profile
        self._rootps_store.append(
            (np.sqrt(original_ps_image.array), original_cf_image.getScale()))

# Make a function for returning Noise correlations
def _Image_getCorrFunc(image):
    """Returns an ImageCorrFunc instance by calculating the correlation function of image pixels.
    """
    return ImageCorrFunc(image)

# Then add this Image method to the Image classes
for Class in galsim.Image.itervalues():
    Class.getCorrFunc = _Image_getCorrFunc

for Class in galsim.ImageView.itervalues():
    Class.getCorrFunc = _Image_getCorrFunc

for Class in galsim.ConstImageView.itervalues():
    Class.getCorrFunc = _Image_getCorrFunc

# Free function for returning a COSMOS noise field correlation function
def get_COSMOS_CorrFunc(file_name, dx_cosmos=0.03, variance=0.):
    """Returns a 2D discrete correlation function representing noise in the HST COSMOS F814W
    unrotated science coadd images.

    See http://cosmos.astro.caltech.edu/astronomer/hst.html for information about the COSMOS survey,
    and Leauthaud et al (2007) for detailed information about the unrotated F814W coadds used for
    weak lensing science.

    This function uses a stacked estimate of the correlation function in COSMOS noise fields, the
    location of which should be input to this function via the `file_name` argument.  This image is
    stored in FITS format, and is generated as described in
    `YOUR/REPO/PATH/GalSim/devel/external/hst/make_cosmos_cfimage.py`.  The image itself can also be
    found within the GalSim repo, located at:

        /YOUR/REPO/PATH/GalSim/examples/data/acs_I_unrot_sci_20_cf.fits

    @param file_name  String containing the path and filename above but modified to match the
                      location of the GalSim repoistory on your system.
    @param dx_cosmos  COSMOS ACS F814W coadd image pixel scale in the units you are using to
                      describe GSObjects and image scales in GalSim: defaults to 0.03 arcsec, see
                      below for more information.
    @variance         Scales the correlation function so that its point variance, equivalent to its
                      value at zero separation distance, matches this value.  The default
                      `variance = 0.` uses the variance in the original COSMOS noise fields.

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
        >>> cf = galsim.correlatednoise.get_COSMOS_CorrFunc(filestring)
        >>> im = galsim.ImageD(300, 300)
        >>> cf.applyNoiseTo(im, dx=0.03)
        >>> im.write('out.fits')

    The FITS file `out.fits` should then contain an image of randomly-generated, COSMOS-like noise.
    """
    # First try to read in the image of the COSMOS correlation function stored in the repository
    import os
    if not os.path.isfile(file_name):
        raise IOError("The input file_name '"+str(file_name)+"' does not exist.")
    try:
        cfimage = galsim.fits.read(file_name)
    except Exception as original_exception:
        # Give a vaguely helpful warning, then raise the original exception for extra diagnostics
        import warnings
        warnings.warn(
            "Function get_COSMOS_CorrFunc() unable to read FITS image from "+str(file_name)+", "+
            "more information on the error in the following Exception...")
        raise original_exception

    # Then check for negative variance before doing anything time consuming
    if variance < 0:
        raise ValueError("Input keyword variance must be zero or positive.")
    
    # Use this info to then generate a correlation function DIRECTLY: note this is non-standard
    # usage, but tolerated since we can be sure that the input cfimage is appropriately symmetric
    # and peaked at the origin
    ret = _CorrFunc(base.InterpolatedImage(
        cfimage, dx=dx_cosmos, normalization="sb", calculate_stepk=False, calculate_maxk=False))
    # If the input keyword variance is non-zero, scale the correlation function to have this
    # variance
    if variance > 0.:
        var_original = ret._profile.xValue(galsim.PositionD(0., 0.))
        ret.scaleVariance(variance / var_original)
    return ret

