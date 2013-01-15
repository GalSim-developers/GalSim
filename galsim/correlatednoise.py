"""@file correlatednoise.py

Python layer documentation and functions for handling correlated noise in GalSim.
"""

import numpy as np
import galsim
from . import base
from . import utilities

class BaseCorrFunc(object):
    """A base class for correlation function objects, will not be instantiated directly in general
    use.

    This class defines the interface with which all GalSim correlation function objects access their
    shared methods and attributes, particularly with respect to the CorrelationFunction attribute (a
    GSObject).
    """
    def __init__(self, gsobject):
        if not isinstance(gsobject, base.GSObject):
            raise TypeError(
                "Correlation function objects must be initialized with a GSObject.")
        
        # Act as a container for the GSObject used to represent the correlation funcion.
        self._GSCorrelationFunction = gsobject
        # Setup a store for later, containing array representations of the sqrt(PowerSpectrum)
        # [useful for later applying noise to images according to this correlation function].
        # List items will store this data as (rootps, dx) tuples.
        self._rootps_store = []

    # Make op+ of two GSObjects work to return an CFAdd object
    def __add__(self, other):
        return AddCorrFunc(self, other)

    # Ditto for op+=
    def __iadd__(self, other):
        return AddCorrFunc(self, other)

    # Make op* and op*= work to scale the overall variance of a correlation function
    def __imul__(self, other):
        self.scaleVariance(other)
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

    # Make a copy of the ImageCorrFunc
    def copy(self):
        """Returns a copy of the correlation function.
        """
        import copy
        cf = galsim._galsim._CorrelationFunction(self._GSCorrelationFunction.SBProfile)
        ret = BaseCorrFunc(base.GSObject(cf))
        ret.__class__ = self.__class__
        ret._rootps_store = copy.deepcopy(self._rootps_store) # possible due to Jim's image pickling
        return ret

    def xValue(self, position):
        """Returns the value of the correlation function at a chosen 2D position in real space.
        
        @param position  A 2D galsim.PositionD/galsim.PositionI instance giving the position in real
                         space.
        """
        return self._GSCorrelationFunction.xValue(position)

    def kValue(self, position):
        """Returns the value of the correlation function at a chosen 2D position in k space.

        @param position  A 2D galsim.PositionD/galsim.PositionI instance giving the position in k 
                         space.
        """
        return self._GSCorrelationFunction.kValue(position)

    def scaleVariance(self, variance_ratio):
        """Multiply the overall variance of the correlation function by variance_ratio.

        @param variance_ratio The factor by which to scale the variance of the correlation function.
        """
        if isinstance(self, AddCorrFunc):
            self._GSCorrelationFunction.SBProfile.scaleFlux(variance_ratio)
        else:
            self._GSCorrelationFunction.SBProfile.scaleVariance(variance_ratio)

    def applyTransformation(self, ellipse):
        """Apply a galsim.Ellipse distortion to this correlation function.
           
        galsim.Ellipse objects can be initialized in a variety of ways (see documentation of this
        class, galsim.ellipse.Ellipse in the doxygen documentation, for details).  Resets the
        internal `_rootps_store` attribute.

        @param ellipse The galsim.Ellipse transformation to apply.
        """
        if not isinstance(ellipse, galsim.Ellipse):
            raise TypeError("Argument to applyTransformation must be a galsim.Ellipse!")
        self._GSCorrelationFunction.applyTransformation(ellipse)
        self._rootps_store = []

    def applyScaling(self, scale):
        """Scale the linear size of this CorrFunc by scale.  
        
        Scales the linear dimensions of the image by the factor scale, e.g.
        `half_light_radius` <-- `half_light_radius * scale`.  Resets the internal `_rootps_store`
        attribute.

        @param scale The linear rescaling factor to apply.
        """
        self.applyTransformation(galsim.Ellipse(np.log(scale))) # _rootps_store is reset here too

    def applyRotation(self, theta):
        """Apply a rotation theta to this object.
           
        After this call, the caller's type will still be a CorrFunc, unlike in the GSObject base
        class implementation of this method.  This is to allow CorrFunc methods to be available
        after transformation, such as .applyNoiseTo().  Resets the internal `_rootps_store`
        attribute.

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        self._GSCorrelationFunction.applyRotation(theta)
        self._rootps_store = []

    def applyShear(self, *args, **kwargs):
        """Apply a shear to this object, where arguments are either a galsim.Shear, or arguments
        that will be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).

        After this call, the caller's type will still be a CorrFunc.  This is to allow CorrFunc
        methods to be available after transformation, such as .applyNoiseTo().
        Resets the internal `_rootps_store` attribute.
        """
        self._GSCorrelationFunction.applyShear(*args, **kwargs)
        self._rootps_store = []

    # Also add methods which create a new GSObject with the transformations applied...
    #
    def createTransformed(self, ellipse):
        """Returns a new CorrFunc by applying a galsim.Ellipse transformation (shear, dilate).

        Note that galsim.Ellipse objects can be initialized in a variety of ways (see documentation
        of this class, galsim.ellipse.Ellipse in the doxygen documentation, for details).

        @param ellipse The galsim.Ellipse transformation to apply
        @returns The transformed GSObject.
        """
        if not isinstance(ellipse, galsim.Ellipse):
            raise TypeError("Argument to createTransformed must be a galsim.Ellipse!")
        ret = self.copy()
        ret.applyTransformation(ellipse)
        return ret

    def createScaled(self, scale):
        """Returns a new GSObject by applying a magnification by the given scale, scaling the linear
        size by scale.  

        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`

        @param scale The linear rescaling factor to apply.
        @returns The rescaled GSObject.
        """
        ret = self.copy()
        ret.applyTransformation(galsim.Ellipse(np.log(scale)))
        return ret

    def createRotated(self, theta):
        """Returns a new GSObject by applying a rotation.

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        @returns The rotated GSObject.
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        ret = self.copy()
        ret.applyRotation(theta)
        return ret

    def createSheared(self, *args, **kwargs):
        """Returns a new GSObject by applying a shear, where arguments are either a galsim.Shear or
        keyword arguments that can be used to create one.

        For more details about the allowed keyword arguments, see the documentation of galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).
        """
        ret = self.copy()
        ret.applyShear(*args, **kwargs)
        return ret

    def draw(self, image=None, dx=None, gain=1., wmult=1., add_to_image=False):
        """Draws an Image of the correlation function, with bounds optionally set by an input Image.

        The draw method is used to draw an Image of the correlation function, typically using
        Fourier space convolution and using interpolation to carry out image transformations such as
        shearing.  This method can create a new Image or can draw into an existing one, depending on
        the choice of the `image` keyword parameter.  Other keywords of particular relevance for
        users are those that set the pixel scale for the image (`dx`) and that decide whether the
        clear the input Image before drawing into it (`add_to_image`).

        @param image  If provided, this will be the image on which to draw the profile.
                      If `image = None`, then an automatically-sized image will be created.
                      If `image != None`, but its bounds are undefined (e.g. if it was 
                        constructed with `image = galsim.ImageF()`), then it will be resized
                        appropriately based on the profile's size (default `image = None`).

        @param dx     If provided, use this as the pixel scale for the image.
                      If `dx` is `None` and `image != None`, then take the provided image's pixel 
                        scale.
                      If `dx` is `None` and `image == None`, then use the Nyquist scale 
                        `= pi/maxK()`.
                      If `dx <= 0` (regardless of image), then use the Nyquist scale `= pi/maxK()`.
                      (Default `dx = None`.)

        @param gain   The number of photons per ADU ("analog to digital units", the units of the 
                      numbers output from a CCD).  (Default `gain =  1.`)

        @param wmult  A factor by which to make an automatically-sized image larger than it would 
                      normally be made.  This factor also applies to any intermediate images during
                      Fourier calculations.  The size of the intermediate images are normally 
                      automatically chosen to reach some preset accuracy targets (see 
                      include/galsim/SBProfile.h); however, if you see strange artifacts in the 
                      image, you might try using `wmult > 1`.  This will take longer of 
                      course, but it will produce more accurate images, since they will have 
                      less "folding" in Fourier space. (Default `wmult = 1.`)

        @param add_to_image  Whether to add flux to the existing image rather than clear out
                             anything in the image before drawing.
                             Note: This requires that image be provided (i.e. `image` is not `None`)
                             and that it have defined bounds (default `add_to_image = False`).

        @returns      The drawn image.

        Note: this method uses the .draw() method of GSObject instances, which are themselves used
        to contain the internal representation of the correlation function.
        """
        # Call the GSObject draw method, but set the normalization to the surface brightness as
        # appropriate for these correlation function objects
        return self._GSCorrelationFunction.draw(
            image=image, dx=dx, gain=gain, wmult=wmult, normalization="sb",
            add_to_image=add_to_image)

    def drawShoot(self, image=None, dx=None, gain=1., wmult=1., add_to_image=False, n_photons=0.,
                  rng=None, max_extra_noise=0., poisson_flux=None):
        """Draw an image of the correlation function by shooting individual photons drawn from the
        internal representation of the profile.

        The drawShoot() method is used to draw an image of a correlation function by shooting a
        number of photons to randomly sample the profile. The resulting image will thus have Poisson
        noise due to the finite number of photons shot.  drawShoot() can create a new Image or use
        an existing one, depending on the choice of the `image` keyword parameter.  Other keywords
        of particular relevance for users are those that set the pixel scale for the image (`dx`),
        that choose the normalization convention for the flux (`normalization`), and that decide
        whether the clear the input Image before shooting photons into it (`add_to_image`).

        It is important to remember that the image produced by drawShoot() represents the
        correlation function as convolved with a square image pixel.  So when using drawShoot()
        instead of draw(), you implicitly include an additional pixel response equivalent to
        convolving with a Pixel GSObject.  In other words, whereas draw() samples the correlation
        function at the location of the pixel centre, in the asymptotic limit of large numbers of
        photons drawShoot() returns the average of many samples taken throughout the area of each
        pixel.

        @param image  If provided, this will be the image on which to draw the profile.
                      If `image = None`, then an automatically-sized image will be created.
                      If `image != None`, but its bounds are undefined (e.g. if it was constructed 
                        with `image = galsim.ImageF()`), then it will be resized appropriately base 
                        on the profile's size.
                      (Default `image = None`.)

        @param dx     If provided, use this as the pixel scale for the image.
                      If `dx` is `None` and `image != None`, then take the provided image's pixel 
                        scale.
                      If `dx` is `None` and `image == None`, then use the Nyquist scale 
                        `= pi/maxK()`.
                      If `dx <= 0` (regardless of image), then use the Nyquist scale `= pi/maxK()`.
                      (Default `dx = None`.)

        @param gain   The number of photons per ADU ("analog to digital units", the units of the 
                      numbers output from a CCD).  (Default `gain =  1.`)

        @param wmult  A factor by which to make an automatically-sized image larger than 
                      it would normally be made. (Default `wmult = 1.`)

        @param add_to_image     Whether to add flux to the existing image rather than clear out
                                anything in the image before drawing.
                                Note: This requires that image be provided (i.e. `image != None`)
                                and that it have defined bounds (default `add_to_image = False`).
                              
        @param n_photons        If provided, the number of photons to use.
                                If not provided (i.e. `n_photons = 0`), use as many photons as
                                  necessary to result in an image with the correct Poisson shot 
                                  noise for the object's flux.  For positive definite profiles, this
                                  is equivalent to `n_photons = flux`.  However, some profiles need
                                  more than this because some of the shot photons are negative 
                                  (usually due to interpolants).
                                (Default `n_photons = 0`).

        @param rng              If provided, a random number generator to use for photon shooting.
                                  (may be any kind of `galsim.BaseDeviate` object)
                                If `rng=None`, one will be automatically created, using the time
                                  as a seed.
                                (Default `rng = None`)

        @param max_extra_noise  If provided, the allowed extra noise in each pixel.
                                  This is only relevant if `n_photons=0`, so the number of photons 
                                  is being automatically calculated.  In that case, if the image 
                                  noise is dominated by the sky background, you can get away with 
                                  using fewer shot photons than the full `n_photons = flux`.
                                  Essentially each shot photon can have a `flux > 1`, which 
                                  increases the noise in each pixel.  The `max_extra_noise` 
                                  parameter specifies how much extra noise per pixel is allowed 
                                  because of this approximation.  A typical value for this might be
                                  `max_extra_noise = sky_level / 100` where `sky_level` is the flux
                                  per pixel due to the sky.  If the natural number of photons 
                                  produces less noise than this value for all pixels, we lower the 
                                  number of photons to bring the resultant noise up to this value.
                                  If the natural value produces more noise than this, we accept it 
                                  and just use the natural value.  Note that this uses a "variance"
                                  definition of noise, not a "sigma" definition.
                                (Default `max_extra_noise = 0.`)

        @param poisson_flux     Whether to allow total object flux scaling to vary according to 
                                Poisson statistics for `n_photons` samples (default 
                                `poisson_flux = True` unless n_photons is given, in which case
                                the default is `poisson_flux = False`).

        @returns  The tuple (image, added_flux), where image is the input with drawn photons 
                  added and added_flux is the total flux of photons that landed inside the image 
                  bounds.

        The second part of the return tuple may be useful as a sanity check that you have provided a
        large enough image to catch most of the flux.  For example:
        
            image, added_flux = obj.drawShoot(image)
            assert added_flux > 0.99 * obj.getFlux()
        
        However, the appropriate threshold will depend things like whether you are keeping 
        `poisson_flux = True`, how high the flux is, how big your images are relative to the size of
        your object, etc.

        Note: this method uses the .drawShoot() method of GSObject instances, which are themselves
        used to contain the internal representation of the correlation function.
        """
        return self._GSCorrelationFunction.drawShoot(
            image=image, dx=dx, gain=gain, wmult=wmult, add_to_image=ad_to_image,
            n_photons=n_photons, rng=rng, max_extra_noise=max_extra_noise,
            poisson_flux=poisson_flux)
    
    def drawK(self, re=None, im=None, dk=None, gain=1., wmult=1., add_to_image=False):
        """Draws the k-space Images (real and imaginary parts) of the correlation function, also
        known as the power spectrum, with bounds optionally set by input Images.

        Normalization is always such that re(0,0) = flux.  The imaginary part of the k-space image
        is expected to be vanishingly small.

        @param re     If provided, this will be the real part of the k-space image.
                      If `re = None`, then an automatically-sized image will be created.
                      If `re != None`, but its bounds are undefined (e.g. if it was 
                        constructed with `re = galsim.ImageF()`), then it will be resized
                        appropriately based on the profile's size (default `re = None`).

        @param im     If provided, this will be the imaginary part of the k-space image.
                      A provided im must match the size and scale of re.
                      If `im = None`, then an automatically-sized image will be created.
                      If `im != None`, but its bounds are undefined (e.g. if it was 
                        constructed with `im = galsim.ImageF()`), then it will be resized
                        appropriately based on the profile's size (default `im = None`).

        @param dk     If provided, use this as the pixel scale for the images.
                      If `dk` is `None` and `re, im != None`, then take the provided images' pixel 
                        scale (which must be equal).
                      If `dk` is `None` and `re, im == None`, then use the Nyquist scale 
                        `= pi/maxK()`.
                      If `dk <= 0` (regardless of image), then use the Nyquist scale `= pi/maxK()`.
                      (Default `dk = None`.)

        @param gain   The number of photons per ADU ("analog to digital units", the units of the 
                      numbers output from a CCD).  (Default `gain =  1.`)

        @param wmult  A factor by which to make an automatically-sized image larger than it would 
                      normally be made.  This factor also applies to any intermediate images during
                      Fourier calculations.  The size of the intermediate images are normally 
                      automatically chosen to reach some preset accuracy targets (see 
                      include/galsim/SBProfile.h); however, if you see strange artifacts in the 
                      image, you might try using `wmult > 1`.  This will take longer of 
                      course, but it will produce more accurate images, since they will have 
                      less "folding" in Fourier space. (Default `wmult = 1.`)

        @param add_to_image  Whether to add to the existing images rather than clear out
                             anything in the image before drawing.
                             Note: This requires that images be provided (i.e. `re`, `im` are
                             not `None`) and that they have defined bounds (default 
                             `add_to_image = False`).

        @returns      (re, im)  (created if necessary)

        Note: this method uses the .drawK() method of GSObject instances, which are themselves used
        to contain the internal representation of the correlation function.
        """
        return self._GSCorrelationFunction(
            re=re, im=im, dk=dk, gain=gain, wmult=wmult, add_to_image=add_to_image)

    def applyNoiseTo(self, image, dx=0., dev=None, add_to_image=True):
        """Add noise as a Gaussian random field with this correlation function to an input Image.

        If the optional image pixel scale `dx` is not specified, `image.getScale()` is used for the
        input image pixel separation.
        
        If an optional random deviate `dev` is supplied, the application of noise will share the
        same underlying random number generator when generating the vector of unit variance
        Gaussians that seed the (Gaussian) noise field.

        @param image The input Image object.
        @param dx    The pixel scale to adopt for the input image; should use the same units the
                     ImageCorrFunc instance for which this is a method.  If is specified,
                     `image.getScale()` is used instead.
        @param dev   Optional random deviate from which to draw pseudo-random numbers in generating
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

        # Set up the Gaussian random deviate we will need later
        if dev is None:
            g = galsim.GaussianDeviate()
        else:
            if isinstance(dev, galsim.BaseDeviate):
                g = galsim.GaussianDeviate(dev)
            else:
                raise TypeError(
                    "Supplied input keyword dev must be a galsim.BaseDeviate or derived class "+
                    "(e.g. galsim.UniformDeviate, galsim.GaussianDeviate).")

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
            self.draw(newcf, dx=None) # setting dx=None here uses the newcf image scale set above

            # Roll to put the origin at the lower left pixel before FT-ing to get the PS...
            rolled_cf_array = utilities.roll2d(
                newcf.array, (-newcf.array.shape[0] / 2, -newcf.array.shape[1] / 2))

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

###
# Then we define the ImageCorrFunc, which generates a correlation function by estimating it directly
# from images:
#
class ImageCorrFunc(BaseCorrFunc):
    """A class describing 2D correlation functions calculated from Images.

    This class uses an internally-stored C++ CorrelationFunction object, wrapped using Boost.Python.
    For more details of the CorrelationFunction object, please see the documentation produced by
    doxygen.

    Initialization
    --------------
    An ImageCorrFunc is initialized using an input Image (or ImageView) instance.  The correlation
    function for that image is then calculated from its pixel values using the NumPy FFT functions.
    Optionally, the pixel scale for the input `image` can be specified using the `dx` keyword
    argument. 

    If `dx` is not set the value returned by `image.getScale()` is used unless this is <= 0, in
    which case a scale of 1 is assumed.

    Basic example:

        >>> cf = galsim.correlatednoise.ImageCorrFunc(image)

    Instantiates an ImageCorrFunc using the pixel scale information contained in image.getScale()
    (assumes the scale is unity if image.getScale() <= 0.)

    Optional Inputs
    ---------------

        >>> cf = galsim.correlatednoise.ImageCorrFunc(image, dx=0.2)

    The example above instantiates an ImageCorrFunc, but forces the use of the pixel scale dx to
    set the units of the internal lookup table.

        >>> cf = galsim.correlatednoise.ImageCorrFunc(image,
        ...     interpolant=galsim.InterpolantXY(galsim.Lanczos(5, tol=1.e-4))

    The example above instantiates a ImageCorrFunc, but forces the use of a non-default interpolant
    for interpolation of the internal lookup table.  Must be an InterpolantXY instance or an
    Interpolant instance (if the latter one-dimensional case is supplied an InterpolantXY will be
    automatically generated from it).

    The default interpolant if None is set is a galsim.InterpolantXY(galsim.Linear(tol=1.e-4)),
    which uses bilinear interpolation.  Initial tests indicate the favourable performance of this
    interpolant in applications involving correlated pixel noise.

    Methods
    -------
    The ImageCorrFunc is not a GSObject, but does inherit some of the GSObject methods (draw(),
    drawShoot(), applyShear() etc.) and operator bindings.  Most of these work in the way you would
    intuitively expect, but see the individual docstrings for details.

    However, some methods are purposefully not implemented, e.g. applyShift(), createShifted().
    """
    def __init__(self, image, dx=0., interpolant=None):
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

        # Store local copies of the original image, power spectrum and modified correlation function
        self.original_image = image
        self.original_ps_image = galsim.ImageViewD(np.ascontiguousarray(ps_array))
        self.original_cf_image = galsim.ImageViewD(np.ascontiguousarray(cf_array))

        # Store the original image scale if set
        if dx > 0.:
            self.original_image.setScale(dx)
            self.original_cf_image.setScale(dx)
        elif image.getScale() > 0.:
            self.original_cf_image.setScale(image.getScale())
        else: # sometimes Images are instantiated with scale=0, in which case we will assume unit
              # pixel scale
            self.original_image.setScale(1.)
            self.original_cf_image.setScale(1.)

        # If interpolant not specified on input, use a high-ish polynomial
        if interpolant == None:
            quintic = galsim.Quintic(tol=1.e-4)
            self.interpolant = galsim.InterpolantXY(quintic)
        else:
            if isinstance(interpolant, galsim.Interpolant):
                self.interpolant = galsim.InterpolantXY(interpolant)
            elif isinstance(interpolant, galsim.InterpolantXY):
                self.interpolant = interpolant
            else:
                raise RuntimeError(
                    'Specified interpolant is not an Interpolant or InterpolantXY instance!')

        # Then initialize...
        BaseCorrFunc.__init__(
            self, base.GSObject(
                galsim._galsim._CorrelationFunction(
                    self.original_cf_image, self.interpolant, dx=self.original_image.getScale())))
        
        # Finally store useful data as a (rootps, dx) tuple for efficient later use:
        self._rootps_store.append(
            (np.sqrt(self.original_ps_image.array), self.original_cf_image.getScale()))

    # Make a copy of the ImageCorrFunc
    def copy(self):
        """Returns a copy of the ImageCorrFunc.

        All attributes (e.g. rootps_store, original_image) are copied.
        """
        import copy
        cf = galsim._galsim._CorrelationFunction(self._GSCorrelationFunction.SBProfile)
        ret = BaseCorrFunc(base.GSObject(cf))
        ret.__class__ = self.__class__
        ret._rootps_store = copy.deepcopy(self._rootps_store) # possible due to Jim's image pickling
        ret.original_image = self.original_image.copy()
        ret.original_cf_image = self.original_cf_image.copy()
        ret.original_ps_image = self.original_ps_image.copy()
        ret.interpolant = self.interpolant
        return ret


class AddCorrFunc(BaseCorrFunc):
    """A class for adding two or more correlation functions.

    The AddCorrFunc class is used to represent the sum of multiple correlation functions.

    Methods
    -------
    The AddCorrFunc is not a GSObject, but does inherit some of the GSObject methods (draw(),
    drawShoot(), applyShear() etc.) and operator bindings.  Most of these work in the way you would
    intuitively expect, but see the individual docstrings for details.

    However, some methods are purposefully not implemented, e.g. applyShift(), createShifted().
    """
    
    # --- Public Class methods ---
    def __init__(self, *args):

        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise ValueError(
                "AddCorrFunc must be initialized with at least one BaseCorrFunc or derived class "+
                "instance.")
        elif len(args) == 1:
            # 1 argument.  Should be either a GSObject or a list of GSObjects
            if isinstance(args[0], galsim.BaseCorrFunc):
                CFList = [args[0]._GSCorrelationFunction.SBProfile]
            elif isinstance(args[0], list):
                CFList = []
                for obj in args[0]:
                    if isinstance(obj, galsim.BaseCorrFunc):
                        GSList.append(obj._GSCorrelationFunction.SBProfile)
                    else:
                        raise TypeError("Input list must contain only GSObjects.")
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
            BaseCorrFunc.__init__(self, galsim.GSObject(galsim.SBAdd(GSList)))
        elif len(args) >= 2:
            # >= 2 arguments.  Convert to a list of SBProfiles
            CFList = [obj._GSCorrelationFunction.SBProfile for obj in args]
            BaseCorrFunc.__init__(self, galsim.GSObject(galsim.SBAdd(CFList)))


# Make a function for returning Noise correlations
def _Image_getCorrFunc(image):
    """Returns a CorrFunc instance by calculating the correlation function of image pixels.
    """
    return ImageCorrFunc(image.view())

# Then add this Image method to the Image classes
for Class in galsim.Image.itervalues():
    Class.getCorrFunc = _Image_getCorrFunc

for Class in galsim.ImageView.itervalues():
    Class.getCorrFunc = _Image_getCorrFunc

for Class in galsim.ConstImageView.itervalues():
    Class.getCorrFunc = _Image_getCorrFunc
