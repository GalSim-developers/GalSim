"""@file correlatednoise.py

Python layer documentation and functions for handling correlated noise in GalSim.
"""

import numpy as np
from . import _galsim
from . import base
from . import utilities

class CorrFunc(base.GSObject):
    """A class describing 2D Correlation Functions calculated from Images.

    Has an SBCorrFunc in the SBProfile attribute.  For more details of the SBCorrFunc object, please
    see the documentation produced by doxygen.

    Initialization
    --------------
    A CorrFunc is initialized using an input Image (or ImageView) instance.  The correlation
    function for that image is then calculated from its pixel values using the NumPy FFT functions.
    Optionally, the pixel scale for the input `image` can be specified using the `dx` keyword
    argument. 

    If `dx` is not set the value returned by `image.getScale()` is used unless this is <= 0, in
    which case a scale of 1 is assumed.

    Basic example:

        >>> cf = galsim.correlatednoise.CorrFunc(image)

    Instantiates a CorrFunc using the pixel scale information contained in image.getScale()
    (assumes the scale is unity if image.getScale() <= 0.)

    Optional Inputs
    ---------------

        >>> cf = galsim.correlatednoise.CorrFunc(image, dx=0.2)

    The example above instantiates a CorrFunc, but forces the use of the pixel scale dx to set the
    units of the internal lookup table.

        >>> cf = galsim.correlatednoise.CorrFunc(image,
        ...     interpolant=galsim.InterpolantXY(galsim.Lanczos(5, tol=1.e-4))

    The example above instantiates a CorrFunc, but forces the use of a non-default interpolant for
    interpolation of the internal lookup table.  Must be an InterpolantXY instance or an Interpolant
    instance (if the latter one-dimensional case is supplied an InterpolantXY will be automatically
    generated from it).

    The default interpolant if None is set is a galsim.InterpolantXY(galsim.Quintic(tol=1.e-4)).

    Methods
    -------
    The CorrFunc is a GSObject, and inherits most of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.  

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
        self.original_ps_image = _galsim.ImageViewD(np.ascontiguousarray(ps_array))
        self.original_cf_image = _galsim.ImageViewD(np.ascontiguousarray(cf_array))

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
            quintic = _galsim.Quintic(tol=1.e-4)
            self.interpolant = _galsim.InterpolantXY(quintic)
        else:
            if isinstance(interpolant, _galsim.Interpolant):
                self.interpolant = galsim.InterpolantXY(interpolant)
            elif isinstance(interpolant, _galsim.InterpolantXY):
                self.interpolant = interpolant
            else:
                raise RuntimeError(
                    'Specified interpolant is not an Interpolant or InterpolantXY instance!')

        # Setup a store for later, containing array representations of the sqrt(PowerSpectrum)
        # [useful for later applying noise to images according to this correlation function].
        # Stores data as (rootps, dx) tuples.
        self._rootps_store = [
            (np.sqrt(self.original_ps_image.array), self.original_cf_image.getScale())]

            #test = np.ones(image.array.shape)
            #testim = _galsim.ImageViewD(test)

        # Then initialize...
        base.GSObject.__init__(
            self, _galsim.SBCorrFunc(self.original_cf_image, self.interpolant, dx=self.original_image.getScale()))

    def applyNoiseTo(self, image, dx=0., dev=None):
        """Add noise as a Gaussian random field with this correlation function to an input Image.

        If the optional image pixel scale `dx` is not specified, `image.getScale()` is used for the
        input image pixel separation.
        
        If an optional random deviate `dev` is supplied, the application of noise will share the
        same underlying random number generator when generating the vector of unit variance
        Gaussians that seed the (Gaussian) noise field.

        @param image The input Image object.
        @param dx    The pixel scale to adopt for the input image; should use the same units the
                     CorrFunc instance for which this is a method.  If is specified,
                     `image.getScale()` is used instead.
        @param dev   Optional random deviate from which to draw pseudo-random numbers in generating
                     the noise field.
        """
        # Note that this uses the (fast) method of going via the power spectrum and FFTs to generate
        # noise according to the correlation function represented by this instance.  An alternative
        # would be to use the covariance matrices and eigendecomposition.  However, it is O(N^6)
        # operations for an NxN image!  FFT-based noise realization is O(2 N^2 log[N]) so we use it
        # for noise generation applications.

        # Set up the Gaussian random deviate we will need later
        if dev is None:
            g = _galsim.GaussianDeviate()
        else:
            if isinstance(dev, _galsim.BaseDeviate):
                g = _galsim.GaussianDeviate(dev)
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
            newcf = _galsim.ImageD(image.bounds) # set the correlation func to be the correct size
            # set the scale based on dx...
            if dx <= 0.:
                if image.getScale() > 0.:
                    newcf.setScale(image.getScale())
                else:
                    newcf.setScale(1.) # sometimes new Images have getScale() = 0
            else:
                newcf.setScale(dx)
            # Then draw this correlation function into an array
            self.draw(newcf, dx=None, normalization="sb") # setting dx=None here uses the newcf
                                                          # image scale set above
            # Roll to put the origin at the lower left pixel before FT-ing to get the PS...
            rolled_cf_array = utilities.roll2d(
                newcf.array, (-newcf.array.shape[0] / 2, -newcf.array.shape[1] / 2))
            # DEBUGGING UNDERSTANDING:
            #import matplotlib.pyplot as plt
            #plt.figure()
            # plot this stored, internal representation of the discrete CF which will be used
            # to generate the requested noise field
            #plt.pcolor(newcf.array[240-15:240+15, 240-15:240+15], vmax=1.1, vmin=-.1)
            #plt.colorbar()
            #plt.title("CF [internal]")
            #plt.savefig('cf_internal.png')

            # Then calculate the sqrt(PS) that will be used to generate the actual noise
            rootps = np.sqrt(np.abs(np.fft.fft2(newcf.array)) * np.product(image.array.shape))

            #plt.figure()
            #plt.pcolor(np.log10(rootps**2), vmax=8, vmin=0.)
            #plt.colorbar()
            # Print some useful info then plot
            #print 'Mean PS [internal]  = {:f}'.format(
            #    np.mean(rootps**2) / np.product(image.array.shape))
            #plt.title("log10(PS) [internal]")
            #plt.savefig('logps_internal.png')

            # Then add this and the relevant scale to the _rootps_store for later use
            self._rootps_store.append((rootps, newcf.getScale()))

        # Finally generate a random field in Fourier space with the right PS, and inverse DFT back,
        # including factor of sqrt(2) to account for only adding noise to the real component:
        gaussvec = _galsim.ImageD(image.bounds)
        gaussvec.addNoise(g)
        noise_array = np.sqrt(2.) * np.fft.ifft2(gaussvec.array * rootps)
        # Make contiguous and add to the image
        image += _galsim.ImageViewD(np.ascontiguousarray(noise_array.real))
        return image

    # Make a copy of the CorrFunc
    def copy(self):
        """Returns a copy of the CorrFunc.

        All attributes (e.g. rootps_store, original_image) are copied.
        """
        import copy
        sbp = self.SBProfile.__class__(self.SBProfile)
        ret = base.GSObject(sbp)
        ret.__class__ = self.__class__
        ret.original_image = self.original_image.copy()
        ret.original_cf_image = self.original_cf_image.copy()
        ret.original_ps_image = self.original_ps_image.copy()
        ret.interpolant = self.interpolant
        ret._rootps_store = copy.deepcopy(self._rootps_store) # possible due to Jim's image pickling
        return ret

    def applyShift(self):
        """The applyShift() method is not available for the CorrFunc.
        """
        raise NotImplementedError("applyShift() not available for CorrFunc objects.")

    def createShifted(self):
        """The createShifted() method is not available for the CorrFunc.
        """
        raise NotImplementedError("createShifted() not available for CorrFunc objects.")

    def applyRotation(self, theta):
        """Apply a rotation theta to this object.
           
        After this call, the caller's type will still be a CorrFunc, unlike in the GSObject base
        class implementation of this method.  This is to allow CorrFunc methods to be available
        after transformation, such as .applyNoiseTo().  Resets the internal _rootps_store.

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        """
        base.GSObject.applyRotation(self, theta)
        self._rootps_store = []
        self.__class__ = CorrFunc

    def applyShear(self, *args, **kwargs):
        """Apply a shear to this object, where arguments are either a galsim.Shear, or arguments
        that will be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).

        After this call, the caller's type will still be a CorrFunc, unlike in the GSObject base
        class implementation of this method.  This is to allow CorrFunc methods to be available
        after transformation, such as .applyNoiseTo().  Resets the internal _rootps_store.
        """
        base.GSObject.applyShear(self, *args, **kwargs)
        self._rootps_store = []
        self.__class__ = CorrFunc

    def applyDilation(self, scale):
        """Apply a dilation of the linear size by the given scale.

        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`

        This operation preserves flux.
        See applyMagnification() for a version that preserves surface brightness, and thus 
        changes the flux.

        After this call, the caller's type will still be a CorrFunc, unlike in the GSObject base
        class implementation of this method.  This is to allow CorrFunc methods to be available
        after transformation, such as .applyNoiseTo().  Resets the internal _rootps_store.

        @param scale The linear rescaling factor to apply.
        """
        base.GSObject.applyDilation(self, scale)
        self._rootps_store = []
        self.__class = CorrFunc

    def applyMagnification(self, scale):
        """"Apply a magnification by the given scale, scaling the linear size by scale and the flux 
        by scale^2.  
        
        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`

        This operation preserves surface brightness, which means that the flux scales 
        with the change in area.  
        See applyDilation for a version that preserves flux.

        After this call, the caller's type will still be a CorrFunc, unlike in the GSObject base
        class implementation of this method.  This is to allow CorrFunc methods to be available
        after transformation, such as .applyNoiseTo().  Resets the internal _rootps_store.

        @param scale The linear rescaling factor to apply.
        """
        base.GSObject.applyMagnification(self, scale)
        self._rootps_store = []
        self.__class__ = CorrFunc

    def applyTransformation(self, ellipse):
        """Apply a galsim.Ellipse distortion to this object.
           
        galsim.Ellipse objects can be initialized in a variety of ways (see documentation of this
        class, galsim.ellipse.Ellipse in the doxygen documentation, for details).

        Note: if the ellipse includes a dilation, then this transformation will not be
        flux-conserving.  It conserves surface brightness instead.  Thus, the flux will increase by
        the increase in area = dilation^2.

        After this call, the caller's type will still be a CorrFunc, unlike in the GSObject base
        class implementation of this method.  This is to allow CorrFunc methods to be available
        after transformation, such as .applyNoiseTo().  Resets the internal _rootps_store.

        @param ellipse The galsim.Ellipse transformation to apply
        """
        base.GSObject.applyTransformation(self, ellipse)
        self._rootps_store = []
        self.__class__ = CorrFunc


# Make a function for returning Noise correlation
def Image_getCorrFunc(image):
    """Returns a CorrFunc instance by calculating the correlation function of image pixels.
    """
    return CorrFunc(image.view())

# Then add this Image method to the Image classes
for Class in _galsim.Image.itervalues():
    Class.getCorrFunc = Image_getCorrFunc

for Class in _galsim.ImageView.itervalues():
    Class.getCorrFunc = Image_getCorrFunc

for Class in _galsim.ConstImageView.itervalues():
    Class.getCorrFunc = Image_getCorrFunc
