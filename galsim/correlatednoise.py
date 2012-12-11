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

    Examples:

    >>> cf = galsim.CorrFunc(image)

    >>> cf = galsim.CorrFunc(image, dx=0.2)

    Methods
    -------
    The CorrFunc is a GSObject, and inherits most of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.  

    However, some methods are purposefully not implemented, e.g. applyShift(), createShifted().
    """
    def __init__(self, image, dx=0., interpolant=None):
        # Build a noise correlation function from the input image, first get the CF using DFTs
        ft_array = np.fft.fft2(image.array)

        # Calculate the power spectrum then correlation function
        ps_array = np.abs(ft_array * ft_array.conj())
        cf_array = (np.fft.ifft2(ps_array)).real / float(np.product(np.shape(ft_array)))

        # Roll CF array to put the centre in image centre.  Remember that numpy stores data [y,x]
        cf_array = utilities.roll2d(cf_array, (cf_array.shape[0] / 2, cf_array.shape[1] / 2))

        # Store local copies of the the original image, power spectrum and correlation function
        self.original_image = image
        self.original_ps_image = _galsim.ImageViewD(np.ascontiguousarray(ps_array))
        self.original_cf_image = _galsim.ImageViewD(np.ascontiguousarray(cf_array))

        # Store the original image scale if set
        if dx > 0.:
            self.original_image.setScale(dx)
            self.original_cf_image.setScale(dx)
        elif image.getScale > 0.:
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
            if isinstance(interpolant, _galsim.InterpolantXY) is False:
                raise RuntimeError('Specified interpolant is not an InterpolantXY!')
            self.interpolant = interpolant

        # Setup a store for later, containing array representations of the sqrt(PowerSpectrum)
        # [useful for later applying noise to images according to this correlation function].
        # Stores data as (rootps, dx) tuples.
        self.rootps_store = [
            (np.sqrt(self.original_ps_image.array), self.original_cf_image.getScale())]

        # Then initialize...
        # TODO: Decide on best default interpolant, and allow optional setting via kwarg on init
        base.GSObject.__init__(
            self, _galsim.SBCorrFunc(self.original_cf_image, self.interpolant, dx=dx))

    def applyNoiseTo(self, image, dx=0., dev=None):
        """Apply noise as a Gaussian random field with this correlation function to an input Image.

        If the optional image pixel scale `dx` is not specified, `image.getScale()` is used for the
        input image pixel separation.
        
        If an optional random deviate `dev` is supplied, the application of noise will share the
        same underlying random number generator when generating the vector of unit variance
        Gaussians that seed the (Gaussian) noise field.
        """
        # Note that this uses the (fast) method of going via the power spectrum and FFTs to generate        # noise according to the correlation function represented by this instance.  An alternative
        # would be to use the covariance matrices and eigendecomposition.  However, although the
        # latter is necessary for whitening, it is an O(N^6) operations for an NxN image!
        # FFT-based noise realization is O(2 N^2 log[N]) so we use it for this simpler (non-
        # whitening) application.

        # Set up the Gaussian random deviate we will need later
        if dev is None:
            g = _galsim.GaussianDeviate()
        else:
            g = _galsim.GaussianDeviate(dev)

        # Then retrieve or redraw the sqrt(power spectrum) needed for making the noise field:

        # First check whether we can just use the stored power spectrum (no drawing necessary if so)
        use_stored = False
        for rootps_array, scale in self.rootps_store:
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
            self.draw(newcf, dx=None) # setting dx=None here uses the newcf image scale set above
            # Roll to put the origin at the lower left pixel before FT-ing to get the PS...
            rolled_cf_array = utilities.roll2d(
                newcf.array, (-newcf.array.shape[0] / 2, -newcf.array.shape[1] / 2))
            rootps = np.sqrt(np.abs(np.fft.fft2(rolled_cf_array)) * np.product(image.array.shape))
            # Then add this and the relevant scale to the rootps_store for later use
            self.rootps_store.append((rootps, newcf.getScale()))

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

        Currently, the interpolant attribute is not being copied over, and getting this fixed is an
        action item.  All other attributes (e.g. rootps_store, original_image) are copyable and
        copied.
        """
        sbp = self.SBProfile.__class__(self.SBProfile)
        ret = base.GSObject(sbp)
        ret.__class__ = self.__class__
        ret.original_image = self.original_image.copy()
        ret.original_cf_image = self.original_cf_image.copy()
        ret.original_ps_image = self.original_ps_image.copy()
        import copy
        #TODO: Currently we cannot copy the interpolant, ask Jim
        #ret.interpolant = _galsim.InterpolantXY(self.interpolant)
        ret.rootps_store = copy.deepcopy(self.rootps_store) # possible thanks to Jim's pickling work
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
        after transformation, such as .applyNoiseTo().

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        """
        base.GSObject.applyRotation(self, theta)
        self.__class__ = CorrFunc

    def applyShear(self, *args, **kwargs):
        """Apply a shear to this object, where arguments are either a galsim.Shear, or arguments
        that will be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).

        After this call, the caller's type will still be a CorrFunc, unlike in the GSObject base
        class implementation of this method.  This is to allow CorrFunc methods to be available
        after transformation, such as .applyNoiseTo().
        """
        base.GSObject.applyShear(self, *args, **kwargs)
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
        after transformation, such as .applyNoiseTo().

        @param scale The linear rescaling factor to apply.
        """
        base.GSObject.applyDilation(self, scale)
        self.__class = CorrFunc

    def applyMagnification(self, scale):
        """"Apply a magnification by the given scale, scaling the linear size by scale and the flux 
        by scale^2.  
        
        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`

        This operation preserves surface brightness, which means that the flux is scales 
        with the change in area.  
        See applyDilation for a version that preserves flux.

        After this call, the caller's type will still be a CorrFunc, unlike in the GSObject base
        class implementation of this method.  This is to allow CorrFunc methods to be available
        after transformation, such as .applyNoiseTo().

        @param scale The linear rescaling factor to apply.
        """
        base.GSObject.applyMagnification(self, scale)
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
        after transformation, such as .applyNoiseTo().

        @param ellipse The galsim.Ellipse transformation to apply
        """
        base.GSObject.applyTransformation(self, ellipse)
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
