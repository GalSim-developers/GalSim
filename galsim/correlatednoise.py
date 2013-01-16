"""@file correlatednoise.py

Python layer documentation and functions for handling correlated noise in GalSim.
"""

import numpy as np
import galsim
from . import base
from . import utilities

class CorrFunc(object):
    """A class describing 2D correlation functions, typically calculated from Images.

    Initialization
    --------------
    The constructor for CorrFunc takes a GSObject, which is then reinterpreted as the 
    correlation function.  However, the typical way to create a CorrFunc is with the 
    factory function ImageCorrFunc.  See that function for more info.

    Members
    -------
    The most useful member is `profile`, which is the internally stored GSObject profile.
    It can be manipulated using the various methods of GSObject.  So the following are
    all legal:
    
        cf.profile.applyShear(s)
        cf.profile = cf1.profile + cf2.profile
        cf.profile /= 2
        cf.profile.draw(im,dx)

    Methods
    -------
    The main way that CorrFunc is used is to add correlated noise to an image.  This is 
    done with

        cf.addNoiseTo(im)
    """
    def __init__(self, gsobject):
        if not isinstance(gsobject, base.GSObject):
            raise TypeError(
                "Correlation function objects must be initialized with a GSObject.")
        
        # Act as a container for the GSObject used to represent the correlation funcion.
        self.profile = gsobject

        # When applying noise to an image, we normally do a calculation. 
        # If store_profile is profile, then it means we can use the other stored values
        # and avoid having to redo the calculation.
        # So for now, we start out with store_profile = None.
        self.profile_for_stored = None

    def copy(self):
        """Returns a copy of the correlation function.
        """
        return CorrFunc(self.profile.copy())

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

        # If the profile has changed since last time (or if we have never been here before),
        # clear out the stored values.
        if self.profile_for_stored is not self.profile:
            self._rootps_store = []
        # Set profile_for_stored for next time.
        self.profile_for_stored = self.profile

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
            self.profile.draw(newcf, dx=None) # setting dx=None here uses the newcf image scale set above

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
def ImageCorrFunc(image, dx=0., interpolant=None):
    """A factory function for making a CorrFunc from an image:

    The correlation function is calculated from its pixel values using the NumPy FFT functions.
    Optionally, the pixel scale for the input `image` can be specified using the `dx` keyword
    argument. 

    If `dx` is not set the value returned by `image.getScale()` is used unless this is <= 0, in
    which case a scale of 1 is assumed.

    Basic example:

        >>> cf = galsim.ImageCorrFunc(image)

    Instantiates an ImageCorrFunc using the pixel scale information contained in image.getScale()
    (assumes the scale is unity if image.getScale() <= 0.)

    Optional Inputs
    ---------------

        >>> cf = galsim.ImageCorrFunc(image, dx=0.2)

    The example above instantiates an ImageCorrFunc, but forces the use of the pixel scale dx to
    set the units of the internal lookup table.

        >>> cf = galsim.ImageCorrFunc(image,
        ...     interpolant=galsim.InterpolantXY(galsim.Lanczos(5, tol=1.e-4))

    The example above instantiates a ImageCorrFunc, but forces the use of a non-default interpolant
    for interpolation of the internal lookup table.  Must be an InterpolantXY instance or an
    Interpolant instance (if the latter one-dimensional case is supplied an InterpolantXY will be
    automatically generated from it).

    The default interpolant if None is set is a galsim.InterpolantXY(galsim.Linear(tol=1.e-4)),
    which uses bilinear interpolation.  Initial tests indicate the favourable performance of this
    interpolant in applications involving correlated pixel noise.
    """
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
    original_image = image.view() # Makes a new object without copying data values
    original_ps_image = galsim.ImageViewD(np.ascontiguousarray(ps_array))
    original_cf_image = galsim.ImageViewD(np.ascontiguousarray(cf_array))

    # Store the original image scale if set
    if dx > 0.:
        original_image.setScale(dx)
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
    profile = base.InterpolatedImage(original_cf_image, interpolant, dx=original_image.getScale())
    ret = CorrFunc(profile)
        
    # Finally store useful data as a (rootps, dx) tuple for efficient later use:
    ret.profile_for_stored = profile
    ret._rootps_store = []
    ret._rootps_store.append(
        (np.sqrt(original_ps_image.array), original_cf_image.getScale()))

    return ret

# Make a function for returning Noise correlations
def _Image_getCorrFunc(image):
    """Returns a CorrFunc instance by calculating the correlation function of image pixels.
    """
    return ImageCorrFunc(image)

# Then add this Image method to the Image classes
for Class in galsim.Image.itervalues():
    Class.getCorrFunc = _Image_getCorrFunc

for Class in galsim.ImageView.itervalues():
    Class.getCorrFunc = _Image_getCorrFunc

for Class in galsim.ConstImageView.itervalues():
    Class.getCorrFunc = _Image_getCorrFunc
