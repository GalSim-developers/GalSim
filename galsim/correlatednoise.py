"""@file correlatednoise.py

Python layer documentation and functions for handling correlated noise in GalSim.
"""

import numpy as np
from . import _galsim
from . import base
from . import utilities

class CorrFunc(base.GSObject):
    """A class describing 2D Correlation Functions calculated from Images.
    """

    def __init__(self, image, dx=0.):
        # Build a noise correlation function from the input image, first get the CF using DFTs
        ft_array = np.fft.fft2(image.array)

        # Calculate the power spectrum then correlation function
        ps_array = (ft_array * ft_array.conj()).real
        cf_array = (np.fft.ifft2(ps_array)).real / float(np.product(np.shape(ft_array)))

        # Roll CF array to put the centre in image centre.  Remember that numpy stores data [y,x]
        cf_array = utilities.roll2d(cf_array, (cf_array.shape[0] / 2, cf_array.shape[1] / 2))

        # Store local copies of the the original power spectrum and correlation function
        self.original_ps_image = _galsim.ImageViewD(np.ascontiguousarray(ps_array))
        self.original_cf_image = _galsim.ImageViewD(np.ascontiguousarray(cf_array))

        # Store the original image scale if set
        if dx > 0.:
            self.original_cf_image.setScale(dx)

        # Then initialize...
        # TODO: Decide on best default interpolant, and allow optional setting via kwarg on init
        base.GSObject.__init__(self, _galsim.SBCorrFunc(self.original_cf_image, dx=dx))

    def applyNoiseTo(self, image, dx=0., dev=None):
        """Apply noise with this correlation function to an input Image.

        If the optional image pixel scale `dx` is not specified, `image.getScale()` is used for the
        input image pixel separation.
        
        If an optional random deviate `dev` is supplied, the application of noise will share the
        same underlying random number generator.
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
        if image.array.shape == self.original_ps_image.array.shape:
            if ((dx <= 0. and self.original_cf_image.getScale() == 1.) or
                (dx == self.original_cf_image.getScale())):
                rootps = np.sqrt(self.original_ps_image.array) # Actually we want sqrt(PS)

        # If not, draw the correlation function to the desired size and resolution, then DFT to
        # generate the required array of the square root of the power spectrum
        else:
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

        # Finally generate a random field in Fourier space with the right PS, and inverse DFT back,
        # including factor of sqrt(2) to account for only adding noise to the real component:
        gaussvec = _galsim.ImageD(image.bounds)
        gaussvec.addNoise(g)
        noise_array = np.sqrt(2.) * np.fft.ifft2(gaussvec.array * rootps)
        # Make contiguous and add to the image
        image += _galsim.ImageViewD(np.ascontiguousarray(noise_array.real))
        return image


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
