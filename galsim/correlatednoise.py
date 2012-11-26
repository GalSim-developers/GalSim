"""@file correlatednoise.py

Python layer documentation and functions for handling correlated noise in GalSim.
"""

import numpy as np
from . import galsim

class NoiseCorrFunc(GSObject):
    """A class describing 2D Noise Correlation Functions.
    """

    def __init__(self, noise_image):
        # Build a noise correlation function from the input image of noise
        noise_array = noise_image.array
        # Get the CF using DFTs
        ft_array = np.fft.fft2(noise_array)
        cf_array = (np.fft.ifft2(ft_array * ft_array.conj())).real / float(np.product(np.shape(ft)))
        # Roll CF array to put the centre in image centre.  Remember that numpy stores data [y,x]
        yxroll = ((cf_array.shape[0] - 1) / 2, (cf_array.shape[1] - 1) / 2)
        cf_array = np.utilities.roll2d(cf_array, yxroll)
        cf_image = galsim.ImageViewD(cf_array, xmin=-yxroll[1], ymin=-yxroll[0])

        GSObject.__init__(
            self, galsim.SBNoiseCF())

def Image_getCorrFunc():


