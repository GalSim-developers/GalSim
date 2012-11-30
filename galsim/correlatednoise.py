"""@file correlatednoise.py

Python layer documentation and functions for handling correlated noise in GalSim.
"""

import numpy as np
from . import galsim
from . import utilities

class NoiseCF(galsim.GSObject):
    """A class describing 2D Noise Correlation Functions.
    """

    def __init__(self, noise_image):
        # Build a noise correlation function from the input image of noise
        noise_array = noise_image.array
        # Get the CF using DFTs
        ft_array = np.fft.fft2(noise_array)
        cf_array = (np.fft.ifft2(ft_array * ft_array.conj())).real / \
            float(np.product(np.shape(ft_array)))
        # Roll CF array to put the centre in image centre.  Remember that numpy stores data [y,x]
        yxroll = (cf_array.shape[0] / 2, cf_array.shape[1] / 2)
        self.cf_array = np.ascontiguousarray(utilities.roll2d(cf_array, yxroll))
        self.cf_image = galsim.ImageViewD(cf_array, xmin=-yxroll[1], ymin=-yxroll[0])
        
#def Image_getNoiseCF():


