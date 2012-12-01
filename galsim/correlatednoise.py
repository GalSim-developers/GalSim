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

    def __init__(self, image):
        # Build a noise correlation function from the input image, first get the CF using DFTs
        ft_array = np.fft.fft2(image.array)
        cf_array = ((np.fft.ifft2(ft_array * ft_array.conj())).real /
            float(np.product(np.shape(ft_array))))
        # Roll CF array to put the centre in image centre.  Remember that numpy stores data [y,x]
        yxroll = (cf_array.shape[0] / 2, cf_array.shape[1] / 2)
        cf_array = np.ascontiguousarray(utilities.roll2d(cf_array, yxroll))
        self.cf_image = _galsim.ImageViewD(cf_array)
        base.GSObject.__init__(self, _galsim.SBNoiseCF(self.cf_image))

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
