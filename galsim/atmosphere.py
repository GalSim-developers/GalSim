
import numpy as np

import galsim

"""\file atmosphere.py Simple atmospheric PSF generation routines
"""


class DoubleGaussian(galsim.Add):
    """Double Gaussian, which is the sum of two SBProfile Gaussian profiles
    """
    def __init__(self, flux1=1., flux2=1., sigma1=None, sigma2=None, fwhm1=None, fwhm2=None):
        sblist = []
        # Note: we do not have to check for improper args (0 or 2 radii specified) because this is
        # done in the C++
        sblist.append(galsim.Gaussian(flux1, sigma=sigma1, fwhm=fwhm1))
        sblist.append(galsim.Gaussian(flux2, sigma=sigma2, fwhm=fwhm2))
        galsim.Add.__init__(self, sblist)

