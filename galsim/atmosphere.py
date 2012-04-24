
import numpy as np

import galsim

"""\file atmosphere.py Simple atmospheric PSF generation routines
"""


class DoubleGaussian(galsim.Add):
    """Double Gaussian, which is the sum of two SBProfile Gaussian profiles
    """
    def __init__(self, flux1=1., sigma1=1., flux2=1., sigma2=1.):
        sblist = []
        sblist.append(galsim.Gaussian(flux1, sigma=sigma1))
        sblist.append(galsim.Gaussian(flux2, sigma=sigma2))
        galsim.Add.__init__(self, sblist)

