
import numpy as np

import galsim

"""\file atmosphere.py Simple atmospheric PSF generation routines
"""


class DoubleGaussian(galsim.GSAdd):
    """Double Gaussian, which is the sum of two SBProfile Gaussian profiles, using the GSAdd
    interface to SBProfile's SBAdd.
    """
    def __init__(self, flux1=1., sigma1=1., flux2=1., sigma2=1.):
        galsim.GSAdd.__init__(self, galsim.SBAdd(galsim.SBGaussian(flux1, sigma1)))
        galsim.GSAdd.add(self, galsim.SBGaussian(flux2, sigma2))

