
import numpy as np

import galsim

"""\file atmosphere.py Simple atmospheric PSF generation routines
"""


class DoubleGaussian(galsim.Add):
    """Double Gaussian, which is the sum of two SBProfile Gaussian profiles
    """
    def __init__(self, flux1=1., flux2=1., sigma1=None, sigma2=None, fwhm1=None, fwhm2=None):
        sblist = []
        if fwhm1!=None and sigma1!=None:
            raise RuntimeError("Both FWHM and sigma specified for first Gaussian!")
        if fwhm2!=None and sigma2!=None:
            raise RuntimeError("Both FWHM and sigma specified for first Gaussian!")
        if fwhm1==None:
            sblist.append(galsim.Gaussian(flux1, sigma=sigma1))
        else:
            sblist.append(galsim.Gaussian(flux1, fwhm=fwhm1))
        if fwhm2==None:
            sblist.append(galsim.Gaussian(flux2, sigma=sigma2))
        else:
            sblist.append(galsim.Gaussian(flux2, fwhm=fwhm2))
        galsim.Add.__init__(self, sblist)

