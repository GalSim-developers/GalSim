
import numpy as np

import galsim

"""\file atmosphere.py Simple atmospheric PSF generation routines
"""

def gaussian(flux=1.0, size=1.0):
    """Initialize a Gaussian PSF. In this case simply return an SBGaussian object with given flux 
    and size.
    """
    return galsim.SBGaussian(flux, size)

def moffat(beta, truncationFWHM=2.0, flux=1.0, re=1.0):
    """Initialize a Moffat PSF. In this case simply return an SBMoffat object with exponent beta,
    flux, and effective radius re. The Moffat profile is truncated at truncationFWHM times the 
    FWHM.
    """
    return galsim.SBMoffat(beta, truncationFWHM, flux, re)
