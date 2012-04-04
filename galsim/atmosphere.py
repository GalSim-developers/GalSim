
import numpy as np

import galsim

"""\file atmosphere.py Simple atmospheric PSF generation routines
"""

class GSAdd:
    """Base class for defining the python interface to the SBAdd C++ class.
    """
    def __init__(self, SBAdd):
        self.SBAdd = SBAdd

    def add(self, profile, scale=1.):
        self.SBAdd.add(profile, scale)

    def draw(self, dx=0., wmult=1):
        if type(wmult) != int:
            raise TypeError("Input wmult must be an int")
        if type(dx) != float:
            raise Warning("Input dx not a float, converting...")
            dx = float(dx)
        return self.SBAdd.draw(dx=dx, wmult=wmult)

    # many more methods exposed to python not yet implemented
    

class DoubleGaussian(GSAdd):
    """Double Gaussian, which is the sum of two SBProfile Gaussian profiles, using the GSAdd
    interface to SBProfile's SBAdd.
    """
    def __init__(self, flux1=1., sigma1=1., flux2=1., sigma2=1.):
        GSAdd.__init__(self, galsim.SBAdd(galsim.SBGaussian(flux1, sigma1)))
        GSAdd.add(self, galsim.SBGaussian(flux2, sigma2))

