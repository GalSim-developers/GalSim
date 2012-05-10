from . import _galsim
import pyfits

"""file @real.py @brief Necessary functions for dealing with real galaxies and their catalogs.

The RealGalaxyCatalog class can be used to represent a catalog containing information about real
galaxies. 

There will also be a function that can manipulate a RealGalaxy to simulate some data with a given
shear and target PSF, a la SHERA - tentative name SimReal, to be fleshed out later
"""

class RealGalaxyCatalog:
    """Class containing a catalog with information about real galaxy training data
    """
    def __init__(self, filename, imagedir):
        cat = pyfits.open(filename)
        self.filename = filename # store the filename from which the catalog was read
        self.imagedir = imagedir # store the directory containing all image files (gal, PSF)
        self.n = len(cat) # number of objects in the catalog
        self.gal_filename = cat.field('gal_filename') # file containing the galaxy image
        self.PSF_filename = cat.field('PSF_filename') # file containing the PSF image
        self.gal_hdu = cat.field('gal_hdu') # HDU containing the galaxy image
        self.PSF_hdu = cat.field('PSF_hdu') # HDU containing the PSF image
        self.pixel_scale = cat.field('pixel_scale') # pixel scale for the image (could be different
        # if we have training data from other datasets... let's be general here and make it a vector)
        self.mag = cat.field('mag') # apparent magnitude
        self.band = cat.field('band') # bandpass in which apparent mag is measured, e.g., "F814W"
        self.weight = cat.field('weight') # weight factor to account for size-dependent probability
        # of galaxy inclusion in training sample

        # note: am assuming that pyfits takes care of error handling, e.g., if the file does not
        # exist, there's no field with that name, etc.
        # also note: will be adding bits of information, like noise properties and galaxy fit params
