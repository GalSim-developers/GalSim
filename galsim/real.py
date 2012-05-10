from . import _galsim

"""@file real.py @brief Necessary functions for dealing with real galaxies and their catalogs.
"""

class RealGalaxyCatalog:
    """@brief Class containing a catalog with information about real galaxy training data.

    The RealGalaxyCatalog class reads in and stores information about a specific training sample of
    realistic galaxies. We assume that all files containing the images (galaxies and PSFs) live in
    one directory; they could be individual files, or multiple HDUs of the same file.  Currently
    there is no functionality that lets this be a FITS data cube, because we assume that the object
    postage stamps will in general need to be different sizes depending on the galaxy size.  For
    example, if the catalog is called 'catalog.fits' and is in the working directory, and the images
    are in a subdirectory called 'images', then the RealGalaxyCatalog can be read in as follows:

    my_rgc = galsim.RealGalaxyCatalog('./catalog.fits','images')

    To explore for the future: scaling with number of galaxies, adding more information as needed,
    and other i/o related issues.

    Parameters
    ----------
    @param filename   The file containing the catalog (including full path).
    @param imagedir   The directory containing the images.
    """
    def __init__(self, filename, imagedir):
        import pyfits
        cat = pyfits.getdata(filename)
        self.filename = filename # store the filename from which the catalog was read
        self.imagedir = imagedir # store the directory containing all image files (gal, PSF)
        self.n = len(cat) # number of objects in the catalog
        self.gal_filename = cat.field('gal_filename') # file containing the galaxy image
        self.PSF_filename = cat.field('PSF_filename') # file containing the PSF image
        self.gal_hdu = cat.field('gal_hdu') # HDU containing the galaxy image
        self.PSF_hdu = cat.field('PSF_hdu') # HDU containing the PSF image
        self.pixel_scale = cat.field('pixel_scale') # pixel scale for the image (could be different
        # if we have training data from other datasets... let's be general here and make it a vector
        # in case of mixed training set)
        self.mag = cat.field('mag') # apparent magnitude
        self.band = cat.field('band') # bandpass in which apparent mag is measured, e.g., "F814W"
        self.weight = cat.field('weight') # weight factor to account for size-dependent probability
        # of galaxy inclusion in training sample

        ## eventually I think we'll want information about the training dataset, i.e. (dataset, ID within dataset)

        # note: am assuming that pyfits takes care of error handling, e.g., if the file does not
        # exist, there's no field with that name, etc.
        # also note: will be adding bits of information, like noise properties and galaxy fit params

def SimReal(real_galaxy, target_PSF, target_pixel_scale, g1 = 0.0, g2 = 0.0, rand_rotate = True,
            target_flux = 1.0):
    # put some documentation here and above for RealGalaxyCatalog
    # do some checking of arguments and so on
    # check SHERA code for some more keywords that might be useful

    # rotate
    # deconvolve
    # shear
    # convolve, resample
    # return simulated image

