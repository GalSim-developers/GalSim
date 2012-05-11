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

def simReal(real_galaxy, target_PSF, target_pixel_scale, g1 = 0.0, g2 = 0.0, rotation_angle = None, 
            rand_rotate = True, target_flux = 1.0):
    """@brief Function to simulate images (no added noise) from real galaxy training data.

    This function takes a RealGalaxy from some training set, and manipulates it as needed to
    simulate a (no-noise-added) image from some lower-resolution telescope.  It thus requires a
    target PSF (which could be an image, or one of our base classes) and a target pixel scale.
    Optionally, the user can specify a rotation angle and a shear.  Or, the user can request
    rotation by a randomly-selected angle.  Finally, they can specify a flux normalization for the
    final image.

    Parameters
    ----------
    @param real_galaxy         The RealGalaxy object to use.
    @param target_PSF          The target PSF, either one of our base classes or an ImageView/Image.
    @param target_pixel_scale  The pixel scale for the final image, in arcsec.
    @param g1                  First component of shear to impose (components defined with respect
                                                                   to pixel coordinates), default 0.
    @param g2                  Second component of shear to impose, default 0.
    @param rotation_angle      Angle by which to rotate the galaxy (must be an Angle instance).
    @param rand_rotate         If true (default) then impose a random rotation on the training
                               galaxy.
    @param target_flux         The target flux in the output galaxy image, default 1.
    """
    # do some checking of arguments
    if not isinstance(real_galaxy, galsim.RealGalaxy):
        raise RuntimeError("Error: simReal requires a RealGalaxy!")
    if isinstance(target_PSF, galsim.Image) or isinstance(target_PSF, galsim.ImageView):
        l5 = galsim.Lanczos(5, True, 1.e-4) # Conserve flux=True and 1.e-4 copied from Shera.py!
        interp2d = galsim.InterpolantXY(l5)
        new_target_PSF = galsim.SBInterpolatedImage(target_PSF.view(), interp2d, dx = target_pixel_scale)
        target_PSF = new_target_PSF
    if not isinstance(target_PSF, galsim.SBProfile):
        raise RuntimeError("Error: target PSF is not an Image, ImageView, or SBProfile!")
    if not isinstance(rotation_angle, galsim.Angle):
        raise RuntimeError("Error: rotation angle is not an Angle instance!")
    if (rotation_angle != None and rand_rotate == True):
        raise RuntimeError("Error: both a random rotation and a specific rotation angle were requested!")
    if (target_pixel_scale < real_galaxy.pixel_scale):
        raise Warning("Warning: requested pixel scale is higher resolution than original!")

    # rotate
    if rotation_angle != None:
        real_galaxy.SBProfile.applyRotation(rotation_angle)
        real_galaxy.PSF.applyRotation(rotation_angle)
    elif rand_rotate == True:
        u = galsim.UniformDeviate()
        rand_angle = galsim.Angle(np.pi*u(), galsim.radians)
        real_galaxy.SBProfile.applyRotation(rand_angle)
        real_galaxy.PSF.applyRotation(rand_angle)

    # set fluxes
    real_galaxy.PSF.setFlux(1.0)
    real_galaxy.SBProfile.setFlux(target_flux)

    # deconvolve
    psf_inv = galsim.Deconvolve(psf)
    deconv = galsim.Convolve([gal, psf_inv])

    # shear
    if (g1 != 0.0 or g2 != 0.0):
        sheared = deconv.createSheared(g1, g2)

    # convolve, resample
    out_gal = galsim.Convolve([sheared, target_PSF])
    out_gal_image = out_gal.draw(dx = target_pixel_scale)

    # return simulated image
    return out_gal_image
