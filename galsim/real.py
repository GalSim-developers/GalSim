import galsim
import utilities

"""@file real.py @brief Necessary functions for dealing with real galaxies and their catalogs.
"""

class RealGalaxyCatalog(object):
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
    @param file_name   The file containing the catalog.
    @param image_dir   The directory containing the images.
    @param dir         The directory of catalog file (optional)
    @param preload     Whether to preload the header information. (default = False)
    """
    _req_params = { 'file_name' : str , 'image_dir' : str }
    _opt_params = { 'dir' : str, 'preload' : bool }
    _single_params = []

    def __init__(self, file_name, image_dir, dir=None, preload=False):
        import os
        # First build full file_name
        self.file_name = file_name
        if not os.path.isdir(image_dir):
            raise RuntimeError(image_dir+' directory does not exist!')
        self.image_dir = image_dir
        if dir is None: dir = image_dir
        self.file_name = os.path.join(dir,self.file_name)

        import pyfits
        try:
            cat = pyfits.getdata(self.file_name)
            self.nobjects = len(cat) # number of objects in the catalog
            ident = cat.field('ident') # ID for object in the training sample
            # We want to make sure that the ident array contains all strings.
            # Strangely, ident.astype(str) produces a string with each element == '1'.
            # Hence this way of doing the conversion:
            self.ident = [ "%s"%val for val in ident ]
            self.gal_file_name = cat.field('gal_filename') # file containing the galaxy image
            self.PSF_file_name = cat.field('PSF_filename') # file containing the PSF image
            self.gal_hdu = cat.field('gal_hdu') # HDU containing the galaxy image
            self.PSF_hdu = cat.field('PSF_hdu') # HDU containing the PSF image
            self.pixel_scale = cat.field('pixel_scale') # pixel scale for the image (could be different
            # if we have training data from other datasets... let's be general here and make it a vector
            # in case of mixed training set)
            self.mag = cat.field('mag') # apparent magnitude
            self.band = cat.field('band') # bandpass in which apparent mag is measured, e.g., "F814W"
            self.weight = cat.field('weight') # weight factor to account for size-dependent probability
        except Exception, e:
            print e
            raise RuntimeError("Unable to read real galaxy catalog %s."%self.file_name)

        if preload:
            self.preload()
            self.preloaded = True
        else:
            self.preloaded = False

        # eventually I think we'll want information about the training dataset, 
        # i.e. (dataset, ID within dataset)
        # also note: will be adding bits of information, like noise properties and galaxy fit params

    def get_index_for_id(self, id):
        """
        Find which index number corresponds to the value ID in the ident field.
        """
        # Just to be completely consistent, convert id to a string in the same way we
        # did above for the ident array:
        id = "%s"%id
        if id in self.ident:
            return self.ident.index(id)
        else:
            raise ValueError('ID %s not found in list of IDs'%id)

    def preload(self):
        """
        Preload the files into memory.
        
        There are memory implications to this, so we don't do this by default.
        However, it can be a big speedup if memory isn't an issue.  Especially
        if many (or all) of the images are stored in the same file as different HDUs.
        """
        import pyfits
        import os
        self.preloaded = True
        self.loaded_files = {}
        for file_name in self.gal_file_name:
            if file_name not in self.loaded_files:
                full_file_name = os.path.join(self.image_dir,file_name)
                self.loaded_files[file_name] = pyfits.open(full_file_name)
        for file_name in self.PSF_file_name:
            if file_name not in self.loaded_files:
                full_file_name = os.path.join(self.image_dir,file_name)
                self.loaded_files[file_name] = pyfits.open(full_file_name)

    def getGal(self,i):
        """
        Returns the galaxy at index i as an ImageViewD object.
        """
        if i >= len(self.gal_file_name):
            raise IndexError(
                'index %d given to getGal is out of range (0..%d)'%(i,len(self.gal_file_name)-1))
        import pyfits
        import os
        import numpy
        if self.preloaded:
            array = self.loaded_files[self.gal_file_name[i]][self.gal_hdu[i]].data
        else:
            file_name = os.path.join(self.image_dir,self.gal_file_name[i])
            array = pyfits.getdata(file_name,self.gal_hdu[i])
        return galsim.ImageViewD(numpy.ascontiguousarray(array.astype(numpy.float64)))

    def getPSF(self,i):
        """
        Returns the PSF at index i as an ImageViewD object.
        """
        if i >= len(self.PSF_file_name):
            raise IndexError(
                'index %d given to getPSF is out of range (0..%d)'%(i,len(self.PSF_file_name)-1))
        import pyfits
        import os
        import numpy
        if self.preloaded:
            array = self.loaded_files[self.PSF_file_name[i]][self.PSF_hdu[i]].data
        else:
            file_name = os.path.join(self.image_dir,self.PSF_file_name[i])
            array = pyfits.getdata(file_name,self.PSF_hdu[i])
        return galsim.ImageViewD(numpy.ascontiguousarray(array.astype(numpy.float64)))


def simReal(real_galaxy, target_PSF, target_pixel_scale, g1 = 0.0, g2 = 0.0, rotation_angle = None, 
            rand_rotate = True, rng = None, target_flux = 1000.0, image=None):
    """@brief Function to simulate images (no added noise) from real galaxy training data.

    This function takes a RealGalaxy from some training set, and manipulates it as needed to
    simulate a (no-noise-added) image from some lower-resolution telescope.  It thus requires a
    target ePSF (which could be an image, or one of our base classes) that represents all PSF
    components including the pixel response, and a target pixel scale.  The default rotation option
    is to impose a random rotation to make irrelevant any real shears in the galaxy training data
    (optionally, the RNG can be supplied).  This default can be turned off by setting rand_rotate =
    False or by requesting a specific rotation angle using the rotation_angle keyword, in which case
    rand_rotate is ignored.  Optionally, the user can specify a shear (default 0).  Finally, they
    can specify a flux normalization for the final image, default 1000.

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
                               galaxy; this is ignored if rotation_angle is set.
    @param rng                 A random number generator to use for the random rotation angle.
                               (may be any kind of BaseDeviate or None) (Default = None)
    @param target_flux         The target flux in the output galaxy image, default 1000.
    @param image               As with the GSObject.draw function, if an image is provided,
                               then it will be used and returned.
                               If image=None, then an appropriately sized image will be created.
    """
    # do some checking of arguments
    if not isinstance(real_galaxy, galsim.RealGalaxy):
        raise RuntimeError("Error: simReal requires a RealGalaxy!")
    for Class in galsim.Image.itervalues():
        if isinstance(target_PSF, Class):
            lan5 = galsim.Lanczos(5, conserve_flux = True, tol = 1.e-4)
            interp2d = galsim.InterpolantXY(lan5)
            target_PSF = galsim.SBInterpolatedImage(target_PSF.view(), interp2d, dx = target_pixel_scale)
            break
    for Class in galsim.ImageView.itervalues():
        if isinstance(target_PSF, Class):
            lan5 = galsim.Lanczos(5, conserve_flux = True, tol = 1.e-4)
            interp2d = galsim.InterpolantXY(lan5)
            target_PSF = galsim.SBInterpolatedImage(target_PSF, interp2d, dx = target_pixel_scale)
            break
    if isinstance(target_PSF, galsim.GSObject):
        target_PSF = target_PSF.SBProfile
    if not isinstance(target_PSF, galsim.SBProfile):
        raise RuntimeError("Error: target PSF is not an Image, ImageView, SBProfile, or GSObject!")
    if rotation_angle != None and not isinstance(rotation_angle, galsim.Angle):
        raise RuntimeError("Error: specified rotation angle is not an Angle instance!")
    if (target_pixel_scale < real_galaxy.pixel_scale):
        import warnings
        message = "Warning: requested pixel scale is higher resolution than original!"
        warnings.warn(message)
    import math # needed for pi, sqrt below
    g = math.sqrt(g1**2 + g2**2)
    if g > 1:
        raise RuntimeError("Error: requested shear is >1!")

    # make sure target PSF is normalized
    target_PSF.setFlux(1.0)

    real_galaxy_copy = real_galaxy.copy()

    # rotate
    if rotation_angle != None:
        real_galaxy_copy.applyRotation(rotation_angle)
    elif rotation_angle == None and rand_rotate == True:
        if rng == None:
            uniform_deviate = galsim.UniformDeviate()
        elif isinstance(rng,galsim.UniformDeviate):
            uniform_deviate = rng
        elif isinstance(rng,galsim.BaseDeviate):
            uniform_deviate = galsim.UniformDeviate(rng)
        else:
            raise TypeError("The rng provided to drawShoot is not a BaseDeviate")
        rand_angle = galsim.Angle(math.pi*uniform_deviate(), galsim.radians)
        real_galaxy_copy.applyRotation(rand_angle)

    # set fluxes
    real_galaxy_copy.setFlux(target_flux)

    # shear
    if (g1 != 0.0 or g2 != 0.0):
        real_galaxy_copy.applyShear(g1=g1, g2=g2)

    # convolve, resample
    out_gal = galsim.Convolve([real_galaxy_copy, galsim.GSObject(target_PSF)])
    image = out_gal.draw(image=image, dx = target_pixel_scale)

    # return simulated image
    return image
