# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file real.py
Functions for dealing with galsim.RealGalaxy objects and the catalogs that store their data.

The galsim.RealGalaxy uses images of galaxies from real astrophysical data (e.g. the Hubble Space
Telescope), along with a PSF model of the optical properties of the telescope that took these
images, to simulate new galaxy images with a different (must be larger) telescope PSF.  A 
description of the simulation method can be found in Section 5 of Mandelbaum et al. (2012; MNRAS, 
540, 1518), although note that the details of the implementation in Section 7 of that work are not 
relevant to the more recent software used here.

This module defines the RealGalaxyCatalog class, used to store all required information about a
real galaxy simulation training sample and accompanying PSF model.  For information about 
downloading GalSim-readable RealGalaxyCatalog data in FITS format, see the RealGalaxy Data Download
page on the GalSim Wiki: 
https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data%20Download%20Page

The function simReal takes this information and uses it to simulate a (no-noise-added) image from 
some lower-resolution telescope.
"""


import galsim
import utilities


class RealGalaxyCatalog(object):
    """Class containing a catalog with information about real galaxy training data.

    The RealGalaxyCatalog class reads in and stores information about a specific training sample of
    realistic galaxies. We assume that all files containing the images (galaxies and PSFs) live in
    one directory; they could be individual files, or multiple HDUs of the same file.  Currently
    there is no functionality that lets this be a FITS data cube, because we assume that the object
    postage stamps will in general need to be different sizes depending on the galaxy size.  

    If only the catalog name (`'real_galaxy_catalog.fits'`) is specified, then the set of galaxy/PSF
    image files (e.g., `'real_galaxy_images_1.fits'`, `'real_galaxy_PSF_images_1.fits'`, etc.) are
    assumed to be in the directory as the catalog file (in the following example, in the current 
    working directory `./`):

        >>> my_rgc = galsim.RealGalaxyCatalog('real_galaxy_catalog.fits')

    If `image_dir` is specified, the set of galaxy/PSF image files is assumed to be in the
    subdirectory of where the catalog is (in the following example, `./images`):

        >>> my_rgc = galsim.RealGalaxyCatalog('real_galaxy_catalog.fits', image_dir='images')

    If the real galaxy catalog is in some far-flung directory, and the galaxy/PSF image files are in 
    its subdirectory, one only needs to specify the long directory name once:

        >>> file_name = '/data3/scratch/user_name/galsim/real_galaxy_data/real_galaxy_catalog.fits'
        >>> image_dir = 'images'
        >>> my_rgc = galsim.RealGalaxyCatalog(file_name, image_dir=image_dir)

    In the above case, the galaxy/PSF image files are in the directory 
    `/data3/scratch/user_name/galsim/real_galaxy_data/images/`.

    The above behavior is changed if the `image_dir` specifies a directory.  In this case, 
    `image_dir` is interpreted as the full path:

        >>> file_name = '/data3/scratch/user_name/galsim/real_galaxy_data/real_galaxy_catalog.fits'
        >>> image_dir = '/data3/scratch/user_name/galsim/real_galaxy_data/images'
        >>> my_rgc = galsim.RealGalaxyCatalog(file_name, image_dir=image_dir)

    When `dir` is specified without `image_dir` being specified, both the catalog and
    the set of galaxy/PSF images will be searched for under the directory `dir`:

        >>> catalog_dir = '/data3/scratch/user_name/galsim/real_galaxy_data'
        >>> file_name = 'real_galaxy_catalog.fits'
        >>> my_rgc = galsim.RealGalaxyCatalog(file_name, dir=catalog_dir)

    If the `image_dir` is specified in addition to `dir`, the catalog name is specified as 
    `dir/file_name`, while the galaxy/PSF image files will be searched for under `dir/image_dir`:

        >>> catalog_dir = '/data3/scratch/user_name/galsim/real_galaxy_data'
        >>> file_name = 'real_galaxy_catalog.fits'
        >>> image_dir = 'images'
        >>> my_rgc = galsim.RealGalaxyCatalog(file_name, image_dir=image_dir, dir=catalog_dir)

    To explore for the future: scaling with number of galaxies, adding more information as needed,
    and other i/o related issues.

    The GalSim repository currently contains an example catalog, in
    `GalSim/examples/data/real_galaxy_catalog_example.fits` (100 galaxies), along with the
    corresponding image data in other files (`real_galaxy_images.fits` and
    `real_galaxy_PSF_images.fits`) in that directory.  For information on how to download a larger
    sample of 26k training galaxies, see the RealGalaxy Data Download Page on the GalSim Wiki:
    https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data%20Download%20Page

    @param file_name  The file containing the catalog.
    @param image_dir  If a string containing no `/`, it is the relative path from the location of
                      the catalog file to the directory containing the galaxy/PDF images.
                      If a path (a string containing `/`), it is the full path to the directory
                      containing the galaxy/PDF images.
    @param dir        The directory of catalog file (optional).
    @param preload    Whether to preload the header information. (default `preload = False`)
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'image_dir' : str , 'dir' : str, 'preload' : bool }
    _single_params = []
    _takes_rng = False

    # nobject_only is an intentionally undocumented kwarg that should be used only by
    # the config structure.  It indicates that all we care about is the nobjects parameter.
    # So skip any other calculations that might normally be necessary on construction.
    def __init__(self, file_name, image_dir=None, dir=None, preload=False, nobjects_only=False):
        import os
        # First build full file_name
        if dir is None:
            self.file_name = file_name
            if image_dir == None:
                self.image_dir = os.path.dirname(file_name)
            elif os.path.dirname(image_dir) == '':
                self.image_dir = os.path.join(os.path.dirname(self.file_name),image_dir)
            else:
                self.image_dir = image_dir
        else:
            self.file_name = os.path.join(dir,file_name)
            if image_dir == None:
                self.image_dir = dir
            else:
                self.image_dir = os.path.join(dir,image_dir)
        if not os.path.isdir(self.image_dir):
            raise RuntimeError(self.image_dir+' directory does not exist!')

        import pyfits
        cat = pyfits.getdata(self.file_name)
        self.nobjects = len(cat) # number of objects in the catalog
        if nobjects_only: return  # Exit early if that's all we needed.
        ident = cat.field('ident') # ID for object in the training sample
        # We want to make sure that the ident array contains all strings.
        # Strangely, ident.astype(str) produces a string with each element == '1'.
        # Hence this way of doing the conversion:
        self.ident = [ "%s"%val for val in ident ]
        self.gal_file_name = cat.field('gal_filename') # file containing the galaxy image
        self.PSF_file_name = cat.field('PSF_filename') # file containing the PSF image
        self.gal_hdu = cat.field('gal_hdu') # HDU containing the galaxy image
        self.PSF_hdu = cat.field('PSF_hdu') # HDU containing the PSF image
        self.pixel_scale = cat.field('pixel_scale') # pixel scale for image (could be different
        # if we have training data from other datasets... let's be general here and make it a 
        # vector in case of mixed training set)
        self.variance = cat.field('noise_variance') # noise variance for image
        self.mag = cat.field('mag')   # apparent magnitude
        self.band = cat.field('band') # bandpass in which apparent mag is measured, e.g., F814W
        self.weight = cat.field('weight') # weight factor to account for size-dependent
                                          # probability

        self.preloaded = False
        self.do_preload = preload

        # eventually I think we'll want information about the training dataset, 
        # i.e. (dataset, ID within dataset)
        # also note: will be adding bits of information, like noise properties and galaxy fit params

    def _get_index_for_id(self, id):
        """Internal function to find which index number corresponds to the value ID in the ident 
        field.
        """
        # Just to be completely consistent, convert id to a string in the same way we
        # did above for the ident array:
        id = "%s"%id
        if id in self.ident:
            return self.ident.index(id)
        else:
            raise ValueError('ID %s not found in list of IDs'%id)

    def preload(self):
        """Preload the files into memory.
        
        There are memory implications to this, so we don't do this by default.  However, it can be 
        a big speedup if memory isn't an issue.  Especially if many (or all) of the images are 
        stored in the same file as different HDUs.
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

    def getGal(self, i):
        """Returns the galaxy at index `i` as an ImageViewD object.
        """
        if i >= len(self.gal_file_name):
            raise IndexError(
                'index %d given to getGal is out of range (0..%d)'%(i,len(self.gal_file_name)-1))
        import pyfits
        import os
        import numpy
        if self.do_preload and not self.preloaded:
            self.preload()
        if self.preloaded:
            array = self.loaded_files[self.gal_file_name[i]][self.gal_hdu[i]].data
        else:
            file_name = os.path.join(self.image_dir,self.gal_file_name[i])
            array = pyfits.getdata(file_name,self.gal_hdu[i])
        return galsim.ImageViewD(numpy.ascontiguousarray(array.astype(numpy.float64)))

    def getPSF(self, i):
        """Returns the PSF at index `i` as an ImageViewD object.
        """
        if i >= len(self.PSF_file_name):
            raise IndexError(
                'index %d given to getPSF is out of range (0..%d)'%(i,len(self.PSF_file_name)-1))
        import pyfits
        import os
        import numpy
        if self.do_preload and not self.preloaded:
            self.preload()
        if self.preloaded:
            array = self.loaded_files[self.PSF_file_name[i]][self.PSF_hdu[i]].data
        else:
            file_name = os.path.join(self.image_dir,self.PSF_file_name[i])
            array = pyfits.getdata(file_name,self.PSF_hdu[i])
        return galsim.ImageViewD(numpy.ascontiguousarray(array.astype(numpy.float64)))


def simReal(real_galaxy, target_PSF, target_pixel_scale, g1=0.0, g2=0.0, rotation_angle=None, 
            rand_rotate=True, rng=None, target_flux=1000.0, image=None):
    """Function to simulate images (no added noise) from real galaxy training data.

    This function takes a RealGalaxy from some training set, and manipulates it as needed to 
    simulate a (no-noise-added) image from some lower-resolution telescope.  It thus requires a
    target PSF (which could be an image, or one of our base classes) that represents all PSF 
    components including the pixel response, and a target pixel scale.  

    The default rotation option is to impose a random rotation to make irrelevant any real shears 
    in the galaxy training data (optionally, the RNG can be supplied).  This default can be turned 
    off by setting `rand_rotate = False` or by requesting a specific rotation angle using the
    `rotation_angle` keyword, in which case `rand_rotate` is ignored.

    Optionally, the user can specify a shear (default 0).  Finally, they can specify a flux 
    normalization for the final image, default 1000.

    @param real_galaxy         The RealGalaxy object to use, not modified in generating the
                               simulated image.
    @param target_PSF          The target PSF, either one of our base classes or an ImageView/Image.
    @param target_pixel_scale  The pixel scale for the final image, in arcsec.
    @param g1                  First component of shear to impose (components defined with respect
                               to pixel coordinates), default `g1 = 0.`
    @param g2                  Second component of shear to impose, default `g2 = 0.`
    @param rotation_angle      Angle by which to rotate the galaxy (must be a galsim.Angle 
                               instance).
    @param rand_rotate         If `rand_rotate = True` (default) then impose a random rotation on 
                               the training galaxy; this is ignored if `rotation_angle` is set.
    @param rng                 A random number generator to use for selection of the random 
                               rotation angle. (optional, may be any kind of galsim.BaseDeviate 
                               or None)
    @param target_flux         The target flux in the output galaxy image, default 
                               `target_flux = 1000.`
    @param image               As with the GSObject.draw() function, if an image is provided,
                               then it will be used and returned.
                               If `image=None`, then an appropriately sized image will be created.
    @return A simulated galaxy image.  The input RealGalaxy is unmodified. 
    """
    # do some checking of arguments
    if not isinstance(real_galaxy, galsim.RealGalaxy):
        raise RuntimeError("Error: simReal requires a RealGalaxy!")
    for Class in galsim.Image.itervalues():
        if isinstance(target_PSF, Class):
            lan5 = galsim.Lanczos(5, conserve_flux = True, tol = 1.e-4)
            interp2d = galsim.InterpolantXY(lan5)
            target_PSF = galsim.SBInterpolatedImage(
                target_PSF.view(), xInterp=interp2d, dx = target_pixel_scale)
            break
    for Class in galsim.ImageView.itervalues():
        if isinstance(target_PSF, Class):
            lan5 = galsim.Lanczos(5, conserve_flux = True, tol = 1.e-4)
            interp2d = galsim.InterpolantXY(lan5)
            target_PSF = galsim.SBInterpolatedImage(target_PSF,
                                                    xInterp=interp2d,
                                                    dx = target_pixel_scale)
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
