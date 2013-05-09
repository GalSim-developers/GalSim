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
from galsim import GSObject


class RealGalaxy(GSObject):
    """A class describing real galaxies from some training dataset.  Has an SBConvolve in the
    SBProfile attribute.

    This class uses a catalog describing galaxies in some training data (for more details, see the
    RealGalaxyCatalog documentation) to read in data about realistic galaxies that can be used for
    simulations based on those galaxies.  Also included in the class is additional information that
    might be needed to make or interpret the simulations, e.g., the noise properties of the training
    data.

    The GSObject drawShoot method is unavailable for RealGalaxy instances.

    Initialization
    --------------
    
        real_galaxy = galsim.RealGalaxy(real_galaxy_catalog, index=None, id=None, random=False, 
                                        rng=None, x_interpolant=None, k_interpolant=None,
                                        flux=None, pad_factor = 0, noise_pad=False, pad_image=None,
                                        use_cache = True)

    This initializes real_galaxy with three SBInterpolatedImage objects (one for the deconvolved
    galaxy, and saved versions of the original HST image and PSF). Note that there are multiple
    keywords for choosing a galaxy; exactly one must be set.  In future we may add more such
    options, e.g., to choose at random but accounting for the non-constant weight factors
    (probabilities for objects to make it into the training sample).  Like other GSObjects, the
    RealGalaxy contains an SBProfile attribute which is an SBConvolve representing the deconvolved
    HST galaxy.

    Note that preliminary tests suggest that for optimal balance between accuracy and speed,
    `k_interpolant` and `pad_factor` should be kept at their default values.  The user should be
    aware that significant inaccuracy can result from using other combinations of these parameters;
    see devel/modules/finterp.pdf, especially table 1, in the GalSim repository.

    @param real_galaxy_catalog  A RealGalaxyCatalog object with basic information about where to
                                find the data, etc.
    @param index                Index of the desired galaxy in the catalog.
    @param id                   Object ID for the desired galaxy in the catalog.
    @param random               If true, then just select a completely random galaxy from the
                                catalog.
    @param rng                  A random number generator to use for selecting a random galaxy 
                                (may be any kind of BaseDeviate or None) and to use in generating
                                any noise field when padding.  This user-input random number
                                generator takes precedence over any stored within a user-input
                                CorrelatedNoise instance (see `noise_pad` param below).
    @param x_interpolant        Either an Interpolant2d (or Interpolant) instance or a string 
                                indicating which real-space interpolant should be used.  Options are 
                                'nearest', 'sinc', 'linear', 'cubic', 'quintic', or 'lanczosN' 
                                where N should be the integer order to use. [default 
                                `x_interpolant = galsim.Lanczos(5,...)'].
    @param k_interpolant        Either an Interpolant2d (or Interpolant) instance or a string 
                                indicating which k-space interpolant should be used.  Options are 
                                'nearest', 'sinc', 'linear', 'cubic', 'quintic', or 'lanczosN' 
                                where N should be the integer order to use.  We strongly recommend
                                leaving this parameter at its default value; see text above for
                                details.  [default `k_interpolant = galsim.Quintic()'].
    @param flux                 Total flux, if None then original flux in galaxy is adopted without
                                change [default `flux = None`].
    @param pad_factor           Factor by which to pad the Image when creating the
                                SBInterpolatedImage; `pad_factor <= 0` results in the use of the
                                default value, 4.  We strongly recommend leaving this parameter at
                                its default value; see text above for details.
                                [Default `pad_factor = 0`.]
    @param noise_pad            When creating the SBProfile attribute for this GSObject, pad the
                                SBInterpolated image with zeros, or with noise of a level specified
                                in the training dataset?  There are several options here: 
                                    Use `noise_pad = False` if you wish to pad with zeros.
                                    Use `noise_pad = True` if you wish to pad with uncorrelated
                                        noise of the proper variance.
                                    Set `noise_pad` equal to a galsim.CorrelatedNoise, an Image, or
                                        a filename containing an Image of an example noise field
                                        that will be used to calculate the noise power spectrum and
                                        generate noise in the padding region.  Any random number
                                        generator passed to the `rng` keyword will take precedence
                                        over that carried in an input galsim.CorrelatedNoise.
                                In the last case, if the same file is used repeatedly, then use of
                                the `use_cache` keyword (see below) can be used to prevent the need
                                for repeated galsim.CorrelatedNoise initializations.
                                [default `noise_pad = False`]
    @param pad_image            Image to be used for deterministically padding the original image.
                                This can be specified in two ways:
                                   (a) as a galsim.Image; or
                                   (b) as a string which is interpreted as a filename containing an
                                       image to use.
                               The size of the image that is passed in is taken to specify the
                                amount of padding, and so the `pad_factor` keyword should be equal
                                to 1, i.e., no padding.  The `pad_image` scale is ignored, and taken
                                to be equal to that of the `image`. Note that `pad_image` can be
                                used together with `noise_pad`.  However, the user should be careful
                                to ensure that the image used for padding has roughly zero mean.
                                The purpose of this keyword is to allow for a more flexible
                                representation of some noise field around an object; if the user
                                wishes to represent the sky level around an object, they should do
                                that when they have drawn the final image instead.  (Default
                                `pad_image = None`.)
    @param use_cache            Specify whether to cache noise_pad read in from a file to save
                                having to build an CorrelatedNoise repeatedly from the same image.
                                (Default `use_cache = True`)
    @param gsparams             You may also specify a gsparams argument.  See the docstring for
                                galsim.GSParams using help(galsim.GSParams) for more information
                                about this option.

    Methods
    -------
    The RealGalaxy is a GSObject, and inherits all of the GSObject methods (draw(), applyShear(), 
    etc. except drawShoot() which is unavailable), and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "x_interpolant" : str ,
                    "k_interpolant" : str,
                    "flux" : float ,
                    "pad_factor" : float,
                    "noise_pad" : str,
                    "pad_image" : str}
    _single_params = [ { "index" : int , "id" : str } ]
    _takes_rng = True
    _cache_noise_pad = {}
    _cache_variance = {}

    # --- Public Class methods ---
    def __init__(self, real_galaxy_catalog, index=None, id=None, random=False,
                 rng=None, x_interpolant=None, k_interpolant=None, flux=None, pad_factor = 0,
                 noise_pad=False, pad_image=None, use_cache=True, gsparams=None):

        import pyfits
        import numpy as np

        # Code block below will be for galaxy selection; not all are currently implemented.  Each
        # option must return an index within the real_galaxy_catalog.        
        if index is not None:
            if id is not None or random is True:
                raise AttributeError('Too many methods for selecting a galaxy!')
            use_index = index
        elif id is not None:
            if random is True:
                raise AttributeError('Too many methods for selecting a galaxy!')
            use_index = real_galaxy_catalog._get_index_for_id(id)
        elif random is True:
            if rng is None:
                uniform_deviate = galsim.UniformDeviate()
            elif isinstance(rng, galsim.BaseDeviate):
                uniform_deviate = galsim.UniformDeviate(rng)
            else:
                raise TypeError("The rng provided to RealGalaxy constructor is not a BaseDeviate")
            use_index = int(real_galaxy_catalog.nobjects * uniform_deviate()) 
        else:
            raise AttributeError('No method specified for selecting a galaxy!')

        # read in the galaxy, PSF images; for now, rely on pyfits to make I/O errors. Should
        # consider exporting this code into fits.py in some function that takes a filename and HDU,
        # and returns an ImageView

        gal_image = real_galaxy_catalog.getGal(use_index)
        PSF_image = real_galaxy_catalog.getPSF(use_index)

        # choose proper interpolant
        if x_interpolant is None:
            lan5 = galsim.Lanczos(5, conserve_flux=True, tol=1.e-4)
            self.x_interpolant = galsim.InterpolantXY(lan5)
        else:
            self.x_interpolant = galsim.utilities.convert_interpolant_to_2d(x_interpolant)
        if k_interpolant is None:
            self.k_interpolant = galsim.InterpolantXY(galsim.Quintic(tol=1.e-4))
        else:
            self.k_interpolant = galsim.utilities.convert_interpolant_to_2d(k_interpolant)

        # read in data about galaxy from FITS binary table; store as normal attributes of RealGalaxy

        # save any other relevant information as instance attributes
        self.catalog_file = real_galaxy_catalog.file_name
        self.index = use_index
        self.pixel_scale = float(real_galaxy_catalog.pixel_scale[use_index])

        # handle padding by an image
        specify_size = False
        padded_size = gal_image.getPaddedSize(pad_factor)
        if pad_image is not None:
            specify_size = True
            if isinstance(pad_image,str):
                pad_image = galsim.fits.read(pad_image)
            if ( not isinstance(pad_image, galsim.BaseImageF) and 
                 not isinstance(pad_image, galsim.BaseImageD) ):
                raise ValueError("Supplied pad_image is not one of the allowed types!")
            # If an image was supplied directly or from a file, check its size:
            #    Cannot use if too small.
            #    Use to define the final image size otherwise.
            deltax = ((1 + pad_image.getXMax() - pad_image.getXMin()) - 
                      (1 + gal_image.getXMax() - gal_image.getXMin()))
            deltay = ((1 + pad_image.getYMax() - pad_image.getYMin()) - 
                      (1 + gal_image.getYMax() - gal_image.getYMin()))
            if deltax < 0 or deltay < 0:
                raise RuntimeError("Image supplied for padding is too small!")
            if pad_factor != 1. and pad_factor != 0.:
                import warnings
                msg =  "Warning: ignoring specified pad_factor because user also specified\n"
                msg += "         an image to use directly for the padding."
                warnings.warn(msg)
        else:
            if isinstance(gal_image, galsim.BaseImageF):
                pad_image = galsim.ImageF(padded_size, padded_size)
            if isinstance(gal_image, galsim.BaseImageD):
                pad_image = galsim.ImageD(padded_size, padded_size)

        # Set up the GaussianDeviate if not provided one, or check that the user-provided one
        # is of a valid type.  Note if random was selected, we can use that uniform_deviate safely.
        if random is True:
            gaussian_deviate = galsim.GaussianDeviate(uniform_deviate)
        else:
            if rng is None:
                gaussian_deviate = galsim.GaussianDeviate()
            elif isinstance(rng,galsim.BaseDeviate):
                # Even if it's already a GaussianDeviate, we still want to make a new Gaussian
                # deviate that would generate the same sequence, because later we change the sigma
                # and we don't want to change it for the original one that was passed in.  So don't
                # distinguish between GaussianDeviate and the other BaseDeviates here.
                gaussian_deviate = galsim.GaussianDeviate(rng)
            else:
                raise TypeError("rng provided to RealGalaxy constructor is not a BaseDeviate")

        # handle noise-padding options
        try:
            noise_pad = galsim.config.value._GetBoolValue(noise_pad,'')
        except:
            pass
        if noise_pad:
            self.pad_variance = float(real_galaxy_catalog.variance[use_index])

            # Check, is it "True" or something else?  If True, we use Gaussian uncorrelated noise
            # using the stored variance in the catalog.  Otherwise, if it's a CorrelatedNoise we use
            # it directly; if it's an Image of some sort we use it to make a CorrelatedNoise; if
            # it's a string, we read in the image from file and make a CorrelatedNoise.
            if type(noise_pad) is not bool:
                if isinstance(noise_pad, galsim.correlatednoise._BaseCorrelatedNoise):
                    cn = noise_pad.copy()
                    if rng: # Let user supplied RNG take precedence over that in user CN
                        cn.setRNG(gaussian_deviate)
                    # This small patch may have different overall variance, so rescale while
                    # preserving the correlation structure by default                  
                    cn.setVariance(self.pad_variance)
                elif (isinstance(noise_pad,galsim.BaseImageF) or 
                      isinstance(noise_pad,galsim.BaseImageD)):
                    cn = galsim.CorrelatedNoise(gaussian_deviate, noise_pad)
                elif use_cache and noise_pad in RealGalaxy._cache_noise_pad:
                    cn = RealGalaxy._cache_noise_pad[noise_pad]
                    # Make sure that we are using the desired RNG by resetting that in this cached
                    # CorrelatedNoise instance
                    if rng:
                        cn.setRNG(gaussian_deviate)
                    # This small patch may have different overall variance, so rescale while
                    # preserving the correlation structure
                    cn.setVariance(self.pad_variance)
                elif isinstance(noise_pad, str):
                    tmp_img = galsim.fits.read(noise_pad)
                    cn = galsim.CorrelatedNoise(gaussian_deviate, tmp_img)
                    if use_cache:
                        RealGalaxy._cache_noise_pad[noise_pad] = cn
                    # This small patch may have different overall variance, so rescale while
                    # preserving the correlation structure
                    cn.setVariance(self.pad_variance)
                else:
                    raise RuntimeError("noise_pad must be either a bool, CorrelatedNoise, Image, "+
                                       "or a filename for reading in an Image")

            # Set the GaussianDeviate sigma          
            gaussian_deviate.setSigma(np.sqrt(self.pad_variance))

            # populate padding image with noise field
            if type(noise_pad) is bool:
                pad_image.addNoise(galsim.DeviateNoise(gaussian_deviate))
            else:
                pad_image.addNoise(cn)
        else:
            self.pad_variance=0.

        # Now we have to check: was the padding determined using pad_factor?  Or by passing in an
        # image for padding?  Treat these cases differently:
        # (1) If the former, then we can simply have the C++ handle the padding process.
        # (2) If the latter, then we have to do the padding ourselves, and pass the resulting image
        # to the C++ with pad_factor explicitly set to 1.
        if specify_size is False:
            # Make the SBInterpolatedImage out of the image.
            self.original_image = galsim.SBInterpolatedImage(
                gal_image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                dx=self.pixel_scale, pad_factor=pad_factor, pad_image=pad_image, gsparams=gsparams)
        else:
            # Leave the original image as-is.  Instead, we shift around the image to be used for
            # padding.  Find out how much x and y margin there should be on lower end:
            x_marg = int(np.round(0.5*deltax))
            y_marg = int(np.round(0.5*deltay))
            # Now reset the pad_image to contain the original image in an even way
            pad_image = pad_image.view()
            pad_image.setScale(self.pixel_scale)
            pad_image.setOrigin(gal_image.getXMin()-x_marg, gal_image.getYMin()-y_marg)
            # Set the central values of pad_image to be equal to the input image
            pad_image[gal_image.bounds] = gal_image
            self.original_image = galsim.SBInterpolatedImage(
                pad_image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                dx=self.pixel_scale, pad_factor=1., gsparams=gsparams)

        # also make the original PSF image, with far less fanfare: we don't need to pad with
        # anything interesting.
        self.original_PSF = galsim.SBInterpolatedImage(
            PSF_image, xInterp=self.x_interpolant, kInterp=self.k_interpolant, dx=self.pixel_scale,
            gsparams=gsparams)

        # recalculate Fourier-space attributes rather than using overly-conservative defaults
        self.original_image.calculateStepK()
        self.original_image.calculateMaxK()
        self.original_PSF.calculateStepK()
        self.original_PSF.calculateMaxK()
        
        if flux != None:
            self.original_image.setFlux(flux)
            self.original_image.__class__ = galsim.SBTransform # correctly reflect SBProfile change
        self.original_PSF.setFlux(1.0)
        self.original_PSF.__class__ = galsim.SBTransform # correctly reflect SBProfile change

        # Calculate the PSF "deconvolution" kernel
        psf_inv = galsim.SBDeconvolve(self.original_PSF, gsparams=gsparams)
        # Initialize the SBProfile attribute
        GSObject.__init__(
            self, galsim.SBConvolve([self.original_image, psf_inv], gsparams=gsparams))

    def getHalfLightRadius(self):
        raise NotImplementedError("Half light radius calculation not implemented for RealGalaxy "
                                   +"objects.")


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
