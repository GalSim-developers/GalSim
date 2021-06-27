# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import os
import numpy as np

from .gsobject import GSObject
from .gsparams import GSParams
from .chromatic import ChromaticSum
from .position import PositionD
from .utilities import lazy_property, doc_inherit, convert_interpolant
from .interpolant import Quintic
from .interpolatedimage import InterpolatedImage, _InterpolatedKImage
from .convolve import Convolve, Deconvolve
from .image import ImageCD
from .correlatednoise import CovarianceSpectrum
from . import _galsim
from .errors import GalSimError, GalSimValueError, GalSimIncompatibleValuesError
from .errors import GalSimIndexError


HST_area = 45238.93416  # Area of HST primary mirror in cm^2 from Synphot User's Guide.

# Currently, have bandpasses available for HST COSMOS, AEGIS, and CANDELS.
# ACS zeropoints (AB magnitudes) from
# http://www.stsci.edu/hst/acs/analysis/zeropoints/old_page/localZeropoints#tablestart
# WFC3 zeropoints (AB magnitudes) from
# http://www.stsci.edu/hst/wfc3/phot_zp_lbn
# Format of dictionary entry is:
#    'KEY' : tuple(bandpass filename, zeropoint)
real_galaxy_bandpasses = {
        'F275W': ('WFC3_uvis_F275W.dat', 24.1305),
        'F336W': ('WFC3_uvis_F336W.dat', 24.6682),
        'F435W': ('ACS_wfc_F435W.dat', 25.65777),
        'F606W': ('ACS_wfc_F606W.dat', 26.49113),
        'F775W': ('ACS_wfc_F775W.dat', 25.66504),
        'F814W': ('ACS_wfc_F814W.dat', 25.94333),
        'F850LP': ('ACS_wfc_F850LP.dat', 24.84245),
        'F105W': ('WFC3_ir_F105W.dat', 26.2687),
        'F125W': ('WFC3_ir_F125W.dat', 26.2303),
        'F160W': ('WFC3_ir_F160W.dat', 25.9463)
}

class RealGalaxy(GSObject):
    """A class describing real galaxies from some training dataset.  Its underlying implementation
    uses a Convolution instance of an `InterpolatedImage` (for the observed galaxy) with a
    `Deconvolution` of another `InterpolatedImage` (for the PSF).

    This class uses a catalog describing galaxies in some training data (for more details, see the
    `RealGalaxyCatalog` documentation) to read in data about realistic galaxies that can be used for
    simulations based on those galaxies.  Also included in the class is additional information that
    might be needed to make or interpret the simulations, e.g., the noise properties of the training
    data.  Users who wish to draw RealGalaxies that have well-defined flux scalings in various
    passbands, and/or parametric representations, should use the COSMOSGalaxy class.

    Because RealGalaxy involves a `Deconvolution`, ``method = 'phot'`` is unavailable for the
    `GSObject.drawImage` function.

    Example::

        >>> real_galaxy = galsim.RealGalaxy(real_galaxy_catalog, index=None, id=None, random=False,
        ...                                 rng=None, x_interpolant=None, k_interpolant=None,
        ...                                 flux=None, pad_factor=4, noise_pad_size=0,
        ...                                 gsparams=None)

    This initializes ``real_galaxy`` with three `InterpolatedImage` objects (one for the deconvolved
    galaxy, and saved versions of the original HST image and PSF). Note that there are multiple
    keywords for choosing a galaxy; exactly one must be set.

    Note that tests suggest that for optimal balance between accuracy and speed, ``k_interpolant``
    and ``pad_factor`` should be kept at their default values.  The user should be aware that
    significant inaccuracy can result from using other combinations of these parameters; more
    details can be found in http://arxiv.org/abs/1401.2636, especially table 1, and in comment
    https://github.com/GalSim-developers/GalSim/issues/389#issuecomment-26166621 and the following
    comments.

    If you don't set a flux, the flux of the returned object will be the flux of the original
    HST data, scaled to correspond to a 1 second HST exposure (though see the ``area_norm``
    parameter below, and also caveats related to using the ``flux`` parameter).  If you want a flux
    appropriate for a longer exposure, or for a telescope with a different collecting area than HST,
    you can either renormalize the object with the ``flux_rescale`` parameter, or by using the
    ``exptime`` and ``area`` parameters to `GSObject.drawImage`.

    Note that RealGalaxy objects use arcsec for the units of their linear dimension.  If you
    are using a different unit for other things (the PSF, WCS, etc.), then you should dilate
    the resulting object with ``gal.dilate(galsim.arcsec / scale_unit)``.

    Parameters:
        real_galaxy_catalog:    A `RealGalaxyCatalog` object with basic information about where to
                                find the data, etc.
        index:                  Index of the desired galaxy in the catalog. [One of ``index``,
                                ``id``, or ``random`` is required.]
        id:                     Object ID for the desired galaxy in the catalog. [One of ``index``,
                                ``id``, or ``random`` is required.]
        random:                 If True, then select a random galaxy from the catalog.  If the
                                catalog has a 'weight' associated with it to allow for correction of
                                selection effects in which galaxies were included, the 'weight'
                                factor is used to remove those selection effects rather than
                                selecting a completely random object.
                                [One of ``index``, ``id``, or ``random`` is required.]
        rng:                    A random number generator to use for selecting a random galaxy
                                (may be any kind of `BaseDeviate` or None) and to use in generating
                                any noise field when padding. [default: None]
        x_interpolant:          Either an `Interpolant` instance or a string indicating which
                                real-space interpolant should be used.  Options are 'nearest',
                                'sinc', 'linear', 'cubic', 'quintic', or 'lanczosN' where N should
                                be the integer order to use. [default: galsim.Quintic()]
        k_interpolant:          Either an `Interpolant` instance or a string indicating which
                                k-space interpolant should be used.  Options are 'nearest', 'sinc',
                                'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the
                                integer order to use.  We strongly recommend leaving this parameter
                                at its default value; see text above for details.
                                [default: galsim.Quintic()]
        flux:                   Total flux, if None then original flux in image is adopted without
                                change.  Note that, technically, this parameter sets the flux of the
                                postage stamp image and not the flux of the contained galaxy.
                                These two values will be strongly correlated when the signal-to-
                                noise ratio of the galaxy is large, but may be considerably
                                different if the flux of the galaxy is small with respect to the
                                noise variations in the postage stamp.  To avoid complications with
                                faint galaxies, consider using the flux_rescale parameter.
                                [default: None]
        flux_rescale:           Flux rescaling factor; if None, then no rescaling is done.  Either
                                ``flux`` or ``flux_rescale`` may be set, but not both.
                                [default: None]
        pad_factor:             Factor by which to pad the `Image` when creating the
                                `InterpolatedImage`.  We strongly recommend leaving this parameter
                                at its default value; see text above for details.  [default: 4]
        noise_pad_size:         If provided, the image will be padded out to this size (in arcsec)
                                with the noise specified in the real galaxy catalog. This is
                                important if you are planning to whiten the resulting image.  You
                                should make sure that the padded image is larger than the postage
                                stamp onto which you are drawing this object.
                                [default: None]
        area_norm:              Area in cm^2 by which to normalize the flux of the returned object.
                                When area_norm=1 (the default), drawing with `GSObject.drawImage`
                                keywords exptime=1 and area=1 will simulate an image with the
                                appropriate number of counts for a 1 second exposure with the
                                original telescope/camera (e.g., with HST when using the COSMOS
                                catalog).
                                If you would rather explicitly specify the collecting area of the
                                telescope when using `GSObject.drawImage` with a `RealGalaxy`,
                                then you should set area_norm equal to the collecting area of the
                                source catalog telescope when creating the `RealGalaxy` (e.g.,
                                area_norm=45238.93416 for HST).  [default: 1]
        gsparams:               An optional `GSParams` argument. [default: None]
        logger:                 A logger object for output of progress statements if the user wants
                                them.  [default: None]
    """
    _opt_params = { "x_interpolant" : str ,
                    "k_interpolant" : str ,
                    "flux" : float ,
                    "flux_rescale" : float ,
                    "pad_factor" : float,
                    "noise_pad_size" : float,
                    "area_norm" : float
                  }
    _single_params = [ { "index" : int , "id" : str , "random" : bool } ]
    _takes_rng = True

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, real_galaxy_catalog, index=None, id=None, random=False,
                 rng=None, x_interpolant=None, k_interpolant=None, flux=None, flux_rescale=None,
                 pad_factor=4, noise_pad_size=0, area_norm=1.0, gsparams=None, logger=None):
        from .random import BaseDeviate, UniformDeviate
        from .correlatednoise import UncorrelatedNoise, BaseCorrelatedNoise
        from .interpolatedimage import InterpolatedImage
        from .convolve import Convolve, Deconvolve
        from .config import LoggerWrapper

        if rng is None:
            rng = BaseDeviate()
        elif not isinstance(rng, BaseDeviate):
            raise TypeError("The rng provided to RealGalaxy is not a BaseDeviate")
        self.rng = rng

        if flux is not None and flux_rescale is not None:
            raise GalSimIncompatibleValuesError(
                "Cannot supply a flux and a flux rescaling factor.",
                flux=flux, flux_rescale=flux_rescale)

        logger = LoggerWrapper(logger)  # So don't need to check `if logger:` all the time.

        if isinstance(real_galaxy_catalog, tuple):
            # Special (undocumented) way to build a RealGalaxy without needing the rgc directly
            # by providing the things we need from it.  Used by COSMOSGalaxy.
            self.gal_image, self.psf_image, noise_image, pixel_scale, var = real_galaxy_catalog
            use_index = 0  # For the logger statements below.
            logger.debug('RealGalaxy %d: Start RealGalaxy constructor.',use_index)
            self.catalog_file = None
            self.catalog = ''
        else:
            # Get the index to use in the catalog
            if index is not None:
                if id is not None or random:
                    raise GalSimIncompatibleValuesError(
                        "Too many methods for selecting a galaxy.",
                        index=index, id=id, random=random)
                use_index = index
            elif id is not None:
                if random:
                    raise GalSimIncompatibleValuesError(
                        "Too many methods for selecting a galaxy.", id=id, random=random)
                use_index = real_galaxy_catalog.getIndexForID(id)
            elif random:
                ud = UniformDeviate(self.rng)
                use_index = int(real_galaxy_catalog.nobjects * ud())
                if real_galaxy_catalog.weight is not None:
                    # If weight factors are available, make sure the random selection uses the
                    # weights to remove the catalog-level selection effects (flux_radius-dependent
                    # probability of making a postage stamp for a given object).
                    while ud() > real_galaxy_catalog.weight[use_index]:
                        # Pick another one to try.
                        use_index = int(real_galaxy_catalog.nobjects * ud())
            else:
                raise GalSimIncompatibleValuesError(
                    "No method specified for selecting a galaxy.",
                    index=index, id=id, random=random)
            logger.debug('RealGalaxy %d: Start RealGalaxy constructor.',use_index)

            # Read in the galaxy, PSF images; for now, rely on pyfits to make I/O errors.
            self.gal_image = real_galaxy_catalog.getGalImage(use_index)
            logger.debug('RealGalaxy %d: Got gal_image',use_index)

            self.psf_image = real_galaxy_catalog.getPSFImage(use_index)
            logger.debug('RealGalaxy %d: Got psf_image',use_index)

            #self._gal_noise = real_galaxy_catalog.getNoise(use_index, self.rng, gsparams)
            # We need to duplication some of the RealGalaxyCatalog.getNoise() function, since we
            # want it to be possible to have the RealGalaxyCatalog in another process, and the
            # BaseCorrelatedNoise object is not picklable.  So we just build it here instead.
            noise_image, pixel_scale, var = real_galaxy_catalog.getNoiseProperties(use_index)
            logger.debug('RealGalaxy %d: Got noise_image',use_index)
            self.catalog_file = real_galaxy_catalog.getFileName()
            self.catalog = real_galaxy_catalog

        self._gsparams = GSParams.check(gsparams)

        if noise_image is None:
            self._gal_noise = UncorrelatedNoise(var, rng=self.rng, scale=pixel_scale,
                                                gsparams=self._gsparams)
        else:
            ii = InterpolatedImage(noise_image, normalization="sb",
                                   calculate_stepk=False, calculate_maxk=False,
                                   x_interpolant='linear', gsparams=self._gsparams)
            self._gal_noise = BaseCorrelatedNoise(self.rng, ii, noise_image.wcs)
            self._gal_noise = self._gal_noise.withVariance(var)
        logger.debug('RealGalaxy %d: Finished building noise',use_index)

        # Save any other relevant information as instance attributes
        self.index = use_index
        self.pixel_scale = float(pixel_scale)
        self._x_interpolant = x_interpolant
        self._k_interpolant = k_interpolant
        self._pad_factor = pad_factor
        self._noise_pad_size = noise_pad_size
        self._input_flux = flux
        self._flux_rescale = flux_rescale
        self._area_norm = area_norm

        # Convert noise_pad to the right noise to pass to InterpolatedImage
        if noise_pad_size:
            noise_pad = self._gal_noise
        else:
            noise_pad = 0.

        # Build the InterpolatedImage of the PSF.
        self.original_psf = InterpolatedImage(
            self.psf_image, x_interpolant=x_interpolant, k_interpolant=k_interpolant,
            flux=1.0, gsparams=self._gsparams)
        logger.debug('RealGalaxy %d: Made original_psf',use_index)

        # Build the InterpolatedImage of the galaxy.
        # Use the stepk value of the PSF as a maximum value for stepk of the galaxy.
        # (Otherwise, low surface brightness galaxies can get a spuriously high stepk, which
        # leads to problems.)
        self.original_gal = InterpolatedImage(
                self.gal_image, x_interpolant=x_interpolant, k_interpolant=k_interpolant,
                pad_factor=pad_factor, noise_pad_size=noise_pad_size,
                calculate_stepk=self.original_psf.stepk,
                calculate_maxk=self.original_psf.maxk,
                noise_pad=noise_pad, rng=self.rng, gsparams=self._gsparams)
        logger.debug('RealGalaxy %d: Made original_gal',use_index)

        # Only alter normalization if a change is requested
        if flux is not None or flux_rescale is not None or area_norm != 1:
            if flux_rescale is None:
                flux_rescale = 1.0
            flux_rescale /= area_norm
            if flux is not None:
                flux_rescale *= flux/self.original_gal.flux
            self.original_gal *= flux_rescale
            self._gal_noise *= flux_rescale**2

        logger.debug('RealGalaxy %d: Finished building RealGalaxy',use_index)

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, **kwargs)
        ret.original_gal = self.original_gal.withGSParams(ret._gsparams, **kwargs)
        ret.original_psf = self.original_psf.withGSParams(ret._gsparams, **kwargs)
        ret._gal_noise = self._gal_noise.withGSParams(ret._gsparams, **kwargs)
        return ret

    @classmethod
    def makeFromImage(cls, image, PSF, xi, **kwargs):
        """Create a `RealGalaxy` directly from image, PSF, and noise description.

        Parameters:
            image:  `Image` of the galaxy you want to simulate.
            PSF:    `GSObject` representing the PSF of the galaxy image.  Note that this PSF
                    should include the response of the pixel convolution.
            xi:     `BaseCorrelatedNoise` object characterizing the noise correlations in the input
                    image.
        """
        noise_image = xi.drawImage()
        pixel_scale = noise_image.scale
        var = xi.getVariance()
        psf_image = PSF.drawImage(method='no_pixel')
        return RealGalaxy((image, psf_image, noise_image, pixel_scale, var))

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, RealGalaxy) and
                 self.catalog == other.catalog and
                 self.index == other.index and
                 self._x_interpolant == other._x_interpolant and
                 self._k_interpolant == other._k_interpolant and
                 self._pad_factor == other._pad_factor and
                 self._noise_pad_size == other._noise_pad_size and
                 self._input_flux == other._input_flux and
                 self._flux_rescale == other._flux_rescale and
                 self._area_norm == other._area_norm and
                 self._gsparams == other._gsparams))

    def __hash__(self):
        return hash(("galsim.RealGalaxy", self.catalog, self.index, self._x_interpolant,
                     self._k_interpolant, self._pad_factor, self._noise_pad_size, self._input_flux,
                     self._flux_rescale, self._area_norm, self._gsparams))

    def __repr__(self):
        s = 'galsim.RealGalaxy(%r, index=%r, '%(self.catalog, self.index)
        if self._x_interpolant is not None:
            s += 'x_interpolant=%r, '%self._x_interpolant
        if self._k_interpolant is not None:
            s += 'k_interpolant=%r, '%self._k_interpolant
        if self._pad_factor != 4:
            s += 'pad_factor=%r, '%self._pad_factor
        if self._noise_pad_size != 0:
            s += 'noise_pad_size=%r, '%self._noise_pad_size
        if self._input_flux is not None:
            s += 'flux=%r, '%self._input_flux
        if self._flux_rescale is not None:
            s += 'flux_rescale=%r, '%self._flux_rescale
        if self._area_norm != 1:
            s += 'area_norm=%r, '%self._area_norm
        s += 'rng=%r, '%self.rng
        s += 'gsparams=%r)'%self._gsparams
        return s

    def __str__(self):
        # I think this is more intuitive without the RealGalaxyCatalog parameter listed.
        return 'galsim.RealGalaxy(index=%s, flux=%s)'%(self.index, self.flux)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_conv',None)
        d.pop('_psf_inv',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @lazy_property
    def _psf_inv(self):
        return Deconvolve(self.original_psf, gsparams=self._gsparams)

    @lazy_property
    def _conv(self):
        return Convolve([self.original_gal, self._psf_inv], gsparams=self._gsparams)

    @property
    def _noise(self):
        # We just store the original noise, not convolved with psf_inv until we need it,
        # mostly so we don't have to invalidate this if gsparams changes.
        return self._gal_noise.convolvedWith(self._psf_inv, self._gsparams)

    @property
    def _maxk(self):
        return self._conv._maxk

    @property
    def _stepk(self):
        return self._conv._stepk

    @property
    def _centroid(self):
        return self._conv._centroid

    @property
    def _flux(self):
        return self._conv._flux

    @property
    def _positive_flux(self):
        return self._conv._positive_flux

    @property
    def _negative_flux(self):
        return self._conv._negative_flux

    @lazy_property
    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

    @property
    def _max_sb(self):
        return self._conv._max_sb

    def _kValue(self, kpos):
        return self._conv._kValue(kpos)

    def _drawKImage(self, image, jac=None):
        self._conv._drawKImage(image, jac)


class RealGalaxyCatalog(object):
    """Class containing a catalog with information about real galaxy training data.

    The RealGalaxyCatalog class reads in and stores information about a specific training sample of
    realistic galaxies. We assume that all files containing the images (galaxies and PSFs) live in
    one directory; they could be individual files, or multiple HDUs of the same file.  Currently
    there is no functionality that lets this be a FITS data cube, because we assume that the object
    postage stamps will in general need to be different sizes depending on the galaxy size.

    Note that when simulating galaxies based on HST but using either realistic or parametric galaxy
    models, the COSMOSCatalog class may be more useful.  It allows the imposition of selection
    criteria and other subtleties that are more difficult to impose with RealGalaxyCatalog.

    While you could create your own catalog to use with this class, the typical use cases would
    be to use one of the catalogs that we have created and distributed.  There are three such
    catalogs currently, which can be use with one of the following initializations:

    1. A small example catalog is distributed with the GalSim distribution.  This catalog only
       has 100 galaxies, so it is not terribly useful as a representative galaxy population.
       But for simplistic use cases, it might be sufficient.  We use it for our unit tests and
       in some of the demo scripts (demo6, demo10, and demo11).  To use this catalog, you would
       initialize with::

           >>> rgc = galsim.RealGalaxyCatalog('real_galaxy_catalog_23.5_example.fits',
                                              dir='path/to/GalSim/examples/data')

    2. There are two larger catalogs based on HST observations of the COSMOS field with around
       26,000 and 56,000 galaxies each with a limiting magnitude of F814W=23.5.  (The former is
       a subset of the latter.) For information about how to download these catalogs, see the
       RealGalaxy Data Download Page on the GalSim Wiki:

       https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data

       Be warned that the catalogs are quite large.  The larger one is around 11 GB after unpacking
       the tarball.  To use one of these catalogs, you would initialize with::

           >>> rgc = galsim.RealGalaxyCatalog('real_galaxy_catalog_23.5.fits',
                                              dir='path/to/download/directory')

    3. There is a catalog containing a random subsample of the HST COSMOS images with a limiting
       magnitude of F814W=25.2.  More information about downloading these catalogs can be found on
       the RealGalaxy Data Download page linked above.

    4. Finally, we provide a program that will download the large COSMOS sample for you and
       put it in the $PREFIX/share/galsim directory of your installation path.  The program is::

           galsim_download_cosmos

       which gets installed in the $PREFIX/bin directory when you install GalSim.  If you use
       this program to download the COSMOS catalog, then you can use it with::

           >>> rgc = galsim.RealGalaxyCatalog()

       GalSim knows the location of the installation share directory, so it will automatically
       look for it there.

    Parameters:
        file_name:  The file containing the catalog. [default: None, which will look for the
                    F814W<25.2 COSMOS catalog in $PREFIX/share/galsim.  It will raise an
                    exception if the catalog is not there telling you to run
                    galsim_download_cosmos.]
        sample:     A keyword argument that can be used to specify the sample to use, i.e.,
                    "23.5" or "25.2".  At most one of ``file_name`` and ``sample`` should be
                    specified.
                    [default: None, which results in the same default as ``file_name=None``.]
        dir:        The directory containing the catalog, image, and noise files, or symlinks to
                    them. [default: None]
        preload:    Whether to preload the header information.  If ``preload=True``, the bulk of
                    the I/O time is in the constructor.  If ``preload=False``, there is
                    approximately the same total I/O time (assuming you eventually use most of
                    the image files referenced in the catalog), but it is spread over the
                    various calls to `getGalImage` and `getPSFImage`.  [default: False]
        logger:     An optional logger object to log progress. [default: None]
    """
    _opt_params = { 'file_name' : str, 'sample' : str, 'dir' : str,
                    'preload' : bool }

    # _nobject_only is an intentionally undocumented kwarg that should be used only by
    # the config structure.  It indicates that all we care about is the nobjects parameter.
    # So skip any other calculations that might normally be necessary on construction.
    def __init__(self, file_name=None, sample=None, dir=None, preload=False, logger=None):
        from ._pyfits import pyfits
        from .config import LoggerWrapper
        from multiprocessing import Lock

        if sample is not None and file_name is not None:
            raise GalSimIncompatibleValuesError(
                "Cannot specify both the sample and file_name.",
                sample=sample, file_name=file_name)

        logger = LoggerWrapper(logger)

        self.file_name, self.image_dir, self.sample = _parse_files_dirs(file_name, dir, sample)

        with pyfits.open(self.file_name) as fits:
            self.cat = fits[1].data
        self.nobjects = len(self.cat) # number of objects in the catalog
        logger.debug('RealGalaxyCatalog %s has %d objects',self.file_name,self.nobjects)

        self._preload = preload
        self.loaded_files = {}
        self.saved_noise_im = {}
        # The pyfits commands aren't thread safe.  So we need to make sure the methods that
        # use pyfits are not run concurrently from multiple threads.
        self.gal_lock = Lock()  # Use this when accessing gal files
        self.psf_lock = Lock()  # Use this when accessing psf files
        self.loaded_lock = Lock()  # Use this when opening new files from disk
        self.noise_lock = Lock()  # Use this for building the noise image(s) (usually just one)


    # Some lazy properties that we set up the first time they are used.
    @lazy_property
    def ident(self):
        ident = self.cat.field('ident') # ID for object in the training sample
        # We want to make sure that the ident array contains all strings.
        # Strangely, ident.astype(str) produces a string with each element == '1'.
        # Hence this way of doing the conversion:
        return [ "%s"%val for val in ident ]

    @lazy_property
    def gal_file_name(self):
        gal_file_name = self.cat.field('gal_filename') # file containing the galaxy image
        # Add the directories:
        # Note the strip call.  Sometimes the filenames have an extra space at the end.
        # This gets rid of that space.
        return [os.path.join(self.image_dir,f.strip()) for f in gal_file_name]

    @lazy_property
    def psf_file_name(self):
        psf_file_name = self.cat.field('PSF_filename') # file containing the PSF image
        return [os.path.join(self.image_dir,f.strip()) for f in psf_file_name]

    @lazy_property
    def noise_file_name(self):
        # We don't require the noise_filename column.  If it is not present, we will use
        # Uncorrelated noise based on the variance column.
        try:
            noise_file_name = self.cat.field('noise_filename') # file containing the noise cf
        except KeyError:
            return None
        else:
            return [os.path.join(self.image_dir,f) for f in noise_file_name]

    @lazy_property
    def gal_hdu(self):
        return self.cat.field('gal_hdu') # HDU containing the galaxy image

    @lazy_property
    def psf_hdu(self):
        return self.cat.field('PSF_hdu') # HDU containing the PSF image

    @lazy_property
    def pixel_scale(self):
        return self.cat.field('pixel_scale') # pixel scale for image (could be different
        # if we have training data from other datasets... let's be general here and make it a
        # vector in case of mixed training set)

    @lazy_property
    def variance(self):
        return self.cat.field('noise_variance') # noise variance for image

    @lazy_property
    def mag(self):
        return self.cat.field('mag')   # apparent magnitude

    @lazy_property
    def band(self):
        return self.cat.field('band') # bandpass in which apparent mag is measured, e.g., F814W

    @lazy_property
    def weight(self):
        # The weight factor should be a float value >=0 (so that random selections of indices can
        # use it to remove any selection effects in the catalog creation process).
        # Here we renormalize by the maximum weight.  If the maximum is below 1, that just means
        # that all galaxies were subsampled at some level, and here we only want to account for
        # relative selection effects within the catalog, not absolute subsampling.  If the maximum
        # is above 1, then our random number generation test used to draw a weighted sample will
        # fail since we use uniform deviates in the range 0 to 1.
        try:
            weight = self.cat.field('weight')
        except KeyError:  # pragma: no cover
            raise OSError("You still have the old COSMOS catalog.  Run the program "
                          "`galsim_download_cosmos -s %s` to upgrade."%(self.sample))
        else:
            return weight/np.max(weight)

    @lazy_property
    def stamp_flux(self):
        try:
            return self.cat.field('stamp_flux')
        except KeyError:  # pragma: no cover
            raise OSError("You still have the old COSMOS catalog.  Run the program "
                          "`galsim_download_cosmos -s %s` to upgrade."%(self.sample))

    def __del__(self):
        # Make sure to clean up pyfits open files if people forget to call close()
        self.close()

    def close(self):
        # Need to close any open files.
        # Make sure to check if loaded_files exists, since the constructor could abort
        # before it gets to the place where loaded_files is built.
        if hasattr(self, 'loaded_files'):
            for f in self.loaded_files.values():
                f.close()
        self.loaded_files = {}

    def getNObjects(self) : return self.nobjects
    def __len__(self): return self.nobjects
    def getFileName(self) : return self.file_name

    def getIndexForID(self, id):
        """Internal function to find which index number corresponds to the value ID in the ident
        field.
        """
        # Just to be completely consistent, convert id to a string in the same way we
        # did above for the ident array:
        id = "%s"%id
        if id in self.ident:
            return self.ident.index(id)
        else:
            raise GalSimValueError('ID not found in list of IDs',id, self.ident)

    def _maybe_preload(self):
        # Preload all files if desired.
        # This is delayed until the first time we might need it, since we might only need
        # to know nobjects and not load the data at all.  The first time we try to do something
        # that needs the files, we'll call preload (if requested).
        if self._preload:
            self.preload()
            self._preload = False  # Once we've loaded them.  Don't do it again.

    def preload(self):
        """Preload the files into memory.

        There are memory implications to this, so we don't do this by default.  However, it can be
        a big speedup if memory isn't an issue.
        """
        from ._pyfits import pyfits
        with self.loaded_lock:
            for file_name in np.concatenate((self.gal_file_name , self.psf_file_name)):
                # numpy sometimes add a space at the end of the string that is not present in
                # the original file.  Stupid.  But this next line removes it.
                file_name = file_name.strip()
                if file_name not in self.loaded_files:
                    # I use memmap=False, because I was getting problems with running out of
                    # file handles in the great3 real_gal run, which uses a lot of rgc files.
                    # I think there must be a bug in pyfits that leaves file handles open somewhere
                    # when memmap = True.  Anyway, I don't know what the performance implications
                    # are (since I couldn't finish the run with the default memmap=True), but I
                    # don't think there is much impact either way with memory mapping in our case.
                    f = pyfits.open(file_name,memmap=False)
                    self.loaded_files[file_name] = f
                    # Access all the data from all hdus to force PyFits to read the data
                    for hdu in f:
                        hdu.data

    def _getFile(self, file_name):
        from ._pyfits import pyfits
        self._maybe_preload()
        if file_name in self.loaded_files:
            f = self.loaded_files[file_name]
        else:
            with self.loaded_lock:
                # Check again in case two processes both hit the else at the same time.
                if file_name in self.loaded_files: # pragma: no cover
                    f = self.loaded_files[file_name]
                else:
                    f = pyfits.open(file_name,memmap=False)
                    self.loaded_files[file_name] = f
        return f

    def getBandpass(self):
        """Returns a `Bandpass` object for the catalog.
        """
        from .bandpass import Bandpass
        try:
            bp = real_galaxy_bandpasses[self.band[0].upper()]
        except KeyError:
            raise GalSimValueError("Bandpass not found.  To use this bandpass, please add an entry "
                                   "to the galsim.real.real_galaxy_bandpasses dictionary.",
                                   self.band[0], real_galaxy_bandpasses.keys())
        return Bandpass(bp[0], wave_type='nm', zeropoint=bp[1])

    def getGalImage(self, i):
        """Returns the galaxy at index ``i`` as an `Image` object.
        """
        from .image import Image
        if i >= len(self.gal_file_name):
            raise GalSimIndexError('index out of range (0..%d)'%(len(self.gal_file_name)-1),i)
        f = self._getFile(self.gal_file_name[i])
        with self.gal_lock:
            array = f[self.gal_hdu[i]].data
        im = Image(np.ascontiguousarray(array.astype(np.float64)), scale=self.pixel_scale[i])
        return im

    def getPSFImage(self, i):
        """Returns the PSF at index ``i`` as an `Image` object.
        """
        from .image import Image
        if i >= len(self.psf_file_name):
            raise GalSimIndexError('index out of range (0..%d)'%(len(self.psf_file_name)-1),i)
        f = self._getFile(self.psf_file_name[i])
        with self.psf_lock:
            array = f[self.psf_hdu[i]].data
        return Image(np.ascontiguousarray(array.astype(np.float64)), scale=self.pixel_scale[i])

    def getPSF(self, i, x_interpolant=None, k_interpolant=None, gsparams=None):
        """Returns the PSF at index ``i`` as a `GSObject`.
        """
        from .interpolatedimage import InterpolatedImage
        psf_image = self.getPSFImage(i)
        return InterpolatedImage(psf_image,
                                 x_interpolant=x_interpolant, k_interpolant=k_interpolant,
                                 flux=1.0, gsparams=gsparams)

    def getNoiseProperties(self, i):
        """Returns the components needed to make the noise correlation function at index ``i``.
           Specifically, the noise image (or None), the pixel_scale, and the noise variance,
           as a tuple (im, scale, var).
        """
        from .image import Image
        if self.noise_file_name is None:
            im = None
        else:
            if i >= len(self.noise_file_name):
                raise GalSimIndexError('index out of range (0..%d)'%(len(self.noise_file_name)-1),i)
            if self.noise_file_name[i] in self.saved_noise_im:
                im = self.saved_noise_im[self.noise_file_name[i]]
            else:
                with self.noise_lock:
                    # Again, a second check in case two processes get here at the same time.
                    if self.noise_file_name[i] in self.saved_noise_im:  # pragma: no cover
                        im = self.saved_noise_im[self.noise_file_name[i]]
                    else:
                        from ._pyfits import pyfits
                        with pyfits.open(self.noise_file_name[i]) as fits:
                            array = fits[0].data
                        im = Image(np.ascontiguousarray(array.astype(np.float64)),
                                        scale=self.pixel_scale[i])
                        self.saved_noise_im[self.noise_file_name[i]] = im

        return im, self.pixel_scale[i], self.variance[i]

    def getNoise(self, i, rng=None, gsparams=None):
        """Returns the noise correlation function at index ``i`` as a `BaseCorrelatedNoise` object.
        """
        from .correlatednoise import UncorrelatedNoise, BaseCorrelatedNoise
        from .interpolatedimage import InterpolatedImage
        im, scale, var = self.getNoiseProperties(i)
        if im is None:
            cf = UncorrelatedNoise(var, rng=rng, scale=scale, gsparams=gsparams)
        else:
            ii = InterpolatedImage(im, normalization="sb",
                                          calculate_stepk=False, calculate_maxk=False,
                                          x_interpolant='linear', gsparams=gsparams)
            cf = BaseCorrelatedNoise(rng, ii, im.wcs)
            cf = cf.withVariance(var)
        return cf

    def __repr__(self):
        return 'galsim.RealGalaxyCatalog(%r)'%self.file_name

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, RealGalaxyCatalog) and
                 self.file_name == other.file_name and
                 self.image_dir == other.image_dir))
    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self): return hash(repr(self))

    def __getstate__(self):
        d = self.__dict__.copy()
        d['loaded_files'] = {}
        d['saved_noise_im'] = {}
        del d['gal_lock']
        del d['psf_lock']
        del d['loaded_lock']
        del d['noise_lock']
        return d

    def __setstate__(self, d):
        from multiprocessing import Lock
        self.__dict__ = d
        self.gal_lock = Lock()
        self.psf_lock = Lock()
        self.loaded_lock = Lock()
        self.noise_lock = Lock()
        pass

def _parse_files_dirs(file_name, image_dir, sample):
    from . import meta_data
    if sample is None:
        if file_name is None:
            use_sample = '25.2'
        elif '25.2' in file_name:
            use_sample = '25.2'
        elif '23.5' in file_name:
            use_sample = '23.5'
        else:
            use_sample = None
    else:
        use_sample = sample

    if file_name is None:
        file_name = 'real_galaxy_catalog_' + use_sample + '.fits'
        if image_dir is None:
            use_meta_dir = True  # Used to give a more helpful error message
            image_dir = os.path.join(meta_data.share_dir,
                                     'COSMOS_'+use_sample+'_training_sample')
        else:
            use_meta_dir = False
        full_file_name = os.path.join(image_dir,file_name)
        if not os.path.isfile(full_file_name) and use_meta_dir:
            if use_sample not in ('23.5', '25.2'):
                raise GalSimValueError("Sample name not recognized.",use_sample, ('23.5', '25.2'))
            else:
                raise OSError('No RealGalaxy catalog found in %s. Run the program '
                              'galsim_download_cosmos -s %s to download catalog and accompanying '
                              'image files.'%(image_dir, use_sample))
    elif image_dir is None:
        full_file_name = file_name
        image_dir = os.path.dirname(file_name)
    else:
        full_file_name = os.path.join(image_dir,file_name)
    if not os.path.isfile(full_file_name):
        raise OSError(full_file_name+' not found.')

    return full_file_name, image_dir, use_sample


class ChromaticRealGalaxy(ChromaticSum):
    """A class describing real galaxies over multiple wavelengths, using some multi-band training
    dataset.  The underlying implementation models multi-band images of individual galaxies
    as chromatic PSF convolutions (and integrations over wavelength) with a sum of profiles
    separable into spatial and spectral components.  The spectral components are specified by the
    user, and the spatial components are determined one Fourier mode at a time by the class.  This
    decomposition can be thought of as a constrained chromatic deconvolution of the multi-band
    images by the associated PSFs, similar in spirit to `RealGalaxy`.

    Because ChromaticRealGalaxy involves an `InterpolatedKImage`, ``method = 'phot'`` is unavailable
    for the `ChromaticObject.drawImage` function.

    Fundamentally, the required inputs for this class are:

    (1) a series of high resolution input `Image` instances of a single galaxy in different bands,
    (2) a list of `Bandpass` corresponding to those images,
    (3) the PSFs of those images as either `GSObject` or `ChromaticObject` instances, and
    (4) the noise properties of the input images as `BaseCorrelatedNoise` instances.

    If you want to specify these inputs directly, that is possible via the `makeFromImages` factory
    method of this class::

        >>> crg = galsim.ChromaticRealGalaxy.makeFromImages(imgs, bands, PSFs, xis, ...)

    Alternatively, you may create a ChromaticRealGalaxy via a list of `RealGalaxyCatalog` that
    correspond to a set of galaxies observed in different bands::

        >>> crg = galsim.ChromaticRealGalaxy(real_galaxy_catalogs, index=0, ...)

    The above will use the 1st object in the catalogs, which should be the same galaxy, just
    observed in different bands.  Note that there are multiple keywords for choosing a galaxy from
    a catalog; exactly one must be set.  In the future we may add more such options, e.g., to
    choose at random but accounting for the non-constant weight factors (probabilities for
    objects to make it into the training sample).

    The flux normalization of the returned object will by default match the original data, scaled to
    correspond to a 1 second HST exposure (though see the ``area_norm`` parameter).  If you want
    a flux appropriate for a longer exposure or telescope with different collecting area, you can
    use the `ChromaticObject.withScaledFlux` method on the returned object, or use the ``exptime``
    and ``area`` keywords to `ChromaticObject.drawImage`.

    Note that while you can also use `ChromaticObject.withFlux`, `ChromaticObject.withMagnitude`,
    and `ChromaticObject.withFluxDensity` to set the absolute normalization, these methods
    technically adjust the flux of the entire postage stamp image (including noise!) and not
    necessarily the flux of the galaxy itself.  (These two fluxes will be strongly correlated for
    high signal-to-noise ratio galaxies, but may be considerably different at low signal-to-noise
    ratio.)

    Note that ChromaticRealGalaxy objects use arcsec for the units of their linear dimension.  If
    you are using a different unit for other things (the PSF, WCS, etc.), then you should dilate the
    resulting object with ``gal.dilate(galsim.arcsec / scale_unit)``.

    Noise from the original images is propagated by this class, though certain restrictions apply
    to when and how that noise is made available.  The propagated noise depends on which `Bandpass`
    the ChromaticRealGalaxy is being imaged through, so the noise is only available after the
    `ChromaticObject.drawImage` method has been called.  Also, since ChromaticRealGalaxy will
    only produce reasonable images when convolved with a (suitably wide) PSF, the noise attribute is
    attached to the `ChromaticConvolution` (or `ChromaticTransformation` of the
    `ChromaticConvolution`) which holds as one of its convolutants the `ChromaticRealGalaxy`.::

        >>> crg = galsim.ChromaticRealGalaxy(...)
        >>> psf = ...
        >>> obj = galsim.Convolve(crg, psf)
        >>> bandpass = galsim.Bandpass(...)
        >>> assert not hasattr(obj, 'noise')
        >>> image = obj.drawImage(bandpass)
        >>> assert hasattr(obj, 'noise')
        >>> noise1 = obj.noise

    Note that the noise attribute is only associated with the most recently used bandpass.  If you
    draw another image of the same object using a different bandpass, the noise object will be
    replaced.::

        >>> bandpass2 = galsim.Bandpass(...)
        >>> image2 = obj.drawImage(bandpass2)
        >>> assert noise1 != obj.noise

    Parameters:
        real_galaxy_catalogs:   A list of `RealGalaxyCatalog` objects from which to create
                                `ChromaticRealGalaxy` objects.  Each catalog should represent the
                                same set of galaxies, and in the same order, just imaged through
                                different filters.
        index:                  Index of the desired galaxy in the catalog. [One of ``index``,
                                ``id``, or ``random`` is required.]
        id:                     Object ID for the desired galaxy in the catalog. [One of ``index``,
                                ``id``, or ``random`` is required.]
        random:                 If True, then just select a completely random galaxy from the
                                catalog.  [One of ``index``, ``id``, or ``random`` is required.]
        rng:                    A random number generator to use for selecting a random galaxy (may
                                be any kind of `BaseDeviate` or None) and to use in generating any
                                noise field when padding.
        SEDs:                   An optional list of `SED` instances to use when representing real
                                galaxies as sums of separable profiles.  By default, it will use
                                ``len(real_galaxy_catalogs)`` SEDs that are polynomials in
                                wavelength.  Note that if given, ``len(SEDs)`` must equal
                                ``len(real_galaxy_catalogs)``. [default: None]
        k_interpolant:          Either an `Interpolant` instance or a string indicating which
                                k-space interpolant should be used.  Options are 'nearest', 'sinc',
                                'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the
                                integer order to use.  We strongly recommend leaving this parameter
                                at its default value; see text above for details.
                                [default: galsim.Quintic()]
        maxk:                   Optional maxk argument.  If you know you will be convolving the
                                resulting `ChromaticRealGalaxy` with a "fat" PSF in a subsequent
                                step, then it can be more efficient to limit the range of Fourier
                                modes used when solving for the sum of separable profiles below.
                                [default: None]
        pad_factor:             Factor by which to internally oversample the Fourier-space images
                                that represent the `ChromaticRealGalaxy` (equivalent to zero-padding
                                the real-space profiles).  We strongly recommend leaving this
                                parameter at its default value; see text in Realgalaxy docstring
                                for details.  [default: 4]
        noise_pad_size:         If provided, the image will be padded out to this size (in arcsec)
                                with the noise specified in the real galaxy catalog. This is
                                important if you are planning to whiten the resulting image.  You
                                should make sure that the padded image is larger than the postage
                                stamp onto which you are drawing this object.
                                [default: None]
        area_norm:              Area in cm^2 by which to normalize the flux of the returned object.
                                When area_norm=1 (the default), using ``exptime=1`` and ``area=1``
                                arguments in `ChromaticObject.drawImage` (also the default) will
                                simulate an image with the appropriate number of counts for a 1
                                second exposure with the original telescope/camera (e.g., with HST
                                when using the COSMOS catalog).
                                If you would rather explicitly specify the collecting area of the
                                telescope when using `ChromaticObject.drawImage` with a
                                `ChromaticRealGalaxy`, then you should set area_norm equal to the
                                collecting area of the source catalog telescope when creating the
                                `ChromaticRealGalaxy` (e.g., area_norm=45238.93416 for HST).
                                [default: 1]
        gsparams:               An optional `GSParams` argument. [default: None]
        logger:                 A logger object for output of progress statements if the user wants
                                them.  [default: None]

    """
    # TODO: SEDs isn't implemented yet in config parser.
    _opt_params = { "k_interpolant" : str ,
                    "maxk" : float,
                    "pad_factor" : float,
                    "noise_pad_size" : float,
                    "area_norm" : float
                  }
    _single_params = [ { "index" : int , "id" : str , "random" : bool } ]
    _takes_rng = True

    def __init__(self, real_galaxy_catalogs, index=None, id=None, random=False, rng=None,
                 gsparams=None, logger=None, **kwargs):
        from .random import BaseDeviate, UniformDeviate
        from .bounds import BoundsI
        from .interpolatedimage import InterpolatedImage
        from .correlatednoise import BaseCorrelatedNoise
        from .config import LoggerWrapper

        if rng is None:
            rng = BaseDeviate()
        elif not isinstance(rng, BaseDeviate):
            raise TypeError("The rng provided to ChromaticRealGalaxy is not a BaseDeviate")
        self.rng = rng

        logger = LoggerWrapper(logger)  # So don't need to check `if logger:` all the time.

        # Get the index to use in the catalog
        if index is not None:
            if id is not None or random:
                raise GalSimIncompatibleValuesError(
                    "Too many methods for selecting a galaxy.", index=index, id=id, random=random)
            use_index = index
        elif id is not None:
            if random:
                raise GalSimIncompatibleValuesError(
                    "Too many methods for selecting a galaxy.", id=id, random=random)
            use_index = real_galaxy_catalogs[0].getIndexForID(id)
        elif random:
            uniform_deviate = UniformDeviate(self.rng)
            use_index = int(real_galaxy_catalogs[0].nobjects * uniform_deviate())
        else:
            raise GalSimIncompatibleValuesError(
                "No method specified for selecting a galaxy.", index=index, id=id, random=random)
        logger.debug('ChromaticRealGalaxy %d: Start ChromaticRealGalaxy constructor.', use_index)
        self.index = use_index

        # Read in the galaxy, PSF images; for now, rely on pyfits to make I/O errors.
        imgs = [rgc.getGalImage(use_index) for rgc in real_galaxy_catalogs]
        logger.debug('ChromaticRealGalaxy %d: Got gal_image', use_index)

        PSFs = [rgc.getPSF(use_index) for rgc in real_galaxy_catalogs]
        logger.debug('ChromaticRealGalaxy %d: Got psf', use_index)

        bands = [rgc.getBandpass() for rgc in real_galaxy_catalogs]

        xis = []
        for rgc in real_galaxy_catalogs:
            noise_image, pixel_scale, var = rgc.getNoiseProperties(use_index)
            # Make sure xi image is odd-sized.
            if noise_image.array.shape[0] % 2 == 0: #pragma: no branch
                bds = noise_image.bounds
                new_bds = BoundsI(bds.xmin+1, bds.xmax, bds.ymin+1, bds.ymax)
                noise_image = noise_image[new_bds]
            ii = InterpolatedImage(noise_image, normalization='sb',
                                   calculate_stepk=False, calculate_maxk=False,
                                   x_interpolant='linear', gsparams=gsparams)
            xi = BaseCorrelatedNoise(self.rng, ii, noise_image.wcs)
            xi = xi.withVariance(var)
            xis.append(xi)
        logger.debug('ChromaticRealGalaxy %d: Got noise_image',use_index)
        self.catalog_files = [rgc.getFileName() for rgc in real_galaxy_catalogs]

        self._initialize(imgs, bands, xis, PSFs, gsparams=gsparams, **kwargs)

    @classmethod
    def makeFromImages(cls, images, bands, PSFs, xis, **kwargs):
        """Create a `ChromaticRealGalaxy` directly from images, bandpasses, PSFs, and noise
        descriptions.  See the `ChromaticRealGalaxy` docstring for more information.

        Parameters:
            images:             An iterable of high resolution `Image` instances of a galaxy
                                through different bandpasses.
            bands:              An iterable of `Bandpass` objects corresponding to the  input
                                images.
            PSFs:               Either an iterable of `GSObject` or `ChromaticObject` indicating
                                the PSFs of the different input images, or potentially a single
                                `GSObject` or `ChromaticObject` that will be used as the PSF for
                                all images.
            xis:                An iterable of `BaseCorrelatedNoise` objects characterizing the
                                noise in the input images.
            SEDs:               An optional list of `SED` instances to use when representing real
                                galaxies as sums of separable profiles.  By default, it will use
                                ``len(images)`` SEDs that are polynomials in wavelength.  Note that
                                if given, ``len(SEDs)`` must equal ``len(images)``. [default: None]
            k_interpolant:      Either an `Interpolant` instance or a string indicating which
                                k-space interpolant should be used.  Options are 'nearest', 'sinc',
                                'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the
                                integer order to use.  We strongly recommend leaving this parameter
                                at its default value; see text above for details.  [default:
                                galsim.Quintic()]
            maxk:               Optional maxk argument.  If you know you will be convolving the
                                resulting `ChromaticRealGalaxy` with a "fat" PSF in a subsequent
                                step, then it can be more efficient to limit the range of Fourier
                                modes used when solving for the sum of separable profiles below.
                                [default: None]
            pad_factor:         Factor by which to internally oversample the Fourier-space images
                                that represent the `ChromaticRealGalaxy` (equivalent to zero-padding
                                the real-space profiles).  We strongly recommend leaving this
                                parameter at its default value; see text in Realgalaxy docstring
                                for details.  [default: 4]
            noise_pad_size:     If provided, the image will be padded out to this size (in arcsec)
                                with the noise specified in the real galaxy catalog. This is
                                important if you are planning to whiten the resulting image.  You
                                should make sure that the padded image is larger than the postage
                                stamp onto which you are drawing this object.
                                [default: None]
            area_norm:          Area in cm^2 by which to normalize the flux of the returned object.
                                When area_norm=1 (the default), using ``exptime=1`` and ``area=1``
                                arguments in `ChromaticObject.drawImage` (also the default) will
                                simulate an image with the appropriate number of counts for a 1
                                second exposure with the original telescope/camera (e.g., with HST
                                when using the COSMOS catalog).
                                If you would rather explicitly specify the collecting area of the
                                telescope when using `ChromaticObject.drawImage` with a
                                `ChromaticRealGalaxy`, then you should set area_norm equal to the
                                collecting area of the source catalog telescope when creating the
                                `ChromaticRealGalaxy` (e.g., area_norm=45238.93416 for HST).
                                [default: 1]
            gsparams:           An optional `GSParams` argument. [default: None]
            logger:             A logger object for output of progress statements if the user wants
                                them.  [default: None]

        """
        from .random import BaseDeviate
        if not hasattr(PSFs, '__iter__'):
            PSFs = [PSFs]*len(images)
        obj = cls.__new__(cls)
        obj.index = None
        obj.catalog_files = None
        obj.rng = kwargs.pop('rng', BaseDeviate())

        if len(images) != len(bands) or len(images) != len(xis) or len(images) != len(PSFs):
            raise GalSimIncompatibleValuesError(
                "The number of images, bands, xis, and PSFs must match.",
                images=images, bands=bands, xis=xis, PSFs=PSFs)

        obj._initialize(images, bands, xis, PSFs, **kwargs)
        return obj

    def _initialize(self, imgs, bands, xis, PSFs,
                    SEDs=None, k_interpolant=None, maxk=None, pad_factor=4., area_norm=1.0,
                    noise_pad_size=0, gsparams=None):

        if SEDs is None:
            SEDs = self._poly_SEDs(bands)
        elif len(SEDs) > len(imgs):
            raise GalSimIncompatibleValuesError(
                "The number of SEDs must be <= the number of images",
                images=imgs, SEDs=SEDs)
        self.SEDs = SEDs

        if k_interpolant is None:
            k_interpolant = Quintic()
        else:
            k_interpolant = convert_interpolant(k_interpolant)

        self._area_norm = area_norm
        self._k_interpolant = k_interpolant
        self._gsparams = GSParams.check(gsparams)

        NSED = len(self.SEDs)
        Nim = len(imgs)
        #assert Nim == len(bands)
        #assert Nim == len(xis)
        #assert Nim == len(PSFs)
        #assert Nim >= NSED

        if area_norm != 1.0:
            imgs = [img/area_norm for img in imgs]
            xis = [xi/area_norm**2 for xi in xis]

        # Need to sample three different types of objects on the same Fourier grid: the input
        # effective PSFs, the input images, and the input correlation-functions/power-spectra.
        # There are quite a few potential options for implementing this Fourier sampling.  Some
        # examples include:
        #   * draw object in real space, interpolate onto the real-space grid conjugate to the
        #     desired Fourier-space grid and then DFT with numpy.fft methods.
        #   * Use numpy.fft methods on pre-sampled real-space input (like the input images), then
        #     use an InterpolatedKImage object to regrid onto desired Fourier grid.
        #   * Create an InterpolatedImage from pre-sampled input then use drawKImage to directly
        #     sample on desired Fourier grid.
        # I'm sure there are other options too.  The options chosen below were chosen empirically
        # based on tests of propagating both (chromatic) galaxy images and images of pure noise.

        # Select maxk by requiring modes to be resolved both by the marginal PSFs (i.e., the
        # achromatic PSFs obtained by evaluating the chromatic PSF at the blue and red edges of
        # each of the filters provided) and also by the input images' pixel scales.

        img_maxk = np.min([np.pi/img.scale for img in imgs])
        marginal_PSFs = [PSF.evaluateAtWavelength(band.blue_limit)
                         for PSF in PSFs for band in bands]
        marginal_PSFs += [PSF.evaluateAtWavelength(band.red_limit)
                          for PSF in PSFs for band in bands]
        psf_maxk = np.min([p.maxk for p in marginal_PSFs])

        # In practice, the output PSF should almost always cut off at smaller maxk than obtained
        # above.  In this case, the user can set the maxk keyword argument for improved efficiency.
        if maxk is None:
            maxk = np.min([img_maxk, psf_maxk])
        else:
            maxk = np.min([img_maxk, psf_maxk, maxk])

        # Setting stepk is trickier.  We'll assume that the postage stamp inputs are already at the
        # critical size to avoid significant aliasing and use the implied stepk.  We'll insist that
        # the WCS is a simple PixelScale.  We'll also use the same trick that InterpolatedImage
        # uses to improve accuracy, namely, increase the Fourier-space resolution a factor of
        # `pad_factor`.
        stepk = np.min([2*np.pi/(img.scale*max(img.array.shape))/pad_factor for img in imgs])
        nk = 2*int(np.floor(maxk/stepk))

        # Create Fourier-space kimages of effective PSFs
        PSF_eff_kimgs = np.empty((Nim, NSED, nk, nk), dtype=np.complex128)
        for i, (img, band, PSF) in enumerate(zip(imgs, bands, PSFs)):
            for j, sed in enumerate(self.SEDs):
                # assume that PSF already includes pixel, so don't convolve one in again.
                PSF_eff_kimgs[i, j] = (PSF * sed).drawKImage(band, nx=nk, ny=nk, scale=stepk).array

        # Get Fourier-space representations of input imgs.
        kimgs = np.empty((Nim, nk, nk), dtype=np.complex128)

        if noise_pad_size == 0:
            noise_pad = 0.

        for i, (img, xi) in enumerate(zip(imgs, xis)):
            if noise_pad_size != 0:
                noise_pad = xi
            ii = InterpolatedImage(img, noise_pad_size=noise_pad_size, noise_pad=noise_pad,
                                   rng=self.rng, pad_factor=pad_factor)
            kimgs[i] = ii.drawKImage(nx=nk, ny=nk, scale=stepk).array

        # Setup input noise power spectra
        pks = np.empty((Nim, nk, nk), dtype=np.float64)
        for i, (img, xi) in enumerate(zip(imgs, xis)):
            pks[i] = xi.drawKImage(nx=nk, ny=nk, scale=stepk).array.real / xi.wcs.pixelArea()
            ny, nx = img.array.shape
            pks[i] *= nx * ny
        w = 1./np.sqrt(pks)

        # Allocate and fill output coefficients and covariances.
        # Note: put NSED axis last, since significantly faster to compute them this way,
        # even though we eventually convert to images which are strided in this format.
        coef = np.zeros((nk, nk, NSED), dtype=np.complex128)
        Sigma = np.empty((nk, nk, NSED, NSED), dtype=np.complex128)

        # Solve the weighted linear least squares problem for each Fourier mode.  This is
        # effectively a constrained chromatic deconvolution.  Take advantage of symmetries.
        _coef = coef.__array_interface__['data'][0]
        _Sigma = Sigma.__array_interface__['data'][0]
        _w = w.__array_interface__['data'][0]
        _kimgs = kimgs.__array_interface__['data'][0]
        _psf = PSF_eff_kimgs.__array_interface__['data'][0]
        _galsim.ComputeCRGCoefficients(_coef, _Sigma, _w, _kimgs, _psf, NSED, Nim, nk, nk)

        # Reorder these so they correspond to (NSED, nky, nkx) and (NSED, NSED, nky, nkx) shapes.
        coef = np.transpose(coef, (2,0,1))
        Sigma = np.transpose(Sigma, (2,3,0,1))

        # Set up obj_list as required of ChromaticSum subclass.
        obj_list = []
        for i, sed in enumerate(self.SEDs):
            obj_list.append(sed * _InterpolatedKImage(
                    ImageCD(coef[i], scale=stepk),
                    k_interpolant=self._k_interpolant,
                    gsparams=self._gsparams))

        Sigma_dict = {}
        for i in range(NSED):
            for j in range(i, NSED):
                obj = _InterpolatedKImage(
                        ImageCD(Sigma[i, j], scale=stepk),
                        k_interpolant=self._k_interpolant,
                        gsparams=self._gsparams)
                obj /= (imgs[0].array.shape[0] * imgs[0].array.shape[1] * imgs[0].scale**2)
                Sigma_dict[(i, j)] = obj

        self.covspec = CovarianceSpectrum(Sigma_dict, self.SEDs)

        ChromaticSum.__init__(self, obj_list)

    @staticmethod
    def _poly_SEDs(bands):
        from .sed import SED
        from .table import LookupTable
        # Use polynomial SEDs by default; up to the number of bands provided.
        waves = []
        for bp in bands:
            waves = np.union1d(waves, bp.wave_list)
        SEDs = []
        for i in range(len(bands)):
            SEDs.append(
                    SED(LookupTable(waves, waves**i, interpolant='linear'), 'nm', 'fphotons')
                    .withFlux(1.0, bands[0]))
        return SEDs

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticRealGalaxy) and
                 self.catalog_files == other.catalog_files and
                 self.index == other.index and
                 self.SEDs == other.SEDs and
                 self._k_interpolant == other._k_interpolant and
                 self._area_norm == other._area_norm and
                 self._gsparams == other._gsparams))
    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self):
        return hash(("galsim.ChromaticRealGalaxy", tuple(self.catalog_files), self.index,
                     tuple(self.SEDs), self._k_interpolant, self._area_norm, self._gsparams))

    def __str__(self):
        return "galsim.ChromaticRealGalaxy(%r, index=%r)"%(self.catalog_files, self.index)

    def __repr__(self):
        return ("galsim.ChromaticRealGalaxy(%r, SEDs=%r, index=%r, k_interpolant=%r, "
                "area_norm=%r, gsparams=%r)"%(self.catalog_files, self.SEDs, self.index,
                                              self._k_interpolant, self._area_norm, self._gsparams))
