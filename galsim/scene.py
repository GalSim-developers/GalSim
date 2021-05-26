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

import numpy as np
import math
import os

from .real import RealGalaxy, RealGalaxyCatalog
from .errors import GalSimError, GalSimValueError, GalSimIncompatibleValuesError
from .errors import GalSimNotImplementedError, galsim_warn
from .utilities import lazy_property

# Below is a number that is needed to relate the COSMOS parametric galaxy fits to quantities that
# GalSim needs to make a GSObject representing that fit.  It is simply the pixel scale, in arcsec,
# in the COSMOS weak lensing reductions used for the fits.
# Note: This isn't used anywhere.  This is just informational, really.
cosmos_pix_scale = 0.03

class COSMOSCatalog(object):
    """
    A class representing a random subsample of galaxies from the COSMOS sample with F814W<25.2
    (default), or alternatively the entire sample with F814W<23.5.

    Depending on the keyword arguments, particularly ``use_real``, the catalog will either have
    information about real galaxies, and/or parametric ones.  To use this with either type of
    galaxies, you need to get the COSMOS datasets in the format that GalSim recognizes; see

    https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy-Data

    option (1) for more information.  Note that if you want to make real galaxies you need to
    download and store the full tarball with all galaxy images, whereas if you want to make
    parametric galaxies you only need the catalog real_galaxy_catalog_25.2_fits.fits (and the
    selection file real_galaxy_catalog_25.2_selection.fits if you want to place cuts on the
    postage stamp quality) and can delete the galaxy and PSF image files.

    Finally, we provide a program that will download the large COSMOS sample for you and
    put it in the $PREFIX/share/galsim directory of your installation path.  The program is::

            galsim_download_cosmos

    which gets installed in the $PREFIX/bin directory when you install GalSim.  If you use
    this program to download the COSMOS catalog, then you can use it with::

            cat = galsim.COSMOSCatalog()

    GalSim knows the location of the installation share directory, so it will automatically
    look for it there.

    In addition to the option of specifying catalog names, this class also accepts a keyword
    argument ``sample`` that can be used to switch between the samples with limiting magnitudes of
    23.5 and 25.2.

    After getting the catalogs, there is a method makeGalaxy() that can make a `GSObject`
    corresponding to any chosen galaxy in the catalog (whether real or parametric).  See
    `makeGalaxy` for more information.  As an interesting application and example of the usage of
    these routines, consider the following code::

        >>> im_size = 64
        >>> pix_scale = 0.05
        >>> bp_file = os.path.join(galsim.meta_data.share_dir, 'wfc_F814W.dat.gz')
        >>> bandpass = galsim.Bandpass(bp_file, wave_type='ang').thin().withZeropoint(25.94)
        >>> cosmos_cat = galsim.COSMOSCatalog()
        >>> psf = galsim.OpticalPSF(diam=2.4, lam=1000.) # bigger than HST F814W PSF.
        >>> indices = np.arange(10)
        >>> real_gal_list = cosmos_cat.makeGalaxy(indices, gal_type='real',
        ...                                       noise_pad_size=im_size*pix_scale)
        >>> param_gal_list = cosmos_cat.makeGalaxy(indices, gal_type='parametric', chromatic=True)
        >>> for ind in indices:
        >>>     real_gal = galsim.Convolve(real_gal_list[ind], psf)
        >>>     param_gal = galsim.Convolve(param_gal_list[ind], psf)
        >>>     im_real = galsim.Image(im_size, im_size)
        >>>     im_param = galsim.Image(im_size, im_size)
        >>>     real_gal.drawImage(image=im_real, scale=pix_scale)
        >>>     param_gal.drawImage(bandpass, image=im_param, scale=pix_scale)
        >>>     im_real.write('im_real_'+str(ind)+'.fits')
        >>>     im_param.write('im_param_'+str(ind)+'.fits')

    This code snippet will draw images of the first 10 entries in the COSMOS catalog, at slightly
    lower resolution than in COSMOS, with a real image and its parametric representation for each of
    those objects.

    Parameters:
        file_name:          The file containing the catalog. [default: None, which will look for the
                            F814W<25.2 COSMOS catalog in $PREFIX/share/galsim.  It will raise an
                            exception if the catalog is not there telling you to run
                            galsim_download_cosmos.]
        sample:             A keyword argument that can be used to specify the sample to use, i.e.,
                            "23.5" or "25.2".  At most one of ``file_name`` and ``sample`` should be
                            specified.  [default: None, which results in the same default as
                            ``file_name=None``.]
        dir:                The directory with the catalog file and, if making realistic galaxies,
                            the image and noise files (or symlinks to them). [default: None, which
                            will look in $PREFIX/share/galsim.]
        preload:            Keyword that is only used for real galaxies, not parametric ones, to
                            choose whether to preload the header information.  If ``preload=True``,
                            the bulk of the I/O time is in the constructor.  If ``preload=False``,
                            there is approximately the same total I/O time (assuming you eventually
                            use most of the image files referenced in the catalog), but it is spread
                            over the calls to makeGalaxy().  [default: False]
        use_real:           Enable the use of realistic galaxies?  [default: True]
                            If this parameter is False, then ``makeGalaxy(gal_type='real')`` will
                            not be allowed, and there will be a (modest) decrease in RAM and time
                            spent on I/O when initializing the COSMOSCatalog. If the real
                            catalog is not available for some reason, it will still be possible to
                            make parametric images.
        exclusion_level:    Level of additional cuts to make on the galaxies based on the quality
                            of postage stamp definition and/or parametric fit quality [beyond the
                            minimal cuts imposed when making the catalog - see Mandelbaum et
                            al. (2012, MNRAS, 420, 1518) for details].  Options:

                            - "none": No cuts.
                            - "bad_stamp": Apply cuts to eliminate galaxies that have failures in
                              postage stamp definition.  These cuts may also eliminate a small
                              subset of the good postage stamps as well.
                            - "bad_fits": Apply cuts to eliminate galaxies that have failures in the
                              parametric fits.  These cuts may also eliminate a small
                              subset of the good parametric fits as well.
                            - "marginal": Apply the above cuts, plus ones that eliminate some more
                              marginal cases.

                            [default: "marginal"]
        min_hlr:            Exclude galaxies whose fitted half-light radius is smaller than this
                            value (in arcsec).  [default: 0, meaning no limit]
        max_hlr:            Exclude galaxies whose fitted half-light radius is larger than this
                            value (in arcsec).  [default: 0, meaning no limit]
        min_flux:           Exclude galaxies whose fitted flux is smaller than this value.
                            [default: 0, meaning no limit]
        max_flux:           Exclude galaxies whose fitted flux is larger than this value.
                            [default: 0, meaning no limit]
        exptime:            The exposure time (in seconds) to assume when creating galaxies.
                            .. note::

                                The processed COSMOS ACS/HST science images have units of
                                counts/second; i.e. they have an effective exposure time of 1
                                second in terms of their flux levels. The default value
                                corresponds to a 1 second exposure on HST, which will match
                                these processed images.

                            [default: 1]
        area:               The effective collecting area (in cm^2) to assume when creating
                            galaxies. [default: None, which means to use the original HST
                            collecting area = pi/4 * 240**2 * (1.-0.33**2)]

    After construction, the following attributes are available:

    Attributes:
        nobjects:       The number of objects in the catalog
    """
    _opt_params = { 'file_name' : str, 'sample' : str, 'dir' : str,
                    'preload' : bool, 'use_real' : bool,
                    'exclusion_level' : str, 'min_hlr' : float, 'max_hlr' : float,
                    'min_flux' : float, 'max_flux' : float
                  }

    hst_eff_area = math.pi * 120**2 * (1-0.33**2)

    def __init__(self, file_name=None, sample=None, dir=None, preload=False,
                 use_real=True, exclusion_level='marginal', min_hlr=0, max_hlr=0.,
                 min_flux=0., max_flux=0., exptime=1., area=None):
        if sample is not None and file_name is not None:
            raise GalSimIncompatibleValuesError(
                "Cannot specify both the sample and file_name.",
                sample=sample, file_name=file_name)

        from ._pyfits import pyfits
        from .real import _parse_files_dirs

        self.use_real = use_real
        self.preload = preload
        self.exptime = exptime
        self.area = area

        # We'll set these up if and when we need them.
        self._bandpass = None
        self._sed = None

        if exclusion_level not in ('none', 'bad_stamp', 'bad_fits', 'marginal'):
            raise GalSimValueError("Invalid value of exclusion_level.", exclusion_level,
                                   ('none', 'bad_stamp', 'bad_fits', 'marginal'))

        # Parse the file name
        self.full_file_name, _, self.use_sample = _parse_files_dirs(file_name, dir, sample)

        try:
            # Read in data.
            with pyfits.open(self.full_file_name) as fits:
                self.param_cat = fits[1].data
            # Check if this was the right file.  It should have a 'fit_status' column.
            self.param_cat['fit_status']
        except KeyError:
            # But if that doesn't work, then the name might be the name of the real catalog,
            # so try adding _fits to it as above.
            param_file_name = self.full_file_name.replace('.fits', '_fits.fits')
            with pyfits.open(param_file_name) as fits:
                self.param_cat = fits[1].data

        # NB. The pyfits FITS_Rec class has a bug where it makes a copy of the full
        # record array in each record (e.g. in getParametricRecord) and then doesn't
        # garbage collect it until the top-level FITS_Record goes out of scope.
        # This leads to a memory leak of order 10MB or so each time we make a parametric
        # galaxy.
        # cf. https://mail.scipy.org/pipermail/astropy/2014-June/003218.html
        # also https://github.com/astropy/astropy/pull/520
        # The simplest workaround seems to be to convert it to a regular numpy recarray.
        # (This also makes it run much faster, as an extra bonus!)
        self.param_cat = np.array(self.param_cat, copy=True)

        self.orig_index = np.arange(len(self.param_cat))
        self._apply_exclusion(exclusion_level, min_hlr, max_hlr, min_flux, max_flux)

    @lazy_property
    def real_cat(self):
        if self.use_real:
            return RealGalaxyCatalog(self.full_file_name, preload=self.preload)
        else:
            return None

    def _apply_exclusion(self, exclusion_level, min_hlr=0, max_hlr=0, min_flux=0, max_flux=0):
        from ._pyfits import pyfits
        mask = np.ones(len(self.orig_index), dtype=bool)
        if exclusion_level in ('marginal', 'bad_stamp'):
            # First, read in what we need to impose selection criteria, if the appropriate
            # exclusion_level was chosen.

            # This should work if the user passed in (or we defaulted to) the real galaxy
            # catalog name:
            selection_file_name = self.full_file_name.replace('.fits', '_selection.fits')
            try:
                with pyfits.open(selection_file_name) as fits:
                    self.selection_cat = fits[1].data
            except (IOError, OSError):
                # There's one more option: full_file_name might be the parametric fit file, so
                # we have to strip off the _fits.fits (instead of just the .fits)
                selection_file_name = self.full_file_name.replace('_fits', '_selection')
                try:
                    with pyfits.open(selection_file_name) as fits:
                        self.selection_cat = fits[1].data
                except (IOError, OSError):  # pragma: no cover
                    raise OSError("File with GalSim selection criteria not found. "
                                  "Run the program `galsim_download_cosmos -s %s` to get the "
                                  "necessary selection file."%(self.use_sample))

            # We proceed to select galaxies in a way that excludes suspect postage stamps (e.g.,
            # with deblending issues), suspect parametric model fits, or both of the above plus
            # marginal ones.  These two options for 'exclusion_level' involve placing cuts on
            # the S/N of the object detection in the original postage stamp, and on issues with
            # masking that can indicate deblending or detection failures.  These cuts were used
            # in GREAT3.  In the case of the masking cut, in some cases there are messed up ones
            # that have a 0 for self.selection_cat['peak_image_pixel_count'].  To make sure we
            # don't divide by zero (generating a RuntimeWarning), and still eliminate those, we
            # will first set that column to 1.e-5.  We choose a sample-dependent mask ratio cut,
            # since this depends on the peak object flux, which will differ for the two samples
            # (and we can't really cut on this for arbitrary user-defined samples).
            if self.use_sample == "23.5":
                cut_ratio = 0.2
                sn_limit = 20.0
            else:
                cut_ratio = 0.8
                sn_limit = 12.0
            div_val = self.selection_cat['peak_image_pixel_count']
            div_val[div_val == 0.] = 1.e-5
            mask &= ( (self.selection_cat['sn_ellip_gauss'] >= sn_limit) &
                      ((self.selection_cat['min_mask_dist_pixels'] > 11.0) |
                       (self.selection_cat['average_mask_adjacent_pixel_count'] / \
                           div_val < cut_ratio)) )

            # Finally, impose a cut that the total flux in the postage stamp should be positive,
            # which excludes a tiny number of galaxies (of order 10 in each sample) with some sky
            # subtraction or deblending errors.  Some of these are eliminated by other cuts when
            # using exclusion_level='marginal'.
            if self.real_cat is not None:
                mask &= self.real_cat.stamp_flux > 0

        if exclusion_level in ('bad_fits', 'marginal'):
            # This 'exclusion_level' involves eliminating failed parametric fits (bad fit status
            # flags).  In this case we only get rid of those with failed bulge+disk AND failed
            # Sersic fits, so there is no viable parametric model for the galaxy.
            sersicfit_status = self.param_cat['fit_status'][:,4]
            bulgefit_status = self.param_cat['fit_status'][:,0]
            mask &= ( ((sersicfit_status > 0) &
                      (sersicfit_status < 5)) |
                      ((bulgefit_status > 0) &
                      (bulgefit_status < 5)) )

        if exclusion_level == 'marginal':
            # We have already placed some cuts (above) in this case, but we'll do some more.  For
            # example, a failed bulge+disk fit often indicates difficulty in fit convergence due to
            # noisy surface brightness profiles, so we might want to toss out those that have a
            # failure in EITHER fit.
            mask &= ( ((sersicfit_status > 0) &
                      (sersicfit_status < 5)) &
                      ((bulgefit_status > 0) &
                      (bulgefit_status < 5)) )

            # Some fit parameters can indicate a likely sky subtraction error: very high sersic n
            # AND abnormally large half-light radius (>1 arcsec).
            if 'hlr' not in self.param_cat.dtype.names:  # pragma: no cover
                raise OSError("You still have the old COSMOS catalog.  Run the program "
                              "`galsim_download_cosmos -s %s` to upgrade."%(self.use_sample))
            hlr = self.param_cat['hlr'][:,0]
            n = self.param_cat['sersicfit'][:,2]
            mask &= ( (n < 5) | (hlr < 1.) )

            # Major flux differences in the parametric model vs. the COSMOS catalog can indicate fit
            # issues, deblending problems, etc.
            mask &= ( np.abs(self.selection_cat['dmag']) < 0.8)

        if min_hlr > 0. or max_hlr > 0. or min_flux > 0. or max_flux > 0.:
            if 'hlr' not in self.param_cat.dtype.names:  # pragma: no cover
                raise OSError("You still have the old COSMOS catalog.  Run the program "
                              "`galsim_download_cosmos -s %s` to upgrade."%(self.use_sample))

            hlr = self.param_cat['hlr'][:,0] # sersic half-light radius
            flux = self.param_cat['flux'][:,0]

            if min_hlr > 0.:
                mask &= (hlr > min_hlr)
            if max_hlr > 0.:
                mask &= (hlr < max_hlr)
            if min_flux > 0.:
                mask &= (flux > min_flux)
            if max_flux > 0.:
                mask &= (flux < max_flux)

        self.orig_index = self.orig_index[mask]
        self.nobjects = len(self.orig_index)

    # We need this method because the config apparatus will use this via a Proxy, and they cannot
    # access attributes directly -- just call methods.  So this is how we get nobjects there.
    def getNObjects(self) : return self.nobjects
    def getUseSample(self): return self.use_sample
    def getOrigIndex(self, index): return self.orig_index[index]
    def getNTot(self) : return len(self.param_cat)
    def __len__(self): return self.nobjects

    def makeGalaxy(self, index=None, gal_type=None, chromatic=False, noise_pad_size=5,
                   deep=False, sersic_prec=0.05, rng=None, n_random=None, gsparams=None):
        """
        Routine to construct one or more `GSObject` instances corresponding to the catalog entry
        with a particular index or indices.

        The flux of the galaxy corresponds to a 1 second exposure time with the Hubble Space
        Telescope.  Users who wish to simulate F814W images with a different telescope and an
        exposure time longer than 1 second should multiply by that exposure time, and by the square
        of the ratio of the effective diameter of their telescope compared to that of HST.
        (Effective diameter may differ from the actual diameter if there is significant
        obscuration.)  See demo11.py for an example that explicitly takes this normalization into
        account.

        Due to the adopted flux normalization, drawing into an image with the COSMOS bandpass,
        zeropoint of 25.94, and pixel scale should give the right pixel values to mimic the actual
        COSMOS science images.  The COSMOS science images that we use are normalized to a count rate
        of 1 second, which is why there is no need to rescale to account for the COSMOS exposure
        time.

        There is an option to make chromatic objects (``chromatic=True``); however, it is important
        to bear in mind that we do not actually have spatially-resolved color information for these
        galaxies, so this keyword can only be True if we are using parametric galaxies.  Even then,
        we simply do the most arbitrary thing possible, which is to assign bulges an elliptical
        `SED`, disks a disk-like `SED`, and `Sersic` galaxies with intermediate values of n some
        intermediate `SED`.  We assume that the photometric redshift is the correct redshift for
        these galaxies (which is a good assumption for COSMOS 30-band photo-z for these bright
        galaxies).  For the given `SED` and redshift, we then normalize to give the right (observed)
        flux in F814W.  Note that for a mock "deep" sample, the redshift distributions of the
        galaxies would be modified, which is not included here.

        For this chromatic option, it is still the case that the output flux normalization is
        appropriate for the HST effective telescope diameter and a 1 second exposure time, so users
        who are simulating another scenario should account for this.

        Note that the returned objects use arcsec for the units of their linear dimension.  If you
        are using a different unit for other things (the PSF, WCS, etc.), then you should dilate
        the resulting object with ``gal.dilate(galsim.arcsec / scale_unit)``.

        Parameters:
            index:          Index of the desired galaxy in the catalog for which a `GSObject`
                            should be constructed.  You may also provide a list or array of
                            indices, in which case a list of objects is returned. If None,
                            then a random galaxy (or more: see n_random kwarg) is chosen,
                            correcting for catalog-level selection effects if weights are
                            available. [default: None]
            gal_type:       Either 'real' or 'parametric'.  This determines which kind of
                            galaxy model is made. [If catalog was loaded with ``use_real=False``,
                            then this defaults to 'parametric', and in fact 'real' is
                            not allowed.  If catalog was loaded with ``use_real=True``, then
                            this defaults to 'real'.]
            chromatic:      Make this a chromatic object, or not?  [default: False]
            noise_pad_size: For realistic galaxies, the size of region to pad with noise,
                            in arcsec.  [default: 5, an arbitrary, but not completely
                            ridiculous choice.]
            deep:           Modify fluxes and sizes of galaxies from the F814W<23.5 sample in
                            order to roughly simulate an F814W<25 sample but with higher S/N, as
                            in GREAT3? [default: False]  Note that this keyword will be ignored
                            (except for issuing a warning) if the input catalog already
                            represents the F814W<25.2 sample.
            sersic_prec:    The desired precision on the Sersic index n in parametric galaxies.
                            GalSim is significantly faster if it gets a smallish number of
                            Sersic values, so it can cache some of the calculations and use
                            them again the next time it gets a galaxy with the same index.
                            If ``sersic_prec`` is 0.0, then use the exact value of index n from
                            the catalog.  But if it is >0, then round the index to that
                            precision.  [default: 0.05]
            rng:            A random number generator to use for selecting a random galaxy
                            (may be any kind of `BaseDeviate` or None) and to use in generating
                            any noise field when padding.  [default: None]
            n_random:       The number of random galaxies to build, if 'index' is None.
                            [default: 1]
            gsparams:       An optional `GSParams` argument. [default: None]

        Returns:
            Either a `GSObject` or a `ChromaticObject` depending on the value of ``chromatic``,
            or a list of them if ``index`` is an iterable.
        """
        return self._makeGalaxy(self, index, gal_type, chromatic, noise_pad_size,
                                deep, sersic_prec, self.exptime, self.area,
                                rng, n_random, gsparams)

    @staticmethod
    def _makeGalaxy(self, index=None, gal_type=None, chromatic=False, noise_pad_size=5,
                    deep=False, sersic_prec=0.05, exptime=1., area=None,
                    rng=None, n_random=None, gsparams=None):
        from .random import BaseDeviate
        if not self.canMakeReal():
            if gal_type is None:
                gal_type = 'parametric'
            elif gal_type != 'parametric':
                raise GalSimIncompatibleValuesError(
                    "Only 'parametric' galaxy type is allowed when use_real == False",
                    gal_type=gal_type, use_real=self.canMakeReal())
        else:
            if gal_type is None:
                gal_type = 'real'

        if gal_type not in ('real', 'parametric'):
            raise GalSimValueError("Invalid galaxy type %r", gal_type, ('real', 'parametric'))

        # Make rng if we will need it.
        if index is None or gal_type == 'real':
            if rng is None:
                rng = BaseDeviate()
            elif not isinstance(rng, BaseDeviate):
                raise TypeError("The rng provided to makeGalaxy is not a BaseDeviate")

        # Select random indices if necessary (no index given).
        if index is None:
            if n_random is None: n_random = 1
            index = self.selectRandomIndex(n_random, rng=rng)
        else:
            if n_random is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot specify both index and n_random", n_random=n_random, index=index)

        if hasattr(index, '__iter__'):
            indices = index
        else:
            indices = [index]

        # Check whether this is a COSMOSCatalog meant to represent real or parametric objects, then
        # call the appropriate helper routine for that case.
        if gal_type == 'real':
            if chromatic:
                raise GalSimNotImplementedError("Cannot yet make real chromatic galaxies!")
            gal_list = []
            for idx in indices:
                real_params = self.getRealParams(idx)
                gal = RealGalaxy(real_params, noise_pad_size=noise_pad_size, rng=rng,
                                 gsparams=gsparams)
                gal_list.append(gal)

        else:
            if chromatic:
                bandpass = self.getBandpass()
                sed = self.getSED()
            else:
                bandpass = None
                sed = None
            gal_list = []
            for idx in indices:
                record = self.getParametricRecord(idx)
                gal = COSMOSCatalog._buildParametric(record, sersic_prec, gsparams,
                                                     chromatic, bandpass, sed)
                gal_list.append(gal)

        flux_scaling = exptime
        if area is not None:
            flux_scaling *= area/self.hst_eff_area
        if flux_scaling != 1.:
            gal_list = [gal * flux_scaling for gal in gal_list]

        # If trying to use the 23.5 sample and "fake" a deep sample, rescale the size and flux as
        # suggested in the GREAT3 handbook.
        if deep:
            if self.getUseSample() == '23.5':
                # Rescale the flux to get a limiting mag of 25 in F814W when starting with a
                # limiting mag of 23.5.  Make the galaxies a factor of 0.6 smaller and appropriately
                # fainter.
                flux_factor = 10.**(-0.4*1.5)
                size_factor = 0.6
                gal_list = [ gal.dilate(size_factor) * flux_factor for gal in gal_list ]
            elif self.getUseSample() == '25.2':
                galsim_warn("Ignoring `deep` argument, because the sample being used already "
                            "corresponds to a flux limit of F814W<25.2")
            else:
                galsim_warn("Ignoring `deep` argument, because the sample being used does not "
                            "corresponds to a flux limit of F814W<23.5")

        # Store the orig_index as gal.index regardless of whether we have a RealGalaxy or not.
        # It gets set as part of making a real galaxy, but not by _buildParametric.
        # And if we are doing the deep scaling, then it gets messed up by that.
        # So just put it in here at the end to be sure.
        for gal, idx in zip(gal_list, indices):
            gal.index = self.getOrigIndex(idx)
            if hasattr(gal, 'original'): gal.original.index = gal.index

        if hasattr(index, '__iter__'):
            return gal_list
        else:
            return gal_list[0]

    def selectRandomIndex(self, n_random=1, rng=None, _n_rng_calls=False):
        """
        Routine to select random indices out of the catalog.  This routine does a weighted random
        selection with replacement (i.e., there is no guarantee of uniqueness of the selected
        indices).  Weighting uses the weight factors available in the catalog, if any; these weights
        are typically meant to remove any selection effects in the catalog creation process.

        Parameters:
            n_random:   Number of random indices to return. [default: 1]
            rng:        A random number generator to use for selecting a random galaxy
                        (may be any kind of `BaseDeviate` or None). [default: None]

        Returns:
            A single index if n_random==1 or a NumPy array containing the randomly-selected
            indices if n_random>1.
        """
        from .random import BaseDeviate
        from . import utilities
        # Set up the random number generator.
        if rng is None:
            rng = BaseDeviate()

        if self.real_cat is not None:
            use_weights = self.real_cat.weight[self.orig_index]
        else:
            galsim_warn("Selecting random object without correcting for catalog-level "
                        "selection effects.  This correction requires the existence of "
                        "real catalog with valid weights in addition to parametric one. "
                        "Create the COSMOSCatalog with use_real=True to avoid this warning.")
            use_weights = None

        # By default, get the number of RNG calls.  We then decide whether or not to return them
        # based on _n_rng_calls.
        index, n_rng_calls = utilities.rand_with_replacement(
                n_random, self.nobjects, rng, use_weights, _n_rng_calls=True)

        if n_random>1:
            if _n_rng_calls:
                return index, n_rng_calls
            else:
                return index
        else:
            if _n_rng_calls:
                return index[0], n_rng_calls
            else:
                return index[0]

    def getBandpass(self):
        from .bandpass import Bandpass
        # Defer making the Bandpass and reading in SEDs until we actually are going to use them.
        # It's not a huge calculation, but the thin() call especially isn't trivial.

        if self._bandpass is None:
            # We have to set an appropriate zeropoint.  This is slightly complicated: The
            # nominal COSMOS zeropoint for single-orbit depth (2000s of usable exposure time,
            # across 4 dithered exposures) is supposedly 25.94.  But the science images that we
            # are using were normalized to count rate, not counts, meaning that an object with
            # mag=25.94 has a count rate of 1 photon/sec, not 1 photon total.  Since we've
            # declared our flux normalization for the outputs to be appropriate for a 1s
            # exposure, we use this zeropoint directly.
            # This means that when drawing chromatic parametric galaxies, the outputs will be
            # properly normalized in terms of counts.
            zp = 25.94
            self._bandpass = Bandpass('ACS_wfc_F814W.dat', wave_type='nm').withZeropoint(zp)
        return self._bandpass

    def getSED(self):
        from .sed import SED
        if self._sed is None:
            # Read in some SEDs.  We are using some fairly truncated and thinned ones, because
            # in any case the SED assignment here is somewhat arbitrary and should not be taken
            # too seriously.
            self._sed = [
                # bulge
                SED('CWW_E_ext_more.sed', wave_type='Ang', flux_type='flambda'),
                # disk
                SED('CWW_Scd_ext_more.sed', wave_type='Ang', flux_type='flambda'),
                # intermediate
                SED('CWW_Sbc_ext_more.sed', wave_type='Ang', flux_type='flambda')
            ]
        return self._sed

    @staticmethod
    def _round_sersic(n, sersic_prec):
        return float(int(n/sersic_prec + 0.5)) * sersic_prec

    @staticmethod
    def _buildParametric(record, sersic_prec, gsparams, chromatic, bandpass=None, sed=None):
        from .angle import radians
        from .exponential import Exponential
        from .sersic import DeVaucouleurs, Sersic
        # Get fit parameters.  For 'sersicfit', the result is an array of 8 numbers for each
        # galaxy:
        #     SERSICFIT[0]: intensity of light profile at the half-light radius.
        #     SERSICFIT[1]: half-light radius measured along the major axis, in units of pixels
        #                   in the COSMOS lensing data reductions (0.03 arcsec).
        #     SERSICFIT[2]: Sersic n.
        #     SERSICFIT[3]: q, the ratio of minor axis to major axis length.
        #     SERSICFIT[4]: boxiness, currently fixed to 0, meaning isophotes are all
        #                   elliptical.
        #     SERSICFIT[5]: x0, the central x position in pixels.
        #     SERSICFIT[6]: y0, the central y position in pixels.
        #     SERSICFIT[7]: phi, the position angle in radians.  If phi=0, the major axis is
        #                   lined up with the x axis of the image.
        # For 'bulgefit', the result is an array of 16 parameters that comes from doing a
        # 2-component sersic fit.  The first 8 are the parameters for the disk, with n=1, and
        # the last 8 are for the bulge, with n=4.
        bparams = record['bulgefit']
        sparams = record['sersicfit']
        if 'hlr' not in record:  # pragma: no cover
            raise OSError("You still have the old COSMOS catalog.  Run the program "
                          "`galsim_download_cosmos -s %s` to upgrade."%(self.use_sample))

        use_bulgefit = record['use_bulgefit']
        if not use_bulgefit and not record['viable_sersic']:  # pragma: no cover
            # This shouldn't be possible I think...
            raise GalSimError("Cannot make parametric model for this galaxy!")

        if use_bulgefit:
            # Bulge parameters:
            # Minor-to-major axis ratio:
            bulge_q = bparams[11]
            # Position angle, now represented as a galsim.Angle:
            bulge_beta = bparams[15]*radians
            disk_q = bparams[3]
            disk_beta = bparams[7]*radians
            bulge_hlr = record['hlr'][1]
            bulge_flux = record['flux'][1]
            disk_hlr = record['hlr'][2]
            disk_flux = record['flux'][2]

            # Make sure the bulge-to-total flux ratio is not nonsense.
            bfrac = bulge_flux/(bulge_flux+disk_flux)
            if bfrac < 0 or bfrac > 1 or np.isnan(bfrac):  # pragma: no cover
                # This shouldn't be possible I think...
                raise GalSimError("Cannot make parametric model for this galaxy")

            # Then combine the two components of the galaxy.
            if chromatic:
                # We define the GSObjects with flux=1, then multiply by an SED defined to have
                # the appropriate (observed) magnitude at the redshift in the COSMOS passband.
                z = record['zphot']
                target_bulge_mag = record['mag_auto']-2.5*math.log10(bfrac)
                bulge_sed = sed[0].atRedshift(z).withMagnitude(
                    target_bulge_mag, bandpass)
                bulge = DeVaucouleurs(half_light_radius=bulge_hlr, gsparams=gsparams)
                bulge *= bulge_sed
                target_disk_mag = record['mag_auto']-2.5*math.log10((1.-bfrac))
                disk_sed = sed[1].atRedshift(z).withMagnitude(target_disk_mag, bandpass)
                disk = Exponential(half_light_radius=disk_hlr, gsparams=gsparams)
                disk *= disk_sed
            else:
                bulge = DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr,
                                             gsparams=gsparams)
                disk = Exponential(flux=disk_flux, half_light_radius=disk_hlr,
                                          gsparams=gsparams)

            # Apply shears for intrinsic shape.
            if bulge_q < 1.:  # pragma: no branch
                bulge = bulge.shear(q=bulge_q, beta=bulge_beta)
            if disk_q < 1.:  # pragma: no branch
                disk = disk.shear(q=disk_q, beta=disk_beta)

            gal = bulge + disk
        else:
            # Do a similar manipulation to the stored quantities for the single Sersic profiles.
            gal_n = sparams[2]
            # Fudge this if it is at the edge of the allowed n values.  Since GalSim (as of #325 and
            # #449) allow Sersic n in the range 0.3<=n<=6, the only problem is that the fits
            # occasionally go as low as n=0.2.  The fits in this file only go to n=6, so there is no
            # issue with too-high values, but we also put a guard on that side in case other samples
            # are swapped in that go to higher value of sersic n.
            if gal_n < 0.3: gal_n = 0.3
            if gal_n > 6.0: gal_n = 6.0
            # GalSim is much more efficient if only a finite number of Sersic n values are used.
            # This (optionally given constructor args) rounds n to the nearest 0.05.
            if sersic_prec > 0.:
                gal_n = COSMOSCatalog._round_sersic(gal_n, sersic_prec)
            gal_q = sparams[3]
            gal_beta = sparams[7]*radians
            gal_hlr = record['hlr'][0]
            gal_flux = record['flux'][0]

            if chromatic:
                gal = Sersic(gal_n, flux=1., half_light_radius=gal_hlr, gsparams=gsparams)
                if gal_n < 1.5:
                    use_sed = sed[1] # disk
                elif gal_n >= 1.5 and gal_n < 3.0:
                    use_sed = sed[2] # intermediate
                else:
                    use_sed = sed[0] # bulge
                target_mag = record['mag_auto']
                z = record['zphot']
                gal *= use_sed.atRedshift(z).withMagnitude(target_mag, bandpass)
            else:
                gal = Sersic(gal_n, flux=gal_flux, half_light_radius=gal_hlr, gsparams=gsparams)

            # Apply shears for intrinsic shape.
            if gal_q < 1.:  # pragma: no branch
                gal = gal.shear(q=gal_q, beta=gal_beta)

        return gal

    def getRealParams(self, index):
        """Get the parameters needed to make a `RealGalaxy` for a given index."""
        # Used by COSMOSGalaxy to circumvent making the RealGalaxy here and potentially having
        # to pickle the result.  These raw materials should be smaller, so quicker to pickle.
        orig_index = self.orig_index[index]
        gal_image = self.real_cat.getGalImage(orig_index)
        psf_image = self.real_cat.getPSFImage(orig_index)
        noise_image, pixel_scale, var = self.real_cat.getNoiseProperties(orig_index)
        return (gal_image, psf_image, noise_image, pixel_scale, var)

    def getParametricRecord(self, index):
        """Get the parametric record for a given index"""
        # Used by _makeGalaxy to circumvent pickling the result.
        record = self.param_cat[self.orig_index[index]]
        # Convert to a dict, since on some systems, the numpy record doesn't seem to
        # pickle correctly.
        record_dict = { k:record[k] for k in record.dtype.names }
        return record_dict

    def canMakeReal(self):
        """Is it permissible to call makeGalaxy with gal_type='real'?"""
        return self.use_real

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, COSMOSCatalog) and
                 self.use_real == other.use_real and
                 self.use_sample == other.use_sample and
                 self.real_cat == other.real_cat and
                 np.array_equal(self.param_cat, other.param_cat) and
                 np.array_equal(self.orig_index, other.orig_index)))
