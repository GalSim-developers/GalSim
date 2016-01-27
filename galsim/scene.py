# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
"""@file scene.py
Routines for defining a "sky scene", i.e., a galaxy or star sample with reasonable properties that
can then be placed throughout a large image.  Currently, this only includes routines for making a
COSMOS-based galaxy sample, but it could be expanded to include star samples as well.
"""

import galsim
import numpy as np
import math
import os

# Below is a number that is needed to relate the COSMOS parametric galaxy fits to quantities that
# GalSim needs to make a GSObject representing that fit.  It is simply the pixel scale, in arcsec,
# in the COSMOS weak lensing reductions used for the fits.
cosmos_pix_scale = 0.03

class COSMOSCatalog(object):
    """
    A class representing a sample of galaxies from the COSMOS sample with F814W<23.5.

    Depending on the keyword arguments, particularly `use_real`, the catalog will either have
    information about real galaxies, or parametric ones.  To use this with either type of galaxies,
    you need to get the COSMOS datasets in the format that GalSim recognizes; see

        https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy-Data

    option (1) for more information.  Note that if you want to make real galaxies you need to
    download and store the full tarball with all galaxy images, whereas if you want to make
    parametric galaxies you only need the catalog real_galaxy_catalog_23.5_fits.fits and can delete
    the other files.

    Finally, we provide a program that will download the large COSMOS sample for you and
    put it in the $PREFIX/share/galsim directory of your installation path.  The program is

            galsim_download_cosmos

    which gets installed in the $PREFIX/bin directory when you install GalSim.  If you use
    this program to download the COSMOS catalog, then you can use it with

            cat = galsim.COSMOSCatalog()

    GalSim knows the location of the installation share directory, so it will automatically
    look for it there.

    After getting the catalogs, there is a method makeGalaxy() that can make a GSObject
    corresponding to any chosen galaxy in the catalog (whether real or parametric).  See
    help(galsim.COSMOSCatalog.makeGalaxy) for more information.  As an interesting application and
    example of the usage of these routines, consider the following code:

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
    those objects.  Note that we are automatically excluding galaxies that do not have parametric
    representations.  These are rare and do not occur in the first ten entries in the catalog, which
    is why we can assume that the real and parametric objects will be comparable.

    Initialization
    --------------

    @param file_name    The file containing the catalog. [default: None, which will look for the
                        COSMOS catalog in $PREFIX/share/galsim.  It will raise an exception if the
                        catalog is not there telling you to run galsim_download_cosmos.]
    @param image_dir    Keyword that is only used for real galaxies, not parametric ones, to specify
                        the directory of the image files.
                        If a string containing no `/`, it is the relative path from the location of
                        the catalog file to the directory containing the galaxy/PDF images.
                        If a path (a string containing `/`), it is the full path to the directory
                        containing the galaxy/PDF images. [default: None]
    @param dir          The directory of catalog file. [default: None]
    @param preload      Keyword that is only used for real galaxies, not parametric ones, to choose
                        whether to preload the header information.  If `preload=True`, the bulk of  
                        the I/O time is in the constructor.  If `preload=False`, there is
                        approximately the same total I/O time (assuming you eventually use most of
                        the image files referenced in the catalog), but it is spread over the
                        various calls to getGal() and getPSF().  [default: False]
    @param noise_dir    Keyword that is only used for real galaxies, not parametric ones.
                        The directory of the noise files if different from the directory of the 
                        image files.  [default: image_dir]
    @param use_real     Enable the use of realistic galaxies?  [default: True]
                        If this parameter is False, then `makeGalaxy(gal_type='real')` will
                        not be allowed.
    @param exclude_fail Exclude galaxies that have failures in the parametric fits? [default: True]
    @param exclude_bad  Exclude those that have evidence of probably being a bad fit?  e.g. n > 5
                        and hlr > 1 arcsec, probably indicates poor sky subtraction. [default: True]
    @param min_hlr      Exclude galaxies whose fitted half-light-radius is smaller than this value
                        (in arcsec).  [default: 0, meaning no limit]
    @param max_hlr      Exclude galaxies whose fitted half-light-radius is larger than this value
                        (in arcsec).  [default: 0, meaning no limit]
    @param min_flux     Exclude galaxies whose fitted flux is smaller than this value.  
                        [default: 0, meaning no limit]
    @param max_flux     Exclude galaxies whose fitted flux is larger than this value.  
                        [default: 0, meaning no limit]

    Attributes
    ----------

    After construction, the following attributes are available:

    nobjects     The number of objects in the catalog
    """
    _req_params = {}
    _opt_params = { 'file_name' : str, 'image_dir' : str , 'dir' : str, 'preload' : bool,
                    'noise_dir' : str, 'use_real' : bool,
                    'exclude_fail' : bool, 'exclude_bad' : bool, 
                    'min_hlr' : float, 'max_hlr' : float, 'min_flux' : float, 'max_flux' : float
                  }
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name=None, image_dir=None, dir=None, preload=False, noise_dir=None,
                 use_real=True, exclude_fail=True, exclude_bad=True, 
                 min_hlr=0, max_hlr=0., min_flux=0., max_flux=0.,
                 _nobjects_only=False):
        from galsim._pyfits import pyfits
        self.use_real = use_real

        if self.use_real:
            if not _nobjects_only:
                # First, do the easy thing: real galaxies.  We make the galsim.RealGalaxyCatalog()
                # constructor do most of the work.  But note that we don't actually need to 
                # bother with this if all we care about is the nobjects attribute.
                self.real_cat = galsim.RealGalaxyCatalog(
                    file_name, image_dir=image_dir, dir=dir, preload=preload, noise_dir=noise_dir)

            # The fits name has _fits inserted before the .fits ending.
            # Note: don't just use k = -5 in case it actually ends with .fits.fz
            k = self.real_cat.file_name.find('.fits') 
            param_file_name = self.real_cat.file_name[:k] + '_fits' + self.real_cat.file_name[k:]
            self.param_cat = pyfits.getdata(param_file_name)

        else:
            # Start by doing the same file_name parsing as we did for the real galaxy
            param_file_name, _, _ = galsim.real._parse_files_dirs(
                    file_name, image_dir, dir, noise_dir)
            try:
                # Read in data.
                self.param_cat = pyfits.getdata(param_file_name)
                # Check if this was the right file.  It should have a 'fit_status' column.
                self.param_cat['fit_status']
            except KeyError:
                # But if that doesn't work, then the name might be the name of the real catalog,
                # so try adding _fits to it as above.
                k = param_file_name.find('.fits')
                param_file_name = param_file_name[:k] + '_fits' + param_file_name[k:]
                self.param_cat = pyfits.getdata(param_file_name)

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
        mask = np.ones(len(self.orig_index), dtype=bool)

        # If requested, select galaxies based on existence of a usable fit.
        if exclude_fail:
            sersicfit_status = self.param_cat['fit_status'][:,4]
            bulgefit_status = self.param_cat['fit_status'][:,0]
            mask &= ( (sersicfit_status > 0) &
                      (sersicfit_status < 5) &
                      (bulgefit_status > 0) &
                      (bulgefit_status < 5) )

        if exclude_bad:
            hlr = self.param_cat['sersicfit'][:,1]
            n = self.param_cat['sersicfit'][:,2]
            mask &= ( (n < 5) | (hlr < 1./cosmos_pix_scale) ) 
            # May add more cuts here if we discover other kinds of problematic objects.

        if min_hlr > 0. or max_hlr > 0. or min_flux > 0. or max_flux > 0.:
            sparams = self.param_cat['sersicfit']
            hlr_pix = sparams[:,1]
            n = sparams[:,2]
            q = sparams[:,3]
            hlr = cosmos_pix_scale*hlr_pix*np.sqrt(q)
            if min_hlr > 0.:
                mask &= (hlr > min_hlr)
            if max_hlr > 0.:
                mask &= (hlr < max_hlr)

            if min_flux > 0. or max_flux > 0.:
                flux_hlr = sparams[:,0]
                # The prefactor for n=4 is 3.607.  For n=1, it is 1.901.
                # It's not linear in these values, but for the sake of efficiency and the 
                # ability to work on the whole array at once, just linearly interpolate.
                # Hopefully, this can be improved as part of issue #693.  Maybe by storing the
                # calculated directly flux in the catalog, rather than just the amplitude of the
                # surface brightness profile at the half-light-radius?
                #prefactor = ( (n-1.)*3.607 + (4.-n)*1.901 ) / (4.-1.)
                prefactor = ((3.607-1.901)/3.) * n + (4.*1.901 - 1.*3.607)/3.
                flux = 2.0*np.pi*prefactor*(hlr**2)*flux_hlr/cosmos_pix_scale**2
                if min_flux > 0.:
                    mask &= (flux > min_flux)
                if max_flux > 0.:
                    mask &= (flux < max_flux)

        self.orig_index = self.orig_index[mask]
        self.nobjects = len(self.orig_index)

    # We need this method because the config apparatus will use this via a Proxy, and they cannot
    # access attributes directly -- just call methods.  So this is how we get nobjects there.
    def getNObjects(self) : return self.nobjects

    def getOrigIndex(self, index): return self.orig_index[index]

    def makeGalaxy(self, index=None, gal_type=None, chromatic=False, noise_pad_size=5,
                   deep=False, sersic_prec=0.05, rng=None, gsparams=None):
        """
        Routine to construct GSObjects corresponding to the catalog entry with a particular index 
        or indices.

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

        There is an option to make chromatic objects (`chromatic=True`); however, it is important
        to bear in mind that we do not actually have spatially-resolved color information for these
        galaxies, so this keyword can only be True if we are using parametric galaxies.  Even then,
        we simply do the most arbitrary thing possible, which is to assign bulges an elliptical
        SED, disks a disk-like SED, and Sersic galaxies with intermediate values of n some
        intermediate SED.  We assume that the photometric redshift is the correct redshift for
        these galaxies (which is a good assumption for COSMOS 30-band photo-z for these bright
        galaxies).  For the given SED and redshift, we then normalize to give the right (observed)
        flux in F814W.  Note that for a mock "deep" sample, the redshift distributions of the
        galaxies would be modified, which is not included here.

        For this chromatic option, it is still the case that the output flux normalization is
        appropriate for the HST effective telescope diameter and a 1 second exposure time, so users
        who are simulating another scenario should account for this.

        @param index            Index of the desired galaxy in the catalog for which a GSObject
                                should be constructed.  You may also provide a list or array of
                                indices, in which case a list of objects is returned. If None,
                                then a single galaxy is chosen at random.  [default: None]
        @param gal_type         Either 'real' or 'parametric'.  This determines which kind of 
                                galaxy model is made. [If catalog was loaded with `use_real=False`,
                                then this defaults to 'parametric', and in fact 'real' is 
                                not allowed.]
        @param chromatic        Make this a chromatic object, or not?  [default: False]
        @param noise_pad_size   For realistic galaxies, the size of region to pad with noise,
                                in arcsec.  [default: 5, an arbitrary, but not completely
                                ridiculous choice.]
        @param deep             Modify fluxes and sizes of galaxies in order to roughly simulate
                                an F814W<25 sample? [default: False]
        @param sersic_prec      The desired precision on the Sersic index n in parametric galaxies.
                                GalSim is significantly faster if it gets a smallish number of
                                Sersic values, so it can cache some of the calculations and use
                                them again the next time it gets a galaxy with the same index.
                                If `sersic_prec` is 0.0, then use the exact value of index n from
                                the catalog.  But if it is >0, then round the index to that
                                precision.  [default: 0.05]
        @param rng              A random number generator to use for selecting a random galaxy
                                (may be any kind of BaseDeviate or None) and to use in generating
                                any noise field when padding.  [default: None]
        @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                                details. [default: None]

        @returns    Either a GSObject or a ChromaticObject depending on the value of `chromatic`,
                    or a list of them if `index` is an iterable.
        """
        if not self.use_real:
            if gal_type is None:
                gal_type = 'parametric'
            elif gal_type != 'parametric':
                raise ValueError("Only 'parametric' galaxy type is allowed when use_real == False")

        if gal_type not in ['real', 'parametric']:
            raise ValueError("Invalid galaxy type %r"%gal_type)

        # We'll set these up if and when we need them.
        self._bandpass = None
        self._sed = None

        # Make rng if we will need it.
        if index is None or gal_type == 'real':
            if rng is None:
                rng = galsim.BaseDeviate()
            elif not isinstance(rng, galsim.BaseDeviate):
                raise TypeError("The rng provided to makeGalaxy is not a BaseDeviate")

        if index is None:
            ud = galsim.UniformDeviate(rng)
            index = int(self.nobjects * ud())

        if hasattr(index, '__iter__'):
            indices = index
        else:
            indices = [index]

        # Check whether this is a COSMOSCatalog meant to represent real or parametric objects, then
        # call the appropriate helper routine for that case.
        if gal_type == 'real':
            if chromatic:
                raise RuntimeError("Cannot yet make real chromatic galaxies!")
            gal_list = self._makeReal(indices, noise_pad_size, rng, gsparams)
        else:
            gal_list = self._makeParametric(indices, chromatic, sersic_prec, gsparams)

        # If deep, rescale the size and flux
        if deep:
            # Rescale the flux to get a limiting mag of 25 in F814W.  Current limiting mag is 23.5,
            # so it's a magnitude difference of 1.5.  Make the galaxies a factor of 0.6 smaller and
            # appropriately fainter.
            flux_factor = 10.**(-0.4*1.5)
            size_factor = 0.6
            gal_list = [ gal.dilate(size_factor) * flux_factor for gal in gal_list ]

        # Store the orig_index as gal.index regardless of whether we have a RealGalaxy or not.
        # It gets set by _makeReal, but not by _makeParametric.
        # And if we are doing the deep scaling, then it gets messed up by that.
        # So just put it in here at the end to be sure.
        for gal, index in zip(gal_list, indices):
            gal.index = self.orig_index[index]

        if hasattr(index, '__iter__'):
            return gal_list
        else:
            return gal_list[0]

    def _makeReal(self, indices, noise_pad_size, rng, gsparams):
        return [ galsim.RealGalaxy(self.real_cat, index=self.orig_index[i],
                                   noise_pad_size=noise_pad_size, rng=rng, gsparams=gsparams)
                 for i in indices ]

    def _makeParametric(self, indices, chromatic, sersic_prec, gsparams):
        if chromatic:
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
                zp = 25.94
                self._bandpass = galsim.Bandpass(
                    os.path.join(galsim.meta_data.share_dir, 'wfc_F814W.dat.gz'),
                    wave_type='ang').thin().withZeropoint(zp)
                # This means that when drawing chromatic parametric galaxies, the outputs will be
                # properly normalized in terms of counts.

                # Read in some SEDs.
                self._sed = [
                    # bulge
                    galsim.SED(os.path.join(galsim.meta_data.share_dir,'CWW_E_ext.sed')),
                    # disk
                    galsim.SED(os.path.join(galsim.meta_data.share_dir,'CWW_Scd_ext.sed')),
                    # intermediate
                    galsim.SED(os.path.join(galsim.meta_data.share_dir,'CWW_Sbc_ext.sed'))]

        gal_list = []
        for index in indices:
            record = self.param_cat[self.orig_index[index]]
            gal = self._buildParametric(record, sersic_prec, gsparams,
                                        chromatic, self._bandpass, self._sed)
            gal_list.append(gal)

        return gal_list

    @staticmethod
    def _round_sersic(n, sersic_prec):
        return float(int(n/sersic_prec + 0.5)) * sersic_prec

    @staticmethod
    def _buildParametric(record, sersic_prec, gsparams, chromatic, bandpass=None, sed=None):
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
        # Get the status flag for the fits.  Entries 0 and 4 in 'fit_status' are relevant for
        # bulgefit and sersicfit, respectively.
        bstat = record['fit_status'][0]
        sstat = record['fit_status'][4]
        # Get the precomputed bulge-to-total flux ratio for the 2-component fits.
        dvc_btt = record['fit_dvc_btt']
        # Get the precomputed median absolute deviation for the 1- and 2-component fits.
        # These quantities are used to ascertain whether the 2-component fit is really
        # justified, or if the 1-component Sersic fit is sufficient to describe the galaxy
        # light profile.
        bmad = record['fit_mad_b']
        smad = record['fit_mad_s']

        # First decide if we can / should use bulgefit, otherwise sersicfit.  This decision
        # process depends on: the status flags for the fits, the bulge-to-total ratios (if near
        # 0 or 1, just use single component fits), the sizes for the bulge and disk (if <=0 then
        # use single component fits), the axis ratios for the bulge and disk (if <0.051 then use
        # single component fits), and a comparison of the median absolute deviations to see
        # which is better.  The reason for the 0.051 cutoff is that the fits were bound at 0.05
        # as a minimum, so anything below 0.051 generally means that the fitter hit the boundary
        # for the 2-component fits, typically meaning that we don't have enough information to
        # make reliable 2-component fits.
        use_bulgefit = True
        if ( bstat < 1 or bstat > 4 or dvc_btt < 0.1 or dvc_btt > 0.9 or
                np.isnan(dvc_btt) or bparams[9] <= 0 or 
                bparams[1] <= 0 or bparams[11] < 0.051 or bparams[3] < 0.051 or
                smad < bmad ):
            use_bulgefit = False
        # Then check if sersicfit is viable; if not, this galaxy is a total failure.
        # Note that we can avoid including these in the catalog in the first place by using
        # `exclude_fail=True` when making the catalog.
        if sstat < 1 or sstat > 4 or sparams[1] <= 0 or sparams[0] <= 0:
            raise RuntimeError("Cannot make parametric model for this galaxy!")

        # If we're supposed to use the 2-component fits, get all the parameters.
        if use_bulgefit:
            # Bulge parameters:
            # Minor-to-major axis ratio:
            bulge_q = bparams[11]
            # Position angle, now represented as a galsim.Angle:
            bulge_beta = bparams[15]*galsim.radians
            # We have to convert from the stored half-light radius along the major axis, to an
            # azimuthally averaged one (multiplying by sqrt(bulge_q)).  We also have to convert
            # to our native units of arcsec, from units of COSMOS pixels.
            bulge_hlr = cosmos_pix_scale*np.sqrt(bulge_q)*bparams[9]
            # The stored quantity is the surface brightness at the half-light radius.  We have
            # to convert to total flux within an n=4 surface brightness profile.
            bulge_flux = 2.0*np.pi*3.607*(bulge_hlr**2)*bparams[8]/cosmos_pix_scale**2
            # Disk parameters, defined analogously:
            disk_q = bparams[3]
            disk_beta = bparams[7]*galsim.radians
            disk_hlr = cosmos_pix_scale*np.sqrt(disk_q)*bparams[1]
            disk_flux = 2.0*np.pi*1.901*(disk_hlr**2)*bparams[0]/cosmos_pix_scale**2
            bfrac = bulge_flux/(bulge_flux+disk_flux)
            # Make sure the bulge-to-total flux ratio is not nonsense.
            if bfrac < 0 or bfrac > 1 or np.isnan(bfrac):
                raise RuntimeError("Cannot make parametric model for this galaxy")

            # Then make the two components of the galaxy.
            if chromatic:
                # We define the GSObjects with flux=1, then multiply by an SED defined to have
                # the appropriate (observed) magnitude at the redshift in the COSMOS passband.
                z = record['zphot']
                target_bulge_mag = record['mag_auto']-2.5*math.log10(bfrac)
                bulge_sed = sed[0].atRedshift(z).withMagnitude(
                        target_bulge_mag, bandpass)
                bulge = galsim.DeVaucouleurs(half_light_radius=bulge_hlr, gsparams=gsparams)
                bulge *= bulge_sed
                target_disk_mag = record['mag_auto']-2.5*math.log10((1.-bfrac))
                disk_sed = sed[1].atRedshift(z).withMagnitude(target_disk_mag, bandpass)
                disk = galsim.Exponential(half_light_radius=disk_hlr, gsparams=gsparams)
                disk *= disk_sed
            else:
                bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr,
                                                gsparams=gsparams)
                disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr,
                                            gsparams=gsparams)

            # Apply shears for intrinsic shape.
            if bulge_q < 1.:
                bulge = bulge.shear(q=bulge_q, beta=bulge_beta)
            if disk_q < 1.:
                disk = disk.shear(q=disk_q, beta=disk_beta)

            gal = bulge + disk
        else:
            # Do a similar manipulation to the stored quantities for the single Sersic profiles.

            gal_n = sparams[2]
            # Fudge this if it is at the edge of the allowed n values.  Since GalSim (as of
            # #325 and #449) allow Sersic n in the range 0.3<=n<=6, the only problem is that
            # the fits occasionally go as low as n=0.2.  The fits in this file only go to n=6,
            # so there is no issue with too-high values, but we also put a guard on that side
            # in case other samples are swapped in that go to higher value of sersic n.
            if gal_n < 0.3: gal_n = 0.3
            if gal_n > 6.0: gal_n = 6.0
            # GalSim is much more efficient if only a finite number of Sersic n values are used.
            # This (optionally given constructor args) rounds n to the nearest 0.05.
            if sersic_prec > 0.:
                gal_n = COSMOSCatalog._round_sersic(gal_n, sersic_prec)
            gal_q = sparams[3]
            gal_beta = sparams[7]*galsim.radians
            gal_hlr = cosmos_pix_scale*np.sqrt(gal_q)*sparams[1]
            # Below is the calculation of the full Sersic n-dependent quantity that goes into
            # the conversion from surface brightness to flux, which here we're calling
            # 'prefactor'.  In the n=4 and n=1 cases above, this was precomputed, but here we
            # have to calculate for each value of n.
            tmp_ser = galsim.Sersic(gal_n, half_light_radius=gal_hlr, gsparams=gsparams)
            gal_flux = sparams[0] / tmp_ser.xValue(0,gal_hlr) / cosmos_pix_scale**2

            if chromatic:
                gal = galsim.Sersic(gal_n, flux=1., half_light_radius=gal_hlr,
                                    gsparams=gsparams)
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
                gal = galsim.Sersic(gal_n, flux=gal_flux, half_light_radius=gal_hlr,
                                    gsparams=gsparams)

            # Apply shears for intrinsic shape.
            if gal_q < 1.:
                gal = gal.shear(q=gal_q, beta=gal_beta)

        return gal

    def getRealParams(self, index):
        """Get the parameters needed to make a RealGalaxy for a given index."""
        # Used by COSMOSGalaxy to circumvent making the RealGalaxy here and potentially having
        # to pickle the result.  These raw materials should be smaller, so quicker to pickle.
        orig_index = self.orig_index[index]
        gal_image = self.real_cat.getGal(orig_index)
        psf_image = self.real_cat.getPSF(orig_index)
        noise_image, pixel_scale, var = self.real_cat.getNoiseProperties(orig_index)
        return (gal_image, psf_image, noise_image, pixel_scale, var)

    def getParametricRecord(self, index):
        """Get the parametric record for a given index"""
        # Used by _makeSingleGalaxy to circumvent pickling the result.
        record = self.param_cat[self.orig_index[index]]
        # Convert to a dict, since on some systems, the numpy record doesn't seem to 
        # pickle correctly.
        #record_dict = { k:record[k] for k in record.dtype.names }  # doesn't work in python 2.6
        record_dict = dict( ((k,record[k]) for k in record.dtype.names) )  # equivalent.
        return record_dict

    def canMakeReal(self):
        """Is it permissible to call makeGalaxy with gal_type='real'?"""
        return self.use_real

    @staticmethod
    def _makeSingleGalaxy(cosmos_catalog, index, gal_type, noise_pad_size=5, deep=False,
                          rng=None, sersic_prec=0.05, gsparams=None):
        # A static function that mimics the functionality of COSMOSCatalog.makeGalaxy()
        # for single index and chromatic=False.
        # The only point of this class is to circumvent some pickling issues when using
        # config objects with type : COSMOSGalaxy.  It's a staticmethod, which means it
        # cannot use any self attributes.  Just methods.  (Which also means we can use it
        # through a proxy COSMOSCatalog object, which we need for the config layer.)

        if not cosmos_catalog.canMakeReal():
            if gal_type is None:
                gal_type = 'parametric'
            elif gal_type != 'parametric':
                raise ValueError("Only 'parametric' galaxy type is allowed when use_real == False")

        if gal_type not in ['real', 'parametric']:
            raise ValueError("Invalid galaxy type %r"%gal_type)

        if gal_type == 'real' and rng is None:
            rng = galsim.BaseDeviate()

        if gal_type == 'real':
            real_params = cosmos_catalog.getRealParams(index)
            gal = galsim.RealGalaxy(real_params, noise_pad_size=noise_pad_size, rng=rng,
                                    gsparams=gsparams)
        else:
            record = cosmos_catalog.getParametricRecord(index)
            gal = COSMOSCatalog._buildParametric(record, sersic_prec, gsparams, chromatic=False)

        # If deep, rescale the size and flux
        if deep:
            # Rescale the flux to get a limiting mag of 25 in F814W.  Current limiting mag is 23.5,
            # so it's a magnitude difference of 1.5.  Make the galaxies a factor of 0.6 smaller and
            # appropriately fainter.
            flux_factor = 10.**(-0.4*1.5)
            size_factor = 0.6
            gal = gal.dilate(size_factor) * flux_factor

        # Store the orig_index as gal.index, since the above RealGalaxy initialization
        # just sets it as 0.  Plus, it isn't set at all if we make a parametric galaxy.
        # And if we are doing the deep scaling, then it gets messed up by that
        gal.index = cosmos_catalog.getOrigIndex(index)

        return gal

    # Since this is a function, not a class, need to use an unconventional location for defining
    # these config parameters.  Also, I thought it would make sense to attach them to the
    # _makeSingleGalaxy method.  But that doesn't work, since it is technically a staticmethod
    # object, not a normal function.  So we attach these to makeGalaxy instead.
    makeGalaxy._req_params = {}
    makeGalaxy._opt_params = { "index" : int,
                               "gal_type" : str,
                               "noise_pad_size" : float,
                               "deep" : bool,
                               "sersic_prec": float,
                             }
    makeGalaxy._single_params = []
    makeGalaxy._takes_rng = True


