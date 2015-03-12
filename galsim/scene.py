# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
from galsim import pyfits
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

    After getting the catalogs, there is a method makeObj() that can make an object corresponding to
    any chosen galaxy in the catalog (whether real or parametric).  See
    help(galsim.COSMOSCatalog.makeObj) for more information.  As an interesting application and
    example of the usage of these routines, consider the following code:

        >>> im_size = 64
        >>> pix_scale = 0.05
        >>> bandpass = galsim.Bandpass('share/wfc_F814W.dat.gz',
                                       wave_type='ang').thin().withZeropoint(25.94)
        >>> real_cat = galsim.COSMOSCatalog()
        >>> param_cat = galsim.COSMOSCatalog(use_real=False)
        >>> psf = galsim.OpticalPSF(diam=2.4, lam=1000.) # bigger than HST F814W PSF.
        >>> indices = np.arange(10)
        >>> real_gal_list = real_cat.makeObj(indices, pad_size=im_size*pix_scale)
        >>> param_gal_list = param_cat.makeObj(indices, chromatic=True)
        >>> for ind in indices:
        >>>     real_obj = galsim.Convolve(real_gal_list[ind], psf)
        >>>     param_obj = galsim.Convolve(param_gal_list[ind], psf)
        >>>     im_real = galsim.Image(im_size, im_size)
        >>>     im_param = galsim.Image(im_size, im_size)
        >>>     real_obj.drawImage(image=im_real, scale=pix_scale)
        >>>     param_obj.drawImage(bandpass, image=im_param, scale=pix_scale)
        >>>     im_real.write('im_real_'+str(ind)+'.fits')
        >>>     im_param.write('im_param_'+str(ind)+'.fits')

    This code snippet will draw images of the first 10 objects in the COSMOS catalog, at slightly
    lower resolution than in COSMOS, with a real image and its parametric representation for each of
    those objects.  Note that we are automatically excluding galaxies that do not have parametric
    representations.  These are rare and do not occur in the first ten objects in the catalog, which
    is why we can assume that the real and parametric objects will be comparable.

    Initialization
    --------------

    @param file_name    The file containing the catalog. [default: None, which will look for the
                        COSMOS catalog in $PREFIX/share/galsim.  It will raise an exception if the
                        catalog is not there telling you to run galsim_download_cosmos.]
    @param use_real     Use realistic galaxies or parametric ones?  [default: True]
    @param preload      Keyword that is only used for real galaxies, not parametric ones, to choose
                        whether to preload the header information.  If `preload=True`, the bulk of  
                        the I/O time is in the constructor.  If `preload=False`, there is
                        approximately the same total I/O time (assuming you eventually use most of
                        the image files referenced in the catalog), but it is spread over the
                        various calls to getGal() and getPSF().  [default: False]
    @param image_dir    Keyword that is only used for real galaxies, not parametric ones, to specify
                        the directory of the image files.
                        If a string containing no `/`, it is the relative path from the location of
                        the catalog file to the directory containing the galaxy/PDF images.
                        If a path (a string containing `/`), it is the full path to the directory
                        containing the galaxy/PDF images. [default: None]
    @param dir          The directory of catalog file. [default: None]
    @param noise_dir    Keyword that is only used for real galaxies, not parametric ones.
                        The directory of the noise files if different from the directory of the 
                        image files.  [default: image_dir]
    @param deep_sample  Modify fluxes and sizes of galaxies in order to roughly simulate an F814W<25
                        sample? [default: False]
    @param exclude_fail For catalogs of parametric galaxies, exclude those that have failures in the
                        parametric fits?  [default: True]

    Attributes
    ----------

    After construction, the following attributes are available:

    nobjects     The number of objects in the catalog
    obj_type     Either 'real' or 'parametric', depending on which way of representing galaxies was
                 chosen.
    deep         Either 'True' or 'False', depending on whether modifications of the object sizes
                 and fluxes were used to simulate a deeper I<25 sample ('True') or not.

    """
    def __init__(self, file_name=None, use_real=True, image_dir=None, dir=None, noise_dir=None,
                      preload=False, deep_sample=False, exclude_fail=True):
        # Make fake deeper sample if necessary.
        if deep_sample:
            # Rescale the flux to get a limiting mag of 25 in F814W.  Current limiting mag is 23.5,
            # so it's a magnitude difference of 1.5.  Make the galaxies a factor of 0.6 smaller and
            # appropriately fainter.
            self.flux_factor = 10.**(-0.4*1.5)
            self.size_factor = 0.6
            self.deep = True
        else:
            self.flux_factor = 1.0
            self.size_factor = 1.0
            self.deep = False

        if use_real:
            # First, do the easy thing: real galaxies.  We make the galsim.RealGalaxyCatalog()
            # constructor do most of the work.
            self.cat = galsim.RealGalaxyCatalog(
                file_name, image_dir=image_dir, dir=dir, preload=preload, noise_dir=noise_dir)
            self.obj_type = 'real'
            self.nobjects = self.cat.nobjects

        else:
            from real import _parse_files_dirs

            # Find the file.
            use_file_name, _, _ = \
                _parse_files_dirs(file_name, image_dir, dir, noise_dir)

            # Read in data.
            cat = pyfits.getdata(use_file_name)

            # If requested, select galaxies based on existence of a usable fit.
            if exclude_fail:
                sersicfit_status = cat['fit_status'][:,4]
                bulgefit_status = cat['fit_status'][:,0]
                use_fit_ind = np.where(
                    (sersicfit_status > 0) &
                    (sersicfit_status < 5) &
                    (bulgefit_status > 0) &
                    (bulgefit_status < 5)
                    )[0]
                self.cat = cat[use_fit_ind]
            else:
                self.cat = cat
            self.obj_type = 'parametric'
            self.nobjects = len(self.cat)

    def makeObj(self, indices, chromatic=False, pad_size=None):
        """
        Routine to construct GSObjects corresponding to catalog entries with particular indices.

        The fluxes are set such that drawing into an image with the COSMOS bandpass and pixel scale
        should give the right pixel values to mimic the actual COSMOS image.

        @param indices    The indices of the catalog entries for which GSObjects should be
                          constructed. This should be either a single number or an iterable.
        @param chromatic  Make this a chromatic object, or not?  [default: False]
                          It is important to bear in mind that we do not actually have
                          spatially-resolved color information for these galaxies, so this keyword
                          can only be True if we are using parametric galaxies.  Even then, we
                          simply do the most arbitrary thing possible, which is to assign bulges an
                          elliptical SED, disks a disk-like SED, and Sersic galaxies with
                          intermediate values of n some intermediate SED.  We assume that the
                          photometric redshift is the correct redshift for these galaxies (which is
                          a good assumption for COSMOS 30-band photo-z for these bright galaxies).
                          For the given SED and redshift, we then normalize to give the right
                          (observed) flux in F814W.  Note that for a mock "deep" sample, the
                          redshift distributions of the galaxies would be modified, which is not
                          included here.
        @param pad_size   For realistic galaxies, the size of region requiring noise padding, in
                          arcsec.  If None, then a region that is 0.25 arcsec in size is used.
                          [default: None]

        @returns A list of GSObjects or chromatic objects representing the galaxy of interest,
        unless `indices` is just a single number, in which case the object for that index is
        returned directly.
        """
        # Check whether this is a COSMOSCatalog meant to represent real or parametric objects, then
        # call the appropriate helper routine for that case.
        if self.obj_type == 'real':
            if pad_size is None:
                pad_size=0.25 # random and not completely ridiculous guess in arcsec
            if chromatic:
                raise RuntimeError("Cannot yet make real chromatic galaxies!")
            return self._makeReal(indices, pad_size=pad_size)
        else:
            return self._makeParam(indices, chromatic=chromatic)

    def _makeReal(self, indices, pad_size):
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        obj_list = []
        noise_pad_size = int(np.ceil(pad_size * np.sqrt(2.)))
        for index in indices:
            gal = galsim.RealGalaxy(self.cat, index=index, noise_pad_size=noise_pad_size)

            # Rescale the galaxy size.
            if self.deep:
                gal.applyDilation(self.size_factor)
                gal *= self.flux_factor
            obj_list.append(gal)

        if len(indices)==1:
            return obj_list[0]
        else:
            return obj_list

    def _makeParam(self, indices, chromatic=False):
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        if chromatic:
            # Defer making the Bandpass and reading in SEDs until we actually are going to use them.
            # It's not a huge calculation, but the thin() call especially isn't trivial.
            if not hasattr(self, '_COSMOS_bandpass'):
                self._COSMOS_bandpass = galsim.Bandpass(
                    os.path.join(galsim.meta_data.share_dir, 'wfc_F814W.dat.gz'),
                    wave_type='ang').thin().withZeropoint(25.94)

                # Read in some SEDs.
                self.sed_bulge = \
                    galsim.SED(os.path.join(galsim.meta_data.share_dir,'CWW_E_ext.sed'))
                self.sed_disk = \
                    galsim.SED(os.path.join(galsim.meta_data.share_dir,'CWW_Scd_ext.sed'))
                self.sed_intermed = \
                    galsim.SED(os.path.join(galsim.meta_data.share_dir,'CWW_Sbc_ext.sed'))

            bandpass = self._COSMOS_bandpass

        obj_list = []
        for index in indices:
            record = self.cat[index]

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
            # These quantities are used to ascertain whether the 2-component fit is really justified, or
            # if the 1-component Sersic fit is sufficient to describe the galaxy light profile.
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
            if bstat<1 or bstat>4 or dvc_btt<0.1 or dvc_btt>0.9 or np.isnan(dvc_btt) or bparams[9]<=0 or \
                    bparams[1]<=0 or bparams[11]<0.051 or bparams[3]<0.051 or smad<bmad:
                use_bulgefit = False
            # Then check if sersicfit is viable; if not, this object is a total failure.
            # Note that we can avoid including these in the catalog in the first place by using
            # `exclude_fail=True` when making the catalog.
            if sstat<1 or sstat>4 or sparams[1]<=0 or sparams[0]<=0:
                raise RuntimeError("Cannot make parametric model for this object!")

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
                    raise RuntimeError("Cannot make parametric model for this object")

                # Then make the two components of the object.
                if chromatic:
                    # We define the GSObjects with flux=1, then multiply by an SED defined to have
                    # the appropriate (observed) magnitude at the object redshift in the COSMOS
                    # passband.
                    z = record['zphot']
                    target_bulge_mag = record['mag_auto']-2.5*math.log10(bfrac*self.flux_factor)
                    bulge_sed = \
                        self.sed_bulge.atRedshift(z).withMagnitude(target_bulge_mag, bandpass)
                    bulge = galsim.DeVaucouleurs(half_light_radius=self.size_factor*bulge_hlr)
                    bulge *= bulge_sed
                    target_disk_mag = record['mag_auto']-2.5*math.log10((1.-bfrac)*self.flux_factor)
                    disk_sed = self.sed_disk.atRedshift(z).withMagnitude(target_disk_mag, bandpass)
                    disk = galsim.Exponential(half_light_radius=self.size_factor*disk_hlr)
                    disk *= disk_sed
                else:
                    bulge = galsim.DeVaucouleurs(flux=self.flux_factor*bulge_flux,
                                                 half_light_radius=self.size_factor*bulge_hlr)
                    disk = galsim.Exponential(flux=self.flux_factor*disk_flux,
                                              half_light_radius=self.size_factor*disk_hlr)
                # Apply shears for intrinsic shape.
                if bulge_q < 1.:
                    bulge = bulge.shear(q=bulge_q, beta=bulge_beta)
                if disk_q < 1.:
                    disk = disk.shear(q=disk_q, beta=disk_beta)
                obj_list.append(bulge+disk)
            else:
                # Do a similar manipulation to the stored quantities for the single Sersic profiles.

                gal_n = sparams[2]
                # Fudge this if it is at the edge of the allowed n values.  Since GalSim (as of #325 and
                # #449) allow Sersic n in the range 0.3<=n<=6, the only problem is that the fits
                # occasionally go as low as n=0.2.  The fits in this file only go to n=6, so there
                # is no issue with too-high values, but we also put a guard on that side in case
                # other samples are swapped in that go to higher value of sersic n.
                if gal_n < 0.3: gal_n = 0.3
                if gal_n > 6.0: gal_n = 6.0
                gal_q = sparams[3]
                gal_beta = sparams[7]*galsim.radians
                gal_hlr = cosmos_pix_scale*np.sqrt(gal_q)*sparams[1]
                # Below is the calculation of the full Sersic n-dependent quantity that goes into
                # the conversion from surface brightness to flux, which here we're calling
                # 'prefactor'.  In the n=4 and n=1 cases above, this was precomputed, but here we
                # have to calculate for each value of n.
                tmp_ser = galsim.Sersic(gal_n, half_light_radius=gal_hlr)
                gal_flux = sparams[0] / tmp_ser.xValue(0,gal_hlr) / cosmos_pix_scale**2

                if chromatic:
                    gal = galsim.Sersic(gal_n, flux=1., half_light_radius=self.size_factor*gal_hlr)
                    if gal_n < 1.5:
                        use_sed = self.sed_disk
                    elif gal_n >= 1.5 and gal_n < 3.0:
                        use_sed = self.sed_intermed
                    else:
                        use_sed = self.sed_bulge
                    target_mag = record['mag_auto']-2.5*math.log10(self.flux_factor)
                    z = record['zphot']
                    gal *= use_sed.atRedshift(z).withMagnitude(target_mag, bandpass)
                else:
                    gal = galsim.Sersic(gal_n, flux=self.flux_factor*gal_flux,
                                        half_light_radius=self.size_factor*gal_hlr)
                if gal_q < 1.:
                    gal = gal.shear(q=gal_q, beta=gal_beta)
                obj_list.append(gal)

        if len(indices)==1:
            return obj_list[0]
        else:
            return obj_list
