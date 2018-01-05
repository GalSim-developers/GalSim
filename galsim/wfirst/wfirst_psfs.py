# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

from past.builtins import basestring
import galsim
import galsim.wfirst
import numpy as np
import os

"""
@file wfirst_psfs.py

Part of the WFIRST module.  This file includes routines needed to define a realistic PSF for WFIRST.
"""

# Define a default set of bandpasses for which this routine works.
default_bandpass_list = ['J129', 'F184', 'W149', 'Y106', 'Z087', 'H158']
# Prefix for files containing information about Zernikes for each SCA for cycle 7.
zemax_filepref = "WFIRST_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727"
zemax_filesuff = '.txt'
zemax_wavelength = 1293. #nm

def getPSF(SCA, bandpass, SCA_pos=None, approximate_struts=False, n_waves=None, extra_aberrations=None,
           logger=None, wavelength=None, high_accuracy=False,
           gsparams=None):
    """
    Get the PSF for WFIRST observations (either a single PSF or a list, depending on the inputs).

    For each PSF that the user wants to create, they must provide a list or NumPy array of SCA
    numbers, and a list of strings indicating the bandpass; this is used when setting up the pupil
    plane configuration and when interpolating chromatic information, if requested.  In general the
    approach is that if a list of length 'n_psf' is given for `SCA`, then it is assumed that the
    user would like 'n_psf' PSFs.  But if just a single value for the other kwargs, it is assumed
    that the single value given for that kwarg should be used for all 'n_psf' PSFs.

    This routine carries out linear interpolation of the aberrations within a given SCA, based on
    the WFIRST Cycle 7 specification of the aberrations as a function of focal plane position.

    The default is to do the calculations using the full specification of the WFIRST pupil plane,
    which is a costly calculation in terms of memory.  For this, we use the provided pupil plane for
    long- and short-wavelength bands for Cycle 7. To avoid using the full pupil plane configuration,
    use the optional keyword `approximate_struts`.  In this case, the pupil plane will have the
    correct obscuration and number of struts, but the struts will be purely radial and evenly spaced
    instead of the true configuration.  The simplicity of this arrangement leads to a much faster
    calculation, and somewhat simplifies the configuration of the diffraction spikes.  Also note
    that currently the orientation of the struts is fixed, rather than rotating depending on the
    orientation of the focal plane.  Rotation of the PSF can easily be affected by the user via

       psf = galsim.wfirst.getPSF(...).rotate(angle)

    which will rotate the entire PSF (including the diffraction spikes and any other features).

    The calculation takes advantage of the fact that the diffraction limit and aberrations have a
    simple, understood wavelength-dependence.  (The WFIRST project webpage for Cycle 7 does in fact
    provide aberrations as a function of wavelength, but the deviation from the expected chromatic
    dependence is sub-percent so we neglect it here.)  For reference, the script use to parse the
    Zernikes given on the webpage and create the files in the GalSim repository can be found in
    `devel/external/parse_wfirst_zernikes_1217.py`.  The resulting chromatic object can be used to
    draw into any of the WFIRST bandpasses, though the pupil plane configuration will only be
    correct for those bands in the same range (i.e., long- or short-wavelength bands).

    For applications that require very high accuracy in the modeling of the PSF, with very limited
    aliasing, the `high_accuracy` option can be set to True.  When using this option, the MTF has a
    value below 1e-4 for all wavenumbers above the band limit when using `approximate_struts=True`,
    or below 3e-4 when using `approximate_struts=False`.  In contrast, when `high_accuracy=False`
    (the default), there are some bumps in the MTF above the band limit that reach an amplitude of
    ~1e-2.

    By default, no additional aberrations are included above the basic design.  However, users can
    provide an optional keyword `extra_aberrations` that will be included on top of those that are
    part of the design.  This should be in the same format as for the ChromaticOpticalPSF class,
    with units of waves at the fiducial wavelength, 1293 nm. Currently, only aberrations up to order
    22 (Noll convention) are simulated.  For WFIRST, the current tolerance for additional
    aberrations is a total of 90 nanometers RMS:
    http://wfirst.gsfc.nasa.gov/science/sdt_public/wps/references/instrument/README_AFTA_C5_WFC_Zernike_and_Field_Data.pdf
    distributed largely among coma, astigmatism, trefoil, and spherical aberrations (NOT defocus).
    This information might serve as a guide for reasonable `extra_aberrations` inputs.

    Jitter and charge diffusion are, by default, not included.  Users who wish to include these can
    find some guidelines for typical length scales of the Gaussians that can represent these
    effects, and convolve the ChromaticOpticalPSF with appropriate achromatic Gaussians.

    The PSFs are always defined assuming the user will specify length scales in arcsec.

    @param    SCA                  Single value or iterable specifying the SCA(s) for which the 
                                   PSF should be loaded.
    @param    bandpass             Single string or list of strings specifying the bandpass to use
                                   when defining the pupil plane configuration and/or interpolation
                                   of chromatic PSFs.  If `approximate_struts` is True (which means
                                   we do not use a realistic pupil plane configuration) and
                                   `n_waves` is None (no interpolation of chromatic PSFs) then
                                   'bandpass' can be None.
    @param    SCA_pos              Single galsim.PositionD or list of galsim.PositionDs indicating
                                   the position within the SCA for which the PSF should be created.
                                   If None, the exact center of the SCA is chosen. [default: None]
    @param    approximate_struts   Should the routine use an approximate representation of the pupil
                                   plane, with 6 equally-spaced radial struts, instead of the exact
                                   representation of the pupil plane?  Setting this parameter to
                                   True will lead to faster calculations, with a slightly less
                                   realistic PSFs.  Can be a single item or list. [default: False]
    @param    n_waves              Number of wavelengths to use for setting up interpolation of the
                                   chromatic PSF objects, which can lead to much faster image
                                   rendering.  If None, then no interpolation is used. Note that
                                   users who want to interpolate can always set up the interpolation
                                   later on even if they do not do so when calling getPSF(). Can be
                                   a single item or a list.
                                   [default: None]
    @param    extra_aberrations    Array of extra aberrations to include in the PSF model, on top of
                                   those that are part of the WFIRST design.  These should be
                                   provided in units of waves at the fiducial wavelength of 1293 nm,
                                   as an array of length 23 with entries 4 through 22 corresponding
                                   to defocus through the 22nd Zernike in the Noll convention.
                                   Can be a single array or a list of arrays.
                                   [default: None]
    @param    logger               A logger object for output of progress statements if the user
                                   wants them.  [default: None]
    @param    wavelength           An option to get an achromatic PSF for a single wavelength, for
                                   users who do not care about chromaticity of the PSF.  If None,
                                   then the fully chromatic PSF is returned.  Alternatively the user
                                   should supply either (a) a wavelength in nanometers, and they
                                   will get achromatic OpticalPSF objects for that wavelength, or
                                   (b) a bandpass object, in which case they will get achromatic
                                   OpticalPSF objects defined at the effective wavelength of that
                                   bandpass.  Can be a single item or a list.
                                   [default: False]
    @param    high_accuracy        If True, make higher-fidelity representations of the PSF in
                                   Fourier space, to minimize aliasing (see plots on
                                   https://github.com/GalSim-developers/GalSim/issues/661 for more
                                   details).  This setting is more expensive in terms of time and
                                   RAM, and may not be necessary for many applications.
                                   Can be a single item or a list.
                                   [default: False]
    @param    gsparams             An optional GSParams argument.  See the docstring for GSParams
                                   for details. Can be a single item or a list.  [default: None]
    @returns  A single PSF object, or a list of PSF objects (either ChromaticOpticalPSFs or
              OpticalPSFs depending on the inputs).
    """
    # Deal with inputs:
    # Use the 'SCA' input to figure out whether we are getting a list of PSFs or what.
    if not hasattr(SCA, '__iter__'):
        SCA = [SCA]
    n_psf = len(SCA)

    # Deal with other kwargs appropriately.
    # Bandpass should either be a list with the same length as 'SCA', or a single value (in which
    # case all PSFs will have the same bandpass).  We have a helper routine for this.
    bandpass = _expand_list(bandpass, n_psf)

    # SCA_pos: if None, then all should just be center of the SCA.
    #          if given, then same rules as for Bandpass apply.
    if SCA_pos is None:
        SCA_pos = galsim.PositionD(galsim.wfirst.n_pix/2, galsim.wfirst.n_pix/2)
    SCA_pos = _expand_list(SCA_pos, n_psf)

    # The rest (except logger, bandpass, and SCA_pos) just get expanded out to a list of length
    # n_psf, if they aren't already in that form.
    approximate_struts = _expand_list(approximate_struts, n_psf)
    n_waves = _expand_list(n_waves, n_psf)
    extra_aberrations = _expand_list(extra_aberrations, n_psf)
    wavelength = _expand_list(wavelength, n_psf)
    high_accuracy = _expand_list(high_accuracy, n_psf)
    gsparams = _expand_list(gsparams, n_psf)

    # Parse the bandpasses to see which pupil plane image is needed, if approximate_struts is False
    # (otherwise just say None).
    pupil_plane_type = [None] * n_psf
    for i in range(n_psf):
        if not approximate_struts[i]:
            if bandpass[i] in galsim.wfirst.longwave_bands:
                pupil_plane_type[i] = 'long'
            elif bandpass[i] in galsim.wfirst.shortwave_bands:
                pupil_plane_type[i] = 'short'
            else:
                raise ValueError("Bad bandpass input: %s"%bandpass[i])
        else:
            # Sanity checking:
            # If we need to use bandpass info, require that it be one of the defaults.
            # If we do not need to use bandpass info, allow it to be None.
            if n_waves[i] is not None:
                if bandpass[i] not in default_bandpass_list:
                    raise ValueError("Bad bandpass input: %s"%bandpass[i])
            else:
                if bandpass[i] not in default_bandpass_list and bandpass[i] is not None:
                    raise ValueError("Bad bandpass input: %s"%bandpass[i])

    # Check cases where reusing is possible.
    # That would be if the SCA and SCA_pos and extra_aberrations are the same,
    # as are the chromatic info.
    reuse_index = [None] * n_psf
    for i in range(n_psf):
        for j in range(i+1, n_psf):
            if SCA[i]==SCA[j] and SCA_pos[i]==SCA_pos[j] and \
                    approximate_struts[i]==approximate_struts[j] and \
                    extra_aberrations[i]==extra_aberrations[j] and \
                    high_accuracy[i]==high_accuracy[j] and \
                    gsparams[i]==gsparams[j] and \
                    pupil_plane_type[i]==pupil_plane_type[j] and \
                    wavelength[i]==wavelength[j] and \
                    n_waves[i]==n_waves[j] and \
                    reuse_index[j] is None:
                reuse_index[j] = i

    # Now loop over the options and call _get_single_PSF() for each one, unless a prior one can be
    # reused.
    psfs = []
    for i in range(n_psf):
        if reuse_index[i] is None:
            psfs.append(
                _get_single_PSF(SCA[i], bandpass[i], SCA_pos[i], approximate_struts[i],
                                n_waves[i], extra_aberrations[i], logger, wavelength[i],
                                high_accuracy[i], pupil_plane_type[i], gsparams[i])
                )
        else:
            psfs.append(psfs[reuse_index[i]])

    if n_psf==1:
        return psfs[0]
    else:
        return psfs

def _get_single_PSF(SCA, bandpass, SCA_pos, approximate_struts,
                    n_waves, extra_aberrations, logger, wavelength,
                    high_accuracy, pupil_plane_type, gsparams):
    """Routine for making a single PSF.  This gets called by getPSF() after it parses all the
       options that were passed in.  Users will not directly interact with this routine."""
    # Deal with some accuracy settings.
    if high_accuracy:
        if approximate_struts:
            oversampling = 3.5
        else:
            oversampling = 2.0

            # In this case, we need to pad the edges of the pupil plane image, so we cannot just use
            # the stored file.
            if pupil_plane_type == 'long':
                tmp_pupil_plane_im = galsim.fits.read(galsim.wfirst.pupil_plane_file_longwave)
            else:
                tmp_pupil_plane_im = galsim.fits.read(galsim.wfirst.pupil_plane_file_shortwave)
            old_bounds = tmp_pupil_plane_im.bounds
            new_bounds = old_bounds.withBorder((old_bounds.xmax+1-old_bounds.xmin)/2)
            pupil_plane_im = galsim.Image(bounds=new_bounds)
            pupil_plane_im[old_bounds] = tmp_pupil_plane_im
            pupil_plane_scale = galsim.wfirst.pupil_plane_scale
    else:
        if approximate_struts:
            oversampling = 1.5
        else:
            oversampling = 1.2
            if pupil_plane_type == 'long':
                pupil_plane_im = galsim.wfirst.pupil_plane_file_longwave
            else:
                pupil_plane_im = galsim.wfirst.pupil_plane_file_shortwave
            pupil_plane_scale = galsim.wfirst.pupil_plane_scale

    if wavelength is None:
        if n_waves is not None:
            # To decide the range of wavelengths to use, check the bandpass.
            bandpass_dict = galsim.wfirst.getBandpasses()
            # Then find the blue and red limit to be used for the imaging bandpasses overall.
            blue_limit, red_limit = _find_limits(bandpass, bandpass_dict)
    elif isinstance(wavelength, float):
        wavelength_nm = wavelength
    elif isinstance(wavelength, galsim.Bandpass):
        wavelength_nm = wavelength.effective_wavelength
    else:
        raise TypeError("wavelength should either be a Bandpass, float, or None.")

    # Start reading in the aberrations for that SCA
    if logger: logger.debug('Beginning to get the PSF aberrations.')
    aberrations, x_pos, y_pos = _read_aberrations(SCA)
    # Do bilinear interpolation, unless we're exactly at the center (default).
    use_aberrations = _interp_aberrations_bilinear(aberrations, x_pos, y_pos, SCA_pos)

    if extra_aberrations is not None:
        use_aberrations += extra_aberrations
    # We don't want to use piston, tip, or tilt aberrations.  The former doesn't affect the
    # appearance of the PSF, and the latter cause centroid shifts.  So, we set the first 4
    # numbers (corresponding to a place-holder, piston, tip, and tilt) to zero.
    use_aberrations[0:4] = 0.

    # Now set up the PSF, including the option to simplify the pupil plane.
    if wavelength is None:
        if approximate_struts:
            PSF = galsim.ChromaticOpticalPSF(
                lam=zemax_wavelength,
                diam=galsim.wfirst.diameter, aberrations=use_aberrations,
                obscuration=galsim.wfirst.obscuration, nstruts=6,
                oversampling=oversampling, gsparams=gsparams)
        else:
            PSF = galsim.ChromaticOpticalPSF(
                lam=zemax_wavelength,
                diam=galsim.wfirst.diameter, aberrations=use_aberrations,
                obscuration=galsim.wfirst.obscuration,
                pupil_plane_im=pupil_plane_im,
                pupil_plane_scale=pupil_plane_scale,
                oversampling=oversampling, pad_factor=2., gsparams=gsparams)
        if n_waves is not None:
            PSF = PSF.interpolate(waves=np.linspace(blue_limit, red_limit, n_waves),
                                  oversample_fac=1.5)
    else:
        tmp_aberrations = use_aberrations * zemax_wavelength / wavelength_nm
        if approximate_struts:
            PSF = galsim.OpticalPSF(lam=wavelength_nm, diam=galsim.wfirst.diameter,
                                    aberrations=tmp_aberrations,
                                    obscuration=galsim.wfirst.obscuration, nstruts=6,
                                    oversampling=oversampling, gsparams=gsparams)
        else:
            PSF = galsim.OpticalPSF(lam=wavelength_nm, diam=galsim.wfirst.diameter,
                                    aberrations=tmp_aberrations,
                                    obscuration=galsim.wfirst.obscuration,
                                    pupil_plane_im=pupil_plane_im,
                                    pupil_plane_scale=pupil_plane_scale,
                                    oversampling=oversampling, pad_factor=2., gsparams=gsparams)

    return PSF

def storePSFImages(PSF_dict, filename, bandpass_list=None, clobber=False):
    """
    This is a routine to store images of chromatic WFIRST PSFs in different bands for each SCA.  It
    takes an output dict of PSFs (`PSF_dict`) directly from getPSF().  The output will be a file
    (`filename`) that has all the images, along with an HDU that contains a FITS table indicating
    the bandpasses, SCAs, and other information needed to reconstruct the PSF information.

    This routine is not meant to work for PSFs from getPSF() that are completely achromatic.  The
    reason for this is that those PSFs are quite fast to generate, so there is little benefit to
    storing them.

    @param PSF_dict            A dict of PSF objects for each SCA, in the same format as output by
                               the getPSF() routine (though it can take versions that have been
                               modified, for example in the inclusion of an SED).
    @param filename            The name of the file to which the images and metadata should be
                               written; extension should be *.fits.
    @param bandpass_list       A list of bandpass names for which images should be generated and
                               stored.  If None, all WFIRST imaging passbands are used.
                               [default: None]
    @param clobber             Should the routine clobber `filename` (if they already exist)?
                               [default: False]
    """
    from galsim._pyfits import pyfits
    # Check for sane input PSF_dict.
    if len(PSF_dict) == 0 or len(PSF_dict) > galsim.wfirst.n_sca or \
            min(PSF_dict.keys()) < 1 or max(PSF_dict.keys()) > galsim.wfirst.n_sca:
        raise galsim.GalSimError("PSF_dict must come from getPSF().")

    # Check if file already exists and warn about clobbering.
    if os.path.isfile(filename):
        if clobber:
            os.remove(filename)
        else:
            raise OSError("Output file %r already exists"%filename)

    # Check that bandpass list input is okay.  It should be strictly a subset of the default list of
    # bandpasses.
    if bandpass_list is None:
        bandpass_list = default_bandpass_list
    else:
        if not isinstance(bandpass_list[0], basestring):
            raise TypeError("Expected input list of bandpass names.")
        if not set(bandpass_list).issubset(default_bandpass_list):
            raise galsim.GalSimValueError("Invalid values in bandpass_list", bandpass_list,
                                          default_bandpass_list)

    # Get all the WFIRST bandpasses.
    bandpass_dict = galsim.wfirst.getBandpasses()

    # Loop through making images and lists of their relevant parameters.
    im_list = []
    bp_name_list = []
    SCA_index_list = []
    for SCA in PSF_dict:
        PSF = PSF_dict[SCA]
        if not isinstance(PSF, galsim.ChromaticOpticalPSF) and \
                not isinstance(PSF, galsim.InterpolatedChromaticObject):
            raise galsim.GalSimValueError("PSFs are not ChromaticOpticalPSFs.", PSF_dict)
        star = galsim.Gaussian(sigma=1.e-8, flux=1.)

        for bp_name in bandpass_list:
            bandpass = bandpass_dict[bp_name]
            star_sed = galsim.SED(lambda x:1, 'nm', 'flambda').withFlux(1, bandpass)
            obj = galsim.Convolve(star*star_sed, PSF)

            im = obj.drawImage(bandpass, scale=0.5*galsim.wfirst.pixel_scale,
                               method='no_pixel')
            im_list.append(im)
            bp_name_list.append(bp_name)
            SCA_index_list.append(SCA)

    # Save images to file.
    galsim.fits.writeMulti(im_list, filename, clobber=clobber)

    # Add data to file, after constructing a FITS table.  Watch out for clobbering.
    bp_names = pyfits.Column(name='bandpass', format='A10', array=np.array(bp_name_list))
    SCA_indices = pyfits.Column(name='SCA', format='J', array=np.array(SCA_index_list))
    cols = pyfits.ColDefs([bp_names, SCA_indices])
    tbhdu = pyfits.BinTableHDU.from_columns(cols)
    f = pyfits.open(filename, mode='update')
    f.append(tbhdu)
    f.flush()
    f.close()

def loadPSFImages(filename):
    """
    Get an achromatic representation of the WFIRST PSF in each passband (originally generated for an
    object with a flat SED).

    If the user has generated WFIRST PSFs and stored their images in each passband using getPSF()
    followed by storePSFImages(), then loadPSFImages() can read in those stored images and
    associated data, and return an InterpolatedImage for each one.  These are intrinsically not
    chromatic objects themselves, so they can be used only if the user does not care about the
    variation of the PSF with wavelength within each passband.  In that case, use of loadPSFImages()
    can represent significant time savings compared to doing the full PSF calculation each time.

    @param filename    Name of file containing the PSF images and metadata from storePSFImages().

    @returns A nested dict containing the GSObject representing the PSF, where the keys are the
    bandpasses, and the values are dicts containing the PSF for each SCA for which results were
    tabulated.
    """
    # Get the image data and metadata.
    hdu, hdu_list, fin = galsim.fits.readFile(filename)
    metadata_hdu = hdu_list.pop()
    im_list = galsim.fits.readMulti(hdu_list=hdu_list)
    bp_list = list(metadata_hdu.data.bandpass)
    # In python3, convert from bytes to str
    bp_list = [ str(bp.decode()) for bp in bp_list ]
    SCA_list = list(metadata_hdu.data.SCA)
    galsim.fits.closeHDUList(hdu_list, fin)

    # Set up the dict of PSF objects, indexed by bandpass (and then SCA).
    full_PSF_dict = {}
    for band_name in set(bp_list):
        band_PSF_dict = {}

        # Find all indices in `bp_list` that correspond to this bandpass.
        bp_indices = []
        idx = -1
        while True:
            try:
                idx = bp_list.index(band_name, idx+1)
                bp_indices.append(idx)
            except ValueError:
                break

        for SCA in SCA_list:
            # Now find which element has both the right band_name and is for this SCA.  There might
            # not be any, depending on what exactly was stored.
            use_idx = -1
            for index in bp_indices:  # pragma: no branch
                if SCA_list[index] == SCA:
                    use_idx = index
                    break

            # Now we know which PSF image is the right one.  So we should just make an
            # InterpolatedImage out of it.
            PSF = galsim.InterpolatedImage(im_list[use_idx])
            band_PSF_dict[SCA]=PSF

        full_PSF_dict[band_name] = band_PSF_dict

    return full_PSF_dict

def _read_aberrations(SCA):
    """
    This is a helper routine that reads in aberrations for a particular SCA and wavelength (given as
    galsim.wfirst.wfirst_psfs.zemax_wavelength) from stored files, and return sthem along with the
    field positions

    @param  SCA      The identifier for the SCA, from 1-18.
    @returns NumPy arrays containing the aberrations, and x and y field positions. 
    """
    # Construct filename.
    sca_str = '_%02d'%SCA
    infile = os.path.join(galsim.meta_data.share_dir,
                          zemax_filepref + sca_str + zemax_filesuff)

    # Read in data.
    dat = np.loadtxt(infile)
    # It actually has 5 field positions, not just 1, to allow us to make position-dependent PSFs
    # within an SCA eventually.  Put it in the required format: an array of length (5 field
    # positions, 23 Zernikes), with the first entry empty (Zernike polynomials are 1-indexed so we
    # use entries 1-22).  The units are waves.
    aberrations = np.zeros((5,23))
    aberrations[:,1:] = dat[:,5:]
    # Also get the field position.  The file gives it in arcsec with respect to the center, but we
    # want it in pixels with respect to the corner.
    x_sca_pos = dat[:,1]/galsim.wfirst.pixel_scale + galsim.wfirst.n_pix/2
    y_sca_pos = dat[:,2]/galsim.wfirst.pixel_scale + galsim.wfirst.n_pix/2
    return aberrations, x_sca_pos, y_sca_pos

def _interp_aberrations_bilinear(aberrations, x_pos, y_pos, SCA_pos):
    """
    This is a helper routine to do bilinear interpolation of aberrations defined at 4 field
    positions: the four corners.  Note that we also have aberrations at the center position,
    but these are generally quite close (within a few percent) of what would come from this bilinear
    interpolation.  So for simplicity, we just do the bilinear interpolation.
    """
    min_x = np.min(x_pos)
    min_y = np.min(y_pos)
    max_x = np.max(x_pos)
    max_y = np.max(y_pos)
    x_frac = (SCA_pos.x - min_x) / (max_x - min_x)
    y_frac = (SCA_pos.y - min_y) / (max_y - min_y)
    lower_x_lower_y_ab = aberrations[(x_pos==min_x) & (y_pos==min_y), :]
    lower_x_upper_y_ab = aberrations[(x_pos==min_x) & (y_pos==max_y), :]
    upper_x_lower_y_ab = aberrations[(x_pos==max_x) & (y_pos==min_y), :]
    upper_x_upper_y_ab = aberrations[(x_pos==max_x) & (y_pos==max_y), :]
    interp_ab = (1.0-x_frac)*(1.0-y_frac)*lower_x_lower_y_ab + \
        (1.0-x_frac)*y_frac*lower_x_upper_y_ab + \
        x_frac*(1.0-y_frac)*upper_x_lower_y_ab + \
        x_frac*y_frac*upper_x_upper_y_ab

    return interp_ab.flatten()

def _find_limits(bandpasses, bandpass_dict):
    """
    This is a helper routine to find the minimum and maximum wavelengths across all bandpasses that
    are to be used.

    It requires a list of bandpasses and a dict containing the actual Bandpass objects, and returns
    the all-inclusive blue and red limits.

    @param bandpasses      List of bandpasses of interest.
    @param bandpass_dict   Dict containing all the bandpass objects.
    @returns blue and red wavelength limits, in nanometers.
    """
    min_wave = 1.e6
    max_wave = -1.e6
    for bandname in bandpasses:
        bp = bandpass_dict[bandname]
        if bp.blue_limit < min_wave: min_wave = bp.blue_limit
        if bp.red_limit > max_wave: max_wave = bp.red_limit
    return min_wave, max_wave

def _expand_list(x, n):
    """
    This is a helper routine to manage the inputs to getPSF.

    If x is iterable, it makes sure it has length n.
    If x is not iterable, it expands it a list of length n (repeating the single unique entry n
    times).
    """
    if hasattr(x, '__iter__'):
        if not len(x) == n:
            raise ValueError('Input lists are mismatched in length!')
        return x
    else:
        return [x] * n
