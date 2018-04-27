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
# Prefix for files containing information about Zernikes for each SCA.
zemax_filepref = "AFTA_C5_WFC_Zernike_Data_150717_SCA"
zemax_filesuff = '.txt'
zemax_wavelength = 1293. #nm

def getPSF(SCAs=None, approximate_struts=False, n_waves=None, extra_aberrations=None,
           wavelength_limits=None, logger=None, wavelength=None, high_accuracy=False,
           gsparams=None):
    """
    Get the PSF for WFIRST observations.

    By default, this routine returns a dict of ChromaticOpticalPSF objects, with the dict indexed by
    the SCA (Sensor Chip Array, the equivalent of a chip in an optical CCD).  The PSF for a given
    SCA corresponds to that for the center of the SCA.  Currently we do not use information about
    PSF variation within each SCA, which is relatively small.

    This routine also takes an optional keyword `SCAs`, which can be a single number or an iterable;
    if this is specified then results are not included for the other SCAs.

    The default is to do the calculations using the full specification of the WFIRST pupil plane,
    which is a costly calculation in terms of memory.  For this, we use the provided pupil plane for
    red bands from

    http://wfirst.gsfc.nasa.gov/science/sdt_public/wps/references/instrument/   (Cycle 5)

    and we neglect for now the fact that the pupil plane configuration is slightly different for
    imaging in Z087, Y106, J129.  To avoid using the full pupil plane configuration, use the
    optional keyword `approximate_struts`.  In this case, the pupil plane will have the correct
    obscuration and number of struts, but the struts will be purely radial and evenly spaced instead
    of the true configuration.  The simplicity of this arrangement leads to a much faster
    calculation, and somewhat simplifies the configuration of the diffraction spikes.  Also note
    that currently the orientation of the struts is fixed, rather than rotating depending on the
    orientation of the focal plane.  Rotation of the PSF can easily be affected by the user via

       psf = galsim.wfirst.getPSF(...).rotate(angle)

    which will rotate the entire PSF (including the diffraction spikes and any other features).

    The calculation takes advantage of the fact that the diffraction limit and aberrations have a
    simple, understood wavelength-dependence.  (The WFIRST project webpage for Cycle 5 does in fact
    provide aberrations as a function of wavelength, but the deviation from the expected chromatic
    dependence is very small and we neglect it here.)  For reference, the script use to parse the
    Zernikes given on the webpage and create the files in the GalSim repository can be found in
    `devel/external/parse_wfirst_zernikes_0715.py`.  The resulting chromatic object can be used to
    draw into any of the WFIRST bandpasses.

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
    11 (Noll convention) can be simulated.  For WFIRST, the current tolerance for additional
    aberrations is a total of 90 nanometers RMS:
    http://wfirst.gsfc.nasa.gov/science/sdt_public/wps/references/instrument/README_AFTA_C5_WFC_Zernike_and_Field_Data.pdf
    distributed largely among coma, astigmatism, trefoil, and spherical aberrations (NOT defocus).
    This information might serve as a guide for reasonable `extra_aberrations` inputs.

    Jitter and charge diffusion are, by default, not included.  Users who wish to include these can
    find some guidelines for typical length scales of the Gaussians that can represent these
    effects, and convolve the ChromaticOpticalPSF with appropriate achromatic Gaussians.

    The PSFs are always defined assuming the user will specify length scales in arcsec.

    @param    SCAs                 Specific SCAs for which the PSF should be loaded.  This can be
                                   either a single number or an iterable.  If None, then the PSF
                                   will be loaded for all SCAs (1...18).  Note that the object that
                                   is returned is a dict indexed by the requested SCA indices.
                                   [default: None]
    @param    approximate_struts   Should the routine use an approximate representation of the pupil
                                   plane, with 6 equally-spaced radial struts, instead of the exact
                                   representation of the pupil plane?  Setting this parameter to
                                   True will lead to faster calculations, with a slightly less
                                   realistic PSFs.  [default: False]
    @param    n_waves              Number of wavelengths to use for setting up interpolation of the
                                   chromatic PSF objects, which can lead to much faster image
                                   rendering.  If None, then no interpolation is used. Note that
                                   users who want to interpolate can always set up the interpolation
                                   later on even if they do not do so when calling getPSF().
                                   [default: None]
    @param    extra_aberrations    Array of extra aberrations to include in the PSF model, on top of
                                   those that are part of the WFIRST design.  These should be
                                   provided in units of waves at the fiducial wavelength of 1293 nm,
                                   as an array of length 12 with entries 4 through 11 corresponding
                                   to defocus through spherical aberrations.  [default: None]
    @param    wavelength_limits    A tuple or list of the blue and red wavelength limits to use for
                                   interpolating the chromatic object, if `n_waves` is not None.  If
                                   None, then it uses the blue and red limits of all imaging
                                   passbands to determine the most inclusive wavelength range
                                   possible.  But this keyword can be used to reduce the range of
                                   wavelengths if only one passband (or a subset of passbands) is to
                                   be used for making the images.
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
                                   bandpass.
                                   [default: False]
    @param    high_accuracy        If True, make higher-fidelity representations of the PSF in
                                   Fourier space, to minimize aliasing (see plots on
                                   https://github.com/GalSim-developers/GalSim/issues/661 for more
                                   details).  This setting is more expensive in terms of time and
                                   RAM, and may not be necessary for many applications.
                                   [default: False]
    @param gsparams                An optional GSParams argument.  See the docstring for GSParams
                                   for details. [default: None]
    @returns  A dict of ChromaticOpticalPSF or OpticalPSF objects for each SCA.
    """
    # Check which SCAs are to be done using a helper routine in this module.
    SCAs = galsim.wfirst._parse_SCAs(SCAs)

    # Deal with some accuracy settings.
    if high_accuracy:
        if approximate_struts:
            oversampling = 3.5
        else:
            oversampling = 2.0

            # In this case, we need to pad the edges of the pupil plane image, so we cannot just use
            # the stored file.
            tmp_pupil_plane_im = galsim.fits.read(galsim.wfirst.pupil_plane_file)
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
            pupil_plane_im = galsim.wfirst.pupil_plane_file
            pupil_plane_scale = galsim.wfirst.pupil_plane_scale

    if wavelength is None:
        if n_waves is not None:
            if wavelength_limits is None:
                # To decide the range of wavelengths to use (if none were passed in by the user),
                # first check out all the bandpasses.
                bandpass_dict = galsim.wfirst.getBandpasses()
                # Then find the blue and red limit to be used for the imaging bandpasses overall.
                blue_limit, red_limit = _find_limits(default_bandpass_list, bandpass_dict)
            else:
                if not isinstance(wavelength_limits, tuple):
                    raise TypeError("Wavelength limits must be entered as a tuple.")
                blue_limit, red_limit = wavelength_limits
                if red_limit <= blue_limit:
                    raise galsim.GalSimIncompatibleValuesError(
                        "Wavelength limits must have red_limit > blue_limit.",
                        blue_limit=blue_limit, red_limit=red_limit)
    elif isinstance(wavelength, float):
        wavelength_nm = wavelength
    elif isinstance(wavelength, galsim.Bandpass):
        wavelength_nm = wavelength.effective_wavelength
    else:
        raise TypeError("wavelength should either be a Bandpass, float, or None.")

    # Start reading in the aberrations for the relevant SCAs.
    aberration_dict = {}
    PSF_dict = {}
    if logger: logger.debug('Beginning to loop over SCAs and get the PSF:')
    for SCA in SCAs:
        aberration_dict[SCA] = _read_aberrations(SCA)

        use_aberrations = aberration_dict[SCA]
        if extra_aberrations is not None:
            use_aberrations += extra_aberrations
        # We don't want to use piston, tip, or tilt aberrations.  The former doesn't affect the
        # appearance of the PSF, and the latter cause centroid shifts.  So, we set the first 4
        # numbers (corresponding to a place-holder, piston, tip, and tilt) to zero.
        use_aberrations[0:4] = 0.

        # Now set up the PSF for this SCA, including the option to simplify the pupil plane.
        if logger: logger.debug('   ... SCA %d'%SCA)
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

        PSF_dict[SCA]=PSF

    return PSF_dict

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
    This is a helper routine that reads in aberrations for a particular SCA and wavelength from
    stored files.  It returns the aberrations in a format required by ChromaticOpticalPSF.

    @param  SCA      The identifier for the SCA, from 1-18.
    @returns a NumPy array containing the aberrations, in the required format for
    ChromaticOpticalPSF.
    """
    # Construct filename.
    sca_str = '%02d'%SCA
    infile = os.path.join(galsim.meta_data.share_dir,
                          zemax_filepref + sca_str + zemax_filesuff)

    # Read in data.
    dat = np.loadtxt(infile).transpose()
    # Put it in the required format: an array of length 12, with the first entry empty (Zernike
    # polynomials are 1-indexed so we use entries 1-11).  The units are waves.
    aberrations = np.zeros(12)
    aberrations[1:] = dat
    return aberrations

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
