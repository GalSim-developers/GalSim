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

import galsim
import galsim.wfirst
import numpy as np
import os
import pyfits

"""
@file wfirst_psfs.py

Part of the WFIRST module.  This file includes routines needed to define a realistic PSF for WFIRST.
"""

# Define a default set of bandpasses for which this routine works.
default_bandpass_list = ['J129', 'F184', 'W149', 'Y106', 'Z087', 'H158']
# Prefix for files containing information about Zernikes for each SCA.
zemax_filepref = os.path.join(galsim.meta_data.share_dir,
                              "AFTA_WFI_v4-2-5_140326_192nmRMS_NoAps_PLT_Zemax_ZernStanTerm_C")
zemax_filesuff = '_F01_W04.txt'
zemax_wavelength = 1293. #nm

def getPSF(SCAs=None, approximate_struts=False, n_waves=None, extra_aberrations=None,
           wavelength_limits=None):
    """
    Get the PSF for WFIRST observations.

    By default, this routine returns a list of ChromaticOpticalPSF objects, with the list index
    corresponding to the SCA.  The PSF corresponds to a location in the center of the SCA.
    Currently we do not have information about variation across the SCAs, which is expected to be
    relatively small.

    This routine also takes an optional keyword `SCAs`, which can be a single value or a list; if
    this is specified then results are not included for the other SCAs.  However, to preserve the
    indexing, the list of results has the same length, with the result being None for the SCAs that
    are not of interest.

    The default is to do the calculations using the full specification of the WFIRST pupil plane,
    which is a costly calculation in terms of memory.  To turn this off, use the optional keyword
    `approximate_struts`.  In this case, the pupil plane will have the correct obscuration and
    number of struts, but the struts will be purely radial and evenly spaced instead of the true
    configuration.  The simplicity of this arrangement leads to a much faster calculation.

    The calculation takes advantage of the fact that the diffraction limit and aberrations have a
    simple, understood wavelength-dependence.  The resulting object can be used to draw into any of
    the WFIRST bandpasses.

    By default, no additional aberrations are included above the basic design.  However, users can
    provide an optional keyword `extra_aberrations` that will be included on top of those that
    are part of the design.  This should be in the same format as for the ChromaticOpticalPSF class,
    with units of nanometers. Currently, only aberrations up to order 11 (Noll convention) can be
    simulated.

    The default is to instantiate the ChromaticOpticalPSF objects with coverage of the full
    wavelength range of all WFIRST passbands.  Since this is typically done with a fixed wavelength
    sampling over a wide wavelength range, there is significant overhead in this process.  If the
    intent is to only use the PSFs for a limited wavelength range (e.g., within a single one of the
    bandpasses), it can be significantly faster to use the `wavelength_limits` keyword to pass in a
    tuple with the lower and upper limits in wavelength.

    Usage: <fill in usage example>

    @param    blah blah.
    @returns  blah blah.
    """
    # Check which SCAs are to be done.  Default is all.
    # SCAs are 1-indexed, so we make a list that is one longer than needed, with the 0th element
    # being None.
    all_SCAs = np.arange(galsim.wfirst.n_sca + 1)
    # Later we will use the list of selected SCAs to decide which ones we're actually going to do
    # the calculations for.  For now, just check for invalid numbers.
    if SCAs is not None:
        if hasattr(SCAs, '__iter__'):
            if max(SCAs) > galsim.wfirst.n_sca:
                raise ValueError("Invalid SCA!  Indices must be <=%d."%galsim.wfirst.n_sca)
        else:
            SCAs = [SCAs]

    if wavelength_limits is None:
        # To decide the range of wavelengths to use (if none were passed in by the user), first
        # check out all the bandpasses.
        bandpass_dict = galsim.wfirst.getBandpasses()
        # Then find the blue and red limit to be used for the imaging bandpasses overall.
        blue_limit, red_limit = _find_limits(default_bandpass_list, bandpass_dict)
    else:
        if not isinstance(wavelength_limits, tuple):
            raise ValueError("Wavelength limits must be entered as a tuple!")
        blue_limit, red_limit = wavelength_limits
    # Decide on the number of linearly spaced wavelengths to use for the ChromaticOpticalPSF:
    if n_waves is None: n_waves = int((red_limit - blue_limit)/20)

    # Start reading in the aberrations for the relevant SCAs.  Take advantage of symmetries, so we
    # don't have to call the reading routine too many times.
    aberration_list = []
    PSF_list = []
    for SCA in all_SCAs:
        # First, if the SCA is zero or if it's not in SCAs (user-supplied list) then just stick None
        # on the list.
        if SCA == 0:
            aberration_list.append(None)
            PSF_list.append(None)
            continue
        if SCAs is not None:
            if SCA not in SCAs:
                aberration_list.append(None)
                PSF_list.append(None)
                continue

        # Check if it's above 10.  If it is, the aberrations are the same as for the SCA with
        # index that is 9 lower.  So if that one is in the list, then just use its results.
        read_SCA = SCA
        if SCA >= 10:
            if aberration_list[SCA-9] is not None:
                aberration_list.append(aberration_list[SCA-9])
                PSF_list.append(PSF_list[SCA-9])
                pass
            else:
                read_SCA -= 9

        # If we got here, then it means we have to read in the aberrations.
        aberrations = _read_aberrations(read_SCA)
        if extra_aberrations: aberrations += extra_aberrations
        aberration_list.append(aberrations)

        # Now set up the PSF for this SCA, including the option to simplify the pupil plane.
        if approximate_struts:
            PSF = galsim.ChromaticOpticalPSF(
                diam=galsim.wfirst.diameter, aberrations=aberration_list[SCA],
                waves=np.linspace(blue_limit, red_limit, n_waves), oversample_fac=1.5,
                obscuration=galsim.wfirst.obscuration, nstruts=6)
        else:
            PSF = galsim.ChromaticOpticalPSF(
                diam=galsim.wfirst.diameter, aberrations=aberration_list[SCA],
                waves=np.linspace(blue_limit, red_limit, n_waves), oversample_fac=1.5,
                obscuration=galsim.wfirst.obscuration, pupil_plane_im=galsim.wfirst.pupil_plane_file,
                oversampling=1.2, pad_factor=2.)
        PSF_list.append(PSF)

    return PSF_list

def tabulatePSFImages(PSF_list, image_filename, data_filename, bandpass_list=default_bandpass_list,
                      clobber=False):
    """
    This is a routine to store images of WFIRST PSFs in different bands for each SCA.  It takes an
    output list of PSFs (`PSF_list`) directly from getPSF().  The output will be a file
    (`image_filename`) that has all the images, and another file (`data_filename`) that contains a
    FITS table indicating the bandpasses, SCAs, and other information needed to reconstruct the PSF
    information.

    Note that the image files can take up space, but if `image_filename` has an extension that
    GalSim recognizes as corresponding to a compressed format, the compression will automatically be
    done.  See galsim.fits.write documentation for more information about this option.

    @param blah
    """
    # Check for sane input PSF_list.
    if len(PSF_list) != galsim.wfirst.n_sca+1:
        raise ValueError("PSF_list must have length %d, as output by getPSF!"
                         %(galsim.wfirst.n_sca+1))

    # Check if file already exists and warn about clobbering.
    if os.path.exists(image_filename) or os.path.exists(data_filename):
        if clobber is False:
            raise ValueError("At least one output file already exists, and clobber is not set!")
        else:
            import warnings
            warnings.warn("At least one output file already exists, and will be clobbered.")

    # Check that bandpass list input is okay.  It should be strictly a subset of the default list of
    # bandpasses.
    if not set(bandpass_list).issubset(default_bandpass_list):
        err_msg = ''
        for item in default_bandpass_list:
            err_msg += item+' '
        raise ValueError("Bandpass list must be a subset of the default list, containing %s"
                         %err_msg)

    # Get all the WFIRST bandpasses.
    bandpass_dict = galsim.wfirst.getBandpasses()

    # Loop through making images and lists of their relevant parameters.
    im_list = []
    bp_name_list = []
    SCA_index_list = []
    for bp_name in bandpass_list:
        bandpass = bandpass_dict[bp_name]

        for SCA in range(galsim.wfirst.n_sca+1):
            if PSF_list[SCA] is None:
                continue

            PSF = PSF_list[SCA]
            im = PSF.drawImage(bandpass, scale=0.5*galsim.wfirst.pixel_scale)
            im_list.append(im)
            bp_name_list.append(bp_name)
            SCA_index_list.append(SCA)

    # Save images to file.
    n_ims = len(im_list)
    galsim.fits.writeMulti(im_list, image_filename, clobber=clobber)
    print 'Saved %d images to file %s'%(n_ims, image_filename)

    # Save data to file, after constructing a FITS table.  Watch out for clobbering.
    bp_names = pyfits.Column(name='bandpass', format='A10', array=np.array(bp_name_list))
    SCA_indices = pyfits.Column(name='SCA', format='J', array=np.array(SCA_index_list))
    cols = pyfits.ColDefs([bp_names, SCA_indices])
    tbhdu = pyfits.new_table(cols)
    prhdu = pyfits.PrimaryHDU()
    thdulist = pyfits.HDUList([prhdu, tbhdu])
    galsim.fits.writeFile(data_filename, thdulist, clobber=clobber)
    print 'Saved metadata to file %s'%data_filename

def getStoredPSF(image_filename, data_filename):
    """
    TODO
    """
    # Get the image data.
    im_list = galsim.fits.readMulti(image_filename)

    # Get the metadata from the FITS table, and close the file when done.
    hdu, hdu_list, fin = galsim.fits.readFile(data_filename)
    bp_list = list(hdu_list[1].data.bandpass)
    SCA_list = list(hdu_list[1].data.SCA)
    galsim.fits.closeHDUList(hdu_list, fin)

    # Set up the dict of PSF objects, indexed by bandpass (and then SCA).
    PSF_dict = {}
    for band_name in default_bandpass_list:
        PSF_list = []

        # Find all indices in `bp_list` that correspond to this bandpass.
        bp_indices = []
        if band_name in bp_list:
            idx = -1
            while True:
                try:
                    idx = bp_list.index(band_name, idx+1)
                    bp_indices.append(idx)
                except ValueError:
                    break

        for SCA in range(galsim.wfirst.n_sca+1):
            if SCA == 0 or len(bp_indices)==0:
                PSF_list.append(None)
                continue

            # Now find which element has both the right band_name and is for this SCA.  There might
            # not be any, depending on what exactly was stored.
            use_idx = -1
            for index in bp_indices:
                if SCA_list[index] == SCA:
                    use_idx = index
                    break
            if use_idx<0:
                PSF_list.append(None)
                continue

            # Now we know which PSF image is the right one.  So we should just make an
            # InterpolatedImage out of it.
            PSF = galsim.InterpolatedImage(im_list[use_idx])
            PSF_list.append(PSF)

        PSF_dict[band_name] = PSF_list

    return PSF_dict

def _read_aberrations(SCA):
    """
    This is a helper routine that reads in aberrations for a particular SCA and wavelength from
    stored files.  It returns the aberrations in a format required by ChromaticOpticalPSF.

    @param  SCA      The identifier for the SCA, from 1-9.
    @returns a NumPy array containing the aberrations, in the required format for
    ChromaticOpticalPSF.
    """
    if SCA < 1 or SCA > 9:
        raise ValueError("SCA requested is out of range: %d"%SCA)

    # Construct filename.
    sca_str = '%02d'%SCA
    infile = zemax_filepref + sca_str + zemax_filesuff

    # Read in data.
    dat = np.loadtxt(infile, skiprows=41, usecols=(2,)).transpose()
    # Put it in the required format: an array of length 12, with the first entry empty (Zernike
    # polynomials are 1-indexed so we use entries 1-11).  The units should be nanometers but the
    # input files are in units of waves, hence the multiplication by the wavelength at which the
    # calculation was done.
    aberrations = np.zeros(12)
    aberrations[1:] = dat*zemax_wavelength
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
