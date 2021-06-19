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
import os
from ..utilities import LRU_Cache

"""
@file roman_psfs.py

Part of the Roman Space Telescope module.  This file includes routines needed to define a realistic
PSF for Roman.
"""

# Define a default set of bandpasses for which this routine works.
default_bandpass_list = ['J129', 'F184', 'W149', 'Y106', 'Z087', 'H158']
# Prefix for files containing information about Zernikes for each SCA for cycle 7.
zemax_filepref = "Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727"
zemax_filesuff = '.txt'
zemax_wavelength = 1293. #nm

# These need 'SCA*' prepended to the start to get the file name, and they live in
# the share/roman directory.
pupil_plane_file_longwave = '_full_mask.fits.gz'
pupil_plane_file_shortwave = '_rim_mask.fits.gz'

def getPSF(SCA, bandpass,
           SCA_pos=None, pupil_bin=4, wcs=None,
           n_waves=None, extra_aberrations=None,
           wavelength=None, gsparams=None,
           logger=None, high_accuracy=None, approximate_struts=None):
    """Get a single PSF for Roman ST observations.

    The user must provide the SCA and bandpass; the latter is used when setting up the pupil
    plane configuration and when interpolating chromatic information, if requested.

    This routine carries out linear interpolation of the aberrations within a given SCA, based on
    the Roman (then WFIRST) Cycle 7 specification of the aberrations as a function of focal plane
    position, more specifically from ``Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727.xlsm``
    downloaded from https://roman.gsfc.nasa.gov/science/Roman_Reference_Information.html.  Phase
    B updates that became available in mid-2019 have not yet been incorporated into this module.
    (Note: the files at that url still use the old WFIRST name.  We have renamed them to use the
    new name of the telescope, Roman, after downloading.)

    The mask images for the Roman pupil plane are available at the Roman Reference Information
    page: https://roman.gsfc.nasa.gov/science/Roman_Reference_Information.html.
    There are separate files for each SCA, since the view of the spider pattern varies somewhat
    across the field of view of the wide field camera. Furthermore, the effect of the obscuration
    is somewhat different at longer wavelengths, so F184 has a different set of files than the
    other filters.  cf. the ``galsm.roman.longwave_bands`` and ``galsim.roman.shortwave_bands``
    attributes, which define which bands use which pupil plane images.  Users usually don't need
    to worry about any of this, as GalSim will select the correct pupil image automatically based
    on the SCA and bandpass provided.

    The full pupil plane images are 4096 x 4096, which use a lot of memory and are somewhat slow
    to use, so we normally bin them by a factor of 4 (resulting in 1024 x 1024 images). This
    provides enough detail for most purposes and is much faster to render than using the full pupil
    plane images.  This bin factor is a settable parameter, called ``pupil_bin``.  If you want the
    more accurate, slower calculation using the full images, you can set it to 1. In the other
    direction, using pupil_bin=8 (resulting in a 512 x 512 image) still provides fairly reasonable
    results and is even faster to render.  It is not generally recommended to use higher binning
    than that, as the diffraction spikes will become noticeably degraded.

    .. note::

        This function will cache the aperture calculation, so repeated calls with the same
        SCA and bandpass should be much faster after the first call, as the pupil plane will
        already be loaded.  If you need to clear the cache for memory reasons, you may call::

            galsim.roman.roman_psfs._make_aperture.clear()

        to recover any memory currently being used for this cache.  Of course, subsequent calls to
        `getPSF` will need to rebuild the aperture at that point.

    The PSF that is returned by default will be oriented with respect to the SCA coordinates,
    not world coordinates as is typical in GalSim.  The pupil plane has a fixed orientation
    with respect to the focal plane, so the PSF rotates with the telescope.  To obtain a
    PSF in world coordinates, which can be convolved with galaxies (that are normally described
    in world coordinates), you may pass in a ``wcs`` parameter to this function.  This will
    project the PSF into world coordinates according to that WCS before returning it.  Otherwise,
    the return value is equivalent to using ``wcs=galim.PixelScale(galsim.roman.pixel_scale)``.

    The calculation takes advantage of the fact that the diffraction limit and aberrations have a
    simple, understood wavelength-dependence.  (The Roman project webpage for Cycle 7 does in fact
    provide aberrations as a function of wavelength, but the deviation from the expected chromatic
    dependence is sub-percent so we neglect it here.)  For reference, the script used to parse the
    Zernikes given on the webpage and create the files in the GalSim repository can be found in
    ``devel/external/parse_roman_zernikes_1217.py``.  The resulting chromatic object can be used to
    draw into any of the Roman bandpasses, though the pupil plane configuration will only be
    correct for those bands in the same range (i.e., long- or short-wavelength bands).

    For applications that require very high accuracy in the modeling of the PSF, with very limited
    aliasing, you may want to lower the folding_threshold in the gsparams.  Otherwise very bright
    stars will show some reflections in the spider pattern and possibly some boxiness at the
    outskirts of the PSF.  Using ``gsparams = GSParams(folding_threshold=2.e-3)`` generally
    provides good results even for very bright (e.g. mag=10) stars.  In these cases, you probably
    also want to reduce ``pupil_bin`` somewhat from the default value of 4.

    By default, no additional aberrations are included above the basic design.  However, users can
    provide an optional keyword ``extra_aberrations`` that will be included on top of those that are
    part of the design.  This should be in the same format as for the ChromaticOpticalPSF class,
    with units of waves at the fiducial wavelength, 1293 nm. Currently, only aberrations up to order
    22 (Noll convention) are simulated.  For Roman, the tolerance for additional
    aberrations was a total of 90 nanometers RMS as of mid-2015, distributed largely among coma,
    astigmatism, trefoil, and spherical aberrations (NOT defocus).  This information might serve as
    a guide for reasonable ``extra_aberrations`` inputs.  The reference for that number is
    an earlier Cycle 5 document:

    http://roman.gsfc.nasa.gov/science/sdt_public/wps/references/instrument/README_AFTA_C5_WFC_Zernike_and_Field_Data.pdf

    However, the default (non-extra) aberrations are from Cycle 7 material linked earlier in this
    docstring.

    Jitter and charge diffusion are, by default, not included.  Users who wish to include these can
    find some guidelines for typical length scales of the Gaussians that can represent these
    effects, and convolve the ChromaticOpticalPSF with appropriate achromatic Gaussians.

    The PSFs are always defined assuming the user will specify length scales in arcsec.

    Users may find they do not have to call `getPSF` for all objects in their simulations; for a
    given SCA and position within the SCA, and a given pupil plane configuration and wavelength
    information, it should be possible to reuse the PSFs.

    Parameters:
        SCA:                Single value specifying the SCA for which the PSF should be
                            loaded.
        bandpass:           Single string specifying the bandpass to use when defining the
                            pupil plane configuration and/or interpolation of chromatic PSFs.
                            You may also pass a string 'long' or 'short' for this argument, in
                            which case, the correct pupil plane configuration will be used for
                            long- or short-wavelength bands (F184 is long, all else is short).
                            In this case, no interpolation can be used, since it is defined
                            using the extent of the chosen bandpass. If ``wavelength`` is given,
                            then bandpass may be None, which will use the short-wavelength pupil
                            plane image.
        SCA_pos:            Single galsim.PositionD indicating the position within the SCA
                            for which the PSF should be created. If None, the exact center of
                            the SCA is chosen. [default: None]
        pupil_bin:          The binning to apply to the pupil plane image. (See discussion above.)
                            [default: 4]
        wcs:                The WCS to use to project the PSF into world coordinates.
                            [default: galsim.PixelScale(galsim.roman.pixel_scale)]
        n_waves:            Number of wavelengths to use for setting up interpolation of the
                            chromatic PSF objects, which can lead to much faster image
                            rendering.  If None, then no interpolation is used. Note that
                            users who want to interpolate can always set up the interpolation
                            later on even if they do not do so when calling `getPSF`.
                            [default: None]
        extra_aberrations:  Array of extra aberrations to include in the PSF model, on top of
                            those that are part of the Roman design.  These should be
                            provided in units of waves at the fiducial wavelength of 1293 nm,
                            as an array of length 23 with entries 4 through 22 corresponding
                            to defocus through the 22nd Zernike in the Noll convention.
                            [default: None]
        wavelength:         An option to get an achromatic PSF for a single wavelength, for
                            users who do not care about chromaticity of the PSF.  If None,
                            then the fully chromatic PSF is returned.  Alternatively the user
                            should supply either (a) a wavelength in nanometers, and they
                            will get achromatic OpticalPSF objects for that wavelength, or
                            (b) a bandpass object, in which case they will get achromatic
                            OpticalPSF objects defined at the effective wavelength of that
                            bandpass.  [default: False]
        gsparams:           An optional GSParams argument.  See the docstring for GSParams
                            for details. [default: None]

    Returns:
        A single PSF object (either a ChromaticOpticalPSF or an OpticalPSF depending on the
        inputs).

    """
    from ..position import PositionD
    from ..errors import GalSimValueError, GalSimRangeError
    from ..bandpass import Bandpass
    from ..wcs import PixelScale
    from . import n_pix, n_sca, longwave_bands, shortwave_bands, pixel_scale

    # Deprecated options
    if high_accuracy:
        if approximate_struts:
            from ..deprecated import depr
            from ..gsparams import GSParams
            depr('high_accuracy=True,approximate_struts=True', 2.3,
                 'pupil_bin=4, gsparams=galsim.GSParams(folding_threshold=2.e-3)',
                 'Note: this is not actually equivalent to the old behavior, but it should '
                 'be both faster and more accurate than the corresponding PSF in v2.2.')
            # Set folding_threshold 2.5x smaller than default.
            gsparams = GSParams.check(gsparams, folding_threshold=2.e-3)
            pupil_bin = 4
        else:
            from ..deprecated import depr
            from ..gsparams import GSParams
            depr('high_accuracy=True', 2.3,
                 'pupil_bin=1, gsparams=galsim.GSParams(folding_threshold=2.e-3)',
                 'Note: this is not actually equivalent to the old behavior, but it should '
                 'be both faster and more accurate than the corresponding PSF in v2.2.')
            # Set folding_threshold 2.5x smaller than default.
            gsparams = GSParams.check(gsparams, folding_threshold=2.e-3)
            pupil_bin = 1
    elif approximate_struts:
        from ..deprecated import depr
        from ..gsparams import GSParams
        depr('approximate_struts=True', 2.3, 'pupil_bin=8',
             'Note: this is not actually equivalent to the old behavior, but it should '
             'be both faster and more accurate than the corresponding PSF in v2.2.')
        pupil_bin = 8
    elif approximate_struts is False or high_accuracy is False:
        # If they are explicitly given, rather than default (None), then trigger this.
        from ..deprecated import depr
        from ..gsparams import GSParams
        depr('approximate_struts=False, high_accuracy=False', 2.3, 'pupil_bin=4',
             'Note: this is not actually equivalent to the old behavior, but it should '
             'be both faster and more accurate than the corresponding PSF in v2.2.')
        pupil_bin = 4

    if SCA <= 0 or SCA > n_sca:
        raise GalSimRangeError("Invalid SCA.", SCA, 1, n_sca)

    # SCA_pos: if None, then all should just be center of the SCA.
    if SCA_pos is None:
        SCA_pos = PositionD(n_pix/2, n_pix/2)

    # Parse the bandpasses to see which pupil plane image is needed
    pupil_plane_type = None
    if bandpass in longwave_bands or bandpass=='long':
        pupil_plane_type = 'long'
    elif bandpass in shortwave_bands or bandpass=='short':
        pupil_plane_type = 'short'
    elif bandpass is None and n_waves is None:
        pupil_plane_type = 'short'
    else:
        raise GalSimValueError("Bandpass not a valid Roman bandpass or 'short'/'long'.",
                               bandpass, default_bandpass_list)

    # If bandpass is 'short'/'long', then make sure that interpolation is not called for, since that
    # requires an actual bandpass.
    if bandpass in ['short','long'] and n_waves is not None:
        raise GalSimValueError("Cannot use bandpass='short'/'long' with interpolation.", bandpass)

    if not isinstance(wavelength, (Bandpass, float, type(None))):
        raise TypeError("wavelength should either be a Bandpass, float, or None.")

    # Now call _get_single_PSF().
    psf = _get_single_PSF(SCA, bandpass, SCA_pos, pupil_bin,
                          n_waves, extra_aberrations, wavelength,
                          pupil_plane_type, gsparams)

    # Apply WCS.
    # The current version is in arcsec units, but oriented parallel to the image coordinates.
    # So to apply the right WCS, project to pixels using the Roman mean pixel_scale, then
    # project back to world coordinates with the provided wcs.
    if wcs is not None:
        scale = PixelScale(pixel_scale)
        psf = wcs.toWorld(scale.toImage(psf), image_pos=SCA_pos)

    return psf

def __make_aperture(SCA, pupil_plane_type, pupil_bin, wave, gsparams):
    from . import diameter, obscuration
    from .. import fits
    from .. import meta_data
    from ..phase_psf import Aperture

    # Load the pupil plane image.
    if pupil_plane_type == 'long':
        pupil_plane_im = os.path.join(meta_data.share_dir, 'roman',
            'SCA%d'%SCA + pupil_plane_file_longwave)
    else:
        pupil_plane_im = os.path.join(meta_data.share_dir, 'roman',
            'SCA%d'%SCA + pupil_plane_file_shortwave)
    pupil_plane_im = fits.read(pupil_plane_im, read_header=True)
    # Native pixel scale in the file is for the exit pupil.  We want the scale of the
    # entrance pupil.  Fortunately, they provide the conversion as PUPILMAG in the header.
    # They also use microns for units, and we want meters, hence the extra 1.e-6.
    pupil_plane_im.scale *= pupil_plane_im.header['PUPILMAG'] * 1.e-6

    pupil_plane_im = pupil_plane_im.bin(pupil_bin,pupil_bin)

    aper = Aperture(lam=wave, diam=diameter,
                    obscuration=obscuration,
                    pupil_plane_im=pupil_plane_im,
                    gsparams=gsparams)
    return aper

# Usually a given run will only need one or a few different apertures for repeated getPSF calls.
# So cache those apertures here to avoid having to remake them.
_make_aperture = LRU_Cache(__make_aperture)

def _get_single_PSF(SCA, bandpass, SCA_pos, pupil_bin,
                    n_waves, extra_aberrations, wavelength,
                    pupil_plane_type, gsparams):
    """Routine for making a single PSF.  This gets called by `getPSF` after it parses all the
       options that were passed in.  Users will not directly interact with this routine.
    """
    from .. import OpticalPSF, ChromaticOpticalPSF
    from . import diameter, obscuration
    from ..bandpass import Bandpass
    from .roman_bandpass import getBandpasses

    if wavelength is None:
        wave = zemax_wavelength
    elif isinstance(wavelength, Bandpass):
        wave = wavelength = wavelength.effective_wavelength
    else:
        wave = wavelength

    # All parameters relevant to the aperture.  We may be able to use a cached version.
    aper = _make_aperture(SCA, pupil_plane_type, pupil_bin, wave, gsparams)

    # Start reading in the aberrations for that SCA
    aberrations, x_pos, y_pos = _read_aberrations(SCA)
    # Do bilinear interpolation, unless we're exactly at the center (default).
    use_aberrations = _interp_aberrations_bilinear(aberrations, x_pos, y_pos, SCA_pos)

    if extra_aberrations is not None:
        use_aberrations[:len(extra_aberrations)] += extra_aberrations
    # We don't want to use piston, tip, or tilt aberrations.  The former doesn't affect the
    # appearance of the PSF, and the latter cause centroid shifts.  So, we set the first 4
    # numbers (corresponding to a place-holder, piston, tip, and tilt) to zero.
    use_aberrations[0:4] = 0.

    # Now set up the PSF, including the option to interpolate over waves
    if wavelength is None:
        PSF = ChromaticOpticalPSF(lam=zemax_wavelength,
                                  diam=diameter, aberrations=use_aberrations,
                                  obscuration=obscuration, aper=aper,
                                  gsparams=gsparams)
        if n_waves is not None:
            # To decide the range of wavelengths to use, check the bandpass.
            bp_dict = getBandpasses()
            bp = bp_dict[bandpass]
            PSF = PSF.interpolate(waves=np.linspace(bp.blue_limit, bp.red_limit, n_waves),
                                  oversample_fac=1.5)
    else:
        tmp_aberrations = use_aberrations * zemax_wavelength / wavelength
        PSF = OpticalPSF(lam=wavelength, diam=diameter,
                         aberrations=tmp_aberrations,
                         obscuration=obscuration, aper=aper,
                         gsparams=gsparams)

    return PSF

def _read_aberrations(SCA):
    """
    This is a helper routine that reads in aberrations for a particular SCA and wavelength (given as
    galsim.roman.roman_psfs.zemax_wavelength) from stored files, and returns them along with the
    field positions.

    Parameters:
        SCA:        The identifier for the SCA, from 1-18.

    Returns:
        NumPy arrays containing the aberrations, and x and y field positions.
    """
    from .. import meta_data
    from . import pixel_scale, n_pix

    # Construct filename.
    sca_str = '_%02d'%SCA
    infile = os.path.join(meta_data.share_dir, 'roman',
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
    x_sca_pos = dat[:,1]/pixel_scale + n_pix/2
    y_sca_pos = dat[:,2]/pixel_scale + n_pix/2
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
