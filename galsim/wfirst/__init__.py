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
"""
The galsim.wfirst module, containing information GalSim needs to simulate images for the WFIRST
project.

This module contains numbers and routines for the WFIRST project.  There is also a demo
illustrating the use of most of this functionality, demo13.py.  Some of the parameters below relate
to the entire wide-field imager.  Others, especially the return values of the functions to get the
PSF and WCS, are specific to each SCA (Sensor Chip Assembly, the equivalent of a chip for an optical
CCD) and therefore are indexed based on the SCA.  All SCA-related arrays are 1-indexed, i.e., the
entry with index 0 is None and the entries from 1 to n_sca are the relevant ones.  This is
consistent with diagrams and so on provided by the WFIRST project, which are 1-indexed.

The NIR detectors that will be used for WFIRST have a different photon detection process from CCDs.
In particular, the photon detection process begins with charge generation.  However, instead of
being read out along columns (as for CCDs), they are read directly from each pixel.  Moreover, the
actual quantity that is measured is technically not charge, but rather voltage.  The charge is
inferred based on the capacitance.  To use a common language with that for CCDs, we will often refer
to quantities measured in units of e-/pixel, but for some detector non-idealities, it is important
to keep in mind that it is voltage that is sensed.

Currently, the module includes the following numbers, which were updated as of WFIRST Cycle 7:

    gain - The gain for all SCAs (sensor chip assemblies) is expected to be the roughly the same,
           and we currently have no information about how different they will be, so this is a
           single value rather than a list of values.  Once the actual detectors exist and have been
           characterized, it might be updated to be a dict with entries for each SCA.

    pixel_scale - The pixel scale in units of arcsec/pixel.  This value is approximate and does not
                  include effects like distortion, which are included in the WCS.

    diameter - The telescope diameter in meters.

    obscuration - The linear obscuration of the telescope, expressed as a fraction of the diameter.

    collecting_area - The actual collecting area after accounting for obscuration, struts, etc. in
                      units of cm^2.

    exptime - The typical exposure time in units of seconds.  The number that is stored is for a
              single dither.  Each location within the survey will be observed with a total of 5-7
              dithers across 2 epochs.

    n_dithers - The number of dithers per filter (typically 5-7, so this is currently 6 as a
                reasonable effective average).

    dark_current - The dark current in units of e-/pix/s.

    nonlinearity_beta - The coefficient of the (counts)^2 term in the detector nonlinearity
                        function.  This will not ordinarily be accessed directly by users; instead,
                        it will be accessed by the convenience function in this module that defines
                        the nonlinearity function as counts_out = counts_in + beta*counts_in^2.
                        Alternatively users can use the galsim.wfirst.applyNonlinearity() routine,
                        which already knows about the expected form of the nonlinearity in the
                        detectors.

    reciprocity_alpha - The normalization factor that determines the effect of reciprocity failure
                        of the detectors for a given exposure time.  Alternatively, users can use
                        the galsim.wfirst.addReciprocityFailure() routine, which knows about this
                        normalization factor already, and allows users to choose an exposure time or
                        use the default WFIRST exposure time.

    read_noise - A total of 8.5e-.  This comes from 20 e- per correlated double sampling (CDS) and a
                 5 e- floor, so the CDS read noise dominates.  The source of CDS read noise is the
                 noise introduced when subtracting a single pair of reads; this can be reduced by
                 averaging over multiple reads.  Also, this read_noise value might be reduced
                 based on improved behavior of newer detectors which have lower CDS noise.

    thermal_backgrounds - The thermal backgrounds (in units of e-/pix/s) are based on a temperature
                          of 282 K, but this plan might change in future.  The thermal backgrounds
                          depend on the band, so this is not a single number; instead, it's a
                          dictionary that is accessed by the name of the optical band, e.g.,
                          `galsim.wfirst.thermal_backgrounds['F184']` (where the names of the
                          bandpasses can be obtained using the getBandpasses() routine described
                          below).

    pupil_plane_file - The name of the file containing the image of the pupil plane for WFIRST,
                       to be used when constructing PSFs.  If using the galsim.wfirst.getPSF()
                       routine, users will not need to supply this filename, since the routine
                       already knows about it.

    pupil_plane_scale - The pixel scale in meters per pixel for the image in pupil_plane_file.

    stray_light_fraction - The fraction of the sky background that is allowed to contribute as stray
                           light.  Currently this is required to be <10% of the background due to
                           zodiacal light, so its value is set to 0.1 (assuming a worst-case).  This
                           could be used to get a total background including stray light.

    ipc_kernel - The 3x3 kernel to be used in simulations of interpixel capacitance (IPC), using
                 galsim.wfirst.applyIPC().

    persistence_coefficients - The retention fraction of the previous eight exposures in a simple,
                               linear model for persistence.

    persistence_fermi_params - The parameters in the fermi persistence model.

    n_sca - The number of SCAs in the focal plane.

    n_pix_tot - Each SCA has n_pix_tot x n_pix_tot pixels.

    n_pix - The number of pixels that are actively used.  The 4 outer rows and columns will be
            attached internally to capacitors rather than to detector pixels, and used to monitor
            bias voltage drifts.  Thus, images seen by users will be n_pix x n_pix.

    jitter_rms - The worst-case RMS jitter per axis for WFIRST in the current design (reality
                 will likely be much better than this).  Units: arcsec.

    charge_diffusion - The per-axis sigma to use for a Gaussian representing charge diffusion for
                       WFIRST.  Units: pixels.

For example, to get the gain value, use galsim.wfirst.gain.  Most numbers related to the nature of
the detectors are subject to change as further lab tests are done.

This module also contains the following routines:

    getBandpasses() - A utility to get a dictionary containing galsim.Bandpass objects for each of
                      the WFIRST imaging bandpasses, which by default have AB zeropoints given using
                      the GalSim zeropoint convention (see getBandpasses() docstring for more
                      details).

    getSkyLevel() - A utility to find the expected sky level due to zodiacal light at a given
                    position, in a given band.

    NLfunc() - A function to take an input image and simulate detector nonlinearity.  This will
               ordinarily be used as an input to GalSim routines for applying nonlinearity, though
               it is not needed for users of galsim.wfirst.applyNonlinearity() (which is what we
               actually recommmend).

    applyNonlinearity() - A routine to apply detector nonlinearity of the type expected for WFIRST.

    addReciprocityFailure() - A routine to include the effects of reciprocity failure in images at
                              the level expected for WFIRST.

    applyIPC() - A routine to incorporate the effects of interpixel capacitance in WFIRST images.

    applyPersistence() - A routine to incorporate the effects of persistence - the residual images
                         from earlier exposures after resetting.

    allDetectorEffects() - A routine to add all sources of noise and all (implemented) detector
                           effects to an image containing astronomical objects plus background.  In
                           principle, users can simply use this routine instead of separately using
                           the various routines like applyNonlinearity().

    getPSF() - A routine to get a chromatic representation of the PSF in a single SCA.

    getWCS() - A routine to get the WCS for each SCA in the focal plane, for a given target RA, dec,
               and orientation angle.

    findSCA() - A routine that can take the WCS from getWCS() and some sky position, and indicate in
                which SCA that position can be found, optionally including half of the gaps between
                SCAs (to identify positions that are in the focal plane array but in the gap between
                SCAs).

    allowedPos() - A routine to check whether WFIRST is allowed to look at a given position on a
                   given date, given the constraints on orientation with respect to the sun.

    bestPA() - A routine to calculate the best observatory orientation angle for WFIRST when looking
               at a given position on a given date.

All of the above routines have docstrings that can be accessed using
help(galsim.wfirst.getBandpasses), and so on.

Another routine that may be necessary is galsim.utilities.interleaveImages().
The WFIRST PSFs at native WFIRST pixel scale are undersampled. A Nyquist-sampled PSF image can be
obtained by a two-step process:
1) Call the galsim.wfirst.getPSF() routine and convolve the PSF with the WFIRST pixel response to get
the effective PSF.
2) Draw the effective PSF onto an Image using drawImage routine, with a pixel scale lesser than the
native pixel scale (using the 'method=no_pixel' option).

However, if pixel-level effects such as nonlinearity and interpixel capacitance must be applied to
the PSF images, then they must drawn at the native pixel scale. A Nyquist-sampled PSF image can be
obtained in such cases by generating multiple images with offsets (a dither sequence) and then
combining them using galsim.utilities.interleaveImages(). See docstring for more details on the
usage.
"""
import os
import galsim
import numpy as np

gain = 1.0
pixel_scale = 0.11  # arcsec / pixel
diameter = 2.37  # meters
obscuration = 0.32
collecting_area = 3.757e4 # cm^2, from Cycle 7
exptime = 140.25  # s
dark_current = 0.015 # e-/pix/s
nonlinearity_beta = -6.e-7
reciprocity_alpha = 0.0065
read_noise = 8.5 # e-
n_dithers = 6
thermal_backgrounds = {'J129': 0.023, # e-/pix/s
                       'F184': 0.179,
                       'Y106': 0.023,
                       'Z087': 0.023,
                       'H158': 0.022,
                       'W149': 0.023}

# F184, W149
longwave_bands = ['F184', 'W149']
pupil_plane_file_longwave = os.path.join(galsim.meta_data.share_dir,
                                         "WFIRST_SRR_WFC_Pupil_Mask_Longwave_2048_reformatted.fits.gz")
# Z087, Y106, J129, H158
shortwave_bands = ['Z087', 'Y106', 'J129', 'H158']
pupil_plane_file_shortwave = os.path.join(galsim.meta_data.share_dir,
                                         "WFIRST_SRR_WFC_Pupil_Mask_Shortwave_2048_reformatted.fits.gz")
pupil_plane_file = pupil_plane_file_shortwave  # Let the canonical pupil be the shortwave one.

# The pupil plane image has non-zero values with a diameter of 2042 pixels.  The WFIRST mirror
# is 2.37 meters.  So the scale is 2.37 / 2042 = 0.00116 meters/pixel.
pupil_plane_scale = diameter / 2042.

stray_light_fraction = 0.1

# IPC kernel is unnormalized at first.  We will normalize it.
ipc_kernel = np.array([ [0.001269938, 0.015399776, 0.001199862], \
                        [0.013800177, 1.0, 0.015600367], \
                        [0.001270391, 0.016129619, 0.001200137] ])
ipc_kernel /= np.sum(ipc_kernel)
ipc_kernel = galsim.Image(ipc_kernel)

persistence_coefficients = np.array([0.045707683,0.014959818,0.009115737,0.00656769,0.005135571,0.004217028,0.003577534,0.003106601])/100.

# parameters in the fermi model = [ A, x0, dx, a, r, half_well]
# The following parameters are for H4RG-lo, the conservative model for low x.
persistence_fermi_parameters = np.array([0.017, 60000., 50000., 0.045, 1., 50000.])

n_sca = 18
n_pix_tot = 4096
n_pix = 4088
jitter_rms = 0.014
charge_diffusion = 0.1

from .wfirst_bandpass import getBandpasses
from .wfirst_backgrounds import getSkyLevel
from .wfirst_psfs import getPSF
from .wfirst_wcs import getWCS, findSCA, allowedPos, bestPA, convertCenter
from .wfirst_detectors import applyNonlinearity, addReciprocityFailure, applyIPC, applyPersistence, allDetectorEffects

def NLfunc(x):
    return x + nonlinearity_beta*(x**2)

def _parse_SCAs(SCAs):
    # This is a helper routine to parse the input SCAs (single number or iterable) and put it into a
    # convenient format.  It is used in wfirst_wcs.py.
    #
    # Check which SCAs are to be done.  Default is all (and they are 1-indexed).
    all_SCAs = np.arange(1, n_sca + 1, 1)
    # Later we will use the list of selected SCAs to decide which ones we're actually going to do
    # the calculations for.  For now, just check for invalid numbers.
    if SCAs is not None:
        # Make sure SCAs is iterable.
        if not hasattr(SCAs, '__iter__'):
            SCAs = [SCAs]
        # Then check for reasonable values.
        if min(SCAs) <= 0 or max(SCAs) > galsim.wfirst.n_sca:
            raise galsim.GalSimRangeError("Invalid SCA.", SCAs, 1, galsim.wfirst.n_sca)
        # Check for uniqueness.  If not unique, make it unique.
        SCAs = list(set(SCAs))
    else:
        SCAs = all_SCAs
    return SCAs
