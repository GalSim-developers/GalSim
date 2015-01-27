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
"""
The galsim.wfirst module, containing information GalSim needs to simulate images for the WFIRST-AFTA
project.

This module contains numbers and routines for the WFIRST-AFTA project.  Some of the parameters below
relate to the entire wide-field imager.  Others, especially the return values of the functions to
get the PSF and WCS, are specific to each SCA (Sensor Chip Assembly, the equivalent of a chip for an
optical CCD) and therefore are indexed based on the SCA.  All SCA-related arrays are 1-indexed,
i.e., the entry with index 0 is None and the entries from 1 to n_sca are the relevant ones.  This is
consistent with diagrams and so on provided by the WFIRST project, which are 1-indexed.

Currently, the module includes the following numbers:

    gain - The gain for all SCAs (sensor chip assemblies) is expected to be the same, so this is a
           single value rather than a list of values.

    pixel_scale - The pixel scale in units of arcsec/pixel.

    diameter - The telescope diameter in meters.

    obscuration - The linear obscuration of the telescope, in terms of fraction of the diameter.

    exptime - The typical exposure time in units of seconds.  The number that is stored is for a
              single dither.  Each location within the survey will be observed with a total of 5-7
              dithers across 2 epochs.  For a single dither, there are 32 up-the-ramp samples, each
              taking 5.423 seconds, but the effective live time is really 31 samples.  This is the
              source of the 168.1s that is currently stored for exptime.

    n_dithers - The number of dithers per filter (typically 5-7, so this is currently 6 as a
                reasonable effective average).

    dark_current - The dark current in units of e-/pix/s.

    nonlinearity_beta - The coefficient of the (counts)^2 term in the detector nonlinearity
                        function.  This will not ordinarily be accessed directly by users; instead,
                        it will be accessed by the convenience function in this module that defines
                        the nonlinearity function.

    reciprocity_alpha - The normalization factor that determines the effect of reciprocity failure
                        of the detectors for a given exposure time.  Typically, users would then
                        simulate the reciprocity failure for a given image `im` and for the default
                        WFIRST exposure time using

        >>>> im_recip = im.addReciprocityFailure(galsim.wfirst.exptime,
                                                 galsim.wfirst.reciprocity_alpha)

    read_noise - A total of 10e-.  This comes from 20 e- per correlated double sampling (CDS) and a
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

    pupil_plane_file - The name of the file containing the image of the pupil plane for WFIRST-AFTA,
                       to be used when constructing PSFs.

    stray_light_fraction - The fraction of the sky background that is allowed to contribute as stray
                           light.  Currently this is required to be <10% of the background due to
                           zodiacal light, so its value is set to 0.1 (assuming a worst-case).  This
                           could be used to get a total background including stray light.

    ipc_kernel - The 3x3 kernel to be used in simulations of interpixel capacitance (IPC); see
                 help(galsim.detectors.applyIPC()) for more information.

    n_sca - The number of SCAs in the focal plane.

    n_pix_tot - Each SCA has n_pix_tot x n_pix_tot pixels.

    n_pix - The number of pixels that are actively used.  The 4 outer rows and columns will be
            attached internally to capacitors rather than to detector pixels, and used to monitor
            bias voltage drifts).  Thus, images seen by users will be n_pix x n_pix.

For example, to get the gain value, use galsim.wfirst.gain.  Some of the numbers related to the
nature of the detectors are subject to change as further lab tests are done.

This module also contains the following routines:

    getBandpasses() - A utility to get a dictionary containing galsim.Bandpass objects for each of
                      the WFIRST-AFTA bandpasses, which by default will have their zero point set
                      for the WFIRST-AFTA effective diameter and typical exposure time.

    getSkyLevel() - A utility to find the expected sky level due to zodiacal light at a given
                    position, in a given band..

    NLfunc() - A function to take an input image and simulate detector nonlinearity.  This will
               ordinarily be used as an input to GalSim routines for applying nonlinearity, i.e., if
               you have an image `im` then you would do

               >>>> im_nl = im.applyNonlinearity(galsim.wfirst.NLfunc)

    getPSF() - A routine to get a chromatic representation of the PSF in each SCAs.  This involves a
               significant overhead when initializing, though certain keywords can be used to speed
               up the process.  Actually using the PSFs to draw images is much faster than the
               initialization.

    tabulatePSFImages() - A routine to take outputs of getPSF() and write them to file as a set of
                          images in a multi-extension FITS file.  This can be used along with the
                          next routine for faster calculations, as long as the application is one
                          for which it does not matter if you ignore the variation of the PSF with
                          wavelength within the passband.

    getStoredPSF() - A routine to read in an image of the PSF stored by tabulatePSFImages(), and
                     make objects corresponding to each of them for later use.  This routine has
                     less overhead than getPSF() but also is less flexible.

    getWCS() - A routine to get the WCS for each SCA in the focal plane, for a given target RA, dec,
               and orientation angle.  Use of this routine requires that GalSim be able to access
               some software that can handle TAN-SIP style WCS (either Astropy, starlink.Ast,
               WCSTools).

    findSCA() - A routine that can take the WCS from getWCS() and some sky position, and indicate in
                which SCA that position can be found, optionally including half of the gaps between
                SCAs (to identify positions that are in the focal plane array but in the gap between
                SCAs).

All of the above routines have docstrings that can be accessed using
help(galsim.wfirst.getBandpasses), and so on.
"""
import os
import galsim
import numpy

gain = 1.0
pixel_scale = 0.11
diameter = 2.36
obscuration = 0.3
exptime = 168.1
dark_current = 0.015
nonlinearity_beta = -3.57e-7
reciprocity_alpha = 0.0065
read_noise = 10.0
n_dithers = 6
thermal_backgrounds = {'J129': 0.06,
                       'F184': 1.18, 
                       'Y106': 0.06,
                       'Z087': 0.06,
                       'H158': 0.08,
                       'W149': 0.06}
pupil_plane_file = os.path.join(galsim.meta_data.share_dir,
                                "WFIRST-AFTA_Pupil_Mask_C5_20141010_PLT.fits.gz")
stray_light_fraction = 0.1
# IPC kernel is unnormalized at first.  We will normalize it.
ipc_kernel = numpy.array([ [0.001269938, 0.015399776, 0.001199862], \
                           [0.013800177, 1.0, 0.015600367], \
                           [0.001270391, 0.016129619, 0.001200137] ])
ipc_kernel /= numpy.sum(ipc_kernel)
ipc_kernel = galsim.Image(ipc_kernel)
n_sca = 18
n_pix_tot = 4096 
n_pix = 4088

from wfirst_bandpass import getBandpasses
from wfirst_backgrounds import getSkyLevel
from wfirst_psfs import getPSF, tabulatePSFImages, getStoredPSF
from wfirst_wcs import getWCS, findSCA

def NLfunc(x):
    return x + nonlinearity_beta*(x**2)
