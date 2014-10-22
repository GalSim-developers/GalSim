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

This module contains numbers and routines for the WFIRST-AFTA project.  Currently, it includes the
following numbers:

    gain - The gain for all SCAs (sensor chip arrays) is expected to be the same, so this is a
           single value rather than a list of values.

    pixel_scale - The pixel scale in units of arcsec/pixel.

    dark_current - The dark current in units of e-/s.

    sky_background - The typical sky background flux in e-/pix/s.  (Band?)

    read_noise - The value of read noise in electrons.  (Band?)

    effective_diameter - The effective telescope diameter in meters.

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

    read_noise - A total of 10e-.  This comes from 20 e- per CDS and a 5 e- floor, so the CDS noise
                 dominates.  This read_noise value might be reduced based on improved behavior of
                 newer detectors.

    thermal_backgrounds - The thermal backgrounds (in units of e-/pix/s) are based on a temperature
                          of 282 K, but this plan might change in future.  The thermal backgrounds
                          depend on the band, so this is not a single number; instead, it's a
                          dictionary that is accessed by the name of the optical band, e.g.,
                          `galsim.wfirst.thermal_backgrounds['F184']` (where the names of the
                          bandpasses can be obtained using the getBandpasses() routine described
                          below).


For example, to get the gain value, use galsim.wfirst.gain.  Some of the numbers related to the
nature of the detectors are subject to change as further lab tests are done.

This module also contains the following routines:

    getBandpasses() - A utility to get a dictionary containing galsim.Bandpass objects for each of
                      the WFIRST-AFTA bandpasses, which by default will have their zero point set
                      for the WFIRST-AFTA effective diameter and typical exposure time.  For more
                      detail, do help(galsim.wfirst.getBandpasses).

    NLfunc() - A function to take an input image and simulate detector nonlinearity.  This will
               ordinarily be used as an input to GalSim routines for applying nonlinearity, i.e., if
               you have an image `im` then you would do

               >>>> im_nl = im.applyNonlinearity(galsim.wfirst.NLfunc)

TODO:
 - obscuration - just get the number and stick it here
 - zero points - fix the issue with effective diameter in the code
 - sky background - implement the integration over the zodi, which is going to involve adapting the
 tables and algorithm from the WFIRST ETC, and writing a converter to ecliptic coordinates for the
 CelestialCoord class.
 - stray light - put a simple fraction of sky background for now
 - WCS stuff - add the data from Jeff and port his WCS-builder to python
 - PSF stuff - finish pupil plane work, then wait for data from WCS optics team.
 - things related to IPC, persistence
"""

gain = 1.0
pixel_scale = 0.11
dark_current = 0.01
effective_diameter = 2.4
exptime = 168.1
dark_current = 0.015
nonlinearity_beta = -3.57e-7
n_dithers = 6
thermal_backgrounds = {'J129': 0.06,
                       'F184': 1.18, 
                       'Y106': 0.06,
                       'Z087': 0.06,
                       'H158': 0.08}

from wfirst_bandpass import *

def NLfunc(x):
    return x + nonlinearity_beta*(x**2)
