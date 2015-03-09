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
@file wfirst_bandpass.py

Part of the WFIRST module.  This file includes any routines needed to define the WFIRST bandpasses.
"""

import galsim
import numpy as np
import os

def getBandpasses(AB_zeropoint=True, exptime=None, thin_err=1.e-4):
    """Utility to get a dictionary containing the WFIRST bandpasses used for imaging.

    This routine reads in a file containing a list of wavelengths and throughput for all WFIRST
    bandpasses, and uses the information in the file to create a dictionary.

    In principle it should be possible to replace the version of the file with another one, provided
    that the format obeys the following rules:

    - There is a column called 'Wave', containing the wavelengths in microns.
    - The other columns are labeled by the name of the bandpass.

    Currently the bandpasses are not truncated.  We leave it to the user to decide whether they wish
    to truncate after getting the bandpasses, and the `thin_err` keyword allows the user to choose
    the relative errors allowed when thinning.

    By default, the routine will set an AB zeropoint using the WFIRST effective diameter and default
    exposure time.  Setting the zeropoint can be avoided by setting `AB_zeropoint=False`; changing
    the exposure time that is used for the zeropoint calculation can be used by setting the
    `exptime` keyword.

    This routine also loads information about sky backgrounds in each filter, to be used by the
    galsim.wfirst.getSkyLevel() routine.  The sky background information is saved as an attribute in
    each Bandpass object.

    There are some subtle points related to the filter edges, which seem to depend on the field
    angle at some level.  This is more important for the grism than for the imaging, so currently
    this effect is not included in the WFIRST bandpasses in GalSim.

    Example usage
    -------------

        >>> wfirst_bandpasses = galsim.wfirst.getBandpasses()
        >>> f184_bp = wfirst_bandpasses['F184']

    @param AB_zeropoint     Should the routine set an AB zeropoint before returning the bandpass?
                            If False, then it is up to the user to set a zero point.  [default:
                            True]
    @param exptime          Exposure time to use for setting the zeropoint; if None, use the default
                            WFIRST exposure time, taken from galsim.wfirst.exptime.  [default: None]
    @param thin_err         Relative error allowed when thinning the bandpasses.  [default: 1e-4]

    @returns A dictionary containing bandpasses for all WFIRST imaging filters.
    """
    # Begin by reading in the file containing the info.
    datafile = os.path.join(galsim.meta_data.share_dir, "afta_throughput.txt")
    # One line with the column headings, and the rest as a NumPy array.
    data = np.genfromtxt(datafile, names=True)
    wave = 1000.*data['Wave']

    if AB_zeropoint:
        # Note that withZeropoint wants an effective diameter in cm, not m.  Also, the effective
        # diameter has to take into account the central obscuration, so d_eff = d sqrt(1 -
        # obs^2).
        d_eff = 100. * galsim.wfirst.diameter * np.sqrt(1.-galsim.wfirst.obscuration**2)

    # Read in and manipulate the sky background info.
    sky_file = os.path.join(galsim.meta_data.share_dir, "wfirst_sky_backgrounds.txt")
    sky_data = np.loadtxt(sky_file).transpose()
    ecliptic_lat = sky_data[0, :]
    ecliptic_lon = sky_data[1, :]

    # Set up a dictionary.
    bandpass_dict = {}
    # Loop over the bands.
    band_list = data.dtype.names[1:]
    for index in range(len(band_list)):
        # Need to skip the prism and grism (not used for weak lensing imaging).
        bp_name = band_list[index]
        if bp_name=='SNPrism' or bp_name=='BAO-Grism':
            continue

        # Initialize the bandpass object.
        bp = galsim.Bandpass(galsim.LookupTable(wave, data[bp_name]), wave_type='nm').thin(thin_err)
        # Set the zeropoint if requested by the user:
        if AB_zeropoint:
            if exptime is None:
                exptime = galsim.wfirst.exptime
            bp = bp.withZeropoint('AB', effective_diameter=d_eff, exptime=exptime)

        # Store the sky level information as an attribute.
        bp._ecliptic_lat = ecliptic_lat
        bp._ecliptic_lon = ecliptic_lon
        bp._sky_level = sky_data[2+index, :]

        # Add it to the dictionary.
        bandpass_dict[bp_name] = bp

    return bandpass_dict

