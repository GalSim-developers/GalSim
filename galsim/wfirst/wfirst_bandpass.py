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

def getBandpasses(AB_zeropoint=True, exptime=None):
    """Utility to get a dictionary containing the WFIRST bandpasses.

    This routine reads in a file containing a list of wavelengths and throughput for all WFIRST
    bandpasses, and uses the information in the file to create a dictionary.

    In principle it should be possible to replace the version of the file with another one, provided
    that the format obeys the following rules:

    - There is a column called 'Wave', containing the wavelengths in microns.
    - The other columns are labeled by the name of the bandpass.

    Currently the bandpasses are not truncated or thinned in any way.  We leave it to the user to
    decide whether they wish to do either of those operations.

    By default, the routine will set an AB zeropoint using the WFIRST effective diameter and default
    exposure time.  Setting the zeropoint can be avoided by setting `AB_zeropoint=False`; changing
    the exposure time that is used for the calculation can be used by setting the `exptime`
    keyword.

    This routine also loads information about sky backgrounds in each filter, to be used by the
    galsim.wfirst.getSkyLevel() routine.  The sky background information is saved as an attribute in
    each Bandpass object.

    Example usage
    -------------

        >>> wfirst_bandpasses = galsim.wfirst.getBandpasses()
        >>> f184 = wfirst_bandpasses['F184']

    @param AB_zeropoint     Should the routine set an AB zeropoint before returning the bandpass?
                            If False, then it is up to the user to set a zero point.  [default:
                            True]
    @param exptime          Exposure time to use for setting the zeropoint; if None, use the default
                            WFIRST exposure time, taken from galsim.wfirst.exptime.  [default: None]

    @returns A dictionary containing bandpasses for all WFIRST filters and grisms.
    """
    # Begin by reading in the file containing the info.
    datafile = os.path.join(galsim.meta_data.share_dir, "afta_throughput.txt")
    # One line with the column headings, and the rest as a NumPy array.
    data = np.loadtxt(datafile, skiprows=1).transpose()
    first_line = open(datafile).readline().rstrip().split()
    if len(first_line) != data.shape[0]:
        raise RuntimeError("Inconsistency in number of columns and header line in file %s"%datafile)

    # Identify the index of the column containing the wavelength in microns.  Get the wavelength and
    # convert to nm.
    wave_ind = first_line.index('Wave')
    wave = 1000.*data[wave_ind,:]

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
    i_band = 0
    # Loop over the bands.
    for index in range(len(first_line)):
        # Need to skip the entry for wavelength.
        if index==wave_ind:
            continue

        # Initialize the bandpass object.
        bp = galsim.Bandpass(galsim.LookupTable(wave, data[index,:]), wave_type='nm')
        # Set the zeropoint if requested by the user:
        if AB_zeropoint:
            if exptime is None:
                exptime = galsim.wfirst.exptime
            bp = bp.withZeropoint('AB', effective_diameter=d_eff, exptime=exptime)

        # Store the sky level information as an attribute.
        bp._ecliptic_lat = ecliptic_lat
        bp._ecliptic_lon = ecliptic_lon
        bp._sky_level = sky_data[2+i_band, :]

        # Add it to the dictionary.
        bandpass_dict[first_line[index]] = bp
        i_band += 1

    return bandpass_dict

