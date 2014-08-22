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

def getBandpasses():
    """Utility to get a dictionary containing the WFIRST bandpasses.

    This routine reads in a file containing a list of wavelengths and throughput for all WFIRST
    bandpasses, and uses the information in the file to create a dictionary.

    In principle it should be possible to replace the version of the file with another one, provided
    that the format obeys the following rules:

    - There is a column called 'Wave', containing the wavelengths in microns.
    - The other columns are labeled by the name of the bandpass.

    Currently the bandpasses are not truncated or thinned in any way.  We leave it to the user to
    decide whether they wish to do either of those operations.
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

    # Set up a dictionary.
    bandpass_dict = {}
    # Loop over the bands.
    for index in range(len(first_line)):
        # Need to skip the entry for wavelength.
        if index==wave_ind:
            continue

        # Initialize the bandpass object.
        bp = galsim.Bandpass(galsim.LookupTable(wave, data[index,:]), wave_type='nm')

        # Add it to the dictionary.
        bandpass_dict[first_line[index]] = bp

    return bandpass_dict

