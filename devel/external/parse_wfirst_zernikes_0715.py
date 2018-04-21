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
This is a helper routine to parse the 07/15 WFIRST Zernike information from

    http://wfirst.gsfc.nasa.gov/science/sdt_public/wps/references/instrument/

More specifically, it takes as input a tab-separated version of the Zernike information in 

    AFTA_C5_WFC_Zernike_and_Field_Data_150717.xlsx

isolates the entries for the center of each SCA and our wavelength that we use as default (1293 nm),
and makes output in the per-SCA format that the `_read_aberrations` routine in
galsim/wfirst/wfirst_psfs.py wants.

GalSim users will not have to run this routine.  It is included as a way to record how the files
with Zernike information in the share/ directory were created.
"""
import numpy as np
import galsim.wfirst
import os

infile = '../../new_zernikes.tsv'
out_dir = '../../share'
zemax_wavelength = 1293.

dat = np.loadtxt(infile).transpose()
sca_num = dat[0,:]
wave = dat[1,:] #in microns
field_pos = dat[2,:]
aberrs = dat[12:23,:]

# select out SCA centers (field_pos = 1) and default wavelength
to_use = np.logical_and.reduce(
    [field_pos == 1,
     1000*wave == zemax_wavelength])
sca_num = sca_num[to_use]
aberrs = aberrs[:,to_use]

for SCA in galsim.wfirst._parse_SCAs(None):

    # Construct filename.
    sca_str = '%02d'%SCA
    outfile = os.path.join(out_dir, galsim.wfirst.wfirst_psfs.zemax_filepref + \
                               sca_str+galsim.wfirst.wfirst_psfs.zemax_filesuff)
    outarr = aberrs[:,sca_num==SCA]
    np.savetxt(outfile, outarr)
