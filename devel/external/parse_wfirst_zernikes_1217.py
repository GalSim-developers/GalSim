# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
This is a helper routine to parse the Cycle 7 WFIRST Zernike information from

    https://wfirst.gsfc.nasa.gov/science/WFIRST_Reference_Information.html

More specifically, it takes as input a tab-separated version of the Zernike information in 

    WFIRST_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727.xlsm

for the wide-field imager.

It takes all field locations within the SCA, and isolates entries for our default value of
wavelength.  Then it makes output in the per-SCA format that the `_read_aberrations` routine in
galsim/wfirst/wfirst_psfs.py wants.

GalSim users will not have to run this routine.  It is included as a way to record how the files
with Zernike information in the share/ directory were created.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import galsim
import galsim.wfirst

dat = np.loadtxt('/Users/rmandelb/new_wfirst/WFIRST_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727.txt').transpose()
n_vals = dat.shape[1]
n_zern = 22
lam_to_use = galsim.wfirst.wfirst_psfs.zemax_wavelength/1000
out_dir = '../../share'

sca = dat[0,:]
lam = dat[1,:]
fieldnum = dat[2,:]
zern = np.zeros((22, n_vals))
for ind in range(0,n_zern):
    zern[ind,:] = dat[19+ind,:]

# Get list of unique SCA and field numbers.
sca_num = list(set(sca))
fieldnum_num = list(set(fieldnum))

# Take the target wavelength.
rows_to_use = (lam == 1.293)
sca = sca[rows_to_use]
fieldnum = fieldnum[rows_to_use]
sca_x = dat[3,rows_to_use] # SCA x position in mm
sca_y = dat[4,rows_to_use]
fpa_x = dat[5,rows_to_use] # FPA x position in mm
fpa_y = dat[6,rows_to_use]
zern = zern[:,rows_to_use]

for sca_val in sca_num:
    to_use = (sca == sca_val)
    sca_str = '_%02d'%sca_val
    outfile = os.path.join(out_dir, galsim.wfirst.wfirst_psfs.zemax_filepref + \
                               sca_str+galsim.wfirst.wfirst_psfs.zemax_filesuff)
    outarr = np.vstack([fieldnum[to_use], sca_x[to_use], sca_y[to_use], fpa_x[to_use],
                        fpa_y[to_use], zern[:,to_use]]).transpose()
    np.savetxt(outfile, outarr, fmt='%d %.2f %.2f %.2f %.2f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f')

