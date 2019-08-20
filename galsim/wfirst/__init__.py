# Copyright (c) 2012-2019 by the GalSim developers team on GitHub
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
"""
import os
import numpy as np
from .. import meta_data, Image

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
pupil_plane_file_longwave = os.path.join(meta_data.share_dir,
        "WFIRST_SRR_WFC_Pupil_Mask_Longwave_2048_reformatted.fits.gz")
# Z087, Y106, J129, H158
shortwave_bands = ['Z087', 'Y106', 'J129', 'H158']
pupil_plane_file_shortwave = os.path.join(meta_data.share_dir,
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
ipc_kernel = Image(ipc_kernel)

persistence_coefficients = np.array([0.045707683,0.014959818,0.009115737,0.00656769,0.005135571,0.004217028,0.003577534,0.003106601])/100.

# parameters in the fermi model = [ A, x0, dx, a, r, half_well]
# The following parameters are for H4RG-lo, the conservative model for low influence level x.
# The info and implementation can be found in wfirst_detectors.applyPersistence() and wfirst_detectors.fermi_linear().
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
from .wfirst_detectors import applyNonlinearity, addReciprocityFailure, applyIPC, applyPersistence, allDetectorEffects, NLfunc


