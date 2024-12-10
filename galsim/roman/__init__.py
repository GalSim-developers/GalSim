# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
The galsim.roman module, containing information GalSim needs to simulate images for the Roman
Space Telescope.
"""
import os
import numpy as np
from .. import meta_data, Image

gain = 1.0
pixel_scale = 0.11  # arcsec / pixel
diameter = 2.36  # meters
obscuration = 0.32
collecting_area = 3.757e4 # cm^2, from Cycle 7
exptime = 139.8  # s
dark_current = 0.015 # e-/pix/s
nonlinearity_beta = -6.e-7
reciprocity_alpha = 0.0065
read_noise = 8.5 # e-
n_dithers = 6

# These are from https://roman.gsfc.nasa.gov/science/WFI_technical.html, as of October, 2023
thermal_backgrounds = {'R062': 0.00, # e-/pix/s
                       'Z087': 0.00,
                       'Y106': 0.00,
                       'J129': 0.00,
                       'H158': 0.04,
                       'F184': 0.17,
                       'K213': 4.52,
                       'W146': 0.98,
                       'SNPrism': 0.00,
                       'Grism_0thOrder': 0.00,
                       'Grism_1stOrder': 0.00,
                       }

# Physical pixel size
pixel_scale_mm = 0.01 # mm

# There are actually separate pupil plane files for each SCA, since the view of the pupil
# obscuration is different from different locations on the focal plane.  It's also modestly
# wavelength dependent, so there is a different file appropriate for F184, the longest wavelength
# filter.  This file is for SCA2, which is near the center and for short wavelengths, so in
# some sense the most typical example of the pupil mask.  If anyone needs a generic pupil
# plane file to use, this one should be fine.
pupil_plane_file = os.path.join(meta_data.share_dir, 'roman', 'SCA2_rim_mask.fits.gz')

# The pupil plane files all keep track of their correct pixel scale, but for the exit pupil,
# rather than the input pupil.  The scaling to use to get to the entrance pupil, which is what
# we actually want, is in the header as PUPILMAG. The result for the above file is given here.
pupil_plane_scale = 0.00111175097

# Which bands should use the long vs short pupil plane files for the PSF.
# F184, K213
longwave_bands = ['F184', 'K213']
# R062, Z087, Y106, J129, H158, W146, SNPrism, Grism_0thOrder, Grism_1stOrder.
# Note that the last three are not imaging bands.
non_imaging_bands = ['Grism_0thOrder', 'Grism_1stOrder', 'SNPrism']
shortwave_bands = ['R062', 'Z087', 'Y106', 'J129', 'H158', 'W146'] \
    + non_imaging_bands

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
# The info and implementation can be found in roman_detectors.applyPersistence() and roman_detectors.fermi_linear().
persistence_fermi_parameters = np.array([0.017, 60000., 50000., 0.045, 1., 50000.])

n_sca = 18
n_pix_tot = 4096
n_pix = 4088
jitter_rms = 0.014
charge_diffusion = 0.1

# Maxinum allowed angle from the telecope solar panels to the sun in degrees.
max_sun_angle = 36.

from .roman_bandpass import getBandpasses
from .roman_backgrounds import getSkyLevel
from .roman_psfs import getPSF
from .roman_wcs import getWCS, findSCA, allowedPos, bestPA, convertCenter
from .roman_detectors import applyNonlinearity, addReciprocityFailure, applyIPC, applyPersistence, allDetectorEffects, NLfunc
from . import roman_config
