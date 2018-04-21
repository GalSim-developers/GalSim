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

# imports, etc.
import galsim
import galsim.wfirst as wf
import datetime
import numpy as np
import matplotlib.pyplot as plt
from radec_to_chip import *

# Make a list of RA/dec central values and nearby values
n_vals = 100
seed = 314159
ud = galsim.UniformDeviate(seed=seed)
min_ra = 0.0
max_ra = 360.0
min_cos_dec = -0.95
max_cos_dec = 0.3
ra_cen_vals = np.zeros(n_vals)
dec_cen_vals = np.zeros(n_vals)
ra_vals = np.zeros(n_vals)
dec_vals = np.zeros(n_vals)
delta_dist = 0.5 # degrees offset allowed for (ra, dec) compared to center of focal plane
chris_sca = np.zeros(n_vals).astype(int)
pa_arr = np.zeros(n_vals)
date = datetime.datetime(2025, 1, 12)
for i in range(n_vals):
    # Keep choosing random FPA center positions until we get one that can be observed on the chosen
    # date.
    pa = None
    while (pa is None):
        ra_cen_vals[i] = min_ra + (max_ra-min_ra)*ud()
        dec_cen_vals[i] = \
            90.0-(180.0/np.pi)*np.arccos(min_cos_dec + (max_cos_dec-min_cos_dec)*ud())
        fpa_center = galsim.CelestialCoord(
            ra=ra_cen_vals[i]*galsim.degrees,
            dec=dec_cen_vals[i]*galsim.degrees)
        pa = wf.bestPA(fpa_center, date)
    pa_arr[i] = pa / galsim.radians
    ra_vals[i] = ra_cen_vals[i] + delta_dist*(ud()-0.5)*np.cos(dec_cen_vals[i]*np.pi/180.)
    dec_vals[i] = dec_cen_vals[i] + delta_dist*(ud()-0.5)
    # Find the SCAs from Chris's code (Python version) for the same points (0=not on an SCA)
    chris_sca[i] = radec_to_chip(np.array([ra_cen_vals[i]*np.pi/180.0]),
                                 np.array([dec_cen_vals[i]*np.pi/180.0]),
                                 np.array([pa]),
                                 np.array([ra_vals[i]*np.pi/180.]),
                                 np.array([dec_vals[i]*np.pi/180.]))

out_data = np.column_stack((ra_cen_vals, dec_cen_vals, ra_vals, dec_vals, pa_arr, chris_sca))
np.savetxt('chris_comparison.txt', out_data, fmt='%.8f %.8f %.8f %.8f %.8f %d')

