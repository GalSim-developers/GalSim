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
Script to generate some test (Fourier) images of Spergelet profiles.
"""

import os
import numpy as np
from scipy.special import gamma
import astropy.io.fits as fits

def trig(q):
    if q > 0:
        return lambda phi: np.cos(2*q*phi)
    elif q < 0:
        return lambda phi: np.sin(2*q*phi)
    elif q == 0:
        return lambda phi: 1.0

def spergelet(nu, j, q):
    tr = trig(q)
    def mu(kxs, kys):
        k2s = kxs**2 + kys**2
        phis = np.arctan2(kys, kxs)
        return k2s**j/(1+k2s)**(1+nu+j) * gamma(nu+j+1) / gamma(nu+1) * tr(phis)
    return mu

#make some images
stamp_size = 63
dx = 0.2
kxs = np.arange(-(stamp_size-1)/2, (stamp_size-1)/2+0.000001) * dx
kys = kxs
kxs, kys = np.meshgrid(kxs, kys)
sr = 1.0

hdulist = fits.HDUList()

for nu in [-0.5, 0.5]:
    for (j,q) in [(0,0), (5,-5), (5,0), (5,5)]:
        img = spergelet(nu, j, q)(kxs, kys)
        output_file = "spergelet_nu{:.2f}_j{}_q{}.fits".format(nu, j, q)

        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU(img)
        hdulist.append(hdu)
        if os.path.isfile(output_file):
            os.remove(output_file)
        hdulist.writeto(output_file)
