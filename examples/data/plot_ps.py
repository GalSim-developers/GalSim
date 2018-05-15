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
import galsim
import numpy as np
import matplotlib.pyplot as plt
import glob

# This is a quick little script to find an analytic approximation to the 
# cosmological power spectrum tabulated in cosmo-dit.zmed1.00.out.
# I use the functional form given by Ma, 1998 for the 3d power spectrum,
# and vary the q/k factor until it matches up reasonably well.
# cf. http://adsabs.harvard.edu/abs/1998ApJ...508L...5M

k, p = np.loadtxt('cosmo-fid.zmed1.00.out', unpack=True)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(k,p, color='black', label='cosmo_fid')
q = 4.e-2*k
A = 1.e-6
# Note: A includes more than just Chung-Pei's A factor.  Also, the power of k differs
# from what she had, since we want the 2d power spectrum, not 3d.
f = A * k**(-0.04) * np.log(1.+2.34*q)**2 / np.sqrt(1.+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)
# Here it is with the factors multiplied in.  This is the form I use in lsst.yaml.
f = 1.e-6 * k**(-0.04) * np.log(1.+0.09*k)**2 / np.sqrt(1.+0.16*k+0.41*k**2+0.01*k**3+0.005*k**4)
ax.plot(k,f, color='blue', label='analytic')
plt.savefig('ps.png')
