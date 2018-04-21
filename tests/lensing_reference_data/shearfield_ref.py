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
import numpy as np
import galsim

# The reference data for this test was generated with this script, using the version of the code on
# branch #304 at commit df4b15e.

# random seed, etc.
outfile = 'shearfield_reference.dat'
rng = galsim.BaseDeviate(14136)

# make grid params
n = 10
dx = 1.

# define power spectrum
ps = galsim.lensing.PowerSpectrum(e_power_function="k**0.5", b_power_function="k")
# get shears and convergences
g1, g2, kappa = ps.buildGrid(grid_spacing=dx, ngrid=n, rng=rng, get_convergence=True)

# write to file
g1vec = g1.reshape(n*n)
g2vec = g2.reshape(n*n)
kappavec = kappa.reshape(n*n)
data = (g1vec, g2vec, kappavec)
data = np.column_stack(data)
np.savetxt(outfile, data)

