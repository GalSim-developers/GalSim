# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
import numpy as np
import galsim

# The reference data for this test was generated with this script, using the version of the code on
# branch #304 at commit 21afd0f86886d5fe3c1bdc510b19352e1ced7f47, i.e.,
# https://github.com/GalSim-developers/GalSim/commit/21afd0f86886d5fe3c1bdc510b19352e1ced7f47

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

