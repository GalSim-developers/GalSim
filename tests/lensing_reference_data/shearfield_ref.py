import numpy as np
import galsim

# The reference data for this test was generated with this script, using the version of the code on
# branch #305 at commit e06b7604090cab8de6ec681cca8b74ace386d240, i.e.,
# https://github.com/GalSim-developers/GalSim/commit/e06b7604090cab8de6ec681cca8b74ace386d240

# random seed, etc.
outfile = 'shearfield_reference.dat'
rng = galsim.BaseDeviate(14136)

# make grid params
n = 10
dx = 1.

# define power spectrum
ps = galsim.lensing.PowerSpectrum(e_power_function="k**2", b_power_function="k")
# get shears
g1, g2 = ps.buildGriddedShears(grid_spacing = dx, ngrid = n, rng=rng)

# write to file
g1vec = g1.reshape(n*n)
g2vec = g2.reshape(n*n)
data = (g1vec, g2vec)
data = np.column_stack(data)
np.savetxt(outfile, data)

