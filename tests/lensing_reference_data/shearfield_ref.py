import numpy as np
import galsim

# The reference data for this test was generated with this script, using the version of the code on
# branch #216 at commit 

# random seed, etc.
outfile = 'shearfield_reference.dat'
rng = galsim.UniformDeviate(14136)

# make grid params
n = 10
dx = 1.

# define power spectrum
ps = galsim.lensing.PowerSpectrum(E_power_function=galsim.lensing.pk2,
                                  B_power_function=galsim.lensing.pk1)
# get shears
g1, g2 = ps.getShear(grid_spacing = dx, grid_nx = n, gaussian_deviate=rng)

# write to file
g1vec = g1.reshape(n*n)
g2vec = g2.reshape(n*n)
data = (g1vec, g2vec)
data = np.column_stack(data)
np.savetxt(outfile, data)

