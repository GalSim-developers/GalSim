# This is a quick utility for using the GREAT10 power spectrum code as translated into python to
# test our code for generating power spectra on a grid.
import galsim
import g10_powerspec
import numpy as np

tmpfile = 'tmp.shears.out'
outfile = 'tmp.testps.out'
grid_nx = 100
grid_ny = grid_nx
grid_spacing = 48.

# make the power spectrum and generate random shears on the grid
print "Making power spectra and generating random shears on a grid using GalSim lensing engine"
my_ps = galsim.lensing.PowerSpectrum(galsim.lensing.pk2)
g1, g2 = my_ps.getShear(grid_spacing = grid_spacing, grid_nx = grid_nx)
g2 = -1.*g2 # not sure about this - the g10_powerspec.py reverses g2, so if I don't do this then I
# get lots of B mode power even if the B mode power spectrum is zero, as for the test case defined
# here

# define grid x, y
x, y = np.meshgrid(np.arange(0., grid_nx*grid_spacing, grid_spacing),
                   np.arange(0., grid_ny*grid_spacing, grid_spacing))
x = 0.5*grid_spacing + x
y = 0.5*grid_spacing + y
g1_1d = np.reshape(g1, grid_nx*grid_ny)
g2_1d = np.reshape(g2, grid_nx*grid_ny)
x_1d = np.reshape(x, grid_nx*grid_ny)
y_1d = np.reshape(y, grid_nx*grid_ny)
# save to file in required format
data_all = (g1_1d, g2_1d, x_1d, y_1d)
data = np.column_stack(data_all)
np.savetxt(tmpfile, data)

# calculate power spectrum
print "Calculating power spectra using GREAT10 code"
ein = g10_powerspec.readells(tmpfile)
myps = g10_powerspec.ps(ein)
myps.setup()
myps.create()
myps.angavg()

# write to file
print "Writing results to file: ",outfile
print "Columns are ell, E power, B power, EB cross-power"
print "Power is calculated as: ell^2*C_ell/(2 pi)"
data_all = (myps.ll, myps.gPowEE, myps.gPowBB, myps.gPowEB)
data = np.column_stack(data_all)
np.savetxt(outfile, data)


