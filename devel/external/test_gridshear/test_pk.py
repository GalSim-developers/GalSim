import galsim
import numpy as np
import g10_powerspec

n_realization = 500
pkfile = 'ps.wmap7lcdm.2000.append0.dat'
n_ell = 50
grid_nx = 50
grid_spacing = 96. # pixels, but assume fixed grid of 10 deg- see below!
theta = 10. # degrees
dtheta = theta/grid_nx # degrees
tmpfile = 'tmp.out'
outfile = 'output/ps.results.input_pb.fine.dat'

ellvals = np.zeros(n_ell)
p_e = np.zeros((n_ell, n_realization))
p_b = np.zeros((n_ell, n_realization))
p_eb = np.zeros((n_ell, n_realization))

test_ps_e=galsim.PowerSpectrum(e_power_function = pkfile, units='radians')
test_ps_b=galsim.PowerSpectrum(b_power_function = pkfile, units='radians')
test_ps_eb=galsim.PowerSpectrum(e_power_function = pkfile, b_power_function = pkfile, units='radians')

# define grid x, y
x, y = np.meshgrid(np.arange(0., grid_nx*grid_spacing, grid_spacing),
                   np.arange(0., grid_nx*grid_spacing, grid_spacing))
x = 0.5*grid_spacing + x
y = 0.5*grid_spacing + y
x_1d = np.reshape(x, grid_nx*grid_nx)
y_1d = np.reshape(y, grid_nx*grid_nx)

for ireal in range(n_realization):
    print "Iteration ",ireal

    print "Getting shears on a grid"
    g1, g2 = test_ps_b.buildGriddedShears(grid_spacing=dtheta, ngrid=grid_nx, units=galsim.degrees)
    g2 = -1.*g2

    g1_1d = np.reshape(g1, grid_nx*grid_nx)
    g2_1d = np.reshape(g2, grid_nx*grid_nx)
    # save to file in required format
    data_all = (g1_1d, g2_1d, x_1d, y_1d)
    data = np.column_stack(data_all)
    np.savetxt(tmpfile, data)
    ein = g10_powerspec.readells(tmpfile, n=grid_nx, step=grid_spacing)
    myps = g10_powerspec.ps(ein, step=grid_spacing, nbin2=n_ell, size=theta) ###!!
    myps.setup()
    myps.create()
    myps.angavg()
    ellvals = myps.ll
    p_e[:,ireal] = myps.gPowEE
    p_b[:,ireal] = myps.gPowBB
    p_eb[:,ireal] = myps.gPowEB

avgp_e = np.mean(p_e,1)
avgp_b = np.mean(p_b,1)
avgp_eb = np.mean(p_eb,1)
avgp_e_pow = 2.*np.pi*avgp_e[ellvals>0.]/ellvals[ellvals>0.]**2
avgp_b_pow = 2.*np.pi*avgp_b[ellvals>0.]/ellvals[ellvals>0.]**2
avgp_eb_pow = 2.*np.pi*avgp_eb[ellvals>0.]/ellvals[ellvals>0.]**2
ellvals = ellvals[ellvals>0.]
data_all = (ellvals, avgp_e_pow, avgp_b_pow, avgp_eb_pow)
data = np.column_stack(data_all)
np.savetxt(outfile, data)
