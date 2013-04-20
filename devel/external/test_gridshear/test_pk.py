import galsim
import numpy as np
import g10_powerspec
import pse

n_realization = 1000
pkfile = 'ps.wmap7lcdm.2000.append0.dat'
n_ell = 15
grid_nx = 50
theta = 10. # degrees
dtheta = theta/grid_nx # degrees
outfile = 'output/ps.results.input_pe.pse.dat'

ellvals = np.zeros(n_ell)
p_e = np.zeros((n_ell, n_realization))
p_b = np.zeros((n_ell, n_realization))
p_eb = np.zeros((n_ell, n_realization))

tab = galsim.LookupTable(file=pkfile, interpolant='linear', x_log=True, f_log=True)
test_ps_e=galsim.PowerSpectrum(e_power_function = tab, units='radians')
test_ps_b=galsim.PowerSpectrum(b_power_function = tab, units='radians')
test_ps_eb=galsim.PowerSpectrum(e_power_function = tab, b_power_function = tab, units='radians')

my_pse = pse.PowerSpectrumEstimator(grid_nx, theta, n_ell)

for ireal in range(n_realization):
    print "Iteration ",ireal

    print "Getting shears on a grid"
    g1, g2 = test_ps_e.buildGrid(grid_spacing=dtheta, ngrid=grid_nx, units=galsim.degrees)
    this_ell, this_pe, this_pb, this_peb, this_theory = my_pse.estimate(g1, g2, theory_func=tab)

    ellvals = this_ell
    p_e[:,ireal] = this_pe
    p_b[:,ireal] = this_pb
    p_eb[:,ireal] = this_peb

avgp_e = np.mean(p_e,1)
avgp_b = np.mean(p_b,1)
avgp_eb = np.mean(p_eb,1)
avgp_e_pow = avgp_e[ellvals>0.]
avgp_b_pow = avgp_b[ellvals>0.]
avgp_eb_pow = avgp_eb[ellvals>0.]
theory = this_theory
ellvals = ellvals[ellvals>0.]
data_all = (ellvals, avgp_e_pow, avgp_b_pow, avgp_eb_pow, theory)
data = np.column_stack(data_all)
np.savetxt(outfile, data)
