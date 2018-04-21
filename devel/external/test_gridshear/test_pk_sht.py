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
import g10_powerspec

n_realization = 5
pkpref = 'sht_outputs/gridshear.peb.'
pksuff = '.out'
grid_nx = 50
grid_spacing = 96. # pixels
tmpfile = 'tmp.out'
outfile = 'output/ps.results.input_peb.sht.dat'
n_ell = 20

ellvals = np.zeros(n_ell)
p_e = np.zeros((n_ell, n_realization))
p_b = np.zeros((n_ell, n_realization))
p_eb = np.zeros((n_ell, n_realization))

# define grid x, y
x, y = np.meshgrid(np.arange(0., grid_nx*grid_spacing, grid_spacing),
                   np.arange(0., grid_nx*grid_spacing, grid_spacing))
x = 0.5*grid_spacing + x
y = 0.5*grid_spacing + y
x_1d = np.reshape(x, grid_nx*grid_nx)
y_1d = np.reshape(y, grid_nx*grid_nx)

for ireal in range(1,n_realization+1):
    infile = pkpref+str(ireal)+pksuff
    
    data=np.loadtxt(infile)
    g1=-1*data[:,2]
    g2=data[:,3]
    data_all = (g1, g2, x_1d, y_1d)
    data = np.column_stack(data_all)
    np.savetxt(tmpfile, data)
    ein = g10_powerspec.readells(tmpfile, n=grid_nx, step=grid_spacing)
    myps = g10_powerspec.ps(ein, step=grid_spacing, nbin2=n_ell)
    myps.setup()
    myps.create()
    myps.angavg()
    ellvals = myps.ll
    p_e[:,ireal-1] = myps.gPowEE
    p_b[:,ireal-1] = myps.gPowBB
    p_eb[:,ireal-1] = myps.gPowEB

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
