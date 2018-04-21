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
import subprocess
from scipy.special import jv

###### UP HERE IS WHERE NUMBERS AND SO ON ARE DEFINED.
###### THEN THERE ARE HELPER ROUTINES.
###### AND THEN THE ACTUAL CODE IS AT THE BOTTOM.
# file containing theoretical P(k), with fake values added above ell=2000
pkfile = 'ps.wmap7lcdm.2000.dat'
# read it in as a lookup table to pass to power spectrum code
theory_tab = galsim.LookupTable(file=pkfile, interpolant='linear')
# N for our grid used for estimating shears
grid_nx = 100
# length of grid in one dimension (degrees)
theta = 10.
# grid spacing
dtheta = theta/grid_nx
# How many iterations to average over, in order to beat down noise?
n_iter = 100
file_prefix = 'test_pse_corr_new' # prefix for output file
kmin_factor = 3
kmax_factor = 2

# parameters for corr2:
min_sep = dtheta # lowest bin starts at grid spacing
max_sep = grid_nx * np.sqrt(2) * dtheta # upper edge of upper bin is at maximum pair separation
nbins = 100 # lots of bins!

# Make the program deterministic by setting this random seed.  So if you run again you should get
# the same answer (assuming you didn't change other stuff)
rng = galsim.BaseDeviate(1234)

class xi_integrand:
    """Helper routine that defines the integrand when taking a theoretical P(k) and converting to
    xi.  Basically, k*P(k)*(appropriate Bessel function)."""
    def __init__(self, pk, r, n):
        self.pk = pk
        self.r = r
        self.n = n
    def __call__(self, k):
        return k * self.pk(k) * jv(self.n, self.r*k)
        
def calculate_xi(r, pk, n):
    """Helper routine that takes a P(k) and a set of separations at which you want xi+ and xi-, and
    does the math to get the correlation functions properly.
    """
    #print 'Start calculate_xi'
    # xi+/-(r) = 1/2pi int(dk k P(k) J0/4(kr), k=0..inf)

    int_min = pk.x_min
    int_max = pk.x_max
    rrad = r * np.pi/180.  # Convert to radians

    xi = np.zeros_like(r)
    for i in range(len(r)):
        integrand = xi_integrand(pk, rrad[i], n)
        xi[i] = galsim.integ.int1d(integrand, int_min, int_max,
                                   rel_err=1.e-6, abs_err=1.e-12)
    xi /= 2. * np.pi
    return xi

def doplot(r, t_xip, t_xim, xip, xim, pref):
    """Helper routine to make the final plots."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nonzero = (xip != 0.)
    ax.plot(r, t_xip, 'black', label='Theory xi+')
    ax.plot(r, t_xim, 'grey', label='Theory xi-')
    ax.plot(r[nonzero], xip[nonzero], 'blue', label='Observed xi+')
    ax.plot(r[nonzero], xim[nonzero], 'green', label='Observed xi-')
    ax.plot(r, -t_xip, 'black', ls='dashed')
    ax.plot(r, -t_xim, 'grey', ls='dashed')
    ax.plot(r[nonzero], -xip[nonzero], 'blue', ls='dashed')
    ax.plot(r[nonzero], -xim[nonzero], 'green', ls='dashed')
    plt.ylim(1e-8,2e-5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r [arcmin]')
    ax.set_ylabel(r'$\xi$')
    ax.set_title('Shear-Shear Correlations')
    plt.legend(loc='upper right')
    figfile = pref + '.png'
    plt.savefig(figfile)
    print 'Wrote to file ',figfile

def run_treecorr(x, y, g1, g2):
    """Helper routine to take outputs of GalSim shear grid routine, and run treecorr on it."""
    import pyfits
    import os
    import treecorr
    # Use fits binary table for faster I/O.
    assert x.shape == y.shape
    assert x.shape == g1.shape
    assert x.shape == g2.shape
    x_col = pyfits.Column(name='x', format='1D', array=x.flatten() )
    y_col = pyfits.Column(name='y', format='1D', array=y.flatten() )
    g1_col = pyfits.Column(name='g1', format='1D', array=g1.flatten() )
    g2_col = pyfits.Column(name='g2', format='1D', array=g2.flatten() )
    cols = pyfits.ColDefs([x_col, y_col, g1_col, g2_col])
    table = pyfits.new_table(cols)
    phdu = pyfits.PrimaryHDU()
    hdus = pyfits.HDUList([phdu,table])
    hdus.writeto('temp.fits',clobber=True)
    # Define the treecorr catalog object.
    cat = treecorr.Catalog('temp.fits',x_units='degrees',y_units='degrees',
                           x_col='x',y_col='y',g1_col='g1',g2_col='g2')
    # Define the corrfunc object
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=0.1, sep_units='degrees')
    # Actually calculate the correlation function.
    gg.process(cat)
    os.remove('temp.fits')
    return gg

# Here's where we actually do stuff.  Start by making the PowerSpectrum object, and defining the
# grid range.
test_ps=galsim.PowerSpectrum(e_power_function = theory_tab, units='radians')
grid_range = dtheta * np.arange(grid_nx)
x, y = np.meshgrid(grid_range, grid_range)

# Now we do the iterations to build the shear grids.
for ind in range(n_iter):
    print 'Building grid %d'%ind
    g1, g2 = test_ps.buildGrid(grid_spacing=dtheta, ngrid=grid_nx,
                               rng=rng, units='degrees', kmin_factor=kmin_factor,
                               kmax_factor=kmax_factor)

    print 'Calculating correlations %d'%ind
    gg = run_treecorr(x,y,g1,g2)
    if ind == 0:
        r = np.exp(gg.meanlogr)
        xip = gg.xip
        xim = gg.xim
    else:
        xip += gg.xip
        xim += gg.xim

# Take the average correlation function when all iterations are done.
xip /= n_iter
xim /= n_iter

print "Converting theory from PS to correlation functions"
theory_xip = calculate_xi(r,theory_tab,0)
theory_xim = calculate_xi(r,theory_tab,4)
theory_xiket = calculate_xi(r,theory_tab,2)

print "Making figures of dimensionless power, and writing to files"
doplot(r, theory_xip, theory_xim,  xip, xim,  file_prefix)

