import galsim
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.special import jv

# This uses Mike Jarvis's corr2 program for calculating the correlation function.
# It is available at https://code.google.com/p/mjarvis/ and needs to be installed separately.

### set up basic parameters ###
# file containing theoretical P(k), with fake values added above ell=2000
pkfile = 'ps.wmap7lcdm.2000.dat'
theory_tab = galsim.LookupTable(file=pkfile, interpolant='linear')
# N for our grid used for estimating shears
grid_nx = 100
# length of grid in one dimension (degrees)
theta = 10. # degrees
dtheta = theta/grid_nx
extra_res = 10      # Extra resolution factor for g1,g2 grid.

# parameters for corr2:
min_sep = dtheta
max_sep = grid_nx * np.sqrt(2) * dtheta
nbins = 100

# Make deterministic
rng = galsim.BaseDeviate(1234)

# To save time debugging, use the existing corr files
use_saved = False

class xi_integrand:
    def __init__(self, pk, r, n):
        self.pk = pk
        self.r = r
        self.n = n
    def __call__(self, k):
        return k * self.pk(k) * jv(self.n, self.r*k)
        
def calculate_xi(r, pk, n):
    """Calculate xi+(r) or xi-(r) from a power spectrum.
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

def doplot(r, t_xip, t_xim, t_xiket, xip, xim, xix, xik, xiket, xikex, pref):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nonzero = (xip != 0.)
    ax.plot(r, t_xip, 'black', label='Theory xi+')
    ax.plot(r, t_xim, 'grey', label='Theory xi-')
    ax.plot(r[nonzero], xip[nonzero], 'blue', label='Observed xi+')
    ax.plot(r[nonzero], xim[nonzero], 'green', label='Observed xi-')
    ax.plot(r[nonzero], xix[nonzero], 'red', label='Observed xix')
    ax.plot(r, -t_xip, 'black', ls='dashed')
    ax.plot(r, -t_xim, 'grey', ls='dashed')
    ax.plot(r[nonzero], -xip[nonzero], 'blue', ls='dashed')
    ax.plot(r[nonzero], -xim[nonzero], 'green', ls='dashed')
    ax.plot(r[nonzero], -xix[nonzero], 'red', ls='dashed')
    plt.ylim(1e-8,2e-5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\xi$')
    ax.set_title('Shear-Shear Correlations')
    plt.legend(loc='upper right')
    figfile = pref + '_e2.jpg'
    plt.savefig(figfile)
    print 'Wrote to file ',figfile

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(r, t_xip, 'black', label='Theory xi_kappa')
    ax.plot(r[nonzero], xik[nonzero], 'blue', label='Observed xi_kappa')
    ax.plot(r, -t_xip, 'black', ls='dashed')
    ax.plot(r[nonzero], -xik[nonzero], 'blue', ls='dashed')
    plt.ylim(1e-8,2e-5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\xi$')
    ax.set_title('Kappa-Kappa Correlations')
    plt.legend(loc='upper right')
    figfile = pref + '_k2.jpg'
    plt.savefig(figfile)
    print 'Wrote to file ',figfile

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(r, t_xiket, 'black', label='Theory <kappa gamma_t>')
    ax.plot(r[nonzero], xiket[nonzero], 'blue', label='Observed <kappa gamma_t>')
    ax.plot(r[nonzero], xikex[nonzero], 'red', label='Observed <kappa gamma_x>')
    ax.plot(r, -t_xiket, 'black', ls='dashed')
    ax.plot(r[nonzero], -xiket[nonzero], 'blue', ls='dashed')
    ax.plot(r[nonzero], -xikex[nonzero], 'red', ls='dashed')
    plt.ylim(1e-8,2e-5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\xi$')
    ax.set_title('Kappa-Shear Correlations')
    plt.legend(loc='upper right')
    figfile = pref + '_ke.jpg'
    plt.savefig(figfile)
    print 'Wrote to file ',figfile

def run_corr2(x, y, g1, g2, k):
    import pyfits
    import os
    # Use fits binary table for faster I/O. (Converting to/from strings is slow.)
    assert x.shape == y.shape
    assert x.shape == g1.shape
    assert x.shape == g2.shape
    assert x.shape == k.shape
    x_col = pyfits.Column(name='x', format='1D', array=x.flatten() )
    y_col = pyfits.Column(name='y', format='1D', array=y.flatten() )
    g1_col = pyfits.Column(name='g1', format='1D', array=g1.flatten() )
    g2_col = pyfits.Column(name='g2', format='1D', array=g2.flatten() )
    k_col = pyfits.Column(name='k', format='1D', array=k.flatten() )
    cols = pyfits.ColDefs([x_col, y_col, g1_col, g2_col, k_col])
    table = pyfits.new_table(cols)
    phdu = pyfits.PrimaryHDU()
    hdus = pyfits.HDUList([phdu,table])
    hdus.writeto('temp.fits',clobber=True)
    subprocess.Popen(['corr2','corr2.params',
                      'e2_file_name=temp.e2', 'k2_file_name=temp.k2',
                      'min_sep=%f'%min_sep,'max_sep=%f'%max_sep,'nbins=%f'%nbins]).wait()
    subprocess.Popen(['corr2','corr2.params',
                      'file_name2=temp.fits', 'ke_file_name=temp.ke',
                      'min_sep=%f'%min_sep,'max_sep=%f'%max_sep,'nbins=%f'%nbins]).wait()
    os.remove('temp.fits')

if use_saved:
    print 'Using existing temp.e2, temp.k2, temp.ke'
else:
    print 'Build Gridded g1,g2,kappa'
    test_ps=galsim.PowerSpectrum(e_power_function = theory_tab, units='radians')
    g1, g2, k = test_ps.buildGrid(grid_spacing=dtheta, ngrid=grid_nx*extra_res,
                                rng=rng, units='degrees', get_convergence=True)
    grid_range = dtheta * np.arange(grid_nx*extra_res)
    x, y = np.meshgrid(grid_range, grid_range)
    
    print 'Calculate correlations'
    run_corr2(x,y,g1,g2,k)

e2 = np.loadtxt('temp.e2')
k2 = np.loadtxt('temp.k2')
ke = np.loadtxt('temp.ke')
#os.remove('temp.e2')
#os.remove('temp.k2')
#os.remove('temp.ke')

r = e2[:,1]
xip = e2[:,2]
xim = e2[:,3]
xix = e2[:,5]
w = e2[:,7]
xik = k2[:,2]
xiket = ke[:,2]
xikex = ke[:,3]

print "Convert between corr and ps"
theory_xip = calculate_xi(r,theory_tab,0)
theory_xim = calculate_xi(r,theory_tab,4)
theory_xiket = calculate_xi(r,theory_tab,2)

print "Making figures of dimensionless power, and writing to files"
doplot(r, theory_xip, theory_xim, theory_xiket, xip, xim, xix, xik, xiket, xikex, 'test_pse_corr')

