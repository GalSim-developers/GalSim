import os
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import test_sersic_highn_basic

YMAX_ZOOMOUT = 0.25
YMAX_ZOOMIN = 2.5e-4
XMIN = -.5
XMAX = .8

results_tuple = cPickle.load(open(test_sersic_highn_basic.OUTFILE, 'rb'))
g1obs_draw = results_tuple[0]
g2obs_draw = results_tuple[1]
sigma_draw = results_tuple[2]
delta_g1obs = results_tuple[3]
delta_g2obs = results_tuple[4]
delta_sigma = results_tuple[5]
err_g1obs = results_tuple[6]
err_g2obs = results_tuple[7]
err_sigma = results_tuple[8]

# Urgh changed the sersic n values in the last commit, so this is TEMPORARY!
test_sersic_highn_basic.SERSIC_N_TEST = [3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.]

ntest = len(test_sersic_highn_basic.SERSIC_N_TEST)

# First do the plots of g1
plt.axhline(ls='--', color='k')
plt.axvline(ls='--', color='k')
plt.xlim(XMIN, XMAX)
for i in range(ntest):
    if i < 7:
        fmt='x'
    else:
        fmt='o'
    plt.errorbar(
        g1obs_draw[:, i], delta_g1obs[:, i], yerr=err_g1obs[:, i], fmt=fmt,
        label="n = "+str(test_sersic_highn_basic.SERSIC_N_TEST[i]))

plt.xlabel(r'g$_1$ (DFT)')
plt.ylabel(r'$\Delta$g$_1$ (DFT - Photon)')
plt.ylim(-YMAX_ZOOMIN, YMAX_ZOOMIN)
plt.legend()
plt.subplots_adjust(left=0.15)
plt.savefig(os.path.join('plots', 'sersic_highn_basic_zoomin_g1.png'))

plt.ylim(-YMAX_ZOOMOUT, YMAX_ZOOMOUT)
plt.subplots_adjust(left=None)
plt.savefig(os.path.join('plots', 'sersic_highn_basic_zoomout_g1.png'))

# Then do the plots of g2
plt.clf()
plt.axhline(ls='--', color='k')
plt.axvline(ls='--', color='k')
plt.xlim(XMIN, XMAX)
for i in range(ntest):
    if i < 7:
        fmt='x'
    else:
        fmt='o'
    plt.errorbar(
        g2obs_draw[:, i], delta_g2obs[:, i], yerr=err_g2obs[:, i], fmt=fmt,
        label="n = "+str(test_sersic_highn_basic.SERSIC_N_TEST[i]))

plt.xlabel(r'g$_2$ (DFT)')
plt.ylabel(r'$\Delta$g$_2$ (DFT - Photon)')
plt.ylim(-YMAX_ZOOMIN, YMAX_ZOOMIN)
plt.legend()
plt.savefig(os.path.join('plots', 'sersic_highn_basic_zoomin_g2.png'))

plt.ylim(-YMAX_ZOOMOUT, YMAX_ZOOMOUT)
plt.savefig(os.path.join('plots', 'sersic_highn_basic_zoomout_g2.png'))

# Then do the plots of sigma
YMAX_ZOOMOUT = .3   # in arcsec
YMAX_ZOOMIN = 4.e-4 # in arcsec
XMIN = 0.
XMAX = .3 # in arcsec
plt.clf()
plt.xlim(XMIN, XMAX)
for i in range(ntest):
    if i < 7:
        fmt='x'
    else:
        fmt='o'
    plt.errorbar(
        sigma_draw[:, i] * test_sersic_highn_basic.PIXEL_SCALE,
        delta_sigma[:, i] * test_sersic_highn_basic.PIXEL_SCALE,
        yerr=err_sigma[:, i] * test_sersic_highn_basic.PIXEL_SCALE, fmt=fmt,
        label="n = "+str(test_sersic_highn_basic.SERSIC_N_TEST[i]))

plt.ylim(-YMAX_ZOOMIN, YMAX_ZOOMIN)
plt.xlabel(r'$\sigma$ (DFT) [arcsec]')
plt.ylabel(r'$\Delta \sigma$ (DFT - Photon) [arcsec]')
plt.legend()
plt.show()
