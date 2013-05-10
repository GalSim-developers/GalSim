import os
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import test_sersic_highn_basic

nobs = test_sersic_highn_basic.NOBS
ntest = len(test_sersic_highn_basic.SERSIC_N_TEST)

for test_case in ("basic", "alias2", "maxk2", "wmult2", "alias2_maxk2_wmult2"):

    outfile = os.path.join("outputs", "sersic_highn_"+test_case+"_output_N"+str(nobs)+".asc")
    print "Generating plots for "+outfile
    # Ready some empty arrays for storing the output
    g1obs_draw = np.empty((nobs, ntest))
    g2obs_draw = np.empty((nobs, ntest))
    sigma_draw = np.empty((nobs, ntest))
    delta_g1obs = np.empty((nobs, ntest))
    delta_g2obs = np.empty((nobs, ntest))
    delta_sigma = np.empty((nobs, ntest))
    err_g1obs = np.empty((nobs, ntest))
    err_g2obs = np.empty((nobs, ntest))
    err_sigma = np.empty((nobs, ntest))
    ntest_output = np.empty((nobs, ntest))
    hlr_output = np.empty((nobs, ntest))
    g1_output = np.empty((nobs, ntest))
    g2_output = np.empty((nobs, ntest))
    # Load the data into these arrays
    data = np.loadtxt(outfile)
    for j in range(ntest):

        g1obs_draw[:, j] = data[range(j, ntest * nobs, ntest), 0]
        g2obs_draw[:, j] = data[range(j, ntest * nobs, ntest), 1]
        sigma_draw[:, j] = data[range(j, ntest * nobs, ntest), 2]
        delta_g1obs[:, j] = data[range(j, ntest * nobs, ntest), 3]
        delta_g2obs[:, j] = data[range(j, ntest * nobs, ntest), 4]
        delta_sigma[:, j] = data[range(j, ntest * nobs, ntest), 5]
        err_g1obs[:, j] = data[range(j, ntest * nobs, ntest), 6]
        err_g2obs[:, j] = data[range(j, ntest * nobs, ntest), 7]
        err_sigma[:, j] = data[range(j, ntest * nobs, ntest), 8]
        ntest_output[:, j] = data[range(j, ntest * nobs, ntest), 9]
        hlr_output[:, j] = data[range(j, ntest * nobs, ntest), 10]
        g1_output[:, j] = data[range(j, ntest * nobs, ntest), 11]
        g2_output[:, j] = data[range(j, ntest * nobs, ntest), 12]

    # First do the plots of g1
    YMAX_ZOOMOUT = 0.25
    YMAX_ZOOMIN = 2.5e-4
    XMIN = -.6
    XMAX = .8
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
            g1obs_draw[:, i], delta_g1obs[:, i], yerr=err_g1obs[:, i], fmt=fmt,
            label="n = "+str(test_sersic_highn_basic.SERSIC_N_TEST[i]))
    plt.xlabel(r'g$_1$ (DFT)')
    plt.ylabel(r'$\Delta$g$_1$ (DFT - Photon)')
    plt.ylim(-YMAX_ZOOMIN, YMAX_ZOOMIN)
    plt.legend()
    plt.title(test_case)
    plt.subplots_adjust(left=0.15)
    plt.savefig(os.path.join('plots', 'sersic_highn_'+test_case+'_zoomin_g1.png'))
    plt.ylim(-YMAX_ZOOMOUT, YMAX_ZOOMOUT)
    plt.subplots_adjust(left=None)
    plt.savefig(os.path.join('plots', 'sersic_highn_'+test_case+'_zoomout_g1.png'))

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
    plt.title(test_case)
    plt.savefig(os.path.join('plots', 'sersic_highn_'+test_case+'_zoomin_g2.png'))
    plt.ylim(-YMAX_ZOOMOUT, YMAX_ZOOMOUT)
    plt.savefig(os.path.join('plots', 'sersic_highn_'+test_case+'_zoomout_g2.png'))

    # Then do the plots of sigma
    YMAX_ZOOMOUT = .3   # in arcsec
    YMAX_ZOOMIN = 2.5e-3 # in arcsec
    XMIN = 0.
    XMAX = 1.25 # in arcsec
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
    plt.title(test_case)
    plt.savefig(os.path.join('plots', 'sersic_highn_'+test_case+'_zoomin_sigma.png'))
    plt.ylim(-YMAX_ZOOMOUT, YMAX_ZOOMOUT)
    plt.savefig(os.path.join('plots', 'sersic_highn_'+test_case+'_zoomout_sigma.png'))
