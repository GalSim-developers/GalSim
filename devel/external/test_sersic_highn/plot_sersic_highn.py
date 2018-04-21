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

import os
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import test_sersic_highn_basic
import fitting

nobs = test_sersic_highn_basic.NOBS
ntest = len(test_sersic_highn_basic.SERSIC_N_TEST)

for test_case in ("basic",):# "alias2", "maxk2", "wmult2", "alias2_maxk2_wmult2", "kvalue10"):
    # "shoot_accuracy2", "shoot_relerr2", "shoot_abserr2"):

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

        g1obs_draw[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 0]
        g2obs_draw[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 1]
        sigma_draw[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 2]
        delta_g1obs[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 3]
        delta_g2obs[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 4]
        delta_sigma[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 5]
        err_g1obs[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 6]
        err_g2obs[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 7]
        err_sigma[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 8]
        ntest_output[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 9]
        hlr_output[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 10]
        g1_output[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 11]
        g2_output[0:data.shape[0]/ntest, j] = data[range(j, data.shape[0], ntest), 12]

    # First do the plots of g1
    YMAX_ZOOMIN = 1e-4
    XMIN = -.6
    XMAX = .5
    plt.clf()
    plt.axhline(ls='--', color='k')
    plt.axvline(ls='--', color='k')
    plt.xlim(XMIN, XMAX)
    for i in range(ntest)[:-1]:
        # First fit a line to the points
        c, m, var_c, cov_cm, var_m = fitting.fitline(
            g1obs_draw[0:data.shape[0]/ntest, i], delta_g1obs[0:data.shape[0]/ntest, i])
        if i < 7:
            fmt='x'
        else:
            fmt='o'
        plt.errorbar(
            g1obs_draw[0:data.shape[0]/ntest, i], delta_g1obs[0:data.shape[0]/ntest, i],
            yerr=err_g1obs[0:data.shape[0]/ntest, i], fmt=fmt,
            label=r"n = %.1f, m = %.2e $\pm$ %.2e" % (
                test_sersic_highn_basic.SERSIC_N_TEST[i], m, np.sqrt(var_m)))
    plt.xlabel(r'g$_1$ (DFT)')
    plt.ylabel(r'$\Delta$g$_1$ (DFT - Photon)')
    plt.ylim(-YMAX_ZOOMIN, YMAX_ZOOMIN)
    plt.legend()
    #plt.title(test_case)
    plt.subplots_adjust(left=0.15)
    plt.savefig(os.path.join('plots', 'sersic_highn_'+test_case+'_zoomin_g1.png'))

    # Then do the plots of g2
    plt.clf()
    plt.axhline(ls='--', color='k')
    plt.axvline(ls='--', color='k')
    plt.xlim(XMIN, XMAX)
    for i in range(ntest)[:-1]:
        # First fit a line to the points
        c, m, var_c, cov_cm, var_m = fitting.fitline(
            g2obs_draw[0:data.shape[0]/ntest, i], delta_g2obs[0:data.shape[0]/ntest, i])
        if i < 7:
            fmt='x'
        else:
            fmt='o'
        plt.errorbar(
             g2obs_draw[0:data.shape[0]/ntest, i], delta_g2obs[0:data.shape[0]/ntest, i],
             yerr=err_g2obs[0:data.shape[0]/ntest, i], fmt=fmt,
             label=r"n = %.1f, m = %.2e $\pm$ %.2e" % (
                test_sersic_highn_basic.SERSIC_N_TEST[i], m, np.sqrt(var_m)))
    plt.xlabel(r'g$_2$ (DFT)')
    plt.ylabel(r'$\Delta$g$_2$ (DFT - Photon)')
    plt.ylim(-YMAX_ZOOMIN, YMAX_ZOOMIN)
    plt.legend()
    #plt.title(test_case)
    plt.savefig(os.path.join('plots', 'sersic_highn_'+test_case+'_zoomin_g2.png'))

    # Then do the plots of sigma
    YMAX_ZOOMIN = 5 # in 1e-5 arcsec
    XMIN = 0.
    XMAX = 0.7 # in arcsec
    plt.clf()
    plt.xlim(XMIN, XMAX)
    print ""
    print ""
    print ""
    for i in range(ntest)[:-1]:
        # First fit a line to the points
        c, m, var_c, cov_cm, var_m = fitting.fitline(
            sigma_draw[0:data.shape[0]/ntest, i], delta_sigma[0:data.shape[0]/ntest, i])
        print "sigma results for n = "+str(test_sersic_highn_basic.SERSIC_N_TEST[i])+":"
        print "c = %.2e +/- %.2e arcsec" % (
            c * test_sersic_highn_basic.PIXEL_SCALE,
            np.sqrt(var_c) * test_sersic_highn_basic.PIXEL_SCALE)
        if i < 7:
            fmt='x'
        else:
            fmt='o'
        plt.errorbar(
            sigma_draw[0:data.shape[0]/ntest, i] * test_sersic_highn_basic.PIXEL_SCALE,
            1e5*delta_sigma[0:data.shape[0]/ntest, i] * test_sersic_highn_basic.PIXEL_SCALE,
            yerr=1e5*err_sigma[0:data.shape[0]/ntest, i] * test_sersic_highn_basic.PIXEL_SCALE, fmt=fmt,
            label=r"n = %.1f, m = %.2e $\pm$ %.2e" % (
                test_sersic_highn_basic.SERSIC_N_TEST[i], m, np.sqrt(var_m)))
    plt.ylim(-.3 * YMAX_ZOOMIN, YMAX_ZOOMIN)
    plt.xlabel(r'$\sigma$ (DFT) [arcsec]')
    plt.ylabel(r'$\Delta \sigma$ (DFT - Photon) [1e-5 arcsec]')
    plt.axhline(ls='--', color='k')
    plt.legend()
    #plt.title(test_case)
    plt.savefig(os.path.join('plots', 'sersic_highn_'+test_case+'_zoomin_sigma.png'))

