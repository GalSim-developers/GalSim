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
import numpy as np
import matplotlib.pyplot as plt
import test_sersic_highn_basic
import test_gaussian_basic

# Now try a trick... import the sersic plotting script (this will run it, but it's quick) and
# use the Sersic n=.5 results from there directly
import plot_sersic_highn

print "Plotting the Gaussian results"
nobs = test_sersic_highn_basic.NOBS

outfile = os.path.join("outputs", "gaussian_basic_output_N"+str(nobs)+".asc")
data = np.loadtxt(outfile)

g1obs_draw = data[:, 0]
g2obs_draw = data[:, 1]
sigma_draw = data[:, 2]
delta_g1obs = data[:, 3]
delta_g2obs = data[:, 4]
delta_sigma = data[:, 5]
err_g1obs = data[:, 6]
err_g2obs = data[:, 7]
err_sigma = data[:, 8]

# First do the plot of g1
YMAX_ZOOMIN = 2.5e-4
XMIN = -.6
XMAX = .8
plt.clf()
plt.axhline(ls='--', color='k')
plt.axvline(ls='--', color='k')
plt.xlim(XMIN, XMAX)
plt.errorbar(g1obs_draw, delta_g1obs, yerr=err_g1obs, fmt='x', label="Gaussian")
plt.xlabel(r'g$_1$ (DFT)')
plt.ylabel(r'$\Delta$g$_1$ (DFT - Photon)')
plt.ylim(-YMAX_ZOOMIN, YMAX_ZOOMIN)
plt.title("Gaussian comparison")
plt.errorbar(
    plot_sersic_highn.g1obs_draw[:, 0], plot_sersic_highn.delta_g1obs[:, 0], # First column is n=.5
    yerr=plot_sersic_highn.err_g1obs[:, 0], fmt='x',
    label="n = "+str(test_sersic_highn_basic.SERSIC_N_TEST[0])+" ("+str(
        plot_sersic_highn.test_case)+")")

plt.legend()
plt.subplots_adjust(left=0.15)
plt.savefig(os.path.join('plots', 'gaussian_zoomin_g1.png'))
# Then do the plot of g2
YMAX_ZOOMIN = 2.5e-4
XMIN = -.6
XMAX = .8
plt.clf()
plt.axhline(ls='--', color='k')
plt.axvline(ls='--', color='k')
plt.xlim(XMIN, XMAX)
plt.errorbar(g2obs_draw, delta_g2obs, yerr=err_g2obs, fmt='x', label="Gaussian")
plt.xlabel(r'g$_2$ (DFT)')
plt.ylabel(r'$\Delta$g$_2$ (DFT - Photon)')
plt.ylim(-YMAX_ZOOMIN, YMAX_ZOOMIN)
plt.title("Gaussian comparison")
plt.errorbar(
    plot_sersic_highn.g2obs_draw[:, 0], plot_sersic_highn.delta_g2obs[:, 0], # First column is n=.5
    yerr=plot_sersic_highn.err_g2obs[:, 0], fmt='x',
    label="n = "+str(test_sersic_highn_basic.SERSIC_N_TEST[0])+" ("+str(
        plot_sersic_highn.test_case)+")")

plt.legend()
plt.subplots_adjust(left=0.15)
plt.savefig(os.path.join('plots', 'gaussian_zoomin_g2.png'))

# Then plot comparisons of the Gaussian DFT versus n=0.5 DFT and photons shooting results
YMAX_ZOOMIN = 2.5e-4
XMIN = -.8
XMAX = .8
plt.clf()
plt.axhline(ls='--', color='k')
plt.axvline(ls='--', color='k')
plt.xlim(XMIN, XMAX)
plt.plot(g1obs_draw, plot_sersic_highn.g1obs_draw[:, 0] - g1obs_draw, '+',
         label=r"Sersic n=0.5 via DFT g$_1$")
plt.plot(g2obs_draw, plot_sersic_highn.g2obs_draw[:, 0] - g2obs_draw, 'x',
         label=r"Sersic n=0.5 via DFT g$_2$")
plt.errorbar(
    g1obs_draw,
    plot_sersic_highn.g1obs_draw[:, 0] - plot_sersic_highn.delta_g1obs[:, 0] - g1obs_draw,
    yerr=err_g1obs, fmt='+', label=r"Sersic n=0.5 via Photon Shooting g$_1$")
plt.errorbar(
    g2obs_draw,
    plot_sersic_highn.g2obs_draw[:, 0] - plot_sersic_highn.delta_g2obs[:, 0] - g2obs_draw,
    yerr=err_g2obs, fmt='x', label=r"Sersic n=0.5 via Photon Shooting g$_2$")
plt.xlabel(r'g$_i$ (Gaussian via DFT)')
plt.ylabel(r'g$_i$ (Sersic n=0.5) - g$_i$ (Gaussian)')
plt.ylim(-YMAX_ZOOMIN, YMAX_ZOOMIN)
plt.title("Gaussian vs Sersic n=0.5 comparison")
plt.legend()
plt.subplots_adjust(left=0.15)
plt.savefig(os.path.join('plots', 'gaussian_vs_Sersic.png'))



