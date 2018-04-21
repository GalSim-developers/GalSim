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

"""
A script to set up some cuts on intrinsic galaxy radius / flux etc. to eliminate galaxies that would
not be simulated in GREAT3.  Used to make cuts imposed in examples/great3/cgc.yaml.
"""
import pyfits
import matplotlib.pyplot as plt
import numpy as np
import os
import galsim

do_plot = False # Toggle this to choose whether to examine histograms or not
gal_dir = '/Users/rmandelb/great3/data-23.5' # Change this if you aren't Rachel
rgc_sel_file = 'real_galaxy_selection_info.fits' # Selection file from GREAT3, can be downloaded
                                                 # from the GREAT3 data website
rgc_fits_file = 'real_galaxy_catalog_23.5_fits.fits'
# Numbers used in GREAT3 selection files, see great3-public repository if you want to read the
# code.  We use these to check how to eliminate galaxies that will fail cuts a priori, rather than
# after simulating them.
min_ground_fwhm = 0.5 # minimum value of FWHM for which results are tabulated
ground_dfwhm = 0.15 # spacing between tabulated FWHM values
ground_nfwhm = 4 # number of FWHM values for ground
fwhm_arr = min_ground_fwhm + ground_dfwhm*np.arange(ground_nfwhm)
noise_var = [0.0088, 0.0060, 0.0046, 0.0040]
noise_fail_val = 1.e-10
sn_min = 17.0
sn_max = 100.0

# Read in and parse original selection files.
selection_catalog = pyfits.getdata(os.path.join(gal_dir, rgc_sel_file))
resolution_arr = selection_catalog.field('resolution')[:,1:]
noise_max_var = selection_catalog.field('max_var')[:,1:]

# Read in and parse fits files.  Use Sersic fits for this, even if models are going to be drawn with
# 2-component bulge + disk fits, just because radius and n are better-defined with a single galaxy
# model.
fit_catalog = pyfits.getdata(os.path.join(gal_dir, rgc_fits_file))
gal_hlr = fit_catalog.field('hlr')[:,0] # sersic HLR
gal_flux = fit_catalog.field('flux')[:,0] # sersic flux
lg_gal_flux = np.log10(gal_flux)

# Check, for each FWHM value, what are the radius and flux distributions for objects that pass/fail
# the resolution and S/N cuts.   Use these distributions to make cuts.
for ind in range(len(fwhm_arr)):
    fwhm = fwhm_arr[ind]
    print 'Beginning work for FWHM=%.2f arcsec:'%fwhm

    # Get the resolution for this FWHM value.
    res = resolution_arr[:,ind]

    # Get the subsets that pass / fail the resolution>=1/3 cut.
    pass_cuts = res >= 1./3
    fail_cuts = (1-pass_cuts).astype(bool)

    # Find the 5th percentile in half-light radius for galaxies that pass the resolution cut.  Then
    # check that if we cut there, what fraction of the galaxies that fail the resolution cut are
    # eliminated.
    cut_val = np.percentile(gal_hlr[pass_cuts], 5.)
    elim_frac = float(np.sum(gal_hlr[fail_cuts] < cut_val))/len(gal_hlr[fail_cuts])
    print '    Radius cut at %.3f arcsec eliminates a fraction %f of res failures'%(cut_val,elim_frac)

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n, bins, patches = ax.hist(gal_hlr[pass_cuts], np.linspace(0.,1.,21), facecolor='green',
                                   alpha=0.75)
        n, bins, patches = ax.hist(gal_hlr[fail_cuts], np.linspace(0.,1.,21), facecolor='red',
                                   alpha=0.75)
        ax.set_xlabel('Half-light radius')
        ax.set_title('FWHM: %.2f arcsec'%fwhm)
        plt.show()

    # Estimate the S/N for each galaxy in this seeing, given the noise variance that will be
    # adopted.
    nmv = noise_max_var[:,ind]
    approx_sn_gal = np.zeros_like(nmv)
    approx_sn_gal[nmv > noise_fail_val] = 20.0*np.sqrt(nmv[nmv > noise_fail_val] / noise_var[ind])
    # Find those that fail the S/N cut at either the high or low end.
    pass_cuts_flux = np.logical_and.reduce([
            approx_sn_gal >= sn_min,
            approx_sn_gal <= sn_max])
    fail_cuts_flux = (1-pass_cuts_flux).astype(bool)
    fail_cuts_highflux = approx_sn_gal >= sn_max
    # The distributions overlap at low flux and there is no clear cut to impose.  But at high flux,
    # there is a clear tail for the failures at S/N>100.  So let's target those, starting at the
    # 95th percentile in galaxies that pass the S/N cuts.
    cut_val = np.percentile(gal_flux[pass_cuts_flux], 95.)
    # Check what fraction of the failing galaxies are eliminated.
    elim_frac = float(np.sum(gal_flux[fail_cuts_flux] > cut_val))/len(gal_flux[fail_cuts_flux])
    # Check what fraction of the failing ones AT HIGH S/N are eliminated.  Should be a lot more!
    elim_frac_high = float(np.sum(gal_flux[fail_cuts_flux] > cut_val))/\
        len(gal_flux[fail_cuts_highflux])
    print '    Flux cut at %.3f eliminates a fraction %f of SNR failures'%(cut_val,elim_frac)
    print '    ...or, a fraction %f of SNR failures at the high end'%elim_frac_high

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n, bins, patches = ax.hist(lg_gal_flux[pass_cuts_flux], np.linspace(-1.,3.,21),
                                   facecolor='green', alpha=0.75)
        n, bins, patches = ax.hist(lg_gal_flux[fail_cuts_flux], np.linspace(-1.,3.,21),
                                   facecolor='red', alpha=0.75)
        ax.set_xlabel('Log Flux')
        ax.set_title('FWHM: %.2f arcsec'%fwhm)
        plt.show()

    if do_plot:
        if False:
            # Just confirm that the previous results were not strongly Sersic n-dependent
            fig = plt.figure()
            H, xedges, yedges = np.histogram2d(gal_hlr[pass_cuts],fit_gal_n[pass_cuts],
                                               bins=20,range=((0,1),(0,6)))
            H = np.rot90(H)
            H = np.flipud(H)
            Hm = np.ma.masked_where(H<=2, H)
            plt.pcolormesh(xedges, yedges, Hm)
            plt.xlabel('Half-light radius')
            plt.ylabel('Sersic n')
            plt.title('Pass cuts for FWHM %.2f arcsec'%fwhm)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Counts')
            plt.show()
            
            fig = plt.figure()
            H, xedges, yedges = np.histogram2d(gal_hlr[fail_cuts],fit_gal_n[fail_cuts],
                                               bins=20,range=((0,1),(0,6)))
            H = np.rot90(H)
            H = np.flipud(H)
            Hm = np.ma.masked_where(H<=2, H)
            plt.pcolormesh(xedges, yedges, Hm)
            plt.xlabel('Half-light radius')
            plt.ylabel('Sersic n')
            plt.title('Fail cuts for FWHM %.2f arcsec'%fwhm)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Counts')
            plt.show()

