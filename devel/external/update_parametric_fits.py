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
In GalSim v1.3, the COSMOSCatalog() class defined in scene.py has to do many calculations (some
expensive) to convert the stored information about parametric galaxy fits into GSObjects.  For
GalSim v1.4, we would like to avoid the need to do those calculations each time, so this script is
meant to precompute all those quantities and store them in the file with the parametric fits.  This
way they can simply be read in.

Set up to point to data that live in a particular spot, so likely nobody else can actually run this
script.
"""

import numpy as np
import pyfits
import os
import shutil
import galsim
import matplotlib.pyplot as plt

cosmos_pix_scale = 0.03
data_dir = '/Users/rmandelb/great3/data-23.5'
tmp_file = 'tmp_old_catalog.fits' # move the old catalog to this location before overwriting
param_fit_file = 'real_galaxy_catalog_23.5_fits.fits' # this is the file that will get replaced

# First copy the original file just in case something gets messed up.
shutil.copy(os.path.join(data_dir, param_fit_file),
            os.path.join(data_dir, tmp_file))

# Load the appropriate data from the parametric fit file.
dat = pyfits.getdata(os.path.join(data_dir, param_fit_file))
# Get fit parameters.  For 'sersicfit', the result is an array of 8 numbers for each galaxy:
#
#     SERSICFIT[0]: intensity of light profile at the half-light radius.
#     SERSICFIT[1]: half-light radius measured along the major axis, in units of pixels
#                   in the COSMOS lensing data reductions (0.03 arcsec).
#     SERSICFIT[2]: Sersic n.
#     SERSICFIT[3]: q, the ratio of minor axis to major axis length.
#     SERSICFIT[4]: boxiness, currently fixed to 0, meaning isophotes are all
#                   elliptical.
#     SERSICFIT[5]: x0, the central x position in pixels.
#     SERSICFIT[6]: y0, the central y position in pixels.
#     SERSICFIT[7]: phi, the position angle in radians.  If phi=0, the major axis is
#                   lined up with the x axis of the image.
# For 'bulgefit', the result is an array of 16 parameters that comes from doing a 2-component sersic
# fit.  The first 8 are the parameters for the disk, with n=1, and the last 8 are for the bulge,
# with n=4.
bparams = dat['bulgefit']
sparams = dat['sersicfit']
# Get the status flag for the fits.  Entries 0 and 4 in 'fit_status' are relevant for bulgefit and
# sersicfit, respectively.
bstat = dat['fit_status'][:,0]
sstat = dat['fit_status'][:,4]
# Get the precomputed bulge-to-total flux ratio for the 2-component fits.
dvc_btt = dat['fit_dvc_btt']
# Get the precomputed median absolute deviation for the 1- and 2-component fits.  These quantities
# are used to ascertain whether the 2-component fit is really justified, or if the 1-component
# Sersic fit is sufficient to describe the galaxy light profile.
bmad = dat['fit_mad_b']
smad = dat['fit_mad_s']

# To compute: use_bulgefit (which says whether to use the Sersic or the bulge+disk model) and
# failure flag.
failure = np.zeros(len(dat)).astype(bool)
use_bulgefit = np.zeros(len(dat)).astype(bool)
viable_sersic = np.zeros(len(dat)).astype(bool)
find_double = np.logical_and.reduce([
        bstat >= 1, 
        bstat <= 4,
        dvc_btt >= 0.1,
        dvc_btt <= 0.9,
        bparams[:,9] > 0,
        bparams[:,1] > 0,
        bparams[:,11] >= 0.051,
        bparams[:,3] >= 0.051,
        smad >= bmad])
use_bulgefit[find_double] = True

# Then check if sersicfit is viable; if not, this galaxy is a total failure.  Note that we can avoid
# including these in the catalog in the first place by using `exclusion_level=bad_fits` or
# `exclusion_level=marginal` when making the COSMOSCatalog in GalSim v1.4.
viable_sersic_flag = np.logical_and.reduce([
        sstat >= 1,
        sstat <= 4,
        sparams[:,1] > 0,
        sparams[:,0] > 0])
viable_sersic[viable_sersic_flag] = True

find_failure_flag = np.logical_and.reduce([
        viable_sersic is False,
        use_bulgefit is False])
failure[find_failure_flag] = True

# To compute: Sersic, bulge, disk hlr.
# To compute: Sersic, bulge, disk, total flux.
hlr = np.zeros((len(dat), 3))
flux = np.zeros((len(dat), 4))
# Sersic first:
gal_n = sparams[viable_sersic_flag,2]
# Fudge this if it is at the edge of the allowed n values.  Since GalSim (as of #325 and #449) allow
# Sersic n in the range 0.3<=n<=6, the only problem is that the fits occasionally go as low as
# n=0.2.  The fits in this file only go to n=6, so there is no issue with too-high values, but we
# also put a guard on that side in case other samples are swapped in that go to higher value of
# sersic n.
gal_n[gal_n < 0.3] = 0.3
gal_n[gal_n > 6.0] = 6.0
# GalSim is much more efficient if only a finite number of Sersic n values are used.  This
# (optionally given constructor args) rounds n to the nearest 0.05.
use_hlr = cosmos_pix_scale*np.sqrt(sparams[viable_sersic_flag,3])*sparams[viable_sersic_flag,1]
hlr[viable_sersic_flag,0] = use_hlr
# Below is the calculation of the full Sersic n-dependent quantity that goes into the conversion
# from surface brightness to flux, which here we're calling 'prefactor'
use_flux = np.zeros_like(use_hlr)
for ind in range(len(gal_n)):
    if ind % 1000 == 0:
        print 'Calculations: %d...'%ind
    tmp_ser = galsim.Sersic(gal_n[ind],
                            half_light_radius=use_hlr[ind])
    use_flux[ind] = sparams[viable_sersic_flag,0][ind] / \
        tmp_ser.xValue(0,use_hlr[ind]) / cosmos_pix_scale**2
flux[viable_sersic_flag,0] = use_flux        
# Then bulge, disk (same considerations about units etc. apply):
hlr[find_double,1] = cosmos_pix_scale*np.sqrt(bparams[find_double,11])*bparams[find_double,9]
flux[find_double,1] = \
    2.0*np.pi*3.607*(hlr[find_double,1]**2)*bparams[find_double,8]/cosmos_pix_scale**2
hlr[find_double,2] = cosmos_pix_scale*np.sqrt(bparams[find_double,3])*bparams[find_double,1]
flux[find_double,2] = \
    2.0*np.pi*1.901*(hlr[find_double,2]**2)*bparams[find_double,0]/cosmos_pix_scale**2
flux[find_double,3] = flux[find_double,1] + flux[find_double,2]

# Make useful diagnostic comments and plots
print 'Total number in catalog:',len(use_bulgefit)
print 'Number for which we are going to use 2-component fits:',np.sum(use_bulgefit.astype(int))
print 'Number with viable Sersic fits (even if not using):',np.sum(viable_sersic.astype(int))
print 'Number of failures: cannot use either type of fit:',np.sum(failure.astype(int))
# Compare the flux from the two types of fits, for cases that have both.
to_plot = np.logical_and.reduce([
        use_bulgefit,
        viable_sersic])
log_flux_bf = np.log10(flux[to_plot,3])
log_flux_s = np.log10(flux[to_plot,0])

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(log_flux_s, log_flux_bf, 'bo', linestyle='None')
    plt.xlabel('Log sersic flux')
    plt.ylabel('Log bulge+disk flux')
    plt.xlim((0.,4.))
    plt.ylim((0.,4.))
    plt.show()
    # Plot radius vs. flux for Sersic fits
    log_flux = np.log10(flux[viable_sersic,0])
    log_rad = np.log10(hlr[viable_sersic,0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(log_flux, log_rad, 'bo', linestyle='None')
    plt.xlabel('Log sersic flux')
    plt.ylabel('Log sersic radius')
    plt.xlim((-1.,4.))
    plt.ylim((-2.5,1.))
    plt.show()

# Stick them together into a single FITS table.  First just make a table with the new stuff.
tbhdu = pyfits.new_table(pyfits.ColDefs([
            pyfits.Column(name='use_bulgefit',
                          format='J',
                          array=use_bulgefit),
            pyfits.Column(name='viable_sersic',
                          format='J',
                          array=viable_sersic),
            pyfits.Column(name='hlr',
                          format='3D',
                          array=hlr),
            pyfits.Column(name='flux',
                          format='4D',
                          array=flux)]
                                        ))
# Then merge them.
new_table = dat.columns + tbhdu.columns
hdu = pyfits.new_table(new_table)

# Output to file.
out_file = os.path.join(data_dir, param_fit_file)
print "Writing to file ",out_file
hdu.writeto(out_file,clobber=True)
