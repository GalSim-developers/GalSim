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
This script updates the subcatalogs for the RealGalaxyCatalog, and writes them to files.
Meant for use for unit tests.
"""
import pyfits
import numpy as np

fitfile = '/Users/rmandelb/great3/data-23.5/real_galaxy_catalog_23.5_fits.fits'
realfile = '/Users/rmandelb/great3/data-23.5/real_galaxy_catalog_23.5.fits'
subfitfile = '../../examples/data/real_galaxy_catalog_example_fits.fits'
subrealfile = '../../examples/data/real_galaxy_catalog_example.fits'
update_fits_cat = False # update parametric fits catalog?
update_real_cat = True # update real catalog?

subrealdat = pyfits.getdata(subrealfile)
realdat = pyfits.getdata(realfile)
fitdat = pyfits.getdata(fitfile)

if update_fits_cat:
    # need to find the stuff to keep
    match_ind = np.zeros(len(subrealdat)).astype(int)
    for ind in range(len(subrealdat)):
        match_ind[ind] = list(fitdat['ident']).index(subrealdat['ident'][ind])

    new_dat = []
    for ind in range(len(fitdat.columns)):
        new_dat.append(pyfits.Column(name=fitdat.columns[ind].name,
                                     format=fitdat.columns[ind].format,
                                     array=fitdat[fitdat.columns[ind].name][match_ind]))
    new_hdu = pyfits.new_table(pyfits.ColDefs(new_dat))
    new_hdu.writeto(subfitfile,clobber=True)

if update_real_cat:
    # need to find the stuff to keep
    match_ind = np.zeros(len(subrealdat)).astype(int)
    for ind in range(len(subrealdat)):
        match_ind[ind] = list(realdat['ident']).index(subrealdat['ident'][ind])

    new_dat = []
    for ind in range(len(realdat.columns)):
        # Only replace if we're not changing the filename / HDU.
        if realdat.columns[ind].name not in \
                ['GAL_FILENAME', 'GAL_HDU', 'PSF_FILENAME', 'PSF_HDU', 'NOISE_FILENAME']:
            new_dat.append(pyfits.Column(name=realdat.columns[ind].name,
                                         format=realdat.columns[ind].format,
                                         array=realdat[realdat.columns[ind].name][match_ind]))
        else:
            # Otherwise keep the old one.
            new_dat.append(pyfits.Column(name=realdat.columns[ind].name,
                                         format=realdat.columns[ind].format,
                                         array=subrealdat[realdat.columns[ind].name]))
    new_hdu = pyfits.new_table(pyfits.ColDefs(new_dat))
    new_hdu.writeto(subrealfile,clobber=True)
