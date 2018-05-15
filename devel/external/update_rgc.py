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
In GalSim v1.4, we want to be able to pre-select (eliminate) the very small number of postage stamps
that have a negative total flux level due to sky subtraction or deblending issues.  This script
precomputes the total flux and stores it in the RealGalaxyCatalog file.

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
data_dirs = ['/Users/rmandelb/great3/data-23.5',
             '/Users/rmandelb/great3/COSMOS_25.2_training_sample']
# this is the file that will get replaced
files = ['real_galaxy_catalog_23.5.fits', 'real_galaxy_catalog_25.2.fits']
tmp_file = 'tmp_old_catalog.fits' # move the old catalog to this location before overwriting
n = len(files)

for ind in range(n):
    file = files[ind]
    data_dir = data_dirs[ind]
    print data_dir, file

    # First copy the original file just in case something gets messed up.
    shutil.copy(os.path.join(data_dir, file),
                os.path.join(data_dir, tmp_file))
    print 'Copied to ',os.path.join(data_dir, tmp_file)

    # Load the appropriate data from the parametric fit file.
    dat = pyfits.getdata(os.path.join(data_dir, file))

    # And set up the COSMOSCatalog
    ccat = galsim.COSMOSCatalog(file, dir=data_dir, exclusion_level='none')
    print 'Read in ',ccat.nobjects,' objects'

    # Compute the flux in the postage stamp
    stamp_flux = []
    for ind_n in range(ccat.nobjects):
        if ind_n % 1000 == 0:
            print '  Working on galaxy %d.'%ind_n
        gal = ccat.makeGalaxy(index=ind_n, gal_type='real')
        stamp_flux.append(gal.gal_image.array.sum())
    stamp_flux = np.array(stamp_flux)
    print len(stamp_flux[stamp_flux<0]),' negative flux stamps'

    # Make a tbhdu for this data.
    tbhdu = pyfits.new_table(pyfits.ColDefs([
                pyfits.Column(name='stamp_flux',
                              format='F',
                              array=stamp_flux)]
                                        ))
    # Then merge with original table.
    new_table = dat.columns + tbhdu.columns
    hdu = pyfits.new_table(new_table)

    # Output to file.
    out_file = os.path.join(data_dir, file)
    print "Writing to file ",out_file
    hdu.writeto(out_file,clobber=True)
    print ''
