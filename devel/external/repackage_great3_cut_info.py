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
The purpose of this script is to take data from assorted GREAT3-related files that store data about
the COSMOS galaxies and their parametric fits, and combine them into a single file that GalSim will
use to make cuts in the COSMOSCatalog class.

Set up to point to data that live in a particular spot, so likely nobody else can actually run this
script.
"""

import numpy as np
import pyfits
import os

data_dir = '/Users/rmandelb/great3/data-23.5'
dmag_file = 'real_galaxy_deltamag_info.fits'
sn_file = 'real_galaxy_image_selection_info.fits'
mask_file = 'real_galaxy_mask_info.fits'
out_file = 'real_galaxy_catalog_23.5_selection.fits'

# Load the appropriate data from each file.
dat = pyfits.getdata(os.path.join(data_dir, dmag_file))
ident = dat['IDENT']
dmag = dat['delta_mag']

dat = pyfits.getdata(os.path.join(data_dir, sn_file))
sn_ellip_gauss = dat['sn_ellip_gauss']

dat = pyfits.getdata(os.path.join(data_dir, mask_file))
min_mask_dist_pixels = dat['min_mask_dist_pixels']
average_mask_adjacent_pixel_count = dat['average_mask_adjacent_pixel_count']
peak_image_pixel_count = dat['peak_image_pixel_count']


# Stick them together into a single FITS table.
tbhdu = pyfits.new_table(pyfits.ColDefs([
            pyfits.Column(name='IDENT',
                          format='J',
                          array=ident),
            pyfits.Column(name='dmag',
                          format='D',
                          array=dmag),
            pyfits.Column(name='sn_ellip_gauss',
                          format='D',
                          array=sn_ellip_gauss),
            pyfits.Column(name='min_mask_dist_pixels',
                          format='D',
                          array=min_mask_dist_pixels),
            pyfits.Column(name='average_mask_adjacent_pixel_count',
                          format='D',
                          array=average_mask_adjacent_pixel_count),
            pyfits.Column(name='peak_image_pixel_count',
                          format='D',
                          array=peak_image_pixel_count)]
                                        ))

# Output to file.
out_file = os.path.join(data_dir, out_file)
print "Writing to file ",out_file
tbhdu.writeto(out_file)
