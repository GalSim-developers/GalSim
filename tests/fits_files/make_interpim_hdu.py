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
"""This script makes some multi-extension FITS files with images to be used for tests of making
interpolated images using the 'hdu' keyword."""

import galsim

sigma = 1.
pix_scale = 0.2
g2_vals = [0., 0.1, 0.7, 0.3]
outfile = 'interpim_hdu_test.fits'
koutfile = 'interpkim_hdu_test.fits'

obj = galsim.Gaussian(sigma=sigma)

im_list = []
kim_list = []
for g2 in g2_vals:
    gal = obj.shear(g2=g2)
    im = gal.drawImage(scale=pix_scale, method='no_pixel')
    print 'For shear ',g2,':',im.FindAdaptiveMom()
    im_list.append(im)
    kim = gal.drawKImage(scale=0.2)
    kim_list.append(kim.real)
    kim_list.append(kim.imag)

galsim.fits.writeMulti(im_list, outfile)
galsim.fits.writeMulti(kim_list, koutfile)



