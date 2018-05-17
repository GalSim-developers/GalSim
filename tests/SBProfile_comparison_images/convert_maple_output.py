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
This program converts the outputs of a Maple program (which cannot write directly to 
a fits file) into images that can be used for the SBProfile tests.
The Maple program is saved in the same directory as moffat_pixel.mw.
And the output that it produces is saved as moffat_pixel.dat.
This program converts that into a fits file called moffat_pixel.fits.
"""

import numpy
from galsim import pyfits
import os

for input_file in [ "moffat_pixel.dat" , "moffat_pixel_distorted.dat" ]:

    output_file = input_file.split('.')[0] + '.fits'
    print input_file, output_file

    nx = 61
    ny = 61

    fin = open(input_file,'r')
    vals = map(float,fin.readlines())

    array = numpy.array(vals).reshape(nx,ny).transpose()

    hdus = pyfits.HDUList()
    hdu = pyfits.PrimaryHDU(array)
    hdus.append(hdu)

    if os.path.isfile(output_file):
        os.remove(output_file)
    hdus.writeto(output_file)
