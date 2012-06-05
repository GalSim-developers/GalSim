#!/usr/bin/env python
"""
This program converts the outputs of a Maple program (which cannot write directly to 
a fits file) into images that can be used for the SBProfile tests.
The Maple program is saved in the same directory as moffat_pixel.mw.
And the output that it produces is saved as moffat_pixel.dat.
This program converts that into a fits file called moffat_pixel.fits.
"""

import numpy
import pyfits
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
