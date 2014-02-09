# Copyright 2012-2014 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
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
