# Copyright 2012, 2013 The GalSim developers:
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
import numpy as np
import os
import sys
from galsim_test_helpers import *

"""Time the different ways to zip and unzip and image
"""

n_iter = 50

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def time_gunzip():
    """Time different functions for gunzip"""
    import time

    file_name = 'Image_comparison_images/test_multiD.fits.gz'
    sum = 0

    t1 = time.time()
    for iter in range(n_iter):
        hdu_list, fin = galsim.fits._read_file.gzip_in_mem(file_name)
        im = galsim.fits.read(hdu_list = hdu_list)
        sum += im(1,1)
        galsim.fits.closeHDUList(hdu_list,fin)
    t2 = time.time()

    for iter in range(n_iter):
        hdu_list, fin = galsim.fits._read_file.pyfits_open(file_name)
        im = galsim.fits.read(hdu_list = hdu_list)
        sum += im(1,1)
        galsim.fits.closeHDUList(hdu_list,fin)
    t3 = time.time()

    for iter in range(n_iter):
        hdu_list, fin = galsim.fits._read_file.gzip_tmp(file_name)
        im = galsim.fits.read(hdu_list = hdu_list)
        sum += im(1,1)
        galsim.fits.closeHDUList(hdu_list,fin)
    t4 = time.time()

    for iter in range(n_iter):
        hdu_list, fin = galsim.fits._read_file.gunzip_call(file_name)
        im = galsim.fits.read(hdu_list = hdu_list)
        sum += im(1,1)
        galsim.fits.closeHDUList(hdu_list,fin)
    t5 = time.time()

    print 'All times are for %d iterations...'%n_iter
    print 'time for gzip_in_mem = %.2f'%(t2-t1)
    print 'time for pyfits_open = %.2f'%(t3-t2)
    print 'time for gzip_tmp = %.2f'%(t4-t3)
    print 'time for gunzip_call = %.2f'%(t5-t4)
    default_order = [ f.__name__ for f in galsim.fits._read_file.gzip_methods ]
    print 'the default order for your system is ',default_order

def time_bunzip():
    """Time different functions for bunzip2"""
    import time

    file_name = 'Image_comparison_images/test_multiD.fits.bz2'
    sum = 0

    t1 = time.time()
    for iter in range(n_iter):
        hdu_list, fin = galsim.fits._read_file.bz2_in_mem(file_name)
        im = galsim.fits.read(hdu_list = hdu_list)
        sum += im(1,1)
        galsim.fits.closeHDUList(hdu_list,fin)
    t2 = time.time()

    for iter in range(n_iter):
        hdu_list, fin = galsim.fits._read_file.bz2_tmp(file_name)
        im = galsim.fits.read(hdu_list = hdu_list)
        sum += im(1,1)
        galsim.fits.closeHDUList(hdu_list,fin)
        pass
    t3 = time.time()

    for iter in range(n_iter):
        hdu_list, fin = galsim.fits._read_file.bunzip2_call(file_name)
        im = galsim.fits.read(hdu_list = hdu_list)
        sum += im(1,1)
        galsim.fits.closeHDUList(hdu_list,fin)
    t4 = time.time()

    print 'All times are for %d iterations...'%n_iter
    print 'time for bz2_in_mem = %.2f'%(t2-t1)
    print 'time for bz2_tmp = %.2f'%(t3-t2)
    print 'time for bunzip2_call = %.2f'%(t4-t3)
    default_order = [ f.__name__ for f in galsim.fits._read_file.bz2_methods ]
    print 'the default order for your system is ',default_order


if __name__ == "__main__":
    time_gunzip()
    time_bunzip()
