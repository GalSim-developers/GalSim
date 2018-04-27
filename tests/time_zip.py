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

"""Time the different ways to zip and unzip and image
"""

from __future__ import print_function
import numpy as np
import os
import sys
from galsim_test_helpers import *
from galsim import pyfits

n_iter = 20

import galsim

big_im = galsim.Image(5000, 5000)
big_im_file = 'big_im_file.fits'
big_im_file_gz = 'big_im_file1.fits.gz'
big_im_file_bz2 = 'big_im_file2.fits.bz2'

medium_im = galsim.Image(1000, 1000)
medium_im_file = 'medium_im_file.fits'
medium_im_file_gz = 'medium_im_file1.fits.gz'
medium_im_file_bz2 = 'medium_im_file2.fits.bz2'

small_im = galsim.Image(200, 200)
small_im_file = 'small_im_file.fits'
small_im_file_gz = 'small_im_file1.fits.gz'
small_im_file_bz2 = 'small_im_file2.fits.bz2'

dir = 'Image_comparison_images'

big_im.addNoise(galsim.GaussianNoise(sigma=20.))
big_im.write(os.path.join(dir,big_im_file))
medium_im.addNoise(galsim.GaussianNoise(sigma=20.))
medium_im.write(os.path.join(dir,medium_im_file))
small_im.addNoise(galsim.GaussianNoise(sigma=20.))
small_im.write(os.path.join(dir,small_im_file))

def time_gzip():
    """Time different functions for gzip"""
    import time

    for file, gzfile, size, n_iter in [ (big_im_file, big_im_file_gz, 5000, 1),
                                        (medium_im_file, medium_im_file_gz, 1000, 20),
                                        (small_im_file, small_im_file_gz, 200, 400) ]:

        infile_name = os.path.join(dir,file)
        hdu_list = pyfits.open(infile_name)

        outfile_name = os.path.join(dir,gzfile)
        t1 = time.time()
        try:
            for iter in range(n_iter):
                if os.path.isfile(outfile_name):
                    os.remove(outfile_name)
                galsim.fits._write_file.gzip_in_mem(hdu_list, outfile_name)
        except:
            # Only report error the first time.
            if n_iter == 1:
                import traceback
                print('gzip_in_mem failed with exception:')
                traceback.print_exc()
        t2 = time.time()

        try:
            for iter in range(n_iter):
                if os.path.isfile(outfile_name):
                    os.remove(outfile_name)
                galsim.fits._write_file.gzip_tmp(hdu_list, outfile_name)
        except:
            if n_iter == 1:
                import traceback
                print('gzip_tmp failed with exception:')
                traceback.print_exc()
        t3 = time.time()

        try:
            for iter in range(n_iter):
                if os.path.isfile(outfile_name):
                    os.remove(outfile_name)
                galsim.fits._write_file.gzip_call(hdu_list, outfile_name)
        except:
            if n_iter == 1:
                import traceback
                print('gzip_call failed with exception:')
                traceback.print_exc()
        t4 = time.time()

        try:
            for iter in range(n_iter):
                if os.path.isfile(outfile_name):
                    os.remove(outfile_name)
                galsim.fits._write_file.gzip_call2(hdu_list, outfile_name)
        except:
            if n_iter == 1:
                import traceback
                print('gzip_call2 failed with exception:')
                traceback.print_exc()
        t5 = time.time()

        print('Times for %d iterations of writing to %s (%d x %d): '%(n_iter, gzfile, size, size))
        print('   time for gzip_in_mem = %.2f'%(t2-t1))
        print('   time for gzip_tmp = %.2f'%(t3-t2))
        print('   time for gzip_call = %.2f'%(t4-t3))
        print('   time for gzip_call2 = %.2f'%(t5-t4))
        hdu_list.close()

    default_order = [ f.__name__ for f in galsim.fits._write_file.gz_methods ]
    print('The current default order for gzip write is ',default_order)
    print()

def time_bzip2():
    """Time different functions for bzip2"""
    import time

    for file, bz2file, size, n_iter in [ (big_im_file, big_im_file_bz2, 5000, 1),
                                         (medium_im_file, medium_im_file_bz2, 1000, 20),
                                         (small_im_file, small_im_file_bz2, 200, 400) ]:

        infile_name = os.path.join(dir,file)
        hdu_list = pyfits.open(infile_name)

        outfile_name = os.path.join(dir,bz2file)
        t1 = time.time()
        try:
            for iter in range(n_iter):
                if os.path.isfile(outfile_name):
                    os.remove(outfile_name)
                galsim.fits._write_file.bz2_in_mem(hdu_list, outfile_name)
        except:
            if n_iter == 1:
                import traceback
                print('bz2_in_mem failed with exception:')
                traceback.print_exc()
        t2 = time.time()

        try:
            for iter in range(n_iter):
                if os.path.isfile(outfile_name):
                    os.remove(outfile_name)
                galsim.fits._write_file.bz2_tmp(hdu_list, outfile_name)
        except:
            if n_iter == 1:
                import traceback
                print('bz2_tmp failed with exception:')
                traceback.print_exc()
        t3 = time.time()

        try:
            for iter in range(n_iter):
                if os.path.isfile(outfile_name):
                    os.remove(outfile_name)
                galsim.fits._write_file.bzip2_call(hdu_list, outfile_name)
        except:
            if n_iter == 1:
                import traceback
                print('bzip2_call failed with exception:')
                traceback.print_exc()
        t4 = time.time()

        try:
            for iter in range(n_iter):
                if os.path.isfile(outfile_name):
                    os.remove(outfile_name)
                galsim.fits._write_file.bzip2_call2(hdu_list, outfile_name)
        except:
            if n_iter == 1:
                import traceback
                print('bzip2_call2 failed with exception:')
                traceback.print_exc()
        t5 = time.time()

        print('Times for %d iterations of writing to %s (%d x %d): '%(n_iter, bz2file, size, size))
        print('   time for bz2_in_mem = %.2f'%(t2-t1))
        print('   time for bz2_tmp = %.2f'%(t3-t2))
        print('   time for bzip2_call = %.2f'%(t4-t3))
        print('   time for bzip2_call2 = %.2f'%(t5-t4))
        hdu_list.close()

    default_order = [ f.__name__ for f in galsim.fits._write_file.bz2_methods ]
    print('The current default order for bzip2 write is ',default_order)
    print()

def time_gunzip():
    """Time different functions for gunzip"""
    import time

    for gzfile, size, n_iter in [ (big_im_file_gz, 5000, 1),
                                  (medium_im_file_gz, 1000, 20),
                                  (small_im_file_gz, 200, 400) ]:

        file_name = os.path.join(dir,gzfile)

        t1 = time.time()
        try:
            for iter in range(n_iter):
                hdu_list, fin = galsim.fits._read_file.gzip_in_mem(file_name)
                im = galsim.fits.read(hdu_list = hdu_list)
                fin.close()
                hdu_list.close()
        except:
            if n_iter == 1:
                import traceback
                print('gzip_in_mem failed with exception:')
                traceback.print_exc()
        t2 = time.time()

        try:
            for iter in range(n_iter):
                hdu_list, fin = galsim.fits._read_file.gzip_tmp(file_name)
                im = galsim.fits.read(hdu_list = hdu_list)
                os.remove(fin)
                hdu_list.close()
        except:
            if n_iter == 1:
                import traceback
                print('gzip_tmp failed with exception:')
                traceback.print_exc()
        t3 = time.time()

        try:
            for iter in range(n_iter):
                hdu_list, fin = galsim.fits._read_file.gunzip_call(file_name)
                im = galsim.fits.read(hdu_list = hdu_list)
                fin.close()
                hdu_list.close()
        except:
            if n_iter == 1:
                import traceback
                print('gunzip_call failed with exception:')
                traceback.print_exc()
        t4 = time.time()

        try:
            for iter in range(n_iter):
                hdu_list, fin = galsim.fits._read_file.pyfits_open(file_name)
                im = galsim.fits.read(hdu_list = hdu_list)
                hdu_list.close()
        except:
            if n_iter == 1:
                import traceback
                print('pyfits_open failed with exception:')
                traceback.print_exc()
        t5 = time.time()

        print('Times for %d iterations of reading %s (%d x %d): '%(n_iter, gzfile, size, size))
        print('   time for gzip_in_mem = %.2f'%(t2-t1))
        print('   time for gzip_tmp = %.2f'%(t3-t2))
        print('   time for gunzip_call = %.2f'%(t4-t3))
        print('   time for pyfits_open = %.2f'%(t5-t4))

    default_order = [ f.__name__ for f in galsim.fits._read_file.gz_methods ]
    print('The current default order for gzip read is ',default_order)
    print()


def time_bunzip2():
    """Time different functions for bunzip2"""
    import time

    for bz2file, size, n_iter in [ (big_im_file_bz2, 5000, 1),
                                   (medium_im_file_bz2, 1000, 20),
                                   (small_im_file_bz2, 200, 400) ]:

        file_name = os.path.join(dir,bz2file)

        t1 = time.time()
        try:
            for iter in range(n_iter):
                hdu_list, fin = galsim.fits._read_file.bz2_in_mem(file_name)
                im = galsim.fits.read(hdu_list = hdu_list)
                fin.close()
                hdu_list.close()
        except:
            if n_iter == 1:
                import traceback
                print('bz2_in_mem failed with exception:')
                traceback.print_exc()
        t2 = time.time()

        try:
            for iter in range(n_iter):
                hdu_list, fin = galsim.fits._read_file.bz2_tmp(file_name)
                im = galsim.fits.read(hdu_list = hdu_list)
                os.remove(fin)
                hdu_list.close()
        except:
            if n_iter == 1:
                import traceback
                print('bz2_tmp failed with exception:')
                traceback.print_exc()
        t3 = time.time()

        try:
            for iter in range(n_iter):
                hdu_list, fin = galsim.fits._read_file.bunzip2_call(file_name)
                im = galsim.fits.read(hdu_list = hdu_list)
                fin.close()
                hdu_list.close()
        except:
            if n_iter == 1:
                import traceback
                print('bunzip_call failed with exception:')
                traceback.print_exc()
        t4 = time.time()

        print('Times for %d iterations of reading %s (%d x %d): '%(n_iter, bz2file, size, size))
        print('   time for bz2_in_mem = %.2f'%(t2-t1))
        print('   time for bz2_tmp = %.2f'%(t3-t2))
        print('   time for bunzip2_call = %.2f'%(t4-t3))

    default_order = [ f.__name__ for f in galsim.fits._read_file.bz2_methods ]
    print('The current default order for bzip2 read is ',default_order)
    print()

if __name__ == "__main__":
    time_gzip()
    time_bzip2()
    time_gunzip()
    time_bunzip2()
