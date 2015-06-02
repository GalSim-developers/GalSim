# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
import numpy as np
import os
import sys

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
import galsim.utilities


testshape = (512, 512)  # shape of image arrays for all tests
decimal = 6     # Last decimal place used for checking equality of float arrays, see
                # np.testing.assert_array_almost_equal(), low since many are ImageF

def test_roll2d_circularity():
    """Test both integer and float arrays are unchanged by full circular roll.
    """
    import time
    t1 = time.time()
    # Make heterogenous 2D array, integers first, test that a full roll gives the same as the inputs
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.utilities.roll2d(int_image, int_image.shape),
                                  err_msg='galsim.utilities.roll2D failed int array circularity')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.utilities.roll2d(flt_image, flt_image.shape),
                                  err_msg='galsim.utilities.roll2D failed flt array circularity')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_roll2d_fwdbck():
    """Test both integer and float arrays are unchanged by unit forward and backward roll.
    """
    import time
    t1 = time.time()
    # Make heterogenous 2D array, integers first, test that a +1, -1 roll gives the same as initial
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.utilities.roll2d(galsim.utilities.roll2d(int_image,
                                                                                  (+1, +1)),
                                                          (-1, -1)),
                                  err_msg='galsim.utilities.roll2D failed int array fwd/back test')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.utilities.roll2d(galsim.utilities.roll2d(flt_image,
                                                                                  (+1, +1)),
                                                          (-1, -1)),
                                  err_msg='galsim.utilities.roll2D failed flt array fwd/back test')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_roll2d_join():
    """Test both integer and float arrays are equivalent if rolling +1/-1 or -/+(shape[i/j] - 1).
    """
    import time
    t1 = time.time()
    # Make heterogenous 2D array, integers first
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(galsim.utilities.roll2d(int_image, (+1, -1)),
                                  galsim.utilities.roll2d(int_image, (-(int_image.shape[0] - 1),
                                                                   +(int_image.shape[1] - 1))),
                                  err_msg='galsim.utilities.roll2D failed int array +/- join test')
    np.testing.assert_array_equal(galsim.utilities.roll2d(int_image, (-1, +1)),
                                  galsim.utilities.roll2d(int_image, (+(int_image.shape[0] - 1),
                                                                   -(int_image.shape[1] - 1))),
                                  err_msg='galsim.utilities.roll2D failed int array -/+ join test')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(galsim.utilities.roll2d(flt_image, (+1, -1)),
                                  galsim.utilities.roll2d(flt_image, (-(flt_image.shape[0] - 1),
                                                                   +(flt_image.shape[1] - 1))),
                                  err_msg='galsim.utilities.roll2D failed flt array +/- join test')
    np.testing.assert_array_equal(galsim.utilities.roll2d(flt_image, (-1, +1)),
                                  galsim.utilities.roll2d(flt_image, (+(flt_image.shape[0] - 1),
                                                                   -(flt_image.shape[1] - 1))),
                                  err_msg='galsim.utilities.roll2D failed flt array -/+ join test')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_kxky():
    """Test that the basic properties of kx and ky are right.
    """
    import time
    t1 = time.time()
    kx, ky = galsim.utilities.kxky((4, 4))
    kxref = np.array([0., 0.25, -0.5, -0.25]) * 2. * np.pi
    kyref = np.array([0., 0.25, -0.5, -0.25]) * 2. * np.pi
    for i in xrange(4):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in xrange(4):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_kxky_plusone():
    """Test that the basic properties of kx and ky are right...
    But increment testshape used in test_kxky by one to test both odd and even cases.
    """
    import time
    t1 = time.time()
    kx, ky = galsim.utilities.kxky((4 + 1, 4 + 1))
    kxref = np.array([0., 0.2, 0.4, -0.4, -0.2]) * 2. * np.pi
    kyref = np.array([0., 0.2, 0.4, -0.4, -0.2]) * 2. * np.pi
    for i in xrange(4 + 1):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in xrange(4 + 1):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_check_all_contiguous():
    """Test all galsim.optics outputs are C-contiguous as required by the galsim.Image class.
    """
    import time
    t1 = time.time()
    #Check that roll2d outputs contiguous arrays whatever the input
    imcstyle = np.random.random(size=testshape)
    rolltest = galsim.utilities.roll2d(imcstyle, (+1, -1))
    assert rolltest.flags.c_contiguous
    imfstyle = np.random.random(size=testshape).T
    rolltest = galsim.utilities.roll2d(imfstyle, (+1, -1))
    assert rolltest.flags.c_contiguous
    # Check kx, ky
    kx, ky = galsim.utilities.kxky(testshape)
    assert kx.flags.c_contiguous
    assert ky.flags.c_contiguous
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_interleaveImages():
    import time
    t1 = time.time()

    # 1a) With galsim Gaussian
    g = galsim.Gaussian(sigma=3.7,flux=1000.)
    gal = galsim.Convolve([g,galsim.Pixel(1.0)])
    im_list = []
    offset_list = []
    n = 2
    for j in xrange(n):
        for i in xrange(n):
            im = galsim.Image(16*n,16*n)
            offset = galsim.PositionD(-(i+0.5)/n+0.5,-(j+0.5)/n+0.5)
            offset_list.append(offset)
            gal.drawImage(image=im,scale=1.0,method='no_pixel',offset=offset)
            #gal.drawImage(image=im,scale=1.0,method='no_pixel',offset=galsim.PositionD(1.*i/n,1.*j/n))
            im_list.append(im)

    # Input to N as an int
    img = galsim.utilities.interleaveImages(im_list,n,offsets=offset_list)
    im = galsim.Image(16*n*n,16*n*n)
    g = galsim.Gaussian(sigma=3.7,flux=1000.*n*n)
    gal = galsim.Convolve([g,galsim.Pixel(1.0)])
    gal.drawImage(image=im,method='no_pixel',offset=galsim.PositionD(0.0,0.0),scale=1.0/n)
    np.testing.assert_array_equal(img.array,im.array,\
        err_msg="Interleaved Gaussian images do not match")
    assert im.scale == img.scale

    # 1b) With im_list and offsets permuted
    offset_list = []
    # An elegant way of generating the default offsets
    DX = np.arange(0.0,-1.0,-1.0/n)
    DX -= DX.mean()
    DY = DX
    for dy in DY:
        for dx in DX:
            offset = galsim.PositionD(dx,dy)
            offset_list.append(offset)

    np.random.seed(42) # for generating the same random permutation everytime
    rand_idx = np.random.permutation(len(offset_list))
    im_list_randperm = [im_list[idx] for idx in rand_idx]
    offset_list_randperm = [offset_list[idx] for idx in rand_idx]
    # Input to N as a tuple
    img_randperm = galsim.utilities.interleaveImages(im_list_randperm,(n,n),offsets=offset_list_randperm)

    np.testing.assert_array_equal(img_randperm.array,img.array,\
        err_msg="Interleaved images do not match when 'offsets' is supplied")
    assert img_randperm.scale == img.scale

    # 1c) Catching errors if offsets
    offset_list = []
    im_list = []
    n = 5
    DX = np.array([-0.67,-0.33,0.,0.33,0.67])
    DY = DX
    for dy in DY:
        for dx in DX:
           offset = galsim.PositionD(dx,dy)
           offset_list.append(offset)
           im = galsim.Image(16,16)
           gal.drawImage(image=im,offset=offset,scale=1.0,method='no_pixel')
           im_list.append(im)

    VE = ValueError('No error')
    try:
        galsim.utilities.interleaveImages(im_list,n,offsets=offset_list,catch_offset_errors=True)
    except ValueError as VE:
        pass
    message =  "'offsets' must be a list of galsim.PositionD instances with x values"\
                          +" spaced by 1/{0} and y values by 1/{1} around 0.".format(n,n)
    assert VE.message == message

    offset_list = []
    im_list = []
    n = 5
    DX = np.arange(0.,1.,1./n)
    DY = DX
    for dy in DY:
        for dx in DX:
            offset = galsim.PositionD(dx,dy)
            offset_list.append(offset)
            im = galsim.Image(16,16)
            gal.drawImage(image=im,offset=offset,scale=1.0,method='no_pixel')
            im_list.append(im)

    VE = ValueError('No error')
    try:
        galsim.utilities.interleaveImages(im_list,n,offsets=offset_list,catch_offset_errors=True)
    except ValueError as VE:
        pass
    assert VE.message == message

    # 2a) Increase resolution along one direction - square to rectangular images
    n = 2
    g = galsim.Gaussian(sigma=3.7,flux=100.)
    g1 = g.shear(g=1.*(n**2-1)/(n**2+1),beta=0.0*galsim.radians)
    gal1 = g1 #galsim.Convolve([g1,galsim.Pixel(1.0)])
    im_list = []
    offset_list = []
  
    # Generating offsets in a natural way
    DY = np.arange(0.0,1.0,1.0/(n*n))
    DY -= DY.mean()
    for dy in DY:
        im = galsim.Image(16,16)
        offset = galsim.PositionD(0.0,dy)
        offset_list.append(offset)
        gal1.drawImage(im,offset=offset,scale=1.0,method='no_pixel')
        im_list.append(im)

    img = galsim.utilities.interleaveImages(im_list,N=[1,n**2],offsets=offset_list,add_flux=False,suppress_warnings=True)
    im = galsim.Image(16,16*n*n)
    g = galsim.Gaussian(sigma=3.7*n,flux=100.)
    gal = g#alsim.Convolve([g,galsim.Pixel(1.0)])
    gal.drawImage(image=im,scale=1.,method='no_pixel')

    np.testing.assert_array_equal(im.array,img.array,err_msg="Sheared gaussian not interleaved correctly")
    assert img.scale is None

    # 2b) Increase resolution along one direction - rectangular to square images
    n = 2
    g = galsim.Gaussian(sigma=3.7,flux=100.)
    g2 = g.shear(g=1.*(n**2-1)/(n**2+1),beta=90.*galsim.degrees)
    gal2 = g2
    im_list = []
    offset_list = []

    # Generating offsets in a natural way
    DX = np.arange(0.0,1.0,1.0/n**2)
    DX -= DX.mean()
    for dx in DX:
         offset = galsim.PositionD(dx,0.0)
         offset_list.append(offset)
         im = galsim.Image(16,16*n*n)
         gal2.drawImage(im,offset=offset,scale=1.0,method='no_pixel')
         im_list.append(im)

    img = galsim.utilities.interleaveImages(im_list,N=np.array([n**2,1]),offsets=offset_list,suppress_warnings=True)
    im = galsim.Image(16*n*n,16*n*n)
    g = galsim.Gaussian(sigma=3.7,flux=100.*n*n)
    gal = g
    gal.drawImage(image=im,scale=1./n,method='no_pixel')

    np.testing.assert_array_equal(im.array,img.array,err_msg="Sheared gaussian not interleaved correctly")
    assert img.scale is None
    
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_roll2d_circularity()
    test_roll2d_fwdbck()
    test_roll2d_join()
    test_kxky()
    test_kxky_plusone()
    test_check_all_contiguous()
    test_interleaveImages()
