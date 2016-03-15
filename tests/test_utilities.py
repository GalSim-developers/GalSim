from __future__ import print_function
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
    print('time for %s = %.2f'%(funcname(),t2-t1))

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
    print('time for %s = %.2f'%(funcname(),t2-t1))

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
    print('time for %s = %.2f'%(funcname(),t2-t1))

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
    print('time for %s = %.2f'%(funcname(),t2-t1))

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
    print('time for %s = %.2f'%(funcname(),t2-t1))

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
    print('time for %s = %.2f'%(funcname(),t2-t1))

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
            gal.drawImage(image=im,method='no_pixel',offset=offset,scale=0.5)
            im_list.append(im)

    scale = im.scale

    # Input to N as an int
    img = galsim.utilities.interleaveImages(im_list,n,offsets=offset_list)
    im = galsim.Image(16*n*n,16*n*n)
    g = galsim.Gaussian(sigma=3.7,flux=1000.*n*n)
    gal = galsim.Convolve([g,galsim.Pixel(1.0)])
    gal.drawImage(image=im,method='no_pixel',offset=galsim.PositionD(0.0,0.0),scale=1.*scale/n)
    np.testing.assert_array_equal(img.array,im.array,\
        err_msg="Interleaved Gaussian images do not match")

    assert im.wcs == img.wcs

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

    # 1c) Catching errors in offsets
    offset_list = []
    im_list = []
    n = 5
    # Generate approximate offsets
    DX = np.array([-0.67,-0.33,0.,0.33,0.67])
    DY = DX
    for dy in DY:
        for dx in DX:
           offset = galsim.PositionD(dx,dy)
           offset_list.append(offset)
           im = galsim.Image(16,16)
           gal.drawImage(image=im,offset=offset,method='no_pixel')
           im_list.append(im)

    try:
        N = (n,n)
        np.testing.assert_raises(ValueError,galsim.utilities.interleaveImages,im_list,N,offset_list)
    except ImportError:
        print("The assert_raises tests require nose")

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
            gal.drawImage(image=im,offset=offset,method='no_pixel')
            im_list.append(im)

    try:
        N = (n,n)
        np.testing.assert_raises(ValueError,galsim.utilities.interleaveImages,im_list,N,offset_list)
    except ImportError:
        print("The assert_raises tests require nose")

    # 2a) Increase resolution along one direction - square to rectangular images
    n = 2
    g = galsim.Gaussian(sigma=3.7,flux=100.)
    gal1 = g.shear(g=1.*(n**2-1)/(n**2+1),beta=0.0*galsim.radians)
    im_list = []
    offset_list = []
  
    # Generating offsets in a natural way
    DY = np.arange(0.0,1.0,1.0/(n*n))
    DY -= DY.mean()
    for dy in DY:
        im = galsim.Image(16,16)
        offset = galsim.PositionD(0.0,dy)
        offset_list.append(offset)
        gal1.drawImage(im,offset=offset,method='no_pixel',scale=2.0)
        im_list.append(im)

    img = galsim.utilities.interleaveImages(im_list,N=[1,n**2],offsets=offset_list,add_flux=False,suppress_warnings=True)
    im = galsim.Image(16,16*n*n)
    # The interleaved image has the total flux averaged out since `add_flux = False'
    gal = galsim.Gaussian(sigma=3.7*n,flux=100.)
    gal.drawImage(image=im,method='no_pixel',scale=2.0)

    np.testing.assert_array_equal(im.array,img.array,err_msg="Sheared gaussian not interleaved correctly")
    assert img.wcs == galsim.JacobianWCS(2.0,0.0,0.0,2./(n**2))

    # 2b) Increase resolution along one direction - rectangular to square images
    n = 2
    g = galsim.Gaussian(sigma=3.7,flux=100.)
    gal2 = g.shear(g=1.*(n**2-1)/(n**2+1),beta=90.*galsim.degrees)
    im_list = []
    offset_list = []

    # Generating offsets in a natural way
    DX = np.arange(0.0,1.0,1.0/n**2)
    DX -= DX.mean()
    for dx in DX:
         offset = galsim.PositionD(dx,0.0)
         offset_list.append(offset)
         im = galsim.Image(16,16*n*n)
         gal2.drawImage(im,offset=offset,method='no_pixel',scale=3.0)
         im_list.append(im)

    img = galsim.utilities.interleaveImages(im_list,N=np.array([n**2,1]),offsets=offset_list,suppress_warnings=True)
    im = galsim.Image(16*n*n,16*n*n)
    gal = galsim.Gaussian(sigma=3.7,flux=100.*n*n)
    scale = im_list[0].scale
    gal.drawImage(image=im,scale=1.*scale/n,method='no_pixel')

    np.testing.assert_array_equal(im.array,img.array,err_msg="Sheared gaussian not interleaved correctly")
    assert img.wcs == galsim.JacobianWCS(1.*scale/n**2,0.0,0.0,scale)
    
    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


def test_python_LRU_Cache():
    f = lambda x: x+1
    size = 10
    # Test correct size cache gets created
    cache = galsim.utilities.LRU_Cache(f, maxsize=size)
    assert len(cache.cache) == size
    # Insert f(0) = 1 into cache and check that we can get it back
    assert cache(0) == f(0)
    assert cache(0) == f(0)

    # Manually manipulate cache so we can check for hit
    cache.cache[(0,)][3] = 2
    assert cache(0) == 2

    # Insert (and check) 1 thru size into cache.  This should bump out the (0,).
    for i in range(1, size+1):
        assert cache(i) == f(i)
    assert (0,) not in cache.cache

    # Test non-destructive cache expansion
    newsize = 20
    cache.resize(newsize)
    for i in range(1, size+1):
        assert (i,) in cache.cache
        assert cache(i) == f(i)
    assert len(cache.cache) == 20

    # Add new items until the (1,) gets bumped
    for i in range(size+1, newsize+2):
        assert cache(i) == f(i)
    assert (1,) not in cache.cache

    # Test mostly non-destructive cache contraction.
    # Already bumped (0,) and (1,), so (2,) should be the first to get bumped
    for i in range(newsize-1, size, -1):
        assert (newsize - (i - 1),) in cache.cache
        cache.resize(i)
        assert (newsize - (i - 1),) not in cache.cache


if __name__ == "__main__":
    test_roll2d_circularity()
    test_roll2d_fwdbck()
    test_roll2d_join()
    test_kxky()
    test_kxky_plusone()
    test_check_all_contiguous()
    test_interleaveImages()
    test_python_LRU_Cache()
