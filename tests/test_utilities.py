# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

from __future__ import print_function
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


@timer
def test_roll2d_circularity():
    """Test both integer and float arrays are unchanged by full circular roll.
    """
    # Make heterogenous 2D array, integers first, test that a full roll gives the same as the inputs
    int_image = np.random.randint(low=0, high=1+1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.utilities.roll2d(int_image, int_image.shape),
                                  err_msg='galsim.utilities.roll2D failed int array circularity')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.utilities.roll2d(flt_image, flt_image.shape),
                                  err_msg='galsim.utilities.roll2D failed flt array circularity')


@timer
def test_roll2d_fwdbck():
    """Test both integer and float arrays are unchanged by unit forward and backward roll.
    """
    # Make heterogenous 2D array, integers first, test that a +1, -1 roll gives the same as initial
    int_image = np.random.randint(low=0, high=1+1, size=testshape)
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


@timer
def test_roll2d_join():
    """Test both integer and float arrays are equivalent if rolling +1/-1 or -/+(shape[i/j] - 1).
    """
    # Make heterogenous 2D array, integers first
    int_image = np.random.randint(low=0, high=1+1, size=testshape)
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


@timer
def test_kxky():
    """Test that the basic properties of kx and ky are right.
    """
    kx, ky = galsim.utilities.kxky((4, 4))
    kxref = np.array([0., 0.25, -0.5, -0.25]) * 2. * np.pi
    kyref = np.array([0., 0.25, -0.5, -0.25]) * 2. * np.pi
    for i in range(4):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in range(4):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))


@timer
def test_kxky_plusone():
    """Test that the basic properties of kx and ky are right...
    But increment testshape used in test_kxky by one to test both odd and even cases.
    """
    kx, ky = galsim.utilities.kxky((4 + 1, 4 + 1))
    kxref = np.array([0., 0.2, 0.4, -0.4, -0.2]) * 2. * np.pi
    kyref = np.array([0., 0.2, 0.4, -0.4, -0.2]) * 2. * np.pi
    for i in range(4 + 1):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in range(4 + 1):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))


@timer
def test_check_all_contiguous():
    """Test all galsim.optics outputs are C-contiguous as required by the galsim.Image class.
    """
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


@timer
def test_deInterleaveImage():
    np.random.seed(84) # for generating the same random instances

    # 1) Check compatability with interleaveImages
    img = galsim.Image(np.random.randn(64,64),scale=0.25)
    img.setOrigin(galsim.PositionI(5,7)) ## for non-trivial bounds
    im_list, offsets = galsim.utilities.deInterleaveImage(img,8)
    img1 = galsim.utilities.interleaveImages(im_list,8,offsets)
    np.testing.assert_array_equal(img1.array,img.array,
            err_msg = "interleaveImages cannot reproduce the input to deInterleaveImage for square "
                      "images")

    assert img.wcs == img1.wcs
    assert img.bounds == img1.bounds

    img = galsim.Image(abs(np.random.randn(16*5,16*2)),scale=0.5)
    img.setCenter(0,0) ## for non-trivial bounds
    im_list, offsets = galsim.utilities.deInterleaveImage(img,(2,5))
    img1 = galsim.utilities.interleaveImages(im_list,(2,5),offsets)
    np.testing.assert_array_equal(img1.array,img.array,
            err_msg = "interleaveImages cannot reproduce the input to deInterleaveImage for "
                      "rectangular images")

    assert img.wcs == img1.wcs
    assert img.bounds == img1.bounds

    # 2) Checking for offsets
    img = galsim.Image(np.random.randn(32,32),scale=2.0)
    im_list, offsets = galsim.utilities.deInterleaveImage(img,(4,2))

    ## Checking if offsets are centered around zero
    assert np.sum([offset.x for offset in offsets]) == 0.
    assert np.sum([offset.y for offset in offsets]) == 0.

    ## Checking if offsets are separated by (i/4,j/2) for some integers i,j
    offset0 = offsets[0]
    for offset in offsets[1:]:
        assert 4.*(offset.x-offset0.x) == int(4.*(offset.x-offset0.x))
        assert 2.*(offset.y-offset0.y) == int(2.*(offset.y-offset0.y))

    # 3a) Decreasing resolution
    g0 = galsim.Gaussian(sigma=0.8,flux=100.)
    img0 = galsim.Image(32,32)
    g0.drawImage(image=img0,method='no_pixel',scale=0.25)

    im_list0, offsets0 = galsim.utilities.deInterleaveImage(img0,2,conserve_flux=True)

    for n in range(len(im_list0)):
        im = galsim.Image(16,16)
        g0.drawImage(image=im,offset=offsets0[n],scale=0.5,method='no_pixel')
        np.testing.assert_array_equal(im.array,im_list0[n].array,
            err_msg="deInterleaveImage does not reduce the resolution of the input image correctly")
        assert im_list0[n].wcs == im.wcs

    # 3b) Increasing directional resolution
    g = galsim.Gaussian(sigma=0.7,flux=1000.)
    img = galsim.Image(16,16)
    g.drawImage(image=img,scale=0.5,method='no_pixel')

    n1,n2 = 4,2
    img1 = galsim.Image(16*n1**2,16)
    img2 = galsim.Image(16,16*n2**2)

    g1 = g.shear(g=(n1**2-1.)/(n1**2+1.),beta=0.*galsim.degrees)
    g2 = g.shear(g=(n2**2-1.)/(n2**2+1.),beta=90.*galsim.degrees)
    g1.drawImage(image=img1,scale=0.5/n1,method='no_pixel')
    g2.drawImage(image=img2,scale=0.5/n2,method='no_pixel')

    im_list1, offsets1 = galsim.utilities.deInterleaveImage(img1,(n1**2,1),conserve_flux=True)
    im_list2, offsets2 = galsim.utilities.deInterleaveImage(img2,[1,n2**2],conserve_flux=False)

    for n in range(n1**2):
        im, offset = im_list1[n], offsets1[n]
        img = g.drawImage(image=img,offset=offset,scale=0.5,method='no_pixel')
        np.testing.assert_array_equal(im.array,img.array,
            err_msg="deInterleaveImage does not reduce the resolution correctly along the "
                    "vertical direction")

    for n in range(n2**2):
        im, offset = im_list2[n], offsets2[n]
        g.drawImage(image=img,offset=offset,scale=0.5,method='no_pixel')
        np.testing.assert_array_equal(im.array*n2**2,img.array,
            err_msg="deInterleaveImage does not reduce the resolution correctly along the "
                     "horizontal direction")
        # im is scaled to account for flux not being conserved


@timer
def test_interleaveImages():
    # 1a) With galsim Gaussian
    g = galsim.Gaussian(sigma=3.7,flux=1000.)
    gal = galsim.Convolve([g,galsim.Pixel(1.0)])
    im_list = []
    offset_list = []
    n = 2
    for j in range(n):
        for i in range(n):
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
    np.testing.assert_almost_equal(img.array,im.array, 6,
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

    np.testing.assert_array_equal(img_randperm.array,img.array,
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
        np.testing.assert_raises(ValueError, galsim.utilities.interleaveImages,
                                 im_list, N, offset_list)
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

    img = galsim.utilities.interleaveImages(im_list, N=[1,n**2], offsets=offset_list,
                                            add_flux=False, suppress_warnings=True)
    im = galsim.Image(16,16*n*n)
    # The interleaved image has the total flux averaged out since `add_flux = False'
    gal = galsim.Gaussian(sigma=3.7*n,flux=100.)
    gal.drawImage(image=im,method='no_pixel',scale=2.0)

    np.testing.assert_array_equal(im.array, img.array,
                                  err_msg="Sheared gaussian not interleaved correctly")
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

    img = galsim.utilities.interleaveImages(im_list, N=np.array([n**2,1]), offsets=offset_list,
                                            suppress_warnings=True)
    im = galsim.Image(16*n*n,16*n*n)
    gal = galsim.Gaussian(sigma=3.7,flux=100.*n*n)
    scale = im_list[0].scale
    gal.drawImage(image=im,scale=1.*scale/n,method='no_pixel')

    np.testing.assert_almost_equal(im.array, img.array, 12,
                                  err_msg="Sheared gaussian not interleaved correctly")
    assert img.wcs == galsim.JacobianWCS(1.*scale/n**2,0.0,0.0,scale)

    # 3) Check compatability with deInterleaveImage
    n = 3
    g = galsim.Gaussian(sigma=3.7,flux=100.)
    # break symmetry to detect possible bugs in deInterleaveImage
    gal = g.shear(g=0.2,beta=0.*galsim.degrees)
    im_list = []
    offset_list = []

    # Generating offsets in the order they would be returned by deInterleaveImage, for convenience
    for i in range(n):
        for j in range(n):
            im = galsim.Image(16*n,16*n)
            offset = galsim.PositionD(-(i+0.5)/n+0.5,-(j+0.5)/n+0.5)
            offset_list.append(offset)
            gal.drawImage(image=im,method='no_pixel',offset=offset,scale=0.5)
            im.setOrigin(3,3) # for non-trivial bounds
            im_list.append(im)

    img = galsim.utilities.interleaveImages(im_list,N=n,offsets=offset_list)
    im_list_1, offset_list_1 = galsim.utilities.deInterleaveImage(img, N=n)

    for k in range(n**2):
        assert offset_list_1[k] == offset_list[k]
        np.testing.assert_array_equal(im_list_1[k].array, im_list[k].array)
        assert im_list_1[k].wcs == im_list[k].wcs

        assert im_list[k].origin == img.origin
        assert im_list[k].bounds == im_list_1[k].bounds

    # Checking for non-default flux option
    img = galsim.utilities.interleaveImages(im_list,N=n,offsets=offset_list,add_flux=False)
    im_list_2, offset_list_2 = galsim.utilities.deInterleaveImage(img,N=n,conserve_flux=True)

    for k in range(n**2):
        assert offset_list_2[k] == offset_list[k]
        np.testing.assert_array_equal(im_list_2[k].array, im_list[k].array)
        assert im_list_2[k].wcs == im_list[k].wcs


@timer
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

@timer
def test_rand_with_replacement():
    """Test routine to select random indices with replacement."""
    # Most aspects of this routine get tested when it's used by COSMOSCatalog.  We just check some
    # of the exception-handling here.
    try:
        np.testing.assert_raises(ValueError, galsim.utilities.rand_with_replacement,
                                 n=1.5, n_choices=10, rng=galsim.BaseDeviate(1234))
        np.testing.assert_raises(TypeError, galsim.utilities.rand_with_replacement,
                                 n=2, n_choices=10, rng='foo')
        np.testing.assert_raises(ValueError, galsim.utilities.rand_with_replacement,
                                 n=2, n_choices=10.5, rng=galsim.BaseDeviate(1234))
        np.testing.assert_raises(ValueError, galsim.utilities.rand_with_replacement,
                                 n=2, n_choices=-11, rng=galsim.BaseDeviate(1234))
        np.testing.assert_raises(ValueError, galsim.utilities.rand_with_replacement,
                                 n=-2, n_choices=11, rng=galsim.BaseDeviate(1234))
        tmp_weights = np.arange(10).astype(float)-3
        np.testing.assert_raises(ValueError, galsim.utilities.rand_with_replacement,
                                 n=2, n_choices=10, rng=galsim.BaseDeviate(1234),
                                 weight=tmp_weights)
        tmp_weights[0] = np.nan
        np.testing.assert_raises(ValueError, galsim.utilities.rand_with_replacement,
                                 n=2, n_choices=10, rng=galsim.BaseDeviate(1234),
                                 weight=tmp_weights)
        tmp_weights[0] = np.inf
        np.testing.assert_raises(ValueError, galsim.utilities.rand_with_replacement,
                                 n=2, n_choices=10, rng=galsim.BaseDeviate(1234),
                                 weight=tmp_weights)
    except ImportError:
        print("The assert_raises tests require nose")

    # Make sure results come out the same whether we use _n_rng_calls or not.
    result_1 = galsim.utilities.rand_with_replacement(n=10, n_choices=100,
                                                      rng=galsim.BaseDeviate(314159))
    result_2, _ = galsim.utilities.rand_with_replacement(n=10, n_choices=100,
                                                         rng=galsim.BaseDeviate(314159),
                                                         _n_rng_calls=True)
    assert np.all(result_1==result_2),"Using _n_rng_calls results in different random numbers"
    weight = np.zeros(100)
    galsim.UniformDeviate(1234).generate(weight)
    result_1 = galsim.utilities.rand_with_replacement(
        n=10, n_choices=100, rng=galsim.BaseDeviate(314159), weight=weight)
    assert not np.all(result_1==result_2),"Weights did not have an effect"
    result_2, _ = galsim.utilities.rand_with_replacement(
        n=10, n_choices=100, rng=galsim.BaseDeviate(314159),
        weight=weight, _n_rng_calls=True)
    assert np.all(result_1==result_2),"Using _n_rng_calls results in different random numbers"

@timer
def test_position_type_promotion():
    pd1 = galsim.PositionD(0.1, 0.2)
    pd2 = galsim.PositionD(-0.3, 0.4)
    pd3 = galsim.PositionD()  # Also test 0-argument initializer here

    pi1 = galsim.PositionI(3, 65)
    pi2 = galsim.PositionI(-4, 4)
    pi3 = galsim.PositionI()

    # First check combinations that should yield a PositionD
    for lhs, rhs in zip([pd1, pd1, pi1, pd1, pi2], [pd2, pi1, pd2, pi3, pd3]):
        assert lhs+rhs == galsim.PositionD(lhs.x+rhs.x, lhs.y+rhs.y)
        assert lhs-rhs == galsim.PositionD(lhs.x-rhs.x, lhs.y-rhs.y)

    # Also check PosI +/- PosI -> PosI
    assert pi1+pi2 == galsim.PositionI(pi1.x+pi2.x, pi1.y+pi2.y)
    assert pi1-pi2 == galsim.PositionI(pi1.x-pi2.x, pi1.y-pi2.y)


@timer
def test_unweighted_moments():
    sigma = 0.8
    gal = galsim.Gaussian(sigma=sigma)
    scale = 0.02    # Use a small scale and a large image so we can neglect the impact of boundaries
    nx = ny = 1024  # and pixelization in tests.
    img1 = gal.drawImage(nx=nx, ny=ny, scale=scale, method='no_pixel')

    mom = galsim.utilities.unweighted_moments(img1)
    shape = galsim.utilities.unweighted_shape(mom)

    # Check that shape derived from moments is same as shape derived from image.
    shape2 = galsim.utilities.unweighted_shape(img1)
    assert shape == shape2

    # Object should show up at the image true center.
    np.testing.assert_almost_equal(mom['Mx'], img1.true_center.x)
    np.testing.assert_almost_equal(mom['My'], img1.true_center.y)
    # And have the right sigma = rsqr/2
    np.testing.assert_almost_equal(mom['Mxx']*scale**2, sigma**2)
    np.testing.assert_almost_equal(mom['Myy']*scale**2, sigma**2)
    np.testing.assert_almost_equal(mom['Mxy'], 0.0)
    np.testing.assert_almost_equal(shape['e1'], 0.0)
    np.testing.assert_almost_equal(shape['e2'], 0.0)


    # Add in some ellipticity and test that
    e1 = 0.2
    e2 = 0.3
    gal = gal.shear(e1=e1, e2=e2)
    img2 = gal.drawImage(nx=nx, ny=ny, scale=scale, method='no_pixel')

    mom2 = galsim.utilities.unweighted_moments(img2)
    shape3 = galsim.utilities.unweighted_shape(mom2)

    # Check that shape derived from moments is same as shape derived from image.
    shape4 = galsim.utilities.unweighted_shape(img2)
    assert shape3 == shape4

    np.testing.assert_almost_equal(mom2['Mx'], img2.true_center.x)
    np.testing.assert_almost_equal(mom2['My'], img2.true_center.y)
    np.testing.assert_almost_equal(shape3['e1'], e1)
    np.testing.assert_almost_equal(shape3['e2'], e2)

    # Check subimage still works
    bds = galsim.BoundsI(15, 1022, 11, 1002)
    subimg = img2[bds]

    mom3 = galsim.utilities.unweighted_moments(subimg)
    shape5 = galsim.utilities.unweighted_shape(subimg)

    for key in mom2:
        np.testing.assert_almost_equal(mom2[key], mom3[key])
    for key in shape3:
        np.testing.assert_almost_equal(shape3[key], shape5[key])

    # Test unweighted_moments origin keyword.  Using origin=true_center should make centroid result
    # (0.0, 0.0)
    mom4 = galsim.utilities.unweighted_moments(img2, origin=img2.true_center)
    np.testing.assert_almost_equal(mom4['Mx'], 0.0)
    np.testing.assert_almost_equal(mom4['My'], 0.0)


def test_dol_to_lod():
    """Check broadcasting behavior of dol_to_lod"""

    i0 = 0
    l1 = [1]
    l2 = [1, 2]
    l3 = [1, 2, 3]
    d1 = {1:1}

    # Should be able to broadcast scalar elements or lists of length 1.
    dd = dict(i0=i0, l2=l2)
    for i, d in enumerate(galsim.utilities.dol_to_lod(dd)):
        assert d == dict(i0=i0, l2=l2[i])

    dd = dict(l1=l1, l2=l2)
    for i, d in enumerate(galsim.utilities.dol_to_lod(dd)):
        assert d == dict(l1=l1[0], l2=l2[i])

    dd = dict(l1=l1, l3=l3)
    for i, d in enumerate(galsim.utilities.dol_to_lod(dd, 3)):
        assert d == dict(l1=l1[0], l3=l3[i])

    # Can't broadcast list of lengths 2 and 3 though.
    try:
        dd = dict(l2=l2, l3=l3)
        np.testing.assert_raises(ValueError, list, galsim.utilities.dol_to_lod(dd))
    except ImportError:
        print('The assert_raises tests require nose')

    # Can't broadcast a dictionary
    try:
        dd = dict(l2=l2, d1=d1)
        np.testing.assert_raises(ValueError, list, galsim.utilities.dol_to_lod(dd))
    except ImportError:
        print('The assert_raises tests require nose')


if __name__ == "__main__":
    test_roll2d_circularity()
    test_roll2d_fwdbck()
    test_roll2d_join()
    test_kxky()
    test_kxky_plusone()
    test_check_all_contiguous()
    test_deInterleaveImage()
    test_interleaveImages()
    test_python_LRU_Cache()
    test_rand_with_replacement()
    test_position_type_promotion()
    test_unweighted_moments()
    test_dol_to_lod()
