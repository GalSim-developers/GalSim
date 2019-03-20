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

from __future__ import print_function
import numpy as np
import os
import sys

import galsim
from galsim_test_helpers import *

testshape = (512, 512)  # shape of image arrays for all tests
decimal = 6     # Last decimal place used for checking equality of float arrays, see
                # np.testing.assert_array_almost_equal(), low since many are ImageF


@timer
def test_pos():
    """Simple tests of Position classes
    """
    pi1 = galsim.PositionI(11,23)
    assert pi1.x == 11
    assert pi1.y == 23
    assert isinstance(pi1.x, int)
    assert isinstance(pi1.y, int)

    pi2 = galsim.PositionI((11,23))
    pi3 = galsim.PositionI(x=11.0, y=23.0)
    pi4 = galsim.PositionI(pi1)
    pi5 = galsim.PositionI(galsim.PositionD(11.0,23.0))
    assert pi2 == pi1
    assert pi3 == pi1
    assert pi4 == pi1
    assert pi5 == pi1
    assert isinstance(pi3.x, int)
    assert isinstance(pi3.y, int)
    assert isinstance(pi5.x, int)
    assert isinstance(pi5.y, int)

    pd1 = galsim.PositionD(11.,23.)
    assert pd1.x == 11.
    assert pd1.y == 23.
    assert isinstance(pd1.x, float)
    assert isinstance(pd1.y, float)

    pd2 = galsim.PositionD((11,23))
    pd3 = galsim.PositionD(x=11.0, y=23.0)
    pd4 = galsim.PositionD(pd1)
    pd5 = galsim.PositionD(pi1)
    assert pd2 == pd1
    assert pd3 == pd1
    assert pd4 == pd1
    assert pd5 == pd1
    assert isinstance(pd3.x, float)
    assert isinstance(pd3.y, float)
    assert isinstance(pd5.x, float)
    assert isinstance(pd5.y, float)

    assert_raises(TypeError, galsim.PositionI, 11)
    assert_raises(TypeError, galsim.PositionI, 11, 23, 9)
    assert_raises(TypeError, galsim.PositionI, x=11, z=23)
    assert_raises(TypeError, galsim.PositionI, x=11)
    assert_raises(TypeError, galsim.PositionD, x=11, y=23, z=17)
    assert_raises(TypeError, galsim.PositionI, 11, 23, x=13, z=21)
    assert_raises(TypeError, galsim.PositionI, 11, 23.5)

    assert_raises(TypeError, galsim.PositionD, 11)
    assert_raises(TypeError, galsim.PositionD, 11, 23, 9)
    assert_raises(TypeError, galsim.PositionD, x=11, z=23)
    assert_raises(TypeError, galsim.PositionD, x=11)
    assert_raises(TypeError, galsim.PositionD, x=11, y=23, z=17)
    assert_raises(TypeError, galsim.PositionD, 11, 23, x=13, z=21)
    assert_raises(ValueError, galsim.PositionD, 11, "blue")

    # Can't use base class directly.
    assert_raises(TypeError, galsim.Position, 11, 23)
    assert_raises(NotImplementedError, galsim.Position)

    # Check arithmetic
    for p1 in [pi1, pd1]:

        p2 = p1 * 2
        assert p2.x == p1.x * 2
        assert p2.y == p1.y * 2

        p3 = p2 / 2
        assert p3 == p1

        p4 = 2 * p1
        assert p4 == p2

        p5 = -p1
        assert p5.x == -p1.x
        assert p5.y == -p1.y

        p6 = p1 + p2
        assert p6.x == 3 * p1.x
        assert p6.y == 3 * p1.y

        p7 = p2 - p1
        assert p7.x == p1.x
        assert p7.y == p1.y

    # Cross type arithemetic -> PositionD
    pd6 = pi1 + pd1
    assert pd6 == 2*pd1
    assert isinstance(pd6, galsim.PositionD)

    pd7 = pd1 + pi1
    assert pd7 == 2*pd1
    assert isinstance(pd7, galsim.PositionD)

    pd8 = pi1 - pd1
    assert pd8 == 0*pd1
    assert isinstance(pd8, galsim.PositionD)

    pd9 = pd1 - pi1
    assert pd9 == 0*pd1
    assert isinstance(pd9, galsim.PositionD)

    assert_raises(TypeError, pd1.__add__, 11)
    assert_raises(TypeError, pd1.__sub__, 11)
    assert_raises(TypeError, pd1.__mul__, "11")
    assert_raises(TypeError, pd1.__mul__, None)
    assert_raises(TypeError, pd1.__div__, "11e")

    assert_raises(TypeError, pi1.__add__, 11)
    assert_raises(TypeError, pi1.__sub__, 11)
    assert_raises(TypeError, pi1.__mul__, "11e")
    assert_raises(TypeError, pi1.__mul__, None)
    assert_raises(TypeError, pi1.__div__, 11.5)

    do_pickle(pi1)
    do_pickle(pd1)

@timer
def test_bounds():
    """Simple tests of Bounds classes
    """
    bi1 = galsim.BoundsI(11,23,17,50)
    assert bi1.xmin == bi1.getXMin() == 11
    assert bi1.xmax == bi1.getXMax() == 23
    assert bi1.ymin == bi1.getYMin() == 17
    assert bi1.ymax == bi1.getYMax() == 50
    assert isinstance(bi1.xmin, int)
    assert isinstance(bi1.xmax, int)
    assert isinstance(bi1.ymin, int)
    assert isinstance(bi1.ymax, int)

    bi2 = galsim.BoundsI(galsim.PositionI(11,17), galsim.PositionI(23,50))
    bi3 = galsim.BoundsI(galsim.PositionD(11.,50.), galsim.PositionD(23.,17.))
    bi4 = galsim.BoundsI(galsim.PositionD(11.,17.)) + galsim.BoundsI(galsim.PositionI(23,50))
    bi5 = galsim.BoundsI(galsim.PositionI(11,17)) + galsim.PositionI(23,50)
    bi6 = galsim.PositionI(11,17) + galsim.BoundsI(galsim.PositionI(23,50))
    bi7 = galsim.BoundsI(bi1)
    bi8 = bi1 + galsim.BoundsI()
    bi9 = galsim.BoundsI() + bi1
    bi10 = galsim.BoundsI() + galsim.PositionI(11,17) + galsim.PositionI(23,50)
    bi11 = galsim.BoundsI(galsim.BoundsD(11.,23.,17.,50.))
    bi12 = galsim.BoundsI(xmin=11,ymin=17,xmax=23,ymax=50)
    bi13 = galsim._BoundsI(11,23,17,50)
    bi14 = galsim.BoundsI()
    bi14 += galsim.PositionI(11,17)
    bi14 += galsim.PositionI(23,50)
    for b in [bi1, bi2, bi3, bi4, bi5, bi6, bi7, bi8, bi9, bi10, bi11, bi12, bi13, bi14]:
        assert b.isDefined()
        assert b == bi1
        assert isinstance(b.xmin, int)
        assert isinstance(b.xmax, int)
        assert isinstance(b.ymin, int)
        assert isinstance(b.ymax, int)
        assert b.origin == galsim.PositionI(11, 17)
        assert b.center == galsim.PositionI(17, 34)
        assert b.true_center == galsim.PositionD(17, 33.5)

    bd1 = galsim.BoundsD(11.,23.,17.,50.)
    assert bd1.xmin == bd1.getXMin() == 11.
    assert bd1.xmax == bd1.getXMax() == 23.
    assert bd1.ymin == bd1.getYMin() == 17.
    assert bd1.ymax == bd1.getYMax() == 50.
    assert isinstance(bd1.xmin, float)
    assert isinstance(bd1.xmax, float)
    assert isinstance(bd1.ymin, float)
    assert isinstance(bd1.ymax, float)

    bd2 = galsim.BoundsD(galsim.PositionI(11,17), galsim.PositionI(23,50))
    bd3 = galsim.BoundsD(galsim.PositionD(11.,50.), galsim.PositionD(23.,17.))
    bd4 = galsim.BoundsD(galsim.PositionD(11.,17.)) + galsim.BoundsD(galsim.PositionI(23,50))
    bd5 = galsim.BoundsD(galsim.PositionI(11,17)) + galsim.PositionD(23,50)
    bd6 = galsim.PositionD(11,17) + galsim.BoundsD(galsim.PositionI(23,50))
    bd7 = galsim.BoundsD(bd1)
    bd8 = bd1 + galsim.BoundsD()
    bd9 = galsim.BoundsD() + bd1
    bd10 = galsim.BoundsD() + galsim.PositionD(11,17) + galsim.PositionD(23,50)
    bd11 = galsim.BoundsD(galsim.BoundsI(11,23,17,50))
    bd12 = galsim.BoundsD(xmin=11.0,ymin=17.0,xmax=23.0,ymax=50.0)
    bd13 = galsim._BoundsD(11,23,17,50)
    bd14 = galsim.BoundsD()
    bd14 += galsim.PositionD(11.,17.)
    bd14 += galsim.PositionD(23,50)
    for b in [bd1, bd2, bd3, bd4, bd5, bd6, bd7, bd8, bd9, bd10, bd11, bd12, bd13, bd14]:
        assert b.isDefined()
        assert b == bd1
        assert isinstance(b.xmin, float)
        assert isinstance(b.xmax, float)
        assert isinstance(b.ymin, float)
        assert isinstance(b.ymax, float)
        assert b.origin == galsim.PositionD(11, 17)
        assert b.center == galsim.PositionD(17, 33.5)
        assert b.true_center == galsim.PositionD(17, 33.5)

    assert_raises(TypeError, galsim.BoundsI, 11)
    assert_raises(TypeError, galsim.BoundsI, 11, 23)
    assert_raises(TypeError, galsim.BoundsI, 11, 23, 9)
    assert_raises(TypeError, galsim.BoundsI, 11, 23, 9, 12, 59)
    assert_raises(TypeError, galsim.BoundsI, xmin=11, xmax=23, ymin=17, ymax=50, z=23)
    assert_raises(TypeError, galsim.BoundsI, xmin=11, xmax=50)
    assert_raises(TypeError, galsim.BoundsI, 11, 23.5, 17, 50.9)
    assert_raises(TypeError, galsim.BoundsI, 11, 23, 9, 12, xmin=19, xmax=2)
    with assert_raises(TypeError):
        bi1 += (11,23)

    assert_raises(TypeError, galsim.BoundsD, 11)
    assert_raises(TypeError, galsim.BoundsD, 11, 23)
    assert_raises(TypeError, galsim.BoundsD, 11, 23, 9)
    assert_raises(TypeError, galsim.BoundsD, 11, 23, 9, 12, 59)
    assert_raises(TypeError, galsim.BoundsD, xmin=11, xmax=23, ymin=17, ymax=50, z=23)
    assert_raises(TypeError, galsim.BoundsD, xmin=11, xmax=50)
    assert_raises(ValueError, galsim.BoundsD, 11, 23, 17, "blue")
    assert_raises(TypeError, galsim.BoundsD, 11, 23, 9, 12, xmin=19, xmax=2)
    with assert_raises(TypeError):
        bd1 += (11,23)

    # Can't use base class directly.
    assert_raises(TypeError, galsim.Bounds, 11, 23, 9, 12)
    assert_raises(NotImplementedError, galsim.Bounds)

    # Check intersection
    assert bi1 == galsim.BoundsI(0,100,0,100) & bi1
    assert bi1 == bi1 & galsim.BoundsI(0,100,0,100)
    assert bi1 == galsim.BoundsI(0,23,0,50) & galsim.BoundsI(11,100,17,100)
    assert bi1 == galsim.BoundsI(0,23,17,100) & galsim.BoundsI(11,100,0,50)
    assert not (bi1 & galsim.BoundsI()).isDefined()
    assert not (galsim.BoundsI() & bi1).isDefined()

    assert bd1 == galsim.BoundsD(0,100,0,100) & bd1
    assert bd1 == bd1 & galsim.BoundsD(0,100,0,100)
    assert bd1 == galsim.BoundsD(0,23,0,50) & galsim.BoundsD(11,100,17,100)
    assert bd1 == galsim.BoundsD(0,23,17,100) & galsim.BoundsD(11,100,0,50)
    assert not (bd1 & galsim.BoundsD()).isDefined()
    assert not (galsim.BoundsD() & bd1).isDefined()

    with assert_raises(TypeError):
        bi1 & galsim.PositionI(1,2)
    with assert_raises(TypeError):
        bi1 & galsim.PositionD(1,2)
    with assert_raises(TypeError):
        bd1 & galsim.PositionI(1,2)
    with assert_raises(TypeError):
        bd1 & galsim.PositionD(1,2)

    # Check withBorder
    assert bi1.withBorder(4) == galsim.BoundsI(7,27,13,54)
    assert bi1.withBorder(0) == galsim.BoundsI(11,23,17,50)
    assert bi1.withBorder(-1) == galsim.BoundsI(12,22,18,49)
    assert bd1.withBorder(4.1) == galsim.BoundsD(6.9,27.1,12.9,54.1)
    assert bd1.withBorder(0) == galsim.BoundsD(11,23,17,50)
    assert bd1.withBorder(-1) == galsim.BoundsD(12,22,18,49)
    assert_raises(TypeError, bi1.withBorder, 'blue')
    assert_raises(TypeError, bi1.withBorder, 4.1)
    assert_raises(TypeError, bi1.withBorder, '4')
    assert_raises(TypeError, bi1.withBorder, None)
    assert_raises(TypeError, bd1.withBorder, 'blue')
    assert_raises(TypeError, bd1.withBorder, '4.1')
    assert_raises(TypeError, bd1.withBorder, None)

    # Check expand
    assert bi1.expand(2) == galsim.BoundsI(5,29,0,67)
    assert bi1.expand(1.1) == galsim.BoundsI(10,24,15,52)
    assert bd1.expand(2) == galsim.BoundsD(5,29,0.5,66.5)
    np.testing.assert_almost_equal(bd1.expand(1.1)._getinitargs(), (10.4,23.6,15.35,51.65))

    # Check shift
    assert bi1.shift(galsim.PositionI(2,5)) == galsim.BoundsI(13,25,22,55)
    assert bd1.shift(galsim.PositionD(2,5)) == galsim.BoundsD(13,25,22,55)
    assert bd1.shift(galsim.PositionD(2.3,5.9)) == galsim.BoundsD(13.3,25.3,22.9,55.9)
    assert_raises(TypeError, bi1.shift, galsim.PositionD(2,5))
    assert_raises(TypeError, bd1.shift, galsim.PositionI(2,5))

    # Check area
    assert bd1.area() == 12 * 33
    assert bi1.area() == 13 * 34
    assert galsim.BoundsI(galsim.PositionI(11,23)).area() == 1
    assert galsim.BoundsD(galsim.PositionI(11,23)).area() == 0

    # Check includes
    for b in [bi1, bd1]:
        assert b.includes(galsim.PositionI(11,23))
        assert b.includes(galsim.BoundsI(14,18,30,38))
        assert b.includes(galsim.BoundsD(14.7,18.1,30.2,38.6))
        assert b.includes(17, 23)
        assert b.includes(17.9, 23.9)
        assert b.includes(galsim.PositionD(11.9,40.7))
        assert b.includes(galsim.PositionI(23,41))

        assert not bd1.includes(galsim.PositionD(10.99,38))
        assert not bd1.includes(galsim.PositionI(11,51))
        assert not bd1.includes(17,16.99)
        assert not bd1.includes(galsim.BoundsD(0,100,0,100))
        assert not bd1.includes(galsim.BoundsI(14,29,20,30))
        assert not bd1.includes(galsim.BoundsD(22,23.01,49,50.01))

        assert_raises(TypeError, b.includes, 'blue')
        assert_raises(TypeError, b.includes)
        assert_raises(TypeError, b.includes, galsim.PositionI(17,23), galsim.PositionI(12,13))
        assert_raises(TypeError, b.includes, 2, 3, 4)

    # Check undefined bounds
    assert not galsim.BoundsI().isDefined()
    assert galsim.BoundsI() == galsim.BoundsI() & bi1
    assert galsim.BoundsI() == bi1 & galsim.BoundsI()
    assert galsim.BoundsI() == galsim.BoundsI() & galsim.BoundsI()
    assert galsim.BoundsI() == galsim.BoundsI() + galsim.BoundsI()
    assert galsim.BoundsI().area() == 0

    assert not galsim.BoundsD().isDefined()
    assert galsim.BoundsD() == galsim.BoundsD() & bd1
    assert galsim.BoundsD() == bd1 & galsim.BoundsD()
    assert galsim.BoundsD() == galsim.BoundsD() & galsim.BoundsD()
    assert galsim.BoundsD() == galsim.BoundsD() + galsim.BoundsD()
    assert galsim.BoundsD().area() == 0

    assert galsim.BoundsI(23, 11, 17, 50) == galsim.BoundsI()
    assert galsim.BoundsI(11, 23, 50, 17) == galsim.BoundsI()
    assert galsim.BoundsD(23, 11, 17, 50) == galsim.BoundsD()
    assert galsim.BoundsD(11, 23, 50, 17) == galsim.BoundsD()

    assert_raises(galsim.GalSimUndefinedBoundsError, getattr, galsim.BoundsI(), 'center')
    assert_raises(galsim.GalSimUndefinedBoundsError, getattr, galsim.BoundsD(), 'center')
    assert_raises(galsim.GalSimUndefinedBoundsError, getattr, galsim.BoundsI(), 'true_center')
    assert_raises(galsim.GalSimUndefinedBoundsError, getattr, galsim.BoundsD(), 'true_center')

    do_pickle(bi1)
    do_pickle(bd1)
    do_pickle(galsim.BoundsI())
    do_pickle(galsim.BoundsD())


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
    from galsim.utilities import deInterleaveImage, interleaveImages

    np.random.seed(84) # for generating the same random instances

    # 1) Check compatability with interleaveImages
    img = galsim.Image(np.random.randn(64,64),scale=0.25)
    img.setOrigin(galsim.PositionI(5,7)) ## for non-trivial bounds
    im_list, offsets = deInterleaveImage(img,8)
    img1 = interleaveImages(im_list,8,offsets)
    np.testing.assert_array_equal(img1.array,img.array,
            err_msg = "interleaveImages cannot reproduce the input to deInterleaveImage for square "
                      "images")

    assert img.wcs == img1.wcs
    assert img.bounds == img1.bounds

    img = galsim.Image(abs(np.random.randn(16*5,16*2)),scale=0.5)
    img.setCenter(0,0) ## for non-trivial bounds
    im_list, offsets = deInterleaveImage(img,(2,5))
    img1 = interleaveImages(im_list,(2,5),offsets)
    np.testing.assert_array_equal(img1.array,img.array,
            err_msg = "interleaveImages cannot reproduce the input to deInterleaveImage for "
                      "rectangular images")

    assert img.wcs == img1.wcs
    assert img.bounds == img1.bounds

    # 2) Checking for offsets
    img = galsim.Image(np.random.randn(32,32),scale=2.0)
    im_list, offsets = deInterleaveImage(img,(4,2))

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

    im_list0, offsets0 = deInterleaveImage(img0,2,conserve_flux=True)

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

    im_list1, offsets1 = deInterleaveImage(img1,(n1**2,1),conserve_flux=True)
    im_list2, offsets2 = deInterleaveImage(img2,[1,n2**2],conserve_flux=False)

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

    assert_raises(TypeError, deInterleaveImage, image=img0.array, N=2)
    assert_raises(TypeError, deInterleaveImage, image=img0, N=2.0)
    assert_raises(TypeError, deInterleaveImage, image=img0, N=(2.0, 2.0))
    assert_raises(TypeError, deInterleaveImage, image=img0, N=(2,2,3))
    assert_raises(ValueError, deInterleaveImage, image=img0, N=7)
    assert_raises(ValueError, deInterleaveImage, image=img0, N=(2,7))
    assert_raises(ValueError, deInterleaveImage, image=img0, N=(7,2))

    # It is legal to have the input image with wcs=None, but it emits a warning
    img0.wcs = None
    with assert_warns(galsim.GalSimWarning):
        deInterleaveImage(img0, N=2)
    # Unless suppress_warnings is True
    deInterleaveImage(img0, N=2, suppress_warnings=True)


@timer
def test_interleaveImages():
    from galsim.utilities import interleaveImages, deInterleaveImage

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
    img = interleaveImages(im_list,n,offsets=offset_list)
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
    img_randperm = interleaveImages(im_list_randperm,(n,n),offsets=offset_list_randperm)

    np.testing.assert_array_equal(img_randperm.array,img.array,
        err_msg="Interleaved images do not match when 'offsets' is supplied")
    assert img_randperm.scale == img.scale

    # 1c) Catching errors in offsets
    offset_list = []
    im_list = []
    n = 5
    # Generate approximate offsets
    DX = np.array([-0.47,-0.23,0.,0.23,0.47])
    DY = DX
    for dy in DY:
        for dx in DX:
            offset = galsim.PositionD(dx,dy)
            offset_list.append(offset)
            im = galsim.Image(16,16)
            gal.drawImage(image=im,offset=offset,method='no_pixel')
            im_list.append(im)

    N = (n,n)
    with assert_raises(ValueError):
        interleaveImages(im_list,N,offset_list)
    # Can turn off the checks and just use these as they are with catch_offset_errors=False
    interleaveImages(im_list,N,offset_list, catch_offset_errors=False)

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

    N = (n,n)
    with assert_raises(ValueError):
        interleaveImages(im_list, N, offset_list)
    interleaveImages(im_list, N, offset_list, catch_offset_errors=False)

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

    img = interleaveImages(im_list, N=[1,n**2], offsets=offset_list,
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

    img = interleaveImages(im_list, N=np.array([n**2,1]), offsets=offset_list,
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

    img = interleaveImages(im_list,N=n,offsets=offset_list)
    im_list_1, offset_list_1 = deInterleaveImage(img, N=n)

    for k in range(n**2):
        assert offset_list_1[k] == offset_list[k]
        np.testing.assert_array_equal(im_list_1[k].array, im_list[k].array)
        assert im_list_1[k].wcs == im_list[k].wcs

        assert im_list[k].origin == img.origin
        assert im_list[k].bounds == im_list_1[k].bounds

    # Checking for non-default flux option
    img = interleaveImages(im_list,N=n,offsets=offset_list,add_flux=False)
    im_list_2, offset_list_2 = deInterleaveImage(img,N=n,conserve_flux=True)

    for k in range(n**2):
        assert offset_list_2[k] == offset_list[k]
        np.testing.assert_array_equal(im_list_2[k].array, im_list[k].array)
        assert im_list_2[k].wcs == im_list[k].wcs

    assert_raises(TypeError, interleaveImages, im_list=img, N=n, offsets=offset_list)
    assert_raises(ValueError, interleaveImages, [img], N=1, offsets=offset_list)
    assert_raises(ValueError, interleaveImages, im_list, n, offset_list[:-1])
    assert_raises(TypeError, interleaveImages, [im.array for im in im_list], n, offset_list)
    assert_raises(TypeError, interleaveImages,
                  [im_list[0]] + [im.array for im in im_list[1:]],
                  n, offset_list)
    assert_raises(TypeError, interleaveImages,
                  [galsim.Image(16+i,16+j,scale=1) for i in range(n) for j in range(n)],
                  n, offset_list)
    assert_raises(TypeError, interleaveImages,
                  [galsim.Image(16,16,scale=i) for i in range(n) for j in range(n)],
                  n, offset_list)
    assert_raises(TypeError, interleaveImages, im_list, N=3.0, offsets=offset_list)
    assert_raises(TypeError, interleaveImages, im_list, N=(3.0, 3.0), offsets=offset_list)
    assert_raises(TypeError, interleaveImages, im_list, N=(3,3,3), offsets=offset_list)
    assert_raises(ValueError, interleaveImages, im_list, N=7, offsets=offset_list)
    assert_raises(ValueError, interleaveImages, im_list, N=(2,7), offsets=offset_list)
    assert_raises(ValueError, interleaveImages, im_list, N=(7,2), offsets=offset_list)
    assert_raises(TypeError, interleaveImages, im_list, N=n)
    assert_raises(TypeError, interleaveImages, im_list, N=n, offsets=offset_list[0])
    assert_raises(TypeError, interleaveImages, im_list, N=n, offsets=range(n*n))

    # It is legal to have the input images with wcs=None, but it emits a warning
    for im in im_list:
        im.wcs = None
    with assert_warns(galsim.GalSimWarning):
        interleaveImages(im_list, N=n, offsets=offset_list)
    # Unless suppress_warnings is True
    interleaveImages(im_list, N=n, offsets=offset_list, suppress_warnings=True)


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

    # "Resize" to same size does nothing.
    cache.resize(newsize)
    assert len(cache.cache) == 20
    assert (1,) not in cache.cache
    for i in range(2, newsize+2):
        assert (i,) in cache.cache

    # Test mostly non-destructive cache contraction.
    # Already bumped (0,) and (1,), so (2,) should be the first to get bumped
    for i in range(newsize-1, size, -1):
        assert (newsize - (i - 1),) in cache.cache
        cache.resize(i)
        assert (newsize - (i - 1),) not in cache.cache

    assert_raises(ValueError, cache.resize, 0)
    assert_raises(ValueError, cache.resize, -20)


@timer
def test_rand_with_replacement():
    """Test routine to select random indices with replacement."""
    # Most aspects of this routine get tested when it's used by COSMOSCatalog.  We just check some
    # of the exception-handling here.

    # Invalid rng
    with assert_raises(TypeError):
        galsim.utilities.rand_with_replacement(
            n=2, n_choices=10, rng='foo')

    # Invalid n
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=1.5, n_choices=10, rng=galsim.BaseDeviate(1234))
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=0, n_choices=10, rng=galsim.BaseDeviate(1234))
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=-2, n_choices=10, rng=galsim.BaseDeviate(1234))

    # Invalid n_choices
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=2, n_choices=10.5, rng=galsim.BaseDeviate(1234))
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=2, n_choices=0, rng=galsim.BaseDeviate(1234))
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=2, n_choices=-11, rng=galsim.BaseDeviate(1234))

    # Negative weights
    tmp_weights = np.arange(10).astype(float)-3
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=2, n_choices=10, rng=galsim.BaseDeviate(1234), weight=tmp_weights)
    # NaN weights
    tmp_weights[0] = np.nan
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=2, n_choices=10, rng=galsim.BaseDeviate(1234), weight=tmp_weights)
    # inf weights
    tmp_weights[0] = np.inf
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=2, n_choices=10, rng=galsim.BaseDeviate(1234), weight=tmp_weights)

    # Wrong length for weights
    with assert_raises(ValueError):
        galsim.utilities.rand_with_replacement(
            n=2, n_choices=10, rng=galsim.BaseDeviate(1234), weight=tmp_weights[:4])

    # Make sure results come out the same whether we use _n_rng_calls or not.
    rng1 = galsim.BaseDeviate(314159)
    rng2 = galsim.BaseDeviate(314159)
    rng3 = galsim.BaseDeviate(314159)
    result_1 = galsim.utilities.rand_with_replacement(n=10, n_choices=100, rng=rng1)
    result_2, n_rng = galsim.utilities.rand_with_replacement(n=10, n_choices=100, rng=rng2,
                                                             _n_rng_calls=True)
    assert np.all(result_1==result_2),"Using _n_rng_calls results in different random numbers"
    rng3.discard(n_rng)
    assert rng1.raw() == rng2.raw() == rng3.raw()

    # Repeat with weights
    weight = np.zeros(100)
    galsim.UniformDeviate(1234).generate(weight)
    result_1 = galsim.utilities.rand_with_replacement(10, 100, rng1, weight=weight)
    assert not np.all(result_1==result_2),"Weights did not have an effect"
    result_2, n_rng = galsim.utilities.rand_with_replacement(10, 100, rng2, weight=weight,
                                                             _n_rng_calls=True)
    assert np.all(result_1==result_2),"Using _n_rng_calls results in different random numbers"
    rng3.discard(n_rng)
    assert rng1.raw() == rng2.raw() == rng3.raw()

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
    s = "abc"

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
    dd = dict(l2=l2, l3=l3)
    with assert_raises(ValueError):
        list(galsim.utilities.dol_to_lod(dd))

    # Can't broadcast a dictionary
    dd = dict(l2=l2, d1=d1)
    with assert_raises(ValueError):
        list(galsim.utilities.dol_to_lod(dd))

    # Strings can either be interpretted as scalar values or as lists of characters
    dd = dict(i0=i0, s=s)
    for i, d in enumerate(galsim.utilities.dol_to_lod(dd, scalar_string=False)):
        assert d == dict(i0=i0, s=s[i])
    for i, d in enumerate(galsim.utilities.dol_to_lod(dd, scalar_string=True)):
        assert d == dict(i0=i0, s=s)
    for i, d in enumerate(galsim.utilities.dol_to_lod(dd, 3, scalar_string=True)):
        assert d == dict(i0=i0, s=s)
    dd = dict(l3=l3, s=s)
    for i, d in enumerate(galsim.utilities.dol_to_lod(dd, scalar_string=True)):
        assert d == dict(l3=l3[i], s=s)
    for i, d in enumerate(galsim.utilities.dol_to_lod(dd, scalar_string=False)):
        assert d == dict(l3=l3[i], s=s[i])


@timer
def test_nCr():
    """Checking combinations utility."""
    # Check some combinations that I can do in my head...
    assert galsim.utilities.nCr(100, 0) == 1
    assert galsim.utilities.nCr(100, 1) == 100
    assert galsim.utilities.nCr(100, 2) == 100*99//2
    assert galsim.utilities.nCr(100, 98) == 100*99//2
    assert galsim.utilities.nCr(100, 99) == 100
    assert galsim.utilities.nCr(100, 100) == 1
    # Check that we get zero if not 0 <= r <= n
    assert galsim.utilities.nCr(100, 101) == 0
    assert galsim.utilities.nCr(100, -1) == 0
    # Check that Sum_r=0^n nCr(n,r) == 2^n
    for n in range(300):
        assert sum([galsim.utilities.nCr(n, r) for r in range(n+1)]) == 2**n

@timer
def test_horner():
    # Make a random polynomial
    coef = [1.2332, 3.43242, 4.1231, -0.2342, 0.4242]

    # Make a random list of values to test
    x = np.empty(20)
    rng = galsim.UniformDeviate(1234)
    rng.generate(x)

    # Check against the direct calculation
    truth = coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3 + coef[4]*x**4
    result = galsim.utilities.horner(x, coef)
    np.testing.assert_almost_equal(result, truth)

    # Also check against the (slower) numpy code
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval(x,coef))

    # Check that trailing zeros give the same answer
    result = galsim.utilities.horner(x, coef + [0]*3)
    np.testing.assert_almost_equal(result, truth)

    # Check that leading zeros give the right answer
    result = galsim.utilities.horner(x, [0]*3 + coef)
    np.testing.assert_almost_equal(result, truth*x**3)

    # Check using a different dtype
    result = galsim.utilities.horner(x, coef, dtype=complex)
    np.testing.assert_almost_equal(result, truth)

    # Check that a single element coef gives the right answer
    result = galsim.utilities.horner([1,2,3], [17])
    np.testing.assert_almost_equal(result, 17)
    result = galsim.utilities.horner(x, [17])
    np.testing.assert_almost_equal(result, 17)
    result = galsim.utilities.horner([1,2,3], [17,0,0,0])
    np.testing.assert_almost_equal(result, 17)
    result = galsim.utilities.horner(x, [17,0,0,0])
    np.testing.assert_almost_equal(result, 17)
    result = galsim.utilities.horner([1,2,3], [0,0,0,0])
    np.testing.assert_almost_equal(result, 0)
    result = galsim.utilities.horner(x, [0,0,0,0])
    np.testing.assert_almost_equal(result, 0)

    # Check that x may be non-contiguous
    result = galsim.utilities.horner(x[::3], coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval(x[::3],coef))

    # Check that coef may be non-contiguous
    result = galsim.utilities.horner(x, coef[::-1])
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval(x,coef[::-1]))

    # Check odd length
    result = galsim.utilities.horner(x[:15], coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval(x[:15],coef))

    # Check unaligned array
    result = galsim.utilities.horner(x[1:], coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval(x[1:],coef))

    # Check length > 64
    xx = np.empty(2000)
    rng.generate(xx)
    result = galsim.utilities.horner(xx, coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval(xx,coef))

    # Check scalar x
    result = galsim.utilities.horner(3.9, coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval([3.9],coef))

    # Check invalid arguments
    with assert_raises(galsim.GalSimValueError):
        galsim.utilities.horner(x, [coef])

@timer
def test_horner2d():
    # Make a random 2d polynomial
    coef = [ [1.2332, 3.4324, 4.1231, -0.2342, 0.4242],
             [-0.2341, 1.4689, 3.5322, -1.0039, 0.02142],
             [4.02342, -2.2352, 0.1414, 2.9352, -1.3521] ]
    coef = np.array(coef)

    # Make a random list of values to test
    x = np.empty(20)
    y = np.empty(20)
    rng = galsim.UniformDeviate(1234)
    rng.generate(x)
    rng.generate(y)

    # Check against the direct calculation
    truth = coef[0,0] + coef[0,1]*y + coef[0,2]*y**2 + coef[0,3]*y**3 + coef[0,4]*y**4
    truth += (coef[1,0] + coef[1,1]*y + coef[1,2]*y**2 + coef[1,3]*y**3 + coef[1,4]*y**4)*x
    truth += (coef[2,0] + coef[2,1]*y + coef[2,2]*y**2 + coef[2,3]*y**3 + coef[2,4]*y**4)*x**2
    result = galsim.utilities.horner2d(x, y, coef)
    np.testing.assert_almost_equal(result, truth)

    # Also check against the (slower) numpy code
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x,y,coef))

    # Check that trailing zeros give the same answer
    result = galsim.utilities.horner2d(x, y, np.hstack([coef, np.zeros((3,1))]))
    np.testing.assert_almost_equal(result, truth)
    result = galsim.utilities.horner2d(x, y, np.hstack([coef, np.zeros((3,6))]))
    np.testing.assert_almost_equal(result, truth)
    result = galsim.utilities.horner2d(x, y, np.vstack([coef, np.zeros((1,5))]))
    np.testing.assert_almost_equal(result, truth)
    result = galsim.utilities.horner2d(x, y, np.vstack([coef, np.zeros((6,5))]))
    np.testing.assert_almost_equal(result, truth)

    # Check that leading zeros give the right answer
    result = galsim.utilities.horner2d(x, y, np.hstack([np.zeros((3,1)), coef]))
    np.testing.assert_almost_equal(result, truth*y)
    result = galsim.utilities.horner2d(x, y, np.hstack([np.zeros((3,6)), coef]))
    np.testing.assert_almost_equal(result, truth*y**6)
    result = galsim.utilities.horner2d(x, y, np.vstack([np.zeros((1,5)), coef]))
    np.testing.assert_almost_equal(result, truth*x)
    result = galsim.utilities.horner2d(x, y, np.vstack([np.zeros((6,5)), coef]))
    np.testing.assert_almost_equal(result, truth*x**6)

    # Check using a different dtype
    result = galsim.utilities.horner2d(x, y, coef, dtype=complex)
    np.testing.assert_almost_equal(result, truth)

    # Check that x,y may be non-contiguous
    result = galsim.utilities.horner2d(x[::3], y[:7], coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x[::3],y[:7],coef))
    result = galsim.utilities.horner2d(x[:7], y[::-3], coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x[:7],y[::-3],coef))

    # Check that coef may be non-contiguous
    result = galsim.utilities.horner2d(x, y, coef[:,::-1])
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x,y,coef[:,::-1]))
    result = galsim.utilities.horner2d(x, y, coef[::-1,:])
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x,y,coef[::-1,:]))

    # Check odd length
    result = galsim.utilities.horner2d(x[:15], y[:15], coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x[:15],y[:15],coef))

    # Check unaligned array
    result = galsim.utilities.horner2d(x[1:], y[1:], coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x[1:],y[1:],coef))
    result = galsim.utilities.horner2d(x[1:], y[:-1], coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x[1:],y[:-1],coef))
    result = galsim.utilities.horner2d(x[:-1], y[1:], coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x[:-1],y[1:],coef))

    # Check length > 64
    xx = np.empty(2000)
    yy = np.empty(2000)
    rng.generate(xx)
    rng.generate(yy)
    result = galsim.utilities.horner2d(xx, yy, coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(xx,yy,coef))

    # Check scalar x, y
    result = galsim.utilities.horner2d(3.9, 1.7, coef)
    np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d([3.9],[1.7],coef))


    # Check the triangle = True option
    coef = [ [1.2332, 3.4324, 4.1231],
             [-0.2341, 1.4689, 0.],
             [4.02342, 0., 0.] ]
    coef = np.array(coef)

    # Check against the direct calculation
    truth = coef[0,0] + coef[0,1]*y + coef[0,2]*y**2
    truth += coef[1,0]*x + coef[1,1]*x*y
    truth += coef[2,0]*x**2
    result = galsim.utilities.horner2d(x, y, coef)
    np.testing.assert_almost_equal(result, truth)
    result = galsim.utilities.horner2d(x, y, coef, triangle=True)
    np.testing.assert_almost_equal(result, truth)

    # Check using a different dtype
    result = galsim.utilities.horner2d(x, y, coef, dtype=complex, triangle=True)
    np.testing.assert_almost_equal(result, truth)

    # Check invalid arguments
    with assert_raises(galsim.GalSimValueError):
        galsim.utilities.horner2d(x, y, [coef])
    with assert_raises(galsim.GalSimValueError):
        galsim.utilities.horner2d(x, y, coef[0])
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        galsim.utilities.horner2d(x, y, coef[0:1], triangle=True)
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        galsim.utilities.horner2d(x, y[:10], coef)


if __name__ == "__main__":
    test_pos()
    test_bounds()
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
    test_nCr()
    test_horner()
    test_horner2d()
