# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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


import os
import numpy as np
import time
from unittest import mock

import galsim
from galsim_test_helpers import *

path, filename = os.path.split(__file__) # Get the path to this file for use below...

TESTDIR=os.path.join(path, "table_comparison_files")

DECIMAL = 14 # Make sure output agrees at 14 decimal places or better

# Some arbitrary args to use for the test:
args1 = range(7)  # Evenly spaced
vals1 = [ x**2 for x in args1 ]
testargs1 = [ 0.1, 0.8, 2.3, 3, 5.6, 5.9 ] # < 0 or > 7 is invalid

args2 = [ 0.7, 3.3, 14.1, 15.6, 29, 34.1, 42.5 ]  # Not evenly spaced
vals2 = [ np.sin(x*np.pi/180) for x in args2 ]
testargs2 = [ 1.1, 10.8, 12.3, 15.6, 25.6, 41.9 ] # < 0.7 or > 42.5 is invalid

interps = [ 'linear', 'spline', 'floor', 'ceil', 'nearest' ]


@timer
def test_table():
    """Test the spline tabulation of the k space Cubic interpolant.

    Compares interpolated values a LookupTable that were created using a previous version of
    the code (commit: e267f058351899f1f820adf4d6ab409d5b2605d5), using the
    script devutils/external/make_table_testarrays.py
    """
    for interp in interps:
        table1 = galsim.LookupTable(x=args1,f=vals1,interpolant=interp)
        testvals1 = [ table1(x) for x in testargs1 ]
        assert len(table1) == len(args1)

        np.testing.assert_array_equal(table1.getArgs(), args1)
        np.testing.assert_array_equal(table1.getVals(), vals1)
        assert table1.interpolant == interp
        assert table1.isLogX() == False
        assert table1.isLogF() == False

        # The 4th item is in the args list, so it should be exactly the same as the
        # corresponding item in the vals list.
        np.testing.assert_almost_equal(testvals1[3], vals1[3], DECIMAL,
                err_msg="Interpolated value for exact arg entry does not match val entry")

        # Compare the results in testvals1 with the results if we reshape the input array to be
        # 2-dimensional.
        np.testing.assert_array_almost_equal(
            np.array(testvals1).reshape((2,3)), table1(np.array(testargs1).reshape((2,3))),
            DECIMAL,
            err_msg="Interpolated values do not match when input array shape changes")

        if interp != 'nearest':
            # Do a full regression test based on a version of the code thought to be working.
            ref1 = np.loadtxt(os.path.join(TESTDIR, 'table_test1_%s.txt'%interp))
            np.testing.assert_array_almost_equal(ref1, testvals1, DECIMAL,
                    err_msg="Interpolated values from LookupTable do not match saved "+
                    "data for evenly-spaced args, with interpolant %s."%interp)

            # Same thing, but now for args that are not evenly spaced.
            # (The Table class uses a different algorithm when the arguments are evenly spaced
            #  than when they are not.)
            table2 = galsim.LookupTable(x=args2,f=vals2,interpolant=interp)
            testvals2 = [ table2(x) for x in testargs2 ]

            np.testing.assert_almost_equal(testvals2[3], vals2[3], DECIMAL,
                    err_msg="Interpolated value for exact arg entry does not match val entry")
            ref2 = np.loadtxt(os.path.join(TESTDIR, 'table_test2_%s.txt'%interp))
            np.testing.assert_array_almost_equal(ref2, testvals2, DECIMAL,
                    err_msg="Interpolated values from LookupTable do not match saved "+
                    "data for non-evenly-spaced args, with interpolant %s."%interp)

        # Check that out of bounds arguments, or ones with some crazy shape, raise an exception:
        assert_raises(ValueError,table1,args1[0]-0.01)
        assert_raises(ValueError,table1,args1[-1]+0.01)
        assert_raises(ValueError,table2,args2[0]-0.01)
        assert_raises(ValueError,table2,args2[-1]+0.01)

        # These shouldn't raise any exception:
        table1(args1[0]+0.01)
        table1(args1[-1]-0.01)
        table2(args2[0]+0.01)
        table2(args2[-1]-0.01)
        table1(np.zeros((3,3))+args1[0]+0.01)
        table1(np.zeros(3)+args1[0]+0.01)
        table1((args1[0]+0.01,args1[0]+0.01))
        table1([args1[0]+0.01,args1[0]+0.01])
        # Check 2d arrays (square)
        table1(np.zeros((3,3))+args1[0])
        # Check 2d arrays (non-square)
        table1(np.array(testargs1).reshape((2,3)))

        # Check picklability
        check_pickle(table1, lambda x: (tuple(x.getArgs()), tuple(x.getVals()), x.getInterp()))
        check_pickle(table2, lambda x: (tuple(x.getArgs()), tuple(x.getVals()), x.getInterp()))
        check_pickle(table1)
        check_pickle(table2)

        table3 = galsim._LookupTable(x=args1,f=vals1,interpolant=interp)
        assert table3 == table1

    # Spline interpolation with only 2 points is just linear.
    table1 = galsim.LookupTable([1,5],[9,3],'linear')
    table2 = galsim.LookupTable([1,5],[9,3],'spline')
    assert np.isclose(table1(3), 6)
    assert np.isclose(table2(3), 6)
    assert table1.interpolant == 'linear'
    assert table2.interpolant == 'spline'

    assert_raises(ValueError, galsim.LookupTable, x=args1, f=vals1, interpolant='invalid')
    assert_raises(ValueError, galsim.LookupTable, x=[1], f=[1], interpolant='linear')
    assert_raises(ValueError, galsim.LookupTable, x=[], f=[])
    assert_raises(ValueError, galsim.LookupTable, x=[1,1,1], f=[1,2,3])
    assert_raises(ValueError, galsim.LookupTable, x=[0,1,2], f=[1,2,3], x_log=True)
    assert_raises(ValueError, galsim.LookupTable, x=[-1,0,1], f=[1,2,3], x_log=True)
    assert_raises(ValueError, galsim.LookupTable, x=[0,1,2], f=[0,1,2], f_log=True)
    assert_raises(ValueError, galsim.LookupTable, x=[0,1,2], f=[2,-1,2], f_log=True)
    assert_raises(ValueError, galsim.LookupTable, x=args2, f=vals2, interpolant=galsim.Linear())
    assert_raises(ValueError, galsim.LookupTable, x=args1, f=vals1, interpolant=galsim.Linear(),
                  x_log=True)


@timer
def test_init():
    """Some simple tests of LookupTable initialization."""

    # Make sure nothing bad happens when we try to read in a stored power spectrum and assume
    # we can use the default interpolant (spline).
    tab_ps = galsim.LookupTable.from_file('../examples/data/cosmo-fid.zmed1.00_smoothed.out')
    check_pickle(tab_ps)

    # Check for bad inputs
    assert_raises(TypeError, galsim.LookupTable, x='foo')
    assert_raises(TypeError, galsim.LookupTable)
    assert_raises(TypeError, galsim.LookupTable, x=tab_ps.x)
    assert_raises(TypeError, galsim.LookupTable, f=tab_ps.f)
    assert_raises(ValueError, galsim.LookupTable, x=tab_ps.x, f=tab_ps.f, interpolant='foo')

    with assert_raises(ValueError):
        # This file is for a different purpose. It has one column, which is invalid here.
        galsim.LookupTable.from_file('table_comparison_files/table_test1_spline.txt')

    # The default reader uses pandas, but numpy is used if pandas not available.
    with mock.patch.dict(sys.modules, {'pandas':None}):
        tab_ps2 = galsim.LookupTable.from_file('../examples/data/cosmo-fid.zmed1.00_smoothed.out')
        # Not precisely equal. But close enough.
        np.testing.assert_allclose(tab_ps2.x, tab_ps.x, rtol=1.e-12, atol=1.e-12)
        np.testing.assert_allclose(tab_ps2.f, tab_ps.f, rtol=1.e-12, atol=1.e-12)

    # We expect that people will normally initialize with sorted x values, but that's not actually
    # required.  Check that we get the smae thing with a different order.
    index = np.arange(len(tab_ps.x))
    np.random.shuffle(index)
    x1 = tab_ps.x[index]
    f1 = tab_ps.f[index]
    tab_ps3 = galsim.LookupTable(x1, f1)
    assert tab_ps3 == tab_ps

@timer
def test_log():
    """Some simple tests of interpolation using logs."""
    # Set up some test vectors that are strictly positive, and others that are negative.
    x = 0.01*np.arange(1000)+0.01
    y = 1.*x
    x_neg = -1.*x
    y_neg = 1.*x_neg

    # Check that interpolation agrees for the positive ones when using log interpolation (for some
    # reasonable tolerance).
    tab_1 = galsim.LookupTable(x=x, f=y)
    tab_2 = galsim.LookupTable(x=x, f=y, x_log=True, f_log=True)
    tab_3 = galsim.LookupTable(x=x, f=y, x_log=True)
    tab_4 = galsim.LookupTable(x=x, f=y, f_log=True)
    test_x_vals = [2.641, 3.985, 8.123125]
    for test_val in test_x_vals:
        result_1 = tab_1(test_val)
        result_2 = tab_2(test_val)
        result_3 = tab_3(test_val)
        result_4 = tab_4(test_val)
        print(result_1, result_2, result_3, result_4)
        np.testing.assert_almost_equal(
            result_2, result_1, decimal=3,
            err_msg='Disagreement when interpolating in log(f) and log(x)')
        np.testing.assert_almost_equal(
            result_3, result_1, decimal=3,
            err_msg='Disagreement when interpolating in log(x)')
        np.testing.assert_almost_equal(
            result_4, result_1, decimal=3,
            err_msg='Disagreement when interpolating in log(f)')

    # Before getting to the range check, numpy will issue a warning about
    # "invalid value encountered in log".
    # Seems like ok additional information for end users.  But we want to ignore that here.
    with assert_raises(galsim.GalSimRangeError):
        with assert_warns(RuntimeWarning):
            tab_2(-1)
    with assert_raises(galsim.GalSimRangeError):
        with assert_warns(RuntimeWarning):
            tab_3(-1)
    with assert_raises(galsim.GalSimRangeError):
        with assert_warns(RuntimeWarning):
            tab_2(x_neg)
    with assert_raises(galsim.GalSimRangeError):
        with assert_warns(RuntimeWarning):
            tab_3(x_neg)

    # Check picklability
    check_pickle(tab_1)
    check_pickle(tab_2)
    check_pickle(tab_3)
    check_pickle(tab_4)

    # Check storage of args and vals for log vs. linear, which should be the same to high precision.
    np.testing.assert_array_almost_equal(tab_1.getArgs(), tab_3.getArgs(), decimal=12,
                                         err_msg='Args differ for linear vs. log storage')
    np.testing.assert_array_almost_equal(tab_1.getVals(), tab_4.getVals(), decimal=12,
                                         err_msg='Vals differ for linear vs. log storage')
    # Check other properties
    assert not tab_1.x_log
    assert not tab_1.f_log
    assert tab_2.x_log
    assert tab_2.f_log
    assert tab_3.x_log
    assert not tab_3.f_log
    assert not tab_1.isLogX()
    assert not tab_1.isLogF()
    assert tab_2.isLogX()
    assert tab_2.isLogF()
    assert tab_3.isLogX()
    assert not tab_3.isLogF()

    # Check that an appropriate exception is thrown when trying to do interpolation using negative
    # ones.
    assert_raises(ValueError, galsim.LookupTable, x=x_neg, f=y_neg, x_log=True)
    assert_raises(ValueError, galsim.LookupTable, x=x_neg, f=y_neg, f_log=True)
    assert_raises(ValueError, galsim.LookupTable, x=x_neg, f=y_neg, x_log=True, f_log=True)

    # Check that doing log transform explicitly matches expected behavior of x_log and f_log.
    expx = np.exp(x)
    expy = np.exp(y)
    for interpolant in ['linear', 'spline', galsim.Quintic()]:
        tab_1 = galsim.LookupTable(x, y, interpolant=interpolant)
        tab_2 = galsim.LookupTable(expx, y, x_log=True, interpolant=interpolant)
        tab_3 = galsim.LookupTable(x, expy, f_log=True, interpolant=interpolant)
        tab_4 = galsim.LookupTable(expx, expy, x_log=True, f_log=True, interpolant=interpolant)
        tab_5 = galsim._LookupTable(expx, expy, x_log=True, f_log=True, interpolant=interpolant)
        for test_val in test_x_vals:
            result_1 = tab_1(test_val)
            result_2 = tab_2(np.exp(test_val))
            result_3 = np.log(tab_3(test_val))
            result_4 = np.log(tab_4(np.exp(test_val)))
            result_5 = np.log(tab_5(np.exp(test_val)))
            np.testing.assert_almost_equal(
                result_2, result_1, decimal=10,
                err_msg='Disagreement when interpolating in log(x)')
            np.testing.assert_almost_equal(
                result_3, result_1, decimal=10,
                err_msg='Disagreement when interpolating in log(f)')
            np.testing.assert_almost_equal(
                result_4, result_1, decimal=10,
                err_msg='Disagreement when interpolating in log(f) and log(x)')
            np.testing.assert_almost_equal(
                result_5, result_1, decimal=10,
                err_msg='Disagreement when interpolating in log(f) and log(x) with _LookupTable')

    # Verify for exception when using x_log with non-equal-spaced galsim.Interpolant
    galsim.LookupTable(x, y, x_log=True, interpolant='linear') # works fine
    assert_raises(ValueError, galsim.LookupTable, x, y, x_log=True, interpolant=galsim.Linear())


@timer
def test_from_func():
    """Test the LookupTable.from_func factory function"""
    x_min = 2
    x_max = 200

    # Linear interpolation
    x1 = np.linspace(x_min, x_max, 2000)
    f1 = [x**3 for x in x1]
    tab1 = galsim.LookupTable(x1, f1, interpolant='linear')
    tab2 = galsim.LookupTable.from_func(lambda x:x**3, x_min, x_max, interpolant='linear')
    print('tab1 = ',tab1, tab1(10))
    print('tab2 = ',tab2, tab2(10))

    # Spline interpolation
    tab3 = galsim.LookupTable(x1, f1)
    tab4 = galsim.LookupTable.from_func(lambda x:x**3, x_min, x_max)
    print('tab3 = ',tab3, tab3(10))
    print('tab4 = ',tab4, tab4(10))

    # Log interpolation
    x5 = np.exp(np.linspace(np.log(x_min), np.log(x_max), 2000))
    f5 = [x**3 for x in x5]
    tab5 = galsim.LookupTable(x5, f5, x_log=True, f_log=True)
    tab6 = galsim.LookupTable.from_func(lambda x:x**3, x_min, x_max, x_log=True, f_log=True)
    print('tab5 = ',tab5, tab5(10))
    print('tab6 = ',tab6, tab6(10))

    test_x_vals = [2.641, 39.85, 81.23125]
    for x in test_x_vals:
        truth = x**3
        f1 = tab1(x)
        f2 = tab2(x)
        f3 = tab3(x)
        f4 = tab4(x)
        f5 = tab5(x)
        f6 = tab6(x)
        print(truth, f1, f2, f3, f4, f5, f6)
        np.testing.assert_almost_equal(f1/truth, 1.0, decimal=2)
        np.testing.assert_almost_equal(f2/truth, 1.0, 2,
                                       "LookupTable.from_func (linear) gave wrong answer")
        np.testing.assert_almost_equal(f3/truth, 1.0, decimal=6)
        np.testing.assert_almost_equal(f4/truth, 1.0, 6,
                                       "LookupTable.from_func (spline) gave wrong answer")
        np.testing.assert_almost_equal(f5/truth, 1.0, decimal=11)
        np.testing.assert_almost_equal(f6/truth, 1.0, 11,
                                       "LookupTable.from_func (log-log) gave wrong answer")
    check_pickle(tab2)
    check_pickle(tab4)
    check_pickle(tab6)


@timer
def test_roundoff():
    table1 = galsim.LookupTable([1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10])
    # These should work without raising an exception
    np.testing.assert_almost_equal(table1(1.0 - 1.e-7), 1.0, decimal=6)
    np.testing.assert_almost_equal(table1(10.0 + 1.e-7), 10.0, decimal=6)
    assert_raises(ValueError, table1, 1.0-1.e5)
    assert_raises(ValueError, table1, 10.0+1.e5)


@timer
def test_table_GSInterp():
    def f(x_):
        return x_ + 2*x_*x_ + 3*np.sin(2*x_)

    x = np.linspace(0.1, 3.3, 25)
    y = np.arange(20) # Need something big enough to avoid having the interpolant fall off the edge
    xx, yy = np.meshgrid(x, y)

    interpolants = ['lanczos3', 'lanczos3F', 'lanczos7', 'sinc', 'quintic']

    for interpolant in interpolants:
        z = f(xx)

        tab = galsim.LookupTable(x, z[0,:], interpolant=interpolant)
        check_pickle(tab)

        # Use InterpolatedImage to validate
        wcs = galsim.JacobianWCS(
            (max(x) - min(x))/(len(x)-1),
            0, 0,
            (max(y) - min(y))/(len(y)-1),
        )
        img = galsim.Image(z, wcs=wcs)
        ii = galsim.InterpolatedImage(
            img,
            use_true_center=True,
            offset=(galsim.PositionD(img.xmin,img.ymin)-img.true_center),
            x_interpolant=interpolant,
            normalization='sb',
            calculate_maxk=False, calculate_stepk=False
        ).shift(min(x), min(y))

        # Check single value functionality.
        x1 = 2.3
        np.testing.assert_allclose(tab(x1), ii.xValue(x1, 10), atol=1e-10, rtol=0)

        # Check vectorized output
        newx = np.linspace(0.2, 3.1, 15)
        np.testing.assert_allclose(
            tab(newx),
            np.array([ii.xValue(x_, 10) for x_ in newx]),
            atol=1e-10, rtol=0
        )

        tab2 = galsim._LookupTable(x, z[0,:], interpolant=interpolant)
        assert tab2 == tab


@timer
def test_table2d():
    """Check LookupTable2D functionality.
    """
    from scipy.interpolate import RectBivariateSpline

    def f(x_, y_):
        return np.sin(x_) * np.cos(y_) + x_

    x = np.linspace(0.1, 3.3, 25)
    y = np.linspace(0.2, 10.4, 75)
    xx, yy = np.meshgrid(x, y)
    z = f(xx, yy)

    tab2d = galsim.LookupTable2D(x, y, z)
    check_pickle(tab2d)

    np.testing.assert_array_equal(tab2d.getXArgs(), x)
    np.testing.assert_array_equal(tab2d.getYArgs(), y)
    np.testing.assert_array_equal(tab2d.getVals(), z)
    assert tab2d.interpolant == 'linear'
    assert tab2d.edge_mode == 'raise'

    newx = np.linspace(0.2, 3.1, 45)
    newy = np.linspace(0.3, 10.1, 85)
    newxx, newyy = np.meshgrid(newx, newy)

    # Compare different ways of evaluating Table2D
    ref = tab2d(newxx, newyy)
    np.testing.assert_array_almost_equal(ref, tab2d(newx, newy, grid=True))
    np.testing.assert_array_almost_equal(ref, np.array([[tab2d(x0, y0)
                                                         for x0 in newx]
                                                        for y0 in newy]))

    scitab2d = RectBivariateSpline(x, y, z.T, kx=1, ky=1)
    np.testing.assert_array_almost_equal(ref, scitab2d(newx, newy).T)

    # Try using linear GSInterp
    tab2d2 = galsim.LookupTable2D(x, y, z, interpolant=galsim.Linear())
    np.testing.assert_array_almost_equal(tab2d(newxx, newyy), tab2d2(newxx, newyy))
    np.testing.assert_array_almost_equal(tab2d(newxx.T, newyy.T), tab2d2(newxx.T, newyy.T))

    # Try again using Nearest()
    tab2d2 = galsim.LookupTable2D(x, y, z, interpolant=galsim.Nearest())
    tab2d3 = galsim.LookupTable2D(x, y, z, interpolant='nearest')
    np.testing.assert_array_almost_equal(tab2d2(newxx, newyy), tab2d3(newxx, newyy))
    np.testing.assert_array_almost_equal(tab2d2(newxx.T, newyy.T), tab2d3(newxx.T, newyy.T))
    # Make sure we exercise the special case in T2DInterpolant2D::interp
    np.testing.assert_array_almost_equal(tab2d2(0.3, 0.4), tab2d3(0.3, 0.4))
    np.testing.assert_array_almost_equal(tab2d2(0.3, 0.4), tab2d3(0.3, 0.4))


    # Test non-equally-spaced table.
    x = np.delete(x, 10)
    y = np.delete(y, 10)
    xx, yy = np.meshgrid(x, y)
    z = f(xx, yy)
    tab2d = galsim.LookupTable2D(x, y, z)
    ref = tab2d(newxx, newyy)
    np.testing.assert_array_almost_equal(ref, tab2d(newx, newy, grid=True))
    np.testing.assert_array_almost_equal(ref, np.array([[tab2d(x0, y0)
                                                         for x0 in newx]
                                                        for y0 in newy]))
    scitab2d = RectBivariateSpline(x, y, z.T, kx=1, ky=1)
    np.testing.assert_array_almost_equal(ref, scitab2d(newx, newy).T)

    # Using a galsim.Interpolant should raise an exception if x/y are not equal spaced.
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        tab2d2 = galsim.LookupTable2D(x, y, z, interpolant=galsim.Linear())

    # Try a simpler interpolation function.  We should be able to interpolate a (bi-)linear function
    # exactly with a linear interpolant.
    def f(x_, y_):
        return 2*x_ + 3*y_

    z = f(xx, yy)
    tab2d = galsim.LookupTable2D(x, y, z)

    np.testing.assert_array_almost_equal(f(newxx, newyy), tab2d(newxx, newyy))
    np.testing.assert_array_almost_equal(f(newxx, newyy), np.array([[tab2d(x0, y0)
                                                                     for x0 in newx]
                                                                    for y0 in newy]))

    # Test edge exception
    with assert_raises(ValueError):
        tab2d(1e6, 1e6)
    with assert_raises(ValueError):
        tab2d(np.array([1e5, 1e6]), np.array([1e5, 1e6]))
    with assert_raises(ValueError):
        tab2d(np.array([1e5, 1e6]), np.array([1e5, 1e6]), grid=True)
    with assert_raises(ValueError):
        tab2d.gradient(1e6, 1e6)
    with assert_raises(ValueError):
        tab2d.gradient(np.array([1e5, 1e6]), np.array([1e5, 1e6]))
    with assert_raises(ValueError):
        tab2d.gradient(np.array([1e5, 1e6]), np.array([1e5, 1e6]), grid=True)
    with assert_raises(galsim.GalSimError):
        galsim.utilities.find_out_of_bounds_position(
            np.array([1, 3]), np.array([2, 4]), galsim.BoundsD(0,5,0,5))
    with assert_raises(galsim.GalSimError):
        galsim.utilities.find_out_of_bounds_position(
            np.array([1, 3]), np.array([2, 4]), galsim.BoundsD(0,5,0,5), grid=True)

    # Check warning mode
    tab2dw = galsim.LookupTable2D(x, y, z, edge_mode='warn', constant=1)
    with assert_warns(galsim.GalSimWarning):
        assert tab2dw(1e6, 1e6) == 1
    with assert_warns(galsim.GalSimWarning):
        np.testing.assert_array_equal(
            tab2dw(np.array([1e5, 1e6]), np.array([1e5, 1e6])),
            np.array([1.0, 1.0])
        )
    with assert_warns(galsim.GalSimWarning):
        np.testing.assert_array_equal(
            tab2dw(np.array([1e5, 1e6]), np.array([1e5, 1e6]), grid=True),
            np.array([[1.0, 1.0], [1.0, 1.0]])
        )
    with assert_warns(galsim.GalSimWarning):
        assert tab2dw.gradient(1e6, 1e6) == (0.0, 0.0)
    with assert_warns(galsim.GalSimWarning):
        np.testing.assert_array_equal(
            tab2dw.gradient(np.array([1e5, 1e6]), np.array([1e5, 1e6])),
            np.zeros((2, 2), dtype=float)
        )
    with assert_warns(galsim.GalSimWarning):
        np.testing.assert_array_equal(
            tab2dw.gradient(np.array([1e5, 1e6]), np.array([1e5, 1e6]), grid=True),
            np.zeros((2, 2, 2), dtype=float)
        )

    # But doesn't warn if in bounds
    tab2dw(1.0, 1.0)
    tab2dw(np.array([1.0]), np.array([1.0]))
    tab2dw(np.array([1.0]), np.array([1.0]), grid=True)
    tab2dw.gradient(1.0, 1.0)
    tab2dw.gradient(np.array([1.0]), np.array([1.0]))

    # Test edge wrapping
    # Check that can't construct table with edge-wrapping if edges don't match
    with assert_raises(ValueError):
        galsim.LookupTable2D(x, y, z, edge_mode='wrap')

    # Extend edges and make vals match
    x = np.append(x, x[-1] + (x[-1]-x[-2]))
    y = np.append(y, y[-1] + (y[-1]-y[-2]))
    z = np.pad(z,[(0,1), (0,1)], mode='wrap')
    tab2d = galsim.LookupTable2D(x, y, z, edge_mode='wrap')

    np.testing.assert_array_almost_equal(tab2d(newxx, newyy), tab2d(newxx+3*(x[-1]-x[0]), newyy))
    np.testing.assert_array_almost_equal(
        tab2d(newx, newy, grid=True), tab2d(newxx+3*(x[-1]-x[0]), newyy))
    np.testing.assert_array_almost_equal(tab2d(newxx, newyy), tab2d(newxx, newyy+13*(y[-1]-y[0])))
    np.testing.assert_array_almost_equal(
        tab2d(newxx, newyy), tab2d(newx, newy+13*(y[-1]-y[0]), grid=True))

    # Test edge_mode='constant'
    tab2d = galsim.LookupTable2D(x, y, z, edge_mode='constant', constant=42)
    assert type(tab2d(x[0]-1, y[0]-1)) in [float, np.float64]
    assert tab2d(x[0]-1, y[0]-1) == 42.0
    # One in-bounds, one out-of-bounds
    np.testing.assert_array_almost_equal(tab2d([x[0], x[0]-1], [y[0], y[0]-1]),
                                         [tab2d(x[0], y[0]), 42.0])


    # Test floor/ceil/nearest interpolant
    x = np.arange(5)
    y = np.arange(5)
    z = x + y[:, np.newaxis]
    tab2d = galsim.LookupTable2D(x, y, z, interpolant='ceil')
    assert tab2d(2.4, 3.6) == 3+4, "Ceil interpolant failed."
    tab2d = galsim.LookupTable2D(x, y, z, interpolant='floor')
    assert tab2d(2.4, 3.6) == 2+3, "Floor interpolant failed."
    tab2d = galsim.LookupTable2D(x, y, z, interpolant='nearest')
    assert tab2d(2.4, 3.6) == 2+4, "Nearest interpolant failed."
    tab2d = galsim.LookupTable2D(x, y, z, interpolant=galsim.Nearest())
    assert tab2d(2.4, 3.6) == 2+4, "Nearest interpolant failed."

    assert_raises(ValueError, galsim.LookupTable2D, x, y, z, interpolant='invalid')
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z, edge_mode='invalid')
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z[:-1,:-1])

    # Test that x,y arrays need to be strictly increasing.
    x[0] = x[1]
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z)
    x[0] = x[1]+1
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z)
    x[0] = x[1]-1
    y[0] = y[1]
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z)
    y[0] = y[1]+1
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z)

    # Test that x,y arrays are larger than interpolant
    x = y = np.arange(2)
    z = x[:,None]+y
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z, interpolant='spline', edge_mode='wrap')

    # Check dfdx input
    x = y = np.arange(20)
    z = x[:,None]+y
    # Not using interpolant='spline'
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z, dfdx=z)
    # Only specifying one derivative
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z, dfdx=z, interpolant='spline')
    # Derivative is wrong shape
    assert_raises(ValueError, galsim.LookupTable2D, x, y, z, dfdx=z, dfdy=z, d2fdxdy=z[::2],
                  interpolant='spline')

    # Check private shortcut instantiation
    new_tab2d = galsim.table._LookupTable2D(
        tab2d.x, tab2d.y, tab2d.f,
        tab2d.interpolant, tab2d.edge_mode,
        tab2d.constant, tab2d.dfdx, tab2d.dfdy, tab2d.d2fdxdy
    )
    assert tab2d == new_tab2d
    assert tab2d(2.4, 3.6) == new_tab2d(2.4, 3.6)


@timer
def test_table2d_gradient():
    """Check LookupTable2D gradient function
    """
    # Same function as the above test
    def f(x_, y_):
        return np.sin(x_) * np.cos(y_) + x_
    # The gradient is analytic for this:
    def dfdx(x_, y_):
        return np.cos(x_) * np.cos(y_) + 1.
    def dfdy(x_, y_):
        return -np.sin(x_) * np.sin(y_)

    x = np.linspace(0.1, 3.3, 250)
    y = np.linspace(0.2, 10.4, 750)
    xx, yy = np.meshgrid(x, y)
    z = f(xx, yy)

    tab2d = galsim.LookupTable2D(x, y, z)

    newx = np.linspace(0.2, 3.1, 45)
    newy = np.linspace(0.3, 10.1, 85)
    newxx, newyy = np.meshgrid(newx, newy)

    # Check single value functionality.
    x1,y1 = 1.1, 4.9
    np.testing.assert_almost_equal(tab2d.gradient(x1,y1), (dfdx(x1,y1), dfdy(x1,y1)), decimal=2)

    # Check that the gradient function comes out close to the analytic derivatives.
    ref_dfdx = dfdx(newxx, newyy)
    ref_dfdy = dfdy(newxx, newyy)
    test_dfdx, test_dfdy = tab2d.gradient(newxx, newyy)
    np.testing.assert_almost_equal(test_dfdx, ref_dfdx, decimal=2)
    np.testing.assert_almost_equal(test_dfdy, ref_dfdy, decimal=2)
    test2_dfdx, test2_dfdy = tab2d.gradient(newx, newy, grid=True)
    np.testing.assert_almost_equal(test2_dfdx, ref_dfdx, decimal=2)
    np.testing.assert_almost_equal(test2_dfdy, ref_dfdy, decimal=2)

    # Check edge wrapping
    tab2d = galsim.LookupTable2D(x, y, z, edge_mode='wrap')

    test_dfdx, test_dfdy = tab2d.gradient(newxx+13*tab2d.xperiod, newyy)
    test2_dfdx, test2_dfdy = tab2d.gradient(newx+13*tab2d.xperiod, newy, grid=True)
    np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx, decimal=2)
    np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy, decimal=2)
    np.testing.assert_array_almost_equal(ref_dfdx, test2_dfdx, decimal=2)
    np.testing.assert_array_almost_equal(ref_dfdy, test2_dfdy, decimal=2)

    test_dfdx, test_dfdy = tab2d.gradient(newxx, newyy+12*tab2d.yperiod)
    np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx, decimal=2)
    np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy, decimal=2)

    # Test single value:
    np.testing.assert_almost_equal(tab2d.gradient(x1, y1), tab2d.gradient(x1-9*tab2d.xperiod, y1))
    np.testing.assert_almost_equal(tab2d.gradient(x1, y1), tab2d.gradient(x1, y1-7*tab2d.yperiod))


    # Check constant edge_mode
    tab2d = galsim.LookupTable2D(x, y, z, edge_mode='constant')

    # Should work the same inside original boundary
    test_dfdx, test_dfdy = tab2d.gradient(newxx, newyy)
    test2_dfdx, test2_dfdy = tab2d.gradient(newx, newy, grid=True)
    np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx, decimal=2)
    np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy, decimal=2)
    np.testing.assert_array_almost_equal(ref_dfdx, test2_dfdx, decimal=2)
    np.testing.assert_array_almost_equal(ref_dfdy, test2_dfdy, decimal=2)

    # Should come out zero outside original boundary
    test_dfdx, test_dfdy = tab2d.gradient(newxx+2*np.max(newxx), newyy+2*np.max(newyy))
    test2_dfdx, test2_dfdy = tab2d.gradient(newx+2*np.max(newx), newy+2*np.max(newy), grid=True)
    np.testing.assert_array_equal(0.0, test_dfdx)
    np.testing.assert_array_equal(0.0, test_dfdy)
    np.testing.assert_array_equal(0.0, test2_dfdx)
    np.testing.assert_array_equal(0.0, test2_dfdy)

    # Test single value
    np.testing.assert_almost_equal(tab2d.gradient(x1, y1), (dfdx(x1,y1), dfdy(x1, y1)), decimal=2)
    np.testing.assert_equal(tab2d.gradient(x1+2*np.max(newxx), y1), (0.0, 0.0))


    # Try a simpler interpolation function.  Derivatives should be exact.
    def f(x_, y_):
        return 2*x_ + 3*y_ - 4*x_*y_

    z = f(xx, yy)

    tab2d = galsim.LookupTable2D(x, y, z)
    test_dfdx, test_dfdy = tab2d.gradient(newxx, newyy)
    test2_dfdx, test2_dfdy = tab2d.gradient(newx, newy, grid=True)
    np.testing.assert_almost_equal(test_dfdx, 2.-4*newyy, decimal=7)
    np.testing.assert_almost_equal(test_dfdy, 3.-4*newxx, decimal=7)
    np.testing.assert_almost_equal(test2_dfdx, 2.-4*newyy, decimal=7)
    np.testing.assert_almost_equal(test2_dfdy, 3.-4*newxx, decimal=7)

    # Check single value functionality.
    np.testing.assert_almost_equal(tab2d.gradient(x1,y1), (2.-4*y1, 3.-4*x1))

    # Check edge wrapping
    tab2d = galsim.LookupTable2D(x, y, z, edge_mode='wrap')

    ref_dfdx, ref_dfdy = tab2d.gradient(newxx, newyy)
    test_dfdx, test_dfdy = tab2d.gradient(newxx+3*tab2d.xperiod, newyy)
    test2_dfdx, test2_dfdy = tab2d.gradient(newx+3*tab2d.xperiod, newy, grid=True)
    np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx)
    np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy)
    np.testing.assert_array_almost_equal(ref_dfdx, test2_dfdx)
    np.testing.assert_array_almost_equal(ref_dfdy, test2_dfdy)

    test_dfdx, test_dfdy = tab2d.gradient(newxx, newyy+13*tab2d.yperiod)
    np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx)
    np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy)

    # Test single value
    np.testing.assert_almost_equal(tab2d.gradient(x1, y1), (2-4*y1, 3-4*x1))
    np.testing.assert_almost_equal(tab2d.gradient(x1+2*tab2d.xperiod, y1), (2-4*y1, 3-4*x1))

    # Test mix of inside and outside original boundary
    test_dfdx, test_dfdy = tab2d.gradient(np.dstack([newxx, newxx+3*tab2d.xperiod]),
                                          np.dstack([newyy, newyy]))

    np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx[:,:,0])
    np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy[:,:,0])
    np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx[:,:,1])
    np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy[:,:,1])

    # Check that edge_mode='constant' behaves as expected.

    # Should work the same inside original boundary
    tab2d = galsim.LookupTable2D(x, y, z, edge_mode='constant')
    test_dfdx, test_dfdy = tab2d.gradient(newxx, newyy)
    np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx)
    np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy)

    # Should come out zero outside original boundary
    test_dfdx, test_dfdy = tab2d.gradient(newxx+2*np.max(newxx), newyy)
    np.testing.assert_array_equal(0.0, test_dfdx)
    np.testing.assert_array_equal(0.0, test_dfdy)

    # Test single value
    np.testing.assert_almost_equal(tab2d.gradient(x1, y1), (2-4*y1, 3-4*x1))
    np.testing.assert_equal(tab2d.gradient(x1+2*np.max(newxx), y1), (0.0, 0.0))

    # Test mix of inside and outside original boundary
    test_dfdx, test_dfdy = tab2d.gradient(np.dstack([newxx, newxx+2*np.max(newxx)]),
                                          np.dstack([newyy, newyy]))

    np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx[:,:,0])
    np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy[:,:,0])
    np.testing.assert_array_equal(0.0, test_dfdx[:,:,1])
    np.testing.assert_array_equal(0.0, test_dfdy[:,:,1])


@timer
def test_table2d_cubic():
    # A few functions that should be exactly interpolatable with bicubic
    # interpolation
    def f1(x_, y_):
        return 2*x_ + 3*y_
    def df1dx(x_, y_):
        return np.ones_like(x_)*2
    def df1dy(x_, y_):
        return np.ones_like(x_)*3

    def f2(x_, y_):
        return 2*x_*x_
    def df2dx(x_, y_):
        return 4*x_
    def df2dy(x_, y_):
        return np.zeros_like(x_)

    def f3(x_, y_):
        return 2*y_*y_
    def df3dx(x_, y_):
        return np.zeros_like(x_)
    def df3dy(x_, y_):
        return 4*y_

    def f4(x_, y_):
        return 2*y_*y_ + 3*x_*x_
    def df4dx(x_, y_):
        return 6*x_
    def df4dy(x_, y_):
        return 4*y_

    def f5(x_, y_):
        return 2*y_*y_ + 3*x_*x_ + 4*x_*y_
    def df5dx(x_, y_):
        return 6*x_ + 4*y_
    def df5dy(x_, y_):
        return 4*y_ + 4*x_

    fs = [f1, f2, f3, f4, f5]
    dfdxs = [df1dx, df2dx, df3dx, df4dx, df5dx]
    dfdys = [df1dy, df2dy, df3dy, df4dy, df5dy]

    x = np.linspace(0.1, 3.3, 250)
    y = np.linspace(0.2, 10.4, 750)
    xx, yy = np.meshgrid(x, y)

    for f, dfdx, dfdy in zip(fs, dfdxs, dfdys):
        z = f(xx, yy)
        tab2d = galsim.LookupTable2D(x, y, z, interpolant='spline')

        # Check single value functionality.
        x1,y1 = 2.3, 1.2
        np.testing.assert_allclose(tab2d(x1,y1), f(x1, y1), atol=1e-11, rtol=0)
        ref_dfdx = dfdx(x1,y1)
        ref_dfdy = dfdy(x1,y1)
        test_dfdx, test_dfdy = tab2d.gradient(x1,y1)
        np.testing.assert_allclose(test_dfdx, ref_dfdx, atol=1e-11, rtol=0)
        np.testing.assert_allclose(test_dfdy, ref_dfdy, atol=1e-11, rtol=0)

        # Check vectorized output
        newx = np.linspace(0.2, 3.1, 45)
        newy = np.linspace(0.3, 10.1, 85)
        newxx, newyy = np.meshgrid(newx, newy)

        np.testing.assert_allclose(tab2d(newxx, newyy), f(newxx, newyy), atol=1e-11, rtol=0)
        ref_dfdx = dfdx(newxx, newyy)
        ref_dfdy = dfdy(newxx, newyy)
        test_dfdx, test_dfdy = tab2d.gradient(newxx, newyy)
        np.testing.assert_allclose(test_dfdx, ref_dfdx, atol=1e-11, rtol=0)
        np.testing.assert_allclose(test_dfdy, ref_dfdy, atol=1e-11, rtol=0)

    # I think it ought to be possible to exactly interpolate the following,
    # provided the exact derivatives are provided.  Use slightly looser tolerances,
    # which seems reasonable since these are varying more rapidly.
    def f6(x_, y_):
        return y_*y_*y_
    def df6dx(x_, y_):
        return np.zeros_like(x_)
    def df6dy(x_, y_):
        return 3*y_*y_
    def d2f6dxdy(x_, y_):
        return np.zeros_like(x_)

    def f7(x_, y_):
        return 2*y_*y_*y_ + 3*x_*x_*x_ + 4*x_*y_*y_
    def df7dx(x_, y_):
        return 9*x_*x_ + 4*y_*y_
    def df7dy(x_, y_):
        return 6*y_*y_ + 8*x_*y_
    def d2f7dxdy(x_, y_):
        return 8*y_

    fs = [f6, f7]
    dfdxs = [df6dx, df7dx]
    dfdys = [df6dy, df7dy]
    d2fdxdys = [d2f6dxdy, d2f7dxdy]

    for f, dfdx, dfdy, d2fdxdy in zip(fs, dfdxs, dfdys, d2fdxdys):
        z = f(xx, yy)
        tab2d = galsim.LookupTable2D(x, y, z, interpolant='spline',
            dfdx=dfdx(xx, yy), dfdy=dfdy(xx, yy), d2fdxdy=d2fdxdy(xx, yy))

        # Check single value functionality.
        x1,y1 = 2.3, 1.2
        np.testing.assert_allclose(tab2d(x1,y1), f(x1, y1), atol=1e-10, rtol=0)
        ref_dfdx = dfdx(x1,y1)
        ref_dfdy = dfdy(x1,y1)
        test_dfdx, test_dfdy = tab2d.gradient(x1,y1)
        np.testing.assert_allclose(test_dfdx, ref_dfdx, atol=1e-10, rtol=0)
        np.testing.assert_allclose(test_dfdy, ref_dfdy, atol=1e-10, rtol=0)

        # Check vectorized output
        newx = np.linspace(0.2, 3.1, 45)
        newy = np.linspace(0.3, 10.1, 85)
        newxx, newyy = np.meshgrid(newx, newy)

        np.testing.assert_allclose(tab2d(newxx, newyy), f(newxx, newyy), atol=1e-10, rtol=0)
        ref_dfdx = dfdx(newxx, newyy)
        ref_dfdy = dfdy(newxx, newyy)
        test_dfdx, test_dfdy = tab2d.gradient(newxx, newyy)
        np.testing.assert_allclose(test_dfdx, ref_dfdx, atol=1e-10, rtol=0)
        np.testing.assert_allclose(test_dfdy, ref_dfdy, atol=1e-10, rtol=0)


@timer
def test_table2d_GSInterp():
    def f(x_, y_):
        return 2*y_*y_ + 3*x_*x_ + 4*x_*y_ - np.cos(x_)

    x = np.linspace(0.1, 3.3, 25)
    y = np.linspace(0.2, 10.4, 75)
    xx, yy = np.meshgrid(x, y)

    interpolants = ['lanczos3', 'lanczos3F', 'lanczos7', 'sinc', 'quintic']

    for interpolant in interpolants:
        z = f(xx, yy)
        tab2d = galsim.LookupTable2D(x, y, z, interpolant=interpolant)
        check_pickle(tab2d)
        # Make sure precomputed-hash gets covered
        hash(tab2d)

        # Use InterpolatedImage to validate
        wcs = galsim.JacobianWCS(
            (max(x) - min(x))/(len(x)-1),
            0, 0,
            (max(y) - min(y))/(len(y)-1),
        )
        img = galsim.Image(z, wcs=wcs)
        ii = galsim.InterpolatedImage(
            img,
            use_true_center=True,
            offset=(galsim.PositionD(img.xmin,img.ymin)-img.true_center),
            x_interpolant=interpolant,
            normalization='sb',
            calculate_maxk=False, calculate_stepk=False
        ).shift(min(x), min(y))

        # Check single value functionality.
        x1,y1 = 2.3, 3.2
        np.testing.assert_allclose(tab2d(x1,y1), ii.xValue(x1, y1), atol=1e-10, rtol=0)

        # Check vectorized output
        newx = np.linspace(0.2, 3.1, 15)
        newy = np.linspace(0.3, 10.1, 25)
        newxx, newyy = np.meshgrid(newx, newy)
        np.testing.assert_allclose(
            tab2d(newxx, newyy).ravel(),
            np.array([ii.xValue(x_, y_)
                      for x_, y_ in zip(newxx.ravel(), newyy.ravel())]).ravel(),
            atol=1e-10, rtol=0
        )
        np.testing.assert_array_almost_equal(tab2d(newxx, newyy), tab2d(newx, newy, grid=True))

        # Check that edge_mode='wrap' works
        tab2d = galsim.LookupTable2D(x, y, z, edge_mode='wrap')

        ref_dfdx, ref_dfdy = tab2d.gradient(newxx, newyy)
        test_dfdx, test_dfdy = tab2d.gradient(newxx+3*tab2d.xperiod, newyy)
        np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx)
        np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy)

        test_dfdx, test_dfdy = tab2d.gradient(newxx, newyy+13*tab2d.yperiod)
        np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx)
        np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy)

        test_dfdx, test_dfdy = tab2d.gradient(newx, newy+13*tab2d.yperiod, grid=True)
        np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx)
        np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy)

        # Test mix of inside and outside original boundary
        test_dfdx, test_dfdy = tab2d.gradient(np.dstack([newxx, newxx+3*tab2d.xperiod]),
                                              np.dstack([newyy, newyy]))

        np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx[:,:,0])
        np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy[:,:,0])
        np.testing.assert_array_almost_equal(ref_dfdx, test_dfdx[:,:,1])
        np.testing.assert_array_almost_equal(ref_dfdy, test_dfdy[:,:,1])


@timer
def test_ne():
    """ Check that inequality works as expected."""
    # These should all compare as unequal.
    x = [1, 2, 3]
    f = [4, 5, 6]
    x2 = [1.1, 2.2, 3.3]
    f2 = [4.4, 5.5, 6.6]
    lts = [galsim.LookupTable(x, f),
           galsim.LookupTable(x, f2),
           galsim.LookupTable(x2, f),
           galsim.LookupTable(x, f, interpolant='floor'),
           galsim.LookupTable(x, f, x_log=True),
           galsim.LookupTable(x, f, f_log=True)]

    ff = f+np.array(f)[:, None]
    ff2 = f2+np.array(f2)[:, None]
    lts.extend([
        galsim.LookupTable2D(x, x, ff),
        galsim.LookupTable2D(x2, x, ff),
        galsim.LookupTable2D(x, x2, ff),
        galsim.LookupTable2D(x, x2, ff2),
        galsim.LookupTable2D(x, x, ff, interpolant='nearest'),
        galsim.LookupTable2D(x, x, ff, interpolant=galsim.Nearest()),
        galsim.LookupTable2D(x, x, ff, edge_mode='wrap'),
        galsim.LookupTable2D(x, x, ff, constant=1),
        galsim.LookupTable2D(x, x, ff, interpolant='spline'),
        galsim.LookupTable2D(x, x, ff, interpolant='spline',
            dfdx=ff, dfdy=ff, d2fdxdy=ff
        )
    ])
    check_all_diff(lts)

@timer
def test_integrate():
    functions = [
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='linear'),
        galsim.LookupTable([0,4], [1,8], interpolant='linear'),
        galsim.LookupTable([0,0.1,0.2,3.5,4], [1,5,5,8,1], interpolant='linear'),
        galsim.LookupTable([0,2.1,2.2,2.5,4], [1,5,5,8,1], interpolant='linear'),
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='floor'),
        galsim.LookupTable([0,4], [1,8], interpolant='floor'),
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='ceil'),
        galsim.LookupTable([0,4], [1,8], interpolant='ceil'),
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='nearest'),
        galsim.LookupTable([0,4], [1,8], interpolant='nearest'),
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='spline'),
        galsim.LookupTable([0,1,4], [1,1,8], interpolant='spline'),
        galsim.LookupTable([0,0.1,0.2,3.5,4], [1,2,2,8,7], interpolant='spline'),
        galsim.LookupTable([0,2.1,2.2,2.5,4], [1,5,5,5,1], interpolant='spline'),
    ]

    for func in functions:
        print('func = ',repr(func))
        for x_min in [None, func.x_min, func.x_min+1, func.x_min-1]:
            for x_max in [None, func.x_max, func.x_max-1, func.x_max+1]:
                func_ans = func.integrate(x_min, x_max)

                # Note: To keep the original values, x_min, x_max are the loop values
                #       and xmin, xmax are what I use for the test integration, which may
                #       need to be adjusted somewhat.
                if x_min is None or x_min < func.x_min:
                    xmin = func.x_min
                else:
                    xmin = x_min
                if x_max is None or x_max > func.x_max:
                    xmax = func.x_max
                else:
                    xmax = x_max
                x = np.linspace(xmin, xmax, 10000)
                f = func(x)
                np_ans = np.trapz(f,x)
                print('integrate range %s..%s = %s  %s'%(xmin,xmax,func_ans,np_ans))
                if func.interpolant in ['linear', 'spline']:
                    rtol = 1.e-7
                else:
                    # ceil, floor, and nearest not super well approximated by trapz
                    rtol = 1.e-3
                np.testing.assert_allclose(func_ans, np_ans, rtol=rtol)

                # if xmin > xmax, return the negative.
                if x_min is not None and x_max is not None:
                    neg_func_ans = func.integrate(x_max, x_min)
                    np.testing.assert_allclose(neg_func_ans, -np_ans, rtol=rtol)

                # if xmin == xmax, result should be 0.
                if x_min is not None and x_max is None:
                    zero_func_ans = func.integrate(x_min, x_min)
                    np.testing.assert_allclose(zero_func_ans, 0, atol=1.e-12)


    # Time how long it takes to integrate a really big table
    x = np.linspace(0,2,1000000)
    y = 4. - x**2
    assert np.min(y) >= 0.
    y = np.sqrt(y)
    t0 = time.time()
    ans1 = galsim._LookupTable(x,y,'linear').integrate()
    t1 = time.time()
    ans2 = galsim._LookupTable(x,y,'spline').integrate()
    t2 = time.time()
    ans3 = galsim._LookupTable(x,y,'floor').integrate()
    t3 = time.time()
    ans4 = galsim._LookupTable(x,y,'ceil').integrate()
    t4 = time.time()
    ans5 = galsim._LookupTable(x,y,'nearest').integrate()
    t5 = time.time()
    ans6 = galsim.trapz(y,x)
    t6 = time.time()
    ans7 = np.trapz(y,x)
    t7 = time.time()
    ans8 = galsim.trapz(y,x, interpolant='spline')
    t8 = time.time()
    # Amazingly, LookupTable(linear) is more than 2x faster than np.trapz, and equally accurate.
    # Spline is a bit slower, but about 2x more accurate.
    # Floor and ceil are slightly faster still, but not nearly as accurate.
    print('           integration time       answer - Pi')
    print('linear         %.6f          %.6e'%(t1-t0,ans1-np.pi))
    print('spline         %.6f          %.6e'%(t2-t1,ans2-np.pi))
    print('floor          %.6f          %.6e'%(t3-t2,ans3-np.pi))
    print('ceil           %.6f          %.6e'%(t4-t3,ans4-np.pi))
    print('nearest        %.6f          %.6e'%(t5-t4,ans5-np.pi))
    print('trapz          %.6f          %.6e'%(t6-t5,ans6-np.pi))
    print('np.trapz       %.6f          %.6e'%(t7-t6,ans7-np.pi))
    print('trapz(spline)  %.6f          %.6e'%(t8-t7,ans8-np.pi))
    np.testing.assert_allclose(ans1, np.pi, atol=2.e-9)
    np.testing.assert_allclose(ans2, np.pi, atol=1.e-9)
    np.testing.assert_allclose(ans3, np.pi, atol=3.e-6)
    np.testing.assert_allclose(ans4, np.pi, atol=3.e-6)
    np.testing.assert_allclose(ans5, np.pi, atol=2.e-9)
    np.testing.assert_allclose(ans6, np.pi, atol=2.e-9)
    np.testing.assert_allclose(ans7, np.pi, atol=2.e-9)
    np.testing.assert_allclose(ans8, np.pi, atol=1.e-9)

    # Check errors
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], x_log=True).integrate()
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], f_log=True).integrate()
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], x_log=True, f_log=True).integrate()
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], interpolant=galsim.Quintic()).integrate()
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], interpolant=galsim.Linear()).integrate()
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], interpolant=galsim.Lanczos(5)).integrate()


@timer
def test_integrate_product():
    functions = [
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='linear'),
        galsim.LookupTable([0,4], [1,8], interpolant='linear'),
        galsim.LookupTable([0,0.1,0.2,3.5,4], [1,5,5,8,1], interpolant='linear'),
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='floor'),
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='ceil'),
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='nearest'),
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='spline'),
        galsim.LookupTable([0,2.1,2.2,2.5,4], [1,5,5,5,1], interpolant='spline'),
    ]

    g_functions = [
        galsim.LookupTable([0,1,2,3,4], [1,5,5,8,1], interpolant='linear'),
        galsim.LookupTable([0,6], [1,8], interpolant='linear'),
        galsim.LookupTable([2,4,6,8], [1,5,8,1], interpolant='spline'),
        galsim.LookupTable([0,2.1,2.2,2.5,4], [1,5,5,5,1], interpolant='spline'),
        lambda x: x**3,
        np.exp,
        lambda x: 17.,
        galsim.LookupTable([5,10], [1,8], interpolant='linear'),
    ]

    for func in functions:
        print('func = ',repr(func))
        for g in g_functions:
            #print('g = ',repr(g))
            for xfact in [1, 1.25, 0.9]:
                #print('xfact = ',xfact)
                for x_min in [None, func.x_min/xfact, func.x_min/xfact+1, func.x_min/xfact-1]:
                    for x_max in [None, func.x_max/xfact, func.x_max/xfact-1, func.x_max/xfact+1]:
                        #print('xmin/xmax = ',x_min,x_max)
                        func_ans = func.integrate_product(g, x_min, x_max, xfact)
                        if x_min is None or x_min < func.x_min:
                            xmin = func.x_min / xfact
                        else:
                            xmin = x_min
                        if x_max is None or x_max > func.x_max:
                            xmax = func.x_max / xfact
                        else:
                            xmax = x_max

                        # Make the version of g that integrate_product approximates it as.
                        if isinstance(g, galsim.LookupTable):
                            if xmin < g.x_min: xmin = g.x_min
                            if xmax > g.x_max: xmax = g.x_max
                            x2 = np.union1d(func.x / xfact, g.x)
                            x2 = x2[(x2>=xmin) & (x2<=xmax)]
                        else:
                            x2 = func.x / xfact
                        if xmin >= xmax:
                            np.testing.assert_allclose(func_ans, 0, atol=1.e-12)
                            continue
                        x2 = np.union1d(x2, [xmin, xmax])
                        gf2 = g(x2)
                        if isinstance(gf2, float): gf2 = gf2 * np.ones(len(x2))
                        g2 = galsim.LookupTable(x2, gf2, 'linear')

                        # Do the integral using trapz with fine x spacing.
                        x = np.linspace(xmin, xmax, 10000)
                        fg = func(x*xfact) * g2(x)
                        np_ans = np.trapz(fg,x)
                        #print('integrate range %s..%s = %s  %s'%(xmin,xmax,func_ans,np_ans))
                        if func.interpolant in ['linear', 'spline']:
                            rtol = 1.e-7
                        else:
                            # ceil, floor, and nearest not super well approximated by trapz
                            rtol = 1.e-3
                        np.testing.assert_allclose(func_ans, np_ans, rtol=rtol)

                        # if xmin > xmax, return the negative.
                        if x_min is not None and x_max is not None:
                            neg_func_ans = func.integrate_product(g, x_max, x_min, xfact)
                            np.testing.assert_allclose(neg_func_ans, -np_ans, rtol=rtol)

                        # if xmin == xmax, result should be 0.
                        if x_min is not None and x_max is None:
                            zero_func_ans = func.integrate_product(g, x_min, x_min, xfact)
                            np.testing.assert_allclose(zero_func_ans, 0, atol=1.e-12)


    # Check errors
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], x_log=True).integrate_product(g)
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], f_log=True).integrate_product(g)
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], x_log=True, f_log=True).integrate_product(g)
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], interpolant=galsim.Quintic()).integrate_product(g)
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], interpolant=galsim.Linear()).integrate_product(g)
    with assert_raises(NotImplementedError):
        galsim.LookupTable([1,2,3,4], [5,5,8,1], interpolant=galsim.Lanczos(5)).integrate_product(g)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
