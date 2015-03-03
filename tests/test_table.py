# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

"""@brief Tests of the LookupTable class.

Compares interpolated values a LookupTable that were created using a previous version of
the code (commit: e267f058351899f1f820adf4d6ab409d5b2605d5), using the
script devutils/external/make_table_testarrays.py
"""
import os
import numpy as np
import pickle

from galsim_test_helpers import *

path, filename = os.path.split(__file__) # Get the path to this file for use below...
try:
    import galsim
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

TESTDIR=os.path.join(path, "table_comparison_files")

DECIMAL = 14 # Make sure output agrees at 14 decimal places or better

# Some arbitrary args to use for the test:
args1 = range(7)  # Evenly spaced
vals1 = [ x**2 for x in args1 ]
testargs1 = [ 0.1, 0.8, 2.3, 3, 5.6, 5.9 ] # < 0 or > 7 is invalid

args2 = [ 0.7, 3.3, 14.1, 15.6, 29, 34.1, 42.5 ]  # Not evenly spaced
vals2 = [ np.sin(x*np.pi/180) for x in args2 ]
testargs2 = [ 1.1, 10.8, 12.3, 15.6, 25.6, 41.9 ] # < 0.7 or > 42.5 is invalid

interps = [ 'linear', 'spline', 'floor', 'ceil' ]

def test_table():
    """Test the spline tabulation of the k space Cubic interpolant.
    """
    import time
    t1 = time.time()

    for interp in interps:
        table1 = galsim.LookupTable(x=args1,f=vals1,interpolant=interp)
        testvals1 = [ table1(x) for x in testargs1 ]

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
        try:
            np.testing.assert_raises(RuntimeError,table1,args1[0]-0.01)
            np.testing.assert_raises(RuntimeError,table1,args1[-1]+0.01)
            np.testing.assert_raises(RuntimeError,table2,args2[0]-0.01)
            np.testing.assert_raises(RuntimeError,table2,args2[-1]+0.01)
            np.testing.assert_raises(ValueError,table1,np.zeros((3,3,3))+args1[0])
        except ImportError:
            print 'The assert_raises tests require nose'

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

        # Check that a LookupTable is picklable.
        p1 = pickle.dumps(table1)
        table1x = pickle.loads(p1)
        np.testing.assert_equal(table1.getArgs(), table1x.getArgs(),
                err_msg="Pickled LookupTable does not preserve correct args")
        np.testing.assert_equal(table1.getVals(), table1x.getVals(),
                err_msg="Pickled LookupTable does not preserve correct vals")
        np.testing.assert_equal(table1.getInterp(), table1x.getInterp(),
                err_msg="Pickled LookupTable does not preserve correct interp")

        p2 = pickle.dumps(table2)
        table2x = pickle.loads(p2)
        np.testing.assert_equal(table2.getArgs(), table2x.getArgs(),
                err_msg="Pickled LookupTable does not preserve correct args")
        np.testing.assert_equal(table2.getVals(), table2x.getVals(),
                err_msg="Pickled LookupTable does not preserve correct vals")
        np.testing.assert_equal(table2.getInterp(), table2x.getInterp(),
                err_msg="Pickled LookupTable does not preserve correct interp")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_init():
    """Some simple tests of LookupTable initialization."""
    import time
    t1 = time.time()

    interp = 'linear'
    try:
        # Check for bad input: 1 column file, or specifying file and x, or just x, or bad
        # interpolant.
        np.testing.assert_raises(ValueError, galsim.LookupTable,
                                 file=os.path.join(TESTDIR, 'table_test1_%s.txt'%interp),
                                 x = interp)
        np.testing.assert_raises(ValueError, galsim.LookupTable,
                                 file=os.path.join(TESTDIR, 'table_test1_%s.txt'%interp))
        np.testing.assert_raises(ValueError, galsim.LookupTable,
                                 x=os.path.join(TESTDIR, 'table_test1_%s.txt'%interp))
        np.testing.assert_raises(ValueError, galsim.LookupTable,
                                 file='../examples/data/cosmo-fid.zmed1.00_smoothed.out',
                                 interpolant='foo')
    except ImportError:
        print 'The assert_raises tests require nose'
    # Also make sure nothing bad happens when we try to read in a stored power spectrum and assume
    # we can use the default interpolant (spline).
    tab_ps = galsim.LookupTable(
        file='../examples/data/cosmo-fid.zmed1.00_smoothed.out')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_log():
    """Some simple tests of interpolation using logs."""
    import time
    t1 = time.time()

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
        print result_1, result_2, result_3, result_4
        np.testing.assert_almost_equal(
            result_2, result_1, decimal=3,
            err_msg='Disagreement when interpolating in log(f) and log(x) vs. using the values directly.')
        np.testing.assert_almost_equal(
            result_3, result_1, decimal=3,
            err_msg='Disagreement when interpolating in log(x) vs. using the values directly.')
        np.testing.assert_almost_equal(
            result_4, result_1, decimal=3,
            err_msg='Disagreement when interpolating in log(f) vs. using the values directly.')

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
    try:
        np.testing.assert_raises(ValueError, galsim.LookupTable, x=x_neg, f=y_neg, x_log=True)
        np.testing.assert_raises(ValueError, galsim.LookupTable, x=x_neg, f=y_neg, f_log=True)
        np.testing.assert_raises(ValueError, galsim.LookupTable, x=x_neg, f=y_neg, x_log=True,
                                 f_log=True)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_roundoff():
    import time
    t1 = time.time()

    table1 = galsim.LookupTable([1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10])
    try:
        table1(1.0 - 1.e-7)
        table1(10.0 + 1.e-7)
    except:
        raise ValueError("c++ LookupTable roundoff guard failed.")
    try:
        np.testing.assert_raises(RuntimeError, table1, 1.0-1.e5)
        np.testing.assert_raises(RuntimeError, table1, 10.0+1.e5)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_table()
    test_init()
    test_log()
    test_roundoff()
