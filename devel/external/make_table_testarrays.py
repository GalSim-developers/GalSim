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

import numpy as np
import galsim
import os

# Tabulate some interpolated values to use as a regression test for Table.h.
#
# To re-run this script:
# - First type `git checkout e267f058351899f1f820adf4d6ab409d5b2605d5`
# - Type `scons install` in the main GalSim directory.
# - Then type `python make_table_testarrays.py` in this directory.

# Some arbitrary args to use for the test:
args1 = range(7)  # Evenly spaced
vals1 = [ x**2 for x in args1 ]
testargs1 = [ 0.1, 0.8, 2.3, 3, 5.6, 5.9 ] # < 0 or > 7 is invalid

args2 = [ 0.7, 3.3, 14.1, 15.6, 29, 34.1, 42.5 ]  # Not evenly spaced
vals2 = [ np.sin(x*np.pi/180) for x in args2 ]
testargs2 = [ 1.1, 10.8, 12.3, 15.6, 25.6, 41.9 ] # < 0.7 or > 42.5 is invalid

interps = [ 'linear', 'spline', 'floor', 'ceil' ]

dir = '../../tests/table_comparison_files'

for interp in interps:
    print 'args1 = ',args1
    print 'vals1 = ',vals1
    print 'interp = ',interp
    table1 = galsim.LookupTable(args1,vals1,interp)
    print 'table1.getArgs() = ',table1.getArgs()
    print 'table1.getVals() = ',table1.getVals()
    print 'testargs1 = ',testargs1
    testvals1 = [ table1(x) for x in testargs1 ]
    print 'testvals1 = ',testvals1

    print 'args2 = ',args2
    print 'vals2 = ',vals2
    print 'interp = ',interp
    table2 = galsim.LookupTable(args2,vals2,interp)
    print 'table2.getArgs() = ',table2.getArgs()
    print 'table2.getVals() = ',table2.getVals()
    print 'testargs2 = ',testargs2
    testvals2 = [ table2(x) for x in testargs2 ]
    print 'testvals2 = ',testvals2

    np.savetxt(os.path.join(dir, 'table_test1_%s.txt'%interp), testvals1)
    np.savetxt(os.path.join(dir, 'table_test2_%s.txt'%interp), testvals2)

