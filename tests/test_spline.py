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

"""@brief Tests for the numerical spline routines used to tabulate some of the k space interpolators
for which an analytic result is not available (Cubic, Quintic, Lanczos).

Compares kValue outputs from an SBInterpolatedImage (sum of Guassians) that were created
using a previous version of the code (commit: 4d71631d7379f76bb0e3ee582b5a1fbdc0def666), using the
script devutils/external/test_spline/make_spline_arrays.py.
"""
import os
import numpy as np

path, filename = os.path.split(__file__) # Get the path to this file for use below...
try:
    import galsim
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

TESTDIR=os.path.join(path, "spline_comparison_files")

DECIMAL = 14 # Make sure output agrees at 14 decimal places or better

# Some arbitrary kx, ky k space values to test
KXVALS = np.array((1.30, 0.71, -4.30)) * np.pi / 2.
KYVALS = np.array((0.80, -0.02, -0.31,)) * np.pi / 2.

absoutk = np.zeros(len(KXVALS)) # result storage arrays

# First make an image that we'll use for interpolation:
g1 = galsim.Gaussian(sigma = 3.1, flux=2.4)
g1.applyShear(g1=0.2,g2=0.1)
g2 = galsim.Gaussian(sigma = 1.9, flux=3.1)
g2.applyShear(g1=-0.4,g2=0.3)
g2.applyShift(-0.3,0.5)
g3 = galsim.Gaussian(sigma = 4.1, flux=1.6)
g3.applyShear(g1=0.1,g2=-0.1)
g3.applyShift(0.7,-0.2)

final = g1 + g2 + g3
image = galsim.ImageD(128,128)
dx = 0.4
# The reference image was drawn with the old convention, which is now use_true_center=False
final.draw(image=image, dx=dx, normalization='sb', use_true_center=False)

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_Cubic_spline():
    """Test the spline tabulation of the k space Cubic interpolant.
    """
    import time
    t1 = time.time()
    interp = galsim.InterpolantXY(galsim.Cubic(tol=1.e-4))
    testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
    testKvals = np.zeros(len(KXVALS))
    # Make test kValues
    for i in xrange(len(KXVALS)):
        posk = galsim.PositionD(KXVALS[i], KYVALS[i])
        testKvals[i] = np.abs(testobj.kValue(posk))
    # Compare with saved array
    refKvals = np.loadtxt(os.path.join(TESTDIR, "absfKCubic_test.txt"))
    np.testing.assert_array_almost_equal(refKvals, testKvals, DECIMAL,
                                         err_msg="Spline-interpolated kValues do not match saved "+
                                                 "data for k space Cubic interpolant.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Quintic_spline():
    """Test the spline tabulation of the k space Quintic interpolant.
    """
    import time
    t1 = time.time()
    interp = galsim.InterpolantXY(galsim.Quintic(tol=1.e-4))
    testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
    testKvals = np.zeros(len(KXVALS))
    # Make test kValues
    for i in xrange(len(KXVALS)):
        posk = galsim.PositionD(KXVALS[i], KYVALS[i])
        testKvals[i] = np.abs(testobj.kValue(posk))
    # Compare with saved array
    refKvals = np.loadtxt(os.path.join(TESTDIR, "absfKQuintic_test.txt"))
    np.testing.assert_array_almost_equal(refKvals, testKvals, DECIMAL,
                                         err_msg="Spline-interpolated kValues do not match saved "+
                                                 "data for k space Quintic interpolant.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Lanczos5_spline():
    """Test the spline tabulation of the k space Lanczos-5 interpolant.
    """
    import time
    t1 = time.time()
    interp = galsim.InterpolantXY(galsim.Lanczos(5, conserve_flux=True, tol=1.e-4))
    testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
    testKvals = np.zeros(len(KXVALS))
    # Make test kValues
    for i in xrange(len(KXVALS)):
        posk = galsim.PositionD(KXVALS[i], KYVALS[i])
        testKvals[i] = np.abs(testobj.kValue(posk))
    # Compare with saved array
    refKvals = np.loadtxt(os.path.join(TESTDIR, "absfKLanczos5_test.txt"))
    np.testing.assert_array_almost_equal(refKvals, testKvals, DECIMAL,
                                         err_msg="Spline-interpolated kValues do not match saved "+
                                                 "data for k space Lanczos-5 interpolant.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
    
def test_Lanczos7_spline():
    """Test the spline tabulation of the k space Lanczos-7 interpolant.
    """
    import time
    t1 = time.time()
    interp = galsim.InterpolantXY(galsim.Lanczos(7, conserve_flux=True, tol=1.e-4))
    testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
    testKvals = np.zeros(len(KXVALS))
    # Make test kValues
    for i in xrange(len(KXVALS)):
        posk = galsim.PositionD(KXVALS[i], KYVALS[i])
        testKvals[i] = np.abs(testobj.kValue(posk))
    # Compare with saved array
    refKvals = np.loadtxt(os.path.join(TESTDIR, "absfKLanczos7_test.txt"))
    np.testing.assert_array_almost_equal(refKvals, testKvals, DECIMAL,
                                         err_msg="Spline-interpolated kValues do not match saved "+
                                                 "data for k space Lanczos-7 interpolant.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_Cubic_spline()
    test_Quintic_spline()
    test_Lanczos5_spline()
    test_Lanczos7_spline()

