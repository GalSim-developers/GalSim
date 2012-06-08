
"""@brief Tests for the numerical spline routines used to tabulate some of the k space interpolators
for which an analytic result is not available (Cubic, Quintic, Lanczos).

Compares kValue outputs from an SBInterpolatedImage (an OpticalPSF in this case) that were created
using a previous version of the code (commit: ffeb22583894bd1c2254dbbd75449996f13a04a2), using the
script devutils/external/test_spline/make_spline_arrays.py.
"""
import os
import numpy as np

path, filename = os.path.split(__file__) # Get the path to this file for use below...
try:
    import galsim
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

TESTDIR=os.path.join(path, "spline_comparison_files")

DECIMAL = 14 # Make sure output agrees at 14 decimal places or better

# Some values used to make the OpticalPSF (uses an SBInterpolatedImage table) interesting and 
# non-symmetric:
COMA1 = 0.17
ASTIG2 = -0.44
DEFOCUS = -0.3
SPHER = 0.027
LAM_OVER_D = 5.
OVERSAMPLING = 2.
PAD_FACTOR=2.

# Some arbitrary kx, ky k space values to test
KXVALS = np.array((1.3, 0.71, -4.3)) * np.pi / 2.
KYVALS = np.array((.8, -2., -.31,)) * np.pi / 2.

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_Cubic_spline():
    """Test the spline tabulation of the k space Cubic interpolant.
    """
    import time
    t1 = time.time()
    interp = galsim.InterpolantXY(galsim.Cubic(tol=1.e-4))
    testobj = galsim.OpticalPSF(lam_over_D=LAM_OVER_D, defocus=DEFOCUS, astig2=ASTIG2, coma1=COMA1,
                                spher=SPHER, interpolantxy=interp, oversampling=OVERSAMPLING,
                                pad_factor=PAD_FACTOR)
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
    testobj = galsim.OpticalPSF(lam_over_D=LAM_OVER_D, defocus=DEFOCUS, astig2=ASTIG2, coma1=COMA1,
                                spher=SPHER, interpolantxy=interp, oversampling=OVERSAMPLING,
                                pad_factor=PAD_FACTOR)
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
    testobj = galsim.OpticalPSF(lam_over_D=LAM_OVER_D, defocus=DEFOCUS, astig2=ASTIG2, coma1=COMA1,
                                spher=SPHER, interpolantxy=interp, oversampling=OVERSAMPLING,
                                pad_factor=PAD_FACTOR)
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
    """Test the spline tabulation of the k space Lanczos-5 interpolant.
    """
    import time
    t1 = time.time()
    interp = galsim.InterpolantXY(galsim.Lanczos(7, conserve_flux=True, tol=1.e-4))
    testobj = galsim.OpticalPSF(lam_over_D=LAM_OVER_D, defocus=DEFOCUS, astig2=ASTIG2, coma1=COMA1,
                                spher=SPHER, interpolantxy=interp, oversampling=OVERSAMPLING,
                                pad_factor=PAD_FACTOR)
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

