#!/usr/bin/env python

import numpy as np
import galsim

# Some values to make the OpticalPSF (uses an SBInterpolatedImage table) interesting and 
# non-symmetric:
COMA1 = 0.17
ASTIG2 = -0.44
DEFOCUS = -0.3
SPHER = 0.027
OVERSAMPLING = 2.

LAM_OVER_D = 5.

# Tabulate some SBInterpolatedImage kValues using the old (working) NR spline routines in Table.h,
# and use these as a regression test on the new (home grown) spline routines.
#
# Uses the NR spline code as preserved in the repo, at commit:
# 8a9b04085b873f63be4fb56ae4fa5a0ca78a0387
#
# ...and the updated OpticalPSF code from commit:
# 6925d74efa8bbec8f2d8de0cdeda58a926a6ade2
#
# Revert to the commit one after this commit (6c077855db16a07486d3753f2d7aec825e5e6414) and 
# `scons -c`, then `scons` to rebuild using old NR splines before re-running this script.

# Some arbitrary kx, ky k space values to test
kxvals = np.array((1.3, 0.71, -4.3)) * np.pi / 2.
kyvals = np.array((.8, -2., -.31,)) * np.pi / 2.

absoutk = np.zeros(len(kxvals)) # result storage arrays

# First make a Cubic interpolant
interp = galsim.InterpolantXY(galsim.Cubic(tol=1.e-4))
testobj = galsim.OpticalPSF(lam_over_D=LAM_OVER_D, defocus=DEFOCUS, astig2=ASTIG2, coma1=COMA1,
                            spher=SPHER, interpolantxy=interp, oversampling=OVERSAMPLING)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt('absfKCubic_test.txt', absoutk)

# Then make a Quintic interpolant
interp = galsim.InterpolantXY(galsim.Quintic(tol=1.e-4))
testobj = galsim.OpticalPSF(lam_over_D=LAM_OVER_D, defocus=DEFOCUS, astig2=ASTIG2, coma1=COMA1,
                            spher=SPHER, interpolantxy=interp, oversampling=OVERSAMPLING)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt('absfKQuintic_test.txt', absoutk)

# Then make a Lanczos5 interpolant
interp = galsim.InterpolantXY(galsim.Lanczos(5, conserve_flux=True, tol=1.e-4))
testobj = galsim.OpticalPSF(lam_over_D=LAM_OVER_D, defocus=DEFOCUS, astig2=ASTIG2, coma1=COMA1,
                            spher=SPHER, interpolantxy=interp, oversampling=OVERSAMPLING)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt('absfKLanczos5_test.txt', absoutk)

# Then make a Lanczos7 interpolant
interp = galsim.InterpolantXY(galsim.Lanczos(7, conserve_flux=True, tol=1.e-4))
testobj = galsim.OpticalPSF(lam_over_D=LAM_OVER_D, defocus=DEFOCUS, astig2=ASTIG2, coma1=COMA1,
                            spher=SPHER, interpolantxy=interp, oversampling=OVERSAMPLING)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt('absfKLanczos7_test.txt', absoutk)


