#!/usr/bin/env python

import numpy as np
import galsim
import os

# Tabulate some SBInterpolatedImage kValues using the old (working) NR spline routines in Table.h,
# and use these as a regression test on the new (home grown) spline routines.
#
# Uses the NR spline code as preserved in the repo, at commit:
# 8a9b04085b873f63be4fb56ae4fa5a0ca78a0387
#
# Revert to the commit ???
# and `scons install` to re-run this script.

# Some arbitrary kx, ky k space values to test
kxvals = np.array((1.3, 0.71, -4.3)) * np.pi / 2.
kyvals = np.array((.8, -2., -.31,)) * np.pi / 2.

absoutk = np.zeros(len(kxvals)) # result storage arrays

# First make an image that we'll use for interpolation:
g1 = galsim.Gaussian(sigma = 3.1, flux=2.4)
g1.applyShear(0.2,0.1)
g2 = galsim.Gaussian(sigma = 1.9, flux=3.1)
g2.applyShear(-0.4,0.3)
g2.applyShift(-0.3,0.5)
g3 = galsim.Gaussian(sigma = 4.1, flux=1.6)
g3.applyShear(0.1,-0.1)
g3.applyShift(0.7,-0.2)

final = g1 + g2 + g3
image = galsim.ImageD(128,128)
dx = 0.4
final.draw(image=image, dx=dx)

dir = '../../../tests/spline_comparison_files'

# First make a Cubic interpolant
interp = galsim.InterpolantXY(galsim.Cubic(tol=1.e-4))
testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt(os.path.join(dir,'absfKCubic_test.txt'), absoutk)

# Then make a Quintic interpolant
interp = galsim.InterpolantXY(galsim.Quintic(tol=1.e-4))
testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt(os.path.join(dir,'absfKQuintic_test.txt'), absoutk)

# Then make a Lanczos5 interpolant
interp = galsim.InterpolantXY(galsim.Lanczos(5, conserve_flux=True, tol=1.e-4))
testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt(os.path.join(dir,'absfKLanczos5_test.txt'), absoutk)

# Then make a Lanczos7 interpolant
interp = galsim.InterpolantXY(galsim.Lanczos(7, conserve_flux=True, tol=1.e-4))
testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt(os.path.join(dir,'absfKLanczos7_test.txt'), absoutk)


