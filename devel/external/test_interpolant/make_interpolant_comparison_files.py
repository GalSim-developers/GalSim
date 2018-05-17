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

# Tabulate some SBInterpolatedImage kValues using the old (working) NR spline routines in Table.h,
# and use these as a regression test on the new (home grown) spline routines.
#
# Uses the NR spline code as preserved in the repo, at commit:
# 8a9b04085b873f63be4fb56ae4fa5a0ca78a0387
#
# To re-run this script:
# - First type `git checkout 5d749338c6092f2f54d5f604eef977fba13c997c`
# - Type `scons install` in the main GalSim directory.
# - Then type `python make_interpolant_comparison_files.py` in this directory.

# Some arbitrary kx, ky k space values to test
kxvals = np.array((1.30, 0.71, -4.30)) * np.pi / 2.
kyvals = np.array((0.80, -0.02, -0.31)) * np.pi / 2.

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

dir = '../../../tests/interpolant_comparison_files'

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
interp = galsim.InterpolantXY(galsim.Lanczos(5, conserve_flux=False, tol=1.e-4))
testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt(os.path.join(dir,'absfKLanczos5_test.txt'), absoutk)

# Then make a Lanczos7 interpolant
interp = galsim.InterpolantXY(galsim.Lanczos(7, conserve_flux=False, tol=1.e-4))
testobj = galsim.SBInterpolatedImage(image.view(), interp, dx=dx)
for i in xrange(len(kxvals)):
    posk = galsim.PositionD(kxvals[i], kyvals[i])
    absoutk[i] = np.abs(testobj.kValue(posk))
print absoutk
np.savetxt(os.path.join(dir,'absfKLanczos7_test.txt'), absoutk)


