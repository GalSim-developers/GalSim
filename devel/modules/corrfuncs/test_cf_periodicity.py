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

"""test_cf_periodicity.py:  See how the assumption of purely positive power spectra holds up under
the periodicity correction, and other effects.

See the full discussion at https://github.com/GalSim-developers/GalSim/issues/430 
"""
import numpy as np
import galsim

TESTDIM = 50

rng = galsim.BaseDeviate()#1233235555)
test_image1 = galsim.ImageD(TESTDIM, TESTDIM)
test_image2 = galsim.ImageD(TESTDIM, TESTDIM)

failed1 = 0
failed2 = 0
failed3 = 0
failed4 = 0
failed5 = 0
failed6 = 0
failed7 = 0

TESTSTART = 25
TESTEND = 125

for dim in range(TESTSTART, TESTEND):
    noise_array = np.random.randn(dim, dim)
    noise_image = galsim.ImageViewD(noise_array)
    cn1 = galsim.CorrelatedNoise(rng, noise_image)
    cn2 = galsim.CorrelatedNoise(rng, noise_image, correct_periodicity=False)
    # First try with (default), then without, periodicity correction
    test_image1.setZero()
    try:
        cn1.applyTo(test_image1)
    except RuntimeError:
        failed1 += 1
    test_image2.setZero()
    try:
        cn2.applyTo(test_image2)
    except RuntimeError:
        failed2 += 1
    # Then try calculating the PS by hand, in the same manner as the CorrelatedNoise internals
    noiseft = np.fft.fft2(noise_array)
    ps = np.abs(noiseft)**2
    cf = np.fft.ifft2(ps)
    # Get the periodicity correction to try this
    periodicity_correction = galsim.correlatednoise._cf_periodicity_dilution_correction(cf.shape)

    # Use the CF in a number of different ways to retrieve the PS and test results
    ps_from_cfreal = np.fft.ifft2(cf.real)
    ps_from_cfabs = np.fft.ifft2(np.abs(cf))
    ps_from_cfreal_with_correction = np.fft.ifft2(cf.real * periodicity_correction)
    import matplotlib.pyplot as plt
    if np.any(ps_from_cfreal.real < -1.e-12 * np.mean(ps_from_cfreal.real)):
        failed3 += 1
    if np.any(ps_from_cfabs.real < -1.e-12 * np.mean(ps_from_cfabs.real)):
        failed4 += 1
    if np.any(
        ps_from_cfreal_with_correction.real < 
        -1.e-12 * np.mean(ps_from_cfreal_with_correction.real)):
        failed5 += 1
    # Then try putting this correlation function into an array of the same dimensions as test_image3
    # as all these tests so far have been for arrays of dimensions (dim, dim).
    if dim <= TESTDIM:
        cf = galsim.utilities.roll2d(cf, ((dim - 1) / 2, (dim - 1)/ 2))
        cf_testsize = np.zeros((TESTDIM, TESTDIM), dtype=np.complex)
        cf_testsize[0:dim, 0:dim] += cf
        cf_testsize = galsim.utilities.roll2d(
            cf_testsize, (-((dim - 1) / 2), -((dim - 1) / 2))) # roll back
    if dim > TESTDIM:
        cf = galsim.utilities.roll2d(cf, (TESTDIM / 2 - 1, TESTDIM / 2 - 1))
        cf_testsize = np.zeros((TESTDIM, TESTDIM), dtype=np.complex)
        cf_testsize += cf[0: TESTDIM, 0: TESTDIM]
        cf_testsize = galsim.utilities.roll2d(
            cf_testsize, (-(TESTDIM / 2 - 1), -(TESTDIM / 2 - 1))) # roll back
    ps_from_cfreal_testsize = np.fft.ifft2(cf_testsize.real)
    if np.any(
        ps_from_cfreal_testsize.real < -1.e-12 * np.mean(ps_from_cfreal_testsize.real)):
        failed6 += 1
    # Then test the Hermitian stuff
    ps_from_halfcfreal_testsize = np.fft.irfft2(cf_testsize[:, 0:TESTDIM / 2 + 1])
    if np.any(
        ps_from_halfcfreal_testsize.real < -1.e-12 * np.mean(ps_from_halfcfreal_testsize.real)):
        failed7 += 1
print ""
print "With periodicity correction failed                                              "+\
    str(failed1)+"/"+str(TESTEND - TESTSTART)+" times"
print "Without periodicity correction failed                                           "+\
    str(failed2)+"/"+str(TESTEND - TESTSTART)+" times"
print "By hand, PS from real part of CF failed                                         "+\
    str(failed3)+"/"+str(TESTEND - TESTSTART)+" times"
print "By hand, PS from abs() of CF failed                                             "+\
    str(failed4)+"/"+str(TESTEND - TESTSTART)+" times"
print "By hand, PS from real part of CF + periodicity correction failed                "+\
    str(failed5)+"/"+str(TESTEND - TESTSTART)+" times"
print "By hand, PS from real part of CF, expanded to test size, failed                 "+\
    str(failed6)+"/"+str(TESTEND - TESTSTART)+" times"
print "By hand, PS from Hermitian half real part of CF, expanded to test size, failed  "+\
    str(failed7)+"/"+str(TESTEND - TESTSTART)+" times"
print ""
