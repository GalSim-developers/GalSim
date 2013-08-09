import numpy as np
import galsim

TESTDIM = 50

rng = galsim.BaseDeviate()
test_image1 = galsim.ImageD(TESTDIM, TESTDIM)
test_image2 = galsim.ImageD(TESTDIM, TESTDIM)

failed1 = 0
failed2 = 0
failed3 = 0
failed4 = 0
failed5 = 0
failed6 = 0

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
    periodicity_correction = galsim.correlatednoise._cf_periodicity_dilution_correction(cf.shape)

    ps_from_cfreal = np.fft.ifft2(cf.real)
    ps_from_cfabs = np.fft.ifft2(np.abs(cf))
    ps_from_cfreal_with_correction = np.fft.ifft2(cf.real * periodicity_correction)
    if np.any(ps_from_cfreal.real < -1.e-12 * np.mean(ps_from_cfreal).real):
        import matplotlib.pyplot as plt
        print ps_from_cfreal
        print ps_from_cfreal.min()
        failed3 += 1
    if np.any(ps_from_cfabs.real < -1.e-12 * np.mean(ps_from_cfabs).real):
        failed4 += 1
    if np.any(
        ps_from_cfreal_with_correction.real < 
        -1.e-12 * np.mean(ps_from_cfreal_with_correction).real):
        failed5 += 1

print "With periodicity correction failed                               "+\
    str(failed1)+"/"+str(TESTEND - TESTSTART)+" times"
print "Without periodicity correction failed                            "+\
    str(failed2)+"/"+str(TESTEND - TESTSTART)+" times"
print "By hand, PS from real part of CF failed                          "+\
    str(failed3)+"/"+str(TESTEND - TESTSTART)+" times"
print "By hand, PS from abs() of CF failed                              "+\
    str(failed4)+"/"+str(TESTEND - TESTSTART)+" times"
print "By hand, PS from real part of CF + periodicity correction failed "+\
    str(failed5)+"/"+str(TESTEND - TESTSTART)+" times"


