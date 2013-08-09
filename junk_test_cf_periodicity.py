import numpy as np
import galsim

TESTDIM = 50

rng = galsim.BaseDeviate()
test_image1 = galsim.ImageD(TESTDIM, TESTDIM)
test_image2 = galsim.ImageD(TESTDIM, TESTDIM)

failed1 = 0
failed2 = 0

TESTSTART = 25
TESTEND = 125

for dim in range(TESTSTART, TESTEND):
    noise_array = np.random.randn(dim, dim)
    noise_image = galsim.ImageViewD(noise_array)
    cn1 = galsim.CorrelatedNoise(rng, noise_image)
    cn2 = galsim.CorrelatedNoise(rng, noise_image, correct_periodicity=False)
    test_image1.setZero()
    try:
        cn1.applyTo(test_image1)
    except RuntimeError:
        failed1 += 1
    test_image2.setZero()
    try:
        cn2.applyTo(test_image2)
    except RuntimeError:
        failed2 +=1

print "With periodicity correction failed    "+str(failed1)+"/"+str(TESTSTART-TESTEND)+" times"
print "Without periodicity correction failed "+str(failed2)+"/"+str(TESTSTART-TESTEND)+" times"
    


