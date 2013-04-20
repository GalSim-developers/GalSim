import cPickle
import numpy as np
import galsim
import matplotlib.pyplot as plt

# For information on where to download the .pkl file below, see the python script
# `devel/external/hst/make_cosmos_cfimage.py`
NOISEIMFILE = "acs_I_unrot_sci_20_noisearrays.pkl"  # Input pickled list filename
CFIMFILE_SUB = "acs_I_unrot_sci_20_cf_subtracted.fits" # Output image of correlation function (sub)
CFIMFILE_UNS = "acs_I_unrot_sci_20_cf_unsubtracted.fits" # Output image of correlation function

RSEED = 12334566

ud = galsim.UniformDeviate(RSEED)

# Case 1: subtract_mean=True; Case 2: subtract_mean=False
cn1 = galsim.getCOSMOSNoise(ud, CFIMFILE_SUB, dx_cosmos=1.)
cn2 = galsim.getCOSMOSNoise(ud, CFIMFILE_UNS, dx_cosmos=1.)

testim1 = galsim.ImageD(7, 7)
testim2 = galsim.ImageD(7, 7)
var1 = 0.
var2 = 0.

noisearrays = cPickle.load(open(NOISEIMFILE, 'rb'))
for noisearray, i in zip(noisearrays, range(len(noisearrays))):
    noise1 = galsim.ImageViewD((noisearray.copy()).astype(np.float64))
    noise2 = galsim.ImageViewD((noisearray.copy()).astype(np.float64))
    noise1.setScale(1.)
    noise2.setScale(1.)
    cn1.applyWhiteningTo(noise1)
    cn2.applyWhiteningTo(noise2)
    var1 += noise1.array.var()
    var2 += noise2.array.var()
    cntest1 = galsim.CorrelatedNoise(ud, noise1)
    cntest2 = galsim.CorrelatedNoise(ud, noise2)
    cntest1.draw(testim1, dx=1., add_to_image=True)
    cntest2.draw(testim2, dx=1., add_to_image=True)
    print "Done "+str(i + 1)+"/"+str(len(noisearrays))

testim1 /= len(noisearrays)
testim2 /= len(noisearrays)
var1 /= len(noisearrays)
var2 /= len(noisearrays)

print var1, var2

delx = np.arange(7) - 3.

import matplotlib.pyplot as plt
plt.semilogy(delx, testim1.array[:, 3] / var1,'r-', label='along y')
plt.semilogy(delx, testim1.array[3, :] / var1,'r--', label='along x')
plt.semilogy(delx, -testim1.array[:, 3] / var1, 'k-')
plt.semilogy(delx, -testim1.array[3, :] / var1, 'k--')
plt.ylim(1.e-4, 2.)
plt.legend()
plt.title('Mean subtracted, av. output variance = '+str(var1))
plt.savefig('mean_subtracted_xy_covariances.png')

plt.figure()
plt.semilogy(delx, -testim2.array[:, 3] / var2, 'k-')
plt.semilogy(delx, -testim2.array[3, :] / var2, 'k--')
plt.semilogy(delx, testim2.array[:, 3] / var2, 'b-', label='along y')
plt.semilogy(delx, testim2.array[3, :] / var2, 'b--', label='along x')
plt.ylim(1.e-4, 2.)
plt.legend()
plt.title('Non-mean subtracted, av. output variance = '+str(var2))
plt.savefig('non_mean_subtracted_xy_covariances.png')

# The variance in the final whitened noise for the non-mean subtracted CF estimate case is a factor
# of 25 larger!  I think that the non-mean subtracted case's reduction in the normalized off-
# diagonal correlations is pretty much consistent with this large excess in the central variance,
# boosting the ratio of variance/covariances.
#
# To me (Barney) this motivates using the mean subtracted COSMOS CF: it produces close-to-white
# (~0.3% off-diagonal covariances) noise fields with an output variance that is x25 smaller than
# when using the non-mean subtracted COSMOS CF.
