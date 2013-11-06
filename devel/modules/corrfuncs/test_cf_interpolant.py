"""test_cf_interpolant.py:  Make upsampled plots of correlation functions to see how their
interpolants behave.

A simple script used to quickly generate plots for the discussion of which interpolant should be
used to describe correlation functions.  See the discussion on: TODO ADD LINK TO PR FOR #430 
"""
import numpy as np
import matplotlib.pyplot as plt
import galsim

CFSIZE=9
UPSAMPLING = 3

rng = galsim.BaseDeviate(752424)
gd = galsim.GaussianNoise(rng)
noise_image = galsim.ImageD(CFSIZE, CFSIZE)
noise_image.addNoise(gd)

cn = galsim.CorrelatedNoise(rng, noise_image, x_interpolant=galsim.Nearest(tol=1.e-4), dx=1.)

test_image = galsim.ImageD(
    2 * UPSAMPLING * CFSIZE, 2 * UPSAMPLING * CFSIZE, scale=1. / float(UPSAMPLING))
cn.applyRotation(30. * galsim.degrees)
cn.draw(test_image)

plt.clf()
plt.pcolor(test_image.array)
plt.colorbar()
plt.savefig('cf_nearest.png')
plt.show()

