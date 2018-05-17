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

"""test_cf_interpolant.py:  Make upsampled plots of correlation functions to see how their
interpolants behave.

A simple script used to quickly generate plots for the discussion of which interpolant should be
used to describe correlation functions.  See the discussion at
https://github.com/GalSim-developers/GalSim/pull/452#discussion-diff-5701561 
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

