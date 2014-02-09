# Copyright 2012-2014 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
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

