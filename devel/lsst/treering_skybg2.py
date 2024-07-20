# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

from __future__ import print_function

import time
import numpy as np
import galsim

seed = 140101
rng = galsim.UniformDeviate(seed)

treering_func = galsim.SiliconSensor.simple_treerings(0.26, 47.)
treering_center = galsim.PositionD(0,0)

skyCounts = 800.
print('skyCounts = ',skyCounts)

# Not an LSST wcs, but just make sure this works properly with a non-trivial wcs.
wcs = galsim.FitsWCS('../../tests/fits_files/tnx.fits')

t0 = time.time()
image = galsim.ImageF(2000, 500, wcs=wcs)
print('image bounds = ',image.bounds)

nrecalc = 1.e300
sensor = galsim.SiliconSensor(rng=rng, nrecalc=nrecalc,
                              treering_func=treering_func, treering_center=treering_center)

# For regular sky photons, we can just use the pixel areas to buidl the sky image.
# At this point the image is blank, so area is just from tree rings.
sensor_area = sensor.calculate_pixel_areas(image)
sensor_area.write('sensor_area.fits')

# We also need to account for the distortion of the wcs across the image.  
# This expects sky_level in ADU/arcsec^2, not ADU/pixel.
image.wcs.makeSkyImage(image, sky_level=1.)
image.write('wcs_area.fits')

# Rescale so that the mean sky level per pixel is skyCounts
mean_pixel_area = image.array.mean()
image *= skyCounts / mean_pixel_area

# Now multiply by the area due to the sensor effects.
image *= sensor_area

# Finally, add noise.  What we have here so far is the expectation value in each pixel.
# We need to realize this according to Poisson statistics with these means.
noise = galsim.PoissonNoise(rng)
image.addNoise(noise)
t1 = time.time()
print('Time to make sky image = ',t1-t0)

image.write('sky.fits')

# Check that the photons follow Poisson statistics
import matplotlib.pyplot as plt
from scipy.stats import poisson

fig = plt.figure()
ax = fig.add_subplot(111)
bin_width = 5
bins = np.arange(0,2*skyCounts,bin_width)
n, bins, p = ax.hist(image.array.ravel(), bins=bins, histtype='step', color='blue', fill=True)

npix = np.prod(image.array.shape)
ax.plot(bins, npix * bin_width * poisson.pmf(bins, skyCounts), color='green')

ax.set_xlabel('photons per pixel')
ax.set_ylabel('n pixels')
plt.tight_layout()
plt.savefig('poisson_test.pdf')
