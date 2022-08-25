# Copyright (c) 2012-2022 by the GalSim developers team on GitHub
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

import sys
import time
import numpy as np
import galsim

# Put the salient numbers up here so they are easy to adjust.
counts_per_iter = 4.e3  # a few thousand is probably fine.  (bigger is faster of course.)
counts_total = 80.e3    # 80K flats
nx = 509
ny = 2000
nborder = 2
nflats = 10
treering_amplitude = 0.0#0.26
treering_period = 47.
treering_center = galsim.PositionD(0,0)
seed = 31415

for counts in range(1,6):
    counts_total = 20000.0 * counts     #  80.e3    # 80K flats

    # This is very similar to treering_skybg2.py, which builds a sky image.
    # But here we build up to a much higher flux level where B/F is important.

    t0 = time.time()

    rng = galsim.UniformDeviate(seed)

    treering_func = galsim.SiliconSensor.simple_treerings(treering_amplitude, treering_period)

    niter = int(counts_total / counts_per_iter + 0.5)
    counts_per_iter = counts_total / niter  # Recalculate in case not even multiple.
    print('Total counts = {} = {} * {}'.format(counts_total,niter,counts_per_iter))

    # Not an LSST wcs, but just make sure this works properly with a non-trivial wcs.
    wcs = galsim.FitsWCS('../../tests/fits_files/tnx.fits')

    base_image = galsim.ImageF(nx+2*nborder, ny+2*nborder, wcs=wcs)
    print('image bounds = ',base_image.bounds)

    # nrecalc is actually irrelevant here, since a recalculation will be forced on each iteration.
    # Which is really the point.  We need to set coundsPerIter appropriately so that the B/F effect
    # doesn't change *too* much between iterations.
    sensor = galsim.SiliconSensor(rng=rng,
                                  treering_func=treering_func, treering_center=treering_center)

    # We also need to account for the distortion of the wcs across the image.  
    # This expects sky_level in ADU/arcsec^2, not ADU/pixel.
    base_image.wcs.makeSkyImage(base_image, sky_level=1.)
    base_image.write('wcs_area.fits')

    # Rescale so that the mean sky level per pixel is skyCounts
    mean_pixel_area = base_image.array.mean()

    sky_level_per_iter = counts_per_iter / mean_pixel_area  # in ADU/arcsec^2 now.
    base_image *= sky_level_per_iter

    # The base_image has the right level to account for the WCS distortion, but not any sensor effects.
    # This is the noise-free level that we want to add each iteration modulated by the sensor.

    noise = galsim.PoissonNoise(rng)

    t1 = time.time()
    print('Initial setup time = ',t1-t0)

    # Make flats
    for n in range(nflats):
        t1 = time.time()
        # image is the image that we will build up in steps.
        # We add on a border of 2 pixels, since the outer row/col get a little messed up by photons
        # falling off the edge, but not coming on from the other direction.
        # We do 2 rows/cols rather than just 1 to be safe, since I think diffusion can probably go
        # 2 pixels, even though the deficit is only really evident on the outer pixel.
        image = galsim.ImageF(nx+2*nborder, ny+2*nborder, wcs=wcs)

        for i in range(niter):
            t2 = time.time()
            # temp is the additional flux we will add to the image in this iteration.
            # Start with the right area due to the sensor effects.
            temp = sensor.calculate_pixel_areas(image)
            temp.write('sensor_area.fits')

            # Multiply by the base image to get the right mean level and wcs effects
            temp *= base_image 
            temp.write('nonoise.fits')

            # Finally, add noise.  What we have here so far is the expectation value in each pixel.
            # We need to realize this according to Poisson statistics with these means.
            temp.addNoise(noise)
            temp.write('withnoise.fits')

            # Add this to the image we are building up.
            image += temp
            t3 = time.time()
            print('Iter {}: time = {}'.format(i,t3-t2))

        # Cut off the outer border where things don't work quite right.
        print('bounds = ',image.bounds)
        image = image.subImage(galsim.BoundsI(1+nborder,nx+nborder,1+nborder,ny+nborder))
        print('bounds => ',image.bounds)
        image.setOrigin(1,1)
        print('bounds => ',image.bounds)

        t4 = time.time()
        print('Total time to make flat image with level {} = {}'.format(counts_total, t4-t1))

        image.write('csl_flats/flat_%d_%02d.fits'%(counts,n))

    t5 = time.time()
    print('Total time to make {} flat images = {}'.format(nflats, t5-t0))
