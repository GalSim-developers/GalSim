# Copyright (c) 2012-2020 by the GalSim developers team on GitHub
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

# Copied from https://github.com/LSSTDESC/imSim/blob/treerings/python/desc/imsim/skyModel.py
def get_bundled_photon_array(image, nphotons, nbundles_per_pix, rng):
    # A convenient way to do that is to have the fluxes of the
    # bundles be generated from a Poisson distribution.

    # Make a PhotonArray to hold the sky photons
    npix = np.prod(image.array.shape)
    nbundles = npix * nbundles_per_pix
    flux_per_bundle = np.float(nphotons) / nbundles
    #print('npix = ',npix)
    #print('nbundles = ',nbundles)
    #print('flux_per_bundle = ',flux_per_bundle)
    photon_array = galsim.PhotonArray(int(nbundles))

    # Generate the x,y values.
    xx, yy = np.meshgrid(np.arange(image.xmin, image.xmax+1),
                         np.arange(image.ymin, image.ymax+1))
    xx = xx.ravel()
    yy = yy.ravel()
    assert len(xx) == npix
    assert len(yy) == npix
    xx = np.repeat(xx, nbundles_per_pix)
    yy = np.repeat(yy, nbundles_per_pix)
    # If the photon_array is smaller than xx and yy,
    # randomly select the corresponding number of xy values.
    if photon_array.size() < len(xx):
        index = (np.random.permutation(np.arange(len(xx)))[:photon_array.size()],)
        xx = xx[index]
        yy = yy[index]
    assert len(xx) == photon_array.size()
    assert len(yy) == photon_array.size()
    galsim.random.permute(rng, xx, yy)  # Randomly reshuffle in place

    # The above values are pixel centers.  Add a random offset within each pixel.
    rng.generate(photon_array.x)  # Random values from 0..1
    photon_array.x -= 0.5
    rng.generate(photon_array.y)
    photon_array.y -= 0.5
    photon_array.x += xx
    photon_array.y += yy

    # Set the flux of the photons
    flux_pd = galsim.PoissonDeviate(rng, mean=flux_per_bundle)
    flux_pd.generate(photon_array.flux)

    return photon_array

def get_photon_array(image, nphotons, rng):
    # Simpler method that has all the pixels with flux=1.
    # Might be too slow, in which case consider switching to the above code.
    photon_array = galsim.PhotonArray(int(nphotons))

    # Generate the x,y values.
    rng.generate(photon_array.x) # 0..1 so far
    photon_array.x *= (image.xmax - image.xmin + 1)
    photon_array.x += image.xmin - 0.5  # Now from xmin-0.5 .. xmax+0.5
    rng.generate(photon_array.y)
    photon_array.y *= (image.ymax - image.ymin + 1)
    photon_array.y += image.ymin - 0.5

    # Flux in this case is simple.  All flux = 1.
    photon_array.flux = 1

    return photon_array


sky_sed_file = 'sky_sed.txt'
band = 'r'
bandpass_file = '%s_band.txt' % band

gs_sed = galsim.SED(sky_sed_file, wave_type='nm', flux_type='flambda').thin(rel_err=0.1)
gs_bandpass = galsim.Bandpass(bandpass_file, wave_type='nm').thin(rel_err=0.1)
print('made sed, bandpass')

seed = 140101
rng = galsim.UniformDeviate(seed)

waves = galsim.WavelengthSampler(sed=gs_sed, bandpass=gs_bandpass, rng=rng)
fratio = 1.234
obscuration = 0.606
angles = galsim.FRatioAngles(fratio, obscuration, rng)

treering_func = galsim.SiliconSensor.simple_treerings(0.26, 47.)
treering_center = galsim.PositionD(0,0)

skyCounts = 800.
print('skyCounts = ',skyCounts)
bundles_per_pix = 0   # Note: bundling doesn't work right if using tree rings.
chunk_size = int(1e7)

image = galsim.ImageF(2000, 500)
print('image bounds = ',image.bounds)

num_photons = int(np.prod(image.array.shape)*skyCounts)
print('num_photons = ',num_photons)

nrecalc = 1.e300
sensor = galsim.SiliconSensor(rng=rng, nrecalc=nrecalc,
                              treering_func=treering_func, treering_center=treering_center)

# This bit is also basically copied from imSim, although I've changed the logic about how
# the chunks work when bundles_per_pix > 0.  Still couldn't get it to work right though
# unless bundles_per_pix == skyCounts, which means it's not any faster to use bundling.
# I think because the photons slipping to neighbor pixels due to tree rings messes up
# the count statistics.
if bundles_per_pix == 0:
    print('using chunk_size = ',chunk_size)
    chunks = [chunk_size]*(num_photons//chunk_size)
    if num_photons % chunk_size > 0:
        chunks.append(num_photons % chunk_size)
else:
    print('bundles_per_pix = ',bundles_per_pix)
    # Each "chunk" is 1 bundle of photons.
    chunks = np.empty(bundles_per_pix)
    flux_per_chunk = float(num_photons) / bundles_per_pix
    pd = galsim.PoissonDeviate(rng, mean=flux_per_chunk)
    pd.generate(chunks)

for ichunk, nphot in enumerate(chunks):
    t0 = time.time()
    sys.stdout.write('{}  {}  {} {}  \n'.format(ichunk, nphot, image.array.min(),image.array.max()))
    sys.stdout.flush()

    if bundles_per_pix == 0:
        photon_array = get_photon_array(image, nphot, rng)
    else:
        photon_array = get_bundled_photon_array(image, nphot, 1., rng)
    waves.applyTo(photon_array)
    angles.applyTo(photon_array)

    # Accumulate the photons on the temporary amp image.
    sensor.accumulate(photon_array, image, resume=(ichunk>0))
    print(time.time() - t0)

image.write('test.fits')

# Check that the photons follow Poisson statistics, even through we use bundling.
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
