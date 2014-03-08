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
import os
import sys
import numpy as np

from galsim_test_helpers import *

"""Tests to determine the cross-over point for using photon shooting vs. fft draw.
"""

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def time_gaussian():
    """Find the cross-over time for Gaussian draw vs drawShoot"""
    import time

    flux = 1.e5
    pixel_scale = 0.44
    stamp_size = 32
    ntile = 1 # In each direction, so 100 tiles total

    rng = galsim.BaseDeviate(72347234)

    gauss = galsim.Gaussian(sigma = 2., flux=flux)
    pixel = galsim.Pixel(scale = pixel_scale)
    withpix = galsim.Convolve([gauss,pixel])
    nopix = gauss

    image = galsim.Image(stamp_size*ntile, stamp_size*ntile, scale = pixel_scale)

    sky_level = 1.e3
    noise = galsim.PoissonNoise(rng, sky_level)
    ntot = ntile**2

    # First time the fft draw.  This is independent of flux, so it doesn't matter what we choose.
    S = 0
    N = 0
    t1 = time.time()
    for ix in range(ntile):
        for iy in range(ntile):
            b = galsim.BoundsI(ix*stamp_size+1, (ix+1)*stamp_size,
                               iy*stamp_size+1, (iy+1)*stamp_size)
            stamp = image[b]
            withpix.draw(stamp)
            stamp.addNoise(noise)
            S += stamp.array.sum()
            N += np.sqrt(sky_level) * stamp_size
    t2 = time.time()
    print 'Mean S/N = ',S/ntot,'/',N/ntot,'=',S/N
    print 'time for fft draw with Gaussian * Pixel = ',t2-t1
    image.write('junk1.fits')

    # Repeat with drawShoot for a few different S/N values 
    times = []
    sky_levels = [ 1000, 4000, 16000, 64000, 256000 ]
    for sky_level in sky_levels:
        S = 0
        N = 0
        noise = galsim.PoissonNoise(rng, sky_level)
        t1 = time.time()
        for ix in range(ntile):
            for iy in range(ntile):
                b = galsim.BoundsI(ix*stamp_size+1, (ix+1)*stamp_size,
                                iy*stamp_size+1, (iy+1)*stamp_size)
                stamp = image[b]
                nopix.drawShoot(stamp, max_extra_noise = 0.01*sky_level)
                stamp.addNoise(noise)
                S += stamp.array.sum()
                N += np.sqrt(sky_level) * stamp_size
        t2 = time.time()
        print 'Mean S/N = ',S/ntot,'/',N/ntot,'=',S/N
        print 'time for phot drawShoot with sky_level = ',sky_level,' = ',t2-t1
        image.write('junk%d.fits'%sky_level)
        times.append(t2-t1)

    print 'time, sky'
    for time, sky in zip(times,sky_levels):
        print time, sky

if __name__ == "__main__":
    time_gaussian()
