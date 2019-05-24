import os
import sys
import time

import galsim

def time_silicon_accumulate():
    nx = 1000
    ny = 1000
    nobj = 1000
    photons_per_obj = 10000
    flux_per_photon = 1

    rng = galsim.UniformDeviate(314159)

    sensor = galsim.SiliconSensor(rng=rng.duplicate(), diffusion_factor=0.0)

    im = galsim.ImageF(nx, ny)

    num_photons = nobj * photons_per_obj
    photons = galsim.PhotonArray(num_photons)

    rng.generate(photons.x)
    photons.x *= nx
    photons.x += 0.5
    rng.generate(photons.y)
    photons.y *= ny
    photons.y += 0.5
    photons.flux = flux_per_photon

    t1 = time.time()
    sensor.accumulate(photons, im)
    t2 = time.time()
    print('Time = ', t2-t1)


if __name__ == "__main__":
    time_silicon_accumulate()
