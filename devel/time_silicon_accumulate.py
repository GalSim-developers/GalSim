# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
