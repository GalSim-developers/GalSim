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

#
# Compute LSST focus curves including effects of Silicon absorption length and
# refraction, and the fast f/1.2 beam.
# Compare this to figure 8 from O'Connor++06 "Study of silicon sensor thickness
# optimization for LSST".  GalSim results below are grossly consistent, but
# differ in the exact shape and values that the focus curves take.

import numpy as np
import matplotlib.pyplot as plt
import galsim

bd = galsim.BaseDeviate(12)
depths = np.linspace(-25, 25, 41)  # microns
obj = galsim.Gaussian(sigma=1e-4)
sed = galsim.SED("1", wave_type='nm', flux_type='flambda')

fig, ax = plt.subplots()
oversampling = 16
for filter in ['g', 'z', 'y']:
    bandpass = galsim.Bandpass("LSST_{}.dat".format(filter), wave_type='nm')
    Ts = []
    for depth in depths:
        depth_pix = depth / 10
        surface_ops = [
            galsim.WavelengthSampler(sed, bandpass, rng=bd),
            galsim.FRatioAngles(1.234, 0.606, rng=bd),
            galsim.FocusDepth(depth_pix),
            galsim.Refraction(3.9)  # approx number for Silicon
        ]
        img = obj.drawImage(
            sensor=galsim.SiliconSensor(),
            method='phot',
            n_photons=1_000_000,
            surface_ops=surface_ops,
            scale=0.2/oversampling,  # oversample pixels to better resolve PSF size
            nx=32*oversampling,  # 6.4 arcsec stamp
            ny=32*oversampling,
        )
        Ts.append(img.calculateMomentRadius())
    Ts = np.array(Ts)/0.2*10*oversampling  # convert arcsec -> micron
    ax.scatter(depths, Ts, label=filter)
ax.set_ylim(2, 7)
ax.axvline(0, c='k')
ax.legend()
ax.set_xlabel("<- intrafocal   focus(micron)   extrafocal ->")
ax.set_ylabel("PSF size (micron)")
plt.show()
