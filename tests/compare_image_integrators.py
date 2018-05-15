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

from __future__ import print_function
import os
import numpy as np
path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))
try:
    import galsim
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# liberal use of globals here...
zenith_angle = 30 * galsim.degrees

# some profile parameters to test with
bulge_n = 4.0
bulge_hlr = 0.5
bulge_e1 = 0.2
bulge_e2 = 0.2

disk_n = 1.0
disk_hlr = 1.0
disk_e1 = 0.4
disk_e2 = 0.2

PSF_hlr = 0.3
PSF_beta = 2.6
PSF_e1 = 0.01
PSF_e2 = 0.06

shear_g1 = 0.01
shear_g2 = 0.02

# load some spectra and a filter
bulge_SED = galsim.SED(os.path.join(datapath, 'CWW_E_ext.sed'), wave_type='A')

disk_SED = galsim.SED(os.path.join(datapath, 'CWW_Sbc_ext.sed'), wave_type='A')

bandpass = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'))
bandpass = bandpass.truncate(relative_throughput=0.001)

bulge_SED = bulge_SED.withFlux(1.0, bandpass)
disk_SED = disk_SED.withFlux(1.0, bandpass)
def silentgetmoments(image1):
    xgrid, ygrid = np.meshgrid(np.arange(image1.array.shape[1]) + image1.getXMin(),
                               np.arange(image1.array.shape[0]) + image1.getYMin())
    mx = np.sum(xgrid * image1.array) / np.sum(image1.array)
    my = np.sum(ygrid * image1.array) / np.sum(image1.array)
    mxx = np.sum(((xgrid-mx)**2) * image1.array) / np.sum(image1.array)
    myy = np.sum(((ygrid-my)**2) * image1.array) / np.sum(image1.array)
    mxy = np.sum((xgrid-mx) * (ygrid-my) * image1.array) / np.sum(image1.array)
    return mx, my, mxx, myy, mxy

def compare_image_integrators():
    import galsim.integ
    import time

    pixel_scale = 0.2
    stamp_size = 128

    gal = galsim.Chromatic(galsim.Gaussian(half_light_radius=0.5), disk_SED)
    PSF_500 = galsim.Gaussian(half_light_radius=PSF_hlr)
    PSF = galsim.ChromaticAtmosphere(PSF_500, 500.0, zenith_angle)

    final = galsim.Convolve([gal, PSF])
    image = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)

    # truth flux
    x = np.union1d(disk_SED.wave_list, bandpass.wave_list)
    x = x[(x <= bandpass.red_limit) & (x >= bandpass.blue_limit)]
    target = np.trapz(disk_SED(x) * bandpass(x), x)
    print('target')
    print('        {:14.11f}'.format(target))

    t1 = time.time()
    print('midpoint')
    for N in [10, 30, 100, 300, 1000, 3000]:
        image = final.drawImage(
            bandpass, image=image,
            integrator=galsim.integ.ContinuousIntegrator(rule=galsim.integ.midpt, N=N))
        mom = silentgetmoments(image)
        outstring = '   {:4d} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f}'
        print(outstring.format(N, image.array.sum(), image.array.sum()-target, *mom))
    t2 = time.time()
    print('time for midpoint = %.2f'%(t2-t1))

    print('trapezoidal')
    for N in [10, 30, 100, 300, 1000, 3000]:
        image = final.drawImage(
            bandpass, image=image,
            integrator=galsim.integ.ContinuousIntegrator(rule=np.trapz, N=N))
        mom = silentgetmoments(image)
        outstring = '   {:4d} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f}'
        print(outstring.format(N, image.array.sum(), image.array.sum()-target, *mom))
    t3 = time.time()
    print('time for trapezoidal = %.2f'%(t3-t2))

if __name__ == '__main__':
    compare_image_integrators()
