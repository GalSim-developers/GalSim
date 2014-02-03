# Copyright 2012, 2013 The GalSim developers:
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
Egal_wave, Egal_flambda = np.genfromtxt(os.path.join(datapath, 'CWW_E_ext.sed')).T
Egal_wave /= 10 # Angstrom -> nm
bulge_SED = galsim.SED(wave=Egal_wave, flambda=Egal_flambda)
bulge_SED.setNormalization(base_wavelength=500.0, normalization=0.3)

Sbcgal_wave, Sbcgal_flambda = np.genfromtxt(os.path.join(datapath, 'CWW_Sbc_ext.sed')).T
Sbcgal_wave /= 10 # Angstrom -> nm
disk_SED = galsim.SED(wave=Sbcgal_wave, flambda=Sbcgal_flambda)
disk_SED.setNormalization(base_wavelength=500.0, normalization=0.3)

filter_wave, filter_throughput = np.genfromtxt(os.path.join(datapath, 'LSST_r.dat')).T
bandpass = galsim.Bandpass(filter_wave, filter_throughput)
bandpass.truncate(relative_throughput=0.001)

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
    pix = galsim.Pixel(pixel_scale)
    PSF_500 = galsim.Gaussian(half_light_radius=PSF_hlr)
    PSF = galsim.ChromaticAtmosphere(PSF_500, 500.0, zenith_angle)

    final = galsim.Convolve([gal, PSF, pix])
    image = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)

    # truth flux
    target = galsim.integ.int1d(lambda w:disk_SED(w) * bandpass(w),
                                bandpass.blue_limit, bandpass.red_limit)
    print 'target'
    print '        {:14.11f}'.format(target)

    t1 = time.time()
    print 'midpoint'
    for N in [10, 30, 100, 300, 1000, 3000]:
        image = galsim.ChromaticObject.draw(final, bandpass, N=N, image=image)
        mom = silentgetmoments(image)
        outstring = '   {:4d} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f}'
        print outstring.format(N, image.array.sum(), image.array.sum()-target, *mom)
    t2 = time.time()
    print 'time for midpoint = %.2f'%(t2-t1)

    print 'trapezoidal'
    for N in [10, 30, 100, 300, 1000, 3000]:
        image = galsim.ChromaticObject.draw(final, bandpass, N=N, image=image,
                                             integrator = galsim.integ.trapezoidal_int_image)
        mom = silentgetmoments(image)
        outstring = '   {:4d} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f}'
        print outstring.format(N, image.array.sum(), image.array.sum()-target, *mom)
    t3 = time.time()
    print 'time for trapezoidal = %.2f'%(t3-t2)

    print 'Simpson\'s'
    for N in [10, 30, 100, 300, 1000, 3000]:
        image = galsim.ChromaticObject.draw(final, bandpass, N=N, image=image,
                                             integrator = galsim.integ.simpsons_int_image)
        mom = silentgetmoments(image)
        outstring = '   {:4d} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f}'
        print outstring.format(N, image.array.sum(), image.array.sum()-target, *mom)
    t4 = time.time()
    print 'time for simpsons = %.2f'%(t4-t3)

    print 'Globally Adaptive Gauss-Kronrod'
    simpsons_image = np.array(image.array) #assume large N Simpson's is truth for comparison...
    for rel_err in [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8]:
        image = galsim.ChromaticObject.draw(final, bandpass, image=image,
                                            integrator = galsim.integ.globally_adaptive_GK_int_image,
                                            rel_err=rel_err, verbose=True)
        mom = silentgetmoments(image)
        outstring = '{:4.1e} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f} {:14.11f}'
        print outstring.format(rel_err, image.array.sum(), image.array.sum()-target, *mom)
        rel_err = (np.sqrt(np.mean((image.array - simpsons_image)**2))
                   / np.abs(image.array).sum())
        print 'relative error: {}'.format(rel_err)

    t5 = time.time()
    print 'time for Globally Adaptive Gauss-Kronrod = %.2f'%(t5-t4)


if __name__ == '__main__':
    compare_image_integrators()
