# A script to time and profile runs of photon shooting using
#
#  - A realistic range of fluxes, so many objects have < 10 photons
#  - Surface layer ops, including wavelenght, angles
#  - Silicon sensor to get brighter-fatter

import galsim
import time
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
#import psutil
import resource
import cProfile
import pstats

# Some global variables that we might want to adjust
nobjects = 10**4

flux_min = 5.
flux_max = 1000.
flux_power = -1.

hlr_min = 0.1
hlr_max = 10.
hlr_power = -2.

xsize = 4096
ysize = 4096

pixel_scale = 0.2
fratio = 1.234
obscuration = 0.606
nrecalc = 10000

sed_list = [ galsim.SED(name, wave_type='ang', flux_type='flambda').thin(0.1) for name in
                ['CWW_E_ext.sed', 'CWW_Im_ext.sed', 'CWW_Sbc_ext.sed', 'CWW_Scd_ext.sed'] ]


def make_plots(times, mem, photons):
    index = np.arange(len(times))

    plt.clf()
    plt.plot(index, times)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('object number')
    plt.ylabel('cpu time (s)')
    plt.savefig('time.png')

    plt.clf()
    plt.plot(index, mem)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('object number')
    plt.ylabel('memory used')
    plt.savefig('mem.png')

    plt.clf()
    plt.plot(index, photons)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('object number')
    plt.ylabel('cumulative photons shot')
    plt.savefig('phot.png')

def generate_powerlaw_pdf(rng, x1, x2, n):
    """Generate a random value drawn from a power law distribution.

    Assume power law between x1 and x2
    p(x)  = A x^n
    Generate by taking cdf = rng() (uniform from 0..1)
    """

    cdf = rng()
    if n == -1:
        # cdf = A exp(x/x1)
        # cdf = ln(f/x1) / ln(x2/x1)
        f = x1 * np.exp(cdf * np.log(x2/x1))
    else:
        # cdf = A/(n+1) (x^(n+1) - x1^(n+1))
        #     = (f^(n+1) - x1^(n+!) / (x2^(n+1) - x1^(n+1))
        f1 = x1**(n+1)
        f2 = x2**(n+1)
        f = (cdf * (f2-f1) + f1)**(1./(n+1))
    return f

def get_flux(rng):
    """Get a random flux drawn from a power law"""
    return generate_powerlaw_pdf(rng, flux_min, flux_max, flux_power)

def get_hlr(rng):
    """Get a random half-light radius drawn from a power law"""
    return generate_powerlaw_pdf(rng, hlr_min, hlr_max, hlr_power)

def get_pos(rng):
    """Get a random position for this object falling somewhere on the image"""
    x = rng() * xsize + 1
    y = rng() * ysize + 1
    return galsim.PositionD(x,y)

def make_psf(rng):
    """Make the PSF."""
    psf = galsim.Moffat(fwhm=0.8, beta=3.5)
    return psf

def make_gal(rng):
    """Make the galaxy."""
    flux = get_flux(rng)
    hlr = get_hlr(rng)
    gal = galsim.Sersic(n=2, half_light_radius=hlr, flux=flux)
    return gal

def get_sed(rng):
    """Get an SED for this galaxy at a random redshift"""
    z = rng() * 4  # uniform from 0 to 4.
    index = int(rng() * len(sed_list))
    # This means there is a new sed object each time, but the disk I/O is amortized.
    sed = sed_list[index].atRedshift(z)
    return sed

def calculate_bounds(obj, pos, image):
    """Calculate a good bounds to use for this object based on its position.
    Also return the offset to use when drawing onto the postage stamp.
    """
    obj_on_image = image.wcs.toImage(obj)
    N = obj_on_image.getGoodImageSize(1.0)
    xmin = int(math.floor(pos.x) - N/2)
    xmax = int(math.ceil(pos.x) + N/2)
    ymin = int(math.floor(pos.y) - N/2)
    ymax = int(math.ceil(pos.y) + N/2)
    bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
    bounds = bounds & image.bounds
    offset = pos - bounds.true_center

    return bounds, offset

def main():
    pr = cProfile.Profile()
    pr.enable()

    rng = galsim.UniformDeviate(8675309)

    image = galsim.Image(xsize, ysize, scale=0.2)
    bandpass = galsim.Bandpass('LSST_r.dat', wave_type='nm').thin(0.1)

    angles = galsim.FRatioAngles(fratio, obscuration, rng)
    sensor = galsim.SiliconSensor(rng=rng, nrecalc=nrecalc)

    #process = psutil.Process(os.getpid())
    times = []
    mem = []
    phot = []

    t0 = time.clock()
    for iobj in range(nobjects):
        sys.stderr.write('.')
        psf = make_psf(rng)
        gal = make_gal(rng)
        obj = galsim.Convolve(psf, gal)

        sed = get_sed(rng)
        waves = galsim.WavelengthSampler(sed=sed, bandpass=bandpass, rng=rng)
        surface_ops = (waves, angles)

        pos = get_pos(rng)
        bounds, offset = calculate_bounds(obj, pos, image)

        obj.drawImage(method='phot', image=image[bounds], offset=offset,
                      rng=rng, sensor=sensor,
                      surface_ops=surface_ops)

        times.append(time.clock() - t0)
        #mem.append(process.memory_info().rss)
        mem.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        phot.append(obj.flux)

    image.write('phot.fits')
    phot = np.cumsum(phot)
    make_plots(times, mem, phot)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('time')
    ps.print_stats(20)

if __name__ == "__main__":
    main()
