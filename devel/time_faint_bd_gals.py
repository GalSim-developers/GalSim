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

import timeit
import galsim
import numpy as np

# This script more or less runs through what happens to faint bulge + disk + knots
# objects in imSim.  In particular, these used to spend a lot of time computing
# SEDs, especially combining wave_lists, which are eventually discarded to instead
# use a "trivial_sed".

bulge_sed = galsim.SED('CWW_E_ext.sed', wave_type='ang', flux_type='flambda')
disk_sed = galsim.SED('CWW_Sbc_ext.sed', wave_type='ang', flux_type='flambda')
knots_sed = galsim.SED('CWW_Scd_ext.sed', wave_type='ang', flux_type='flambda')
trivial_sed = galsim.SED(galsim.LookupTable([100, 2000], [1,1], interpolant='linear'),
                         wave_type='nm', flux_type='fphotons')

bandpass = galsim.Bandpass('LSST_r.dat', 'nm')

rng = galsim.BaseDeviate(1234)

# This is from the DoubleGaussian class in imSim.
# It's cached in imsim, so put it a global scope here.
fwhm1=0.6
fwhm2=0.12
wgt1=1.0
wgt2=0.1
r1 = fwhm1/2.355
r2 = fwhm2/2.355
norm = 1.0/(wgt1 + wgt2)
gaussian1 = galsim.Gaussian(sigma=r1)
gaussian2 = galsim.Gaussian(sigma=r2)
double_gaussian = norm*(wgt1*gaussian1 + wgt2*gaussian2)


def draw_faint(rng):
    ud = galsim.UniformDeviate(rng)
    bulge = galsim.Sersic(n=4,
                          half_light_radius= 0.7 + ud() * 0.5)
    disk = galsim.Exponential(half_light_radius= 1.0 + ud() * 0.5)
    knots = galsim.RandomKnots(profile=disk, rng=rng, npoints=int(20+ud()*20))

    bulge_fraction = ud() * 0.5
    knots_fraction = ud() * 0.2
    disk_fraction = 1 - knots_fraction - bulge_fraction
    gal = galsim.Add(bulge * bulge_fraction * bulge_sed,
                     disk * disk_fraction * disk_sed,
                     knots * knots_fraction * knots_sed)
    gal.flux = ud() * 20  # Anything fainter than flux=100 is considered *faint* in imSim.

    # This isn't the psf we use, but it's suitably complicated enough for this test.
    psf = galsim.ChromaticAtmosphere(galsim.Moffat(fwhm=0.8, beta=2.3),
                                     base_wavelength=500, zenith_angle=15*galsim.degrees)

    realized_flux = galsim.PoissonDeviate(rng, mean=gal.flux)()

    if realized_flux == 0:
        return None

    if realized_flux < 10:
        image_size = 32
    else:
        psf = double_gaussian
        gal_achrom = gal.evaluateAtWavelength(bandpass.effective_wavelength)
        obj = galsim.Convolve(gal_achrom, psf).withFlux(realized_flux)
        image_size = obj.getGoodImageSize(0.2)

    image = galsim.Image(ncol=image_size, nrow=image_size, scale=0.2)  # Simple WCS.

    faint = realized_flux < 100
    assert faint  # This script is designed for this to always be true.

    gal = gal.evaluateAtWavelength(bandpass.effective_wavelength)
    gal = gal * trivial_sed
    gal = gal.withFlux(realized_flux, bandpass)

    gal.drawImage(bandpass, method='phot', rng=rng, n_photons=realized_flux, image=image,
                  poisson_flux=False)
    return image

n = 1000
t1 = min(timeit.repeat(lambda: draw_faint(rng), number=n))

print(f'Time for {n} iterations of draw_faint = {t1}')

with galsim.utilities.Profile(filename='faint.pstats'):
    for i in range(n):
        draw_faint(rng)
