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
"""
Demo #12

The twelvth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script currently doesn't have an equivalent demo*.yaml or demo*.json file.  The API for catalog
level chromatic objects has not been written yet.

This script introduces the chromatic objects module galsim.chromatic, which handles wavelength-
dependent profiles.  Three uses of this module are demonstrated:

1) A chromatic object representing a Sersic galaxy with an early-type SED at redshift 0.8 is
created.  The galaxy is then drawn using the six LSST filter throughput curves to demonstrate that
the galaxy is a g-band dropout.

2) A two-component bulge+disk galaxy, in which the bulge and disk have different SEDs, is created
and then drawn using LSST filters.

3) A wavelength-dependent PSF is created to represent atmospheric effects of differential chromatic
refraction, and the wavelength dependence of Kolmogorov-turbulence-induced seeing.  This PSF is used
to draw a single Sersic galaxy in LSST filters.

For all cases, suggested parameters for viewing in ds9 are also included.

New features introduced in this demo:

- gal = galsim.ChromaticGSObject(GSObject, wave, photons)
- obj = galsim.ChromaticConvolve([list of ChromaticObjects and GSObjects])
- obj = galsim.ChromaticAdd([list of ChromaticObjects])
- ChromaticObject.draw(filter_wave, filter_throughput)
- PSF = galsim.ChromaticShiftAndDilate(GSObject, shift_fn, dilate_fn)
"""

import sys
import os
import math
import numpy
import logging
import galsim

def main(argv):
    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo12")

    # initialize (pseudo-)random number generator
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.1)

    # read in SEDs
    SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
    SEDs = {}
    for SED_name in SED_names:
        wave, flambda = numpy.genfromtxt('data/{}.sed'.format(SED_name)).T
        wave /= 10 # convert from Angstroms to nanometers
        photons = flambda*wave
        #normalize SEDs to 1 photon per nm at 502nm
        photons /= photons[wave == 502]
        SEDs[SED_name] = {'wave':wave, 'photons':photons}
    logger.debug('Successfully read in SEDs')

    # read in the r-, i-, and z-band LSST filters
    filter_names = 'ugrizy'
    filters = {}
    for filter_name in filter_names:
        wave, throughput = numpy.genfromtxt('data/LSST_{}.dat'.format(filter_name)).T
        filters[filter_name] = {'wave':wave, 'throughput':throughput}
    logger.debug('Read in filters')

    #-----------------------------------------------------------------------------------------------
    # Part A: chromatic Sersic galaxy

    # Here we create a chromatic version of a Sersic profile using the ChromaticGSObject class.
    # This class lets one create chromatic versions of any galsim GSObject class.  The first argument
    # is the GSObject class to be chromaticized, and the second and third arguments are the
    # wavelength and photon array for the profile's SED.  Any required or optional arguments for the
    # GSObject are then inserted, such as the `n=1` and `half_light_radius=0.5` needed for
    # galsim.Sersic profile as seen below.

    logger.info('Starting part A: chromatic Sersic galaxy')
    redshift = 0.8
    gal = galsim.ChromaticGSObject(galsim.Sersic,
                                   SEDs['CWW_E_ext']['wave'] * (1+redshift),
                                   SEDs['CWW_E_ext']['photons'],
                                   n=1, half_light_radius=0.5)
    # You can still shear, shift, and dilate the resulting chromatic object.
    gal.applyShear(g1=0.5, g2=0.3)
    gal.applyDilation(1.05)
    gal.applyShift((0.0, 0.1))
    logger.debug('Created ChromaticGSObject')

    # now place this galaxy in a scene
    pixel_scale = 0.2 * galsim.arcsec
    pix = galsim.Pixel(pixel_scale / galsim.arcsec)
    PSF = galsim.Moffat(fwhm=0.6, beta=2.5)
    scn = galsim.ChromaticConvolve([gal, pix, PSF])
    logger.debug('Created scene')

    for filter_name in filter_names:
        filter_ = filters[filter_name]
        img = galsim.ImageF(64, 64, pixel_scale / galsim.arcsec)
        scn.draw(wave=filter_['wave'], throughput=filter_['throughput'], image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        galsim.fits.write(img, 'output/demo12a_{}.fits'.format(filter_name))
        logger.debug('Wrote {}-band image to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    # You can display the output in ds9 with a command line that looks something like:
    # `ds9 output/demo12a_*.fits -match scale`

    #-----------------------------------------------------------------------------------------------
    # Part B: chromatic bulge+disk galaxy

    logger.info('Starting part B: chromatic bulge+disk galaxy')
    redshift = 0.8
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.02)
    bulge = galsim.ChromaticGSObject(galsim.DeVaucouleurs,
                                     SEDs['CWW_E_ext']['wave'] * (1+redshift),
                                     SEDs['CWW_E_ext']['photons'],
                                     half_light_radius=0.5)
    bulge.applyShear(g1=0.12, g2=0.07)
#    bulge.applyShift((0,-2))
    logger.debug('Created bulge component')
    disk =  galsim.ChromaticGSObject(galsim.Exponential,
                                     SEDs['CWW_Im_ext']['wave'] * (1+redshift),
                                     SEDs['CWW_Im_ext']['photons'],
                                     half_light_radius=2.0)
    disk.applyShear(g1=0.4, g2=0.2)
#    disk.applyShift((0, 2))
    logger.debug('Created disk component')
    bdgal = bulge+disk*5
    bdscn = galsim.ChromaticConvolve([bdgal, pix, PSF])
    logger.debug('Created bulge+disk galaxy scene')

    for filter_name in filter_names:
        filter_ = filters[filter_name]
        img = galsim.ImageF(64, 64, pixel_scale / galsim.arcsec)
        bdscn.draw(wave=filter_['wave'], throughput=filter_['throughput'], image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        galsim.fits.write(img, 'output/demo12b_{}.fits'.format(filter_name))
        logger.debug('Wrote {}-band image to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    # You can display the output in ds9 with a command line that looks something like:
    # `ds9 -rgb -blue -scale limits -0.2 0.8 output/demo12b_r.fits -green -scale limits -0.25 1 output/demo12b_i.fits -red -scale limits -0.25 1 output/demo12b_z.fits`

    #-----------------------------------------------------------------------------------------------
    # Part C: chromatic bulge+disk galaxy

    logger.info('Starting part C: chromatic PSF')
    redshift = 0.0
    gal = galsim.ChromaticGSObject(galsim.Sersic,
                                   SEDs['CWW_Im_ext']['wave'] * (1+redshift),
                                   SEDs['CWW_Im_ext']['photons'],
                                   n=1, half_light_radius=0.5)
    gal.applyShear(g1=0.5, g2=0.3)
    logger.debug('Created ChromaticGSObject')

    # Create chromatic PSF implementing differential chromatic refraction (DCR) and chromatic seeing
    #
    # DCR shifts the position of the PSF as a function of wavelength.  The galsim.dcr module contains
    # code to compute the magnitude of this shift.
    #
    # Kolmogorov turbulence in the atmosphere leads to a seeing size (e.g., FWHM) that scales with
    # wavelength to the (-0.2) power.
    #
    # These two effects can be modeled together using the galsim.ChromaticShiftAndDilate class.

    # First we define the shifting function due to DCR.  We normalize to the shift at 500 nm so the
    # images don't fall off of the postage stamp completely.
    zenith_angle = 30 * galsim.degrees
    R500 = (galsim.dcr.atmosphere_refraction_angle(500.0, zenith_angle) / galsim.radians
            / pixel_scale.rad())
    shift_fn = lambda w:(0,(galsim.dcr.atmosphere_refraction_angle(w, zenith_angle) / galsim.radians
                            / pixel_scale.rad()) - R500)

    # Second define the dilation function due to Kolmogorov turbulence.
    dilate_fn = lambda w: (w/500.0)**(-0.2)

    # galsim.ChromaticShiftAndDilate functions similarly to ChromaticGSObject, in that it
    # chromaticizes and existing GSObject.  In this case, we'll use a Moffat profile as the fiducial
    # PSF.  The way we've set up the shifting and dilating functions, this is equivalent to the PSF
    # at 500nm.  Note that arguments to the fiducial Moffat profile are passed after the shift and
    # dilate functions.
    PSF = galsim.ChromaticShiftAndDilate(galsim.Moffat, shift_fn, dilate_fn, beta=2.5, fwhm=0.6)

    # now place this galaxy in a scene
    pix = galsim.Pixel(pixel_scale / galsim.arcsec)
    scn = galsim.ChromaticConvolve([gal, pix, PSF])
    logger.debug('Created scene')

    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.03)
    for filter_name in filter_names:
        filter_ = filters[filter_name]
        img = galsim.ImageF(64, 64, pixel_scale / galsim.arcsec)
        scn.draw(wave=filter_['wave'], throughput=filter_['throughput'], image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        galsim.fits.write(img, 'output/demo12c_{}.fits'.format(filter_name))
        logger.debug('Wrote {}-band image to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))
    # You can display the output in ds9 with a command line that looks something like:
    # `ds9 output/demo12a_*.fits -match scale`

if __name__ == "__main__":
    main(sys.argv)
