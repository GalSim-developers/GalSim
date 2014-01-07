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

This script currently doesn't have an equivalent demo*.yaml or demo*.json file.  The API for this
still needs to be written.

This script introduces the chromatic objects module galsim.chromatic, which handles wavelength-
dependent profiles.  Three uses of this module are demonstrated:

1) A chromatic object representing a Sersic galaxy with an XXXX-type SED at redshift XXXXX is
created.  The galaxy is then drawn using the six LSST filter total throughput curves to demonstrate
that the galaxy is an r-band dropout.

2) A two-component bulge+disk galaxy, in which the bulge and disk have different SEDs, is created
and then drawn using LSST g-, r-, and i-band filters to create a color rgbcube fits file, which will
appear chromatic when viewed using ds9.

3) A wavelength-dependent PSF is created to represent atmospheric effects of differential chromatic
refraction, and the wavelength dependence of Kolmogorov-turbulence-induced seeing.  This PSF is used
to draw a single Sersic galaxy in LSST g-, r-, and i-band filters, which will appear chromatic when
viewed using ds9.

New features introduced in this demo:

- gal = galsim.ChromaticBaseObject(GSObject, wave, photons)
- obj = galsim.ChromaticConvolve([list of ChromaticObjects and GSObjects])
- obj = galsim.ChromaticAdd([list of ChromaticObjects and GSObjects])
- ChromaticObject.draw(wave, throughput)
- psf = galsim.ChromaticShiftAndDilate(GSObject, shift_fn, dilate_fn)
"""

import sys
import os
import math
import numpy
import logging
import time
import galsim

def main(argv):
    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo12")

    # initialize (pseudo-)random number generator
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)
    gaussian_noise = galsim.GaussianNoise(rng, sigma=1e5)

    # read in SED and redshift it
    redshift = 2.1
    SED_name = 'CWW_E_ext'
    wave, flambda = numpy.genfromtxt('data/{}.sed'.format(SED_name)).T
    wave /= 10 # convert from Angstroms to nanometers
    wave *= (1+redshift) #redshift the spectrum
    photons = flambda*wave
    logger.debug('Successfully read in SEDs')

    # read in the r-, i-, and z-band LSST filters
    filter_names = 'riz'
    filters = {}
    for filter_name in filter_names:
        fwave, throughput = numpy.genfromtxt('data/LSST_{}.dat'.format(filter_name)).T
        filters[filter_name] = {'wave':fwave, 'throughput':throughput}
    logger.debug('Read in filters')

    # Here we create a chromatic version of a Sersic profile using the ChromaticBaseObject class.
    # This class lets one create chromatic versions of any galsim GSObject class.  The first argument
    # is the GSObject class to be chromaticized, and the second and third arguments are the
    # wavelength and photon array for the profile's SED.  Any required or optional arguments for the
    # GSObject are then inserted, such as the `n=1` and `half_light_radius=0.5` needed for
    # galsim.Sersic profile as seen below.
    gal = galsim.ChromaticBaseObject(galsim.Sersic, wave, photons,
                                     n=1, half_light_radius=0.5)
    # You can still shear, shift,  and dilate the resulting chromatic object.
    gal.applyShear(g1=0.5, g2=0.3)
    gal.applyDilation(1.05)
    gal.applyShift((0.0, 0.1))
    logger.debug('Created ChromaticBaseObject')

    # now place this galaxy in a scene
    pix = galsim.Pixel(0.2)
    psf = galsim.Moffat(fwhm=0.6, beta=2.5)
    scn = galsim.ChromaticConvolve([gal, pix, psf])
    logger.debug('Created scene')

    for filter_name in filter_names:
        filter_ = filters[filter_name]
        img = galsim.ImageF(64, 64, 0.2)
        scn.draw(wave=filter_['wave'], throughput=filter_['throughput'], image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        galsim.fits.write(img, 'output/demo12_{}.fits'.format(filter_name))
        logger.debug('Wrote {}-band image to disk'.format(filter_name))


if __name__ == "__main__":
    main(sys.argv)
