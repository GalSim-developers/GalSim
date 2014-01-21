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

- gal = galsim.Chromatic(GSObject, wave, photons)
- obj = galsim.ChromaticAdd([list of ChromaticObjects])
- ChromaticObject.draw(throughput_fn, bluelim, redlim)
- PSF = galsim.ChromaticShiftAndDilate(GSObject, shift_fn, dilate_fn)
"""

import sys
import os
import math
import numpy
import logging
import galsim

def main(argv):
    # Where to find and output data
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "data/"))
    outpath = os.path.abspath(os.path.join(path, "output/"))

    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo12")

    # initialize (pseudo-)random number generator
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)

    # read in SEDs
    SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
    SEDs = {}
    for SED_name in SED_names:
        SED_filename = os.path.join(datapath, '{}.sed'.format(SED_name))
        wave, flambda = numpy.genfromtxt(SED_filename).T
        wave /= 10 # convert from Angstroms to nanometers
        # Create SED and normalize such that photon density is 1 photon per nm at 500 nm
        SEDs[SED_name] = galsim.SED(wave=wave, flambda=flambda, base_wavelength=500, norm=1.0)
    logger.debug('Successfully read in SEDs')

    # read in the LSST filters
    filter_names = 'ugrizy'
    filters = {}
    for filter_name in filter_names:
        filter_filename = os.path.join(datapath, 'LSST_{}.dat'.format(filter_name))
        wave, throughput = numpy.genfromtxt(filter_filename).T
        filters[filter_name] = galsim.LookupTable(wave, throughput)
    logger.debug('Read in filters')

    pixel_scale = 0.2 # arcseconds

    #-----------------------------------------------------------------------------------------------
    # Part A: chromatic Sersic galaxy

    # Here we create a chromatic version of an Exponential profile using the Chromatic class.
    # This class lets one create chromatic versions of any galsim GSObject class.  The first argument
    # is the GSObject instance to be chromaticized, and the second and third arguments are the
    # wavelength and photon array for the profile's SED.

    logger.info('')
    logger.info('Starting part A: chromatic Sersic galaxy')
    redshift = 0.8
    mono_gal = galsim.Exponential(half_light_radius=0.5)
    SEDs['CWW_E_ext'].setRedshift(redshift)
    gal = galsim.Chromatic(mono_gal, SEDs['CWW_E_ext'])

    # You can still shear, shift, and dilate the resulting chromatic object.
    gal.applyShear(g1=0.5, g2=0.3)
    gal.applyDilation(1.05)
    gal.applyShift((0.0, 0.1))
    logger.debug('Created Chromatic')

    # convolve with pixel and PSF to make final profile
    pix = galsim.Pixel(pixel_scale)
    PSF = galsim.Moffat(fwhm=0.6, beta=2.5)
    final = galsim.Convolve([gal, pix, PSF])
    logger.debug('Created final profile')

    # draw profile through LSST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.1)
    for filter_name, filter_ in filters.iteritems():
        img = galsim.ImageF(64, 64, scale=pixel_scale)
        final.draw(filter_, 300, 1100, image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12a_{}.fits'.format(filter_name))
        galsim.fits.write(img, out_filename)
        logger.debug('Wrote {}-band image to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 output/demo12a_*.fits -match scale -zoom 2 -match frame image')

    #-----------------------------------------------------------------------------------------------
    # Part B: chromatic bulge+disk galaxy

    logger.info('')
    logger.info('Starting part B: chromatic bulge+disk galaxy')
    redshift = 0.8
    # make a bulge ...
    mono_bulge = galsim.DeVaucouleurs(half_light_radius=0.5)
    SEDs['CWW_E_ext'].setRedshift(redshift)
    bulge = galsim.Chromatic(mono_bulge, SEDs['CWW_E_ext'])
    bulge.applyShear(g1=0.12, g2=0.07)
    logger.debug('Created bulge component')
    # ... and a disk ...
    mono_disk = galsim.Exponential(half_light_radius=2.0)
    SEDs['CWW_Im_ext'].setRedshift(redshift)
    disk = galsim.Chromatic(mono_disk, SEDs['CWW_Im_ext'])
    disk.applyShear(g1=0.4, g2=0.2)
    logger.debug('Created disk component')
    # ... and then combine them.
    bdgal = bulge+disk*5 # you can add and multiply ChromaticObjects just like GSObjects
    bdfinal = galsim.Convolve([bdgal, pix, PSF])
    logger.debug('Created bulge+disk galaxy final profile')

    # draw profile through LSST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.02)
    for filter_name, filter_ in filters.iteritems():
        img = galsim.ImageF(64, 64, scale=pixel_scale)
        bdfinal.draw(filter_, 300, 1100, image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12b_{}.fits'.format(filter_name))
        galsim.fits.write(img, out_filename)
        logger.debug('Wrote {}-band image to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output/demo12b_r.fits -green -scale limits'
                +' -0.25 1 output/demo12b_i.fits -red -scale limits -0.25 1 output/demo12b_z.fits'
                +' -zoom 2')

    #-----------------------------------------------------------------------------------------------
    # Part C: chromatic PSF

    logger.info('')
    logger.info('Starting part C: chromatic PSF')
    redshift = 0.0
    mono_gal = galsim.Exponential(half_light_radius=0.5)
    SEDs['CWW_Im_ext'].setRedshift(redshift)
    gal = galsim.Chromatic(mono_gal, SEDs['CWW_Im_ext'])
    gal.applyShear(g1=0.5, g2=0.3)
    logger.debug('Created Chromatic')

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
    # galaxy images land close to (0,0).
    zenith_angle = 30 * galsim.degrees
    # get_refraction returns its output as a galsim.AngleUnit
    R500 = galsim.dcr.get_refraction(500.0, zenith_angle) / galsim.arcsec
    # we'll assume the y-direction is the zenith direction, so only shift along the second component.
    shift_fn = lambda w:(0,(galsim.dcr.get_refraction(w, zenith_angle) / galsim.arcsec) - R500)

    # Define the dilation function due to Kolmogorov turbulence.
    # Normalize to 500 nm, at which wavelength the PSF will have FWHM=0.6 as defined below.
    dilate_fn = lambda w: (w/500.0)**(-0.2)

    # galsim.ChromaticShiftAndDilate functions similarly to Chromatic, in that it
    # chromaticizes an existing GSObject.  In this case, the existing object is a fiducial PSF which
    # gets Shifted and Dilated according to shift_fn and dilate_fn.  We'll use a Moffat profile as
    # the fiducial PSF.
    PSF = galsim.ChromaticShiftAndDilate(galsim.Moffat(beta=2.5, fwhm=0.5), shift_fn, dilate_fn)

    # convolve with pixel and PSF to create final profile
    pix = galsim.Pixel(pixel_scale)
    final = galsim.Convolve([gal, pix, PSF])
    logger.debug('Created chromatic PSF finale profile')

    # Draw profile through LSST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.03)
    for filter_name, filter_ in filters.iteritems():
        img = galsim.ImageF(64, 64, scale=pixel_scale)
        final.draw(filter_, 300, 1100, image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12c_{}.fits'.format(filter_name))
        galsim.fits.write(img, out_filename)
        logger.debug('Wrote {}-band image to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))
    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 output/demo12c_*.fits -match scale -zoom 2 -match frame image -blink')

if __name__ == "__main__":
    main(sys.argv)
