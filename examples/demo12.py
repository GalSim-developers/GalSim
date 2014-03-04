# Copyright 2012-2014 The GalSim developers:
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

The twelfth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script currently doesn't have an equivalent demo*.yaml or demo*.json file.  The API for catalog
level chromatic objects has not been written yet.

This script introduces the chromatic objects module galsim.chromatic, which handles wavelength-
dependent profiles.  Three uses of this module are demonstrated:

1) A chromatic object representing an Exponential galaxy with an early-type SED at redshift 0.8 is
created.  The galaxy is then drawn using the six LSST filter throughput curves to demonstrate that
the galaxy is a g-band dropout.

2) A two-component bulge+disk galaxy, in which the bulge and disk have different SEDs, is created
and then drawn using LSST filters.

3) A wavelength-dependent PSF is created to represent atmospheric effects of differential chromatic
refraction, and the wavelength dependence of Kolmogorov-turbulence-induced seeing.  This PSF is used
to draw a single Sersic galaxy in the LSST filters.

For all cases, suggested parameters for viewing in ds9 are also included.

New features introduced in this demo:

- SED = galsim.SED(wave, flambda)
- SED2 = SED.atRedshift(redshift)
- bandpass = galsim.Bandpass(filename)
- bandpass2 = bandpass.truncate(relative_throughput=X)
- bandpass3 = bandpass2.thin(step)
- gal = galsim.Chromatic(GSObject, wave, photons)
- gal = GSObject * SED
- obj = galsim.Add([list of ChromaticObjects])
- ChromaticObject.draw(bandpass)
- PSF = galsim.ChromaticAtmosphere(GSObject, base_wavelength, zenith_angle)
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
        # Wavelengths read are in nanometers, flux in flambda.  This is important for the
        # differential chromatic refraction routines below, which assume wavelengths are in
        # nanometers. Without the DCR routines, then it's enough to just match the wavelength
        # units of the SED and the bandpass.
        SED = galsim.SED(SED_filename, wave_type='Ang')
        # The normalization of SEDs affects how many photons are eventually drawn into an image.
        # One way to control this normalization is to specify the flux density in photons per nm
        # at a particular wavelength.  For example, here we normalize such that photon density is
        # 1 photon per nm at 500 nm.
        SEDs[SED_name] = SED.withFluxDensity(target_flux_density=1.0, wavelength=500)
    logger.debug('Successfully read in SEDs')

    # read in the LSST filters
    filter_names = 'ugrizy'
    filters = {}
    for filter_name in filter_names:
        filter_filename = os.path.join(datapath, 'LSST_{}.dat'.format(filter_name))
        # Columns required to be 1) Wavelength in nm, 2) Dimensionless throughput
        filters[filter_name] = galsim.Bandpass(filter_filename)
        # don't waste time integrating where there's less than 1% relative throughput.
        filters[filter_name] = filters[filter_name].truncate(relative_throughput=0.01)
        # and thin out the wavelength sampling by a factor of 10 to also improve speed.
        filters[filter_name] = filters[filter_name].thin(10)
    logger.debug('Read in filters')

    pixel_scale = 0.2 # arcseconds

    #-----------------------------------------------------------------------------------------------
    # Part A: chromatic de Vaucouleurs galaxy

    # Here we create a chromatic version of a de Vaucouleurs profile using the Chromatic class.
    # This class lets one create chromatic versions of any galsim GSObject class.  The first
    # argument is the GSObject instance to be chromaticized, and the second argument is the
    # profile's SED.

    logger.info('')
    logger.info('Starting part A: chromatic De Vaucouleurs galaxy')
    redshift = 0.8
    mono_gal = galsim.Exponential(half_light_radius=0.5)
    SED = SEDs['CWW_E_ext'].atRedshift(redshift)
    gal = galsim.Chromatic(mono_gal, SED)

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
        final.draw(filter_, image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12a_{}.fits'.format(filter_name))
        galsim.fits.write(img, out_filename)
        logger.debug('Wrote {}-band image to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 output/demo12a_*.fits -match scale -zoom 2 -match frame image &')

    #-----------------------------------------------------------------------------------------------
    # Part B: chromatic bulge+disk galaxy

    logger.info('')
    logger.info('Starting part B: chromatic bulge+disk galaxy')
    redshift = 0.8
    # make a bulge ...
    mono_bulge = galsim.DeVaucouleurs(half_light_radius=0.5)
    bulge_SED = SEDs['CWW_E_ext'].atRedshift(redshift)
    # Here's a shortcut for creating a chromatic version of a GSObject:
    bulge = mono_bulge * bulge_SED
    bulge.applyShear(g1=0.12, g2=0.07)
    logger.debug('Created bulge component')
    # ... and a disk ...
    mono_disk = galsim.Exponential(half_light_radius=2.0)
    disk_SED = SEDs['CWW_Im_ext'].atRedshift(redshift)
    disk = mono_disk * disk_SED
    disk.applyShear(g1=0.4, g2=0.2)
    logger.debug('Created disk component')
    # ... and then combine them.
    bdgal = 1.1 * (0.8*bulge+4*disk) # you can add and multiply ChromaticObjects just like GSObjects
    bdfinal = galsim.Convolve([bdgal, pix, PSF])
    # Note that at this stage, our galaxy is chromatic but our PSF is still achromatic.  Part C)
    # below will dive into chromatic PSFs.
    logger.debug('Created bulge+disk galaxy final profile')

    # draw profile through LSST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.02)
    for filter_name, filter_ in filters.iteritems():
        img = galsim.ImageF(64, 64, scale=pixel_scale)
        bdfinal.draw(filter_, image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12b_{}.fits'.format(filter_name))
        galsim.fits.write(img, out_filename)
        logger.debug('Wrote {}-band image to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output/demo12b_r.fits -green -scale limits'
                +' -0.25 1.0 output/demo12b_i.fits -red -scale limits -0.25 1.0 output/demo12b_z.fits'
                +' -zoom 2 &')

    #-----------------------------------------------------------------------------------------------
    # Part C: chromatic PSF

    logger.info('')
    logger.info('Starting part C: chromatic PSF')
    redshift = 0.0
    mono_gal = galsim.Exponential(half_light_radius=0.5)
    SED = SEDs['CWW_Im_ext'].atRedshift(redshift)
    # Here's another way to set the normalization of the SED.  If we want 50 counts to be drawn
    # when observing an object with this SED through the g-band, for instance, then we do:
    SED = SED.withFlux(50.0, filters['g'])
    # The flux drawn through other bands, which sample different parts of the SED and have different
    # throughputs, will, of course, be differ.
    gal = mono_gal * SED
    gal.applyShear(g1=0.5, g2=0.3)
    logger.debug('Created `Chromatic` galaxy')

    # For a ground-based PSF, two chromatic effects are introduced by the atmosphere:
    # (i) differential chromatic refraction (DCR), and (ii) wavelength-dependent seeing.
    #
    # DCR shifts the position of the PSF as a function of wavelength.  Blue light is shifted
    # toward the zenith slightly more than red light.
    #
    # Kolmogorov turbulence in the atmosphere leads to a seeing size (e.g., FWHM) that scales with
    # wavelength to the (-0.2) power.
    #
    # These effects are both implemented in the ChroamticAtmosphere class as perturbations to a
    # fiducial PSF at some fiducial wavelength.

    # First we define a monochromatic PSF that will be the basis for perturbation as a function of
    # wavelength
    PSF_500 = galsim.Moffat(beta=2.5, fwhm=0.5)
    # Then we use ChromaticAtmosphere to manipulate this fiducial PSF as a function of wavelength.
    # We also specify the wavelength of our fiducial PSF, and the zenith_angle of the observation
    # so that the DCR can be computed.  We can also optionally specify the position angle that
    # points to zenith.  The default of 0.0 implies that zenith is "up".  A position angle of 90
    # degrees indicates a zenith directly to the right.
    PSF = galsim.ChromaticAtmosphere(PSF_500, 500.0, zenith_angle=30.0 * galsim.degrees,
                                     position_angle=0.0 * galsim.degrees)

    # convolve with pixel and PSF to create final profile
    pix = galsim.Pixel(pixel_scale)
    final = galsim.Convolve([gal, pix, PSF])
    logger.debug('Created chromatic PSF finale profile')

    # Draw profile through LSST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.03)
    for filter_name, filter_ in filters.iteritems():
        img = galsim.ImageF(64, 64, scale=pixel_scale)
        final.draw(filter_, image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12c_{}.fits'.format(filter_name))
        galsim.fits.write(img, out_filename)
        logger.debug('Wrote {}-band image to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 output/demo12c_*.fits -match scale -zoom 2 -match frame image -blink &')

if __name__ == "__main__":
    main(sys.argv)
