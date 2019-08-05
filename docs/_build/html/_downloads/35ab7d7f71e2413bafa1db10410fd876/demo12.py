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
"""
Demo #12

The twelfth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script currently doesn't have an equivalent demo*.yaml or demo*.json file.  The API for catalog
level chromatic objects has not been written yet.

This script introduces the chromatic objects module galsim.chromatic, which handles wavelength-
dependent profiles.  Three uses of this module are demonstrated:

1) A chromatic object representing a De Vaucouleurs galaxy with an early-type SED at redshift 0.8 is
created.  The galaxy is then drawn using the six LSST filter throughput curves to demonstrate that
the galaxy is a g-band dropout.

2) A two-component bulge+disk galaxy, in which the bulge and disk have different SEDs, is created
and then drawn using LSST filters.

3) A wavelength-dependent PSF is created to represent atmospheric effects of differential chromatic
refraction, and the wavelength dependence of Kolmogorov-turbulence-induced seeing.  This PSF is used
to draw a single Sersic galaxy in the LSST filters.

For all cases, suggested parameters for viewing in ds9 are also included.

New features introduced in this demo:

- SED = galsim.SED(wave, flambda, wave_type, flux_type)
- SED2 = SED.atRedshift(redshift)
- bandpass = galsim.Bandpass(filename, wave_type)
- bandpass2 = bandpass.truncate(relative_throughput)
- bandpass3 = bandpass2.thin(rel_err)
- gal = GSObject * SED
- obj = galsim.Add([list of ChromaticObjects])
- ChromaticObject.drawImage(bandpass)
- PSF = galsim.ChromaticAtmosphere(GSObject, base_wavelength, zenith_angle)
"""

import sys
import os
import logging
import galsim

def main(argv):
    # Where to find and output data
    path, filename = os.path.split(__file__)
    outpath = os.path.abspath(os.path.join(path, "output/"))
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    # Make output directory if not already present.
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    datapath = galsim.meta_data.share_dir

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
        SED_filename = os.path.join(datapath, 'SEDs/{0}.sed'.format(SED_name))
        # Here we create some galsim.SED objects to hold star or galaxy spectra.  The most
        # convenient way to create realistic spectra is to read them in from a two-column ASCII
        # file, where the first column is wavelength and the second column is flux. Wavelengths in
        # the example SED files are in Angstroms, flux in flambda.  We use a set of files that are
        # distributed with GalSim in the share/ directory.
        SED = galsim.SED(SED_filename, wave_type='Ang', flux_type='flambda')
        # The normalization of SEDs affects how many photons are eventually drawn into an image.
        # One way to control this normalization is to specify the flux density in photons per nm
        # at a particular wavelength.  For example, here we normalize such that the photon density
        # is 1 photon per nm at 500 nm.
        SEDs[SED_name] = SED.withFluxDensity(target_flux_density=1.0, wavelength=500)
    logger.debug('Successfully read in SEDs')

    # read in the LSST filters
    filter_names = 'ugrizy'
    filters = {}
    for filter_name in filter_names:
        filter_filename = os.path.join(datapath, 'bandpasses/LSST_{0}.dat'.format(filter_name))
        # Here we create some galsim.Bandpass objects to represent the filters we're observing
        # through.  These include the entire imaging system throughput including the atmosphere,
        # reflective and refractive optics, filters, and the CCD quantum efficiency.  These are
        # also conveniently read in from two-column ASCII files where the first column is
        # wavelength and the second column is dimensionless throughput. The example filter files
        # units of nanometers for the wavelength type, so we specify that using the required
        # `wave_type` argument.
        filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='nm')
        # For speed, we can thin out the wavelength sampling of the filter a bit.
        # In the following line, `rel_err` specifies the relative error when integrating over just
        # the filter (however, this is not necessarily the relative error when integrating over the
        # filter times an SED).
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)
    logger.debug('Read in filters')

    pixel_scale = 0.2 # arcseconds

    #-----------------------------------------------------------------------------------------------
    # Part A: chromatic de Vaucouleurs galaxy

    # Here we create a chromatic version of a de Vaucouleurs profile by multipying a GSObject by an
    # SED.  This is how one generally constructs _separable_ ChromaticObjects in GalSim, that is,
    # those objects whose spatial dependence and wavelength dependence factor.

    logger.info('')
    logger.info('Starting part A: chromatic De Vaucouleurs galaxy')
    redshift = 0.8
    mono_gal = galsim.DeVaucouleurs(half_light_radius=0.5)
    SED = SEDs['CWW_E_ext'].atRedshift(redshift)
    gal = mono_gal * SED

    # You can still shear, shift, and dilate the resulting chromatic object.
    gal = gal.shear(g1=0.5, g2=0.3).dilate(1.05).shift((0.0, 0.1))
    logger.debug('Created separable ChromaticObject')

    # convolve with PSF to make final profile
    PSF = galsim.Moffat(fwhm=0.6, beta=2.5)
    final = galsim.Convolve([gal, PSF])
    logger.debug('Created final profile')

    # draw profile through LSST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.1)
    for filter_name, filter_ in filters.items():
        img = galsim.ImageF(64, 64, scale=pixel_scale)
        final.drawImage(filter_, image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {0}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12a_{0}.fits'.format(filter_name))
        galsim.fits.write(img, out_filename)
        logger.debug('Wrote {0}-band image to disk'.format(filter_name))
        logger.info('Added flux for {0}-band image: {1}'.format(filter_name, img.added_flux))

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
    bulge = mono_bulge * bulge_SED
    bulge = bulge.shear(g1=0.12, g2=0.07)
    logger.debug('Created bulge component')
    # ... and a disk ...
    mono_disk = galsim.Exponential(half_light_radius=2.0)
    disk_SED = SEDs['CWW_Im_ext'].atRedshift(redshift)
    disk = mono_disk * disk_SED
    disk = disk.shear(g1=0.4, g2=0.2)
    logger.debug('Created disk component')
    # ... and then combine them.
    bdgal = 1.1 * (0.8*bulge+4*disk) # you can add and multiply ChromaticObjects just like GSObjects
    bdfinal = galsim.Convolve([bdgal, PSF])
    # Note that at this stage, our galaxy is chromatic but our PSF is still achromatic.  Part C)
    # below will dive into chromatic PSFs.
    logger.debug('Created bulge+disk galaxy final profile')

    # draw profile through LSST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.02)
    for filter_name, filter_ in filters.items():
        img = galsim.ImageF(64, 64, scale=pixel_scale)
        bdfinal.drawImage(filter_, image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {0}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12b_{0}.fits'.format(filter_name))
        galsim.fits.write(img, out_filename)
        logger.debug('Wrote {0}-band image to disk'.format(filter_name))
        logger.info('Added flux for {0}-band image: {1}'.format(filter_name, img.added_flux))

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
    # when observing an object with this SED through the LSST g-band filter, for instance, then we
    # can do:
    SED = SED.withFlux(50.0, filters['g'])
    # The flux drawn through other bands, which sample different parts of the SED and have different
    # throughputs, will, of course, be different.
    gal = mono_gal * SED
    gal = gal.shear(g1=0.5, g2=0.3)
    logger.debug('Created chromatic galaxy')

    # For a ground-based PSF, two chromatic effects are introduced by the atmosphere:
    # (i) differential chromatic refraction (DCR), and (ii) wavelength-dependent seeing.
    #
    # DCR shifts the position of the PSF as a function of wavelength.  Blue light is shifted
    # toward the zenith slightly more than red light.
    #
    # Kolmogorov turbulence in the atmosphere leads to a seeing size (e.g., FWHM) that scales with
    # wavelength to the (-0.2) power.
    #
    # The ChromaticAtmosphere function will attach both of these effects to a fiducial PSF at
    # some fiducial wavelength.

    # First we define a monochromatic PSF to be the fiducial PSF.
    PSF_500 = galsim.Moffat(beta=2.5, fwhm=0.5)
    # Then we use ChromaticAtmosphere to manipulate this fiducial PSF as a function of wavelength.
    # ChromaticAtmosphere also needs to know the wavelength of the fiducial PSF, and the location
    # and orientation of the object with respect to the zenith.  This final piece of information
    # can be specified in several ways (see the ChromaticAtmosphere docstring for all of them).
    # Here are a couple ways: let's pretend our object is located near M101 on the sky, we observe
    # it 1 hour before it transits and we're observing from Mauna Kea.
    ra = galsim.Angle.from_hms("14:03:13") # hours : minutes : seconds
    dec = galsim.Angle.from_dms("54:20:57") # degrees : minutes : seconds
    m101 = galsim.CelestialCoord(ra, dec)
    latitude = 19.8207 * galsim.degrees # latitude of Mauna Kea
    HA = -1.0 * galsim.hours # Hour angle = one hour before transit

    # Then we can compute the zenith angle and parallactic angle (which is is the position angle
    # of the zenith measured from North through East) of this object:
    za, pa = galsim.dcr.zenith_parallactic_angles(m101, HA=HA, latitude=latitude)
    # And then finally, create the chromatic PSF
    PSF = galsim.ChromaticAtmosphere(PSF_500, 500.0, zenith_angle=za, parallactic_angle=pa)
    # We could have also just passed `m101`, `latitude` and `HA` to ChromaticAtmosphere directly:
    PSF = galsim.ChromaticAtmosphere(PSF_500, 500.0, obj_coord=m101, latitude=latitude, HA=HA)
    # and proceed like normal.

    # convolve with galaxy to create final profile
    final = galsim.Convolve([gal, PSF])
    logger.debug('Created chromatic PSF final profile')

    # Draw profile through LSST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.03)
    for filter_name, filter_ in filters.items():
        img = galsim.ImageF(64, 64, scale=pixel_scale)
        final.drawImage(filter_, image=img)
        img.addNoise(gaussian_noise)
        logger.debug('Created {0}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12c_{0}.fits'.format(filter_name))
        galsim.fits.write(img, out_filename)
        logger.debug('Wrote {0}-band image to disk'.format(filter_name))
        logger.info('Added flux for {0}-band image: {1}'.format(filter_name, img.added_flux))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 output/demo12c_*.fits -match scale -zoom 2 -match frame image -blink &')

if __name__ == "__main__":
    main(sys.argv)
